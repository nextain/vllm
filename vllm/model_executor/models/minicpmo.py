# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only MiniCPM-O model compatible with HuggingFace weights."""

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

import numpy as np
import torch
from torch import nn
from transformers import BatchFeature
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.whisper.modeling_whisper import (
    ACT2FN,
    WhisperAttention,
    WhisperConfig,
    WhisperEncoder,
)

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargsItems
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    NestedTensors,
)
from vllm.multimodal.parse import (
    AudioItem,
    AudioProcessorItems,
    DictEmbeddingItems,
    ModalityData,
    ModalityDataItems,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.transformers_utils.dynamic_module import try_get_class_from_dynamic_module
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import SupportsAudioOutput
from .minicpmv import (
    _MAX_FRAMES_PER_VIDEO,
    MiniCPMV2_6,
    MiniCPMV4_5,
    MiniCPMVDummyInputsBuilder,
    MiniCPMVMultiModalDataParser,
    MiniCPMVMultiModalProcessor,
    MiniCPMVProcessingInfo,
    _minicpmv_field_config,
)
from .utils import AutoWeightsLoader, cast_overflow_tensors, maybe_prefix

logger = init_logger(__name__)

CPU_DEVICE = torch.device("cpu")

if os.getenv("USE_FLAGOS") == "1":
    import flag_gems

    FLAG_GEMS_CONFIG = [
        "sort",
        "sort_stable",
        "layer_norm",
        "clamp_",
        "cos",
        "embedding",
        "exp",
        "exponential_",
        "full",
        "gather",
        "gelu",
        "index",
        "le",
        "lt",
        "lt_scalar",
        "masked_fill_",
        "max",
        "ones",
        "pow_scalar",
        "prod_dim",
        "rand_like",
        "reciprocal",
        "repeat",
        "scatter",
        "scatter_",
        "sin",
        "sub",
        "true_divide",
        "true_divide_",
        "uniform_",
        "where_scalar_self",
        "where_self_out",
        "zeros",
        "zeros_like",
    ]
    flag_gems.only_enable(record=False, include=FLAG_GEMS_CONFIG)


class MiniCPMOAudioFeatureInputs(TensorSchema):
    """
    Dimensions:
        - bns: Batch size * number of audios * number of slices
        - bn: Batch size * number of audios
        - c: Number of channels
        - l: Length
        - s: Number of slices
    """

    type: Literal["audio_features"] = "audio_features"

    audio_features: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bns", "c", "l", dynamic_dims={"l"}),
    ]
    """
    Slice here means chunk. Audio that is too long will be split into slices,
    which is the same as image. Padding is used therefore `audio_features` is 
    `torch.Tensor`.
    """

    audio_feature_lens: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "s"),
    ]
    """
    This should be feature length of each audio slice, 
    which equals to `audio_features.shape[-1]`
    """


class MiniCPMOAudioEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of audios
        - s: Number of slices
        - h: Hidden size (must match language model backbone)

    Length of each slice may vary, so pass it as a list.
    """

    type: Literal["audio_embeds"] = "audio_embeds"

    audio_embeds: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "s", "h", dynamic_dims={"s"}),
    ]


MiniCPMOAudioInputs: TypeAlias = (
    MiniCPMOAudioFeatureInputs | MiniCPMOAudioEmbeddingInputs
)


def _minicpmo_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    return dict(
        **_minicpmv_field_config(hf_inputs),
        audio_features=MultiModalFieldConfig.batched("audio"),
        audio_feature_lens=MultiModalFieldConfig.batched("audio"),
        audio_embeds=MultiModalFieldConfig.batched("audio"),
    )


class MiniCPMOAudioEmbeddingItems(DictEmbeddingItems):
    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        fields_factory: Callable[
            [Mapping[str, torch.Tensor]],
            Mapping[str, MultiModalFieldConfig],
        ],
    ) -> None:
        super().__init__(
            data,
            modality="image",
            required_fields={"audio_embeds"},
            fields_factory=fields_factory,
        )


class MiniCPMOMultiModalDataParser(MiniCPMVMultiModalDataParser):
    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[AudioItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return MiniCPMOAudioEmbeddingItems(
                data,
                fields_factory=_minicpmo_field_config,
            )

        return super()._parse_audio_data(data)


class MiniCPMOProcessingInfo(MiniCPMVProcessingInfo):
    audio_pattern = "(<audio>./</audio>)"

    def get_data_parser(self):
        return MiniCPMOMultiModalDataParser(
            target_sr=self.get_default_audio_sampling_rate(),
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {**super().get_supported_mm_limits(), "audio": None}

    def get_audio_placeholder(
        self,
        audio_lens: int,
        chunk_input: bool = True,
        chunk_length: int = 1,
    ) -> str:
        hf_processor = self.get_hf_processor()

        return hf_processor.get_audio_placeholder(
            audio_lens,
            chunk_input=chunk_input,
            chunk_length=chunk_length,
        )

    def get_default_audio_pool_step(self) -> int:
        hf_config = self.get_hf_config()
        # MiniCPM-o 4.5 uses pool_step=5, older versions use 2
        return getattr(hf_config, "audio_pool_step", 2)

    def get_default_audio_sampling_rate(self) -> int:
        return 16000

    def get_chunk_length(self) -> int:
        return self.get_hf_config().audio_chunk_length

    def get_max_audio_tokens_per_chunk(self) -> int:
        pool_step = self.get_default_audio_pool_step()
        fbank_feat_in_chunk = 100
        cnn_feat_in_chunk = (fbank_feat_in_chunk - 1) // 2 + 1
        return (cnn_feat_in_chunk - pool_step) // pool_step + 1

    def get_max_audio_chunks_with_most_features(self) -> int:
        return 30

    def get_max_audio_tokens(self) -> int:
        num_chunks = self.get_max_audio_chunks_with_most_features()
        return self.get_max_audio_tokens_per_chunk() * num_chunks

    def get_audio_len_by_num_chunks(self, num_chunks: int) -> int:
        sampling_rate = self.get_default_audio_sampling_rate()
        num_tokens_per_chunk = self.get_max_audio_tokens_per_chunk()
        return int(num_chunks * sampling_rate / num_tokens_per_chunk) + 1

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        max_images = mm_counts.get("image", 0)
        max_videos = mm_counts.get("video", 0)
        max_audios = mm_counts.get("audio", 0)

        max_image_tokens = self.get_max_image_tokens() * max_images
        max_audio_tokens = self.get_max_audio_tokens() * max_audios
        max_total_frames = self.get_max_video_frames(
            seq_len - max_image_tokens - max_audio_tokens
        )
        max_frames_per_video = min(
            max_total_frames // max(max_videos, 1), _MAX_FRAMES_PER_VIDEO
        )

        return max(max_frames_per_video, 1)


class MiniCPMODummyInputsBuilder(MiniCPMVDummyInputsBuilder[MiniCPMOProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        audio_prompt_texts = self.info.audio_pattern * num_audios

        return super().get_dummy_text(mm_counts) + audio_prompt_texts

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        audio_len = (
            self.info.get_max_audio_chunks_with_most_features()
            * self.info.get_default_audio_sampling_rate()
        )

        audio_overrides = mm_options.get("audio")

        audio_mm_data = {
            "audio": self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }

        return {
            **super().get_dummy_mm_data(seq_len, mm_counts, mm_options),
            **audio_mm_data,
        }


class MiniCPMOMultiModalProcessor(MiniCPMVMultiModalProcessor[MiniCPMOProcessingInfo]):
    def get_audio_prompt_texts(
        self,
        audio_lens: int,
        chunk_input: bool = True,
        chunk_length: int = 1,
    ) -> str:
        return self.info.get_audio_placeholder(
            audio_lens,
            chunk_input=chunk_input,
            chunk_length=chunk_length,
        )

    def process_audios(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        if (audios := mm_data.get("audios")) is None:
            return {}

        mm_items = self.info.parse_mm_data({"audio": audios}, validate=False)
        parsed_audios = mm_items.get_items(
            "audio", (MiniCPMOAudioEmbeddingItems, AudioProcessorItems)
        )

        if isinstance(parsed_audios, MiniCPMOAudioEmbeddingItems):
            audio_inputs = {}
        else:
            audio_inputs = self._base_call_hf_processor(
                prompts=[self.info.audio_pattern] * len(parsed_audios),
                mm_data={"audios": [[audio] for audio in parsed_audios]},
                mm_kwargs={**mm_kwargs, "chunk_input": True},
                tok_kwargs=tok_kwargs,
                out_keys={"audio_features", "audio_feature_lens"},
            )

            # Avoid padding since we need the output for each audio to be
            # independent of other audios for the cache to work correctly
            unpadded_audio_features = [
                feat[:, :feature_len]
                for feat, feature_len in zip(
                    audio_inputs["audio_features"],
                    audio_inputs["audio_feature_lens"],
                )
            ]
            audio_inputs["audio_features"] = unpadded_audio_features

        return audio_inputs

    def process_mm_inputs(
        self,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        return {
            **super().process_mm_inputs(mm_data, mm_kwargs, tok_kwargs),
            **self.process_audios(mm_data, mm_kwargs, tok_kwargs),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        base_updates = super()._get_prompt_updates(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            out_mm_kwargs=out_mm_kwargs,
        )

        audio_placeholder = self.info.audio_pattern

        def get_audio_replacement(item_idx: int):
            audios = mm_items.get_items(
                "audio", (MiniCPMOAudioEmbeddingItems, AudioProcessorItems)
            )

            if isinstance(audios, MiniCPMOAudioEmbeddingItems):
                single_audio_embeds = audios.get(item_idx)["audio_embeds"]
                audio_len = self.info.get_audio_len_by_num_chunks(
                    sum(map(len, single_audio_embeds))
                )
            else:
                audio_len = audios.get_audio_length(item_idx)

            return PromptUpdateDetails.select_text(
                self.get_audio_prompt_texts(audio_len),
                "<unk>",
            )

        return [
            *base_updates,
            PromptReplacement(
                modality="audio",
                target=audio_placeholder,
                replacement=get_audio_replacement,
            ),
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _minicpmo_field_config(hf_inputs)


class MultiModalProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.relu(self.linear1(audio_features))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class MiniCPMWhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig, layer_idx: int):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        outputs = (hidden_states,)

        return outputs


class MiniCPMWhisperEncoder(WhisperEncoder):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                MiniCPMWhisperEncoderLayer(config, layer_idx=i)
                for i in range(config.encoder_layers)
            ]
        )

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPast:
        # Ignore copy
        input_features = input_features.to(
            dtype=self.conv1.weight.dtype, device=self.conv1.weight.device
        )

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight

        embed_pos = embed_pos[: inputs_embeds.shape[1], :]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        encoder_states = ()

        for idx, encoder_layer in enumerate(self.layers):
            encoder_states = encoder_states + (hidden_states,)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            # Ignore copy
            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                )

                hidden_states = layer_outputs[0]

        hidden_states = self.layer_norm(hidden_states)
        encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
        )


class MiniCPMOBaseModel:
    """Base mixin class for MiniCPM-O models with audio support."""

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "(<image>./</image>)"
        if modality.startswith("video"):
            return "(<video>./</video>)"
        if modality.startswith("audio"):
            return "(<audio>./</audio>)"

        raise ValueError("Only image, video or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        with self._mark_tower_model(vllm_config, "audio"):
            self.apm = self.init_audio_module(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "apm")
            )

    def init_audio_module(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Do not use parameters temporarily
        audio_config = self.config.audio_config
        model = MiniCPMWhisperEncoder(audio_config)
        audio_output_dim = int(audio_config.encoder_ffn_dim // 4)
        self.audio_avg_pooler = nn.AvgPool1d(
            self.config.audio_pool_step, stride=self.config.audio_pool_step
        )
        self.audio_projection_layer = MultiModalProjector(
            in_dim=audio_output_dim, out_dim=self.embed_dim
        )
        self.audio_encoder_layer = -1
        return model

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["tts"])
        return loader.load_weights(weights)

    def subsequent_chunk_mask(
        self,
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = CPU_DEVICE,
        num_lookhead: int = 0,
    ) -> torch.Tensor:
        ret = torch.zeros(size, size, device=device, dtype=torch.bool)
        # Vectorized computation of row indices and chunk boundaries
        row_indices = torch.arange(size, device=device)
        chunk_indices = row_indices // chunk_size
        if num_left_chunks < 0:
            # If num_left_chunks < 0, start is always 0 for all rows
            start_indices = torch.zeros_like(row_indices)
        else:
            # Compute start indices vectorially
            start_chunk_indices = torch.clamp(chunk_indices - num_left_chunks, min=0)
            start_indices = start_chunk_indices * chunk_size
        # Compute ending indices vectorially
        end_chunk_indices = chunk_indices + 1
        end_indices = torch.clamp(
            end_chunk_indices * chunk_size + num_lookhead, max=size
        )
        # Create column indices for broadcasting
        col_indices = torch.arange(size, device=device).unsqueeze(0)
        start_indices = start_indices.unsqueeze(1)
        end_indices = end_indices.unsqueeze(1)
        # Vectorized mask creation
        ret = (col_indices >= start_indices) & (col_indices < end_indices)
        return ret

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn - self.config.audio_pool_step
        ) // self.config.audio_pool_step + 1
        input_lengths_after_pooling = input_lengths_after_pooling.to(dtype=torch.int32)

        return input_lengths_after_cnn, input_lengths_after_pooling

    def get_audio_hidden_states(
        self, data: MiniCPMOAudioFeatureInputs
    ) -> list[torch.Tensor]:
        chunk_length = self.config.audio_chunk_length

        # (bs, 80, frames) or [], multi audios need filled in advance
        wavforms_raw = data["audio_features"]
        if isinstance(wavforms_raw, list):
            B = len(wavforms_raw)
            C = wavforms_raw[0].shape[-2]
            L = max(item.shape[-1] for item in wavforms_raw)
            device = wavforms_raw[0].device
            dtype = wavforms_raw[0].dtype

            wavforms = torch.zeros((B, C, L), dtype=dtype, device=device)
            for i, wavforms_item in enumerate(wavforms_raw):
                L_item = wavforms_item.shape[-1]
                wavforms[i, ..., :L_item] = wavforms_item
        else:
            wavforms = wavforms_raw

        # list, [[x1, x2], [y1], [z1]]
        audio_feature_lens_raw = data["audio_feature_lens"]
        if isinstance(audio_feature_lens_raw, torch.Tensor):
            audio_feature_lens_raw = audio_feature_lens_raw.unbind(0)

        audio_feature_lens = torch.hstack(audio_feature_lens_raw)
        batch_size, _, max_mel_seq_len = wavforms.shape
        max_seq_len = (max_mel_seq_len - 1) // 2 + 1

        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(
                0,
                max_seq_len,
                dtype=audio_feature_lens.dtype,
                device=audio_feature_lens.device,
            )
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feature_lens.unsqueeze(1).expand(batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range >= lengths_expand  # 1 for padded values

        audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
            batch_size, 1, max_seq_len, max_seq_len
        )
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.apm.conv1.weight.dtype, device=self.apm.conv1.weight.device
        )

        if chunk_length > 0:
            chunk_num_frame = int(chunk_length * 50)
            chunk_mask = self.subsequent_chunk_mask(
                size=max_seq_len,
                chunk_size=chunk_num_frame,
                num_left_chunks=-1,
                device=audio_attention_mask_.device,
            )
            audio_attention_mask_ = torch.logical_or(
                audio_attention_mask_, torch.logical_not(chunk_mask)
            )

        audio_attention_mask[audio_attention_mask_] = float("-inf")
        audio_states = self.apm(
            wavforms, attention_mask=audio_attention_mask
        ).hidden_states[self.audio_encoder_layer]
        audio_embeds = self.audio_projection_layer(audio_states)

        audio_embeds = audio_embeds.transpose(1, 2)
        audio_embeds = self.audio_avg_pooler(audio_embeds)
        audio_embeds = audio_embeds.transpose(1, 2)

        _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(
            audio_feature_lens
        )

        num_audio_tokens = feature_lens_after_pooling

        final_audio_embeds = list[torch.Tensor]()
        idx = 0
        for i in range(len(audio_feature_lens_raw)):
            target_audio_embeds_lst = list[torch.Tensor]()
            for _ in range(len(audio_feature_lens_raw[i])):
                target_audio_embeds_lst.append(
                    audio_embeds[idx, : num_audio_tokens[idx], :]
                )
                idx += 1

            final_audio_embeds.append(torch.cat(target_audio_embeds_lst))

        return final_audio_embeds

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> MiniCPMOAudioInputs | None:
        audio_features = kwargs.pop("audio_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)

        if audio_features is None and audio_embeds is None:
            return None

        if audio_embeds is not None:
            return MiniCPMOAudioEmbeddingInputs(
                type="audio_embeds",
                audio_embeds=audio_embeds,
            )

        audio_feature_lens = kwargs.pop("audio_feature_lens")

        return MiniCPMOAudioFeatureInputs(
            type="audio_features",
            audio_features=audio_features,
            audio_feature_lens=audio_feature_lens,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = super()._parse_and_validate_multimodal_inputs(**kwargs)

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("audio_features", "audio_embeds")
                and "audios" not in modalities
            ):
                modalities["audios"] = self._parse_and_validate_audio_input(**kwargs)

        return modalities

    def _process_audio_input(
        self,
        audio_input: MiniCPMOAudioInputs,
    ) -> torch.Tensor | list[torch.Tensor]:
        if audio_input["type"] == "audio_embeds":
            return audio_input["audio_embeds"]

        return self.get_audio_hidden_states(audio_input)

    def _process_multimodal_inputs(self, modalities: dict):
        multimodal_embeddings = super()._process_multimodal_inputs(modalities)

        for modality in modalities:
            if modality == "audios":
                audio_input = modalities["audios"]
                audio_embeddings = self._process_audio_input(audio_input)
                multimodal_embeddings += tuple(audio_embeddings)

        return multimodal_embeddings


class MiniCPMO2_6(MiniCPMOBaseModel, MiniCPMV2_6):
    """MiniCPM-O 2.6 model with Qwen2 backbone."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        with self._mark_tower_model(vllm_config, "audio"):
            self.apm = self.init_audio_module(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "apm")
            )


class MiniCPMO4_5(MiniCPMOBaseModel, MiniCPMV4_5, SupportsAudioOutput):
    """MiniCPM-O 4.5 model with Qwen3 backbone.

    Supports audio output via the Token2wav TTS pipeline
    (``SupportsAudioOutput``).

    By default, TTS weights (``tts.*`` in the checkpoint) are skipped so that
    serving text-only requests is not penalised by the extra VRAM.  Pass
    ``--hf-overrides '{"enable_audio_output": true}'`` (or set
    ``enable_audio_output=True`` in the model config) to load the TTS weights
    in-process and enable ``decode_audio_tokens()``.

    When ``enable_audio_output=True``:
    * ``self.tts`` is initialised as a ``MiniCPMTTS`` module via
      ``trust_remote_code`` (requires ``--trust-remote-code``).
    * ``Token2wav`` (``self.tts.audio_tokenizer``) is loaded automatically
      from ``assets/token2wav/`` in the model directory at the end of
      ``load_weights()``.  No separate ``init_tts()`` call is required.
    """

    # SupportsAudioOutput class variables
    supports_audio_output: ClassVar[Literal[True]] = True
    audio_output_sample_rate: ClassVar[int] = 24000

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        with self._mark_tower_model(vllm_config, "audio"):
            self.apm = self.init_audio_module(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "apm")
            )

        # Store for use in load_weights() (Token2wav path resolution).
        self._model_name_or_path: str = vllm_config.model_config.model

        if getattr(self.config, "enable_audio_output", False):
            self._init_tts_module(vllm_config)

    def _init_tts_module(self, vllm_config: VllmConfig) -> None:
        """Initialise ``self.tts`` (``MiniCPMTTS``) when
        ``enable_audio_output=True``.

        ``MiniCPMTTS`` is the TTS sub-model in the HuggingFace
        ``modeling_minicpmo.py``.  It is loaded via ``trust_remote_code``
        so that weight loading works correctly.  If the class cannot be
        resolved a warning is logged and TTS is left uninitialised —
        ``decode_audio_tokens()`` will raise at call time with a clear
        message.
        """
        tts_config = getattr(self.config, "tts_config", None)
        if tts_config is None:
            logger.warning(
                "enable_audio_output=True but tts_config is absent from "
                "model config; TTS module will not be initialised."
            )
            return

        model_name = vllm_config.model_config.model
        trust_remote = vllm_config.model_config.trust_remote_code
        # MiniCPM-o 4.5 uses MiniCPMTTS (not ConditionalChatTTS which was
        # the class name in earlier model versions).
        tts_cls = try_get_class_from_dynamic_module(
            "modeling_minicpmo.MiniCPMTTS",
            model_name,
            trust_remote_code=trust_remote,
        )
        if tts_cls is None:
            logger.warning(
                "Could not load MiniCPMTTS from %s "
                "(trust_remote_code=%s). "
                "Start vLLM with --trust-remote-code to enable audio output.",
                model_name,
                trust_remote,
            )
            return

        self.tts: nn.Module = tts_cls(config=tts_config, audio_tokenizer=None)
        logger.info(
            "MiniCPMTTS initialised for audio output "
            "(Token2wav will be loaded from assets/token2wav/ in load_weights)."
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        enable_tts = getattr(self.config, "enable_audio_output", False)
        skip: list[str] = [] if enable_tts else ["tts"]
        loader = AutoWeightsLoader(self, skip_prefixes=skip)
        loaded = loader.load_weights(weights)
        if enable_tts and hasattr(self, "tts"):
            self._init_token2wav(self._model_name_or_path)
        return loaded

    def _init_token2wav(self, model_name_or_path: str) -> None:
        """Load ``Token2wav`` from ``assets/token2wav/`` and assign to
        ``self.tts.audio_tokenizer``.

        ``Token2wav`` is provided by the ``stepaudio2`` package
        (``pip install minicpmo-utils[all]``).  The assets are expected at
        ``assets/token2wav/`` inside the model directory (local or HF hub).
        If loading fails a warning is logged and
        ``decode_audio_tokens()`` will raise at call time.
        """
        try:
            from stepaudio2 import Token2wav
        except ImportError:
            logger.warning(
                "stepaudio2 is not installed; audio output will not be "
                "available.  Install with: pip install minicpmo-utils[all]"
            )
            return

        if os.path.isdir(model_name_or_path):
            token2wav_dir = os.path.join(
                model_name_or_path, "assets", "token2wav"
            )
        else:
            try:
                from huggingface_hub import snapshot_download

                repo_dir = snapshot_download(
                    repo_id=model_name_or_path,
                    allow_patterns=["assets/token2wav/**"],
                )
                token2wav_dir = os.path.join(repo_dir, "assets", "token2wav")
            except Exception as exc:
                logger.warning(
                    "Could not download assets/token2wav/ from %s: %s. "
                    "Audio output will not be available.",
                    model_name_or_path,
                    exc,
                )
                return

        if not os.path.isdir(token2wav_dir):
            logger.warning(
                "assets/token2wav/ not found at %s. "
                "Audio output will not be available.",
                token2wav_dir,
            )
            return

        try:
            self.tts.audio_tokenizer = Token2wav(token2wav_dir)  # type: ignore[union-attr]
            logger.info("Token2wav loaded from %s.", token2wav_dir)
        except Exception as exc:
            logger.warning(
                "Failed to load Token2wav from %s: %s. "
                "Audio output will not be available.",
                token2wav_dir,
                exc,
            )

    def decode_audio_tokens(
        self,
        token_ids: list[int],
    ) -> np.ndarray:
        """Decode audio token IDs to a waveform via MiniCPMTTS + Token2wav.

        MiniCPM-o 4.5 uses a two-stage TTS pipeline:

        1. The main LLM generates text that includes TTS spans delimited by
           ``<|tts_bos|>`` and ``<|tts_eos|>`` special tokens.
        2. ``MiniCPMTTS`` (``self.tts``) is an autoregressive model that
           takes the TTS text and generates VQ audio codes.
        3. ``Token2wav`` (``self.tts.audio_tokenizer``, from the
           ``stepaudio2`` package) decodes the VQ codes into a waveform.

        The model must be started with
        ``--hf-overrides '{"enable_audio_output": true}'`` and
        ``--trust-remote-code``.  The ``stepaudio2`` package must be
        installed (``pip install minicpmo-utils[all]``).

        Args:
            token_ids: Token IDs from the main LLM output that include the
                TTS text span between ``<|tts_bos|>`` and ``<|tts_eos|>``.

        Returns:
            Float32 waveform, shape ``[num_samples]``.

        Raises:
            RuntimeError: If TTS is not initialised or Token2wav is not
                loaded (see individual error messages for remediation).
        """
        if not hasattr(self, "tts"):
            raise RuntimeError(
                "Audio output not initialised. "
                "Start vLLM with --hf-overrides '{\"enable_audio_output\": true}' "
                "and --trust-remote-code."
            )
        audio_tok = getattr(self.tts, "audio_tokenizer", None)
        if audio_tok is None:
            raise RuntimeError(
                "Token2wav audio tokenizer not loaded. "
                "Install stepaudio2 via: pip install minicpmo-utils[all] "
                "and ensure assets/token2wav/ is present in the model directory."
            )
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "self.tokenizer is required to decode token_ids to text "
                "for TTS synthesis."
            )
        import io
        import tempfile

        import soundfile as sf

        # torchaudio 2.10+ defaults to TorchCodec which requires FFmpeg.
        # Force the soundfile (libsndfile) backend so that s3tokenizer's
        # internal torchaudio.load() call works without FFmpeg installed.
        # The env-var is checked at call time (not import time), so setting
        # it here before audio_tokenizer() is invoked is sufficient.
        os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"
        try:
            import torchaudio
            torchaudio.set_audio_backend("soundfile")
        except Exception:
            pass  # API absent in some torchaudio versions; env-var is primary

        # Extract the TTS-destined span between the special markers.
        text: str = tokenizer.decode(token_ids, skip_special_tokens=False)
        if "<|tts_bos|>" in text:
            text = text.split("<|tts_bos|>")[-1]
        if "<|tts_eos|>" in text:
            text = text.split("<|tts_eos|>")[0]

        # Re-encode the extracted text span to obtain clean TTS input IDs.
        tts_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)

        device = self.tts.emb_text.weight.device

        # Build inputs_embeds for MiniCPMTTS.generate().
        # MiniCPM-o 4.5 condition_type="hidden_text_merge" normally merges
        # emb_text(token_ids) with projector_semantic(llm_hidden_states).
        # In the vLLM serving path LLM hidden states are not available, so
        # only the token embedding component is used.  This is a known quality
        # compromise versus the full HF chat() pipeline; the hidden-state path
        # requires engine-level changes to surface per-token hidden states.
        tts_tokens = torch.tensor(tts_ids, device=device, dtype=torch.long)
        tts_embeds = self.tts.emb_text(tts_tokens)  # [L, H]
        text_eos_embed = self.tts.emb_text(
            torch.tensor(
                [self.tts.config.text_eos_token_id],
                device=device,
                dtype=torch.long,
            )
        )  # [1, H]
        audio_bos_embed = self.tts.emb_text(
            torch.tensor(
                [self.tts.audio_bos_token_id],
                device=device,
                dtype=torch.long,
            )
        )  # [1, H]
        # Shape: [1, L+2, H]
        inputs_embeds = torch.cat(
            [tts_embeds, text_eos_embed, audio_bos_embed], dim=0
        ).unsqueeze(0)

        eos_token = torch.tensor(
            [self.tts.config.num_audio_tokens - 1],
            dtype=torch.long,
            device=device,
        )
        # MiniCPMTTS.generate() is decorated with @torch.inference_mode().
        gen_out = self.tts.generate(
            inputs_embeds=inputs_embeds,
            eos_token=eos_token,
            show_tqdm=False,
        )

        # gen_out.new_ids: [1, T, num_vq] — VQ audio code sequences.
        # Token2wav.__call__(tokens, ref_wav_path) decodes VQ codes to WAV
        # bytes using campplus.onnx for speaker embedding.  ref_wav_path must
        # point to a real WAV file; None is not valid.  Write a 1-second
        # silent 16 kHz WAV to a temp file to serve as a neutral default
        # speaker reference when no reference audio is available.
        ref_wav: np.ndarray = np.zeros(16000, dtype=np.float32)
        tmp_ref_path: str | None = None
        wav_bytes: bytes = b""
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as f:
                tmp_ref_path = f.name
            sf.write(tmp_ref_path, ref_wav, 16000)
            wav_bytes = self.tts.audio_tokenizer(
                gen_out.new_ids.squeeze(0).tolist(),
                tmp_ref_path,
            )
        finally:
            if tmp_ref_path is not None:
                try:
                    os.unlink(tmp_ref_path)
                except OSError:
                    pass

        waveform, _ = sf.read(io.BytesIO(wav_bytes))
        return np.array(waveform, dtype=np.float32)


_MINICPMO_SUPPORT_VERSION = {
    (2, 6): MiniCPMO2_6,
    (4, 5): MiniCPMO4_5,
}


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMOMultiModalProcessor,
    info=MiniCPMOProcessingInfo,
    dummy_inputs=MiniCPMODummyInputsBuilder,
)
class MiniCPMO(MiniCPMOBaseModel, MiniCPMV2_6):
    """
    MiniCPM-O model with audio support.
    Different versions use different LLM backbones:
    - Version 2.6: Uses Qwen2
    - Version 4.5: Uses Qwen3
    """

    def __new__(cls, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config

        # Determine version from config
        if hasattr(config, "version"):
            try:
                version_str = str(config.version)
                version_parts = version_str.split(".")
                version = tuple(int(x) for x in version_parts[:2])
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid model version format in config: {config.version}. "
                    "Expected a dot-separated version string like '4.5'."
                ) from e
        else:
            # Default to 2.6 for backward compatibility
            version = (2, 6)

        # Dispatch class based on version
        instance_cls = _MINICPMO_SUPPORT_VERSION.get(version)
        if instance_cls is None:
            supported_versions = ", ".join(
                [f"{v[0]}.{v[1]}" for v in sorted(_MINICPMO_SUPPORT_VERSION.keys())]
            )
            raise ValueError(
                f"Currently, MiniCPMO only supports versions "
                f"{supported_versions}. Got version: {version}"
            )

        return instance_cls(vllm_config=vllm_config, prefix=prefix)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # This __init__ won't be called due to __new__ returning a different class
        pass
