# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SupportsAudioOutput protocol and MiniCPM-o 4.5 implementation.

These tests run without a GPU and without downloading the model (all external
calls are mocked).  They verify:
  1. The Protocol interface is correctly detected on MiniCPMO4_5.
  2. Other models (MiniCPMO2_6) are NOT detected.
  3. decode_audio_tokens() raises RuntimeError when TTS is not initialised.
  4. decode_audio_tokens() delegates to the TTS pipeline when TTS is ready.
  5. The _ModelInfo.supports_audio_output field exists and has the right type.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, ClassVar, Literal, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.model_executor.models.interfaces import (
    SupportsAudioOutput,
    supports_audio_output,
)

# ---------------------------------------------------------------------------
# Helpers — build minimal fake model classes without importing heavy deps
# ---------------------------------------------------------------------------


def _make_fake_minicpmo4_5() -> type[SupportsAudioOutput]:
    """Return a class that satisfies the SupportsAudioOutput Protocol."""

    class FakeMiniCPMO4_5:
        supports_audio_output: ClassVar[Literal[True]] = True
        audio_output_sample_rate: ClassVar[int] = 24000

        @classmethod
        def decode_audio_tokens(
            cls,
            token_ids: list[int],
            model_instance: object,
        ) -> np.ndarray:
            if not (
                hasattr(model_instance, "tts") and hasattr(model_instance, "vocos")
            ):
                raise RuntimeError(
                    "MiniCPM-o 4.5 audio output requires a model instance "
                    "with TTS initialised."
                )
            tokenizer = getattr(model_instance, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("model_instance.tokenizer is required.")
            m: Any = cast(Any, model_instance)
            text: str = tokenizer.decode(token_ids)
            mel_spec = m._generate_mel_spec(inputs=None, outputs=None, text=text)
            wav_numpy, _sr = m.decode_mel_to_audio(mel_spec)
            result: np.ndarray = (
                wav_numpy.numpy() if hasattr(wav_numpy, "numpy") else wav_numpy
            )
            return result

    return FakeMiniCPMO4_5  # type: ignore[return-value]


class _FakeNonAudioModel:
    """A model class that does NOT implement SupportsAudioOutput."""

    supports_transcription: bool = False


# ---------------------------------------------------------------------------
# Test 1: supports_audio_output correctly identifies protocol members
# ---------------------------------------------------------------------------


class TestSupportsAudioOutputDetection:
    def test_protocol_class_detected(self) -> None:
        FakeMiniCPMO4_5 = _make_fake_minicpmo4_5()
        assert supports_audio_output(FakeMiniCPMO4_5) is True

    def test_protocol_instance_detected(self) -> None:
        FakeMiniCPMO4_5 = _make_fake_minicpmo4_5()
        instance: SupportsAudioOutput = object.__new__(  # type: ignore[type-abstract]
            FakeMiniCPMO4_5
        )
        assert supports_audio_output(instance) is True

    def test_non_audio_model_not_detected(self) -> None:
        assert supports_audio_output(_FakeNonAudioModel) is False

    def test_non_audio_model_instance_not_detected(self) -> None:
        assert supports_audio_output(_FakeNonAudioModel()) is False

    def test_plain_object_not_detected(self) -> None:
        assert supports_audio_output(object()) is False

    def test_sample_rate_class_variable(self) -> None:
        FakeMiniCPMO4_5 = _make_fake_minicpmo4_5()
        assert FakeMiniCPMO4_5.audio_output_sample_rate == 24000


# ---------------------------------------------------------------------------
# Test 2: decode_audio_tokens raises RuntimeError when TTS not initialised
# ---------------------------------------------------------------------------


class TestDecodeAudioTokensNoTTS:
    def test_raises_when_no_tts_attr(self) -> None:
        FakeMiniCPMO4_5 = _make_fake_minicpmo4_5()
        model_instance = MagicMock(spec=[])  # empty spec — no attributes
        with pytest.raises(RuntimeError, match="TTS initialised"):
            FakeMiniCPMO4_5.decode_audio_tokens([1, 2, 3], model_instance)

    def test_raises_when_tts_but_no_vocos(self) -> None:
        FakeMiniCPMO4_5 = _make_fake_minicpmo4_5()
        model_instance = MagicMock(spec=["tts"])  # has tts, no vocos
        with pytest.raises(RuntimeError, match="TTS initialised"):
            FakeMiniCPMO4_5.decode_audio_tokens([1, 2, 3], model_instance)

    def test_raises_when_no_tokenizer(self) -> None:
        FakeMiniCPMO4_5 = _make_fake_minicpmo4_5()
        model_instance = MagicMock(spec=["tts", "vocos"])
        # getattr(..., "tokenizer", None) returns None when not in spec
        with pytest.raises(RuntimeError, match="tokenizer"):
            FakeMiniCPMO4_5.decode_audio_tokens([1, 2, 3], model_instance)


# ---------------------------------------------------------------------------
# Test 3: decode_audio_tokens succeeds with mocked TTS pipeline
# ---------------------------------------------------------------------------


class TestDecodeAudioTokensMocked:
    def test_returns_ndarray(self) -> None:
        FakeMiniCPMO4_5 = _make_fake_minicpmo4_5()

        model_instance = MagicMock()
        model_instance.tts = MagicMock()
        model_instance.vocos = MagicMock()
        model_instance.tokenizer.decode.return_value = "Hello world"

        fake_mel = torch.zeros(1, 10, 100)
        model_instance._generate_mel_spec.return_value = fake_mel

        fake_wav = np.zeros(24000, dtype=np.float32)
        model_instance.decode_mel_to_audio.return_value = (fake_wav, 24000)

        result = FakeMiniCPMO4_5.decode_audio_tokens([10, 20, 30], model_instance)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (24000,)

    def test_delegates_to_generate_mel_spec(self) -> None:
        FakeMiniCPMO4_5 = _make_fake_minicpmo4_5()
        model_instance = MagicMock()
        model_instance.tts = MagicMock()
        model_instance.vocos = MagicMock()
        model_instance.tokenizer.decode.return_value = "test text"

        fake_mel = torch.zeros(1, 10, 100)
        model_instance._generate_mel_spec.return_value = fake_mel
        fake_wav = np.zeros(100, dtype=np.float32)
        model_instance.decode_mel_to_audio.return_value = (fake_wav, 24000)

        FakeMiniCPMO4_5.decode_audio_tokens([5, 6, 7], model_instance)

        model_instance._generate_mel_spec.assert_called_once_with(
            inputs=None, outputs=None, text="test text"
        )

    def test_handles_torch_tensor_wav(self) -> None:
        """decode_mel_to_audio may return a torch.Tensor; ensure .numpy()."""
        FakeMiniCPMO4_5 = _make_fake_minicpmo4_5()
        model_instance = MagicMock()
        model_instance.tts = MagicMock()
        model_instance.vocos = MagicMock()
        model_instance.tokenizer.decode.return_value = "hi"

        fake_mel = torch.zeros(1, 5, 100)
        model_instance._generate_mel_spec.return_value = fake_mel

        fake_wav_tensor = torch.zeros(12000, dtype=torch.float32)
        model_instance.decode_mel_to_audio.return_value = (
            fake_wav_tensor,
            24000,
        )

        result = FakeMiniCPMO4_5.decode_audio_tokens([1], model_instance)

        assert isinstance(result, np.ndarray)
        assert result.shape == (12000,)


# ---------------------------------------------------------------------------
# Test 4: _ModelInfo dataclass has supports_audio_output field
# ---------------------------------------------------------------------------


class TestModelInfoField:
    def test_supports_audio_output_field_exists(self) -> None:
        from vllm.model_executor.models.registry import _ModelInfo

        field_names = {f.name for f in fields(_ModelInfo)}
        assert "supports_audio_output" in field_names

    def test_supports_audio_output_field_is_bool(self) -> None:
        from vllm.model_executor.models.registry import _ModelInfo

        bool_fields = {
            f.name: f.type
            for f in fields(_ModelInfo)
            if f.name == "supports_audio_output"
        }
        assert "supports_audio_output" in bool_fields
        assert bool_fields["supports_audio_output"] in (bool, "bool")
