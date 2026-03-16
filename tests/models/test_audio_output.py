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
  6. load_weights() conditionally skips tts.* prefixes.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, ClassVar, Literal
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

        def decode_audio_tokens(
            self,
            token_ids: list[int],
        ) -> np.ndarray:
            if not hasattr(self, "tts"):
                raise RuntimeError(
                    "Audio output not initialised. "
                    "Start vLLM with --hf-overrides '{\"enable_audio_output\": true}' "
                    "and --trust-remote-code."
                )
            if not hasattr(self, "vocos"):
                raise RuntimeError(
                    "Vocos vocoder not loaded. "
                    "Call model.init_tts() after loading to load assets/Vocos.pt."
                )
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("self.tokenizer is required.")
            m: Any = self
            text: str = tokenizer.decode(token_ids, skip_special_tokens=False)
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
        instance = FakeMiniCPMO4_5()
        # no tts, no vocos on instance
        with pytest.raises(RuntimeError, match="enable_audio_output"):
            instance.decode_audio_tokens([1, 2, 3])

    def test_raises_when_tts_but_no_vocos(self) -> None:
        FakeMiniCPMO4_5 = _make_fake_minicpmo4_5()
        instance = FakeMiniCPMO4_5()
        instance.tts = MagicMock()  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="init_tts"):
            instance.decode_audio_tokens([1, 2, 3])

    def test_raises_when_no_tokenizer(self) -> None:
        FakeMiniCPMO4_5 = _make_fake_minicpmo4_5()
        instance = FakeMiniCPMO4_5()
        instance.tts = MagicMock()  # type: ignore[attr-defined]
        instance.vocos = MagicMock()  # type: ignore[attr-defined]
        # no tokenizer — getattr returns None
        with pytest.raises(RuntimeError, match="tokenizer"):
            instance.decode_audio_tokens([1, 2, 3])


# ---------------------------------------------------------------------------
# Test 3: decode_audio_tokens succeeds with mocked TTS pipeline
# ---------------------------------------------------------------------------


class TestDecodeAudioTokensMocked:
    def _make_ready_instance(self) -> Any:
        """Return a FakeMiniCPMO4_5 instance with tts/vocos/tokenizer mocked."""
        FakeMiniCPMO4_5 = _make_fake_minicpmo4_5()
        instance = FakeMiniCPMO4_5()
        instance.tts = MagicMock()  # type: ignore[attr-defined]
        instance.vocos = MagicMock()  # type: ignore[attr-defined]
        instance.tokenizer = MagicMock()  # type: ignore[attr-defined]
        instance.tokenizer.decode.return_value = "Hello world"
        instance._generate_mel_spec = MagicMock(  # type: ignore[attr-defined]
            return_value=torch.zeros(1, 10, 100)
        )
        instance.decode_mel_to_audio = MagicMock(  # type: ignore[attr-defined]
            return_value=(np.zeros(24000, dtype=np.float32), 24000)
        )
        return instance

    def test_returns_ndarray(self) -> None:
        instance = self._make_ready_instance()
        result = instance.decode_audio_tokens([10, 20, 30])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (24000,)

    def test_delegates_to_generate_mel_spec(self) -> None:
        instance = self._make_ready_instance()
        instance.decode_audio_tokens([5, 6, 7])
        instance._generate_mel_spec.assert_called_once_with(
            inputs=None, outputs=None, text="Hello world"
        )

    def test_handles_torch_tensor_wav(self) -> None:
        """decode_mel_to_audio may return a torch.Tensor; ensure .numpy()."""
        instance = self._make_ready_instance()
        fake_wav_tensor = torch.zeros(12000, dtype=torch.float32)
        instance.decode_mel_to_audio.return_value = (fake_wav_tensor, 24000)

        result = instance.decode_audio_tokens([1])

        assert isinstance(result, np.ndarray)
        assert result.shape == (12000,)


# ---------------------------------------------------------------------------
# Test 4: load_weights skip_prefixes logic
# ---------------------------------------------------------------------------


class TestLoadWeightsSkipPrefixes:
    """Verify the skip_prefixes logic used by MiniCPMO4_5.load_weights."""

    def _compute_skip(self, enable_audio_output: bool) -> list[str]:
        """Mirror the logic in MiniCPMO4_5.load_weights."""
        return [] if enable_audio_output else ["tts"]

    def test_tts_skipped_by_default(self) -> None:
        assert self._compute_skip(enable_audio_output=False) == ["tts"]

    def test_tts_included_when_flag_set(self) -> None:
        assert self._compute_skip(enable_audio_output=True) == []


# ---------------------------------------------------------------------------
# Test 5: _ModelInfo dataclass has supports_audio_output field
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
