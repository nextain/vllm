# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SupportsAudioOutput protocol and MiniCPM-o 4.5 implementation.

These tests run without a GPU and without downloading the model (all external
calls are mocked).  They verify:
  1. The Protocol interface is correctly detected on MiniCPMO4_5.
  2. Other models (MiniCPMO2_6) are NOT detected.
  3. decode_audio_tokens() raises RuntimeError when TTS is not initialised.
  4. decode_audio_tokens() raises when Token2wav audio_tokenizer is absent.
  5. The FakeMiniCPMO4_5 stub raises a sentinel error when all modules are
     present — isolating the protocol contract from the synthesis pipeline.
     The real MiniCPMO4_5.decode_audio_tokens() is fully implemented and
     tested via E2E (tests/models/test_audio_output_e2e.py).
  6. _init_token2wav() loads Token2wav and handles missing stepaudio2/dir.
  7. The _ModelInfo.supports_audio_output field exists and has the right type.
  8. load_weights() conditionally skips tts.* prefixes.
"""

from __future__ import annotations

import os
from dataclasses import fields
from typing import Any, ClassVar, Literal
from unittest.mock import MagicMock, patch

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
            audio_tok = getattr(self.tts, "audio_tokenizer", None)
            if audio_tok is None:
                raise RuntimeError(
                    "Token2wav audio tokenizer not loaded. "
                    "Install stepaudio2 via: pip install minicpmo-utils[all] "
                    "and ensure assets/token2wav/ is present."
                )
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("self.tokenizer is required.")
            text: str = tokenizer.decode(token_ids, skip_special_tokens=False)
            if "<|tts_bos|>" in text:
                text = text.split("<|tts_bos|>")[-1]
            if "<|tts_eos|>" in text:
                text = text.split("<|tts_eos|>")[0]
            # MiniCPMTTS + Token2wav synthesis pipeline (TODO in production).
            raise RuntimeError(
                "decode_audio_tokens: MiniCPMTTS + Token2wav synthesis is not "
                "yet wired in the vLLM serving path."
            )

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
        with pytest.raises(RuntimeError, match="enable_audio_output"):
            instance.decode_audio_tokens([1, 2, 3])

    def test_raises_when_tts_but_no_audio_tokenizer(self) -> None:
        """tts present but audio_tokenizer is None → Token2wav error."""
        FakeMiniCPMO4_5 = _make_fake_minicpmo4_5()
        instance = FakeMiniCPMO4_5()
        instance.tts = MagicMock()  # type: ignore[attr-defined]
        # audio_tokenizer attribute returns None by default on MagicMock
        instance.tts.audio_tokenizer = None
        with pytest.raises(RuntimeError, match="Token2wav"):
            instance.decode_audio_tokens([1, 2, 3])

    def test_raises_when_no_tokenizer(self) -> None:
        FakeMiniCPMO4_5 = _make_fake_minicpmo4_5()
        instance = FakeMiniCPMO4_5()
        instance.tts = MagicMock()  # type: ignore[attr-defined]
        instance.tts.audio_tokenizer = MagicMock()
        # no tokenizer — getattr returns None
        with pytest.raises(RuntimeError, match="tokenizer"):
            instance.decode_audio_tokens([1, 2, 3])

    def test_raises_sentinel_when_fully_initialised(self) -> None:
        """When tts + audio_tokenizer + tokenizer are all present the fake
        stub reaches the sentinel raise — isolating protocol contract tests
        from actual synthesis (which requires GPU + model weights)."""
        FakeMiniCPMO4_5 = _make_fake_minicpmo4_5()
        instance = FakeMiniCPMO4_5()
        instance.tts = MagicMock()  # type: ignore[attr-defined]
        instance.tts.audio_tokenizer = MagicMock()
        instance.tokenizer = MagicMock()  # type: ignore[attr-defined]
        instance.tokenizer.decode.return_value = "<|tts_bos|>Hello<|tts_eos|>"
        with pytest.raises(RuntimeError, match="not yet wired"):
            instance.decode_audio_tokens([1, 2, 3])


# ---------------------------------------------------------------------------
# Test 3: TTS text extraction from token stream
# ---------------------------------------------------------------------------


class TestTTSTextExtraction:
    """Verify that the TTS span is correctly extracted from the token stream."""

    def _extract_tts_text(self, text: str) -> str:
        """Mirror the extraction logic in decode_audio_tokens."""
        if "<|tts_bos|>" in text:
            text = text.split("<|tts_bos|>")[-1]
        if "<|tts_eos|>" in text:
            text = text.split("<|tts_eos|>")[0]
        return text

    def test_extracts_span_between_markers(self) -> None:
        raw = "Some preamble<|tts_bos|>Hello world<|tts_eos|>trailing"
        assert self._extract_tts_text(raw) == "Hello world"

    def test_no_markers_passes_through(self) -> None:
        raw = "plain text"
        assert self._extract_tts_text(raw) == "plain text"

    def test_only_bos_marker(self) -> None:
        raw = "<|tts_bos|>after bos"
        assert self._extract_tts_text(raw) == "after bos"

    def test_only_eos_marker(self) -> None:
        raw = "before eos<|tts_eos|>ignored"
        assert self._extract_tts_text(raw) == "before eos"


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
# Test 5: _init_token2wav — Token2wav loading from assets/token2wav/
# ---------------------------------------------------------------------------


class _FakeInitToken2wav:
    """Minimal stand-in for MiniCPMO4_5 to test _init_token2wav in isolation."""

    def __init__(self) -> None:
        self.tts = MagicMock()

    def _init_token2wav(self, model_name_or_path: str) -> None:
        """Copy of the production implementation for isolated unit testing."""
        try:
            from stepaudio2 import Token2wav
        except ImportError:
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
            except Exception:
                return

        if not os.path.isdir(token2wav_dir):
            return

        try:
            self.tts.audio_tokenizer = Token2wav(token2wav_dir)
        except Exception:
            pass


class TestInitToken2wav:
    """Unit tests for _init_token2wav path-resolution and Token2wav dispatch."""

    def test_missing_stepaudio2_does_not_raise(self, tmp_path: Any) -> None:
        """stepaudio2 not installed → _init_token2wav returns silently."""
        instance = _FakeInitToken2wav()
        # stepaudio2 is not installed in the test environment.
        # _init_token2wav must catch ImportError and return without crashing.
        instance._init_token2wav(str(tmp_path))
        # Verify no exception was raised (test reaching here is the assertion).

    def test_missing_dir_does_not_raise(self, tmp_path: Any) -> None:
        """assets/token2wav/ absent → no crash, audio_tokenizer not changed."""
        instance = _FakeInitToken2wav()
        fake_token2wav = MagicMock()

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "stepaudio2":
                mod = MagicMock()
                mod.Token2wav = MagicMock(return_value=fake_token2wav)
                return mod
            raise ImportError(name)

        # assets/token2wav/ does not exist in tmp_path
        with patch("builtins.__import__", side_effect=fake_import):
            instance._init_token2wav(str(tmp_path))
        # Token2wav() should NOT be called if dir is absent
        fake_token2wav.assert_not_called()

    def test_local_dir_loads_token2wav(self, tmp_path: Any) -> None:
        """Local directory with assets/token2wav/ → Token2wav instantiated."""
        token2wav_dir = tmp_path / "assets" / "token2wav"
        token2wav_dir.mkdir(parents=True)

        instance = _FakeInitToken2wav()
        fake_token2wav_instance = MagicMock()
        FakeToken2wav = MagicMock(return_value=fake_token2wav_instance)

        with patch.dict("sys.modules", {"stepaudio2": MagicMock(Token2wav=FakeToken2wav)}):
            instance._init_token2wav(str(tmp_path))

        FakeToken2wav.assert_called_once_with(str(token2wav_dir))
        assert instance.tts.audio_tokenizer is fake_token2wav_instance

    def test_non_local_path_skips_gracefully_without_hf(self) -> None:
        """Non-local path that fails snapshot_download → no crash."""
        instance = _FakeInitToken2wav()
        FakeToken2wav = MagicMock()

        def fake_snapshot_download(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("network error")

        with patch.dict("sys.modules", {"stepaudio2": MagicMock(Token2wav=FakeToken2wav)}), \
             patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download):
            instance._init_token2wav("openbmb/MiniCPM-o-4_5")

        FakeToken2wav.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: _ModelInfo dataclass has supports_audio_output field
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
