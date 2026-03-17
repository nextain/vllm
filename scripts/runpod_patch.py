#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Surgical patch for vllm 0.17.1 installed package.

Applies SupportsAudioOutput additions to interfaces.py and registry.py
without touching any ScoreType-related code (ScoreType does not exist in
0.17.1 and is unrelated to our changes).

Run on RunPod after ssh-ing in:
    python3 /workspace/vllm-patch/scripts/runpod_patch.py

Safe to run multiple times (idempotent).
"""
import sys

VLLM_ROOT = "/usr/local/lib/python3.11/dist-packages/vllm"

# ---------------------------------------------------------------------------
# interfaces.py — append SupportsAudioOutput Protocol + helper
# ---------------------------------------------------------------------------

INTERFACES_ADDITION = '''

# ---------------------------------------------------------------------------
# SupportsAudioOutput — text-in → audio-out (TTS) interface
# Added by nextain/vllm fork for MiniCPM-o 4.5 audio output support.
# ---------------------------------------------------------------------------

from typing import runtime_checkable  # noqa: F811 (re-import guard)
try:
    from typing import TypeIs  # Python 3.13+
except ImportError:
    from typing_extensions import TypeIs  # type: ignore[assignment]


@runtime_checkable
class SupportsAudioOutput(Protocol):
    """Interface for models that generate audio tokens as output.

    Models implementing this interface produce audio tokens alongside or
    instead of text tokens.  The token sequence is decoded by a vocoder
    (e.g. Token2wav) to produce waveform audio.

    Only MiniCPM-o 4.5 implements this interface today.  The Protocol is
    model-agnostic — other models can implement it without changes here.
    """

    supports_audio_output: ClassVar[Literal[True]] = True
    """Flag indicating audio-output capability.

    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """

    audio_output_sample_rate: ClassVar[int]
    """Sample rate (Hz) of generated audio.  E.g. 24000 for MiniCPM-o 4.5.

    Required.  Every implementing class must define this attribute.
    """

    def decode_audio_tokens(
        self,
        token_ids: list[int],
    ) -> "tuple[np.ndarray, str] | None":
        """Decode audio token IDs to a waveform and TTS transcript.

        Called on the serving model instance.  The model must have been started
        with ``--hf-overrides \'{"enable_audio_output": true}\'``.

        Args:
            token_ids: Token IDs from the main LLM output that contain the
                TTS text span (``<|tts_bos|>...<|tts_eos|>``).

        Returns:
            ``(waveform, transcript)`` where *waveform* is a float32 array
            with shape ``[num_samples]`` and *transcript* is the plain text
            of the TTS span.  Returns ``None`` if ``token_ids`` contain no
            TTS span (i.e. no ``<|tts_bos|>`` marker).

        Raises:
            RuntimeError: If TTS is not initialised on this model instance.
        """
        ...


@overload
def supports_audio_output(
    model: type[object],
) -> "TypeIs[type[SupportsAudioOutput]]": ...


@overload
def supports_audio_output(model: object) -> "TypeIs[SupportsAudioOutput]": ...


def supports_audio_output(
    model: "type[object] | object",
) -> "TypeIs[type[SupportsAudioOutput]] | TypeIs[SupportsAudioOutput]":
    return getattr(model, "supports_audio_output", False)
'''

# ---------------------------------------------------------------------------
# Patch interfaces.py
# ---------------------------------------------------------------------------

interfaces_path = f"{VLLM_ROOT}/model_executor/models/interfaces.py"
print(f"Patching {interfaces_path} ...")

with open(interfaces_path) as f:
    content = f.read()

if "SupportsAudioOutput" in content:
    print("  [SKIP] SupportsAudioOutput already present.")
else:
    # Ensure numpy is imported (it is in 0.17.1, but be safe)
    if "import numpy as np" not in content:
        content = content.replace(
            "import torch", "import numpy as np\nimport torch", 1
        )
    # Append the SupportsAudioOutput block
    content = content.rstrip() + "\n" + INTERFACES_ADDITION + "\n"
    with open(interfaces_path, "w") as f:
        f.write(content)
    print("  [OK] SupportsAudioOutput appended.")

# ---------------------------------------------------------------------------
# Patch registry.py
# ---------------------------------------------------------------------------

registry_path = f"{VLLM_ROOT}/model_executor/models/registry.py"
print(f"Patching {registry_path} ...")

with open(registry_path) as f:
    reg_content = f.read()

if "supports_audio_output" in reg_content:
    print("  [SKIP] supports_audio_output already present.")
else:
    # 1) Add supports_audio_output to the .interfaces import block.
    #    Find the closing paren of the 'from .interfaces import (' block.
    #    We look for 'supports_transcription,' (last alphabetically before ours)
    #    and insert after it, or just before the closing ')'.
    import re

    # Pattern: find the from .interfaces import block
    interfaces_import_pattern = r"(from \.interfaces import \([^)]+)"
    match = re.search(interfaces_import_pattern, reg_content, re.DOTALL)
    if match:
        old_import = match.group(0)
        # Add supports_audio_output after 'supports_pp,' or before closing paren
        if "supports_pp," in old_import:
            new_import = old_import.replace(
                "supports_pp,",
                "supports_audio_output,\n    supports_pp,",
            )
        else:
            # Fallback: insert before closing paren
            new_import = old_import.rstrip() + "\n    supports_audio_output,"
        reg_content = reg_content.replace(old_import, new_import, 1)
        print("  [OK] supports_audio_output added to .interfaces import.")
    else:
        print("  [WARN] Could not find .interfaces import block — manual edit needed.")

    # 2) Add supports_audio_output field to _ModelInfo dataclass.
    #    Insert before 'supports_transcription: bool'
    if "supports_transcription: bool" in reg_content:
        reg_content = reg_content.replace(
            "    supports_transcription: bool",
            "    supports_audio_output: bool\n"
            "    \"\"\"Whether the model implements SupportsAudioOutput.\"\"\"\n"
            "    supports_transcription: bool",
            1,
        )
        print("  [OK] supports_audio_output field added to _ModelInfo.")
    else:
        print("  [WARN] Could not find supports_transcription field — manual edit needed.")

    # 3) Add supports_audio_output= to from_model_cls() constructor call.
    #    Insert before 'supports_transcription=supports_transcription(model)'
    if "supports_transcription=supports_transcription(model)" in reg_content:
        reg_content = reg_content.replace(
            "            supports_transcription=supports_transcription(model)",
            "            supports_audio_output=supports_audio_output(model),\n"
            "            supports_transcription=supports_transcription(model)",
            1,
        )
        print("  [OK] supports_audio_output= added to from_model_cls().")
    else:
        print("  [WARN] Could not find supports_transcription= call — manual edit needed.")

    with open(registry_path, "w") as f:
        f.write(reg_content)

print("")
print("Verifying patches...")

# Quick import test
import subprocess
result = subprocess.run(
    [sys.executable, "-c",
     "from vllm.model_executor.models.interfaces import supports_audio_output; "
     "from vllm.model_executor.models.registry import _ModelInfo; "
     "print('OK')"],
    capture_output=True, text=True
)
if "OK" in result.stdout:
    print("[PASS] imports verified.")
else:
    print("[FAIL] import error:")
    print(result.stderr[-2000:])
    sys.exit(1)
