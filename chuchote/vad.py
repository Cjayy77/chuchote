"""End-of-turn detection via Silero VAD (Phase 3).

Uses pysilero-vad (onnxruntime, bundles the model, no torch). Without this we'd
either cut people off or wait forever — it's what makes always-listening feel
like real turn-taking rather than a walkie-talkie.
"""

from __future__ import annotations

import numpy as np

from .config import Config

# Silero (16 kHz) consumes fixed 512-sample (32 ms) frames.
FRAME_SAMPLES = 512


class EndpointDetector:
    def __init__(self, config: Config):
        self.config = config
        self._vad = None

    def _ensure(self):
        if self._vad is None:
            from pysilero_vad import SileroVoiceActivityDetector

            self._vad = SileroVoiceActivityDetector()
        return self._vad

    def preload(self) -> None:
        self._ensure()

    def reset(self) -> None:
        if self._vad is not None:
            self._vad.reset()

    def speech_prob(self, frame_i16: np.ndarray) -> float:
        """Speech probability (0..1) for exactly FRAME_SAMPLES int16 samples."""
        vad = self._ensure()
        if len(frame_i16) != FRAME_SAMPLES:
            raise ValueError(f"VAD expects {FRAME_SAMPLES} samples, got {len(frame_i16)}")
        return float(vad(frame_i16.astype(np.int16).tobytes()))
