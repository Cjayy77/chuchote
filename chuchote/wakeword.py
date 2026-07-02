"""Always-on wake word detection via openWakeWord (Phase 3).

openWakeWord is fully local and needs no API key. We run it on onnxruntime
(already present for Piper), so no extra heavy runtime. Pretrained models are
downloaded once on first use.
"""

from __future__ import annotations

import numpy as np

from .config import Config

# openWakeWord's feature models expect 16 kHz int16 audio; 1280 samples (80 ms)
# is the recommended prediction frame size.
FRAME_SAMPLES = 1280


class WakeWordDetector:
    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._buffer = np.zeros(0, dtype=np.int16)

    def _ensure_model(self):
        if self._model is None:
            import openwakeword
            from openwakeword.model import Model

            print(f"[loading wake word '{self.config.wake_model}'...]", flush=True)
            # Downloads the melspectrogram/embedding feature models plus the
            # pretrained wake-word models (once; cached afterwards).
            openwakeword.utils.download_models()
            self._model = Model(
                wakeword_models=[self.config.wake_model],
                inference_framework="onnx",
            )
        return self._model

    def preload(self) -> None:
        self._ensure_model()

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.int16)
        if self._model is not None:
            self._model.reset()

    def feed(self, frame_i16: np.ndarray) -> bool:
        """Feed an int16 mono frame; return True if the wake word fired.

        Accepts arbitrary frame sizes and re-chunks to openWakeWord's preferred
        prediction size internally.
        """
        model = self._ensure_model()
        self._buffer = np.concatenate([self._buffer, frame_i16])
        fired = False
        while len(self._buffer) >= FRAME_SAMPLES:
            chunk = self._buffer[:FRAME_SAMPLES]
            self._buffer = self._buffer[FRAME_SAMPLES:]
            scores = model.predict(chunk)
            if any(score >= self.config.wake_threshold for score in scores.values()):
                fired = True
        return fired
