"""Speech-to-text via faster-whisper.

The model is loaded lazily on first use so `chuchote start` prints quickly and
we only pay the load cost once, before the first turn.
"""

from __future__ import annotations

import numpy as np

from .config import Config


class Transcriber:
    def __init__(self, config: Config):
        self.config = config
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            print(f"[loading whisper '{self.config.whisper_model}'...]", flush=True)
            self._model = WhisperModel(
                self.config.whisper_model,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute_type,
            )
        return self._model

    def preload(self) -> None:
        """Load the model now (fail fast + avoid first-turn latency)."""
        self._ensure_model()

    def _language(self) -> str | None:
        """Language code passed to whisper, or None to auto-detect.

        English-only models (*.en) can only do English. Otherwise honour the
        configured language, treating "auto"/empty as detect.
        """
        if self.config.whisper_model.endswith(".en"):
            return "en"
        lang = (self.config.language or "auto").lower()
        return None if lang == "auto" else lang

    def transcribe(self, samples: np.ndarray) -> str:
        """Transcribe mono float32 samples (already at config.sample_rate)."""
        if samples.size == 0:
            return ""

        model = self._ensure_model()
        # faster-whisper accepts a float32 numpy array at 16 kHz directly.
        segments, _info = model.transcribe(
            samples.astype(np.float32),
            language=self._language(),
            # Beam search (default 5) is noticeably more accurate than greedy;
            # transcription is a small slice of the turn, so it's worth it.
            beam_size=self.config.whisper_beam_size,
            vad_filter=True,  # trims leading/trailing silence
            # Each capture is an independent utterance, so don't let a previous
            # turn's text bias decoding (avoids repetition/hallucination loops).
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
