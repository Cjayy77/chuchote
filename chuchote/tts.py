"""Text-to-speech via Piper.

Piper voices ship as an `.onnx` model plus a matching `.onnx.json` config.
Grab one from https://github.com/rhasspy/piper/blob/master/VOICES.md and point
`config.piper_voice` at the `.onnx` file (or drop it in ./voices).

Downloading a voice is a one-time local setup step, not a runtime network call,
so it doesn't touch the local-first core loop.
"""

from __future__ import annotations

import glob
import os

import numpy as np

from .config import Config


class Speaker:
    def __init__(self, config: Config):
        self.config = config
        self._voice = None
        self.sample_rate: int | None = None

    def _find_voice_path(self) -> str:
        if self.config.piper_voice:
            path = self.config.piper_voice
        elif os.environ.get("CHUCHOTE_VOICE"):
            path = os.environ["CHUCHOTE_VOICE"]
        else:
            path = None
            for d in self.config.voice_search_dirs:
                matches = sorted(glob.glob(os.path.join(d, "*.onnx")))
                if matches:
                    path = matches[0]
                    break

        if not path or not os.path.exists(path):
            searched = ", ".join(
                os.path.abspath(d) for d in self.config.voice_search_dirs
            )
            raise RuntimeError(
                "No Piper voice found. Download a voice (.onnx + .onnx.json) from "
                "https://github.com/rhasspy/piper/blob/master/VOICES.md, then either "
                "pass --voice PATH, set CHUCHOTE_VOICE, or drop it in one of: "
                f"{searched}"
            )
        return path

    def preload(self) -> None:
        """Load the voice now (fail fast + avoid first-turn latency)."""
        self._ensure_voice()

    def _ensure_voice(self):
        if self._voice is None:
            from piper import PiperVoice

            path = self._find_voice_path()
            print(f"[loading piper voice '{os.path.basename(path)}'...]", flush=True)
            self._voice = PiperVoice.load(path)
            self.sample_rate = self._voice.config.sample_rate
        return self._voice

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesise `text` into an int16 mono numpy array at self.sample_rate."""
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.int16)

        voice = self._ensure_voice()
        chunks: list[np.ndarray] = []

        # piper-tts's API has shifted across versions; support the common shapes.
        if hasattr(voice, "synthesize"):
            for chunk in voice.synthesize(text):
                if hasattr(chunk, "audio_int16_array"):
                    chunks.append(np.asarray(chunk.audio_int16_array, dtype=np.int16))
                elif hasattr(chunk, "audio_int16_bytes"):
                    chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
                else:  # raw bytes
                    chunks.append(np.frombuffer(bytes(chunk), dtype=np.int16))
        else:  # older API
            for raw in voice.synthesize_stream_raw(text):
                chunks.append(np.frombuffer(raw, dtype=np.int16))

        if not chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(chunks)
