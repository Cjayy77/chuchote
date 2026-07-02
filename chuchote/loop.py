"""The core round-trip.

    wake word → capture (Silero VAD end-of-turn) → faster-whisper → Ollama
              → Piper → speaker  ↺

Phase 3 adds always-listening turn-taking (wake word + VAD) on top of the
Phase 1/2 round-trip and SQLite memory. Push-to-talk remains as a fallback
mode. Barge-in (interrupting playback) is still to come (Phase 4).
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

from .audio import MicStream, Player, PushToTalkRecorder, make_chime
from .config import Config
from .llm import Reasoner
from .memory import Memory
from .stt import Transcriber
from .text import drain_sentences
from .tts import Speaker
from .vad import FRAME_SAMPLES as VAD_FRAME
from .vad import EndpointDetector
from .wakeword import WakeWordDetector


class Assistant:
    def __init__(self, config: Config):
        self.config = config
        self.recorder = PushToTalkRecorder(config)
        self.transcriber = Transcriber(config)
        self.reasoner = Reasoner(config)
        self.speaker = Speaker(config)
        self.player = Player()
        self.memory = Memory(config)
        self.wake = WakeWordDetector(config)
        self.vad = EndpointDetector(config)

    # --- shared pipeline --------------------------------------------------
    def _speak(self, text: str) -> None:
        samples = self.speaker.synthesize(text)
        if samples.size and self.speaker.sample_rate:
            self.player.play(samples, self.speaker.sample_rate)

    def _chime(self) -> None:
        """Acknowledge the wake word with a short tone (blocks until done)."""
        if not self.config.wake_chime:
            return
        self.player.play(make_chime(self.config.sample_rate), self.config.sample_rate)
        self.player.wait()

    def _reply_to(self, prompt: str) -> None:
        """Reason over history + prompt, speak the streamed reply, persist it."""
        messages = self.memory.recent() + [{"role": "user", "content": prompt}]

        print("chuchote: ", end="", flush=True)
        buffer = ""
        reply = ""
        for piece in self.reasoner.stream_reply(messages):
            print(piece, end="", flush=True)
            reply += piece
            buffer += piece
            sentences, buffer = drain_sentences(buffer)
            for sentence in sentences:
                self._speak(sentence)
        if buffer.strip():
            self._speak(buffer)
        print()

        self.memory.add("user", prompt)
        self.memory.add("assistant", reply)
        self.player.wait()

    def _transcribe(self, audio: np.ndarray) -> str:
        print("[transcribing...]", flush=True)
        return self.transcriber.transcribe(audio)

    # --- push-to-talk mode ------------------------------------------------
    def _ptt_turn(self) -> None:
        audio = self.recorder.record()
        prompt = self._transcribe(audio)
        if not prompt:
            print("(heard nothing — try again)", flush=True)
            return
        print(f"you: {prompt}")
        self._reply_to(prompt)

    # --- wake-word mode ---------------------------------------------------
    def _wait_for_wake(self, mic: MicStream) -> None:
        """Block until the wake word fires (or Ctrl+C between reads)."""
        self.wake.reset()
        print(
            f"\n[listening — say the wake word ('{self.config.wake_model}')...]",
            flush=True,
        )
        while True:
            frame = mic.read(timeout=0.5)
            if frame is None:
                continue
            if self.wake.feed(frame):
                return

    def _capture_utterance(self, mic: MicStream) -> np.ndarray | None:
        """Capture one utterance, ending on trailing silence (Silero VAD)."""
        cfg = self.config
        sr = cfg.sample_rate
        silence_needed = math.ceil(cfg.vad_silence_ms / 1000 * sr / VAD_FRAME)
        max_samples = int(cfg.vad_max_utterance_s * sr)
        start_deadline = time.monotonic() + cfg.vad_start_timeout_s

        self.vad.reset()
        collected: list[np.ndarray] = []
        started = False
        trailing_silence = 0

        while True:
            frame = mic.read(timeout=0.5)
            if frame is None or len(frame) != VAD_FRAME:
                if not started and time.monotonic() > start_deadline:
                    return None
                continue

            is_speech = self.vad.speech_prob(frame) >= cfg.vad_speech_threshold
            if is_speech:
                started = True
                trailing_silence = 0
                collected.append(frame)
            elif started:
                trailing_silence += 1
                collected.append(frame)  # keep a little trailing silence for whisper
                if trailing_silence >= silence_needed:
                    break
            elif time.monotonic() > start_deadline:
                return None

            if started and sum(len(f) for f in collected) >= max_samples:
                break

        if not collected:
            return None
        return np.concatenate(collected).astype(np.float32) / 32768.0

    def _wake_session(self) -> None:
        with MicStream(self.config) as mic:
            while True:
                try:
                    self._wait_for_wake(mic)
                    print("[yes?]", flush=True)
                    self._chime()
                    # Drop the chime + wake-word bleed before capturing speech.
                    mic.flush()
                    audio = self._capture_utterance(mic)
                    if audio is None or audio.size == 0:
                        print("(didn't catch anything — say the wake word again)")
                        continue
                    # Pause the mic during the CPU-heavy transcribe + reply: no
                    # barge-in yet, and it avoids input-overflow warnings.
                    mic.pause()
                    try:
                        prompt = self._transcribe(audio)
                        if not prompt:
                            print("(heard nothing — try again)")
                            continue
                        print(f"you: {prompt}")
                        self._reply_to(prompt)
                    finally:
                        mic.resume()
                        mic.flush()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001 — keep the daemon alive
                    print(f"\n[turn failed: {exc}]", file=sys.stderr)

    # --- entry point ------------------------------------------------------
    def run(self) -> None:
        mode = self.config.mode
        label = "push-to-talk" if mode == "ptt" else f"wake word: {self.config.wake_model}"
        print(f"Chuchote — local voice assistant ({label} + memory)")
        print(f"model: {self.config.ollama_model}  |  whisper: {self.config.whisper_model}")

        # Fail fast + warm up so the first turn isn't slow and missing
        # voice/model/Ollama is caught up front.
        self.reasoner.check_ready()
        print("[warming up models...]", flush=True)
        self.speaker.preload()
        self.transcriber.preload()
        if mode == "wake":
            self.wake.preload()
            self.vad.preload()

        try:
            if mode == "ptt":
                print("Ready. Press Ctrl+C to quit.")
                while True:
                    try:
                        self._ptt_turn()
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        print(f"\n[turn failed: {exc}]", file=sys.stderr)
            else:
                print("Ready. Press Ctrl+C to quit.")
                self._wake_session()
        except KeyboardInterrupt:
            print("\nGoodbye.")
            sys.exit(0)
