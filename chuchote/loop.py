"""The core round-trip.

    push-to-talk → faster-whisper → Ollama → Piper → speaker  ↺

Phase 2 adds SQLite-backed memory: recent turns are injected into Ollama's
context and every exchange is persisted. Wake word, VAD, and barge-in are still
to come (see CLAUDE.md build order).
"""

from __future__ import annotations

import sys

from .audio import Player, PushToTalkRecorder
from .config import Config
from .llm import Reasoner
from .memory import Memory
from .stt import Transcriber
from .text import drain_sentences
from .tts import Speaker


class Assistant:
    def __init__(self, config: Config):
        self.config = config
        self.recorder = PushToTalkRecorder(config)
        self.transcriber = Transcriber(config)
        self.reasoner = Reasoner(config)
        self.speaker = Speaker(config)
        self.player = Player()
        self.memory = Memory(config)

    def _speak(self, text: str) -> None:
        samples = self.speaker.synthesize(text)
        if samples.size and self.speaker.sample_rate:
            self.player.play(samples, self.speaker.sample_rate)

    def _handle_turn(self) -> None:
        audio = self.recorder.record()
        print("[transcribing...]", flush=True)
        prompt = self.transcriber.transcribe(audio)

        if not prompt:
            print("(heard nothing — try again)", flush=True)
            return

        print(f"you: {prompt}")

        # Inject recent history, then this turn's prompt.
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
        # Speak whatever's left that didn't end in punctuation.
        if buffer.strip():
            self._speak(buffer)
        print()

        # Persist the exchange only after a successful generation.
        self.memory.add("user", prompt)
        self.memory.add("assistant", reply)

        self.player.wait()

    def run(self) -> None:
        print("Chuchote — local voice assistant (push-to-talk + memory)")
        print(f"model: {self.config.ollama_model}  |  whisper: {self.config.whisper_model}")

        # Fail fast with clear messages, and warm up the heavy models now so the
        # first turn isn't slow (and a missing voice/model is caught up front,
        # not after we've already generated a reply).
        self.reasoner.check_ready()
        print("[warming up models...]", flush=True)
        self.speaker.preload()
        self.transcriber.preload()

        print("Ready. Press Ctrl+C to quit.")
        try:
            while True:
                try:
                    self._handle_turn()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001 — keep the daemon alive
                    print(f"\n[turn failed: {exc}]", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nGoodbye.")
            sys.exit(0)
