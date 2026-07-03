"""The core round-trip.

    wake word → capture (Silero VAD end-of-turn) → faster-whisper → Ollama
              → Piper → speaker  ↺   (barge-in to interrupt)

Phase 4 adds barge-in — cutting off a reply when the user speaks again — plus
overlapped synthesis (sentences are synthesised on a worker thread while the
model keeps generating). Built on the Phase 1-3 round-trip, memory, and
always-listening wake word + VAD. Push-to-talk remains as a fallback mode.
"""

from __future__ import annotations

import math
import sys
import threading
import time

import numpy as np

from .audio import MicStream, PushToTalkRecorder, SpeechEngine, make_chime
from .banner import render as render_banner
from .config import Config
from .llm import Reasoner
from .memory import Memory
from .stt import Transcriber
from .text import drain_sentences, explain_failure
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
        self.speech = SpeechEngine(self.speaker)
        self.memory = Memory(config)
        self.wake = WakeWordDetector(config)
        self.vad = EndpointDetector(config)

    # --- shared pipeline --------------------------------------------------
    def _chime(self) -> None:
        """Acknowledge the wake word with a short tone (blocks until done)."""
        if not self.config.wake_chime:
            return
        self.speech.play_pcm(make_chime(self.config.sample_rate), self.config.sample_rate)
        self.speech.wait()

    def _reply_to(self, prompt: str, watch=None) -> bool:
        """Reason over history + prompt and speak the streamed reply.

        `watch(interrupted, stop)` is an optional callable run on a monitor
        thread that sets `interrupted` (and cuts off speech) on barge-in.
        Returns True if the reply was cut off.
        """
        messages = self.memory.recent() + [{"role": "user", "content": prompt}]

        interrupted = threading.Event()
        stop_monitor = threading.Event()
        monitor: threading.Thread | None = None
        if watch is not None:
            self.speech.clear_interrupt()
            monitor = threading.Thread(
                target=watch,
                args=(interrupted, stop_monitor),
                daemon=True,
            )
            monitor.start()

        print("chuchote: ", end="", flush=True)
        buffer = ""
        reply = ""
        try:
            for piece in self.reasoner.stream_reply(messages):
                if interrupted.is_set():
                    break
                print(piece, end="", flush=True)
                reply += piece
                buffer += piece
                sentences, buffer = drain_sentences(buffer)
                for sentence in sentences:
                    if interrupted.is_set():
                        break
                    self.speech.say(sentence)
            if not interrupted.is_set() and buffer.strip():
                self.speech.say(buffer)
            # Blocks until playback finishes, or returns early if barge-in
            # drained the queue.
            self.speech.wait()
        finally:
            stop_monitor.set()
            if monitor is not None:
                monitor.join(timeout=1.0)
        print()

        # Persist what was actually produced (even a cut-off reply is context).
        self.memory.add("user", prompt)
        if reply.strip():
            self.memory.add("assistant", reply)

        if interrupted.is_set():
            print("[interrupted]", flush=True)
            self.speech.clear_interrupt()
        return interrupted.is_set()

    def _barge_in_monitor(
        self, mic: MicStream, interrupted: threading.Event, stop: threading.Event
    ) -> None:
        """Watch the mic during a reply. Fires `interrupted` + cuts off speech.

        Always drains the mic (so the input buffer never overflows while we're
        talking); in "off" mode it only drains and never interrupts.
        """
        mode = self.config.barge_in_mode
        if mode == "wake":
            self.wake.reset()
        elif mode == "vad":
            self.vad.reset()

        while not stop.is_set():
            frame = mic.read(timeout=0.2)
            if frame is None:
                continue
            fired = False
            if mode == "wake":
                fired = self.wake.feed(frame)
            elif mode == "vad" and len(frame) == VAD_FRAME:
                fired = self.vad.speech_prob(frame) >= self.config.vad_speech_threshold
            if fired:
                interrupted.set()
                self.speech.interrupt()
                return

    def _ptt_barge_monitor(
        self, interrupted: threading.Event, stop: threading.Event
    ) -> None:
        """Barge-in for push-to-talk: pressing the PTT key cuts off the reply."""
        from pynput import keyboard

        def on_press(key):
            if self.recorder._key_matches(key):
                interrupted.set()
                self.speech.interrupt()
                return False  # stop listener

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        while not stop.is_set() and not interrupted.is_set():
            time.sleep(0.05)
        listener.stop()

    def _transcribe(self, audio: np.ndarray) -> str:
        print("[transcribing...]", flush=True)
        return self.transcriber.transcribe(audio)

    # --- push-to-talk mode ------------------------------------------------
    def _ptt_turn(self) -> None:
        audio = self.recorder.record()
        prompt = self._transcribe(audio)
        if not prompt:
            print("(heard nothing - try again)", flush=True)
            return
        print(f"you: {prompt}")
        # Barge-in via the PTT key (unless disabled).
        watch = None if self.config.barge_in_mode == "off" else self._ptt_barge_monitor
        self._reply_to(prompt, watch)

    # --- wake-word mode ---------------------------------------------------
    def _wait_for_wake(self, mic: MicStream) -> None:
        """Block until the wake word fires (or Ctrl+C between reads)."""
        self.wake.reset()
        print(
            f"\n[listening - say the wake word ('{self.config.wake_model}')...]",
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

    def _converse(self, mic: MicStream) -> None:
        """Handle turns after a wake word. A barge-in rolls straight into the
        next turn without needing the wake word again."""
        while True:
            self._chime()
            mic.flush()  # drop the chime + any wake-word bleed
            audio = self._capture_utterance(mic)
            if audio is None or audio.size == 0:
                print("(didn't catch anything - say the wake word again)")
                return
            prompt = self._transcribe(audio)
            if not prompt:
                print("(heard nothing - try again)")
                return
            print(f"you: {prompt}")
            # Mic monitor always runs (it also drains the mic to prevent
            # overflow); in "off" mode it drains without interrupting.
            interrupted = self._reply_to(
                prompt, lambda i, s: self._barge_in_monitor(mic, i, s)
            )
            mic.flush()
            if not interrupted:
                return  # reply finished cleanly → back to waiting for the wake word

    def _wake_session(self) -> None:
        with MicStream(self.config) as mic:
            while True:
                try:
                    self._wait_for_wake(mic)
                    self._converse(mic)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001 — keep the daemon alive
                    print(
                        f"\n[turn failed: {explain_failure(exc, self.config.ollama_model)}]",
                        file=sys.stderr,
                    )

    # --- entry point ------------------------------------------------------
    def run(self) -> None:
        mode = self.config.mode
        banner = render_banner(self.config)
        if banner:
            print(banner)
        label = "push-to-talk" if mode == "ptt" else f"wake word: {self.config.wake_model}"
        print(f"mode: {label}  |  model: {self.config.ollama_model}"
              f"  |  whisper: {self.config.whisper_model}")

        # Fail fast + warm up so the first turn isn't slow and missing
        # voice/model/Ollama is caught up front.
        self.reasoner.check_ready()
        print("[warming up models...]", flush=True)
        self.speaker.preload()
        self.transcriber.preload()
        if mode == "wake":
            self.wake.preload()
            self.vad.preload()

        print("Ready. Press Ctrl+C to quit.")
        try:
            if mode == "ptt":
                while True:
                    try:
                        self._ptt_turn()
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"\n[turn failed: {explain_failure(exc, self.config.ollama_model)}]",
                            file=sys.stderr,
                        )
            else:
                self._wake_session()
        except KeyboardInterrupt:
            print("\nGoodbye.")
            sys.exit(0)
