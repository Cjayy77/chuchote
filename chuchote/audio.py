"""Microphone capture (push-to-talk) and speaker playback.

Phase 1 uses push-to-talk: hold a key, speak, release. No wake word or VAD
yet — that's Phase 3. Capture and playback both run through sounddevice so we
share one audio backend.
"""

from __future__ import annotations

import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd

from .config import Config


class PushToTalkRecorder:
    """Records mono float32 audio while a key is held down.

    Uses pynput so it works whether or not the terminal has focus, which
    matters for a hands-free-ish daemon. Returns a single numpy array of
    samples captured between key-press and key-release.
    """

    def __init__(self, config: Config):
        self.config = config
        self._recording = threading.Event()
        self._released = threading.Event()

    def _key_matches(self, key) -> bool:
        from pynput import keyboard

        want = self.config.ptt_key.lower()
        # Named special keys (space, ctrl, etc.)
        if isinstance(key, keyboard.Key):
            return key.name == want
        # Character keys
        if isinstance(key, keyboard.KeyCode) and key.char is not None:
            return key.char.lower() == want
        return False

    def record(self) -> np.ndarray:
        """Block until the PTT key is pressed, record until released.

        The mic stream is opened *before* the keypress so it's already warm
        (no startup clipping), and capture continues for a short tail after
        release so the OS audio buffer flushes and the end isn't cut off.
        """
        from pynput import keyboard

        self._recording.clear()
        self._released.clear()
        capturing = threading.Event()
        frames: list[np.ndarray] = []

        def on_press(key):
            if self._key_matches(key) and not self._recording.is_set():
                self._recording.set()

        def on_release(key):
            if self._key_matches(key) and self._recording.is_set():
                self._released.set()
                return False  # stop listener

        def callback(indata, n_frames, time_info, status):  # noqa: ANN001
            if status:
                print(f"(audio: {status})", file=sys.stderr)
            # Only keep audio between key-press and (release + tail).
            if capturing.is_set():
                frames.append(indata.copy())

        print(
            f"\n[hold '{self.config.ptt_key}' to talk, release to send] ",
            end="",
            flush=True,
        )

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        # Open the stream up front so it's running by the time the key goes
        # down; frames are discarded until `capturing` is set on key-press.
        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="float32",
            callback=callback,
        ):
            self._recording.wait()
            capturing.set()
            print("* listening...", flush=True)

            self._released.wait()
            # Keep the callback appending for a moment so trailing audio still
            # in the OS buffer is captured before we stop.
            time.sleep(max(0.0, self.config.ptt_tail_seconds))
            capturing.clear()

        listener.join()

        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames, axis=0).reshape(-1)


class Player:
    """Serialised speaker playback.

    A single worker thread drains a queue of (samples, sample_rate) chunks so
    sentence-chunked TTS can be enqueued as it's synthesised and played back
    gaplessly in order.
    """

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                continue
            samples, sample_rate = item
            try:
                sd.play(samples, samplerate=sample_rate)
                sd.wait()
            except Exception as exc:  # keep the loop alive on playback errors
                print(f"(playback error: {exc})", file=sys.stderr)
            finally:
                self._q.task_done()

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        self._q.put((samples, sample_rate))

    def wait(self) -> None:
        """Block until everything queued so far has finished playing."""
        self._q.join()


# Wake-word and VAD both operate on 16 kHz int16 audio; Silero needs exactly
# 512-sample frames, so that's the stream block size and the unit both consume.
FRAME_SAMPLES = 512


class MicStream:
    """Continuous 16 kHz int16 microphone stream for always-listening mode.

    Yields fixed-size int16 frames via read(). Used as a context manager so the
    stream stays open across wake-word detection and utterance capture.
    """

    def __init__(self, config: Config, blocksize: int = FRAME_SAMPLES):
        self.config = config
        self.blocksize = blocksize
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None

    def __enter__(self) -> "MicStream":
        def callback(indata, n_frames, time_info, status):  # noqa: ANN001
            if status:
                print(f"(audio: {status})", file=sys.stderr)
            self._q.put(indata.copy().reshape(-1))

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="int16",
            blocksize=self.blocksize,
            callback=callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self, timeout: float | None = 0.5) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush(self) -> None:
        """Drop any buffered frames (e.g. the assistant's own TTS output)."""
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def pause(self) -> None:
        """Stop capturing (e.g. during transcription/reply) to avoid input
        overflow and free the CPU. No barge-in yet, so we don't need the mic
        while responding."""
        if self._stream is not None and self._stream.active:
            self._stream.stop()

    def resume(self) -> None:
        if self._stream is not None and not self._stream.active:
            self._stream.start()


def make_chime(sample_rate: int) -> np.ndarray:
    """A short two-note rising chime to acknowledge the wake word."""

    def tone(freq: float, seconds: float) -> np.ndarray:
        t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
        wave = 0.25 * np.sin(2 * np.pi * freq * t)
        # 10 ms fade in/out so it doesn't click.
        fade = max(1, int(0.01 * sample_rate))
        env = np.ones_like(wave)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        return wave * env

    return np.concatenate([tone(660, 0.09), tone(990, 0.11)]).astype(np.float32)
