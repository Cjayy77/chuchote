"""Configuration for the core round-trip.

Phase 1 keeps this to an in-code dataclass with sensible defaults. A real
config file (`~/.config/chuchote/config.toml` or similar) is Phase 5 work —
don't add file loading here until we get there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def user_data_dir() -> str:
    """Platform-appropriate per-user data directory (CWD-independent)."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "chuchote")


def user_voice_dir() -> str:
    """Per-user voices directory."""
    return os.path.join(user_data_dir(), "voices")


def default_db_path() -> str:
    """Per-user SQLite database path for conversation memory."""
    return os.path.join(user_data_dir(), "memory.db")


@dataclass
class Config:
    # --- Audio -------------------------------------------------------------
    sample_rate: int = 16_000  # whisper wants 16 kHz mono
    channels: int = 1
    ptt_key: str = "space"  # hold to talk (see chuchote.audio for key names)
    # Keep capturing briefly after key-release so the OS audio buffer flushes
    # and the tail of your sentence isn't clipped.
    ptt_tail_seconds: float = 0.4

    # --- Turn-taking mode -------------------------------------------------
    # "wake": always-on, triggered by a wake word + Silero VAD end-of-turn.
    # "ptt":  push-to-talk fallback (hold ptt_key).
    mode: str = "wake"

    # --- Wake word (openWakeWord) ----------------------------------------
    # Pretrained model name shipped with openWakeWord, e.g. "hey_jarvis",
    # "alexa", "hey_mycroft". Downloaded once via openwakeword.
    wake_model: str = "hey_jarvis"
    wake_threshold: float = 0.5  # detection score 0..1; raise to reduce false triggers
    wake_chime: bool = True  # play a short tone to acknowledge the wake word

    # --- VAD (Silero, end-of-turn) ---------------------------------------
    vad_speech_threshold: float = 0.5  # per-frame speech probability cutoff
    vad_silence_ms: int = 800  # trailing silence that ends a turn
    vad_start_timeout_s: float = 8.0  # give up if no speech after the wake word
    vad_max_utterance_s: float = 30.0  # hard cap on a single utterance

    # --- STT (faster-whisper) ---------------------------------------------
    # "base.en" is a good latency/quality tradeoff for a first loop on CPU.
    whisper_model: str = "base.en"
    whisper_device: str = "auto"  # "cpu", "cuda", or "auto"
    whisper_compute_type: str = "default"

    # --- Reasoning (Ollama) ----------------------------------------------
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    system_prompt: str = (
        "You are Chuchote, a concise, friendly voice assistant. "
        "You are being spoken to out loud and your replies are read aloud, "
        "so keep answers short and conversational. Avoid markdown, lists, "
        "code blocks, and emoji — just speak in plain sentences."
    )

    # --- TTS (Piper) ------------------------------------------------------
    # Path to a Piper .onnx voice model. Piper voices ship as a .onnx +
    # .onnx.json pair; download one from
    # https://github.com/rhasspy/piper/blob/master/VOICES.md
    # If left None, chuchote looks in ./voices and CHUCHOTE_VOICE.
    piper_voice: str | None = None

    # Directories searched for a Piper voice when piper_voice is unset. The
    # per-user dir works no matter where `chuchote` is launched from; ./voices
    # and . are dev conveniences when running inside the repo.
    voice_search_dirs: list[str] = field(
        default_factory=lambda: [user_voice_dir(), "voices", "."]
    )

    # --- Memory (SQLite) --------------------------------------------------
    # Conversation history is persisted here and recent turns are fed back
    # into Ollama's context each turn. Persists across restarts.
    db_path: str = field(default_factory=default_db_path)
    # How many prior messages (user + assistant lines) to inject as context.
    # ~12 ≈ the last 6 exchanges — enough for continuity without bloating the
    # prompt or slowing generation.
    history_messages: int = 12
