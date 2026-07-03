"""Configuration for Chuchote.

Defaults live in the `Config` dataclass. `Config.load()` overlays a TOML config
file (`chuchote init` writes a template), and the CLI overlays flags on top of
that — so precedence is defaults < config file < command-line flags.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields


def user_data_dir() -> str:
    """Platform-appropriate per-user data directory (CWD-independent)."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "chuchote")


def user_config_dir() -> str:
    """Platform-appropriate per-user config directory."""
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "chuchote")


def default_config_path() -> str:
    """Per-user TOML config path."""
    return os.path.join(user_config_dir(), "config.toml")


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

    # --- Barge-in (interrupt playback) -----------------------------------
    # "wake": say the wake word again to cut off a reply (echo-robust, works
    #         with open speakers — the default).
    # "vad":  any speech interrupts (only reliable with headphones; open
    #         speakers cause the assistant to interrupt its own audio).
    # "off":  never interrupt; let each reply finish.
    barge_in_mode: str = "wake"

    # --- VAD (Silero, end-of-turn) ---------------------------------------
    vad_speech_threshold: float = 0.5  # per-frame speech probability cutoff
    vad_silence_ms: int = 800  # trailing silence that ends a turn
    vad_start_timeout_s: float = 8.0  # give up if no speech after the wake word
    vad_max_utterance_s: float = 30.0  # hard cap on a single utterance

    # --- Language ---------------------------------------------------------
    # Recognition language, e.g. "en", "fr", "de", "es", "it", "zh". "auto"
    # lets whisper detect it. English-only whisper models (*.en) always use
    # English regardless. For any non-English language you also need a
    # MULTILINGUAL whisper model (e.g. "small", not "small.en") and a matching
    # Piper voice — see the "Languages" section of the README.
    language: str = "auto"

    # --- STT (faster-whisper) ---------------------------------------------
    # "base.en" loads reliably even on 8 GB machines with other apps open;
    # "small.en" is noticeably more accurate but needs ~1 GB free RAM to
    # load. Use the multilingual variants ("base", "small", ...) for
    # non-English languages.
    whisper_model: str = "base.en"
    whisper_device: str = "auto"  # "cpu", "cuda", or "auto"
    # int8 quantization: ~4x less memory and faster on CPU, near-identical
    # accuracy. "default" (the model's saved float16) gets converted to
    # float32 on CPUs, which can exhaust RAM on smaller machines.
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 5  # >1 = beam search (more accurate); 1 = greedy (faster)

    # --- Reasoning (Ollama) ----------------------------------------------
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    system_prompt: str = (
        "You are Chuchote, a concise, friendly voice assistant. "
        "You are being spoken to out loud and your replies are read aloud, "
        "so keep answers short and conversational. Avoid markdown, lists, "
        "code blocks, and emoji — just speak in plain sentences. "
        "Always reply in the same language the person is speaking to you."
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

    # --- Interface --------------------------------------------------------
    banner: bool = True  # print the startup banner

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        """Build a Config from defaults overlaid with a TOML file if present.

        Unknown keys are warned about and ignored. Keys map 1:1 to the field
        names below (top-level, or under an optional [chuchote] table).
        """
        cfg = cls()
        path = path or default_config_path()
        if not os.path.exists(path):
            return cfg

        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10: stdlib tomllib is 3.11+
            import tomli as tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)
        if "chuchote" in data and isinstance(data["chuchote"], dict):
            data = data["chuchote"]

        known = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key in known:
                setattr(cfg, key, value)
            else:
                print(f"(config: ignoring unknown key '{key}')", file=sys.stderr)
        return cfg


# Template written by `chuchote init`. Every setting is shown commented at its
# default; uncomment and edit what you want to change.
CONFIG_TEMPLATE = """\
# Chuchote configuration. Uncomment a line to override the default.
# Command-line flags still take precedence over this file.

# --- Reasoning (Ollama) ---
# ollama_model = "llama3.2"
# ollama_host = "http://localhost:11434"

# --- Turn-taking ---
# mode = "wake"              # "wake" (always-on) or "ptt" (push-to-talk)

# --- Wake word ---
# wake_model = "hey_jarvis"  # also: alexa, hey_mycroft, hey_rhasspy
# wake_threshold = 0.5       # 0..1; raise to reduce false triggers
# wake_chime = true          # tone acknowledging the wake word

# --- Barge-in (interrupt a reply) ---
# barge_in_mode = "wake"     # "wake", "vad" (headphones), or "off"

# --- Voice activity detection (end-of-turn) ---
# vad_speech_threshold = 0.5
# vad_silence_ms = 800       # trailing silence that ends your turn
# vad_start_timeout_s = 8.0
# vad_max_utterance_s = 30.0

# --- Push-to-talk ---
# ptt_key = "space"
# ptt_tail_seconds = 0.4

# --- Language ---
# language = "auto"          # "en", "fr", "de", "es", "zh", ... or "auto".
                             # Non-English also needs a multilingual whisper
                             # model + a matching Piper voice (see README).

# --- Speech-to-text (faster-whisper) ---
# whisper_model = "base.en"  # "small.en" = more accurate, needs ~1 GB free RAM;
                             # use "base"/"small" (no .en) for other languages
# whisper_device = "auto"    # "cpu", "cuda", or "auto"
# whisper_compute_type = "int8"  # low-memory + fast on CPU; "default" = model's own
# whisper_beam_size = 5      # >1 = more accurate; 1 = greedy/faster

# --- Text-to-speech (Piper) ---
# piper_voice = ""           # path to a .onnx voice; blank = auto-discover

# --- Memory ---
# history_messages = 12      # prior messages fed back into context

# --- Interface ---
# banner = true
"""


def write_default_config(path: str | None = None) -> str:
    """Write the commented config template. Returns the path written."""
    path = path or default_config_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(CONFIG_TEMPLATE)
    return path
