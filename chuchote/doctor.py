"""`chuchote doctor` — preflight checks so problems surface before a session.

Verifies the things that otherwise fail mid-run: Ollama connectivity + the
configured model, a usable Piper voice, working audio devices, and (in wake
mode) the wake-word/VAD dependencies.
"""

from __future__ import annotations

import os

from .config import Config, default_config_path

OK, WARN, FAIL = "ok", "warn", "fail"
_TAG = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[fail]"}


def _check_config() -> tuple[str, str]:
    path = default_config_path()
    if os.path.exists(path):
        return OK, f"Config file: {path}"
    return OK, "Config file: none (using defaults; 'chuchote init' to create one)"


def _check_ollama(config: Config) -> tuple[str, str]:
    try:
        from ollama import Client

        models = {m.model for m in Client(host=config.ollama_host).list().models}
    except Exception as exc:  # noqa: BLE001
        return FAIL, (
            f"Ollama unreachable at {config.ollama_host} "
            f"({exc.__class__.__name__}). Start it with 'ollama serve'."
        )
    want = config.ollama_model
    if want in models or f"{want}:latest" in models:
        return OK, f"Ollama reachable; model '{want}' available"
    return FAIL, f"Ollama reachable, but model '{want}' not pulled. Run 'ollama pull {want}'."


def _check_voice(config: Config) -> tuple[str, str]:
    from .tts import Speaker

    try:
        path = Speaker(config)._find_voice_path()
        return OK, f"Piper voice: {os.path.basename(path)}"
    except Exception as exc:  # noqa: BLE001
        return FAIL, str(exc)


def _check_audio() -> list[tuple[str, str]]:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
    except Exception as exc:  # noqa: BLE001
        return [(FAIL, f"Audio system error: {exc}")]

    has_in = any(d["max_input_channels"] > 0 for d in devices)
    has_out = any(d["max_output_channels"] > 0 for d in devices)
    return [
        (OK, "Microphone detected") if has_in else (FAIL, "No microphone/input device found"),
        (OK, "Speaker detected") if has_out else (FAIL, "No speaker/output device found"),
    ]


def _free_ram_gb() -> float | None:
    """Available physical memory in GB, or None if unknowable (best-effort)."""
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(stat)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return None
            return stat.ullAvailPhys / 1024**3
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 1024**3
    except Exception:  # noqa: BLE001
        return None


def _check_memory(config: Config) -> tuple[str, str] | None:
    """Warn when free RAM is too tight to load the models.

    Loading whisper needs headroom beyond the model's final size (weights are
    converted in memory), and Ollama's LLM lives in RAM too. Low free memory
    is exactly what turns into an opaque `mkl_malloc` crash at startup.
    """
    free = _free_ram_gb()
    if free is None:
        return None
    if free < 1.5:
        return WARN, (
            f"Very low free RAM ({free:.1f} GB). Even speech models may fail to "
            "load - close some apps, or use a smaller whisper_model (tiny.en)."
        )
    if free < 3.0:
        return WARN, (
            f"Free RAM: {free:.1f} GB - enough for chuchote itself, but Ollama "
            f"may fail to load '{config.ollama_model}' (a ~3B model needs "
            "~2.5 GB). Close some apps or use a smaller model (e.g. llama3.2:1b)."
        )
    return OK, f"Free RAM: {free:.1f} GB"


def _check_language(config: Config) -> tuple[str, str] | None:
    lang = (config.language or "auto").lower()
    if lang in ("auto", "en"):
        return None
    if config.whisper_model.endswith(".en"):
        return WARN, (
            f"Language '{lang}' set, but whisper model '{config.whisper_model}' "
            "is English-only. Use a multilingual model like 'small'."
        )
    return OK, f"Language: {lang} (make sure your Piper voice matches)"


def _check_wake_deps(config: Config) -> tuple[str, str] | None:
    if config.mode != "wake":
        return None
    try:
        import openwakeword  # noqa: F401
        import pysilero_vad  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return FAIL, f"Wake word/VAD deps missing: {exc}"
    return OK, "Wake word + VAD deps present (models download on first run)"


def run(config: Config) -> bool:
    """Print all checks. Return True if nothing failed."""
    print("Chuchote doctor\n")

    results: list[tuple[str, str]] = [_check_config(), _check_ollama(config)]
    results.append(_check_voice(config))
    results.extend(_check_audio())
    for optional in (
        _check_memory(config),
        _check_language(config),
        _check_wake_deps(config),
    ):
        if optional is not None:
            results.append(optional)

    for status, message in results:
        print(f"  {_TAG[status]} {message}")

    fails = sum(1 for status, _ in results if status == FAIL)
    print()
    if fails:
        print(f"{fails} problem(s) found - fix the [fail] items above before 'chuchote start'.")
        return False
    print("All checks passed. Run 'chuchote start' to begin.")
    return True
