"""Startup banner. Pure ASCII so it renders on any console (incl. Windows)."""

from __future__ import annotations

import os
import sys

from . import __version__
from .config import Config

_ART = r"""
       _                _           _
   ___| |__  _   _  ___| |__   ___ | |_ ___
  / __| '_ \| | | |/ __| '_ \ / _ \| __/ _ \
 | (__| | | | |_| | (__| | | | (_) | ||  __/
  \___|_| |_|\__,_|\___|_| |_|\___/ \__\___|
"""

_TAGLINE = "local-first, hands-free voice assistant for Ollama"
_NAME = "CJ_"  # italicised in the credit when the terminal supports it
_CREDIT_TEMPLATE = "by {name} (Cjayy77) - github.com/Cjayy77/chuchote"

_ITALIC_ON = "\033[3m"
_ITALIC_OFF = "\033[23m"

_vt_ready: bool | None = None


def _enable_ansi() -> bool:
    """Ensure the console can process ANSI escapes. On Windows this turns on
    virtual-terminal processing so codes render (or are cleanly ignored)
    instead of printing as literal garbage. Cached after the first call."""
    global _vt_ready
    if _vt_ready is not None:
        return _vt_ready
    if os.name != "nt":
        _vt_ready = True
        return _vt_ready
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
            _vt_ready = True
        else:
            _vt_ready = False
    except Exception:
        _vt_ready = False
    return _vt_ready


def _supports_style() -> bool:
    """ANSI styling only when writing to an interactive terminal (honour
    NO_COLOR). Avoids escape-code garbage in piped/redirected output."""
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR") and _enable_ansi()


def _italic(text: str) -> str:
    return f"{_ITALIC_ON}{text}{_ITALIC_OFF}" if _supports_style() else text


def render(config: Config) -> str:
    """The banner string, or empty if disabled in config."""
    if not config.banner:
        return ""
    credit = _CREDIT_TEMPLATE.format(name=_italic(_NAME))
    return f"{_ART}\n  {_TAGLINE}\n  v{__version__}  -  {credit}\n"
