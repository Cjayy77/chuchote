"""Startup banner. Pure ASCII so it renders on any console (incl. Windows)."""

from __future__ import annotations

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
_CREDIT = "by Cedric (Cjayy77) - github.com/Cjayy77/chuchote"


def render(config: Config) -> str:
    """The banner string, or empty if disabled in config."""
    if not config.banner:
        return ""
    return f"{_ART}\n  {_TAGLINE}\n  v{__version__}  -  {_CREDIT}\n"
