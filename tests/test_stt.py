"""Tests for whisper language selection (no model load required)."""

from chuchote.config import Config
from chuchote.stt import Transcriber


def _lang(**overrides):
    return Transcriber(Config(**overrides))._language()


def test_english_only_model_forces_english():
    assert _lang(whisper_model="small.en", language="fr") == "en"


def test_multilingual_auto_detects():
    assert _lang(whisper_model="small", language="auto") is None


def test_multilingual_pins_configured_language():
    assert _lang(whisper_model="small", language="fr") == "fr"


def test_language_is_case_insensitive():
    assert _lang(whisper_model="small", language="DE") == "de"
