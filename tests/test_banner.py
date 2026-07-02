"""Tests for the startup banner and its ANSI styling gate."""

from chuchote import banner
from chuchote.config import Config


class _FakeTTY:
    def isatty(self):
        return True


class _FakePipe:
    def isatty(self):
        return False


def test_disabled_banner_is_empty():
    assert banner.render(Config(banner=False)) == ""


def test_enabled_banner_has_tagline_and_credit():
    out = banner.render(Config(banner=True))
    assert "voice assistant for Ollama" in out
    assert "Cjayy77" in out


def test_no_ansi_when_not_a_terminal(monkeypatch):
    monkeypatch.setattr(banner.sys, "stdout", _FakePipe())
    out = banner.render(Config(banner=True))
    assert "\033[" not in out


def test_italic_applied_on_a_real_terminal(monkeypatch):
    monkeypatch.setattr(banner, "_vt_ready", True)
    monkeypatch.setattr(banner.sys, "stdout", _FakeTTY())
    monkeypatch.delenv("NO_COLOR", raising=False)
    out = banner.render(Config(banner=True))
    assert f"{banner._ITALIC_ON}{banner._NAME}{banner._ITALIC_OFF}" in out


def test_no_color_env_disables_styling(monkeypatch):
    monkeypatch.setattr(banner, "_vt_ready", True)
    monkeypatch.setattr(banner.sys, "stdout", _FakeTTY())
    monkeypatch.setenv("NO_COLOR", "1")
    out = banner.render(Config(banner=True))
    assert "\033[" not in out
