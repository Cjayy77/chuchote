"""Tests for the `chuchote doctor` result aggregation (checks are stubbed)."""

from chuchote import doctor
from chuchote.config import Config


def _stub(monkeypatch, ollama_status, voice_status=doctor.OK, wake=None):
    monkeypatch.setattr(doctor, "_check_config", lambda: (doctor.OK, "cfg"))
    monkeypatch.setattr(doctor, "_check_ollama", lambda c: (ollama_status, "ollama"))
    monkeypatch.setattr(doctor, "_check_voice", lambda c: (voice_status, "voice"))
    monkeypatch.setattr(
        doctor, "_check_audio", lambda: [(doctor.OK, "mic"), (doctor.OK, "spk")]
    )
    monkeypatch.setattr(doctor, "_check_wake_deps", lambda c: wake)


def test_all_ok_passes(monkeypatch, capsys):
    _stub(monkeypatch, doctor.OK)
    assert doctor.run(Config()) is True
    assert "All checks passed" in capsys.readouterr().out


def test_any_fail_reports_and_returns_false(monkeypatch, capsys):
    _stub(monkeypatch, doctor.FAIL, voice_status=doctor.FAIL)
    assert doctor.run(Config()) is False
    out = capsys.readouterr().out
    assert "2 problem(s) found" in out
    assert "[fail]" in out


def test_wake_check_included_when_present(monkeypatch, capsys):
    _stub(monkeypatch, doctor.OK, wake=(doctor.OK, "wake ready"))
    assert doctor.run(Config()) is True
    assert "wake ready" in capsys.readouterr().out


def test_language_check_skipped_for_english_or_auto():
    assert doctor._check_language(Config(language="auto")) is None
    assert doctor._check_language(Config(language="en")) is None


def test_language_check_warns_on_english_only_model():
    status, msg = doctor._check_language(
        Config(language="fr", whisper_model="small.en")
    )
    assert status == doctor.WARN and "multilingual" in msg


def test_language_check_ok_with_multilingual_model():
    status, _ = doctor._check_language(Config(language="fr", whisper_model="small"))
    assert status == doctor.OK


def test_memory_check_warns_when_tight(monkeypatch):
    monkeypatch.setattr(doctor, "_free_ram_gb", lambda: 0.4)
    status, msg = doctor._check_memory(Config())
    assert status == doctor.WARN and "0.4" in msg


def test_memory_check_warns_when_llm_wont_fit(monkeypatch):
    monkeypatch.setattr(doctor, "_free_ram_gb", lambda: 2.0)
    status, msg = doctor._check_memory(Config(ollama_model="qwen2.5-coder:3b"))
    assert status == doctor.WARN and "qwen2.5-coder:3b" in msg


def test_memory_check_ok_with_headroom(monkeypatch):
    monkeypatch.setattr(doctor, "_free_ram_gb", lambda: 6.0)
    status, _ = doctor._check_memory(Config())
    assert status == doctor.OK


def test_memory_check_skipped_when_unknown(monkeypatch):
    monkeypatch.setattr(doctor, "_free_ram_gb", lambda: None)
    assert doctor._check_memory(Config()) is None
