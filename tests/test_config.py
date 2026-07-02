"""Tests for config loading precedence and the init template."""

from chuchote.config import Config, write_default_config


def test_defaults_when_no_file(tmp_path):
    cfg = Config.load(str(tmp_path / "does-not-exist.toml"))
    assert cfg.ollama_model == "llama3.2"
    assert cfg.mode == "wake"


def test_file_overrides_defaults(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        'ollama_model = "qwen2.5-coder:3b"\n'
        "wake_threshold = 0.7\n"
        "vad_silence_ms = 500\n",
        encoding="utf-8",
    )
    cfg = Config.load(str(p))
    assert cfg.ollama_model == "qwen2.5-coder:3b"
    assert cfg.wake_threshold == 0.7
    assert cfg.vad_silence_ms == 500
    assert cfg.whisper_model == Config().whisper_model  # untouched keys keep defaults


def test_unknown_keys_are_ignored(tmp_path, capsys):
    p = tmp_path / "config.toml"
    p.write_text('bogus_key = 1\nollama_model = "x"\n', encoding="utf-8")
    cfg = Config.load(str(p))
    assert cfg.ollama_model == "x"
    assert not hasattr(cfg, "bogus_key")
    assert "bogus_key" in capsys.readouterr().err


def test_optional_chuchote_table(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[chuchote]\nmode = "ptt"\n', encoding="utf-8")
    assert Config.load(str(p)).mode == "ptt"


def test_written_template_loads_to_defaults(tmp_path):
    p = tmp_path / "config.toml"
    write_default_config(str(p))
    loaded = Config.load(str(p))
    defaults = Config()
    # A fresh template is entirely commented out → identical to defaults.
    assert loaded.ollama_model == defaults.ollama_model
    assert loaded.mode == defaults.mode
    assert loaded.barge_in_mode == defaults.barge_in_mode
