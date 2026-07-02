"""Tests for SQLite-backed conversation memory."""

from chuchote.config import Config
from chuchote.memory import Memory


def _memory(tmp_path, **overrides):
    cfg = Config(db_path=str(tmp_path / "memory.db"), **overrides)
    return cfg, Memory(cfg)


def test_recent_returns_chronological_order(tmp_path):
    _, mem = _memory(tmp_path)
    mem.add("user", "hi")
    mem.add("assistant", "hello")
    mem.add("user", "bye")
    rows = mem.recent()
    assert [r["content"] for r in rows] == ["hi", "hello", "bye"]
    assert rows[0]["role"] == "user" and rows[1]["role"] == "assistant"


def test_recent_is_limited_to_history_messages(tmp_path):
    _, mem = _memory(tmp_path, history_messages=2)
    for i in range(5):
        mem.add("user", f"m{i}")
    rows = mem.recent()
    assert [r["content"] for r in rows] == ["m3", "m4"]  # last 2, in order


def test_blank_messages_are_skipped(tmp_path):
    _, mem = _memory(tmp_path)
    mem.add("user", "   ")
    mem.add("assistant", "")
    assert mem.recent() == []


def test_persists_across_reopen(tmp_path):
    cfg, mem = _memory(tmp_path)
    mem.add("user", "remember me")
    mem.close()
    reopened = Memory(cfg)
    assert [r["content"] for r in reopened.recent()] == ["remember me"]


def test_clear_empties_history(tmp_path):
    _, mem = _memory(tmp_path)
    mem.add("user", "hi")
    mem.clear()
    assert mem.recent() == []


def test_recent_with_zero_limit(tmp_path):
    _, mem = _memory(tmp_path)
    mem.add("user", "hi")
    assert mem.recent(0) == []
