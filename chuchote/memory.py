"""SQLite-backed conversation memory (Phase 2).

Persists every exchange and hands the most recent messages back to the loop so
they can be injected into Ollama's context each turn. Survives restarts, so
Chuchote remembers the conversation across sessions.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

from .config import Config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    role    TEXT NOT NULL,
    content TEXT NOT NULL
);
"""


class Memory:
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(os.path.dirname(os.path.abspath(config.db_path)), exist_ok=True)
        # check_same_thread=False: the player/worker threads never touch this,
        # but keeping it off avoids surprises if that changes later.
        self._conn = sqlite3.connect(config.db_path, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def add(self, role: str, content: str) -> None:
        content = content.strip()
        if not content:
            return
        self._conn.execute(
            "INSERT INTO messages (ts, role, content) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), role, content),
        )
        self._conn.commit()

    def recent(self, limit: int | None = None) -> list[dict]:
        """Return the last `limit` messages in chronological order."""
        limit = self.config.history_messages if limit is None else limit
        if limit <= 0:
            return []
        rows = self._conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    def clear(self) -> None:
        self._conn.execute("DELETE FROM messages")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
