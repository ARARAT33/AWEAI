"""SQLite-backed memory store for AWEAI.

Provides persistent storage of conversation messages and arbitrary
key-value metadata with simple keyword search.  All database access is
synchronous (sqlite3) with a thread-safe connection factory, so it can be
used from async code without blocking the event loop for long.

Schema:

* ``messages`` — role, content, created_at, session_id
* ``kv`` — arbitrary key/value metadata
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    """Persistent memory backed by a single SQLite database file.

    Args:
        db_path: Path to the SQLite file (``:memory:`` supported).
    """

    def __init__(self, db_path: str = "aweai.db") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL DEFAULT 'default',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages (session_id, id);

                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        session_id: str = "default",
    ) -> int:
        """Append a message and return its row id."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, _utcnow()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_messages(
        self,
        session_id: str = "default",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return messages for a session, oldest first."""
        if limit is not None:
            # Fetch the most recent ``limit`` rows, then reverse to
            # chronological order.
            rows = self._conn.execute(
                "SELECT id, session_id, role, content, created_at "
                "FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = self._conn.execute(
                "SELECT id, session_id, role, content, created_at "
                "FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_session(self, session_id: str = "default") -> int:
        """Delete all messages in a session; returns deleted count."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            self._conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    # Key-value metadata
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Store an arbitrary JSON-serializable value under ``key``."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO kv (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "value = excluded.value, updated_at = excluded.updated_at",
                (key, json.dumps(value, ensure_ascii=False), _utcnow()),
            )
            self._conn.commit()

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key."""
        row = self._conn.execute(
            "SELECT value FROM kv WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def delete(self, key: str) -> bool:
        """Delete a key; returns True if it existed."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM kv WHERE key = ?", (key,))
            self._conn.commit()
            return cur.rowcount > 0

    def keys(self) -> List[str]:
        """List all stored keys."""
        rows = self._conn.execute("SELECT key FROM kv ORDER BY key").fetchall()
        return [r["key"] for r in rows]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Naive keyword search over message content (LIKE-based)."""
        like = f"%{query}%"
        rows = self._conn.execute(
            "SELECT id, session_id, role, content, created_at "
            "FROM messages WHERE content LIKE ? "
            "ORDER BY id DESC LIMIT ?",
            (like, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Lifecycle / utilities
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return quick database statistics."""
        messages = self._conn.execute(
            "SELECT COUNT(*) AS c FROM messages"
        ).fetchone()["c"]
        kv = self._conn.execute("SELECT COUNT(*) AS c FROM kv").fetchone()["c"]
        return {"messages": messages, "kv_keys": kv, "db_path": self.db_path}

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
