from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class KVBackend:
    def set(self, namespace: str, key: str, value: str, ttl: Optional[float] = None) -> None:
        raise NotImplementedError

    def get(self, namespace: str, key: str) -> Optional[str]:
        raise NotImplementedError

    def delete(self, namespace: str, key: str) -> None:
        raise NotImplementedError

    def keys(self, namespace: str) -> List[str]:
        raise NotImplementedError

    def cleanup(self, now: float) -> int:
        return 0


class SQLiteBackend(KVBackend):
    def __init__(self, path: str) -> None:
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS kv_store (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                ttl REAL,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (namespace, key)
            );
            CREATE INDEX IF NOT EXISTS idx_kv_ns ON kv_store(namespace);
            CREATE INDEX IF NOT EXISTS idx_kv_ttl ON kv_store(ttl);
        """)
        self.conn.commit()

    def set(self, namespace: str, key: str, value: str, ttl: Optional[float] = None) -> None:
        ts = datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO kv_store (namespace, key, value, ttl, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (namespace, key, value, ttl, ts, ts),
        )
        self.conn.commit()

    def get(self, namespace: str, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value, ttl FROM kv_store WHERE namespace = ? AND key = ?", (namespace, key)).fetchone()
        if row is None:
            return None
        if row["ttl"] is not None and time.time() > row["ttl"]:
            self.delete(namespace, key)
            return None
        return row["value"]

    def delete(self, namespace: str, key: str) -> None:
        self.conn.execute("DELETE FROM kv_store WHERE namespace = ? AND key = ?", (namespace, key))
        self.conn.commit()

    def keys(self, namespace: str) -> List[str]:
        rows = self.conn.execute("SELECT key FROM kv_store WHERE namespace = ?", (namespace,)).fetchall()
        return [r["key"] for r in rows]

    def cleanup(self, now: float) -> int:
        cursor = self.conn.execute("DELETE FROM kv_store WHERE ttl IS NOT NULL AND ttl <= ?", (now,))
        self.conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        self.conn.close()


class KVStore:
    def __init__(self, path: Optional[str] = None, backend: str = "sqlite") -> None:
        if path is None:
            path = str(Path.home() / ".aweai" / "kv.db")
        self.path = Path(path)
        self.backend = backend
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if backend == "sqlite":
            self._backend: KVBackend = SQLiteBackend(str(self.path))
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def set(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl_val = time.time() + ttl if ttl is not None else None
        self._backend.set(namespace, key, json.dumps(value), ttl_val)

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        raw = self._backend.get(namespace, key)
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except Exception:
            return raw

    def delete(self, namespace: str, key: str) -> None:
        self._backend.delete(namespace, key)

    def exists(self, namespace: str, key: str) -> bool:
        return self._backend.get(namespace, key) is not None

    def keys(self, namespace: str, pattern: str = "*") -> List[str]:
        all_keys = self._backend.keys(namespace)
        if pattern == "*":
            return all_keys
        import fnmatch
        return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]

    def get_all(self, namespace: str) -> Dict[str, Any]:
        all_keys = self._backend.keys(namespace)
        result = {}
        for k in all_keys:
            v = self.get(namespace, k)
            if v is not None:
                result[k] = v
        return result

    def compare_and_swap(self, namespace: str, key: str, expected: Any, new_value: Any) -> bool:
        current = self.get(namespace, key, default=None)
        if current != expected:
            return False
        self.set(namespace, key, new_value)
        return True

    def increment(self, namespace: str, key: str, delta: float = 1.0) -> float:
        current = self.get(namespace, key, default=0.0)
        new_val = float(current) + delta
        self.set(namespace, key, new_val)
        return new_val

    def cleanup_expired(self) -> int:
        return self._backend.cleanup(time.time())

    def list_namespaces(self) -> List[str]:
        return []

    def delete_namespace(self, namespace: str) -> int:
        keys = self._backend.keys(namespace)
        for k in keys:
            self._backend.delete(namespace, k)
        return len(keys)

    def close(self) -> None:
        if hasattr(self._backend, "close"):
            self._backend.close()

    def __enter__(self) -> KVStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
