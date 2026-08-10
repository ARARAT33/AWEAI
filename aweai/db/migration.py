from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class MigrationRunner:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_migrations_table()

    def _init_migrations_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT
            )
        """)
        self._conn.commit()

    def get_current_version(self) -> int:
        row = self._conn.execute("SELECT MAX(version) as v FROM schema_migrations").fetchone()
        return int(row["v"]) if row and row["v"] is not None else 0

    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute("SELECT version, name, applied_at FROM schema_migrations ORDER BY version ASC").fetchall()
        return [dict(r) for r in rows]

    def migrate(self, migrations: List[Dict[str, Any]]) -> List[int]:
        current = self.get_current_version()
        applied = []
        for mig in migrations:
            version = int(mig["version"])
            if version <= current:
                continue
            name = str(mig["name"])
            sql = str(mig["sql"])
            self._conn.execute(sql)
            self._conn.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                (version, name, datetime.utcnow().isoformat()),
            )
            applied.append(version)
        self._conn.commit()
        return applied

    def rollback(self, steps: int = 1) -> List[int]:
        current = self.get_current_version()
        rows = self._conn.execute("SELECT version, name FROM schema_migrations ORDER BY version DESC LIMIT ?", (steps,)).fetchall()
        rolled_back = []
        for row in rows:
            version = int(row["version"])
            rollback_sql = f"DROP TABLE IF EXISTS migration_rollback_{version}"
            try:
                self._conn.execute(rollback_sql)
            except Exception:
                pass
            self._conn.execute("DELETE FROM schema_migrations WHERE version = ?", (version,))
            rolled_back.append(version)
        self._conn.commit()
        return rolled_back

    def pending(self, migrations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        current = self.get_current_version()
        return [m for m in migrations if int(m["version"]) > current]

    def schema_diff(self, migrations: List[Dict[str, Any]]) -> Dict[str, Any]:
        current = self.get_current_version()
        pending_migs = [m for m in migrations if int(m["version"]) > current]
        return {
            "current_version": current,
            "pending_count": len(pending_migs),
            "pending": pending_migs,
            "latest_version": max((int(m["version"]) for m in migrations), default=0),
            "needs_migration": len(pending_migs) > 0,
        }

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> Any:
        cursor = self._conn.execute(sql, list(params) if params else [])
        self._conn.commit()
        return cursor

    def query(self, sql: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        rows = self._conn.execute(sql, list(params) if params else []).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> MigrationRunner:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
