from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class QueryBuilder:
    def __init__(self, table: str) -> None:
        self.table = table
        self._conditions: List[str] = []
        self._params: List[Any] = []

    def where(self, condition: str, *params: Any) -> QueryBuilder:
        self._conditions.append(condition)
        self._params.extend(params)
        return self

    def eq(self, column: str, value: Any) -> QueryBuilder:
        return self.where(f"{column} = ?", value)

    def gt(self, column: str, value: Any) -> QueryBuilder:
        return self.where(f"{column} > ?", value)

    def lt(self, column: str, value: Any) -> QueryBuilder:
        return self.where(f"{column} < ?", value)

    def gte(self, column: str, value: Any) -> QueryBuilder:
        return self.where(f"{column} >= ?", value)

    def lte(self, column: str, value: Any) -> QueryBuilder:
        return self.where(f"{column} <= ?", value)

    def like(self, column: str, value: str) -> QueryBuilder:
        return self.where(f"{column} LIKE ?", value)

    def in_(self, column: str, values: Sequence[Any]) -> QueryBuilder:
        placeholders = ", ".join("?" for _ in values)
        return self.where(f"{column} IN ({placeholders})", *values)

    def build_select(self, columns: str = "*", order_by: Optional[str] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> Tuple[str, List[Any]]:
        sql = f"SELECT {columns} FROM {self.table}"
        params: List[Any] = []
        if self._conditions:
            sql += " WHERE " + " AND ".join(self._conditions)
            params = list(self._params)
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += f" LIMIT {limit}"
        if offset is not None:
            sql += f" OFFSET {offset}"
        return sql, params


class DBAdapter:
    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> Any:
        raise NotImplementedError

    def query(self, sql: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def executescript(self, sql: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SQLiteAdapter(DBAdapter):
    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> Any:
        cursor = self.conn.execute(sql, list(params) if params else [])
        self.conn.commit()
        return cursor

    def query(self, sql: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        rows = self.conn.execute(sql, list(params) if params else []).fetchall()
        return [dict(r) for r in rows]

    def executescript(self, sql: str) -> None:
        self.conn.executescript(sql)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class MetadataDB:
    def __init__(self, path: Optional[str] = None, adapter: Optional[DBAdapter] = None) -> None:
        if path is None:
            path = str(Path.home() / ".aweai" / "metadata.db")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.adapter = adapter or SQLiteAdapter(str(self.path))
        self._init_tables()

    def _init_tables(self) -> None:
        self.adapter.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                config TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                experiment_id TEXT,
                name TEXT,
                status TEXT DEFAULT 'running',
                config TEXT,
                metrics TEXT,
                artifacts TEXT,
                start_time TEXT,
                end_time TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value REAL,
                step INTEGER,
                timestamp TEXT,
                meta TEXT
            );
            CREATE TABLE IF NOT EXISTS params (
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                PRIMARY KEY (run_id, key)
            );
            CREATE TABLE IF NOT EXISTS configs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT,
                architecture TEXT,
                params TEXT,
                metrics TEXT,
                path TEXT,
                tags TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_run_key ON metrics(run_id, key);
            CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
            CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);
        """)

    def create_experiment(self, name: str, description: str = "", config: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None) -> str:
        exp_id = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        now = datetime.utcnow().isoformat()
        self.adapter.execute(
            "INSERT INTO experiments (id, name, description, tags, config, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (exp_id, name, description, json.dumps(tags or []), json.dumps(config or {}), now, now),
        )
        return exp_id

    def create_run(self, experiment_id: str, name: str = "", config: Optional[Dict[str, Any]] = None) -> str:
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        now = datetime.utcnow().isoformat()
        self.adapter.execute(
            "INSERT INTO runs (id, experiment_id, name, config, start_time, status) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, experiment_id, name, json.dumps(config or {}), now, "running"),
        )
        return run_id

    def log_metric(self, run_id: str, key: str, value: float, step: Optional[int] = None, meta: Optional[Dict[str, Any]] = None) -> None:
        self.adapter.execute(
            "INSERT INTO metrics (run_id, key, value, step, timestamp, meta) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, key, float(value), step, datetime.utcnow().isoformat(), json.dumps(meta or {})),
        )

    def log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            self.adapter.execute(
                "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?, ?, ?)",
                (run_id, key, json.dumps(value)),
            )

    def log_model(self, name: str, version: str, architecture: str, params: Dict[str, Any], metrics: Dict[str, Any], path: str = "", tags: Optional[List[str]] = None) -> str:
        model_id = f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        now = datetime.utcnow().isoformat()
        self.adapter.execute(
            "INSERT INTO models (id, name, version, architecture, params, metrics, path, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (model_id, name, version, architecture, json.dumps(params), json.dumps(metrics), path, json.dumps(tags or []), now, now),
        )
        return model_id

    def save_config(self, name: str, content: Dict[str, Any], tags: Optional[List[str]] = None) -> str:
        config_id = f"cfg_{name}"
        now = datetime.utcnow().isoformat()
        self.adapter.execute(
            "INSERT OR REPLACE INTO configs (id, name, content, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (config_id, name, json.dumps(content), json.dumps(tags or []), now, now),
        )
        return config_id

    def get_metrics(self, run_id: str, key: Optional[str] = None) -> List[Dict[str, Any]]:
        qb = QueryBuilder("metrics").eq("run_id", run_id)
        if key is not None:
            qb.eq("key", key)
        sql, params = qb.build_select(order_by="step ASC, timestamp ASC")
        return self.adapter.query(sql, params)

    def get_runs(self, experiment_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        qb = QueryBuilder("runs")
        if experiment_id is not None:
            qb.eq("experiment_id", experiment_id)
        if status is not None:
            qb.eq("status", status)
        sql, params = qb.build_select(order_by="start_time DESC")
        return self.adapter.query(sql, params)

    def get_experiments(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        qb = QueryBuilder("experiments")
        if tags:
            for tag in tags:
                qb.like("tags", f"%{tag}%")
        sql, params = qb.build_select(order_by="created_at DESC")
        return self.adapter.query(sql, params)

    def get_models(self, name: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        qb = QueryBuilder("models")
        if name is not None:
            qb.eq("name", name)
        if tags:
            for tag in tags:
                qb.like("tags", f"%{tag}%")
        sql, params = qb.build_select(order_by="created_at DESC")
        return self.adapter.query(sql, params)

    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        row = self.adapter.query("SELECT * FROM configs WHERE id = ?", (f"cfg_{name}",))
        if not row:
            return None
        data = row[0]
        data["content"] = json.loads(data["content"])
        data["tags"] = json.loads(data["tags"])
        return data

    def query(self, sql: str, params: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        return self.adapter.query(sql, params)

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> int:
        cursor = self.adapter.execute(sql, params)
        return cursor.rowcount

    def backup(self, path: str) -> None:
        if isinstance(self.adapter, SQLiteAdapter):
            with open(path, "w", encoding="utf-8") as f:
                for line in self.adapter.conn.iterdump():
                    f.write(f"{line}\n")

    def restore(self, path: str) -> None:
        if isinstance(self.adapter, SQLiteAdapter):
            with open(path, "r", encoding="utf-8") as f:
                self.adapter.executescript(f.read())

    def close(self) -> None:
        self.adapter.close()

    def __enter__(self) -> MetadataDB:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
