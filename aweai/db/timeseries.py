from __future__ import annotations

import json
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from aweai.db.metadata import QueryBuilder


class TimeSeriesDB:
    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            path = str(Path.home() / ".aweai" / "timeseries.db")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _init_tables(self) -> None:
        conn = self._connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS timeseries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                tags TEXT,
                meta TEXT
            );
            CREATE TABLE IF NOT EXISTS downsampled (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                bucket TEXT NOT NULL,
                count INTEGER,
                sum REAL,
                min REAL,
                max REAL,
                mean REAL,
                tags TEXT,
                meta TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ts_key_ts ON timeseries(key, timestamp);
            CREATE INDEX IF NOT EXISTS idx_ds_key_bucket ON downsampled(key, bucket);
        """)
        conn.commit()

    def insert(self, key: str, value: float, timestamp: Optional[str] = None, tags: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None) -> None:
        conn = self._connect()
        ts = timestamp or datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO timeseries (key, value, timestamp, tags, meta) VALUES (?, ?, ?, ?, ?)",
            (key, float(value), ts, json.dumps(tags or []), json.dumps(meta or {})),
        )
        conn.commit()

    def insert_batch(self, key: str, values: Sequence[float], timestamps: Optional[Sequence[str]] = None, tags: Optional[List[str]] = None, meta: Optional[Dict[str, Any]] = None) -> None:
        conn = self._connect()
        rows = []
        ts = datetime.utcnow().isoformat()
        for i, v in enumerate(values):
            rows.append((key, float(v), timestamps[i] if timestamps and i < len(timestamps) else ts, json.dumps(tags or []), json.dumps(meta or {})))
        conn.executemany(
            "INSERT INTO timeseries (key, value, timestamp, tags, meta) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()

    def query(self, key: str, start: Optional[str] = None, end: Optional[str] = None, tags: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        qb = QueryBuilder("timeseries").eq("key", key)
        if start is not None:
            qb.gte("timestamp", start)
        if end is not None:
            qb.lte("timestamp", end)
        if tags:
            for tag in tags:
                qb.like("tags", f"%{tag}%")
        sql, params = qb.build_select(order_by="timestamp ASC", limit=limit)
        conn = self._connect()
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def aggregate(self, key: str, bucket_size: str = "1h", start: Optional[str] = None, end: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = self.query(key, start=start, end=end)
        if not rows:
            return []
        buckets: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            ts = datetime.fromisoformat(r["timestamp"])
            if bucket_size == "1h":
                bucket = ts.strftime("%Y-%m-%dT%H:00:00")
            elif bucket_size == "1d":
                bucket = ts.strftime("%Y-%m-%dT00:00:00")
            else:
                bucket = ts.strftime("%Y-%m-%dT%H:%M:00")
            buckets[bucket].append(r["value"])
        out = []
        for bucket, vals in sorted(buckets.items()):
            arr = np.array(vals)
            out.append({
                "bucket": bucket,
                "count": int(len(vals)),
                "sum": float(arr.sum()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
            })
        return out

    def downsample(self, key: str, bucket_size: str = "1h", retention: str = "30d") -> None:
        agg = self.aggregate(key, bucket_size)
        conn = self._connect()
        conn.execute("DELETE FROM downsampled WHERE key = ? AND bucket >= ?", (key, retention))
        for a in agg:
            conn.execute(
                "INSERT OR REPLACE INTO downsampled (key, bucket, count, sum, min, max, mean) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, a["bucket"], a["count"], a["sum"], a["min"], a["max"], a["mean"]),
            )
        conn.commit()

    def detect_anomalies(self, key: str, threshold: float = 3.0) -> List[Dict[str, Any]]:
        rows = self.query(key)
        if len(rows) < 3:
            return []
        values = np.array([r["value"] for r in rows])
        mean = values.mean()
        std = values.std()
        if std == 0:
            return []
        z_scores = np.abs((values - mean) / std)
        anomalies = []
        for i, z in enumerate(z_scores):
            if z > threshold:
                anomalies.append({**rows[i], "z_score": float(z), "mean": float(mean), "std": float(std)})
        return anomalies

    def stats(self, key: Optional[str] = None) -> Dict[str, Any]:
        conn = self._connect()
        where = "WHERE key = ?" if key else ""
        params = (key,) if key else ()
        row = conn.execute(f"SELECT COUNT(*) as count, MIN(value) as min_v, MAX(value) as max_v, AVG(value) as avg_v FROM timeseries {where}", params).fetchone()
        return dict(row) if row else {}

    def delete(self, key: str, start: Optional[str] = None, end: Optional[str] = None) -> int:
        qb = QueryBuilder("timeseries").eq("key", key)
        if start is not None:
            qb.gte("timestamp", start)
        if end is not None:
            qb.lte("timestamp", end)
        sql, params = qb.build_select()
        conn = self._connect()
        rows = conn.execute(sql, params).fetchall()
        delete_sql = f"DELETE FROM timeseries WHERE id IN ({','.join(str(r['id']) for r in rows)})"
        cursor = conn.execute(delete_sql)
        conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def __enter__(self) -> TimeSeriesDB:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
