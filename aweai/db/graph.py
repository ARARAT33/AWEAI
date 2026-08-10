from __future__ import annotations

import json
import sqlite3
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from aweai.db.metadata import QueryBuilder


class GraphDB:
    def __init__(self, path: Optional[str] = None) -> None:
        if path is None:
            path = str(Path.home() / ".aweai" / "graph.db")
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
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                properties TEXT,
                tags TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                label TEXT NOT NULL,
                properties TEXT,
                weight REAL DEFAULT 1.0,
                created_at TEXT,
                FOREIGN KEY (source) REFERENCES nodes(id),
                FOREIGN KEY (target) REFERENCES nodes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
            CREATE INDEX IF NOT EXISTS idx_edges_label ON edges(label);
            CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);
        """)
        conn.commit()

    def add_node(self, node_id: str, label: str, properties: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None) -> str:
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO nodes (id, label, properties, tags, created_at) VALUES (?, ?, ?, ?, ?)",
            (node_id, label, json.dumps(properties or {}), json.dumps(tags or []), datetime.utcnow().isoformat()),
        )
        conn.commit()
        return node_id

    def add_edge(self, edge_id: str, source: str, target: str, label: str, properties: Optional[Dict[str, Any]] = None, weight: float = 1.0) -> str:
        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO edges (id, source, target, label, properties, weight, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (edge_id, source, target, label, json.dumps(properties or {}), float(weight), datetime.utcnow().isoformat()),
        )
        conn.commit()
        return edge_id

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["properties"] = json.loads(data["properties"])
        data["tags"] = json.loads(data["tags"])
        return data

    def get_nodes(self, label: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        qb = QueryBuilder("nodes")
        if label is not None:
            qb.eq("label", label)
        if tags:
            for tag in tags:
                qb.like("tags", f"%{tag}%")
        sql, params = qb.build_select(order_by="created_at DESC")
        conn = self._connect()
        rows = conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            data = dict(r)
            data["properties"] = json.loads(data["properties"])
            data["tags"] = json.loads(data["tags"])
            out.append(data)
        return out

    def get_edges(self, source: Optional[str] = None, target: Optional[str] = None, label: Optional[str] = None) -> List[Dict[str, Any]]:
        qb = QueryBuilder("edges")
        if source is not None:
            qb.eq("source", source)
        if target is not None:
            qb.eq("target", target)
        if label is not None:
            qb.eq("label", label)
        sql, params = qb.build_select(order_by="created_at DESC")
        conn = self._connect()
        rows = conn.execute(sql, params).fetchall()
        out = []
        for r in rows:
            data = dict(r)
            data["properties"] = json.loads(data["properties"])
            out.append(data)
        return out

    def neighbors(self, node_id: str, direction: str = "both") -> List[Dict[str, Any]]:
        conn = self._connect()
        if direction == "out":
            rows = conn.execute("SELECT target FROM edges WHERE source = ?", (node_id,)).fetchall()
            ids = [r["target"] for r in rows]
        elif direction == "in":
            rows = conn.execute("SELECT source FROM edges WHERE target = ?", (node_id,)).fetchall()
            ids = [r["source"] for r in rows]
        else:
            rows = conn.execute("SELECT source FROM edges WHERE target = ? UNION SELECT target FROM edges WHERE source = ?", (node_id, node_id)).fetchall()
            ids = [r["source"] for r in rows]
        return [self.get_node(nid) for nid in ids if self.get_node(nid) is not None]

    def bfs(self, start: str, max_depth: int = 10) -> List[Dict[str, Any]]:
        visited = set()
        queue = deque([(start, 0)])
        result = []
        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited or depth > max_depth:
                continue
            visited.add(node_id)
            node = self.get_node(node_id)
            if node:
                result.append(node)
            for neighbor in self.neighbors(node_id, direction="both"):
                if neighbor["id"] not in visited:
                    queue.append((neighbor["id"], depth + 1))
        return result

    def dfs(self, start: str, max_depth: int = 10) -> List[Dict[str, Any]]:
        visited = set()
        stack = [(start, 0)]
        result = []
        while stack:
            node_id, depth = stack.pop()
            if node_id in visited or depth > max_depth:
                continue
            visited.add(node_id)
            node = self.get_node(node_id)
            if node:
                result.append(node)
            for neighbor in self.neighbors(node_id, direction="both"):
                if neighbor["id"] not in visited:
                    stack.append((neighbor["id"], depth + 1))
        return result

    def pattern_match(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        matches = []
        nodes = self.get_nodes()
        for node in nodes:
            match = True
            for k, v in pattern.items():
                if k == "label" and node.get("label") != v:
                    match = False
                    break
                if k == "tags" and not set(v).issubset(set(node.get("tags", []))):
                    match = False
                    break
                if k == "properties":
                    for pk, pv in v.items():
                        if node.get("properties", {}).get(pk) != pv:
                            match = False
                            break
            if match:
                matches.append(node)
        return matches

    def shortest_path(self, source: str, target: str) -> List[Dict[str, Any]]:
        from collections import deque
        visited = {source}
        queue = deque([(source, [source])])
        while queue:
            node, path = queue.popleft()
            if node == target:
                return [self.get_node(n) for n in path]
            for neighbor in self.neighbors(node, direction="both"):
                if neighbor["id"] not in visited:
                    visited.add(neighbor["id"])
                    queue.append((neighbor["id"], path + [neighbor["id"]]))
        return []

    def lineage(self, model_id: str) -> Dict[str, List[Dict[str, Any]]]:
        ancestors = []
        descendants = []
        queue = deque([model_id])
        while queue:
            nid = queue.popleft()
            for e in self.get_edges(target=nid):
                if e["source"] not in [a["id"] for a in ancestors]:
                    src = self.get_node(e["source"])
                    if src:
                        ancestors.append(src)
                    queue.append(e["source"])
            for e in self.get_edges(source=nid):
                if e["target"] not in [d["id"] for d in descendants]:
                    tgt = self.get_node(e["target"])
                    if tgt:
                        descendants.append(tgt)
                    queue.append(e["target"])
        return {"ancestors": ancestors, "descendants": descendants}

    def delete_node(self, node_id: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM edges WHERE source = ? OR target = ?", (node_id, node_id))
        conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        conn.commit()

    def delete_edge(self, edge_id: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        conn.commit()

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def __enter__(self) -> GraphDB:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
