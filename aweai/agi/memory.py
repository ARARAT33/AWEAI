from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MemoryItem:
    key: str
    content: Any
    embedding: np.ndarray
    timestamp: float = field(default_factory=time.time)
    strength: float = 1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decay(self, rate: float = 0.01) -> None:
        self.strength = max(0.0, self.strength - rate * (time.time() - self.timestamp) / 3600.0)

    def is_weak(self, threshold: float = 0.2) -> bool:
        return self.strength < threshold


class WorkingMemory:
    def __init__(self, capacity: int = 7) -> None:
        self.capacity = capacity
        self._buffer: List[MemoryItem] = []
        self._attention_weights: Dict[str, float] = {}

    def store(self, content: Any, key: Optional[str] = None, tags: Optional[List[str]] = None) -> MemoryItem:
        if len(self._buffer) >= self.capacity:
            self._buffer.pop(0)
        k = key or hashlib.sha256(str(content).encode("utf-8")).hexdigest()[:12]
        embedding = self._embed(content)
        item = MemoryItem(key=k, content=content, embedding=embedding, tags=tags or [])
        self._buffer.append(item)
        return item

    def retrieve(self, key: str) -> Optional[MemoryItem]:
        for item in self._buffer:
            if item.key == key:
                item.strength = min(1.0, item.strength + 0.1)
                return item
        return None

    def recall(self, query: Any, top_k: int = 3) -> List[MemoryItem]:
        q_emb = self._embed(query)
        scored = []
        for item in self._buffer:
            sim = float(np.dot(q_emb, item.embedding))
            att = self._attention_weights.get(item.key, 1.0)
            scored.append((sim * att, item))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def set_attention(self, key: str, weight: float) -> None:
        self._attention_weights[key] = max(0.0, min(1.0, weight))

    def clear(self) -> None:
        self._buffer.clear()
        self._attention_weights.clear()

    def snapshot(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": item.key,
                "content": str(item.content)[:200],
                "strength": item.strength,
                "tags": item.tags,
            }
            for item in self._buffer
        ]

    def _embed(self, content: Any) -> np.ndarray:
        text = str(content)
        vec = np.zeros(64, dtype=float)
        for i, char in enumerate(text):
            h = hash(char) % 64
            vec[h] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


class LongTermMemory:
    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self._store: Dict[str, MemoryItem] = {}
        self._index: List[Tuple[str, np.ndarray]] = []

    def store(self, content: Any, key: Optional[str] = None, tags: Optional[List[str]] = None) -> MemoryItem:
        k = key or hashlib.sha256(str(content).encode("utf-8")).hexdigest()[:12]
        embedding = self._embed(content)
        item = MemoryItem(key=k, content=content, embedding=embedding, tags=tags or [])
        self._store[k] = item
        self._index.append((k, embedding))
        return item

    def recall(self, query: Any, top_k: int = 5) -> List[MemoryItem]:
        q_emb = self._embed(query)
        scored = [(float(np.dot(q_emb, emb)), k) for k, emb in self._index]
        scored.sort(key=lambda t: t[0], reverse=True)
        results = []
        for _, k in scored[:top_k]:
            item = self._store.get(k)
            if item:
                item.strength = min(1.0, item.strength + 0.05)
                results.append(item)
        return results

    def search_by_tags(self, tags: Sequence[str]) -> List[MemoryItem]:
        return [item for item in self._store.values() if any(t in item.tags for t in tags)]

    def forget(self, threshold: float = 0.1) -> List[str]:
        to_remove = []
        for k, item in self._store.items():
            item.decay()
            if item.is_weak(threshold):
                to_remove.append(k)
        for k in to_remove:
            del self._store[k]
            self._index = [(ki, emb) for ki, emb in self._index if ki != k]
        return to_remove

    def consolidate(self, working_memory: WorkingMemory) -> List[MemoryItem]:
        consolidated = []
        for item in working_memory._buffer:
            if item.strength > 0.5 and item.key not in self._store:
                lt = self.store(item.content, key=item.key, tags=item.tags)
                consolidated.append(lt)
        return consolidated

    def save(self, path: str) -> None:
        data = {
            k: {
                "content": str(v.content)[:1000],
                "embedding": v.embedding.tolist(),
                "timestamp": v.timestamp,
                "strength": v.strength,
                "tags": v.tags,
                "metadata": v.metadata,
            }
            for k, v in self._store.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._store.clear()
        self._index.clear()
        for k, v in data.items():
            emb = np.array(v["embedding"], dtype=float)
            item = MemoryItem(
                key=k,
                content=v["content"],
                embedding=emb,
                timestamp=v["timestamp"],
                strength=v["strength"],
                tags=v.get("tags", []),
                metadata=v.get("metadata", {}),
            )
            self._store[k] = item
            self._index.append((k, emb))

    def stats(self) -> Dict[str, Any]:
        return {"total_items": len(self._store), "index_size": len(self._index)}

    def _embed(self, content: Any) -> np.ndarray:
        text = str(content)
        vec = np.zeros(self.dim, dtype=float)
        for i, char in enumerate(text):
            h = hash(char) % self.dim
            vec[h] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


class EpisodicMemory:
    def __init__(self) -> None:
        self._episodes: List[Dict[str, Any]] = []

    def record(self, event: Dict[str, Any]) -> None:
        episode = {
            "id": hashlib.sha256(json.dumps(event, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12],
            "event": event,
            "timestamp": time.time(),
            "context": {"working_memory_snapshot": None, "emotional_valence": 0.0},
        }
        self._episodes.append(episode)
        if len(self._episodes) > 5000:
            self._episodes = self._episodes[-5000:]

    def replay(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        scored = []
        q_lower = query.lower()
        for ep in self._episodes:
            text = json.dumps(ep["event"], default=str).lower()
            score = text.count(q_lower)
            recency = 1.0 / (1.0 + (time.time() - ep["timestamp"]) / 86400.0)
            scored.append((score + recency, ep))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        return list(self._episodes[-limit:])

    def count(self) -> int:
        return len(self._episodes)


class AutobiographicalMemory:
    def __init__(self) -> None:
        self._timeline: List[Dict[str, Any]] = []

    def record_event(self, description: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self._timeline.append({
            "description": description,
            "timestamp": time.time(),
            "metadata": metadata or {},
        })

    def get_life_summary(self) -> Dict[str, Any]:
        return {
            "total_events": len(self._timeline),
            "first_event": self._timeline[0]["timestamp"] if self._timeline else None,
            "last_event": self._timeline[-1]["timestamp"] if self._timeline else None,
        }

    def search(self, query: str) -> List[Dict[str, Any]]:
        return [e for e in self._timeline if query.lower() in e["description"].lower()]


class ProceduralMemory:
    def __init__(self) -> None:
        self._skills: Dict[str, Dict[str, Any]] = {}

    def learn(self, skill_name: str, procedure: Callable, context: Optional[Dict[str, Any]] = None) -> None:
        self._skills[skill_name] = {
            "procedure": procedure,
            "context": context or {},
            "mastery": 0.0,
            "invocation_count": 0,
            "success_count": 0,
        }

    def execute(self, skill_name: str, **kwargs: Any) -> Any:
        if skill_name not in self._skills:
            raise KeyError(f"Unknown skill: {skill_name}")
        skill = self._skills[skill_name]
        skill["invocation_count"] += 1
        try:
            result = skill["procedure"](**kwargs)
            skill["success_count"] += 1
            skill["mastery"] = min(1.0, skill["success_count"] / max(skill["invocation_count"], 1))
            return result
        except Exception as e:
            return {"error": str(e)}

    def get_mastery(self, skill_name: str) -> float:
        return self._skills.get(skill_name, {}).get("mastery", 0.0)

    def list_skills(self) -> List[str]:
        return list(self._skills.keys())


class MemoryConsolidator:
    def __init__(self, working: WorkingMemory, long_term: LongTermMemory) -> None:
        self.working = working
        self.long_term = long_term
        self._consolidation_log: List[Dict[str, Any]] = []

    def consolidate(self) -> List[MemoryItem]:
        items = self.long_term.consolidate(self.working)
        record = {
            "timestamp": time.time(),
            "consolidated_count": len(items),
            "items": [i.key for i in items],
        }
        self._consolidation_log.append(record)
        return items

    def forgetting_curve(self, item: MemoryItem, elapsed_hours: float) -> float:
        base = 0.5
        decay = 0.1
        return base * math.exp(-decay * elapsed_hours)

    def log(self) -> List[Dict[str, Any]]:
        return list(self._consolidation_log)
