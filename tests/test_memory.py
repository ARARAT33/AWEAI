"""Tests for aweai.memory."""

from aweai.memory import MemoryStore


def test_add_and_get_messages(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    try:
        store.add_message("user", "hello")
        store.add_message("assistant", "hi there")
        rows = store.get_messages()
        assert len(rows) == 2
        assert rows[0]["role"] == "user"
        assert rows[0]["content"] == "hello"
        assert rows[1]["role"] == "assistant"
    finally:
        store.close()


def test_limit_returns_most_recent_chronological(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    try:
        for i in range(5):
            store.add_message("user", f"msg-{i}")
        rows = store.get_messages(limit=3)
        assert [r["content"] for r in rows] == ["msg-2", "msg-3", "msg-4"]
    finally:
        store.close()


def test_sessions_are_isolated(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    try:
        store.add_message("user", "alpha", session_id="s1")
        store.add_message("user", "beta", session_id="s2")
        assert len(store.get_messages(session_id="s1")) == 1
        assert len(store.get_messages(session_id="s2")) == 1
    finally:
        store.close()


def test_clear_session(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    try:
        store.add_message("user", "hello")
        assert store.clear_session() == 1
        assert store.get_messages() == []
    finally:
        store.close()


def test_kv_roundtrip(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    try:
        store.set("name", "AWE")
        store.set("nested", {"a": [1, 2, 3]})
        assert store.get("name") == "AWE"
        assert store.get("nested") == {"a": [1, 2, 3]}
        assert store.get("missing", "fallback") == "fallback"
        assert store.delete("name") is True
        assert store.delete("name") is False
        assert store.keys() == ["nested"]
    finally:
        store.close()


def test_search(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    try:
        store.add_message("user", "I love Armenian coffee")
        store.add_message("user", "I prefer tea")
        results = store.search("coffee")
        assert len(results) == 1
        assert "coffee" in results[0]["content"]
    finally:
        store.close()


def test_stats(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.db"))
    try:
        store.add_message("user", "hi")
        store.set("k", 1)
        stats = store.stats()
        assert stats["messages"] == 1
        assert stats["kv_keys"] == 1
    finally:
        store.close()


def test_in_memory_db():
    store = MemoryStore(":memory:")
    try:
        store.add_message("user", "hi")
        assert store.stats()["messages"] == 1
    finally:
        store.close()
