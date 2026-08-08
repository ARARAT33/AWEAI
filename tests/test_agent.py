"""Tests for aweai.agent."""

import pytest

from aweai.agent import Agent
from aweai.config import AWEConfig
from aweai.llm import EchoClient
from aweai.memory import MemoryStore
from aweai.tools import ToolRegistry


class FakeLLM:
    """Scripted LLM client: returns one reply per call from a queue."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    async def complete(self, messages, **kwargs):
        self.calls += 1
        if not self.replies:
            return "done"
        return self.replies.pop(0)

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_chat_persists_messages(tmp_path):
    agent = Agent(
        config=AWEConfig(db_path=str(tmp_path / "agent.db")),
        client=EchoClient(),
        tools=ToolRegistry(),
    )
    try:
        reply = await agent.chat("hello")
        assert reply == "hello"
        rows = agent.memory.get_messages()
        assert [r["role"] for r in rows] == ["user", "assistant"]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_chat_returns_plain_reply(tmp_path):
    agent = Agent(
        config=AWEConfig(db_path=str(tmp_path / "agent.db")),
        client=FakeLLM(["plain answer"]),
        tools=ToolRegistry(),
    )
    try:
        reply = await agent.chat("hi")
        assert reply == "plain answer"
        assert agent.memory.get_messages()[-1]["content"] == "plain answer"
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_chat_executes_tool_calls(tmp_path):
    registry = ToolRegistry()

    @registry.add
    async def double(x: int) -> int:
        """Double a number."""
        return x * 2

    script = [
        '[{"name": "double", "arguments": {"x": 21}}]',
        "the answer is 42",
    ]
    agent = Agent(
        config=AWEConfig(db_path=str(tmp_path / "agent.db")),
        client=FakeLLM(script),
        tools=registry,
    )
    try:
        reply = await agent.chat("compute")
        assert reply == "the answer is 42"
        # Only user + final assistant are persisted to memory.
        roles = [r["role"] for r in agent.memory.get_messages()]
        assert roles == ["user", "assistant"]
        # The tool was actually executed (2 LLM calls).
        assert agent.client.calls == 2
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_chat_surfaces_tool_error(tmp_path):
    registry = ToolRegistry()

    @registry.add
    async def boom() -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    agent = Agent(
        config=AWEConfig(db_path=str(tmp_path / "agent.db")),
        client=FakeLLM(['[{"name": "boom", "arguments": {}}]']),
        tools=registry,
    )
    try:
        # FakeLLM returns "done" on the second call; the tool error must not
        # crash the loop and the agent must still produce a final answer.
        reply = await agent.chat("go")
        assert reply == "done"
        assert agent.client.calls == 2
    finally:
        await agent.aclose()


def test_parse_tool_calls():
    assert Agent._parse_tool_calls("not json") == []
    assert Agent._parse_tool_calls(
        '[{"name": "add", "arguments": {"a": 1, "b": 2}}]'
    ) == [{"name": "add", "arguments": {"a": 1, "b": 2}}]
    assert Agent._parse_tool_calls(
        '{"tool_calls": [{"name": "x", "arguments": "{\"k\": 1}"}]}'
    ) == [{"name": "x", "arguments": {"k": 1}}]
    assert Agent._parse_tool_calls("") == []
