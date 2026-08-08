"""Tests for aweai.tools."""

import pytest

from aweai.tools import ToolRegistry, default_registry, tool


@tool
async def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b


@pytest.mark.asyncio
async def test_tool_call():
    registry = ToolRegistry([divide])
    result = await registry.call("divide", {"a": 10, "b": 2})
    assert result == 5.0


@pytest.mark.asyncio
async def test_tool_call_missing_required():
    registry = ToolRegistry([divide])
    with pytest.raises(ValueError, match="missing required"):
        await registry.call("divide", {"a": 10})


@pytest.mark.asyncio
async def test_unknown_tool():
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        await registry.call("nope", {})


def test_duplicate_registration():
    registry = ToolRegistry()
    with pytest.raises(ValueError):
        registry.register(divide)
        registry.register(divide)


def test_schema_generation():
    schema = divide.schema()
    fn = schema["function"]
    assert fn["name"] == "divide"
    assert fn["description"] == "Divide two numbers."
    assert "a" in fn["parameters"]["properties"]
    assert fn["parameters"]["required"] == ["a", "b"]


def test_default_registry():
    registry = default_registry()
    assert "add" in registry
    assert "multiply" in registry
    assert "now_utc" in registry
    assert len(registry.names()) == 4


@pytest.mark.asyncio
async def test_add_and_multiply():
    registry = default_registry()
    assert await registry.call("add", {"a": 2, "b": 3}) == 5
    assert await registry.call("multiply", {"a": 4, "b": 5}) == 20
