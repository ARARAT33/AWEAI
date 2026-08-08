"""Tests for aweai.llm."""

import pytest

from aweai.llm import EchoClient, OpenAICompatClient, make_client
from aweai.config import AWEConfig


@pytest.mark.asyncio
async def test_echo_client():
    client = EchoClient()
    reply = await client.complete(
        [{"role": "user", "content": "ping"}, {"role": "user", "content": "pong"}]
    )
    assert reply == "pong"


@pytest.mark.asyncio
async def test_echo_client_empty():
    client = EchoClient()
    assert await client.complete([]) == ""


def test_openai_client_requires_key_or_base_url():
    with pytest.raises(ValueError):
        OpenAICompatClient(api_key=None, base_url=None)


def test_make_client_echo_without_key(monkeypatch):
    monkeypatch.delenv("AWEAI_API_KEY", raising=False)
    config = AWEConfig()
    assert isinstance(make_client(config), EchoClient)
