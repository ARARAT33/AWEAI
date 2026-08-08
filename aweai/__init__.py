"""
AWEAI — AI-Worker Engine for Agents & Intelligence.

A lightweight, modular Python toolkit for building AI-powered assistants:
configuration, LLM client abstraction, tool registry, SQLite-backed
memory, and a simple agent runner — plus a CLI and examples.

Public API:
    from aweai.config import AWEConfig
    from aweai.llm import LLMClient, EchoClient
    from aweai.tools import ToolRegistry, tool
    from aweai.memory import MemoryStore
    from aweai.agent import Agent
"""

from aweai.config import AWEConfig
from aweai.llm import EchoClient, LLMClient, OpenAICompatClient
from aweai.tools import ToolRegistry, tool
from aweai.memory import MemoryStore
from aweai.agent import Agent

__all__ = [
    "AWEConfig",
    "LLMClient",
    "EchoClient",
    "OpenAICompatClient",
    "ToolRegistry",
    "tool",
    "MemoryStore",
    "Agent",
]

__version__ = "0.1.0"
