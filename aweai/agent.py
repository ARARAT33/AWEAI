"""Agent runner for AWEAI.

The :class:`Agent` ties together an LLM client, a tool registry, and a
memory store.  It supports a simple function-calling loop: the model may
request tool calls, the agent executes them, feeds results back, and
continues until a final answer is produced (or the step budget runs out).
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from aweai.config import AWEConfig
from aweai.llm import LLMClient, make_client
from aweai.memory import MemoryStore
from aweai.tools import ToolRegistry, default_registry

logger = logging.getLogger("aweai.agent")

DEFAULT_MAX_STEPS = 8


class Agent:
    """A minimal tool-using conversational agent.

    Args:
        config: Application configuration.
        client: Optional LLM client (created from config when omitted).
        tools: Optional tool registry (defaults to built-in tools).
        memory: Optional memory store (created when omitted).
        system_prompt: Override the configured system prompt.
    """

    def __init__(
        self,
        config: Optional[AWEConfig] = None,
        client: Optional[LLMClient] = None,
        tools: Optional[ToolRegistry] = None,
        memory: Optional[MemoryStore] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.config = config or AWEConfig()
        self.client = client or make_client(self.config)
        self.tools = tools if tools is not None else default_registry()
        self.memory = memory or MemoryStore(self.config.db_path)
        self.system_prompt = system_prompt or self.config.system_prompt or (
            "You are AWEAI, a helpful AI assistant. "
            "Use tools when they help answer the user's question."
        )

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------

    def _base_messages(self) -> List[Dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}]

    def _history_messages(
        self, session_id: str = "default", limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        rows = self.memory.get_messages(session_id, limit=limit)
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def _tool_messages(self) -> List[Dict[str, Any]]:
        return [{"type": "function", "function": t} for t in self.tools.schemas()]

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    async def chat(
        self,
        user_message: str,
        session_id: str = "default",
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        history_limit: Optional[int] = 20,
    ) -> str:
        """Send a user message and return the final assistant reply.

        The conversation is persisted to memory; history is loaded from
        the same session to keep context across calls.
        """
        self.memory.add_message("user", user_message, session_id)
        messages = self._base_messages()
        messages.extend(self._history_messages(session_id, limit=history_limit))

        steps = 0
        while steps < max_steps:
            steps += 1
            reply = await self.client.complete(messages + self._tool_messages())
            tool_calls = self._parse_tool_calls(reply)

            if not tool_calls:
                self.memory.add_message("assistant", reply, session_id)
                return reply

            messages.append({"role": "assistant", "content": reply})
            for call in tool_calls:
                name = call.get("name", "")
                arguments = call.get("arguments", {})
                logger.info("Executing tool: %s(%s)", name, arguments)
                try:
                    result = await self.tools.call(name, arguments)
                    content = (
                        result if isinstance(result, str)
                        else json.dumps(result, ensure_ascii=False)
                    )
                except Exception as exc:  # noqa: BLE001 - surface to model
                    content = f"ERROR: {type(exc).__name__}: {exc}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "name": name,
                        "content": content,
                    }
                )

        logger.warning("Agent reached max_steps=%s without final answer", max_steps)
        return "I could not finish answering within the step budget."

    async def stream_chat(
        self,
        user_message: str,
        session_id: str = "default",
        *,
        history_limit: Optional[int] = 20,
    ) -> AsyncIterator[str]:
        """Stream the assistant reply for a message (no tool loop)."""
        self.memory.add_message("user", user_message, session_id)
        messages = self._base_messages()
        messages.extend(self._history_messages(session_id, limit=history_limit))

        collected: List[str] = []
        async for chunk in self.client.stream(messages):
            collected.append(chunk)
            yield chunk

        reply = "".join(collected)
        if reply:
            self.memory.add_message("assistant", reply, session_id)

    # ------------------------------------------------------------------
    # Tool-call parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tool_calls(reply: str) -> List[Dict[str, Any]]:
        """Best-effort parse of tool calls from a model reply.

        Accepts either a JSON array of ``{name, arguments}`` objects or a
        JSON object with a ``tool_calls`` key.  Returns an empty list when
        the reply is not a tool-call payload.
        """
        text = reply.strip()
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []

        candidates: List[Dict[str, Any]] = []
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            candidates = data.get("tool_calls", data.get("calls", []))

        calls: List[Dict[str, Any]] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("function", {}).get("name")
            arguments = item.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if name and isinstance(arguments, dict):
                calls.append({"name": name, "arguments": arguments})
        return calls

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying HTTP client (if any)."""
        close = getattr(self.client, "close", None)
        if callable(close):
            await close()
        self.memory.close()
