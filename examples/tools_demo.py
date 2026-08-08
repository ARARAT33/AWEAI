"""Example: custom tool registration and usage.

Run:
    python examples/tools_demo.py
"""

import asyncio

from aweai.agent import Agent
from aweai.tools import ToolRegistry, tool


@tool
async def greet(name: str, excitement: bool = False) -> str:
    """Greet a person by name."""
    suffix = "!" if excitement else "."
    return f"Hello, {name}{suffix}"


async def main() -> None:
    registry = ToolRegistry()
    registry.add(greet)
    registry.register(__import__("aweai.tools", fromlist=["add"]).add)

    print("Registered tools:", registry.names())
    print("Call greet:", await registry.call("greet", {"name": "AWE"}))
    print("Call add:  ", await registry.call("add", {"a": 40, "b": 2}))

    # A very small custom agent that only knows our tools.
    agent = Agent(tools=registry, system_prompt="You are a demo agent.")
    try:
        reply = await agent.chat("demo mode")
        print(f"Agent reply: {reply}")
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
