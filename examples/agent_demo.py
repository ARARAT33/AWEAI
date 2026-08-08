#!/usr/bin/env python
"""Example: ReAct agent with built-in tools.

Run:  python examples/agent_demo.py
"""

from __future__ import annotations

from aweai.agents.engine import AgentEngine


def main() -> None:
    agent = AgentEngine.create()
    result = agent.run(
        "Use the now tool, then the calculate tool to compute 7*8, "
        "then give a final answer.",
        max_steps=4,
        verbose=True,
    )
    print("Final:", result["final"])
    print("Tool calls:", result["tool_calls"])


if __name__ == "__main__":
    main()
