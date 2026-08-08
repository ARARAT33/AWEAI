"""Example: run the ReAct agent with built-in tools."""

from aweai.agents.engine import AgentEngine

agent = AgentEngine.create()
result = agent.run(
    "List the files in /tmp and calculate 15 * 4",
    max_steps=3,
)
print("Final:", result["final"])
print("Tool calls:", result["tool_calls"])
