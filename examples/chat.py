"""Example: one-shot chat with the agent (works offline via EchoClient).

Run:
    python examples/chat.py "Hello from AWEAI!"
"""

import asyncio
import sys

from aweai.agent import Agent


async def main() -> None:
    message = sys.argv[1] if len(sys.argv) > 1 else "Hello from AWEAI!"
    agent = Agent()
    try:
        reply = await agent.chat(message)
        print(f"you>   {message}")
        print(f"aweai> {reply}")
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
