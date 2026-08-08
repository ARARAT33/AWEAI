"""Command-line interface for AWEAI.

Subcommands:

* ``aweai chat`` — interactive chat with the agent (or one-shot with ``-m``)
* ``aweai demo tools`` — run the built-in tool registry demo
* ``aweai demo memory`` — exercise the SQLite memory store
* ``aweai config`` — show resolved configuration
* ``aweai stats`` — show memory database statistics

Examples:

    export AWEAI_API_KEY=sk-...
    aweai chat --model gpt-4o-mini
    aweai chat -m "What can you do?"
    aweai demo tools
    aweai stats --db ./data/memory.db
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Optional

from aweai import __version__
from aweai.config import AWEConfig
from aweai.llm import make_client
from aweai.memory import MemoryStore
from aweai.tools import default_registry

from aweai.agent import Agent


# ----------------------------------------------------------------------
# Chat
# ----------------------------------------------------------------------


async def _chat(args: argparse.Namespace) -> int:
    config = AWEConfig(model=args.model, db_path=args.db)
    agent = Agent(config=config, system_prompt=args.system_prompt)
    try:
        if args.message:
            reply = await agent.chat(args.message)
            print(reply)
            return 0

        print("AWEAI chat — type 'exit' or Ctrl-D to quit.")
        while True:
            try:
                user = input("you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            user = user.strip()
            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break
            reply = await agent.chat(user)
            print(f"aweai> {reply}")
        return 0
    finally:
        await agent.aclose()


# ----------------------------------------------------------------------
# Demos
# ----------------------------------------------------------------------


async def _demo_tools(args: argparse.Namespace) -> int:
    registry = default_registry()
    print(f"Registered tools ({len(registry)}):")
    for schema in registry.schemas():
        fn = schema["function"]
        print(f"  - {fn['name']}: {fn['description']}")

    print("\nRunning demo calls:")
    print("  add(2, 3)       ->", await registry.call("add", {"a": 2, "b": 3}))
    print("  multiply(4, 5)  ->", await registry.call("multiply", {"a": 4, "b": 5}))
    print("  now_utc()       ->", await registry.call("now_utc", {}))
    return 0


async def _demo_memory(args: argparse.Namespace) -> int:
    store = MemoryStore(args.db)
    try:
        store.add_message("user", "Hello AWEAI!")
        store.add_message("assistant", "Hello! How can I help?")
        store.set("user_name", "Ararat")
        store.set("prefs", {"language": "hy", "theme": "dark"})

        print("Messages in 'default' session:")
        for row in store.get_messages():
            print(f"  [{row['role']:9}] {row['content']}")
        print("\nKV store:")
        print("  user_name =", store.get("user_name"))
        print("  prefs     =", store.get("prefs"))
        print("\nSearch 'Hello':")
        for row in store.search("Hello"):
            print(f"  [{row['role']}] {row['content']}")
        print("\nStats:", store.stats())
        return 0
    finally:
        store.close()


# ----------------------------------------------------------------------
# Config / stats
# ----------------------------------------------------------------------


async def _config(args: argparse.Namespace) -> int:
    config = AWEConfig()
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    return 0


async def _stats(args: argparse.Namespace) -> int:
    store = MemoryStore(args.db)
    try:
        print(json.dumps(store.stats(), indent=2, ensure_ascii=False))
    finally:
        store.close()
    return 0


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aweai",
        description="AWEAI — AI-Worker Engine for Agents & Intelligence",
    )
    parser.add_argument("--version", action="version", version=f"aweai {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="chat with the agent")
    chat.add_argument("-m", "--message", help="one-shot message instead of REPL")
    chat.add_argument("--model", default=None, help="model identifier")
    chat.add_argument("--db", default="aweai.db", help="memory database path")
    chat.add_argument("--system-prompt", default=None, help="override system prompt")
    chat.set_defaults(func=_chat)

    demo = sub.add_parser("demo", help="run a demo")
    demo_sub = demo.add_subparsers(dest="demo", required=True)
    demo_tools = demo_sub.add_parser("tools", help="tool registry demo")
    demo_tools.set_defaults(func=_demo_tools)
    demo_memory = demo_sub.add_parser("memory", help="memory store demo")
    demo_memory.add_argument("--db", default="aweai.db", help="memory database path")
    demo_memory.set_defaults(func=_demo_memory)

    config = sub.add_parser("config", help="show resolved configuration")
    config.set_defaults(func=_config)

    stats = sub.add_parser("stats", help="show memory database statistics")
    stats.add_argument("--db", default="aweai.db", help="memory database path")
    stats.set_defaults(func=_stats)

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(args.func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
