# AWEAI

**AI-Worker Engine for Agents & Intelligence** — a lightweight, modular
Python toolkit for building AI-powered assistants. It provides configuration
handling, an LLM client abstraction, a tool registry, SQLite-backed memory,
and a simple agent runner — plus a CLI and examples.

> Created by [ARARAT33](https://github.com/ARARAT33). Works offline out of
> the box (via a built-in echo client) and can connect to any
> OpenAI-compatible API when a key is provided.

## ✨ Features

- **🧩 Modular architecture** — `config`, `llm`, `tools`, `memory`, `agent`
  and `cli` are independent, importable modules.
- **🤖 LLM client abstraction** — swap OpenAI, Groq, Together, vLLM/Ollama
  and more without changing agent code; `EchoClient` lets you run offline.
- **🔧 Tool registry** — register async functions as tools with automatic
  JSON-schema generation, argument validation and dispatch.
- **💾 SQLite memory** — persistent conversation history and key/value
  metadata with keyword search (zero external services).
- **⚡ Agent runner** — a minimal function-calling loop with tool execution,
  error surfacing, and a step budget.
- **🖥️ CLI** — `chat`, `demo`, `config`, `stats` subcommands.

## 📦 Installation

```bash
# From the repository root (editable install for development)
pip install -e ".[dev]"

# Or just run without installing:
python -m aweai --help
```

Requires Python **3.10+**. The only runtime dependency is `httpx`.

## 🚀 Quick start

### Offline (no API key)

```bash
python -m aweai chat -m "Hello from AWEAI!"
# Hello from AWEAI!
```

The default `EchoClient` echoes the user message back, so every command works
without any configuration.

### With an OpenAI-compatible API

```bash
export AWEAI_API_KEY=sk-...
export AWEAI_MODEL=gpt-4o-mini        # optional
export AWEAI_BASE_URL=https://api.openai.com/v1   # optional

python -m aweai chat          # interactive REPL
python -m aweai chat -m "What can you do?"
```

### Demos

```bash
python -m aweai demo tools    # tool registry demo
python -m aweai demo memory   # memory store demo
python -m aweai config        # show resolved configuration
python -m aweai stats         # show database statistics
```

## 🐍 Library usage

```python
import asyncio
from aweai.agent import Agent

async def main():
    agent = Agent()                     # EchoClient, SQLite memory
    try:
        reply = await agent.chat("Hello!")
        print(reply)
    finally:
        await agent.aclose()

asyncio.run(main())
```

### Custom tools

```python
from aweai.tools import ToolRegistry, tool

@tool
async def greet(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}!"

registry = ToolRegistry()
registry.add(greet)
```

### Memory

```python
from aweai.memory import MemoryStore

store = MemoryStore("aweai.db")
store.add_message("user", "Remember this.")
store.set("theme", "dark")
print(store.get("theme"))          # dark
print(store.search("Remember"))    # keyword search over messages
store.close()
```

## 🧪 Running tests

```bash
pip install -e ".[dev]"
pytest -q
```

## 🗂️ Project layout

```
aweai/
├── aweai/
│   ├── __init__.py     # public API
│   ├── __main__.py     # python -m aweai
│   ├── config.py       # AWEConfig (env / file / defaults)
│   ├── llm.py          # LLMClient, EchoClient, OpenAICompatClient
│   ├── tools.py        # ToolRegistry, @tool, built-in tools
│   ├── memory.py       # MemoryStore (SQLite)
│   ├── agent.py        # Agent runner with function-calling loop
│   └── cli.py          # command-line interface
├── examples/           # chat.py, tools_demo.py, memory_demo.py
├── tests/              # pytest suite (30 tests)
├── pyproject.toml
└── README.md
```

## 🔌 Environment variables

| Variable           | Default          | Description                          |
| ------------------ | ---------------- | ------------------------------------ |
| `AWEAI_API_KEY`    | *(none)*         | API key for the LLM provider         |
| `AWEAI_MODEL`      | `gpt-4o-mini`    | Model identifier                     |
| `AWEAI_BASE_URL`   | *(none)*         | Custom OpenAI-compatible endpoint    |
| `AWEAI_DB_PATH`    | `aweai.db`       | SQLite memory database path          |

Configuration can also be loaded from a JSON file:

```bash
python -c "from aweai.config import AWEConfig; c = AWEConfig.from_file('config.json'); print(c.to_dict())"
```

## 🛣️ Roadmap

- [x] Core modules: config, llm, tools, memory, agent
- [x] CLI and examples
- [x] Test suite and packaging
- [ ] Streaming + real function-calling protocol support
- [ ] Plugins / community tool packs
- [ ] Embedding-based semantic memory search
- [ ] GitHub Actions CI

## 📄 License

MIT
