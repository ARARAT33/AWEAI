"""AWEAI — AI-Worker Engine for Agents & Intelligence.

Modular Python toolkit for building AI-powered assistants.

Subpackages/modules:

* ``aweai.config`` — configuration (env / file / defaults)
* ``aweai.llm``    — LLM client abstraction (Echo / OpenAI-compatible)
* ``aweai.tools``  — tool registry and built-in tools
* ``aweai.memory`` — SQLite-backed memory store
* ``aweai.agent``  — agent runner with a function-calling loop
* ``aweai.cli``    — command-line interface
"""

__version__ = "0.1.0"

from aweai.cli import main as _main

if __name__ == "__main__":
    import sys

    sys.exit(_main())
