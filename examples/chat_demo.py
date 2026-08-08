#!/usr/bin/env python
"""Example: terminal chat with AWEAI.

Run:  python examples/chat_demo.py ["your question"]
"""

from __future__ import annotations

import sys

from aweai.models.inference import LLM


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "What is Yerevan?"
    llm = LLM()
    print(f"[model] {llm.model_id or 'auto'}")
    print(f"Q: {question}")
    print(f"A: {llm.chat([{'role': 'user', 'content': question}])}")


if __name__ == "__main__":
    main()
