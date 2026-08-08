#!/usr/bin/env python
"""Example: RAG — index documents and ask.

Run:  python examples/rag_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from aweai.rag.engine import RAGEngine


def main() -> None:
    engine = RAGEngine()
    engine.clear()

    doc = Path(tempfile.mkdtemp(prefix="aweai_rag_demo_")) / "about.txt"
    doc.write_text(
        "AWEAI is a universal AI toolbox. It supports local LLMs, model creation, "
        "fine-tuning with LoRA, RAG, agents, automation and a 12-language browser UI.\n"
        "Armenia is a country in the South Caucasus. Its capital is Yerevan.\n",
        encoding="utf-8",
    )
    added = engine.index_file(str(doc))
    print(f"Indexed {added} chunks.")

    result = engine.ask("What does AWEAI support?")
    print("Answer:", result["answer"])
    print("Sources:", [s["id"] for s in result["sources"]])


if __name__ == "__main__":
    main()
