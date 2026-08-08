#!/usr/bin/env python
"""Example: train a new model from scratch (torch or n-gram fallback).

Run:  python examples/train_demo.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from aweai.models.trainer import train_scratch


def main() -> None:
    # Build a tiny sample dataset
    data = Path(tempfile.mkdtemp(prefix="aweai_demo_")) / "data.jsonl"
    lines = [
        {"text": "AWEAI is the universal AI toolbox."},
        {"text": "Բարեւ աշխարհ։ AWEAI-ը համընդհանուր AI գործիք է։"},
        {"text": "Yerevan is the capital of Armenia."},
        {"text": "RAG stands for retrieval-augmented generation."},
        {"text": "LoRA is a lightweight fine-tuning method."},
        {"text": "Agents can use tools to complete tasks."},
        {"text": "The UI supports 12 languages."},
    ]
    data.write_text(
        "\n".join(json.dumps(l, ensure_ascii=False) for l in lines), encoding="utf-8"
    )

    print("Training a new model from scratch…")
    result = train_scratch("demo_model", str(data), epochs=2)
    print(f"Done in {result.duration_s:.1f}s -> {result.path}")
    for m in result.messages:
        print("  •", m)


if __name__ == "__main__":
    main()
