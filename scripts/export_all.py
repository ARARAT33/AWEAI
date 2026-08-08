#!/usr/bin/env python3
"""Export every model in the zoo to every format (ONNX, TorchScript, raw, JSON)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aweai.management import list_models, export_model


def main() -> None:
    rows = list_models()
    if not rows:
        print("No models in zoo — train one first (e.g. python examples/train_demo.py)")
        return
    for row in rows:
        name = row["name"]
        for fmt in ("json", "raw", "onnx", "torchscript"):
            try:
                path = export_model(name, fmt=fmt)
                print(f"  ✓ {name} → {fmt}: {path}")
            except Exception as e:
                print(f"  ✗ {name} → {fmt}: {e}")


if __name__ == "__main__":
    main()
