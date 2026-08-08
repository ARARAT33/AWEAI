#!/usr/bin/env python3
"""Guard: ensure no UNCONDITIONAL Hugging Face imports in the codebase.

AWEAI's policy is "no Hugging Face dependency" for the core runtime.
Optional integrations (e.g. `transformers` inside a function or try/except)
are allowed and do NOT count — only imports at module top-level (directly
under `ast.Module`, including inside a top-level try/except) fail.

Exit code 0 = clean, 1 = violations found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

BANNED = {"transformers", "datasets", "sentence_transformers", "peft", "huggingface_hub"}


def _build_parents(tree: ast.Module):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _is_lazy(node: ast.AST, parents) -> bool:
    """True if the import lives inside a function/method (lazy import)."""
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        cur = parents.get(cur)
    return False


def main() -> int:
    hits = []
    for p in sorted(Path(".").rglob("*.py")):
        if ".git" in p.parts:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        parents = _build_parents(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in BANNED and not _is_lazy(node, parents):
                        hits.append(f"{p}:{node.lineno}: top-level import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in BANNED and not _is_lazy(node, parents):
                    hits.append(f"{p}:{node.lineno}: top-level from {node.module} import ...")

    if hits:
        print("Unconditional (top-level) Hugging Face imports found:")
        for h in hits:
            print("  ", h)
        print("(lazy imports inside functions / methods are allowed: optional integrations)")
        return 1
    print("OK: no top-level Hugging Face imports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
