# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Standalone entry point used by PyInstaller packaging.

Builds a single binary that behaves like the `aweai` console script:
    aweai <command> [options]

When run with no arguments it prints the CLI help (same as `aweai --help`).
AWEAI is CLI-only — there is no UI entry point.
"""

from __future__ import annotations

from aweai.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
