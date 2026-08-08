#!/usr/bin/env python
"""Example: natural-language automation studio.

Run:  python examples/actions_demo.py "hardware"
"""

from __future__ import annotations

import json
import sys

from aweai.actions.runner import ActionsRunner


def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else "hardware"
    runner = ActionsRunner(verbose=True)
    result = runner.run(text)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
