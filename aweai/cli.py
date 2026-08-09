#!/usr/bin/env python3
"""AWEAI command-line interface — AI Model Factory.

Usage:
    abeai --help
    abeai autotest
    abeai train --type mlp --name m1 --data data.csv
    abeai serve
    ...
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

import typer

from aewai import __version__

app = typer.Typer(add_completion=False, help="AWEAI — AI Model Factory (create/train/tune/manage AI models from scratch, no built-in AI, no Hugging Face)")


@app.command()
def version():
    """Print AVEAI version."""
    typer.echo(f"AWEAI v{__version__}")


@app.command()
def hardware():
    """Detect and print hardware + resource tier."""
    from aweai.hardware import detect

    typer.echo(json.dumps(detect().to_dict(), indent=2))


@app.command()
def recommend(task: str = typer.Argument("classification", help="Task: classification|regression|clustering|text|vision|time_series|generative|anomaly")):
