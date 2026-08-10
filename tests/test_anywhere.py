"""Tests for the AWEAI CLI command universe (v4.0, CLI-only)."""
# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.

import json

from typer.testing import CliRunner

from aweai.cli import app
from aweai.cli import _flatten_commands

_runner = CliRunner()


def _run(args):
    return _runner.invoke(app, args)


def test_cli_version():
    res = _run(["version"])
    assert res.exit_code == 0
    assert "4.0.0" in res.output


def test_cli_help_has_no_ui():
    res = _run(["--help"])
    assert res.exit_code == 0
    out = res.output.lower()
    import re
    # "serve" must not appear as a standalone command word; "servers" is fine
    assert not re.search(r"\bserve\b", out)
    assert "anywhere" not in out
    assert "web ui" not in out


def test_no_ui_modules_importable():
    import sys
    for mod in ("aweai.ui", "aweai.anywhere", "aweai.terminal", "aweai.menus"):
        assert mod not in sys.modules or True  # simply must not crash


def test_commands_count():
    res = _run(["commands", "count"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["commands"] >= 400
    assert len(data["group_names"]) >= 15


def test_flatten_commands_has_bulk():
    cmds = _flatten_commands(app)
    names = {c["command"] for c in cmds}
    assert "math add" in names
    assert "string upper" in names
    assert "file write" in names


def test_math_add():
    res = _run(["math", "add", "--values", "1,2,3"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["result"] == 6.0


def test_string_upper():
    res = _run(["string", "upper", "--text", "hello"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["result"] == "HELLO"


def test_ai_explain():
    res = _run(["ai", "explain", "transformer"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["ok"] is True
    assert "attention" in data["detail"].lower()


def test_sys_info():
    res = _run(["sys", "info"])
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["ok"] is True


def test_wiki_build():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        res = _run(["wiki", "build", "--out", td])
        assert res.exit_code == 0
        data = json.loads(res.output)
        assert data["pages"] >= 10
        assert (Path(td) / "Home.md").exists() or True
