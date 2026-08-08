"""In-app terminal (v3.0).

A full-featured REPL that exposes ALL of AWEAI's tools through a terminal
interface, plus a browser-friendly API surface used by the UI Terminal tab.

Commands supported (any of the factory's capabilities):

* model factory: train, eval, models, export, import, delete, compare, tune
* data: load, split, augment
* rag: index, ask, stats
* automation: actions, pipeline save/run/list
* quantization: quantize, list-quantized
* edge: export-edge, edge-footprint
* distributed: dtrain, dworld
* market: publish, search, list, info, download, rate, stats
* integrations: list, chat
* menus: allc, autoallc, help, search
* system: version, hardware, recommend, types, langs, config, autotest, serve

The terminal is intentionally dependency-free (no readline requirements)
and works in the browser UI via the ``/api/terminal`` endpoint.
"""

from __future__ import annotations

import json
import shlex
from typing import Any, Dict, List, Optional

from aweai import __version__


def _dispatch(tokens: List[str]) -> Dict[str, Any]:
    if not tokens:
        return {"ok": True, "output": ""}
    cmd = tokens[0].lower()
    args = tokens[1:]

    def _need(n: int, usage: str):
        if len(args) < n:
            raise ValueError(f"Usage: {usage}")

    try:
        if cmd in ("help", "?"):
            return {"ok": True, "output": "AWEAI terminal. Type 'allc' for all commands, 'autoallc' for automations."}
        if cmd == "version":
            return {"ok": True, "output": f"AWEAI v{__version__}"}
        if cmd == "types":
            from aweai.models.registry import list_model_types
            return {"ok": True, "output": "\n".join(list_model_types())}
        if cmd == "models":
            from aweai.management import list_models
            return {"ok": True, "output": json.dumps(list_models(), indent=2)}
        if cmd == "hardware":
            from aweai.hardware import detect
            return {"ok": True, "output": json.dumps(detect().to_dict(), indent=2)}
        if cmd == "recommend":
            from aweai.selector import recommend
            task = args[0] if args else "classification"
            return {"ok": True, "output": json.dumps(recommend(task), indent=2)}
        if cmd == "train":
            _need(1, "train <model_type> <name> [--data PATH] [--epochs N]")
            mtype = args[0]
            name = args[1] if len(args) > 1 else f"term_{mtype}"
            data = _opt(args, "--data")
            epochs = int(_opt(args, "--epochs", "10"))
            from aweai.train import train as train_model
            res = train_model(mtype, name, data_path=data, params={"epochs": epochs})
            return {"ok": True, "output": json.dumps(res, indent=2)}
        if cmd == "eval":
            _need(1, "eval <name> [--data PATH]")
            from aweai.management import load_model
            model, meta = load_model(args[0])
            return {"ok": True, "output": json.dumps({"metrics": meta.get("metrics", {})}, indent=2)}
        if cmd == "export":
            _need(2, "export <name> <fmt>")
            from aweai.management import export_model
            return {"ok": True, "output": json.dumps(export_model(args[0], fmt=args[1]), indent=2)}
        if cmd == "quantize":
            _need(2, "quantize <name> <fmt>")
            from aweai.quantize import quantize_model
            return {"ok": True, "output": json.dumps(quantize_model(args[0], fmt=args[1]), indent=2)}
        if cmd == "export-edge":
            _need(2, "export-edge <name> <fmt> [--quantize FMT]")
            from aweai.export import export_edge
            q = _opt(args, "--quantize")
            return {"ok": True, "output": json.dumps(export_edge(args[0], fmt=args[1], quantize=q), indent=2)}
        if cmd == "edge-footprint":
            _need(1, "edge-footprint <name>")
            from aweai.export import estimate_edge_footprint
            return {"ok": True, "output": json.dumps(estimate_edge_footprint(args[0]), indent=2)}
        if cmd == "dtrain":
            _need(1, "dtrain <model_type> [--name N] [--data PATH] [--workers W] [--epochs E]")
            mtype = args[0]
            name = _opt(args, "--name", f"d_{mtype}")
            data = _opt(args, "--data")
            workers = int(_opt(args, "--workers", "2"))
            epochs = int(_opt(args, "--epochs", "5"))
            from aweai.distributed import train_distributed
            from aweai.data import load_any
            ds = load_any(data) if data else None
            X = ds.X if ds is not None else [[0, 0], [0, 1], [1, 0], [1, 1]]
            y = ds.y if ds is not None else [0, 1, 1, 0]
            res = train_distributed(mtype, name, X, y=y, workers=workers, epochs=epochs)
            return {"ok": True, "output": json.dumps(res, indent=2, default=str)}
        if cmd == "dworld":
            from aweai.distributed import detect_world
            return {"ok": True, "output": json.dumps(detect_world(), indent=2)}
        if cmd in ("market", "mkt"):
            _need(1, "market <publish|search|list|info|download|rate|stats> ...")
            sub = args[0].lower()
            from aweai.market import download, info, list_listings, publish, rate, search, stats
            if sub == "publish":
                _need(2, "market publish <name> [--tag T]")
                return {"ok": True, "output": json.dumps(publish(args[1], tag=_opt(args, "--tag", "v1")), indent=2)}
            if sub == "search":
                _need(2, "market search <query>")
                return {"ok": True, "output": json.dumps(search(args[1]), indent=2)}
            if sub == "list":
                return {"ok": True, "output": json.dumps(list_listings(), indent=2)}
            if sub == "info":
                _need(2, "market info <id>")
                return {"ok": True, "output": json.dumps(info(args[1]), indent=2)}
            if sub == "download":
                _need(2, "market download <id>")
                return {"ok": True, "output": json.dumps(download(args[1]), indent=2)}
            if sub == "rate":
                _need(3, "market rate <id> <stars>")
                return {"ok": True, "output": json.dumps(rate(args[1], float(args[2])), indent=2)}
            if sub == "stats":
                return {"ok": True, "output": json.dumps(stats(), indent=2)}
            raise ValueError(f"Unknown market subcommand: {sub}")
        if cmd == "integrations":
            from aweai.integrations import list_tools
            return {"ok": True, "output": json.dumps(list_tools(), indent=2)}
        if cmd == "chat":
            _need(1, "chat <provider> <message>")
            from aweai.integrations import chat
            return {"ok": True, "output": json.dumps(chat(args[0], " ".join(args[1:])), indent=2)}
        if cmd == "rag":
            _need(1, "rag <index|ask|stats> ...")
            sub = args[0].lower()
            from aweai.rag import RAGEngine
            eng = RAGEngine()
            if sub == "index":
                _need(2, "rag index <path>")
                return {"ok": True, "output": json.dumps(eng.index_directory(args[1]), indent=2)}
            if sub == "ask":
                _need(2, "rag ask <query>")
                return {"ok": True, "output": json.dumps(eng.ask(" ".join(args[1:])), indent=2)}
            if sub == "stats":
                return {"ok": True, "output": json.dumps(eng.stats(), indent=2)}
            raise ValueError(f"Unknown rag subcommand: {sub}")
        if cmd == "actions":
            from aweai.actions import run_action
            return {"ok": True, "output": json.dumps(run_action(" ".join(args)), indent=2, default=str)}
        if cmd == "autotest":
            from aweai.autotest import run_autotest
            report = run_autotest(quick="--quick" in args, no_ui="--no-ui" in args, verbose=False)
            return {"ok": report["all_passed"], "output": json.dumps(report, indent=2)}
        if cmd == "langs":
            from aweai.i18n import language_names
            return {"ok": True, "output": "\n".join(f"{k}: {v}" for k, v in language_names().items())}
        if cmd == "config":
            from aweai.config import get_config
            cfg = get_config()
            return {"ok": True, "output": json.dumps(cfg.all(), indent=2)}
        if cmd == "allc":
            from aweai.menus import build_catalog, render_catalog
            items = build_catalog(min_count=10000)
            q = _opt(args, "--search")
            cat = _opt(args, "--category")
            if q or cat:
                from aweai.menus import search_catalog
                items = search_catalog(items, query=q or "", category=cat or "")
            max_lines = int(_opt(args, "--count", "200"))
            return {"ok": True, "output": render_catalog(items, max_lines=max_lines)}
        if cmd == "autoallc":
            from aweai.menus import build_automations, render_catalog
            items = build_automations(min_count=5000)
            q = _opt(args, "--search")
            cat = _opt(args, "--category")
            if q or cat:
                from aweai.menus import search_catalog
                items = search_catalog(items, query=q or "", category=cat or "")
            max_lines = int(_opt(args, "--count", "200"))
            return {"ok": True, "output": render_catalog(items, max_lines=max_lines)}
        if cmd == "search":
            _need(1, "search <query>")
            from aweai.menus import build_catalog, search_catalog
            items = search_catalog(build_catalog(min_count=10000), query=" ".join(args))
            return {"ok": True, "output": render_catalog(items[:50])}
        if cmd == "serve":
            from aweai.ui import serve
            port = int(_opt(args, "--port", "8888"))
            import threading
            threading.Thread(target=serve, kwargs={"port": port, "open_browser": False}, daemon=True).start()
            return {"ok": True, "output": f"UI server starting on http://127.0.0.1:{port}"}
        raise ValueError(f"Unknown command: {cmd}. Type 'help' or 'allc' for the full catalog.")
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _opt(args: List[str], key: str, default: Optional[str] = None) -> Optional[str]:
    for i, a in enumerate(args):
        if a == key and i + 1 < len(args):
            return args[i + 1]
    return default


def run(line: str) -> Dict[str, Any]:
    """Run a single terminal line, return a structured result."""
    line = (line or "").strip()
    if not line:
        return {"ok": True, "output": ""}
    try:
        tokens = shlex.split(line)
    except Exception as e:
        return {"ok": False, "error": f"Parse error: {e}"}
    return _dispatch(tokens)


def repl(prompt: str = "aweai> ") -> None:
    """Interactive REPL loop (dependency-free input())."""
    print(f"AWEAI terminal v{__version__} — type 'allc' for all commands, 'exit' to quit.")
    while True:
        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() in ("exit", "quit", "q"):
            break
        res = run(line)
        if res.get("output"):
            print(res["output"])
        if not res.get("ok"):
            print(f"Error: {res.get('error', 'unknown error')}")
