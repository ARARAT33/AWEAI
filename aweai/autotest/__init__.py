"""Autotest: one-command full system self-check.

`aweai autotest` (or the UI Autotest button) runs:
  1. dependencies   — required packages importable
  2. module imports — every aweai.* module imports
  3. smoke-train    — all model types train briefly
  4. RAG            — index -> search -> reload from disk
  5. actions        — natural-language parsing works
  6. i18n           — 10+ languages load
  7. UI             — server boots, /api/health responds
  8. CLI            — all commands registered

Returns a report dict with per-step status, and prints a summary.
"""

from __future__ import annotations

import importlib
import json
import time
from typing import Any, Dict, List, Optional

import numpy as np

from aweai import __version__


def _check(step: str) -> Any:
    return {"step": step, "ok": False, "detail": ""}


def check_dependencies() -> Dict[str, Any]:
    r = _check("dependencies")
    missing = []
    for mod in ("numpy",):
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if missing:
        r["detail"] = f"missing: {missing}"
    else:
        r["ok"] = True
        r["detail"] = "numpy OK"
    return r


def check_module_imports() -> Dict[str, Any]:
    r = _check("module_imports")
    modules = [
        "aweai.config", "aweai.errors", "aweai.ports", "aweai.utils",
        "aweai.hardware", "aweai.selector", "aweai.i18n",
        "aweai.data", "aweai.models", "aweai.train", "aweai.eval",
        "aweai.management", "aweai.export", "aweai.rag", "aweai.actions",
        "aweai.autotest", "aweai.ui", "aweai.cli",
    ]
    failed = []
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as e:
            failed.append(f"{m}: {e}")
    if failed:
        r["detail"] = "; ".join(failed)
    else:
        r["ok"] = True
        r["detail"] = f"{len(modules)} modules OK"
    return r


def _smoke_train() -> Dict[str, Any]:
    r = _check("smoke_train")
    X = np.random.rand(24, 4).astype(float)
    y = (X[:, 0] > 0.5).astype(int)
    texts = ["the quick brown fox jumps over the lazy dog", "hello world from the model factory", "numpy from scratch no hugging face"]
    results = {}
    failed = []
    for mtype in ("mlp", "linear", "logistic", "kmeans", "ngram", "autoencoder", "gan", "rnn", "lstm", "cnn", "transformer"):
        try:
            if mtype == "ngram":
                from aweai.models.registry import create_model

                m = create_model(mtype, n=2)
                m.fit(texts)
                m.generate(5)
            elif mtype == "cnn":
                from aweai.models.registry import create_model

                Xc = X[:, :16]
                m = create_model(mtype, input_dim=16, height=4, channels=[2, 4], num_classes=2)
                m.fit(Xc, y=y, epochs=2)
                m.predict(Xc[:4])
            elif mtype in ("mlp",):
                from aweai.models.registry import create_model

                m = create_model(mtype, input_dim=4, hidden=[6, 4], output_dim=2)
                m.fit(X, y=y, epochs=3)
                m.predict(X[:4])
            elif mtype in ("linear", "logistic"):
                from aweai.models.registry import create_model

                m = create_model(mtype, input_dim=4)
                m.fit(X, y=y if mtype == "logistic" else X[:, 0], epochs=2)
                m.predict(X[:4])
            elif mtype == "kmeans":
                from aweai.models.registry import create_model

                m = create_model(mtype, k=2)
                m.fit(X)
                m.predict(X[:4])
            elif mtype in ("rnn", "lstm"):
                from aweai.models.registry import create_model

                seq = X[:4].reshape(4, 4, 1)
                m = create_model(mtype, input_dim=1, hidden=4, output_dim=1)
                m.fit(seq, epochs=2)
                m.predict(seq)
            elif mtype == "transformer":
                from aweai.models.registry import create_model

                Xt = np.random.randint(0, 8, (4, 6))
                yt = np.random.randint(0, 2, (4,))
                m = create_model(mtype, vocab_size=10, d_model=8, nhead=2, layers=1, num_classes=2)
                m.fit(Xt, y=yt, epochs=2)
                m.predict(Xt)
            elif mtype == "autoencoder":
                from aweai.models.registry import create_model

                m = create_model(mtype, input_dim=4, hidden=[4, 2])
                m.fit(X, epochs=2)
                m.reconstruct(X[:4])
            elif mtype == "gan":
                from aweai.models.registry import create_model

                m = create_model(mtype, input_dim=4, latent=2, hidden=[4, 4])
                m.fit(X, epochs=2, batch_size=8)
                m.generate(3)
            results[mtype] = "ok"
        except Exception as e:
            failed.append(f"{mtype}: {e}")
            results[mtype] = str(e)
    if failed:
        r["detail"] = "; ".join(failed)
    else:
        r["ok"] = True
        r["detail"] = "all model types smoke-trained"
    r["results"] = results
    return r


def check_rag() -> Dict[str, Any]:
    r = _check("rag")
    try:
        from tempfile import TemporaryDirectory

        from aweai.rag import RAGEngine

        with TemporaryDirectory() as td:
            eng = RAGEngine(index_path=f"{td}/index.json")
            eng.index_documents(["AWEAI is a model factory for creating AI models from scratch.",
                                 "The quick brown fox jumps over the lazy dog.",
                                 "Numpy provides fast array operations for machine learning."])
            hits = eng.search("model factory")
            assert hits, "no hits"
            # reload from disk (the index_file shadowing fix)
            eng2 = RAGEngine(index_path=f"{td}/index.json")
            stats = eng2.stats()
            assert stats["chunks"] >= 3, "reload lost chunks"
        r["ok"] = True
        r["detail"] = "index->search->reload OK"
    except Exception as e:
        r["detail"] = str(e)
    return r


def check_actions() -> Dict[str, Any]:
    r = _check("actions")
    try:
        from aweai.actions import parse_action

        a = parse_action("train an mlp model named demo1")
        assert a["action"] == "train" and a["kwargs"].get("model_type") == "mlp", a
        b = parse_action("list all models")
        assert b["action"] == "list", b
        r["ok"] = True
        r["detail"] = "natural-language parsing OK"
    except Exception as e:
        r["detail"] = str(e)
    return r


def check_i18n() -> Dict[str, Any]:
    r = _check("i18n")
    try:
        from aweai.i18n import LANGUAGES, t

        langs = list(LANGUAGES.keys())
        if len(langs) < 10:
            raise RuntimeError(f"Only {len(langs)} languages")
        val = t("common.dashboard", lang="hy")
        assert val, "empty translation"
        r["ok"] = True
        r["detail"] = f"{len(langs)} languages OK (hy: {val})"
    except Exception as e:
        r["detail"] = str(e)
    return r


def check_ui(no_ui: bool = False) -> Dict[str, Any]:
    r = _check("ui")
    if no_ui:
        r["ok"] = True
        r["detail"] = "skipped (--no-ui)"
        return r
    try:
        from aweai.ports import resolve_port
        from aweai.ui import create_app
        from fastapi.testclient import TestClient

        port = resolve_port(8999)
        app = create_app()
        client = TestClient(app)
        resp = client.get("/api/health")
        assert resp.status_code == 200, resp.status_code
        data = resp.json()
        assert "status" in data and data["status"] == "ok"
        r["ok"] = True
        r["detail"] = "/api/health OK"
    except Exception as e:
        r["detail"] = str(e)
    return r


def check_cli() -> Dict[str, Any]:
    r = _check("cli")
    try:
        from aweai.cli import app

        names = []
        for c in app.registered_commands:
            n = getattr(c, "name", None)
            if n:
                names.append(n)
            elif getattr(c, "callback", None) is not None:
                # Newer typer: name lives on the callback function.
                names.append(c.callback.__name__)
        required = {"train", "eval", "models", "export", "data", "rag", "actions", "serve", "autotest"}
        missing = required - set(names)
        if missing:
            raise RuntimeError(f"missing commands: {missing}")
        r["ok"] = True
        r["detail"] = f"{len(names)} commands OK"
    except Exception as e:
        r["detail"] = str(e)
    return r


def run_autotest(quick: bool = False, no_ui: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """Run all checks and return a report."""
    start = time.time()
    checks = [
        ("dependencies", check_dependencies),
        ("module_imports", check_module_imports),
    ]
    if not quick:
        checks += [
            ("smoke_train", _smoke_train),
            ("rag", check_rag),
            ("actions", check_actions),
            ("i18n", check_i18n),
            ("ui", lambda: check_ui(no_ui=no_ui)),
            ("cli", check_cli),
        ]
    else:
        checks += [("quick_cli", check_cli)]
    results = []
    for name, fn in checks:
        try:
            results.append(fn())
        except Exception as e:
            results.append({"step": name, "ok": False, "detail": str(e)})
    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    report = {
        "version": __version__,
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
        "duration_s": round(time.time() - start, 2),
        "results": results,
    }
    if verbose:
        print(f"AWEAI autotest v{__version__} — {passed}/{total} passed ({report['duration_s']}s)")
        for r in results:
            mark = "✓" if r["ok"] else "✗"
            print(f"  {mark} {r['step']}: {r.get('detail', '')}")
    return report
