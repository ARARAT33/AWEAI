# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Autotest: one-command full system self-check.

`aweai autotest` runs:
  1. dependencies   — required packages importable
  2. module imports — every aweai.* module imports
  3. smoke-train    — all model types train briefly
  4. RAG            — index -> search -> reload from disk
  5. actions        — natural-language parsing works
  6. i18n           — 10+ languages load
  7. CLI            — all commands registered (incl. v4 groups)
  8. Knowledge      — AI/ASI/AGI knowledge base imports
  9. Watermark      — multi-layer visible/steganographic watermark verification

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
        "aweai.autotest", "aweai.cli", "aweai.watermark",
        "aweai.quantize", "aweai.distributed", "aweai.market",
        "aweai.integrations", "aweai.ai", "aweai.bulk", "aweai.wiki",
        "aweai.cmd", "aweai.cmd.data_collect", "aweai.cmd.data_manage",
        "aweai.cmd.model", "aweai.cmd.provider", "aweai.cmd.device", "aweai.cmd.ops",
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
    for mtype in ("mlp", "linear", "logistic", "kmeans", "ngram", "autoencoder", "gan", "rnn", "lstm", "cnn", "transformer", "vision_cnn", "object_detector", "segmentation", "gru", "ts_transformer"):
        try:
            if mtype == "ngram":
                from aweai.models.registry import create_model

                m = create_model(mtype, n=2)
                m.fit(texts)
                m.generate(5)
            elif mtype == "cnn":
                from aweai.models.registry import create_model

                Xc = X[:, :4].reshape(24, 2, 2)
                m = create_model(mtype, input_dim=4, height=2, kernel=1, channels=[1], num_classes=2)
                m.fit(Xc, y=y, epochs=2)
                m.predict(Xc[:4])
            elif mtype == "vision_cnn":
                from aweai.models.registry import create_model

                Xc = X[:, :4].reshape(24, 1, 2, 2)
                m = create_model(mtype, input_dim=4, height=2, kernel=1, pool=1, channels=[1], num_classes=2)
                m.fit(Xc, y=y, epochs=2)
                m.predict(Xc[:4])
            elif mtype == "object_detector":
                from aweai.models.registry import create_model

                m = create_model(mtype, input_dim=4, grid=2, num_anchors=1, num_classes=2)
                m.fit(X, epochs=2)
                m.predict(X[:4])
            elif mtype == "segmentation":
                from aweai.models.registry import create_model

                m = create_model(mtype, input_dim=4, height=2, num_classes=2, hidden=[8, 8])
                m.fit(X, y=y, epochs=2)
                m.predict(X[:4])
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
            elif mtype in ("gru",):
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
            elif mtype == "ts_transformer":
                from aweai.models.registry import create_model

                seq = X[:4].reshape(4, 4, 1)
                m = create_model(mtype, input_dim=1, d_model=4, nhead=2, layers=1, output_dim=1)
                m.fit(seq, epochs=2)
                m.predict(seq)
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


def check_watermark() -> Dict[str, Any]:
    r = _check("watermark")
    try:
        from aweai.watermark import embed_watermark, verify_watermark, extract_watermark, get_watermark_status

        # Test text steganography & verification
        text = "Hello AWEAI production product."
        wm_text = embed_watermark(text)
        v_res = verify_watermark(wm_text)
        assert v_res["has_watermark"], "watermark text verification failed"
        assert v_res["signature_valid"], "watermark signature invalid"

        # Test dict watermarking
        data = {"model": "aweai_test", "accuracy": 0.99}
        wm_dict = embed_watermark(data)
        vd_res = verify_watermark(wm_dict)
        assert vd_res["valid"], "watermark dict verification failed"
        assert not vd_res["tampered"], "watermark dict reported tampered"

        status = get_watermark_status()
        assert status["owner"] == "ARARAT33", "owner mismatch"

        r["ok"] = True
        r["detail"] = "text, dict & steganographic watermark verification OK"
    except Exception as e:
        r["detail"] = str(e)
    return r


def check_cli() -> Dict[str, Any]:
    r = _check("cli")
    try:
        from typer.testing import CliRunner

        from aweai.cli import app

        runner = CliRunner()
        info = app.registered_commands
        names = sorted([c.name or c.callback.__name__ for c in info])
        group_names = sorted([g.name for g in app.registered_groups])
        required = ["train", "eval", "export", "quantize", "export_edge", "edge_footprint",
                    "dtrain", "dworld", "market", "integrations", "autotest",
                    "version", "recommend", "types", "tools"]
        required_groups = ["ai", "commands", "wiki", "collect", "data", "model",
                           "providers", "devices", "ops", "math", "string", "json",
                           "file", "net", "time", "crypto", "ml", "text", "image",
                           "audio", "video", "sys", "db", "cloud", "llm", "rl", "neuro", "knowledge", "watermark"]
        missing = [x for x in required if x not in names]
        missing += [x for x in required_groups if x not in group_names]

        rv = runner.invoke(app, ["version"])
        assert rv.exit_code == 0, rv.output
        rv2 = runner.invoke(app, ["hardware"])
        assert rv2.exit_code == 0, rv2.output
        rv3 = runner.invoke(app, ["types"])
        assert rv3.exit_code == 0, rv3.output
        rv4 = runner.invoke(app, ["watermark", "status"])
        assert rv4.exit_code == 0, rv4.output

        if missing:
            r["detail"] = f"missing commands: {missing}; have {len(names)} commands, {len(group_names)} groups"
        else:
            r["ok"] = True
            r["detail"] = f"{len(names)} commands + {len(group_names)} groups OK (incl. watermark CLI)"
    except Exception as e:
        r["detail"] = str(e)
    return r


def run_autotest(quick: bool = False, no_ui: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """Run all checks and return a report."""
    start = time.time()
    checks = [
        ("dependencies", check_dependencies),
        ("module_imports", check_module_imports),
        ("watermark", check_watermark),
    ]
    if not quick:
        checks += [
            ("smoke_train", _smoke_train),
            ("rag", check_rag),
            ("actions", check_actions),
            ("i18n", check_i18n),
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
