# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""AWEAI command-line interface — the universal AI/ASI/AGI engineering CLI.

Usage:
    aweai --help
    aweai train --type mlp --name m1 --data data.csv
    aweai data-collect synthetic --rows 100
    aweai model quantize m1 --fmt int8
    aweai ai explain transformer
    aweai commands list
    aweai wiki build
    aweai autotest

The CLI is pure terminal (Typer): no web UI, no GUI, no server.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

import typer

from aweai import __version__

app = typer.Typer(add_completion=False, help="AWEAI — universal CLI for AI/ASI/AGI engineering (pure terminal, no UI)")


# ---------------------------------------------------------------------------
# Core commands (model factory + system)
# ---------------------------------------------------------------------------
@app.command()
def version():
    """Print AWEAI version."""
    typer.echo(f"AWEAI v{__version__}")


@app.command()
def hardware():
    """Detect and print hardware + resource tier."""
    from aweai.hardware import detect

    typer.echo(json.dumps(detect().to_dict(), indent=2))


@app.command()
def recommend(task: str = typer.Argument("classification", help="Task: classification|regression|clustering|text|vision|time_series|generative|anomaly")):
    """Resource-adaptive recommendation: best model type for THIS machine."""
    from aweai.selector import recommend

    typer.echo(json.dumps(recommend(task), indent=2))


@app.command()
def types():
    """List available from-scratch model types."""
    from aweai.models.registry import MODEL_TYPES

    for name, info in MODEL_TYPES.items():
        typer.echo(f"{name:<14} {info['task']:<14} {info['desc']}")


@app.command()
def train(
    model_type: str = typer.Option("mlp", "--type", "-t", help="Model type"),
    name: str = typer.Option(..., "--name", "-n", help="Model name"),
    data_path: Optional[str] = typer.Option(None, "--data", "-d", help="CSV/JSON/JSONL/text file"),
    target: Optional[str] = typer.Option(None, "--target", help="Target column (CSV)"),
    text_path: Optional[str] = typer.Option(None, "--text", help="Text file for ngram/rnn/lstm"),
    params: Optional[str] = typer.Option(None, "--params", help="JSON hyperparameters, e.g. '{\"epochs\": 10}'"),
    epochs: int = typer.Option(None, "--epochs", "-e", help="Override epochs"),
    normalize: Optional[str] = typer.Option(None, "--normalize", help="standardize|minmax"),
):
    """Train a model from scratch and save it into the model zoo."""
    from aweai.train import train as train_model

    p = json.loads(params) if params else {}
    if epochs:
        p["epochs"] = epochs
    try:
        res = train_model(model_type, name, data_path=data_path, text_path=text_path,
                          target=target, params=p, normalize=normalize)
        typer.echo(json.dumps(res, indent=2))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def continue_train(
    name: str = typer.Argument(..., help="Existing model name"),
    data_path: Optional[str] = typer.Option(None, "--data", "-d"),
    epochs: int = typer.Option(10, "--epochs", "-e"),
):
    """Continue/fine-tune an existing model on new data."""
    from aweai.train import continue_training

    try:
        res = continue_training(name, data_path=data_path, epochs=epochs)
        typer.echo(json.dumps(res, indent=2))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def eval(
    name: str = typer.Argument(..., help="Model name"),
    data_path: Optional[str] = typer.Option(None, "--data", "-d", help="Evaluation CSV/JSON"),
    target: Optional[str] = typer.Option(None, "--target", help="Target column"),
):
    """Evaluate a model on data (accuracy/precision/recall/F1 + confusion matrix)."""
    from aweai.data import load_any
    from aweai.eval import classification_report
    from aweai.management import load_model

    try:
        model, meta = load_model(name)
        if data_path:
            ds = load_any(data_path, target_column=target or None)
            pred = model.predict(ds.X if ds.X is not None else ds.texts)
            if ds.y is not None:
                report = classification_report(ds.y, pred)
            else:
                report = {"pred": pred.tolist()}
        else:
            report = {"metrics": meta.get("metrics", {}), "history": meta.get("history", {})}
        typer.echo(json.dumps(report, indent=2))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def models():
    """List all models in the zoo."""
    from aweai.management import list_models

    rows = list_models()
    typer.echo(json.dumps(rows, indent=2))


@app.command()
def export(
    name: str = typer.Argument(..., help="Model name"),
    fmt: str = typer.Option("json", "--fmt", "-f", help="json|raw|onnx|torchscript"),
):
    """Export a model to json/raw/onnx/torchscript."""
    from aweai.management import export_model

    try:
        typer.echo(json.dumps(export_model(name, fmt=fmt), indent=2))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def import_model(
    file: str = typer.Argument(..., help="Path to model.json"),
    name: Optional[str] = typer.Option(None, "--name", "-n"),
):
    """Import a model from a JSON export."""
    from aweai.management import import_model as _import

    try:
        typer.echo(json.dumps(_import(file, name=name), indent=2))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def delete(
    name: str = typer.Argument(..., help="Model name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete a model from the zoo."""
    from aweai.management import delete_model

    if not yes:
        confirm = typer.confirm(f"Delete model '{name}'?")
        if not confirm:
            typer.echo("Aborted")
            raise typer.Exit(code=1)
    typer.echo(json.dumps({"ok": True, "deleted": delete_model(name)}))


@app.command()
def compare(names: List[str] = typer.Argument(..., help="Model names to compare")):
    """Compare models side by side."""
    from aweai.management import compare_models

    typer.echo(json.dumps(compare_models(names), indent=2))


@app.command()
def tune(
    model_type: str = typer.Argument(..., help="Model type"),
    data_path: str = typer.Option(..., "--data", "-d"),
    target: Optional[str] = typer.Option(None, "--target"),
    method: str = typer.Option("grid", "--method", "-m", help="grid|random"),
):
    """Hyperparameter search (grid/random) on a model type."""
    from aweai.data import load_any
    from aweai.train import tune as tune_model

    try:
        ds = load_any(data_path, target_column=target or None)
        best = tune_model(model_type, ds.X, y=ds.y, method=method)
        typer.echo(json.dumps(best, indent=2))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def data(
    action: str = typer.Argument(..., help="load|split|augment"),
    path: Optional[str] = typer.Option(None, "--path", "-p"),
    target: Optional[str] = typer.Option(None, "--target"),
    ratio: float = typer.Option(0.8, "--ratio", "-r"),
):
    """Data tools: load a dataset, show info, split, augment."""
    from aweai.data import load_any, train_test_split, augment

    try:
        if action == "load":
            ds = load_any(path, target_column=target or None) if path else None
            typer.echo(json.dumps(ds.to_dict() if ds else {"error": "no path"}, indent=2))
        elif action == "split":
            ds= load_any(path, target_column=target or None)
            parts = train_test_split(ds.X, ds.y, ratio=ratio)
            typer.echo(json.dumps({k: (v.shape if hasattr(v, "shape") else len(v)) for k, v in parts.items()}, indent=2))
        elif action == "augment":
            ds= load_any(path, target_column=target or None)
            out = augment(X=ds.X, texts=ds.texts)
            typer.echo(json.dumps({k: (v.shape if hasattr(v, "shape") else len(v)) for k, v in out.items()}, indent=2))
        else:
            typer.echo(f"Unknown action: {action}", err=True)
            raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def rag(
    action: str = typer.Argument(..., help="index|ask|stats"),
    path: Optional[str] = typer.Option(None, "--path", "-p", help="Directory for index"),
    query: Optional[str] = typer.Option(None, "--query", "-q"),
):
    """RAG tools: index documents, ask questions."""
    from aweai.rag import RAGEngine

    try:
        eng = RAGEngine()
        if action == "index":
            typer.echo(json.dumps(eng.index_directory(path) if path else {"error": "need --path"}, indent=2))
        elif action == "ask":
            typer.echo(json.dumps(eng.ask(query or ""), indent=2))
        elif action == "stats":
            typer.echo(json.dumps(eng.stats(), indent=2))
        else:
            typer.echo(f"Unknown action: {action}", err=True)
            raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def actions(text: str = typer.Argument(..., help="Natural-language instruction")):
    """Run a natural-language automation action."""
    from aweai.actions import run_action

    try:
        typer.echo(json.dumps(run_action(text), indent=2, default=str))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def pipeline(
    action: str = typer.Argument(..., help="save|run|list"),
    name: Optional[str] = typer.Option(None, "--name", "-n"),
    steps: Optional[str] = typer.Option(None, "--steps", help="JSON list of steps"),
):
    """Automation pipelines: save/run/list."""
    from aweai.actions import save_pipeline, list_pipelines, run_pipeline

    try:
        if action == "save":
            typer.echo(json.dumps(save_pipeline(name or "p1", json.loads(steps or "[]")), indent=2))
        elif action == "list":
            typer.echo(json.dumps(list_pipelines(), indent=2))
        elif action == "run":
            typer.echo(json.dumps(run_pipeline(name or "p1"), indent=2, default=str))
        else:
            typer.echo(f"Unknown action: {action}", err=True)
            raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def autotest(
    quick: bool = typer.Option(False, "--quick", help="Skip smoke-train/RAG/i18n"),
    no_ui: bool = typer.Option(False, "--no-ui", help="(accepted for compatibility; no UI exists)"),
):
    """Run the full system autotest (deps, imports, smoke-train all model types, RAG, actions, i18n, CLI)."""
    from aweai.autotest import run_autotest

    report = run_autotest(quick=quick, no_ui=no_ui)
    if not report[ "all_passed"]:
        raise typer.Exit(code=1)


@app.command()
def langs():
    """List supported languages."""
    from aweai.i18n import language_names

    for code, name in language_names().items():
        typer.echo(f"{code:<4} {name}")


@app.command()
def config(
    action: str = typer.Argument(..., help="get|set|show"),
    key: Optional[str] = typer.Option(None, "--key", "-k"),
    value: Optional[str] = typer.Option(None, "--value", "-v"),
):
    """Get/set/show configuration (e.g. language)."""
    from aweai.config import get_config

    cfg = get_config()
    if action == "show":
        typer.echo(json.dumps(cfg.all(), indent=2))
    elif action == "get":
        typer.echo(str(cfg.get(key, "")))
    elif action == "set":
        cfg.set(key, value)
        typer.echo(f"{key}={value}")
    else:
        typer.echo(f"Unknown action: {action}", err=True)
        raise typer.Exit(code=1)


@app.command()
def quantize(
    name: str = typer.Argument(..., help="Model name"),
    fmt: str = typer.Option("int8", "--fmt", "-f", help="float16|int8|uint8|int4"),
):
    """Quantize a model (float16/int8/uint8/int4)."""
    from aweai.quantize import quantize_model

    try:
        typer.echo(json.dumps(quantize_model(name, fmt=fmt), indent=2))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def export_edge(
    name: str = typer.Argument(..., help="Model name"),
    fmt: str = typer.Option("onnx", "--fmt", "-f", help="onnx|tflite|torchscript|edge_json"),
    quantize_fmt: Optional[str] = typer.Option(None, "--quantize", help="float16|int8|uint8|int4"),
):
    """Export a model to an edge format (ONNX/TFLite/TorchScript)."""
    from aweai.export import export_edge as _export_edge

    try:
        typer.echo(json.dumps(_export_edge(name, fmt=fmt, quantize=quantize_fmt), indent=2))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def edge_footprint(name: str = typer.Argument(..., help="Model name")):
    """Estimate on-device footprint (fp32/fp16/int8)."""
    from aweai.export import estimate_edge_footprint

    try:
        typer.echo(json.dumps(estimate_edge_footprint(name), indent=2))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def dtrain(
    model_type: str = typer.Argument(..., help="Model type"),
    name: str = typer.Option(..., "--name", "-n", help="Model name"),
    data_path: Optional[str] = typer.Option(None, "--data", "-d", help="CSV/JSON dataset"),
    target: Optional[str] = typer.Option(None, "--target", help="Target column"),
    workers: int = typer.Option(0, "--workers", "-w", help="Workers (0=auto)"),
    backend: str = typer.Option("auto", "--backend", "-b", help="auto|thread|torch"),
    epochs: int = typer.Option(30, "--epochs", "-e"),
):
    """Distributed training (multi-GPU / multi-node / multi-thread)."""
    from aweai.distributed import train_distributed

    try:
        if data_path:
            from aweai.data import load_any
            ds = load_any(data_path, target_column=target or None)
            X, y = ds.X, ds.y
        else:
            X, y = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]
        res = train_distributed(model_type, name, X, y=y, workers=workers,
                                backend=backend, epochs=epochs)
        typer.echo(json.dumps(res, indent=2, default=str))
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def dworld():
    """Detect the distributed world (GPUs, nodes, backend)."""
    from aweai.distributed import detect_world

    typer.echo(json.dumps(detect_world(), indent=2))

@app.command()
def market(
    action: str = typer.Argument(..., help="publish|search|list|info|download|rate|stats"),
    arg: Optional[str] = typer.Argument(None, help="Model name / query / id"),
    tag: Optional[str] = typer.Option(None, "--tag", help="Publish tag"),
    description: Optional[str] = typer.Option(None, "--description", help="Publish description"),
):
    """Model marketplace: publish/download/rate/search models."""
    from aweai import market as mkt

    try:
        if action == "publish":
            if not arg:
                raise ValueError("publish requires a model name")
            typer.echo(json.dumps(mkt.publish(arg, tag=tag or "v1", description=description or ""), indent=2))
        elif action == "search":
            typer.echo(json.dumps(mkt.search(arg or ""), indent=2))
        elif action == "list":
            typer.echo(json.dumps(mkt.list_listings(), indent=2))
        elif action == "info":
            if not arg:
                raise ValueError("info requires a model id")
            typer.echo(json.dumps(mkt.info(arg), indent=2))
        elif action == "download":
            if not arg:
                raise ValueError("download requires a model id")
            typer.echo(json.dumps(mkt.download(arg), indent=2))
        elif action == "rate":
            if not arg:
                raise ValueError("rate requires <id> <stars>")
            parts = arg.split()
            if len(parts) < 2:
                raise ValueError("rate requires <id> <stars>")
            typer.echo(json.dumps(mkt.rate(parts[0], float(parts[1])), indent=2))
        elif action == "stats":
            typer.echo(json.dumps(mkt.stats(), indent=2))
        else:
            typer.echo(f"Unknown action: {action}", err=True)
            raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def integrations(
    action: str = typer.Argument("list", help="list|chat"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="openai|google|microsoft|anthropic|huggingface"),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Chat message"),
):
    """AI-tool integrations (BYO+): OpenAI/Google/Microsoft/Anthropic/HF."""
    from aweai.integrations import list_tools

    if action == "list":
        typer.echo(json.dumps(list_tools(), indent=2))
    elif action == "chat":
        from aweai.integrations import chat

        if not provider or not message:
            typer.echo("chat requires --provider and --message", err=True)
            raise typer.Exit(code=1)
        typer.echo(json.dumps(chat(provider, message), indent=2))
    else:
        typer.echo(f"Unknown action: {action}", err=True)
        raise typer.Exit(code=1)


@app.command()
def tools(
    action: str = typer.Argument("list", help="list | run | describe | categories"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Tool name (run/describe)"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    params: Optional[str] = typer.Option(None, "--params", help="JSON kwargs for run"),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Hundreds of extension tools (list/run/describe/categories)."""
    from aweai.tools import list_categories, list_tools, run_tool

    if action == "categories":
        cats = list_categories()
        if as_json:
            typer.echo(json.dumps(cats))
        else:
            typer.echo(f"Tool categories: {len(cats)}")
            for c in cats:
                typer.echo(f"  {c['category']:<16} {c['count']} tools")
        return
    if action == "list":
        items = list_tools(category=category)
        if as_json:
            typer.echo(json.dumps(items))
        else:
            typer.echo(f"AWEAI tools: {len(items)} registered")
            for t in items:
                typer.echo(f"  {t['name']:<40} [{t['category']}] {t['purpose']}")
        return
    if action == "describe":
        from aweai.tools import get_tool

        meta = get_tool(name or "")
        if meta is None:
            typer.echo(f"Unknown tool: {name}")
            raise typer.Exit(code=1)
        if as_json:
            typer.echo(json.dumps({k: v for k, v in meta.items() if k != "fn"}))
        else:
            typer.echo(f"Tool: {meta['name']}")
            typer.echo(f"Category: {meta['category']}")
            typer.echo(f"Purpose: {meta['purpose']}")
            typer.echo(f"Signature: {meta['signature']}")
        return
    if action == "run":
        kwargs = json.loads(params) if params else {}
        res = run_tool(name or "", **kwargs)
        typer.echo(json.dumps(res, ensure_ascii=False, indent=2))
        return
    typer.echo(f"Unknown action: {action}")
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# v4.0: command groups (pure CLI, no UI)
# ---------------------------------------------------------------------------
from aweai.cmd import data_collect, data_manage, device, infra, model, ops, provider  # noqa: E402

app.add_typer(data_collect.app, name="collect", help="Data collection: scraping, crawling, import/export, cleaning, synthetic data")
app.add_type(data_manage.app, name="data", help="Data management: datasets, pipelines, preprocessing, tokenization, embedding")
app.add_typer(model.app, name="model", help="Models: train/eval/manage 16+ types, fine-tune, quantize, export")
app.add_type(provider.app, name="providers", help="Providers & integrations: API keys, external models, fine-tuning")
app.add_type(device.app, name="devices", help="Devices & servers: SSH, remote, cluster, distributed training")
app.add_type(ops.app, name="ops", help="Operations: users, auth, billing, workflows, schedulers, AGI, RAG, security, monitoring, backup")
app.add_type(infra.arch_app, name="arch", help="Architecture: change model type/shape (MoE/Transformer/RNN/CNN/hybrid/custom), any size")
app.add_type(infra.scale_app, name="scale", help="Scale: train ANY model size (unlimited params), precision, offload, checkpoint/resume")
app.add_type(infra.cluster_app, name="cluster", help="Cluster: multi-node servers, GPU allocation, health, auto-scaling")
app.add_type(infra.dbops_app, name="dbops", help="Databases: connect, ingest training data, snapshot, restore, query")


# ---------------------------------------------------------------------------
# v4.0: bulk command groups (declarative commands from aweai.bulk)
# ---------------------------------------------------------------------------
from aweai import bulk as _bulk  # noqa: E402
import aweai.bulk_extra as _bulk_extra  # noqa: E402,F401  (registers extra groups)
import aweai.bulk_v5 as _bulk_v5  # noqa: E402,F401  (v4.1 batch 1: agi/safety/secret/audit/nlp/prompt/quality/admin/ds/http/env/test)
import aweai.bulk_v6 as _bulk_v6  # noqa: E402,F401  (v4.1 batch 2: vision/audio/video/dataset/feature/quant/deploy/api/dataops/crypto2/geo/stats2/mcp2/agent2/workflow2/monitor2)
import aweai.bulk_v7 as _bulk_v7  # noqa: E402,F401  (v4.2 batch: arch/model-size/distributed/precision/sharding/checkpoint/gpu/training/database/clusterops)


def _make_bulk_command(group: str, spec: Dict[str, Any]):
    """Build a Typer command function with explicit options from a spec."""
    name = spec["name"]
    help_text = spec["help"]
    params = spec["params"]
    fn = spec["fn"]

    def _pytype(default: Any) -> str:
        if isinstance(default, bool):
            return "bool"
        if isinstance(default, int):
            return "int"
        if isinstance(default, float):
            return "float"
        return "str"

    arg_defs = []
    call_args = []
    for pname, default, phelp in params:
        t = _pytype(default)
        arg_defs.append("    %s: %s = typer.Option(%r, '--%s', help=%r)" % (pname, t, default, pname, phelp))
        call_args.append("'%s': %s" % (pname, pname))
    kwargs_str = ", ".join(call_args)
    src = (
        "def _cmd(\n"
        + ",\n".join(arg_defs) + "\n"
        "):\n"
        "    _kwargs = {%s}\n" % kwargs_str
        + "    _res = fn(_kwargs)\n"
        + "    typer.echo(json.dumps({'group': group, 'command': name, **_res}, indent=2, ensure_ascii=False, default=str))\n"
    )
    ns: Dict[str, Any] = {"typer": typer, "json": json, "fn": fn,
                          "group": group, "name": name}
    exec(compile(src, f"<bulk:{group}:{name}>", "exec"), ns)
    cmd = ns["_cmd"]
    cmd.__doc__ = help_text
    return cmd


_bulk_apps: Dict[str, typer.Typer] = {}
for _g in _bulk.groups():
    _sub = typer.Typer(help=f"{_g} utilities (bulk command group)")
    for _spec in _bulk.specs_for(_g):
        _sub.command(name=_spec["name"], help=_spec["help"])(_make_bulk_command(_g, _spec))
    _bulk_apps[_g] = _sub
    app.add_typer(_sub, name=_g, help=f"{_g} utilities")


# ---------------------------------------------------------------------------
# v4.0: AI / ASI / AGI knowledge
# ---------------------------------------------------------------------------
ai_app = typer.Typer(help="AI/ASI/AGI knowledge base, roadmap, concepts")


@ai_app.command("explain")
def ai_explain(term: str = typer.Argument(..., help="Concept, e.g. transformer/rag/alignment")):
    """Explain an AI/ASI/AGI concept from the knowledge base."""
    from aweai.ai import get_concept, search_concepts

    entry = get_concept(term)
    if entry is None:
        found = search_concepts(term, limit=5)
        typer.echo(json.dumps({"ok": False, "error": f"concept '{term}' not found",
                               "suggestions": found}, indent=2))
        raise typer.Exit(code=1)
    typer.echo(json.dumps({"ok": True, "name": term.lower().replace(" ", "_"),
                           "category": entry["category"], "summary": entry["summary"],
                           "detail": entry["detail"]}, indent=2, ensure_ascii=False))


@ai_app.command("search")
def ai_search(query: str = typer.Argument(..., help="Search text"), category: Optional[str] = typer.Option(None, "--category", "-c")):
    """Search the AI knowledge base."""
    from aweai.ai import search_concepts

    typer.echo(json.dumps(search_concepts(query, category=category), indent=2, ensure_ascii=False))


@ai_app.command("list")
def ai_list(category: Optional[str] = typer.Option(None, "--category", "-c")):
    """List knowledge-base concepts."""
    from aweai.ai import CONCEPTS, categories

    typer.echo(json.dumps({"concepts": [{"name": k, "category": v["category"], "summary": v["summary"]}
                                        for k, v in CONCEPTS.items()
                                        if not category or v["category"] == category]}, indent=2, ensure_ascii=False))


@ai_app.command("categories")
def ai_categories():
    """List knowledge categories."""
    from aweai.ai import categories

    typer.echo(json.dumps(categories(), indent=2))


@ai_app.command("timeline")
def ai_timeline():
    """Show AI history timeline."""
    from aweai.ai import TIMELINE

    typer.echo(json.dumps(TIMELINE, indent=2, ensure_ascii=False))


@ai_app.command("roadmap")
def ai_roadmap():
    """Show AGI/ASI roadmap phases."""
    from aweai.ai import ROADMAP

    typer.echo(json.dumps(ROADMAP, indent=2, ensure_ascii=False))


@ai_app.command("levels")
def ai_levels():
    """Show AGI capability levels."""
    from aweai.ai import AGI_LEVELS

    typer.echo(json.dumps(AGI_LEVELS, indent=2, ensure_ascii=False))


@ai_app.command("self-improve")
def ai_self_improve():
    """Show recursive self-improvement hooks."""
    from aweai.ai import SELF_IMPROVEMENT_HOOKS

    typer.echo(json.dumps(SELF_IMPROVEMENT_HOOKS, indent=2, ensure_ascii=False))


@ai_app.command("about")
def ai_about():
    """Knowledge base statistics."""
    from aweai.ai import about

    typer.echo(json.dumps(about(), indent=2))

app.add_typer(ai_app, name="ai", help="AI/ASI/AGI knowledge base, roadmap, concepts")

# ---------------------------------------------------------------------------
# v4.0: commands registry (list/search/describe the whole CLI)
# ---------------------------------------------------------------------------
cmd_app = typer.Typer(help="Inspect the AWEAI command universe")


def _flatten_commands(typer_app: typer.Typer, prefix: str = "") -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for c in typer_app.registered_commands:
        name = c.name or c.callback.__name__
        out.append({"command": f"{prefix}{name}".strip(), "help": (c.help or c.callback.__doc__ or "").strip()})
    for g in typer_app.registered_groups:
        info = g.typer_instance
        out.extend(_flatten_commands(info, prefix=f"{prefix}{g.name} "))
    return out


@cmd_app.command("list")
def commands_list(
    group: Optional[str] = typer.Option(None, "--group", "-g", help="Filter by group"),
    as_json: bool = typer.Option(False, "--json", help="Raw JSON"),
):
    """List every command in the AWEAI CLI (hundreds)."""
    cmds = _flatten_commands(app)
    if group:
        cmds = [c for c in cmds if c["command"].startswith(group)]
    if as_json:
        typer.echo(json.dumps(cmds, indent=2, ensure_ascii=False))
    else:
        typer.echo(f"AWEAI commands: {len(cmds)}")
        for c in cmds:
            typer.echo(f"  aweai {c['command']:<38} {c['help'][:60]}")


@cmd_app.command("search")
def commands_search(query: str = typer.Argument(..., help="Search text")):
    """Search commands by keyword."""
    cmds = _flatten_commands(app)
    q = query.lower()
    hits = [c for c in cmds if q in c["command"].lower() or q in c["help"].lower()]
    typer.echo(json.dumps({"query": query, "matches": hits}, indent=2, ensure_ascii=False))


@cmd_app.command("describe")
def commands_describe(command: str = typer.Argument(..., help="Full command path, e.g. 'math add' or 'model train'")):
    """Describe a single command."""
    cmds = _flatten_commands(app)
    hits = [c for c in cmds if c["command"] == command or c["command"].endswith(f" {command}")]
    if not hits:
        typer.echo(json.dumps({"ok": False, "error": f"command '{command}' not found"}, indent=2))
        raise typer.Exit(code=1)
    typer.echo(json.dumps({"ok": True, "command": hits[0]["command"], "help": hits[0]["help"],
                           "usage": f"aweai {hits[0]['command']} --help"}, indent=2, ensure_ascii=False))


@cmd_app.command("count")
def commands_count():
    """Count total commands and groups."""
    cmds = _flatten_commands(app)
    groups = [g.name for g in app.registered_groups]
    typer.echo(json.dumps({"ok": True, "commands": len(cmds), "groups": len(groups), "group_names": groups}, indent=2))


app.add_type(cmd_app, name="commands", help="Inspect the AWEAI command universe")


# ---------------------------------------------------------------------------
# v4.0: wiki generator (docs/wiki/*.md)
# ---------------------------------------------------------------------------
wiki_app = typer.Typer(help="Generate the AWEAI wiki (docs/wiki/*.md)")


@wiki_app.command("build")
def wiki_build(
    out_dir: str = typer.Option("docs/wiki", "--out", "-o", help="Output directory"),
    max_specs: int = typer.Option(500, "--max-specs", help="Max bulk specs per page"),
):
    """Generate Markdown wiki pages for every command group."""
    from aweai.wiki import build_wiki

    report = build_wiki(out_dir=out_dir, max_specs=max_specs)
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))


@wiki_app.command("index")
def wiki_index(out_dir: str = typer.Option("docs/wiki", "--out", "-o")):
    """Generate the wiki home/index page."""
    from aweai.wiki import build_wiki_index

    typer.echo(json.dumps(build_wiki_index(out_dir=out_dir), indent=2, ensure_ascii=False))


app.add_type(wiki_app, name="wiki", help="Generate the AWEAI wiki (docs/wiki/*.md)")


if __name__ == "__main__":
    app()
