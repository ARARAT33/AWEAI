#!/usr/bin/env python3
"""AWEAI command-line interface — AI Model Factory.

Usage:
    aweai --help
    aweai autotest
    aweai train --type mlp --name m1 --data data.csv
    aweai serve
    ...
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

import typer

from aweai import __version__

app = typer.Typer(add_completion=False, help="AWEAI — AI Model Factory (create/train/tune/manage AI models from scratch, no built-in AI, no Hugging Face)")


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
    typer.echo(json.dumps({"deleted": delete_model(name)}))


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
            ds = load_any(path, target_column=target or None)
            parts = train_test_split(ds.X, ds.y, ratio=ratio)
            typer.echo(json.dumps({k: (v.shape if hasattr(v, "shape") else len(v)) for k, v in parts.items()}, indent=2))
        elif action == "augment":
            ds = load_any(path, target_column=target or None)
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
    quick: bool = typer.Option(False, "--quick", help="Skip smoke-train/RAG/i18n/UI"),
    no_ui: bool = typer.Option(False, "--no-ui", help="Skip UI check"),
):
    """Run the full system autotest (deps, imports, smoke-train all model types, RAG, actions, i18n, UI, CLI)."""
    from aweai.autotest import run_autotest

    report = run_autotest(quick=quick, no_ui=no_ui)
    if not report["all_passed"]:
        raise typer.Exit(code=1)


@app.command()
def serve(
    port: int = typer.Option(8888, "--port", "-p", help="Preferred port (auto +1 if busy)"),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open browser"),
):
    """Launch the browser UI on smart port 8888 (+1)."""
    from aweai.ui import serve as serve_ui

    serve_ui(port=port, host=host, open_browser=not no_open)


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


# ---------------------------------------------------------------------------
# v2.2 quantization & edge export
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# v3.0 distributed training & marketplace
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# v3.0 integrations
# ---------------------------------------------------------------------------
@app.command()
def integrations(
    action: str = typer.Argument("list", help="list|chat"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="openai|google|microsoft|anthropic|huggingface"),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Chat message"),
):
    """AI-tool integrations (BYOK): OpenAI/Google/Microsoft/Anthropic/HF."""
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


# ---------------------------------------------------------------------------
# v3.0 megamenus & terminal
# ---------------------------------------------------------------------------
@app.command()
def allc(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search text"),
    count: int = typer.Option(200, "--count", help="Max lines to render"),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Print ALL commands & instructions (10,000+)."""
    from aweai.menus import build_catalog, catalog_stats, render_catalog, search_catalog

    items = build_catalog(min_count=10000)
    if category or search:
        items = search_catalog(items, query=search or "", category=category or "")
    if as_json:
        typer.echo(json.dumps(items))
    else:
        stats = catalog_stats(items)
        typer.echo(f"AWEAI instruction catalog: {stats['total']} entries, {stats['categories']} categories")
        typer.echo(render_catalog(items, max_lines=count))


@app.command()
def autoallc(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search text"),
    count: int = typer.Option(200, "--count", help="Max lines to render"),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Print ALL automations (10,000+)."""
    from aweai.menus import build_automations, catalog_stats, render_catalog, search_catalog

    items = build_automations(min_count=10000)
    if category or search:
        items = search_catalog(items, query=search or "", category=category or "")
    if as_json:
        typer.echo(json.dumps(items))
    else:
        stats = catalog_stats(items)
        typer.echo(f"AWEAI automation catalog: {stats['total']} entries, {stats['categories']} categories")
        typer.echo(render_catalog(items, max_lines=count))


@app.command()
def terminal():
    """Launch the in-app terminal (full REPL with all tools)."""
    from aweai.terminal import repl

    repl()


if __name__ == "__main__":
    app()
