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


if __name__ == "__main__":
    app()
