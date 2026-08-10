# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Model commands: create, train, evaluate, manage (16+ types),
fine-tuning, transfer learning, quantization and export."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from aweai.cmd.common import err, jdump, ok

app = typer.Typer(help="Models: create/train/evaluate/manage 16+ types, fine-tune, quantize, export")


def _try_train_model(model_type: str, name: str, data_path: Optional[str], target: Optional[str],
                     text_path: Optional[str], params: Optional[str], epochs: Optional[int],
                     normalize: Optional[str]) -> Dict[str, Any]:
    from aweai.train import train as train_model

    p = json.loads(params) if params else {}
    if epochs:
        p["epochs"] = epochs
    return train_model(model_type, name, data_path=data_path, text_path=text_path,
                       target=target, params=p, normalize=normalize)


@app.command("train")
def train(
    model_type: str = typer.Option("mlp", "--type", "-t", help="Model type"),
    name: str = typer.Option(..., "--name", "-n", help="Model name"),
    data_path: Optional[str] = typer.Option(None, "--data", "-d", help="CSV/JSON/JSONL/text file"),
    target: Optional[str] = typer.Option(None, "--target", help="Target column"),
    text_path: Optional[str] = typer.Option(None, "--text", help="Text file for ngram/rnn/lstm"),
    params: Optional[str] = typer.Option(None, "--params", help="JSON hyperparameters"),
    epochs: int = typer.Option(None, "--epochs", "-e", help="Override epochs"),
    normalize: Optional[str] = typer.Option(None, "--normalize", help="standardize|minmax"),
):
    """Train a model from scratch and save it to the zoo."""
    try:
        res = _try_train_model(model_type, name, data_path, target, text_path, params, epochs, normalize)
        typer.echo(jdump(res))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("types")
def types():
    """List all available model types (16+)."""
    try:
        from aweai.models.registry import MODEL_TYPES

        typer.echo(jdump(MODEL_TYPES))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("list")
def list_models():
    """List all models in the zoo."""
    try:
        from aweai.management import list_models

        typer.echo(jdump(list_models()))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("info")
def info(name: str = typer.Argument(..., help="Model name")):
    """Show model metadata and metrics."""
    try:
        from aweai.management import load_model

        model, meta = load_model(name)
        typer.echo(jdump({"name": name, "model_type": meta.get("model_type"),
                          "metrics": meta.get("metrics", {}), "history": meta.get("history", {}),
                          "created": meta.get("created_at"), "params": _count_params(model)}))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


def _count_params(model: Any) -> int:
    total = 0
    for v in vars(model).values():
        if hasattr(v, "shape") and hasattr(v, "size"):
            try:
                total += int(v.size)
            except Exception:
                pass
        elif isinstance(v, dict):
            for vv in v.values():
                if hasattr(vv, "shape") and hasattr(vv, "size"):
                    try:
                        total += int(vv.size)
                    except Exception:
                        pass
    return total


@app.command("eval")
def evaluate(
    name: str = typer.Argument(..., help="Model name"),
    data_path: Optional[str] = typer.Option(None, "--data", "-d", help="Evaluation CSV/JSON"),
    target: Optional[str] = typer.Option(None, "--target", help="Target column"),
):
    """Evaluate a model (accuracy/precision/recall/F1 + confusion matrix)."""
    try:
        from aweai.data import load_any
        from aweai.eval import classification_report
        from aweai.management import load_model

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
        typer.echo(jdump(report))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("predict")
def predict(
    name: str = typer.Argument(..., help="Model name"),
    input: str = typer.Option(..., "--input", "-i", help="CSV/JSON with features, or text"),
    target: Optional[str] = typer.Option(None, "--target", help="Target column (optional)"),
):
    """Run predictions with a saved model."""
    try:
        from aweai.data import load_any
        from aweai.management import load_model

        model, meta = load_model(name)
        ds = load_any(input, target_column=target or None)
        pred = model.predict(ds.X if ds.X is not None else ds.texts)
        out = pred.tolist() if hasattr(pred, "tolist") else list(pred)
        typer.echo(jdump(ok(name=name, predictions=out)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("continue")
def continue_train(
    name: str = typer.Argument(..., help="Existing model name"),
    data_path: Optional[str] = typer.Option(None, "--data", "-d"),
    epochs: int = typer.Option(10, "--epochs", "-e"),
):
    """Continue/fine-tune an existing model on new data."""
    try:
        from aweai.train import continue_training

        res = continue_training(name, data_path=data_path, epochs=epochs)
        typer.echo(jdump(res))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("fine-tune")
def fine_tune(
    name: str = typer.Argument(..., help="Base model name"),
    data_path: str = typer.Option(..., "--data", "-d", help="New data"),
    target: Optional[str] = typer.Option(None, "--target"),
    epochs: int = typer.Option(5, "--epochs", "-e"),
    new_name: Optional[str] = typer.Option(None, "--new-name", help="Save as new name"),
):
    """Fine-tune (transfer learning) a model on new data."""
    try:
        from aweai.data import load_any
        from aweai.management import load_model, save_model
        from aweai.train import fit_model

        model, meta = load_model(name)
        ds = load_any(data_path, target_column=target or None)
        model = fit_model(model, ds.X, y=ds.y, epochs=epochs)
        final_name = new_name or f"{name}_ft"
        save_model(model, final_name, meta={**meta, "base_model": name, "fine_tuned": True})
        typer.echo(jdump(ok(name=final_name, base=name, epochs=epochs)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("transfer")
def transfer(
    name: str = typer.Argument(..., help="Source model"),
    data_path: str = typer.Option(..., "--data", "-d"),
    target: Optional[str] = typer.Option(None, "--target"),
    new_name: Optional[str] = typer.Option(None, "--new-name"),
):
    """Transfer learning: reuse a model's weights on a new dataset."""
    return fine_tune(name, data_path, target, 3, new_name)


@app.command("tune")
def tune(
    model_type: str = typer.Argument(..., help="Model type"),
    data_path: str = typer.Option(..., "--data", "-d"),
    target: Optional[str] = typer.Option(None, "--target"),
    method: str = typer.Option("grid", "--method", "-m", help="grid|random"),
):
    """Hyperparameter search (grid/random)."""
    try:
        from aweai.data import load_any
        from aweai.train import tune as tune_model

        ds = load_any(data_path, target_column=target or None)
        best = tune_model(model_type, ds.X, y=ds.y, method=method)
        typer.echo(jdump(best))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("quantize")
def quantize(
    name: str = typer.Argument(..., help="Model name"),
    fmt: str = typer.Option("int8", "--fmt", "-f", help="float16|int8|uint8|int4"),
):
    """Quantize a model (float16/int8/uint8/int4)."""
    try:
        from aweai.quantize import quantize_model

        typer.echo(jdump(quantize_model(name, fmt=fmt)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("export")
def export(
    name: str = typer.Argument(..., help="Model name"),
    fmt: str = typer.Option("json", "--fmt", "-f", help="json|raw|onnx|torchscript"),
    out_dir: Optional[str] = typer.Option(None, "--out-dir", help="Output directory"),
):
    """Export a model to json/raw/onnx/torchscript."""
    try:
        from aweai.management import export_model

        typer.echo(jdump(export_model(name, fmt=fmt, out_dir=out_dir)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("export-edge")
def export_edge(
    name: str = typer.Argument(..., help="Model name"),
    fmt: str = typer.Option("onnx", "--fmt", "-f", help="onnx|tflite|torchscript|edge_json"),
    quantize_fmt: Optional[str] = typer.Option(None, "--quantize", help="float16|int8|uint8|int4"),
):
    """Export a model to edge formats (ONNX/TFLite/TorchScript)."""
    try:
        from aweai.export import export_edge as _export_edge

        typer.echo(jdump(_export_edge(name, fmt=fmt, quantize=quantize_fmt)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("footprint")
def footprint(name: str = typer.Argument(..., help="Model name")):
    """Estimate on-device footprint (fp32/fp16/int8)."""
    try:
        from aweai.export import estimate_edge_footprint

        typer.echo(jdump(estimate_edge_footprint(name)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("import")
def import_model(
    file: str = typer.Argument(..., help="Path to model.json"),
    name: Optional[str] = typer.Option(None, "--name", "-n"),
):
    """Import a model from a JSON export."""
    try:
        from aweai.management import import_model as _import

        typer.echo(jdump(_import(file, name=name)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("delete")
def delete(
    name: str = typer.Argument(..., help="Model name"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete a model from the zoo."""
    try:
        from aweai.management import delete_model

        if not yes:
            confirm = typer.confirm(f"Delete model '{name}'?")
            if not confirm:
                typer.echo(jdump(ok(aborted=True)))
                raise typer.Exit(code=0)
        typer.echo(jdump(ok(deleted=delete_model(name))))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("compare")
def compare(names: List[str] = typer.Argument(..., help="Model names to compare")):
    """Compare models side by side."""
    try:
        from aweai.management import compare_models

        typer.echo(jdump(compare_models(names)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("recommend")
def recommend(task: str = typer.Argument("classification", help="classification|regression|clustering|text|vision|time_series|generative|anomaly")):
    """Recommend the best model type for a task on this machine."""
    try:
        from aweai.selector import recommend

        typer.echo(jdump(recommend(task)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("distributed")
def distributed(
    model_type: str = typer.Argument(..., help="Model type"),
    name: str = typer.Option(..., "--name", "-n"),
    data_path: Optional[str] = typer.Option(None, "--data", "-d"),
    target: Optional[str] = typer.Option(None, "--target"),
    workers: int = typer.Option(0, "--workers", "-w", help="0=auto"),
    backend: str = typer.Option("auto", "--backend", "-b", help="auto|thread|torch"),
    epochs: int = typer.Option(30, "--epochs", "-e"),
):
    """Distributed training (multi-thread/multi-node)."""
    try:
        from aweai.distributed import train_distributed

        if data_path:
            from aweai.data import load_any
            ds = load_any(data_path, target_column=target or None)
            X, y = ds.X, ds.y
        else:
            X, y = [[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]
        res = train_distributed(model_type, name, X, y=y, workers=workers, backend=backend, epochs=epochs)
        typer.echo(jdump(res))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("world")
def world():
    """Detect the distributed world (GPUs, nodes, backend)."""
    try:
        from aweai.distributed import detect_world

        typer.echo(jdump(detect_world()))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)
