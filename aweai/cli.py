"""AWEAI command-line interface.

Usage:
    aweai --help
    aweai chat
    aweai serve [--port 8888] [--no-browser]
    aweai models
    aweai hardware
    aweai train --data data.jsonl --name my_model
    aweai finetune --base Qwen/Qwen2.5-0.5B-Instruct --data data.jsonl --name tuned
    aweai continue --checkpoint path/to/model --data data.jsonl
    aweai rag index --path docs/
    aweai rag ask --query "..."
    aweai agent --task "..."
    aweai action "new model with this data"
    aweai config set key=value
    aweai langs
    aweai doctor
"""

from __future__ import annotations

import json
import sys
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from aweai import __version__
from aweai.config import get_config

app = typer.Typer(add_completion=False, help="AWEAI — Universal AI Toolbox")
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"AWEAI v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-V", callback=_version_callback, is_eager=True),
) -> None:
    """AWEAI: everything AI in one lightweight toolbox."""


# ---------- chat ----------
@app.command()
def chat(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model id to use"),
    one_shot: Optional[str] = typer.Option(None, "--prompt", "-p", help="Single prompt, non-interactive"),
) -> None:
    """Chat with AWEAI in the terminal."""
    from aweai.models.inference import LLM

    llm = LLM(model_id=model)
    console.print(f"[bold cyan]AWEAI[/] ready. Type 'exit' to quit. (model={model or 'auto'})")
    history: List[dict] = []
    if one_shot:
        reply = llm.chat([{"role": "user", "content": one_shot}])
        console.print(reply)
        return
    while True:
        try:
            prompt = typer.prompt("You", default="")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip().lower() in ("exit", "quit", "q", "выход", "ելք"):
            break
        if not prompt.strip():
            continue
        history.append({"role": "user", "content": prompt})
        try:
            reply = llm.chat(history)
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
            continue
        console.print(f"[bold green]AWEAI[/]: {reply}")
        history.append({"role": "assistant", "content": reply})


# ---------- serve ----------
@app.command()
def serve(
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Preferred port (default 8888, +1 if busy)"),
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open the browser"),
) -> None:
    """Launch the browser UI (port 8888, auto +1 on conflict)."""
    from aweai.ui.api import serve as ui_serve

    ui_serve(port=port, host=host, open_browser=not no_browser)


# ---------- models ----------
@app.command("models")
def models_cmd() -> None:
    """List the model catalog and installed models."""
    from aweai.models.registry import ModelRegistry
    from aweai.hardware import detect
    from aweai.models.selector import pick_best_model

    hw = detect()
    best = pick_best_model(hw)
    console.print(f"[bold]Hardware:[/] {hw.summary()}")
    console.print(f"[bold green]Recommended model:[/] {best['id'] if best else '—'}")

    reg = ModelRegistry()
    table = Table(title="Model Catalog")
    table.add_column("ID")
    table.add_column("Family")
    table.add_column("Params")
    table.add_column("Context")
    table.add_column("Min RAM")
    table.add_column("License")
    for m in reg.catalog():
        table.add_row(m["id"], m["family"], f"{m['params_b']}B", str(m["context"]), f"{m['min_ram_gb']}GB", m["license"])
    console.print(table)

    installed = reg.installed()
    if installed:
        console.print(f"[bold]Installed local models:[/] {len(installed)}")
        for m in installed:
            console.print(f"  • {m.get('name')} ({m.get('mode', 'local')}) @ {m.get('path')}")


# ---------- hardware ----------
@app.command()
def hardware() -> None:
    """Detect resources and show the best model for this machine."""
    from aweai.hardware import detect
    from aweai.models.selector import pick_best_model, suggest_models

    hw = detect()
    best = pick_best_model(hw)
    table = Table(title="Hardware Detection")
    table.add_column("Property")
    table.add_column("Value")
    for k, v in hw.to_dict().items():
        if k != "recommended_tier":
            table.add_row(k, str(v))
    console.print(table)
    console.print(f"[bold green]Best model:[/] {best['id']} ({best['params_b']}B) — {best['hf']}")
    console.print("[bold]Other suggestions:[/] " + ", ".join(m["id"] for m in suggest_models(hw, limit=3)))


# ---------- train ----------
@app.command()
def train(
    data: str = typer.Option(..., "--data", "-d", help="Path to data (.jsonl/.json/.txt)"),
    name: str = typer.Option("my_model", "--name", "-n"),
    epochs: int = typer.Option(3, "--epochs", "-e"),
    lr: float = typer.Option(0.003, "--lr", help="Learning rate"),
    seed_text: str = typer.Option("Բարեւ AWEAI", "--seed"),
) -> None:
    """Create a brand-new model from scratch on the given data."""
    from aweai.models.trainer import train_scratch

    console.print(f"[bold]Training new model[/] '{name}' on {data} (epochs={epochs})…")
    result = train_scratch(name, data, epochs=epochs, lr=lr, seed_text=seed_text)
    console.print(f"[bold green]Done[/] in {result.duration_s:.1f}s: {result.path}")
    for msg in result.messages:
        console.print(f"  • {msg}")


# ---------- finetune ----------
@app.command()
def finetune(
    base: str = typer.Option(..., "--base", "-b", help="Base HF model id"),
    data: str = typer.Option(..., "--data", "-d"),
    name: str = typer.Option("tuned", "--name", "-n"),
    epochs: int = typer.Option(1, "--epochs", "-e"),
    lora_r: int = typer.Option(8, "--lora-r"),
    lora_alpha: int = typer.Option(16, "--lora-alpha"),
) -> None:
    """Fine-tune an existing model with LoRA."""
    from aweai.models.trainer import finetune

    console.print(f"[bold]Fine-tuning[/] {base} on {data}…")
    try:
        result = finetune(base, name, data, epochs=epochs, lora_r=lora_r, lora_alpha=lora_alpha)
        console.print(f"[bold green]Done[/]: {result.path}")
        for msg in result.messages:
            console.print(f"  • {msg}")
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)


# ---------- continue ----------
@app.command("continue")
def continue_cmd(
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Path to existing model"),
    data: str = typer.Option(..., "--data", "-d"),
    name: str = typer.Option("continued", "--name", "-n"),
    epochs: int = typer.Option(1, "--epochs", "-e"),
) -> None:
    """Continue training an existing model checkpoint."""
    from aweai.models.trainer import continue_training

    console.print(f"[bold]Continuing training[/] {checkpoint} on {data}…")
    result = continue_training(name, checkpoint, data, epochs=epochs)
    console.print(f"[bold green]Done[/]: {result.path}")


# ---------- rag ----------
@app.command()
def rag(
    action: str = typer.Argument("ask", help="index | ask | stats | clear"),
    path: Optional[str] = typer.Option(None, "--path", "-p"),
    query: Optional[str] = typer.Option(None, "--query", "-q"),
    top_k: int = typer.Option(4, "--top-k"),
) -> None:
    """RAG: index documents and ask questions."""
    from aweai.rag.engine import RAGEngine

    engine = RAGEngine()
    if action == "index":
        if not path:
            console.print("[red]--path required for index[/]")
            raise typer.Exit(1)
        p = path
        import os

        added = engine.index_directory(p) if os.path.isdir(p) else engine.index_file(p)
        console.print(f"[bold green]Indexed[/] {added} chunks. {engine.stats()}")
    elif action == "ask":
        if not query:
            console.print("[red]--query required for ask[/]")
            raise typer.Exit(1)
        result = engine.ask(query, top_k=top_k)
        console.print(f"[bold]Answer:[/] {result['answer']}")
        console.print("[dim]Sources:[/]")
        for s in result["sources"]:
            console.print(f"  • {s['id']} ({s['score']})")
    elif action == "stats":
        console.print(engine.stats())
    elif action == "clear":
        engine.clear()
        console.print("Cleared.")


# ---------- agent ----------
@app.command()
def agent(
    task: str = typer.Option(..., "--task", "-t", help="Task description"),
    max_steps: int = typer.Option(5, "--max-steps"),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
) -> None:
    """Run the ReAct agent on a task."""
    from aweai.agents.engine import AgentEngine

    console.print(f"[bold]Agent[/] task: {task}")
    agent = AgentEngine.create()
    result = agent.run(task, max_steps=max_steps, verbose=verbose)
    console.print(f"[bold green]Final:[/] {result['final']}")
    console.print(f"Tool calls: {result['tool_calls']}")


# ---------- action ----------
@app.command()
def action(
    text: str = typer.Argument(..., help="Natural-language action, e.g. 'new model with this data'"),
    lang: str = typer.Option("en", "--lang", "-l"),
) -> None:
    """Run the automation studio: parse intent and execute."""
    from aweai.actions.runner import ActionsRunner

    console.print(f"[bold]Action:[/] {text}")
    runner = ActionsRunner(lang=lang)
    result = runner.run(text)
    console.print(json.dumps(result, indent=2, ensure_ascii=False))


# ---------- config ----------
@app.command()
def config(
    key: Optional[str] = typer.Argument(None, help="key=value to set, or nothing to show"),
) -> None:
    """Show or update configuration (key=value)."""
    cfg = get_config()
    if not key:
        console.print(json.dumps(cfg.all(), indent=2, ensure_ascii=False))
        return
    if "=" in key:
        k, v = key.split("=", 1)
        # try to coerce
        if v.lower() in ("true", "false"):
            cfg.set(k, v.lower() == "true")
        else:
            try:
                cfg.set(k, int(v))
            except ValueError:
                cfg.set(k, v)
        console.print(f"Set {k} = {cfg.get(k)}")
    else:
        console.print(f"{key} = {cfg.get(key)}")


# ---------- langs ----------
@app.command("langs")
def langs_cmd() -> None:
    """List supported UI languages."""
    from aweai.i18n import available_languages

    table = Table(title="Languages")
    table.add_column("Code")
    table.add_column("Name")
    for code, name in available_languages().items():
        table.add_row(code, name)
    console.print(table)


# ---------- doctor ----------
@app.command()
def doctor() -> None:
    """Check the installation and environment."""
    from aweai.config import get_platform, which_ok
    from aweai.hardware import detect

    console.print("[bold]AWEAI Doctor[/]")
    console.print(f"Version: {__version__}")
    console.print(f"Platform: {get_platform()}")

    checks = [
        ("typer", True),
        ("fastapi", _import_ok("fastapi")),
        ("uvicorn", _import_ok("uvicorn")),
        ("torch", _import_ok("torch")),
        ("transformers", _import_ok("transformers")),
        ("peft", _import_ok("peft")),
        ("chromadb", _import_ok("chromadb")),
        ("faiss", _import_ok("faiss")),
    ]
    for name, ok in checks:
        console.print(f"  {'✓' if ok else '✗'} {name}")

    console.print("[bold]Hardware:[/] " + detect().summary())
    console.print("Tip: `pip install aweai[all]` for the full ML stack.")


def _import_ok(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    app()
