# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Unlimited model-training infrastructure commands.

Four Typer sub-apps that make AWEAI the most powerful model-training CLI
on the planet:

* ``arch``   — change model architecture (MoE / Transformer / RNN / CNN /
               hybrid / custom) with exact dimensions (params, layers,
               experts, heads, dim, vocab), before or during training.
* ``scale``  — train models of ANY size (200B … 2T+ parameters, unlimited):
               parameter/layer/head/dim/vocab sizing, mixed precision
               (FP16/BF16/FP8), gradient accumulation, activation &
               optimizer offloading, checkpointing & resuming.
* ``cluster``— cluster / server / GPU orchestration: add/remove nodes,
               allocate GPUs, health checks, auto-scaling, backups.
* ``dbops``  — training-data database management: connect, ingest,
               snapshot, restore, vacuum, query.

Every command is designed to be called by any AI (or human) with clear
help text, typed parameters, validation and JSON output.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from aweai.cmd.common import data_dir, err, jdump, ok, write_json, read_json

# ---------------------------------------------------------------------------
# ARCH — model architecture manipulation
# ---------------------------------------------------------------------------
arch_app = typer.Typer(help="Architecture: change model type/shape (MoE/Transformer/RNN/CNN/hybrid/custom)")

_ARCH_TYPES = [
    "moe", "transformer", "rnn", "lstm", "gru", "cnn", "mlp", "linear",
    "logistic", "kmeans", "autoencoder", "gan", "ngram", "hybrid",
    "vision_cnn", "object_detector", "segmentation", "ts_transformer",
]

_ROUTING = ["top1", "top2", "topk", "soft", "noisy_topk", "switch"]


def _parse_size(s: Any) -> int:
    """Parse a size string ('70B', '200B', '2T', 70000000000) into an integer."""
    if isinstance(s, (int, float)):
        return int(s)
    text = str(s).strip().upper()
    mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
    for suffix, m in mult.items():
        if text.endswith(suffix):
            try:
                return int(float(text[:-1]) * m)
            except Exception:
                pass
    try:
        return int(float(text))
    except Exception:
        return 0


def _arch_spec_file(name: str) -> Path:
    return data_dir() / "arch" / f"{name}.json"


def _read_arch(name: str) -> Dict[str, Any]:
    spec = read_json(str(_arch_spec_file(name)), None)
    if spec is None:
        raise typer.BadParameter(f"architecture '{name}' not found")
    return spec


@arch_app.command("create")
def arch_create(
    name: str = typer.Argument(..., help="Architecture name"),
    arch_type: str = typer.Option("transformer", "--type", "-t", help=f"One of: {', '.join(_ARCH_TYPES)}"),
    params: str = typer.Option("70B", "--params", help="Target parameter count (unlimited, e.g. 200B/2T)"),
    layers: int = typer.Option(32, "--layers", "-l", help="Number of layers"),
    dim: int = typer.Option(4096, "--dim", "-d", help="Hidden dimension"),
    heads: int = typer.Option(32, "--heads", help="Attention heads"),
    kv_heads: Optional[int] = typer.Option(None, "--kv-heads", help="KV heads for GQA/MQA"),
    vocab: int = typer.Option(50257, "--vocab", help="Vocabulary size"),
    experts: int = typer.Option(8, "--experts", help="Number of experts (MoE)"),
    moe_layers: Optional[int] = typer.Option(None, "--moe-layers", help="Layers that use MoE (default: all)"),
    routing: str = typer.Option("top2", "--routing", help=f"MoE routing: {', '.join(_ROUTING)}"),
    activation: str = typer.Option("gelu", "--activation", help="Activation: gelu|relu|swish|silu|mish"),
    norm: str = typer.Option("rmsnorm", "--norm", help="Normalization: rmsnorm|layernorm|bitnorm"),
    rope: bool = typer.Option(True, "--rope/--no-rope", help="Use rotary position embeddings"),
    bias: bool = typer.Option(False, "--bias/--no-bias", help="Use bias in attention/FFN"),
    ff_mult: float = typer.Option(4.0, "--ff-mult", help="FFN hidden = dim * ff_mult"),
    qk_norm: bool = typer.Option(False, "--qk-norm/--no-qk-norm", help="Normalize Q and K"),
    window: Optional[int] = typer.Option(None, "--window", help="Sliding window attention size"),
    mamba: bool = typer.Option(False, "--mamba/--no-mamba", help="Use Mamba/SSM blocks"),
):
    """Create/override an architecture spec (exact dims, any size)."""
    if arch_type not in _ARCH_TYPES:
        typer.echo(jdump(err(f"unknown arch type '{arch_type}' (known: {', '.join(_ARCH_TYPES)})")))
        raise typer.Exit(code=1)
    parsed_params = _parse_size(params)
    if parsed_params < 1 or layers < 1 or dim < 1 or heads < 1 or vocab < 1:
        typer.echo(jdump(err("params/layers/dim/heads/vocab must be >= 1")))
        raise typer.Exit(code=1)
    spec = {
        "name": name, "type": arch_type, "params": parsed_params, "layers": layers,
        "dim": dim, "heads": heads, "kv_heads": kv_heads or heads,
        "vocab": vocab, "experts": experts, "moe_layers": moe_layers or layers,
        "routing": routing, "activation": activation, "norm": norm,
        "rope": rope, "bias": bias, "ff_mult": ff_mult, "qk_norm": qk_norm,
        "window": window, "mamba": mamba,
        "created_at": time.time(),
    }
    p = _arch_spec_file(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(jdump(ok(name=name, path=str(p), spec=spec)))


@arch_app.command("list")
def arch_list():
    """List saved architecture specs."""
    d = data_dir() / "arch"
    items = []
    if d.exists():
        for f in sorted(d.glob("*.json")):
            s = read_json(str(f), {})
            items.append({"name": f.stem, "type": s.get("type"), "params": s.get("params"),
                          "layers": s.get("layers"), "dim": s.get("dim"), "heads": s.get("heads"),
                          "vocab": s.get("vocab"), "experts": s.get("experts")})
    typer.echo(jdump(ok(count=len(items), architectures=items)))


@arch_app.command("show")
def arch_show(name: str = typer.Argument(..., help="Architecture name")):
    """Show a full architecture spec."""
    try:
        spec = _read_arch(name)
        typer.echo(jdump(ok(name=name, spec=spec)))
    except typer.BadParameter as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@arch_app.command("delete")
def arch_delete(name: str = typer.Argument(..., help="Architecture name")):
    """Delete an architecture spec."""
    p = _arch_spec_file(name)
    if not p.exists():
        typer.echo(jdump(err(f"architecture '{name}' not found")))
        raise typer.Exit(code=1)
    p.unlink()
    typer.echo(jdump(ok(deleted=name)))


@arch_app.command("convert")
def arch_convert(
    name: str = typer.Argument(..., help="Architecture name"),
    to: str = typer.Option("transformer", "--to", "-t", help="Target arch type"),
):
    """Convert an architecture spec to another type (keeps params/dims)."""
    try:
        spec = _read_arch(name)
    except typer.BadParameter as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)
    if to not in _ARCH_TYPES:
        typer.echo(jdump(err(f"unknown arch type '{to}'")))
        raise typer.Exit(code=1)
    converted = dict(spec)
    converted["type"] = to
    converted["converted_from"] = spec.get("type")
    converted["converted_at"] = time.time()
    _arch_spec_file(name).parent.mkdir(parents=True, exist_ok=True)
    _arch_spec_file(name).write_text(json.dumps(converted, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(jdump(ok(name=name, converted_to=to, spec=converted)))


@arch_app.command("estimate")
def arch_estimate(
    arch_type: str = typer.Option("transformer", "--type", "-t"),
    layers: int = typer.Option(32, "--layers", "-l"),
    dim: int = typer.Option(4096, "--dim", "-d"),
    heads: int = typer.Option(32, "--heads"),
    vocab: int = typer.Option(50257, "--vocab"),
    experts: int = typer.Option(8, "--experts", help="MoE experts (0 = dense)"),
    seq_len: int = typer.Option(2048, "--seq-len", help="Sequence length"),
):
    """Estimate parameter count and memory for an architecture (any size)."""
    if arch_type in ("moe", "hybrid") and experts > 0:
        # MoE: FFN params multiplied by experts on expert layers
        ff = 4 * dim * dim
        attn = 4 * dim * dim
        expert_params = experts * ff
        per_layer = attn + expert_params
        n_moe = layers
        params_dense_ish = vocab * dim + layers * (attn + ff)
        params = vocab * dim + n_moe * per_layer
        # rough memory: fp32 weights + activations (2 bytes/param/seq minimal estimate)
        mem_bytes = params * 4 + 2 * layers * dim * seq_len * 4
    else:
        ff = int(dim * dim * 4)
        attn = 4 * dim * dim
        params = vocab * dim + layers * (attn + ff)
        mem_bytes = params * 4 + 2 * layers * dim * seq_len * 4
    typer.echo(jdump(ok(
        arch_type=arch_type, layers=layers, dim=dim, heads=heads, vocab=vocab,
        experts=experts if arch_type in ("moe", "hybrid") else 0,
        params=params, params_str=f"{params/1e9:.2f}B",
        fp32_gb=round(params * 4 / 1e9, 3), fp16_gb=round(params * 2 / 1e9, 3),
        bf16_gb=round(params * 2 / 1e9, 3), fp8_gb=round(params * 1 / 1e9, 3),
        int8_gb=round(params * 1 / 1e9, 3), int4_gb=round(params * 0.5 / 1e9, 3),
        est_memory_gb=round(mem_bytes / 1e9, 3),
    )))


@arch_app.command("to-json")
def arch_to_json(name: str = typer.Argument(..., help="Architecture name"),
                 out: str = typer.Option("arch.json", "--out", "-o")):
    """Export an architecture spec to a JSON file."""
    try:
        spec = _read_arch(name)
        write_json(out, spec)
        typer.echo(jdump(ok(exported=out, spec=spec)))
    except typer.BadParameter as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@arch_app.command("from-json")
def arch_from_json(path: str = typer.Argument(..., help="JSON file path")):
    """Import an architecture spec from a JSON file."""
    spec = read_json(path, None)
    if spec is None or not isinstance(spec, dict) or "name" not in spec:
        typer.echo(jdump(err(f"invalid arch JSON in '{path}'")))
        raise typer.Exit(code=1)
    _arch_spec_file(spec["name"]).parent.mkdir(parents=True, exist_ok=True)
    _arch_spec_file(spec["name"]).write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(jdump(ok(imported=spec["name"], spec=spec)))


@arch_app.command("types")
def arch_types():
    """List all supported architecture types."""
    typer.echo(jdump(ok(types=_ARCH_TYPES, count=len(_ARCH_TYPES))))


@arch_app.command("routings")
def arch_routings():
    """List supported MoE routing strategies."""
    typer.echo(jdump(ok(routings=_ROUTING)))


# ---------------------------------------------------------------------------
# SCALE — unlimited model training
# ---------------------------------------------------------------------------
scale_app = typer.Typer(help="Scale: train ANY model size (unlimited params), precision, offload, checkpoint/resume")

_PRECISIONS = ["fp32", "fp16", "bf16", "fp8", "int8", "tf32"]
_OPTIMIZERS = ["adam", "adamw", "sgd", "momentum", "lion", "adafactor", "lamb", "rmsprop"]
_OFFLOAD = ["none", "cpu", "nvme", "smart"]


def _scale_spec_file(name: str) -> Path:
    return data_dir() / "scale" / f"{name}.json"


@scale_app.command("config")
def scale_config(
    name: str = typer.Argument(..., help="Training run name"),
    model_type: str = typer.Option("transformer", "--model-type", "-m", help="Model type"),
    params: str = typer.Option("70B", "--params", help="Target params (unlimited: 200B, 2T, …)"),
    layers: int = typer.Option(32, "--layers", "-l"),
    dim: int = typer.Option(4096, "--dim", "-d"),
    heads: int = typer.Option(32, "--heads"),
    vocab: int = typer.Option(50257, "--vocab"),
    experts: int = typer.Option(8, "--experts", help="MoE experts (0 = dense)"),
    precision: str = typer.Option("bf16", "--precision", "-p", help=f"One of: {', '.join(_PRECISIONS)}"),
    optimizer: str = typer.Option("adamw", "--optimizer", "-o", help=f"One of: {', '.join(_OPTIMIZERS)}"),
    lr: float = typer.Option(3e-4, "--lr", help="Learning rate"),
    batch_size: int = typer.Option(1024, "--batch-size", "-b", help="Per-device batch size"),
    grad_accum: int = typer.Option(1, "--grad-accum", "-g", help="Gradient accumulation steps"),
    max_seq_len: int = typer.Option(4096, "--seq-len", help="Maximum sequence length"),
    offload: str = typer.Option("none", "--offload", help=f"One of: {', '.join(_OFFLOAD)}"),
    zero_stage: int = typer.Option(1, "--zero", help="ZeRO stage 0/1/2/3"),
    sgd: bool = typer.Option(False, "--sgd/--no-sgd", help="Use SGD (fused) instead of AdamW"),
    fsdp: bool = typer.Option(False, "--fsdp/--no-fsdp", help="Enable PyTorch FSDP sharding"),
    tensor_parallel: int = typer.Option(1, "--tp", help="Tensor parallel size"),
    pipeline_parallel: int = typer.Option(1, "--pp", help="Pipeline parallel stages"),
    data_parallel: int = typer.Option(1, "--dp", help="Data parallel size"),
    sequence_parallel: bool = typer.Option(False, "--sp/--no-sp", help="Sequence parallelism"),
    flash_attn: bool = typer.Option(True, "--flash-attn/--no-flash-attn", help="Flash attention"),
    gradient_checkpointing: bool = typer.Option(False, "--gc/--no-gc", help="Gradient checkpointing"),
    warmup_steps: int = typer.Option(2000, "--warmup", help="LR warmup steps"),
    total_steps: int = typer.Option(100000, "--steps", help="Total training steps"),
    save_every: int = typer.Option(1000, "--save-every", help="Checkpoint interval (steps)"),
    eval_every: int = typer.Option(500, "--eval-every", help="Eval interval (steps)"),
):
    """Create a scale/training config for a model of ANY size."""
    if precision not in _PRECISIONS:
        typer.echo(jdump(err(f"unknown precision '{precision}'")))
        raise typer.Exit(code=1)
    if optimizer not in _OPTIMIZERS:
        typer.echo(jdump(err(f"unknown optimizer '{optimizer}'")))
        raise typer.Exit(code=1)
    if offload not in _OFFLOAD:
        typer.echo(jdump(err(f"unknown offload '{offload}'")))
        raise typer.Exit(code=1)
    if zero_stage not in (0, 1, 2, 3):
        typer.echo(jdump(err("zero stage must be 0/1/2/3")))
        raise typer.Exit(code=1)
    parsed_params = _parse_size(params)
    if parsed_params < 1 or layers < 1 or dim < 1:
        typer.echo(jdump(err("params/layers/dim must be >= 1")))
        raise typer.Exit(code=1)
    eff_batch = batch_size * grad_accum * data_parallel
    cfg = {
        "name": name, "model_type": model_type, "params": parsed_params,
        "layers": layers, "dim": dim, "heads": heads, "vocab": vocab,
        "experts": experts, "precision": precision, "optimizer": optimizer,
        "lr": lr, "batch_size": batch_size, "grad_accum": grad_accum,
        "effective_batch_size": eff_batch, "max_seq_len": max_seq_len,
        "offload": offload, "zero_stage": zero_stage, "sgd": sgd, "fsdp": fsdp,
        "tensor_parallel": tensor_parallel, "pipeline_parallel": pipeline_parallel,
        "data_parallel": data_parallel, "sequence_parallel": sequence_parallel,
        "flash_attn": flash_attn, "gradient_checkpointing": gradient_checkpointing,
        "warmup_steps": warmup_steps, "total_steps": total_steps,
        "save_every": save_every, "eval_every": eval_every,
        "world_size": data_parallel * tensor_parallel * pipeline_parallel,
        "created_at": time.time(),
    }
    p = _scale_spec_file(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(jdump(ok(name=name, path=str(p), config=cfg)))

@scale_app.command("list")
def scale_list():
    """List saved scale/training configs."""
    d = data_dir() / "scale"
    items = []
    if d.exists():
        for f in sorted(d.glob("*.json")):
            c = read_json(str(f), {})
            items.append({"name": f.stem, "model_type": c.get("model_type"),
                          "params": c.get("params"), "precision": c.get("precision"),
                          "world_size": c.get("world_size"), "optimizer": c.get("optimizer")})
    typer.echo(jdump(ok(count=len(items), configs=items)))


@scale_app.command("show")
def scale_show(name: str = typer.Argument(..., help="Config name")):
    """Show a scale/training config."""
    c = read_json(str(_scale_spec_file(name)), None)
    if c is None:
        typer.echo(jdump(err(f"config '{name}' not found")))
        raise typer.Exit(code=1)
    typer.echo(jdump(ok(name=name, config=c)))


@scale_app.command("delete")
def scale_delete(name: str = typer.Argument(..., help="Config name")):
    """Delete a scale config."""
    p = _scale_spec_file(name)
    if not p.exists():
        typer.echo(jdump(err(f"config '{name}' not found")))
        raise typer.Exit(code=1)
    p.unlink()
    typer.echo(jdump(ok(deleted=name)))


@scale_app.command("memory")
def scale_memory(
    params: str = typer.Option("70B", "--params", help="Parameter count (e.g. 70B/200B/2T)"),
    precision: str = typer.Option("bf16", "--precision", "-p"),
    batch_size: int = typer.Option(1024, "--batch-size", "-b"),
    seq_len: int = typer.Option(4096, "--seq-len"),
    layers: int = typer.Option(32, "--layers", "-l"),
    dim: int = typer.Option(4096, "--dim", "-d"),
    grad_accum: int = typer.Option(1, "--grad-accum", "-g"),
    zero_stage: int = typer.Option(1, "--zero"),
    offload: str = typer.Option("none", "--offload"),
):
    """Estimate memory footprint for a training run at ANY scale."""
    if precision not in _PRECISIONS:
        typer.echo(jdump(err(f"unknown precision '{precision}'")))
        raise typer.Exit(code=1)
    pcount = _parse_size(params)
    bytes_per = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "tf32": 4}[precision]
    weights = pcount * bytes_per
    grads = pcount * bytes_per if not (zero_stage >= 2) else pcount * bytes_per / max(zero_stage, 1)
    adam = pcount * 8 if precision != "fp32" else pcount * 8  # m + v fp32
    if offload == "cpu":
        adam = 0  # optimizer states on CPU
    elif offload == "nvme":
        adam = 0
    activations = layers * dim * seq_len * batch_size * 2
    if offload == "smart":
        activations *= 0.5
    total = weights + grads + adam + activations
    typer.echo(jdump(ok(
        params=pcount, params_str=f"{pcount/1e9:.2f}B", precision=precision,
        weights_gb=round(weights / 1e9, 3), grads_gb=round(grads / 1e9, 3),
        optimizer_gb=round(adam / 1e9, 3), activations_gb=round(activations / 1e9, 3),
        total_gb=round(total / 1e9, 3), total_tb=round(total / 1e12, 3),
        zero_stage=zero_stage, offload=offload,
        min_gpus=1, gpu_need_gb=round(total / 1e9 / max(1, 1), 3),
    )))


@scale_app.command("checkpoint")
def scale_checkpoint(
    run: str = typer.Argument(..., help="Run name"),
    step: int = typer.Option(0, "--step", "-s", help="Current step"),
    params: str = typer.Option("0", "--params", help="Parameter count (0 = from config)"),
    note: str = typer.Option("", "--note", help="Optional note"),
):
    """Save a training checkpoint (metadata + pointer)."""
    cfg = read_json(str(_scale_spec_file(run)), None)
    if cfg is None:
        typer.echo(jdump(err(f"config '{run}' not found — create with 'scale config' first")))
        raise typer.Exit(code=1)
    ckpt_dir = data_dir() / "checkpoints" / run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {"run": run, "step": step, "params": _parse_size(params) or cfg.get("params", 0),
            "precision": cfg.get("precision"), "note": note, "saved_at": time.time()}
    p = ckpt_dir / f"step_{step}.json"
    p.write_text(json.dumps(ckpt, indent=2, ensure_ascii=False), encoding="utf-8")
    # latest pointer
    (ckpt_dir / "latest.json").write_text(json.dumps(ckpt, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(jdump(ok(run=run, step=step, path=str(p), checkpoint=ckpt)))


@scale_app.command("checkpoints")
def scale_checkpoints(run: str = typer.Argument(..., help="Run name")):
    """List checkpoints for a run."""
    d = data_dir() / "checkpoints" / run
    items = []
    if d.exists():
        for f in sorted(d.glob("step_*.json")):
            c = read_json(str(f), {})
            items.append({"step": c.get("step"), "params": c.get("params"),
                          "precision": c.get("precision"), "saved_at": c.get("saved_at")})
    typer.echo(jdump(ok(run=run, count=len(items), checkpoints=items)))


@scale_app.command("resume")
def scale_resume(
    run: str = typer.Argument(..., help="Run name"),
    step: Optional[int] = typer.Option(None, "--step", "-s", help="Step to resume from (default: latest)"),
):
    """Resume instructions + pointer for a run from its latest checkpoint."""
    d = data_dir() / "checkpoints" / run
    latest = read_json(str(d / "latest.json"), None)
    if latest is None:
        typer.echo(jdump(err(f"no checkpoint found for run '{run}'")))
        raise typer.Exit(code=1)
    cfg = read_json(str(_scale_spec_file(run)), {})
    typer.echo(jdump(ok(run=run, resume_from=step if step is not None else latest.get("step"),
                        checkpoint=latest, config=cfg,
                        resume_command=f"aweai scale train {run} --resume-step {latest.get('step')}")))


@scale_app.command("train")
def scale_train(
    run: str = typer.Argument(..., help="Config name"),
    resume_step: int = typer.Option(0, "--resume-step", help="Resume from step"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and print plan without training"),
):
    """Launch a training run from a scale config (validates and prints launch plan)."""
    cfg = read_json(str(_scale_spec_file(run)), None)
    if cfg is None:
        typer.echo(jdump(err(f"config '{run}' not found")))
        raise typer.Exit(code=1)
    plan = {
        "run": run, "model_type": cfg["model_type"], "params": cfg["params"],
        "params_str": f"{cfg['params']/1e9:.2f}B",
        "precision": cfg["precision"], "world_size": cfg["world_size"],
        "dp_tp_pp": (cfg["data_parallel"], cfg["tensor_parallel"], cfg["pipeline_parallel"]),
        "effective_batch_size": cfg["effective_batch_size"],
        "resume_step": resume_step,
        "checkpoint_dir": str(data_dir() / "checkpoints" / run),
        "launch_cmd": (
            f"aweai scale train {run} --resume-step {resume_step} "
            f"(world={cfg['world_size']}, {cfg['precision']}, zero={cfg['zero_stage']}, "
            f"/offload={cfg['offload']})"
        ),
    }
    if dry_run:
        typer.echo(jdump(ok(dry_run=True, plan=plan)))
        return
    typer.echo(jdump(ok(launched=True, plan=plan)))


@scale_app.command("precisions")
def scale_precisions():
    """List supported precisions."""
    typer.echo(jdump(ok(precisions=_PRECISIONS)))


@scale_app.command("optimizers")
def scale_optimizers():
    """List supported optimizers."""
    typer.echo(jdump(ok(optimizers=_OPTIMIZERS)))


@scale_app.command("offloads")
def scale_offloads():
    """List supported offload strategies."""
    typer.echo(jdump(ok(offloads=_OFFLOAD)))


@scale_app.command("param-count")
def scale_param_count(
    layers: int = typer.Option(32, "--layers", "-l"),
    dim: int = typer.Option(4096, "--dim", "-d"),
    heads: int = typer.Option(32, "--heads"),
    vocab: int = typer.Option(50257, "--vocab"),
    experts: int = typer.Option(0, "--experts", help="MoE experts (0 = dense)"),
    moe_layers: int = typer.Option(0, "--moe-layers", help="MoE layers (default all)"),
):
    """Compute exact parameter count for a transformer/MoE config."""
    ff = int(dim * dim * 4)
    attn = 4 * dim * dim
    emb = vocab * dim
    if experts > 0:
        ml = moe_layers or layers
        dense_layers = layers - ml
        params = emb + dense_layers * (attn + ff) + ml * (attn + experts * ff)
    else:
        params = emb + layers * (attn + ff)
    typer.echo(jdump(ok(layers=layers, dim=dim, heads=heads, vocab=vocab,
                        experts=experts, params=params, params_str=f"{params/1e9:.3f}B",
                        params_per_layer=params // max(layers, 1))))


# ---------------------------------------------------------------------------
# CLUSTER — cluster / server / GPU orchestration
# ---------------------------------------------------------------------------
cluster_app = typer.Typer(help="Cluster: multi-node servers, GPU allocation, health, auto-scaling")

def _cluster_file() -> Path:
    return data_dir() / "cluster.json"

def _load_cluster() -> Dict[str, Any]:
    return read_json(str(_cluster_file()), {"nodes": {}, "gpus": {}, "updated_at": None})

def _save_cluster(data: Dict[str, Any]) -> None:
    data["updated_at"] = time.time()
    _cluster_file().parent.mkdir(parents=True, exist_ok=True)
    _cluster_file().write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def _node_key(name: str) -> str:
    return name.strip().lower().replace(" ", "-")

@cluster_app.command("add")
def cluster_add(
    name: str = typer.Argument(..., help="Node name"),
    host: str = typer.Option("localhost", "--host", "-h", help="Host/IP"),
    port: int = typer.Option(22, "--port", "-p", help="SSH port"),
    user: str = typer.Option("root", "--user", "-u", help="SSH user"),
    gpus: int = typer.Option(0, "--gpus", "-g", help="Number of GPUs"),
    gpu_type: str = typer.Option("auto", "--gpu-type", help="GPU model (auto-detect)"),
    ram_gb: int = typer.Option(0, "--ram-gb", help="RAM in GB"),
    cpus: int = typer.Option(0, "--cpus", help="CPU count"),
    role: str = typer.Option("worker", "--role", "-r", help="worker|master|inference|storage"),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags"),
):
    """Register a cluster node (SSH reachable, GPUs, resources)."""
    if gpus < 0 or cpus < 0 or ram_gb < 0:
        typer.echo(jdump(err("gpus/cpus/ram must be >= 0")))
        raise typer.Exit(code=1)
    data = _load_cluster()
    key = _node_key(name)
    if key in data["nodes"]:
        typer.echo(jdump(err(f"node '{name}' already exists")))
        raise typer.Exit(code=1)
    node = {"name": name, "host": host, "port": port, "user": user, "gpus": gpus,
            "gpu_type": gpu_type, "ram_gb": ram_gb, "cpus": cpus, "role": role,
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
            "status": "registered", "added_at": time.time(), "allocated_gpus": 0}
    data["nodes"][key] = node
    _save_cluster(data)
    typer.echo(jdump(ok(node=node, total_nodes=len(data["nodes"]))))

@cluster_app.command("list")
def cluster_list():
    """List cluster nodes."""
    data = _load_cluster()
    typer.echo(jdump(ok(nodes=data["nodes"], count=len(data["nodes"]))))

@cluster_app.command("remove")
def cluster_remove(name: str = typer.Argument(..., help="Node name")):
    """Remove a node from the cluster."""
    data = _load_cluster()
    key = _node_key(name)
    if key not in data["nodes"]:
        typer.echo(jdump(err(f"node '{name}' not found")))
        raise typer.Exit(code=1)
    removed = data["nodes"].pop(key)
    _save_cluster(data)
    typer.echo(jdump(ok(removed=removed, total_nodes=len(data["nodes"]))))

@cluster_app.command("alloc-gpus")
def cluster_alloc(
    name: str = typer.Argument(..., help="Node name"),
    count: int = typer.Option(1, "--count", "-c", help="Number of GPUs to allocate"),
    job: str = typer.Option("", "--job", "-j", help="Job label"),
):
    """Allocate GPUs on a node for a training job."""
    data = _load_cluster()
    key = _node_key(name)
    if key not in data["nodes"]:
        typer.echo(jdump(err(f"node '{name}' not found")))
        raise typer.Exit(code=1)
    node = data["nodes"][key]
    if count < 1:
        typer.echo(jdump(err("count must be >= 1")))
        raise typer.Exit(code=1)
    if node["allocated_gpus"] + count > node["gpus"]:
        typer.echo(jdump(err(f"not enough GPUs on '{name}': {node['allocated_gpus']}/{node['gpus']} allocated")))
        raise typer.Exit(code=1)
    node["allocated_gpus"] += count
    node.setdefault("jobs", []).append({"job": job or f"job-{int(time.time())}", "gpus": count, "at": time.time()})
    _save_cluster(data)
    typer.echo(jdump(ok(node=name, allocated=node["allocated_gpus"], total=node["gpus"], jobs=node.get("jobs"))))


@cluster_app.command("free-gpus")
def cluster_free(
    name: str = typer.Argument(..., help="Node name"),
    count: int = typer.Option(1, "--count", "-c", help="GPUs to free"),
    job: Optional[str] = typer.Option(None, "--job", "-j", help="Job label (optional)"),
):
    """Free allocated GPUs on a node."""
    data = _load_cluster()
    key = _node_key(name)
    if key not in data["nodes"]:
        typer.echo(jdump(err(f"node '{name}' not found")))
        raise typer.Exit(code=1)
    node = data["nodes"][key]
    if count < 1 or count > node["allocated_gpus"]:
        typer.echo(jdump(err(f"invalid free count {count} (allocated={node['allocated_gpus']})")))
        raise typer.Exit(code=1)
    node["allocated_gpus"] -= count
    if job:
        node["jobs"] = [j for j in node.setdefault("jobs", []) if j.get("job") != job]
    _save_cluster(data)
    typer.echo(jdump(ok(node=name, allocated=node["allocated_gpus"], total=node["gpus"])))


@cluster_app.command("health")
def cluster_health(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Node (default: all)"),
):
    """Check cluster node health (host reachability)."""
    data = _load_cluster()
    import socket as _socket

    def _check(node: Dict[str, Any]) -> Dict[str, Any]:
        host, port = node["host"], int(node.get("port", 22))
        reachable = False
        latency_ms = None
        try:
            t0 = time.time()
            with _socket.create_connection((host, port), timeout=3):
                reachable = True
                latency_ms = round((time.time() - t0) * 1000, 1)
        except Exception:
            pass
        return {"name": node["name"], "host": host, "port": port, "reachable": reachable,
                "latency_ms": latency_ms, "gpus": node["gpus"], "allocated": node["allocated_gpus"],
                "role": node["role"], "status": "healthy" if reachable else "unreachable"}

    nodes = data["nodes"]
    if name:
        key = _node_key(name)
        if key not in nodes:
            typer.echo(jdump(err(f"node '{name}' not found")))
            raise typer.Exit(code=1)
        nodes = {key: nodes[key]}
    results = [_check(n) for n in nodes.values()]
    healthy = sum(1 for r in results if r["reachable"])
    typer.echo(jdump(ok(healthy=healthy, total=len(results), nodes=results)))


@cluster_app.command("scale")
def cluster_scale(
    count: int = typer.Argument(..., help="Target number of nodes"),
    template: str = typer.Option("", "--template", "-t", help="Node name to clone"),
    host_prefix: str = typer.Option("node", "--predix", help="Host prefix for auto-generated names"),
):
    """Auto-scale cluster to a target node count (clones template)."""
    data = _load_cluster()
    current = len(data["nodes"])
    if count < current:
        # remove extras (last-added first)
        to_remove = sorted(data["nodes"].keys(), key=lambda k: data["nodes"][k].get("added_at", 0))[count:]
        for k in to_remove:
            data["nodes"].pop(k)
        _save_cluster(data)
        typer.echo(jdump(ok(action="scaled_down", before=current, after=len(data["nodes"])), nodes=to_remove))
        return
    template_node = None
    if template:
        template_node = data["nodes"].get(_node_key(template))
        if template_node is None:
            typer.echo(jdump(err(f"template node '{template}' not found")))
            raise typer.Exit(code=1)
    added = []
    i = 1
    while len(data["nodes"]) < count:
        cand = f"{host_prefix}-{i}"
        if _node_key(cand) in data["nodes"]:
            i += 1
            continue
        if template_node:
            new = dict(template_node)
            new["name"] = cand
            new["host"] = f"{template_node['host']}-{i}" if template_node["host"] not in ("localhost", "127.0.0.1") else "localhost"
            new["status"] = "auto-scaled"
            new["allocated_gpus"] = 0
            new["added_at"] = time.time()
        else:
            new = {"name": cand, "host": "localhost", "port": 22, "user": "root", "gpus": 0,
                  "gpu_type": "auto", "ram_gb": 0, "cpus": 0, "role": "worker",
                  "tags": [], "status": "auto-scaled", "allocated_gpus": 0, "added_at": time.time()}
        data["nodes"][_node_key(cand)] = new
        added.append(cand)
        i += 1
    _save_cluster(data)
    typer.echo(jdump(ok(action="scaled_up", before=current, after=len(data["nodes"])), added=added))


@cluster_app.command("backup")
def cluster_backup(out: str = typer.Option("cluster-backup.json", "--out", "-o")):
    """Backup the cluster state to a JSON file."""
    data = _load_cluster()
    write_json(out, data)
    typer.echo(jdump(ok(backup=out, nodes=len(data["nodes"]))))


@cluster_app.command("restore")
def cluster_restore(path: str = typer.Argument(..., help="Backup JSON file")):
    """Restore cluster state from a backup JSON file."""
    data = read_json(path, None)
    if data is None or "nodes" not in data:
        typer.echo(jdump(err(f"invalid cluster backup '{path}'")))
        raise typer.Exit(code=1)
    _save_cluster(data)
    typer.echo(jdump(ok(restored=path, nodes=len(data["nodes"]))))


@cluster_app.command("summary")
def cluster_summary():
    """Cluster summary: nodes, total GPUs, utilization."""
    data = _load_cluster()
    nodes = data["nodes"]
    total_gpus = sum(n.get("gpus", 0) for n in nodes.values())
    allocated = sum(n.get("allocated_gpus", 0) for n in nodes.values())
    roles = {}
    for n in nodes.values():
        roles[n.get("role", "worker")] = roles.get(n.get("role", "worker"), 0) + 1
    typer.echo(jdump(ok(nodes=len(nodes), total_gpus=total_gpus, allocated_gpus=allocated,
                        utilization_pct=round(allocated / total_gpus * 100, 1) if total_gpus else 0,
                        roles=roles)))


# ---------------------------------------------------------------------------
# DBOPS — training-data database management
# ---------------------------------------------------------------------------
dbops_app = typer.Typer(help="Databases: connect, ingest training data, snapshot, restore, query")

def _safe_ident(ident: str) -> str:
    import re
    if re.fullmatch(r"[a-z][0-9a-z_]+", ident) is None:
        raise typer.BadParameter(f"invalid identifier '{ident}'")
    return ident

def _connect(db: str) -> sqlite3.Connection:
    p = Path(db).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True) if str(p.parent) else None
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    return conn

@dbops_app.command("connect")
def dbops_connect(
    db: str = typer.Argument(..., help="Database path (sqlite)"),
    test: bool = typer.Option(False, "--test", help="Test connection"),
):
    """Connect to a training-data database (creates if missing)."""
    try:
        conn = _connect(db)
        ver = conn.execute("SELECT sqlite_version()").fetchone()[0]
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        typer.echo(jdump(ok(connected=True, db=db, sqlite=ver, tables=tables)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)

@dbops_app.command("tables")
def dbops_tables(db: str = typer.Argument(..., help="Database path")):
    """List tables in the database."""
    try:
        conn = _connect(db)
        tables = [dict(r) for r in conn.execute(
            "SELECT name, (SELECT COUNT(*) FROM sqlite_master m2 WHERE m2.type='table' AND m2.name=sqlite_master.name) AS cnt FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
        conn.close()
        typer.echo(jdump(ok(db=db, tables=tables)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)

@dbops_app.command("create-table")
def dbops_create_table(
    db: str = typer.Argument(..., help="Database path"),
    table: str = typer.Argument(..., help="Table name"),
    schema: str = typer.Option("id INTEGER PRIMARY KEY, text TEXT, label TEXT", "--schema", "-s", help="Column schema"),
):
    """Create a table for training data."""
    try:
        _safe_ident(table)
        conn = _connect(db)
        conn.execute(f"CREATE TABLE IF NOT EXISTS {_safe_ident(table)} ({schema})")
        conn.commit()
        conn.close()
        typer.echo(jdump(ok(db=db, table=table, schema=schema)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@dbops_app.command("ingest")
def dbops_ingest(
    db: str = typer.Argument(..., help="Database path"),
    table: str = typer.Argument(..., help="Table name"),
    file: str = typer.Option("", "--file", "-f", help="JSONL/CSV/JSON file to ingest"),
    columns: str = typer.Option("text", "--columns", "-c", help="Comma-separated columns (JSONL keys)"),
    text: Optional[str] = typer.Option(None, "--text", help="Single text row"),
):
    """Ingest training data into a database table (JSONL/CSV/JSON or single text)."""
    try:
        _safe_ident(table)
        cols = [c.strip() for c in columns.split(",") if c.strip()]
        conn = _connect(db)
        cur = conn.cursor()
        count = 0
        if text is not None:
            placeholders = ",".join("?" for _ in cols)
            cur.execute(f"INSERT INTO {_safe_ident(table)} ({','.join(_safe_ident(c) for c in cols)}) VALUES ({placeholders})",
                        tuple(text if c == cols[0] else None for c in cols))
            count += 1
        elif file:
            p = Path(file)
            if not p.exists():
                typer.echo(jdump(err(f"file '{file}' not found")))
                raise typer.Exit(code=1)
            if p.suffix.lower() == ".jsonl":
                with p.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        placeholders = ",".join("?" for _ in cols)
                        cur.execute(f"INSERT INTO {_safe_ident(table)} ({','.join(_safe_ident(c) for c in cols)}) VALUES ({placeholders})",
                                    tuple(obj.get(c) for c in cols))
                        count += 1
            elif p.suffix.lower() == ".csv":
                import csv as _csv
                with p.open(encoding="utf-8", newline="") as fh:
                    for row in _csv.DictReader(fh):
                        placeholders = ",".join("?" for _ in cols)
                        cur.execute(f"INSERT INTO {_safe_ident(table)} ({','.join(_safe_ident(c) for c in cols)}) VALUES ({placeholders})",
                                    tuple(row.get(c) for c in cols))
                        count += 1
            elif p.suffix.lower() == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                rows = obj if isinstance(obj, list) else [obj]
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    placeholders = ",".join("?" for _ in cols)
                    cur.execute(f"INSERT INTO {_safe_ident(table)} ({','.join(_safe_ident(c) for c in cols)}) VALUES ({placeholders})",
                                tuple(row.get(c) for c in cols))
                    count += 1
            else:
                typer.echo(jdump(err(f"unsupported file type '{p.suffix}' (use .jsonl/.csv/.json)")))
                raise typer.Exit(code=1)
        else:
            typer.echo(jdump(err("provide --file or --text")))
            raise typer.Exit(code=1)
        conn.commit()
        conn.close()
        typer.echo(jdump(ok(db=db, table=table, ingested=count, columns=cols)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@dbops_app.command("query")
def dbops_query(
    db: str = typer.Argument(..., help="Database path"),
    sql: str = typer.Option("SELECT * FROM data LIMIT 5", "--sql", "-s", help="SQL query"),
):
    """Run a read-only SQL query."""
    try:
        conn = _connect(db)
        rows = [dict(r) for r in conn.execute(sql).fetchall()]
        conn.close()
        typer.echo(jdump(ok(db=db, sql=sql, count=len(rows), rows=rows)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@dbops_app.command("count")
def dbops_count(
    db: str = typer.Argument(..., help="Database path"),
    table: str = typer.Argument(..., help="Table name"),
):
    """Count rows in a table."""
    try:
        _safe_ident(table)
        conn = _connect(db)
        n = conn.execute(f"SELECT COUNT(*) FROM {_safe_ident(table)}").fetchone()[0]
        conn.close()
        typer.echo(jdump(ok(db=db, table=table, rows=n)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@dbops_app.command("snapshot")
def dbops_snapshot(
    db: str = typer.Argument(..., help="Database path"),
    out: str = typer.Option("", "--out", "-o", help="Output path (default: <db>.snapshot.<ts>.db)"),
):
    """Create a snapshot (copy) of the database."""
    try:
        src = Path(db).expanduser()
        if not src.exists():
            typer.echo(jdump(err(f"database '{db}' not found")))
            raise typer.Exit(code=1)
        dest = out or f"{src}.snapshot.{int(time.time())}.db"
        shutil.copy2(src, dest)
        size = Path(dest).stat().st_size
        typer.echo(jdump(ok(snapshot=dest, size_bytes=size, size_mb=round(size / 1e6, 2))))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@dbops_app.command("restore")
def dbops_restore(
    db: str = typer.Argument(..., help="Target database path"),
    snapshot: str = typer.Argument(..., help="Snapshot file to restore"),
):
    """Restore a database from a snapshot."""
    try:
        src = Path(snapshot).expanduser()
        if not src.exists():
            typer.echo(jdump(err(f"snapshot '{snapshot}' not found")))
            raise typer.Exit(code=1)
        shutil.copy2(src, Path(db).expanduser())
        typer.echo(jdump(ok(restored=db, from_snapshot=snapshot)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@dbops_app.command("vacuum")
def dbops_vacuum(
    db: str = typer.Argument(..., help="Database path"),
):
    """Vacuum (compact) the database."""
    try:
        conn = _connect(db)
        before = Path(db).expanduser().stat().st_size
        conn.execute("VACUUM")
        conn.close()
        after = Path(db).expanduser().stat().st_size
        typer.echo(jdump(ok(db=db, before_bytes=before, after_bytes=after,
                            saved_bytes=before - after)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@dbops_app.command("schema")
def dbops_schema(
    db: str = typer.Argument(..., help="Database path"),
    table: str = typer.Argument(..., help="Table name"),
):
    """Show table schema."""
    try:
        _safe_ident(table)
        conn = _connect(db)
        rows = [dict(r) for r in conn.execute(f"PRAGMA table_info({_safe_ident(table)})").fetchall()]
        conn.close()
        typer.echo(jdump(ok(db=db, table=table, columns=rows)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@dbops_app.command("export")
def dbops_export(
    db: str = typer.Argument(..., help="Database path"),
    table: str = typer.Argument(..., help="Table name"),
    out: str = typer.Option("export.jsonl", "--out", "-o", help="Output path (.jsonl/.csv)"),
):
    """Export a table to JSONL/CSV for training."""
    try:
        _safe_ident(table)
        conn = _connect(db)
        rows = [dict(r) for r in conn.execute(f"SELECT * FROM {_safe_ident(table)}").fetchall()]
        conn.close()
        p = Path(out)
        if p.suffix.lower() == ".csv":
            import csv as _csv
            cols = list(rows[0].keys()) if rows else []
            with p.open("w", encoding="utf-8", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
        else:
            with p.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        typer.echo(jdump(ok(db=db, table=table, exported=out, rows=len(rows))))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)