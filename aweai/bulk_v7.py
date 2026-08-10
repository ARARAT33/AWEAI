# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""AWEAI v4.2 — unlimited model-training bulk command specs.

Adds hundreds of declarative commands for the model-training universe:

  arch          — architecture math: MoE/transformer/RNN/CNN/hybrid helpers
  model-size    — parameter counting & scaling at ANY size (200B..2T+)
  distributed   — data/model/tensor/pipeline parallel & ZeRO helpers
  precision     — FP16/BF16/FP8/INT8 mixed-precision helpers
  sharding      — sharding & offloading helpers (FSDP, NVMe, smart)
  checkpoint    — checkpoint/resume helpers
  gpu           — GPU memory & kernel helpers
  training      — training-loop helpers (LR schedule, grad clip, loss)
  database      — training-data DB helpers (SQLite)
  clusterops    — cluster/GPU orchestration helpers

Every spec is registered into the main bulk registry (aweai.bulk).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import statistics
import struct
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import aweai.bulk as _bulk

S = _bulk.S
spec = _bulk.spec

_OK = _bulk._ok
_ERR = _bulk._err


def _ok(**kw: Any) -> Dict[str, Any]:
    return {"ok": True, **kw}


def _err(msg: str) -> Dict[str, Any]:
    return {"ok": False, "error": msg}


def _floats(s: str) -> List[float]:
    try:
        return [float(x) for x in str(s).replace(" ", ",").split(",") if str(x).strip() != ""]
    except Exception:
        return []


def _ints(s: str) -> List[int]:
    try:
        return [int(x) for x in str(s).replace(" ", ",").split(",") if str(x).strip() != ""]
    except Exception:
        return []


def _parse_size(s: Any) -> int:
    """Parse '700M' / '2B' / '2T' / 70000000000 into an integer."""
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


def _fmt_params(n: int) -> str:
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def _transformer_params(layers: int, dim: int, vocab: int, ff_mult: float = 4.0,
                        heads: Optional[int] = None, kv_heads: Optional[int] = None,
                        experts: int = 0, moe_layers: Optional[int] = None) -> Dict[str, Any]:
    heads = heads or max(1, dim // 64)
    kv_heads = kv_heads or heads
    ff = int(dim * dim * ff_mult)
    attn = 4 * dim * dim if kv_heads >= heads else (dim * dim + 2 * dim * (dim // kv_heads))
    emb = vocab * dim
    if experts > 0:
        ml = moe_layers or layers
        dense = layers - ml
        params = emb + dense * (attn + ff) + ml * (attn + experts * ff)
        moe_pct = round(ml / layers * 100, 1) if layers else 0
    else:
        params = emb + layers * (attn + ff)
        moe_pct = 0.0
    return {"params": params, "params_str": _fmt_params(params),
            "embedding": emb, "per_layer": attn + ff,
            "attention": attn, "ffn": ff, "moe_pct": moe_pct}


# ===========================================================================
# ARCH group — architecture math helpers
# ===========================================================================

spec("architect", "types", "List supported architecture families.",
     [], lambda p: _ok(types=["moe", "transformer", "rnn", "lstm", "gru", "cnn", "mlp",
                              "linear", "autoencoder", "gan", "hybrid", "mamba"],
                       count=12))
spec("architect", "moe-params", "Parameter count for a MoE config.",
     [("layers", 32, "Layers"), ("dim", 4096, "Hidden dim"), ("vocab", 50257, "Vocab"),
      ("experts", 8, "Experts"), ("moe_layers", None, "MoE layers (None=all)"),
      ("ff_mult", 4.0, "FFN multiplier")],
     lambda p: _ok(**_transformer_params(int(p["layers"]), int(p["dim"]), int(p["vocab"]),
                                          float(p["ff_mult"]), experts=int(p["experts"]),
                                          moe_layers=int(p["moe_layers"]) if p["moe_layers"] else None)))
spec("architect", "dense-params", "Parameter count for a dense transformer.",
     [("layers", 32, "Layers"), ("dim", 4096, "Hidden dim"), ("vocab", 50257, "Vocab"),
      ("ff_mult", 4.0, "FFN multiplier"), ("heads", None, "Heads (default dim//64)")],
     lambda p: _ok(**_transformer_params(int(p["layers"]), int(p["dim"]), int(p["vocab"]),
                                          float(p["ff_mult"]), heads=int(p["heads"]) if p["heads"] else None)))
spec("architect", "rnn-params", "Parameter count for an RNN/LSTM/GRU stack.",
     [("layers", 4, "Layers"), ("input", 512, "Input dim"), ("hidden", 1024, "Hidden dim"),
      ("type", "lstm", "rnn|lstm|gru"), ("vocab", 50257, "Vocab (embedding)")],
     lambda p: _rnn_params(p))
spec("architect", "cnn-params", "Parameter count for a CNN stack.",
     [("channels", "64,128,256", "Channel sizes (comma)"), ("kernels", "3,3,3", "Kernel sizes"),
      ("input_channels", 3, "Input channels"), ("input_hw", 224, "Input HxW")],
     lambda p: _cnn_params(p))
spec("architect", "hybrid-pct", "Dense/MoE/SSM split percentage for hybrid arch.",
     [("layers", 48, "Total layers"), ("dense", 16, "Dense layers"),
      ("moe", 16, "MoE layers"), ("ssm", 16, "SSM/Mamba layers")],
     lambda p: _ok(dense_pct=round(int(p["dense"])/int(p["layers"])*100, 1) if int(p["layers"]) else 0,
                   moe_pct=round(int(p["moe"])/int(p["layers"])*100, 1) if int(p["layers"]) else 0,
                   ssm_pct=round(int(p["ssm"])/int(p["layers"])*100, 1) if int(p["layers"]) else 0))
spec("architect", "heads-for-dim", "Suggested attention heads for a dim.",
     [("dim", 4096, "Hidden dim"), ("head_dim", 64, "Head dim")],
     lambda p: _ok(heads=max(1, int(p["dim"]) // int(p["head_dim"])), head_dim=int(p["head_dim"])))
spec("architect", "recommend-dim", "Recommend dim/layers for a target param budget.",
     [("params", "70B", "Target size (e.g. 70B / 200B / 2T)"), ("vocab", 50257, "Vocab")],
     lambda p: _arch_recommend(p))


def _rnn_params(p: Dict[str, Any]) -> Dict[str, Any]:
    layers, inp, hid, typ, vocab = int(p["layers"]), int(p["input"]), int(p["hidden"]), p["type"], int(p["vocab"])
    gate = {"rnn": 1, "gru": 3, "lstm": 4}[typ if typ in ("rnn", "gru", "lstm") else "lstm"]
    emb = vocab * inp
    per = gate * (inp * hid + hid * hid + hid)
    return _ok(type=typ, layers=layers, params=emb + layers * per,
               params_str=_fmt_params(emb + layers * per), per_layer=per)


def _cnn_params(p: Dict[str, Any]) -> Dict[str, Any]:
    chans = _ints(p["channels"]); kerns = _ints(p["kernels"])
    if not chans:
        return _err("channels required")
    kerns = (kerns + [3] * len(chans))[:len(chans)]
    total = 0; prev = int(p["input_channels"])
    out_h = int(p["input_hw"])
    for c, k in zip(chans, kerns):
        total += prev * c * k * k + c
        prev = c
        out_h = (out_h - k) // 2 + 1 if out_h > k else 1
    return _ok(params=total, params_str=_fmt_params(total),
               layers=len(chans), output_hw=out_h)


def _arch_recommend(p: Dict[str, Any]) -> Dict[str, Any]:
    target = _parse_size(p["params"])
    vocab = int(p["vocab"])
    if target <= 0:
        return _err("invalid params")
    # heuristic: dim^2 * 8 * layers ~ params (dense)
    for layers in (8, 12, 16, 24, 32, 48, 64, 80, 96, 128):
        dim = int((target / (layers * 8)) ** 0.5)
        if dim >= 256:
            est = _transformer_params(layers, dim, vocab)
            return _ok(target=target, target_str=p["params"], layers=layers, dim=dim,
                       heads=max(1, dim // 64), est_params=est["params"],
                       est_str=est["params_str"])
    return _ok(target=target, layers=128, dim=512, heads=8)


# ===========================================================================
# MODEL-SIZE group — any-size model math
# ===========================================================================

spec("model-size", "parse", "Parse a size string (700M/2B/2T/70000000000).",
     [("size", "70B", "Size string")], lambda p: _ok(size=p["size"], params=_parse_size(p["size"])))
spec("model-size", "format", "Format params as human string.",
     [("params", 200000000000, "Param count")], lambda p: _ok(params=int(p["params"]),
                                                                formatted=_fmt_params(int(p["params"]))))
spec("model-size", "per-token", "Compute params x tokens compute (FLOPs).",
     [("params", "70B", "Params"), ("tokens", "300B", "Training tokens")],
     lambda p: _ok(flops=6 * _parse_size(p["params"]) * _parse_size(p["tokens"]),
                   flops_str=f"{6*_parse_size(p['params'])*_parse_size(p['tokens'])/1e21:.3f} ZFLOPs"))
spec("model-size", "chinchilla", "Optimal tokens for params (Chinchilla ~20 tokens/param).",
     [("params", "70B", "Params")],
     lambda p: _ok(params=int(_parse_size(p["params"])),
                   recommended_tokens=int(_parse_size(p["params"]) * 20),
                   recommended_tokens_str=_fmt_params(_parse_size(p["params"]) * 20)))
spec("model-size", "kaplan", "Compute-optimal params for a FLOPs budget (Kaplan).",
     [("flops", 1e23, "FLOPs budget")],
     lambda p: _ok(flops=float(p["flops"]),
                   optimal_params=int((float(p["flops"]) / 6) ** 0.5),
                   optimal_params_str=_fmt_params(int((float(p["flops"]) / 6) ** 0.5))))
spec("model-size", "layers-from-depth", "Total layers for a model depth budget.",
     [("depth", 128, "Total depth"), ("tower", 64, "Tower depth (encoder/decoder)")],
     lambda p: _ok(layers=int(p["depth"]), tower=int(p["tower"]),
                   encoder_layers=int(p["tower"]), decoder_layers=int(p["depth"]) - int(p["tower"])))
spec("model-size", "embedding-mem", "Embedding memory at precision.",
     [("vocab", 50257, "Vocab"), ("dim", 4096, "Dim"), ("bytes", 2, "Bytes per param (2=bf16,4=fp32)")],
     lambda p: _ok(mem_bytes=int(p["vocab"]) * int(p["dim"]) * int(p["bytes"]),
                   mem_gb=round(int(p["vocab"]) * int(p["dim"]) * int(p["bytes"]) / 1e9, 3)))
spec("model-size", "weights-mem", "Weights memory at precision.",
     [("params", "70B", "Params"), ("bytes", 2, "Bytes per param")],
     lambda p: _ok(mem_gb=round(_parse_size(p["params"]) * int(p["ybtes"]) / 1e9, 3),
                   mem_tb=round(_parse_size(p["params"]) * int(p["bytes"]) / 1e12, 3)))
spec("model-size", "compare", "Compare two model sizes.",
     [("a", "70B", "Size A"), ("b", "405B", "Size B")],
     lambda p: _ok(a=_parse_size(p["a"]), b=_parse_size(p["b"]),
                   ratio=round(_parse_size(p["b"]) / max(_parse_size(p["a"]), 1), 2),
                   delta=_parse_size(p["b"]) - _parse_size(p["a"])))
spec("model-size", "gpu-count", "Min GPUs to fit model at precision.",
     [("params", "70B", "Params"), ("gpu_gb", 80, "GPU memory GB"), ("bytes", 2, "Bytes per param")],
     lambda p: _gpu_count(p))
spec("model-size", "tokens-per-day", "Throughput: tokens/day for a GPU cluster.",
     [("gpus", 1000, "GPU count"), ("tokens_per_gpu_per_sec", 20000, "Tokens/sec/GPU"),
      ("uptime", 0.9, "Uptime ratio")],
     lambda p: _ok(tokens_per_day=int(int(p["gpus"]) * float(p["tokens_per_gpu_per_sec"]) * 86400 * float(p["uptime"])),
                   tokens_str=_fmt_params(int(int(p["gpus"]) * float(p["tokens_per_gpu_per_sec"]) * 86400 * float(p["uptime"])))))

def _gpu_count(p: Dict[str, Any]) -> Dict[str, Any]:
    params = _parse_size(p["params"])
    gpu_bytes = int(p["gpu_gb"]) * 1e9
    mem = params * int(p["bytes"])
    count = math.ceil(mem / gpu_bytes)
    return _ok(params=params, mem_gb=round(mem / 1e9, 3), gpu_gb=int(p["gpu_gb"]),
               min_gpus=max(count, 1), fits_in_one=mem <= gpu_bytes)

# ===========================================================================
# DISTRIBUTED group — parallelism helpers
# ===========================================================================

spec("distributed", "world-size", "World size from dp/tp/pp.",
     [("dp", 8, "Data parallel"), ("tp", 8, "Tensor parallel"), ("pp", 1, "Pipeline parallel")],
     lambda p: _ok(world_size=int(p["dp"]) * int(p["tp"]) * int(p["pp"]),
                   dp=int(p["dp"]), tp=int(p["tp"]), pp=int(p["pp"])))
spec("distributed", "effective-batch", "Effective batch size.",
     [("er_gpu", 1024, "Per-GPU batch"), ("grad_accum", 1, "Grad accumulation"), ("dp", 8, "Data parallel")],
     lambda p: _ok(effective=int(p["per_gpu"]) * int(p["grad_accum"]) * int(p["dp"])))
spec("distributed", "zero-mem", "ZeRO memory savings by stage.",
     [("params", "70B", "Params"), ("stage", 3, "ZeRO stage 0-3"), ("bytes", 2, "Bytes per param"),
      ("gpus", 8, "GPU count")],
     lambda p: _zero_mem(p))
spec("distributed", "tp-comm", "Tensor-parallel communication volume per layer.",
     [("dim", 4096, "Dim"), ("seq", 2048, "Seq len"), ("tp", 8, "TP size")],
     lambda p: _ok(comm_bytes=int(p["dim"]) * int(p["seq"]) * 2 * 2 * int(p["tp"]),
                   comm_mb=round(int(p["dim"]) * int(p["seq"]) * 2 * 2 * int(p["tp"]) / 1e6, 2)))
spec("distributed", "dp-vs-tp", "Which parallel strategy for a world.",
     [("world", 64, "World size"), ("params", "70B", "Params"), ("gpu_gb", 80, "GPU GB"],
     lambda p: _dp_vs_tp(p))
spec("distributed", "ring-time", "All-reduce Ring time estimate.",
     [("size_mb", 1024, "Payload MB"), ("bandwidth_gbps", 400, "Link Gbps"), ("nodes", 8, "Nodes")],
     lambda p: _ok(time_ms=round(int(p["size_mb"]) * 8 / (float(p["bandwidth_gbps"]) * 1e9 / 1e3) * int(p["nodes"]), 3)))
spec("distributed", "node-partition", "Partition layers across nodes.",
     [("layers", 96, "Layers"), ("nodes", 8, "Nodes")],
     lambda p: _ok(per_node=int(p["layers"]) // int(p["nodes"]),
                   remainder=int(p["layers"]) % int(p["nodes"])))
spec("distributed", "micro-batches", "Micro-batches for pipeline fill.",
     [("global_batch", 2048, "Global batch"), ("micro", 128, "Micro batch")],
     lambda p: _ok(micro_batches=max(1, int(p["global_batch"]) // max(int(p["micro"]), 1)),
                   pipeline_bubbles=round((int(p["global_batch"]) // max(int(p["micro"]), 1)) / 4.0, 1)))
spec("distributed", "recommend", "Recommend parallel strategy from world+model.",
     [("world", 64, "World size"), ("params", "70B", "Params"), ("gpu_gb", 80, "GPU GB")],
     lambda p: _dp_vs_tp(p))

def _zero_mem(p: Dict[str, Any]) -> Dict[str, Any]:
    params = _parse_size(p["params"]); stage = int(p["stage"]); gpus = max(int(p["gpus"]), 1)
    b = int(p["bytes"])
    weights = params * b
    grads = params * b / (gpus if stage >= 2 else 1)
    optim = params * 8 / (gpus if stage >= 3 else 1)
    total = weights + grads + optim
    return _ok(params=params, stage=stage, gpus=gpus,
               weights_gb=round(weights / 1e9, 3),
               grads_gb=round(grads / 1e9, 3),
               optimizer_gb=round(optim / 1e9, 3),
               total_per_gpu_gb=round(total / 1e9 / (gpus if stage >= 2 else 1), 3),
               savings_pct=round((1 - total / max(weights * 3, 1)) * 100, 1))

def _dp_vs_tp(p: Dict[str, Any]) -> Dict[str, Any]:
    world = int(p["world"]); params = _parse_size(p["params"])
    gpu_bytes = int(p["gpu_gb"]) * 1e9
    bytes_per = 2
    fits_tp1 = params * bytes_per * 3 <= gpu_bytes  # weights+grads+optim approx
    if fits_tp1:
        return _ok(strategy="data_parallel", dp=world, tp=1, pp=1, reason="model fits on one GPU")
    tp = max(1, math.ceil(params * bytes_per * 3 / gpu_bytes / 4))
    tp = min(tp, world)
    pp = 1
    if params * bytes_per * 3 / (tp * gpu_bytes) > 1:
        pp = max(1, math.ceil(params * bytes_per * 3 / (tp * gpu_bytes)))
    dp = max(1, world // (tp * pp))
    return _ok(strategy="hybrid", dp=dp, tp=tp, pp=pp,
               reason=f"model needs sharding: tp={tp}, pp={pp}, dp={dp}")

# ===========================================================================
# PRECISION group — mixed-precision helpers
# ===========================================================================

spec("precision", "types", "List supported precisions.",
     [], lambda p: _ok(precisions=["fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4"],
                       bytes_per={"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}))
spec("precision", "convert", "Convert a float to target precision bits.",
     [("value", 0.123456789, "Value"), ("bits", 16, "Bits (8/16/32)")],
     lambda p: _precision_convert(p))
spec("precision", "save", "Memory saved vs fp32.",
     [("params", "70B", "Params"), ("from_prec", "fp32", "From precision"), ("to_prec", "bf16", "To precision")],
     lambda p: _precision_save(p))
spec("precision", "loss-scale", "Dynamic loss scaling policy.",
     [("init", 65536, "Initial scale"), ("growth", 2.0, "Growth factor"), ("backoff", 0.5, "Backoff"),
      ("threshold", 2000, "Steps between growth")],
     lambda p: _ok(policy="dynamic", init_scale=float(p["init"]), growth=float(p["growth"]),
                   backoff=float(p["backoff"]), overflow_steps=int(p["threshold"])))
spec("precision", "overflow", "Check if value overflows target bits.",
     [("value", 65504.0, "Value"), ("bits", 16, "Bits"), ("signed", True, "Signed")],
     lambda p: _precision_overflow(p))
spec("precision", "range", "Numeric range of a precision.",
     [("bits", 16, "Bits"), ("signed", True, "Signed")],
     lambda p: _precision_range(p))
spec("precision", "fp8-scale", "FP8 scaling for a tensor.",
     [("values", "-1,0.5,2", "Values")],
     lambda p: _fp8_scale(p))
spec("precision", "recommend", "Recommended precision for a task.",
     [("task", "training", "training|inference|edge|quantization")],
     lambda p: _ok(recommendation={
         "training": "bf16 (best stability) or fp16 with loss scaling; fp8 for accelerator with native support",
         "inference": "fp16/bf16 on GPU, int8 on CPU, int4 for edge",
         "edge": "int8/int4 quantized with calibration",
         "quantization": "fp32 master weights + int8/bfloat16 compute"}[p["task"] if p["task"] in ("training", "inference", "edge", "quantization") else "training"]),)

def _precision_convert(p: Dict[str, Any]) -> Dict[str, Any]:
    import struct as _st
    val = float(p["value"]); bits = int(p["bits"])
    if bits == 32:
        return _ok(bits=32, value=val, quantized=val)
    if bits == 16:
        # IEEE 754 half via struct (may overflow)
        try:
            q = _st.unpack("e", _st.pack("e", val))[0]
            return _ok(bits=16, value=val, quantized=round(float(q), 6))
        except Exception:
            return _ok(bits=16, value=val, quantized=round(val, 4), note="approx")
    if bits == 8:
        q = max(-128, min(127, round(val * 127)))
        return _ok(bits=8, value=val, quantized=q, scale=round(q / 127 if q else 0, 6))
    return _ok(bits=bits, value=val, quantized=val)

def _precision_save(p: Dict[str, Any]) -> Dict[str, Any]:
    b = {"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
    f = b.get(p["per_prec"], 4); t = b.get(p["to_prec"], 2)
    params = _parse_size(p["params"])
    return _ok(params=params, from_bytes=f, to_bytes=t,
               saved_gb=round(params * (f - t) / 1e9, 3),
               saved_pct=round((1 - t / f) * 100, 1))

def _precision_overflow(p: Dict[str, Any]) -> Dict[str, Any]:
    val = float(p["value"]); bits = int(p["bits"])
    if bits == 16:
        max_v = 65504.0
        return _ok(overflows=abs(val) > max_v, max_value=max_v)
    if bits == 8:
        max_v = 127 if p["signed"] else 255
        return _ok(overflows=abs(val) > max_v, max_value=max_v)
    return _ok(overflows=False, max_value=3.4e38)

def _precision_range(p: Dict[str, Any]) -> Dict[str, Any]:
    bits = int(p["bits"]); signed = p["signed"]
    if bits == 16:
        return _ok(min=-65504.0, max=65504.0, smallest_normal=6.1e-5, note="IEEE half")
    if bits == 8:
        return _ok(min=-128 if signed else 0, max=127 if signed else 255, note="integer")
    if bits == 32:
        return _ok(min=-3.4e38, max=3.4e38, note="IEEE single")
    return _ok(min=0, max=2 ** bits - 1, note="integer")

def _fp8_scale(p: Dict[str, Any]) -> Dict[str, Any]:
    vals = _floats(p["values"])
    if not vals:
        return _err("no values")
    amax = max(abs(v) for v in vals)
    scale = amax / 448.0 if amax else 1.0  # fp8 e4m3 max ~448
    return _ok(scale=round(scale, 8), amax=amax,
               quantized=[max(-448, min(448, round(v / scale))) for v in vals])

# ===========================================================================
# SHARDING group — FSDP / offload / sharding helpers
# ===========================================================================

spec("sharding", "fsdp-shards", "FSDP shards needed for a model.",
     [("params", "70B", "Params"), ("gpu_gb", 80, "GPU GB"), ("bytes", 2, "Bytes per param")],
     lambda p: _fsdp_shards(p))
spec("sharding", "offload-mem", "Memory with CPU/NVMe offload.",
     [("params", "70B", "Params"), ("offload", "cpu", "none|cpu|nvme|smart"),
      ("gpu_gb", 80, "GPU GB"), ("gpus", 8, "GPU count")],
     lambda p: _offload_mem(p))
spec("sharding", "activation-checkpoint", "Activation memory saved by gradient checkpointing.",
     [("layers", 32, "Layers"), ("dim", 4096, "Dim"), ("seq", 2048, "Seq"), ("batch", 1024, "Batch"),
      ("bytes", 2, "Bytes per element")],
     lambda p: _act_ckpt(p))
spec("sharding", "tensor-shard", "Shard a tensor across N devices.",
     [("size", 4096, "Tensor size"), ("devices", 8, "Devices")],
     lambda p: _ok(per_device=int(p["size"]) // int(p["devices"]),
                   remainder=int(p["size"]) % int(p["devices"])))
spec("sharding", "expert-shard", "Experts per device for MoE sharding.",
     [("experts", 256, "Total experts"), ("devices", 32, "Devices")],
     lambda p: _ok(experts_per_device=max(1, int(p["experts"]) // int(p["devices"])),
                   remainder=int(p["experts"]) % int(p["devices"])))

def _fsdp_shards(p: Dict[str, Any]) -> Dict[str, Any]:
    params = _parse_size(p["params"]); gpu_bytes = int(p["gpu_gb"]) * 1e9
    mem = params * int(p["bytes"])
    shards = max(1, math.ceil(mem / gpu_bytes))
    return _ok(params=params, mem_gb=round(mem / 1e9, 3), gpu_gb=int(p["gpu_gb"]),
               shards=shards, fits=shards == 1)

def _offload_mem(p: Dict[str, Any]) -> Dict[str, Any]:
    params = _parse_size(p["params"]); gpus = max(int(p["gpus"]), 1)
    b = int(p["bytes"])
    weights = params * b / gpus
    if p["offload"] == "none":
        total = weights * 3  # + grads + optim
    elif p["offload"] == "cpu":
        total = weights * 1.2  # only weights + small buffer
    elif p["offload"] == "nvme":
        total = weights * 1.05
    else:  # smart
        total = weights * 1.5
    return _ok(offload=p["offload"], gpus=gpus,
               per_gpu_gb=round(total / 1e9, 3),
               total_gb=round(total * gpus / 1e9, 3))

def _act_ckpt(p: Dict[str, Any]) -> Dict[str, Any]:
    full = int(p["layers"]) * int(p["dim"]) * int(p["seq"]) * int(p["batch"]) * int(p["bytes"])
    saved = full - (int(p["dim"]) * int(p["seq"]) * int(p["batch"]) * int(p["bytes"]))
    return _ok(full_gb=round(full / 1e9, 3), checkpointed_gb=round(saved / 1e9, 3),
               saved_gb=round((full - saved) / 1e9, 3),
               saved_pct=round((1 - saved / max(full, 1)) * 100, 1))

# ===========================================================================
# CHECKPOINT group — checkpoint/resume helpers
# ===========================================================================

spec("checkpoint", "plan", "Checkpoint plan for a run.",
     [("ceckpoint", "plan", "Checkpoint plan for a run.")],
     lambda p: _ok())
spec("checkpoint", "size", "Checkpoint file size estimate.",
     [("params", "70B", "Params"), ("bytes", 2, "Bytes per param"), ("optimizer", True, "Include optimizer")],
     lambda p: _ok(weights_gb=round(_parse_size(p["params"]) * int(p["bytes"]) / 1e9, 3),
                   optimizer_gb=round(_parse_size(p["params"]) * 8 / 1e9, 3) if p["optimizer"] else 0,
                   total_gb=round(_parse_size(p["params"]) * (int(p["bytes"]) + (8 if p["optimizer"] else 0)) / 1e9, 3)))
spec("checkpoint", "shard-count", "Checkpoint shards for FSDP.",
     [("world", 64, "World size"), ("shard_every", 8, "Shards per rank")],
     lambda p: _ok(shards=int(p["world"]) * int(p["shard_every"])))
spec("checkpoint", "eta", "Estimated time remaining.",
     [("step", 40000, "Current step"), ("total", 100000, "Total steps"), ("seconds_per_step", 0.35, "Sec/step")],
     lambda p: _ok(steps_left=int(p["total"]) - int(p["step"]),
                   seconds_left=round((int(p["total"]) - int(p["step"])) * float(p["seconds_per_step"]), 1),
                   hours_left=round((int(p["total"]) - int(p["step"])) * float(p["seconds_per_step"]) / 3600, 2)))
spec("checkpoint", "resume-info", "What to restore from a checkpoint.",
     [("optimizer", True, "Optimizer state"), ("lr", True, "LR schedule"), ("rng", True, "RNG state"),
      ("data", True, "Data loader state")],
     lambda p: _ok(restore=["optimizer", "lr_scheduler", "rng", "dataloader"] if p["optimizer"] else ["weights"],
                   note="resume restores weights + optional optimizer/lr/rng/dataloader"))

# ===========================================================================
# GPU group — GPU helpers
# ===========================================================================

spec("gpu", "mem-usage", "Memory breakdown for a model on GPU.",
     [("params", "70B", "Params"), ("batch", 1024, "Batch"), ("seq", 2048, "Seq"),
      ("layers", 32, "Layers"), ("dim", 4096, "Dim"), ("bytes", 2, "Bytes per param")],
     lambda p: _gpu_mem(p))
spec("gpu", "tf32-note", "TF32 compute capability.",
     [], lambda p: _ok(tf32_supported="Ampere+ (A100/H100), RTX 30/40 series", note="Use for matmul with 10x fp32 range, ~8x speed vs fp32"))
spec("gpu", "cuda-cores", "Estimate CUDA cores from SM count.",
     [("sms", 132, "SM count"), ("cores_per_sm", 128, "Cores per SM")],
     lambda p: _ok(cuda_cores=int(p["sms"]) * int(p["cores_per_sm"])))
spec("gpu", "bandwidth", "Transfer time for tensor over NVLink/PCIe.",
     [("gb", 10, "Size GB"), ("gbps", 900, "Bandwidth GB/s")],
     lambda p: _ok(seconds=round(float(p["gb"]) / float(p["gbps"]), 3),
                   ms=round(float(p["gb"]) / float(p["gbps"]) * 1000, 1)))
spec("gpu", "util-check", "Check GPU utilization thresholds.",
     [("util", 85, "Util %"), ("warn", 70, "Warn below"), ("crit", 40, "Critical below")],
     lambda p: _ok(level="ok" if int(p["util"]) >= int(p["warn"]) else ("warn" if int(p["util"]) >= int(p["crit"]) else "critical"),
                   util=int(p["util"])))

def _gpu_mem(p: Dict[str, Any]) -> Dict[str, Any]:
    params = _parse_size(p["params"]); b = int(p["bytes"])
    weights = params * b
    grads = params * b
    optim = params * 8
    acts = int(p["layers"]) * int(p["dim"]) * int(p["seq"]) * int(p["batch"]) * 2
    total = weights + grads + optim + acts
    return _ok(weights_gb=round(weights / 1e9, 3), grads_gb=round(grads / 1e9, 3),
               optimizer_gb=round(optim / 1e9, 3), activations_gb=round(acts / 1e9, 3),
               total_gb=round(total / 1e9, 3))

# ===========================================================================
# TRAINING group — training-loop helpers
# ===========================================================================

spec("training", "lr-schedule", "Learning rate at step (cosine/warmup).",
     [("step", 5000, "Step"), ("total", 100000, "Total"), ("base", 3e-4, "Base LR"),
      ("warmup", 2000, "Warmup steps"), ("min", 3e-5, "Min LR")],
     lambda p: _lr_schedule(p))
spec("training", "grad-clip", "Gradient clipping norm.",
     [("grads", "0.1,-0.5,2.0,1.5", "Gradients"), ("max_norm", 1.0, "Max norm")],
     lambda p: _grad_clip(p))
spec("training", "loss", "Cross-entropy / MSE loss.",
     [("loss_type", "ce", "ce|mse|mae|huber"), ("pred", "0.1,0.7,0.2", "Predictions"), ("label", "1", "Label index or value")],
     lambda p: _loss(p))
spec("training", "ppl", "Perplexity from loss.",
     [("loss", 1.5, "CE loss")], lambda p: _ok(ppl=round(math.exp(float(p["loss"])), 4)))
spec("training", "flops-per-step", "FLOPs per training step.",
     [("flops-budget", "60FLOPs", "FLOPs budget")],
     lambda p: _ok(flops="60FLOPs", flops_str="60FLOPs"))
spec("training", "steps-for-tokens", "Steps to process N tokens.",
     [("flops-budget", "60FLOPs", "FLOPs budget")],
     lambda p: _ok(flops="60FLOPs"))
spec("training", "throughput", "Training throughput tokens/sec.",
     [("gpus", 1000, "GPUs"), ("tps", 20000, "Tokens/sec/GPU")],
     lambda p: _ok(tokens_per_sec=int(p["gpus"]) * int(p["tps"]),
                   tokens_per_day=_fmt_params(int(p["gpus"]) * int(p["tps"]) * 86400)))
spec("training", "epochs", "Epochs from steps and dataset size.",
     [("steps", 100000, "Steps"), ("batch", 1024, "Batch"), ("dataset", 1000000, "Dataset size")],
     lambda p: _ok(epochs=round(int(p["steps"]) * int(p["batch"]) / max(int(p["dataset"]), 1), 2)))
spec("training", "warmup-lr", "Linear warmup LR.",
     [("step", 1000, "Step"), ("warmup", 2000, "Warmup"), ("base", 3e-4, "Base LR")],
     lambda p: _ok(lr=round(float(p["base"]) * min(1.0, int(p["step"]) / max(int(p["warmup"]), 1)), 8)))
spec("training", "batch-tokens", "Tokens in a batch.",
     [("batch", 1024, "Batch"), ("seq", 2048, "Seq")],
     lambda p: _ok(tokens=int(p["batch"]) * int(p["seq"]))))

def _lr_schedule(p: Dict[str, Any]) -> Dict[str, Any]:
    step = int(p["step"]); total = max(int(p["total"]), 1); warmup = int(p["warmup"])
    base = float(p["base"]); mn = float(p["min"])
    if step < warmup:
        lr = base * step / max(warmup, 1)
    else:
        prog = (step - warmup) / max(total - warmup, 1)
        lr = mn + 0.5 * (base - mn) * (1 + math.cos(math.pi * min(prog, 1.0)))
    return _ok(step=step, lr=round(lr, 8))

def _grad_clip(p: Dict[str, Any]) -> Dict[str, Any]:
    grads = _floats(p["grads"])
    if not grads:
        return _err("no gradients")
    norm = math.sqrt(sum(g * g for g in grads))
    max_norm = float(p["max_norm"])
    if norm <= max_norm:
        return _ok(norm=round(norm, 4), clipped=False, max_norm=max_norm)
    scale = max_norm / norm
    return _ok(norm=round(norm, 4), clipped=True, max_norm=max_norm,
               scaled=[round(g * scale, 6) for g in grads])

def _loss(p: Dict[str, Any]) -> Dict[str, Any]:
    lt = p["loss_type"]
    if lt == "ce":
        preds = _floats(p["pred"])
        if not preds:
            return _err("no predictions")
        import math as _m
        z = sum(_m.exp(x) for x in preds)
        label = int(p["label"])
        if label < 0 or label >= len(preds):
            return _err("label out of range")
        ce = -_m.log(_m.exp(preds[label]) / z)
        return _ok(loss=round(ce, 6), loss_type="cross_entropy")
    pairs = _floats(p["pred"])
    label = float(p["label"])
    if not pairs:
        return _err("no predictions")
    if lt == "mse":
        return _ok(loss=round((pairs[0] - label) ** 2, 6), loss_type="mse")
    if lt == "mae":
        return _ok(loss=round(abs(pairs[0] - label), 6), loss_type="mae")
    if lt == "huber":
        d = abs(pairs[0] - label); delta = 1.0
        return _ok(loss=round((0.5 * d * d if d <= delta else delta * (d - 0.5 * delta)), 6), loss_type="huber")
    return _err("unknown loss type")


# ===========================================================================
# DATABASE group — training-data DB helpers
# ===========================================================================

spec("database", "create", "Create SQLite training-data DB with schema.",
     [("path", "train.db", "DB path"), ("table", "data", "Table"), ("schema", "id INTEGER PRIMARY KEY, text TEXT, label TEXT", "Schema")],
     lambda p: _db_create(p))
spec("database", "tables", "List tables in a DB.",
     [("path", "train.db", "DB path")], lambda p: _db_tables(p))
spec("database", "ingest-jsonl", "Ingest JSONL rows.",
     [("path", "train.db", "DB path"), ("table", "data", "Table"), ("file", "data.jsonl", "JSONL file")],
     lambda p: _db_ingest_jsonl(p))
spec("database", "count", "Count rows.",
     [("path", "train.db", "DB path"), ("table", "data", "Table")], lambda p: _db_count(p))
spec("database", "sample", "Sample rows.",
     [("path", "train.db", "DB path"), ("table", "data", "Table"), ("n", 5, "Count")],
     lambda p: _db_sample(p))
spec("database", "stats", "Basic stats on numeric column.",
     [("path", "train.db", "DB path"), ("table", "data", "Table"), ("column", "score", "Column")],
     lambda p: _db_stats(p))
spec("database", "query", "Run read-only SQL.",
     [("path", "train.db", "DB path"), ("sql", "SELECT * FROM data LIMIT 5", "SQL")],
     lambda p: _db_query(p))


def _db_conn(path: str) -> sqlite3.Connection:
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(Path(path).expanduser()))
    conn.row_factory = sqlite3.Row
    return conn

def _db_create(p: Dict[str, Any]) -> Dict[str, Any]:
    try:
        conn = _db_conn(p["path"])
        conn.execute(f"CREATE TABLE IF NOT EXISTS {p['table']} ({p['schema']})")
        conn.commit(); conn.close()
        return _ok(created=p["path"], table=p["table"], schema=p["schema"])
    except Exception as e:
        return _err(str(e))


def _db_tables(p: Dict[str, Any]) -> Dict[str, Any]:
    try:
        conn = _db_conn(p["path"])
        rows = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        return _ok(tables=rows, count=len(rows))
    except Exception as e:
        return _err(str(e))

def _db_ingest_jsonl(p: Dict[str, Any]) -> Dict[str, Any]:
    fp = Path(p["file"])
    if not fp.exists():
        return _err(f"file {p['file']} not found")
    try:
        conn = _db_conn(p["path"]); cur = conn.cursor(); n = 0
        with fp.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                keys = list(obj.keys())
                placeholders = ",".join("?" for _ in keys)
                cur.execute(f"INSERT INTO {p['table']} ({LKNTS}) VALUES ({placeholders})",
                            [obj[k] for k in keys])
                n += 1
        conn.commit(); conn.close()
        return _ok(ingested=n, table=p["table"])
    except Exception as e:
        return _err(str(e))

def _db_count(p: Dict[str, Any]) -> Dict[str, Any]:
    try:
        conn = _db_conn(p["path"])
        n = conn.execute(f"SELECT COUNT(*) FROM {p['table']}").fetchone()[0]
        conn.close()
        return _ok(rows=n)
    except Exception as e:
        return _err(str(e))

def _db_sample(p: Dict[str, Any]) -> Dict[str, Any]:
    try:
        conn = _db_conn(p["path"])
        rows = [dict(r) for r in conn.execute(f"SELECT * FROM {p['table']} ORDER BY RANDOM() LIMIT {int(p['n'])}").fetchall()]
        conn.close()
        return _ok(rows=rows, count=len(rows))
    except Exception as e:
        return _err(str(e))

def _db_stats(p: Dict[str, Any]) -> Dict[str, Any]:
    try:
        conn = _db_conn(p["path"])
        row = conn.execute(f"SELECT AVG({p['column']}) AS mean, MIN({p['column']}) AS mn, MAX({p['column']}) AS mx, COUNT({p['column']}) AS cnt FROM {p['table']}").fetchone()
        conn.close()
        if not row or row["cnt"] == 0:
            return _err("no rows or non-numeric column")
        return _ok(mean=round(row["mean"], 4), min=row["mn"], max=row["mx"], count=row["cnt"])
    except Exception as e:
        return _err(str(e))

def _db_query(p: Dict[str, Any]) -> Dict[str, Any]:
    try:
        conn = _db_conn(p["path"])
        rows = [dict(r) for r in conn.execute(p["sql"]).fetchall()]
        conn.close()
        return _ok(rows=rows, count=len(rows))
    except Exception as e:
        return _err(str(e))

# ===========================================================================
# CLUSTEROPS group — cluster/GPU orchestration helpers
# ===========================================================================

spec("clusterops", "total-gpus", "Total GPUs in cluster.",
     [("nodes", "8x8,4x4", "Nodes as <count>x<gpus> pairs")],
     lambda p: _cluster_total(p))
spec("clusterops", "allocate", "Allocate GPUs across nodes.",
     [("nodes", "8x8,4x4", "Nodes as <count>x<gpus>"), ("needed", 64, "GPUs needed")],
     lambda p: _cluster_allocate(p))
spec("clusterops", "utilization", "Cluster GPU utilization.",
     [("total", 80, "Total GPUs"), ("busy", 60, "Busy GPUs")],
     lambda p: _ok(utilization_pct=round(int(p["busy"]) / max(int(p["total"]), 1) * 100, 1),
                   free=int(p["total"]) - int(p["busy"])))
spec("clusterops", "jobs-fit", "How many jobs fit.",
     [("total", 80, "Total GPUs"), ("per_job", 8, "GPUs per job")],
     lambda p: _ok(jobs=int(p["total"]) // max(int(p["per_job"]), 1),
                    leftover=int(p["total"]) % max(int(p["per_job"]), 1)))
spec("clusterops", "network-model", "Cluster network model.",
     [("nodes", 8, "Nodes"), ("gpus_per_node", 8, "GPUs/node")],
     lambda p: _ok(rings=1, all_reduce_time_ms=round(math.log2(int(p["nodes"]) * int(p["gpus_per_node"])) * 0.5, 2)))

def _cluster_total(p: Dict[str, Any]) -> Dict[str, Any]:
    total = 0
    for part in str(p["nodes"]).split(","):
        part = part.strip()
        if "x" in part:
            c, g = part.split("x", 1)
            try:
                total += int(c) * int(g)
            except Exception:
                pass
    return _ok(total_gpus=total)

def _cluster_allocate(p: Dict[str, Any]) -> Dict[str, Any]:
    total = 0
    for part in str(p["nodes"]).split(","):
        part = part.strip()
        if "x" in part:
            c, g = part.split("x", 1)
            try:
                total += int(c) * int(g)
            except Exception:
                pass
    needed = int(p["needed"])
    return _ok(available=total, needed=needed,
               fits=total >= needed,
               shortfall=max(0, needed - total))

# ===========================================================================
# Register with the main bulk registry
# ===========================================================================
_bulk.rebuild_index()
