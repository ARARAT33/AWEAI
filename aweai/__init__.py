# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""AWEAI — Universal CLI for AI/ASI/AGI engineering.

AWEAI is a powerful, pure-terminal (CLI-only) toolkit for CREATING,
TRAINING, TUNING and MANAGING AI models, plus a vast command universe
covering data, providers, devices, operations, AGI orchestration, RAG,
security and an AI/ASI/AGI knowledge base.

It ships with **no built-in AI model** and **no Hugging Face
dependency**: every architecture (MLP, CNN, RNN/LSTM/GRU, mini-Transformer,
time-series transformer, n-gram LM, GAN, autoencoder, vision CNN, object
detection, segmentation, clustering, classification, regression,
time-series, NLP, vision) is implemented from zero on a light stack
(numpy / torch / scikit-learn).

v4.0 highlights (CLI-only — no web UI, no GUI, no server):
    CLI-only          — every feature is reachable from the terminal (Typer)
    collect           — scraping, crawling, import/export, cleaning, synthetic data
    data              — datasets, pipelines, preprocessing, tokenization, embedding
    model             — create/train/evaluate/manage 16+ types, fine-tune, transfer, quantize, export
    providers         — OpenAI / Google / Microsoft / Anthropic / Hugging Face adapters (BYOK)
    devices           — SSH, remote hosts, cluster, distributed training, orchestration
    ops               — users, roles, auth, billing, workflows, schedulers, cron, agents
    agi               — orchestration hooks, memory, reasoning, self-improvement (RSI)
    rag               — retrieval-augmented generation (index + search + ground)
    security          — secret scanning, hashing, monitoring, backup
    bulk commands     — 300+ declarative utilities (math, string, json, file, net, time,
                        crypto, ml, text, image, audio, video, db, cloud, llm, rl, neuro)
    ai                — AI/ASI/AGI knowledge base (concepts, timeline, roadmap, levels)
    commands          — inspect the full command universe (list/search/describe/count)
    wiki              — generate the Markdown wiki (docs/wiki/*.md)
    tools             — 1000+ extension tools (list/run/describe/categories)
    autotest          — one-command full-system self-check

Modules:
    aweai.models      — model zoo (from-scratch architectures)
    aweai.data        — data pipeline (loaders, split, normalize, augment, tokenize)
    aweai.train       — training engine (scratch, continue, fine-tune, tuning)
    aweai.eval        — evaluation (metrics, curves, confusion matrix)
    aweai.management  — model zoo manager (save/load/export/import/version/compare)
    aweai.quantize    — model quantization (float16/int8/uint8/int4)
    aweai.export      — export to ONNX / TorchScript / TFLite / edge formats
    aweai.distributed — distributed training (multi-GPU/multi-node)
    aweai.market      — model marketplace (publish/download/rate)
    aweai.ai          — AI/ASI/AGI knowledge base
    aweai.bulk        — declarative bulk command engine (300+ commands)
    aweai.cmd         — domain command groups (collect/data/model/providers/devices/ops)
    aweai.rag         — retrieval-augmented generation (index + search + ground)
    aweai.actions     — automation: natural-language actions, pipelines, batch jobs
    aweai.autotest    — one-command system self-check (autotest)
    aweai.cli         — full command-line interface (Typer, CLI-only)
"""

__version__ = "4.0.0"
__title__ = "AWEAI"
__description__ = "Universal CLI for AI/ASI/AGI engineering — create, train, tune and manage AI models from scratch. No built-in AI, no Hugging Face, no UI."


def about() -> dict:
    """Return package metadata as a dict."""
    return {
        "name": __title__,
        "version": __version__,
        "description": __description__,
        "builtin_ai": False,
        "huggingface_free": True,
        "cli_only": True,
        "ui": False,
        "stack": ["numpy", "torch", "scikit-learn"],
        "features": [
            "cli-only", "data-collection", "data-management", "models",
            "providers", "devices", "ops", "agi", "rag", "security",
            "bulk-commands", "ai-knowledge", "wiki", "tools", "autotest",
        ],
    }
