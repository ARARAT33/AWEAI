# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""AWEAI — Universal CLI for AI/ASI/AGI engineering.

AWEAI is a powerful, pure-terminal (CLI-only) toolkit for CREATING,
TRAINING, TUNING and MANAGING AI models, plus a vast command universe
covering data, providers, devices, operations, AGI orchestration, RAG,
security and an AI/ASI/AGI knowledge base.

It ships with no built-in AI model and no Hugging Face dependency.

v4.1 highlights:
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
    product           — contracts, health gates, artifact lineage, benchmarks, release gates
    production_ops    — SLOs, canary rollout, cost budgets, reproducibility, incident state
    bulk commands     — 300+ declarative utilities
    commands          — inspect the full command universe (list/search/describe/count)
    wiki              — generate the Markdown wiki
    tools             — extension tools (list/run/describe/categories)
    autotest          — one-command full-system self-check

The product and production layers are model-agnostic engineering primitives;
they do not turn AWEAI into a chat or autonomous-agent product.
"""

__version__ = "4.1.0"
__title__ = "AWEAI"
__description__ = "Universal CLI for AI/ASI/AGI engineering and production operations. No built-in AI, no UI."


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
            "product-control-plane", "production-slos", "canary-rollouts",
            "cost-budgets", "reproducibility", "incident-state",
        ],
    }
