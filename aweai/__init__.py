"""AWEAI — AI Model Factory.

AWEAI is a powerful, fully-automated toolkit for CREATING, TRAINING,
TUNING and MANAGING AI models from scratch. It ships with **no built-in
AI model** and **no Hugging Face dependency**: every architecture
(MLP, CNN, RNN/LSTM/GRU, mini-Transformer, time-series transformer,
n-gram LM, GAN, autoencoder, vision CNN, object detection, segmentation,
clustering, classification, regression, time-series, NLP, vision) is
implemented from zero on a light stack (numpy / torch / scikit-learn).

The factory is resource-adaptive: it inspects your hardware and picks the
best model type and size that will actually run on your machine, so it
works on laptops, servers, edge devices and Android phones.

v3.0 highlights:
    vision           — VisionCNN, ObjectDetector, SegmentationNet (from scratch)
    time-series      — GRU, TimeSeriesTransformer forecasting
    quantization     — float16 / int8 / uint8 / int4 quantization
    edge export      — ONNX / TFLite / TorchScript / edge-optimized artifacts
    distributed      — multi-GPU / multi-node / multi-thread training
    marketplace      — publish / download / rate models (local-first registry)
    menus            — 10,000+ command & instruction catalog (`aweai allc`)
    automations      — 5,000+ automation catalog (`aweai autoallc`)
    terminal         — full in-app terminal with every tool
    integrations     — OpenAI / Google / Microsoft / Anthropic / Hugging Face adapters (BYOK)
    autotest         — one-command full-system self-check

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
    aweai.menus       — megamenus: 10,000+ commands, automations, search
    aweai.terminal    — in-app terminal (full REPL)
    aweai.integrations— AI-tool adapters (OpenAI/Google/Microsoft/Anthropic/HF)
    aweai.rag         — retrieval-augmented generation (index + search + ground)
    aweai.actions     — automation: natural-language actions, pipelines, batch jobs
    aweai.autotest    — one-command system self-check (autotest)
    aweai.ui          — powerful browser UI (dashboard, wizards, live curves, terminal)
    aweai.cli         — full command-line interface
"""

__version__ = "3.0.0"
__title__ = "AWEAI"
__description__ = "AI Model Factory — create, train, tune and manage AI models from scratch. No built-in AI, no Hugging Face."


def about() -> dict:
    """Return package metadata as a dict."""
    return {
        "name": __title__,
        "version": __version__,
        "description": __description__,
        "builtin_ai": False,
        "huggingface_free": True,
        "stack": ["numpy", "torch", "scikit-learn"],
        "features": [
            "vision", "time-series", "quantization", "edge-export",
            "distributed", "marketplace", "megamenus", "terminal",
            "integrations", "autotest",
        ],
    }
