"""AWEAI — AI Model Factory.

AWEAI is a powerful, fully-automated toolkit for CREATING, TRAINING,
TUNING and MANAGING AI models from scratch. It ships with **no built-in
AI model** and **no Hugging Face dependency**: every architecture
(MLP, CNN, RNN/LSTM, mini-Transformer, n-gram LM, GAN, autoencoder,
clustering, classification, regression, time-series, NLP, vision) is
implemented from zero on a light stack (numpy / torch / scikit-learn).

The factory is resource-adaptive: it inspects your hardware and picks the
best model type and size that will actually run on your machine, so it
works on laptops, servers, edge devices and Android phones.

Modules:
    aweai.models      — model zoo (from-scratch architectures)
    aweai.data        — data pipeline (loaders, split, normalize, augment, tokenize)
    aweai.train       — training engine (scratch, continue, fine-tune, tuning)
    aweai.eval        — evaluation (metrics, curves, confusion matrix)
    aweai.management  — model zoo manager (save/load/export/import/version/compare)
    aweai.export      — export to ONNX / TorchScript / raw weights / JSON config
    aweai.rag         — retrieval-augmented generation (index + search + ground)
    aweai.actions     — automation: natural-language actions, pipelines, batch jobs
    aweai.autotest    — one-command system self-check (autotest)
    aweai.ui          — powerful browser UI (dashboard, wizards, live curves)
    aweai.cli         — full command-line interface
"""

__version__ = "2.0.0"
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
    }
