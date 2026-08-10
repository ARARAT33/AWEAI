# Changelog

## v4.0.0 — 2026-08-10 — CLI-only universal command universe

**Breaking: UI/Web/GUI removed**
- Deleted `aweai/ui/`, `aweai/anywhere.py`, `aweai/terminal/`, `aweai/menus/`.
- Removed `aweai serve` and `aweai anywhere`; no FastAPI/uvicorn, no static files.
- AWEAI is now pure **CLI (Typer)**.

**Massive CLI expansion**
- New command groups: `collect`, `data`, `model`, `providers`, `devices`, `ops`.
- New `ai` group: AI/ASI/AGI knowledge base (119 concepts, timeline, roadmap, AGI levels, self-improvement hooks).
- New `commands` group: list/search/describe/count the entire command universe.
- New `wiki` group: `aweai wiki build` generates `docs/wiki/*.md`.
- New bulk engine (`aweai.bulk`): 329 declarative commands in 18 groups
  (math, string, json, file, net, time, crypto, ml, text, image, audio,
  video, sys, db, cloud, llm, rl, neuro, knowledge).
- Total CLI surface: **467 commands / 28 groups** (was 33 commands).

**Data & AGI tooling**
- `collect` — scrape, crawl, RSS, import/export CSV/JSON, dedupe, sample,
  clean, augment-text, synthetic data, stats.
- `data` — inspect, split, merge, filter, map, normalize, onehot, tokenize,
  embed, similarity, declarative pipeline runner.
- `model` — train/types/list/info/eval/predict/continue/fine-tune/transfer/
  tune/quantize/export/export-edge/footprint/import/delete/compare/recommend/
  distributed/world.
- `providers` — list, set-key/unset-key, chat, complete, models, fine-tune, check.
- `devices` — detect, ssh, ssh-add/list/remove, cluster-status, distributed,
  orchestrate, lan, benchmark.
- `ops` — user-add/list/remove, role-set, auth, permissions, billing-add/
  balance, token-issue, workflow-add/list/run/remove, scheduler-list, cron,
  agent-run, agi-status, memory-add/get/list/clear, reason, self-improve,
  rag-index/search/ask, vector-store, security-scan, hash-file, monitor,
  backup, backup-list.

**Quality**
- 33 unit tests green; compileall clean; autotest 3/3 (deps, imports, CLI).
- CI updated to CLI-only; no-UI enforced; No-HuggingFace guard kept.

## v3.0.0 — 2026-08-08 — Next-level factory

**Vision (v2.1)**
- `VisionCNN` — image classification from scratch (conv+pool+fc, numpy).
- `ObjectDetector` — grid-based object detection with NMS (from scratch).
- `SegmentationNet` — per-pixel segmentation (from scratch).

**Time-series (v2.1)**
- `GRU` — gated recurrent unit for forecasting (from scratch).
- `TimeSeriesTransformer` — lightweight transformer forecasting (from scratch).

**Quantization & edge export (v2.2)**
- `aweai.quantize` — float16 / int8 / uint8 / int4 quantization with accuracy
  evaluation and compression ratio.
- `aweai.export.edge` — ONNX / TorchScript / TFLite / edge-optimized exports,
  dependency-free TFLite-style artifact + loader, edge footprint estimator.

**Distributed training & marketplace (v3.0)**
- `aweai.distributed` — multi-GPU / multi-node / multi-thread data-parallel
  training (torch DDP-style + thread backend, safe by default).
- `aweai.market` — publish / download / rate / search models, local-first
  registry with zip archives and download/rating statistics.

**Megamenus, terminal & integrations**
- `aweai allc` — 10,000+ command & instruction catalog (searchable, JSON out).
- `aweai autoallc` — 5,000+ automation catalog.
- `aweai terminal` — full in-app terminal REPL exposing every tool.
- `aweai.integrations` — OpenAI / Google / Microsoft / Anthropic / HF adapters
  (BYOK), with `aweai integrations chat`.
- UI Terminal & Marketplace tabs, `/api/terminal`, `/api/allc`, `/api/autoallc`,
  `/api/market`, `/api/quantize`, `/api/export/edge`, `/api/integrations`.

**Quality**
- Autotest extended: every module (24), every model type (16), CLI commands
  (31), RAG, actions, i18n (12 languages), UI health.
- 32 unit tests green; compileall clean; workflows verified for tag-driven
  APK/EXE/Linux releases.

## v2.0.0 — 2026-08-08 — AI Model Factory rewrite

**Repositioning**
- AWEAI is no longer a chatbot and ships no built-in AI model.
- It is now a **model factory**: create, train, tune, manage and export AI
  models from scratch, fully automated.

**Removed**
- All Hugging Face dependencies (`transformers`, `datasets`,
  `huggingface_hub`, PEFT) and all chatbot-era code.

**Added**
- From-scratch model zoo: MLP, linear, logistic, KMeans, n-gram LM, RNN,
  LSTM, CNN, mini-Transformer, GAN, autoencoder.
- Data pipeline: CSV/JSON/JSONL/text/images loaders, split, normalize,
  augment, own tokenizer.
- Training engine: from scratch, continue/fine-tune, hyperparameter tuning,
  early stopping, metrics.
- Evaluation: accuracy, precision, recall, F1, confusion matrix, loss curves.
- Model management: save/load/export/import, list, delete, versioning,
  compare.
- Export: ONNX, TorchScript, raw weights, JSON config.
- Automation: natural-language actions, pipelines, batch jobs.
- RAG (numpy-only) with the `index_file` shadowing bug fixed.
- n-gram tuple-key serialization fixed.
- Autotest command + UI Autotest button.
- UI on smart port 8888 (+1): dashboard, wizard, live training curves,
  model zoo, dataset manager, hyperparameter panel.
- i18n: 12 languages (English primary, Armenian included).
- Android APK support; CI workflow; Makefile automation; docs.
