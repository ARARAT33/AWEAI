# Roadmap

## v2.1 — foundations
- Model factory core: MLP, linear, logistic, KMeans, n-gram
- CLI training / evaluation / model zoo
- Basic browser UI + REST API

## v2.2 — scale & automation
- RNN / LSTM / GRU / CNN / Transformer
- Data pipelines, RAG index/ask, natural-language actions
- Automation engine (`aweai autoallc`), autotest suite
- i18n (12+ languages)

## v3.0 — vision, edge, distribution, marketplace
- Vision CNNs (VisionCNN, ObjectDetector, SegmentationNet), time-series Transformer
- Quantization (fp16/int8/uint8/int4) and edge export (ONNX/TFLite/TorchScript)
- Distributed training (multi-GPU / multi-node / multi-thread)
- Marketplace (publish / search / download / rate)
- In-app terminal (`aweai terminal`), BYOK integrations
- Megamenus: `aweai allc` = 10,000+ commands, `aweai autoallc` = 5,000+ automations
- Release pipeline: APK (buildozer + NDK r25b), EXE / Linux / macOS / AppImage / web

## v3.0.x (planned maintenance)
- Keep CI green on the latest runner images
- Reproducible Android builds (NDK/Python pins documented in buildozer.spec)

## v4.0 (idea backlog)
- WASM/edge in-browser inference
- Federated learning over the marketplace
- Plugin SDK for community model types
