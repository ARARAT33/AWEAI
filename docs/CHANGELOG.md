# Changelog

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
