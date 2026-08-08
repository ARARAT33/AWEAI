# Autotest

`aweai autotest` verifies the whole system in one command:

1. **dependencies** — required packages importable
2. **module imports** — every `aweai.*` module imports
3. **smoke-train all model types** — mlp, linear, logistic, kmeans, ngram, autoencoder, gan, rnn, lstm, cnn, transformer
4. **RAG** — index → search → reload from disk (verifies the index_file fix)
5. **actions** — natural-language parsing works
6. **i18n** — 10+ languages load
7. **UI** — server boots, `/api/health` responds
8. **CLI** — all commands registered

There is also an **Autotest button** in the UI dashboard.
