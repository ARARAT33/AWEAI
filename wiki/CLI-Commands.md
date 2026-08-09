# CLI Commands

Install with `pip install -e .` and run `aweai` from any terminal. The binary builds (EXE / Linux / macOS) expose the same CLI.

## Top-level commands

```
aweai version | hardware | recommend | types
aweai train --type TYPE --name NAME [--data PATH] [--params JSON]
aweai continue-train NAME [--data PATH] [--epochs N]
aweai eval NAME [--data PATH] [--target COL]
aweai models | export NAME --fmt FMT | import | delete | compare
aweai quantize NAME --fmt float16|int8|uint8|int4
aweai export-edge NAME --fmt onnx|tflite|torchscript|edge_json [--quantize FMT]
aweai edge-footprint NAME
aweai dtrain TYPE --name NAME [--data PATH] [--workers N] [--backend auto|thread|torch]
aweai dworld
aweai market publish|search|list|info|download|rate|stats ...
aweai integrations list|chat --provider P --message M
aweai allc [--category C] [--search Q] [--count N] [--json]
aweai autoallc [--category C] [--search Q] [--count N] [--json]
aweai terminal
aweai data load/split/augment | rag index/ask | actions "..." | pipeline ...
aweai autotest [--quick] [--no-ui]
aweai serve [--port N] [--host H]
```

## The 10,000+ command catalog

- `aweai allc` prints **10,000+ commands & instructions**, searchable by category and keyword.
- `aweai autoallc` prints **10,000+ automations** (pipelines, batch jobs, workflows).
- Options: `--category`, `--search`, `--count`, `--json`.

```bash
aweai allc --category train        # all training instructions
aweai allc --search quantize       # find quantization instructions
aweai autoallc --category rag      # all RAG automations
```

## v3.0 features from the CLI

| Feature | CLI |
|---------|-----|
| Vision CNNs | `aweai train --type vision_cnn ...` |
| Sequence models | `aweai train --type gru ...` / `--type lstm ...` |
| Quantization | `aweai quantize NAME --fmt int8` |
| Edge export | `aweai export-edge NAME --fmt tflite` |
| Distributed training | `aweai dtrain TYPE --name NAME --workers 4` |
| Marketplace | `aweai market publish/search/download/rate` |
| In-app terminal | `aweai terminal` |
| Integrations | `aweai integrations chat --provider openai --message hi` |
| Full audit | `aweai autotest` |
