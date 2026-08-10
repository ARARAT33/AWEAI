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
aweai allc [--category C] [--search Q] [--count N] [--json] [--huge]
aweai autoallc [--category C] [--search Q] [--count N] [--json]
aweai commands list
aweai data load/split/augment | rag index/ask | actions "..." | pipeline ...
aweai autotest [--quick] [--no-ui]
aweai wiki build
```

## The 10,000+ / 100,000+ command catalog

- `aweai allc` prints **10,000+ commands & instructions**, searchable by category and keyword.
- `aweai allc --huge` prints the **100,000+ entry catalog** (v3.1).
- `aweai autoallc` prints **10,000+ automations** (pipelines, batch jobs, workflows).
- Options: `--category`, `--search`, `--count`, `--json`, `--huge`.

```bash
aweai allc --category train        # all training instructions
aweai allc --search quantize       # find quantization instructions
aweai allc --huge --count 20       # first 20 lines of the 100k catalog
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
| Command registry | `aweai commands list` |
| Integrations | `aweai integrations chat --provider openai --message hi` |
| Full audit | `aweai autotest` |

## The 1800+ tool toolkit

AWEAI registers **1800+ unique-purpose tools** (`aweai tools` family) across
categories such as core, security, devops, datascience, media, automation,
networking, aiagents, codegen, testing, monitoring, creative, the 640+
generated `mega` tools and the **697 new `mega2` tools (v3.1)**: crypto
(hash/hmac/cipher/otp/strength), ml (metrics/activations/losses/kmeans),
web (url/html/cors), db (sqlite helpers), cloud (s3/gs/blob/colab detect),
i18n (12 languages), config (ini/env/json), quant (int8/int4/scale), rag
(chunk/tf/cosine), market, quality, ui (color/contrast/responsive),
net (cidr/ports), sys2, data2, math2, str2, json2, time2, gen2, code2, fs2,
sec2 (validators/luhn/otp), fmt2, valid2, csv2, xml2, yaml2, env, combo,
chart (ascii), rep, note, menu, dist, sched2, monitor2 (apdex/sla),
backup2, ai2, auto2, ops (semver), test2, media2.

```bash
aweai tools categories                          # 80+ categories with counts
aweai tools list                                # list all tools
aweai tools list --category security            # list one category
aweai tools describe --name hash_sha256         # purpose + signature
aweai tools run --name hash_sha256 --params '{"text":"hello"}'
aweai tools run --name math_fibonacci --params '{"n":12}'
aweai tools run --name geo_distance_km --params '{"lat1":40,"lon1":44,"lat2":41,"lon2":45}'
aweai tools run --name crypto_hash_sha256 --params '{"s":"hello"}'
aweai tools run --name ml_f1 --params '{"y_true":"[1,0,1]","y_pred":"[1,0,0]"}'
aweai tools run --name env_detect
aweai tools run --name combo_permutations --params '{"xs":"[1,2,3]","r":2}'
```

Every tool also runs from the UI (`/api/tools/run`), the in-app terminal and
the megamenus — so the **100,000+ menu/page structure** covers all of them.
