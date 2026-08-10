# AWEAI Model Training — Unlimited Scale Architecture

> Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.

AWEAI v4.2 turns the CLI into the planet's most powerful model-training
tool: train models of **any size** — from 200 million to 200 billion,
2 trillion and beyond (unlimited). Change architecture before or during
training, orchestrate GPU clusters, manage training-data databases and
estimate memory/precision at any scale.

```
Total commands: ~962+   |   New groups: arch, scale, cluster, dbops, architect,
                         model-size, distributed, precision, sharding,
                         checkpoint, gpu, training, database, clusterops
```

---

## 1. `aweai arch` — Architecture (create / convert / estimate)

Create, convert and inspect model architecture specs with **exact
dimensions**: params (any size), layers, heads, kv-heads, dim, vocab,
experts (MoE), routing, activation, normalization, RoPE, sliding window,
Mamba/SSM blocks.

| Command | Description |
| --- | --- |
| `arch create` | Create/override an architecture spec (any type, any size) |
| `arch list` | List saved architecture specs |
| `arch show` | Show a full architecture spec |
| `arch delete` | Delete an architecture spec |
| `arch convert` | Convert a spec to another type (keeps params/dims) |
| `arch estimate` | Estimate params + memory for a config |
| `arch to-json` / `arch from-json` | Export / import specs as JSON |
| `arch types` / `arch routings` | List supported types / MoE routings |

Examples:

```bash
# 2-trillion-parameter MoE spec
aweai arch create gpt-moe --type moe --params 2T --layers 128 --dim 16384 \
    --heads 128 --experts 16 --moe-layers 120 --routing top2

# Dense transformer
aweai arch create gpt-dense --type transformer --params 70B \
    --layers 80 --dim 8192 --heads 64 --vocab 100352

# Convert dense -> MoE before training
aweai arch convert gpt-dense --to moe

# Estimate memory for a 200B MoE model
aweai arch estimate --type moe --layers 64 --dim 8192 --experts 8
```

---

## 2. `aweai scale` — Train ANY model size (unlimited)

Configure and launch training runs of unlimited size: parameter count,
precision (FP32/FP16/BF16/FP8/INT8/TF32), optimizer (AdamW/Adam/SGD/Lion/
Adafactor/LAMB), offloading (none/CPU/NVMe/smart), ZeRO stages 0–3, FSDP,
tensor/pipeline/data parallelism, sequence parallelism, flash attention,
gradient checkpointing, gradient accumulation, LR warmup, checkpointing
and resuming.

| Command | Description |
| --- | --- |
| `scale config` | Create a training config for ANY model size |
| `scale list` / `scale show` / `scale delete` | Manage configs |
| `scale memory` | Estimate memory at any scale/precision |
| `scale checkpoint` | Save a checkpoint (metadata + pointer) |
| `scale checkpoints` | List checkpoints for a run |
| `scale resume` | Resume instructions from latest checkpoint |
| `scale train` | Launch a run (validates config, prints plan) |
| `scale param-count` | Exact params for a transformer/MoE config |
| `scale precisions` / `scale optimizers` / `scale offloads` | Enumerations |

Examples:

```bash
# 200B BF16 training config with ZeRO-3 + CPU offload
aweai scale config run-200b --model-type transformer --params 200B \
    --layers 96 --dim 12288 --precision bf16 --zero 3 --offload cpu \
    --dp 64 --tp 8 --pp 4 --grad-accum 2 --batch-size 1024

# Memory estimate for that run
aweai scale memory --params 200B --precision bf16 --zero 3 --offload cpu

# Exact parameter count for a MoE config
aweai scale param-count --layers 96 --dim 12288 --experts 16 --moe-layers 64

# Checkpoint + resume
aweai scale checkpoint run-200b --step 40000 --note "mid-training"
aweai scale resume run-200b
aweai scale train run-200b --resume-step 40000 --dry-run
```

---

## 3. `aweai cluster` — Cluster / GPU orchestration

Register and manage multi-node clusters: SSH-reachable nodes, GPU counts,
roles (worker/master/inference/storage), GPU allocation/freeing, health
checks (TCP reachability + latency), auto-scaling, backup/restore and a
cluster summary.

| Command | Description |
| --- | --- |
| `cluster add` | Register a node (host, SSH port, GPUs, RAM, role, tags) |
| `cluster list` / `cluster remove` | List / remove nodes |
| `cluster alloc-gpus` / `cluster free-gpus` | Allocate / free GPUs per node |
| `cluster health` | Health check (reachability + latency) |
| `cluster scale` | Auto-scale to a target node count (clones template) |
| `cluster backup` / `cluster restore` | Backup / restore cluster state |
| `cluster summary` | Nodes, total GPUs, utilization, roles |

Examples:

```bash
aweai cluster add node-a --host 10.0.0.1 --port 22 --user root --gpus 8 --role worker
aweai cluster add node-b --host 10.0.0.2 --gpus 8
aweai cluster health
aweai cluster alloc-gpus node-a --count 4 --job training-200b
aweai cluster summary
aweai cluster scale 4 --template node-a --prefix worker
aweai cluster backup --out cluster-backup.json
```

---

## 4. `aweai dbops` — Training-data databases

Manage databases for training data: connect, list tables, create tables,
ingest JSONL/CSV/JSON or single text, run read-only SQL, count rows,
snapshot, restore, vacuum and export to JSONL/CSV.

| Command | Description |
| --- | --- |
| `dbops connect` | Connect to a DB (creates if missing) |
| `dbops tables` / `dbops schema` | List tables / show schema |
| `dbops create-table` | Create a table for training data |
| `dbops ingest` | Ingest JSONL/CSV/JSON or single text |
| `dbops query` / `dbops count` | Query / count rows |
| `dbops snapshot` / `dbops restore` | Backup / restore DB |
| `dbops vacuum` | Compact DB |
| `dbops export` | Export table to JSONL/CSV |

Examples:

```bash
aweai dbops connect train.db
aweai dbops create-table train.db data --schema "id INTEGER PRIMARY KEY, text TEXT, label TEXT"
aweai dbops ingest train.db data --file dataset.jsonl --columns text,label
aweai dbops count train.db data
aweai dbops query train.db --sql "SELECT label, COUNT(*) FROM data GROUP BY label"
aweai dbops snapshot train.db
aweai dbops export train.db data --out train.jsonl
```

---

## 5. `aweai architect` — Architecture math (bulk)

Pure helpers for architecture math (declarative bulk group):

| Command | Description |
| --- | --- |
| `architect types` | List supported architecture families |
| `architect moe-params` | Parameter count for a MoE config |
| `architect dense-params` | Parameter count for a dense transformer |
| `architect rnn-params` | RNN/LSTM/GRU parameter count |
| `architect cnn-params` | CNN stack parameter count |
| `architect hybrid-pct` | Dense/MoE/SSM split percentage |
| `architect heads-for-dim` | Suggested heads for a dim |
| `architect recommend-dim` | Recommend dim/layers for a param budget |

```bash
aweai architect moe-params --layers 64 --dim 8192 --experts 8
aweai architect recommend-dim --params 200B
```

---

## 6. `aweai model-size` — Any-size model math

| Command | Description |
| --- | --- |
| `model-size parse` / `model-size format` | Parse/format size strings (70B/2T) |
| `model-size per-token` | Compute FLOPs (6 × params × tokens) |
| `model-size chinchilla` | Optimal tokens (~20 × params) |
| `model-size kaplan` | Compute-optimal params for FLOPs budget |
| `model-size weights-mem` / `embedding-mem` | Memory at precision |
| `model-size gpu-count` | Min GPUs to fit a model |
| `model-size compare` / `tokens-per-day` | Compare sizes / throughput |

```bash
aweai model-size gpu-count --params 2T --gpu_gb 80
aweai model-size chinchilla --params 70B
```

---

## 7. `aweai distributed` — Parallelism helpers

| Command | Description |
| --- | --- |
| `distributed world-size` | World = dp × tp × pp |
| `distributed effective-batch` | Effective batch size |
| `distributed zero-mem` | ZeRO memory savings by stage |
| `distributed tp-comm` | Tensor-parallel comm volume |
| `distributed dp-vs-tp` / `recommend` | Choose strategy |
| `distributed ring-time` / `node-partition` / `micro-batches` | Cluster planning |

```bash
aweai distributed zero-mem --params 70B --stage 3 --gpus 8
aweai distributed recommend --world 64 --params 70B --gpu_gb 80
```

---

## 8. `aweai precision` — Mixed precision (FP16/BF16/FP8)

| Command | Description |
| --- | --- |
| `precision types` / `range` / `convert` | Precision math |
| `precision save` | Memory saved vs fp32 |
| `precision loss-scale` | Dynamic loss scaling policy |
| `precision overflow` | Overflow check |
| `precision fp8-scale` | FP8 scaling for a tensor |
| `precision recommend` | Recommended precision for a task |

```bash
aweai precision save --params 70B --from_prec fp32 --to_prec bf16
aweai precision fp8-scale --values "-1,0.5,2"
```

---

## 9. `aweai sharding` — FSDP / offload / sharding

| Command | Description |
| --- | --- |
| `sharding fsdp-shards` | FSDP shards needed |
| `sharding offload-mem` | Memory with CPU/NVMe offload |
| `sharding activation-checkpoint` | Memory saved by gradient checkpointing |
| `sharding tensor-shard` / `expert-shard` | Shard sizes |

```bash
aweai sharding fsdp-shards --params 200B --gpu_gb 80
aweai sharding activation-checkpoint --layers 96 --dim 12288 --seq 4096 --batch 1024
```

---

## 10. `aweai checkpoint` — Checkpoint/resume helpers

| Command | Description |
| --- | --- |
| `checkpoint plan` | Checkpoint plan for a run |
| `checkpoint size` | Checkpoint file size estimate |
| `checkpoint shard-count` | FSDP checkpoint shards |
| `checkpoint eta` | Estimated time remaining |
| `checkpoint resume-info` | What to restore |

```bash
aweai checkpoint size --params 200B --optimizer true
aweai checkpoint eta --step 40000 --total 100000 --seconds_per_step 0.35
```

---

## 11. `aweai gpu` — GPU helpers

| Command | Description |
| --- | --- |
| `gpu mem-usage` | Memory breakdown for a model on GPU |
| `gpu tf32-note` | TF32 compute info |
| `gpu cuda-cores` | Estimate CUDA cores from SMs |
| `gpu bandwidth` | Transfer time over NVLink/PCIe |
| `gpu util-check` | Utilization levels |

```bash
aweai gpu mem-usage --params 70B --batch 1024 --seq 2048
aweai gpu bandwidth --gb 10 --gbps 900
```

---

## 12. `aweai training` — Training-loop helpers

| Command | Description |
| --- | --- |
| `training lr-schedule` | LR at step (cosine/warmup) |
| `training grad-clip` | Gradient clipping norm |
| `training loss` | CE / MSE / MAE / Huber loss |
| `training ppl` | Perplexity from loss |
| `training flops-per-step` | FLOPs per training step |
| `training steps-for-tokens` / `epochs` / `throughput` | Planning |

```bash
aweai training lr-schedule --step 5000 --total 100000 --base 0.0003 --warmup 2000
aweai training flops-per-step --params 200B --batch 1024 --seq 2048
```

---

## 13. `aweai database` — Training-data DB helpers (bulk)

| Command | Description |
| --- | --- |
| `database create` | Create SQLite DB with schema |
| `database tables` / `count` / `sample` / `stats` | Introspection |
| `database ingest-jsonl` | Ingest JSONL rows |
| `database query` | Run read-only SQL |

```bash
aweai database create --path train.db --table data
aweai database ingest-jsonl --path train.db --table data --file dataset.jsonl
aweai database stats --path train.db --table data --column score
```

---

## 14. `aweai clusterops` — Cluster orchestration helpers (bulk)

| Command | Description |
| --- | --- |
| `clusterops total-gpus` | Total GPUs from `<count>x<gpus>` pairs |
| `clusterops allocate` | Allocate GPUs across nodes |
| `clusterops utilization` | Cluster GPU utilization |
| `clusterops jobs-fit` | How many jobs fit |
| `clusterops network-model` | Cluster network model |

```bash
aweai clusterops allocate --nodes "8x8,4x4" --needed 64
aweai clusterops utilization --total 80 --busy 60
```

---

## Quick reference — the full training flow

```bash
# 1. Architect the model
aweai arch create my-2t --type moe --params 2T --layers 128 --dim 16384 --experts 16
aweai arch estimate --type moe --layers 128 --dim 16384 --experts 16

# 2. Add cluster nodes
aweai cluster add n1 --host 10.0.0.1 --gpus 8
aweai cluster add n2 --host 10.0.0.2 --gpus 8
aweai cluster health

# 3. Ingest training data
aweai dbops create-table train.db data --schema "id INTEGER PRIMARY KEY, text TEXT"
aweai dbops ingest train.db data --file dataset.jsonl --columns text

# 4. Configure & launch an unlimited-size run
aweai scale config run-2t --model-type moe --params 2T --precision bf16 --zero 3 --dp 256 --tp 8 --pp 4
aweai scale memory --params 2T --precision bf16 --zero 3 --offload cpu
aweai scale train run-2t --dry-run

# 5. Checkpoint & resume
aweai scale checkpoint run-2t --step 10000
aweai scale resume run-2t
```

Every command prints JSON (`{"ok": true, ...}`) so any AI can parse the
result reliably. All sizes accept `K/M/B/T` suffixes or plain integers —
there is **no upper limit** on model size.
