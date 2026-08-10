# User Guide

## Quick start

```bash
pip install -e .
aweai autotest        # verify the whole system
aweai commands count      # inspect the command universe
```

## Train your first model

```bash
aweai train --type mlp --name my_model --data data.csv --target label --params '{"epochs": 50}'
```

or in Python:

```python
from aweai.train import train
res = train("mlp", "my_model", X=[[0,0],[1,1]], y=[0,1], params={"epochs": 10})
```

## Vision & time-series (v2.1)

```bash
# image classification
aweai train --type vision_cnn --name v1 --data images.csv --target label

# object detection (grid-based)
aweai train --type object_detector --name od1 --data boxes.csv

# segmentation
aweai train --type segmentation --name seg1 --data masks.csv

# time-series forecasting
aweai train --type gru --name ts1 --data series.csv
aweai train --type ts_transformer --name f1 --data series.csv
```

## Quantization & edge export (v2.2)

```bash
aweai quantize my_model --fmt int8          # float16/int8/uint8/int4
aweai export-edge my_model --fmt tflite     # onnx/tflite/torchscript/edge_json
aweai export-edge my_model --fmt onnx --quantize int8
aweai edge-footprint my_model               # fp32/fp16/int8 footprint
```

## Distributed training & marketplace (v3.0)

```bash
aweai dworld                                 # detect GPUs/nodes/backend
aweai dtrain mlp --name d1 --data train.csv --workers 4
aweai market publish my_model --tag v1 --description "first model"
aweai market search "mlp"
aweai market download <id>
aweai market rate <id> 5
```

## Integrations (BYOK)

```bash
aweai integrations list
aweai integrations chat --provider openai --message "hello"
# set OPENAI_API_KEY / GOOGLE_API_KEY / AZURE_OPENAI_KEY / ANTHROPIC_API_KEY / HF_TOKEN
```

## Megamenus & terminal

```bash
aweai allc                    # 10,000+ commands & instructions
aweai allc --search "quantize"
aweai autoallc                # 5,000+ automations
aweai commands list         # full command universe
```

## CLI overview

- `aweai hardware` — show detected hardware and resource tier
- `aweai recommend --task classification` — best model type for this machine
- `aweai train/continue/eval/models/export/delete/compare` — model lifecycle
- `aweai quantize/export-edge/edge-footprint` — quantization & edge
- `aweai dtrain/dworld` — distributed training
- `aweai market ...` — model marketplace
- `aweai data load/split/augment` — data tools
- `aweai rag index/ask` — RAG
- `aweai actions "..."` — natural-language automation
- `aweai allc/autoallc` — megamenus
- `aweai wiki build` — generate the Markdown wiki
- `aweai autotest` — full system self-check
- `aweai ai explain <term>` — AI/ASI/AGI knowledge
