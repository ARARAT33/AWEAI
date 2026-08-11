# RoPE

Rotary Position Embedding (RoPE) encodes positional information using rotation matrices, enabling better length generalization.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | `int` | `512` | Model dimension |
| `max_seq_len` | `int` | `2048` | Maximum sequence length |
| `base` | `int` | `10000` | RoPE base frequency |

## Related Pages

- [ALiBi](ALiBi.md) — Attention with Linear Biases
- [RMSNorm](RMSNorm.md) — RMS Normalization
- [FlashAttention](FlashAttention.md) — FlashAttention
