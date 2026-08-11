# ALiBi

Attention with Linear Biases (ALiBi) replaces positional embeddings with linear biases, enabling better length extrapolation.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | `int` | `512` | Model dimension |
| `nhead` | `int` | `8` | Attention heads |
| `num_layers` | `int` | `6` | Number of layers |

## Related Pages

- [RoPE](RoPE.md) — Rotary Position Embedding
- [RMSNorm](RMSNorm.md) — RMS Normalization
- [FlashAttention](FlashAttention.md) — FlashAttention
