# RMSNorm

RMS Normalization (RMSNorm) is a simplified layer normalization that only normalizes by the root mean square, without mean centering.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | `int` | `512` | Model dimension |
| `eps` | `float` | `1e-6` | Epsilon for numerical stability |

## Related Pages

- [RoPE](RoPE.md) — Rotary Position Embedding
- [ALiBi](ALiBi.md) — Attention with Linear Biases
- [FlashAttention](FlashAttention.md) — FlashAttention
