# MultiQueryAttention

Multi-Query Attention (MQA) shares key and value heads across all query heads, reducing memory and increasing speed.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | `int` | `512` | Model dimension |
| `nhead` | `int` | `8` | Number of query heads |
| `kv_heads` | `int` | `1` | Number of key/value heads |

## Related Pages

- [GroupedQueryAttention](GroupedQueryAttention.md) — Grouped-Query Attention
- [FlashAttention](FlashAttention.md) — FlashAttention
- [RoPE](RoPE.md) — Rotary Position Embedding
