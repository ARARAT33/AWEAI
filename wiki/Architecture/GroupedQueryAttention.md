# GroupedQueryAttention

Grouped-Query Attention (GQA) divides query heads into groups, each sharing a key/value head, balancing efficiency and quality.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | `int` | `512` | Model dimension |
| `nhead` | `int` | `8` | Number of query heads |
| `kv_heads` | `int` | `4` | Number of key/value heads |

## Related Pages

- [MultiQueryAttention](MultiQueryAttention.md) — Multi-Query Attention
- [FlashAttention](FlashAttention.md) — FlashAttention
- [RoPE](RoPE.md) — Rotary Position Embedding
