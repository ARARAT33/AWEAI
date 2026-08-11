# HierarchicalMoE

Hierarchical MoE organizes experts into multiple levels, enabling coarse-to-fine specialization.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_levels` | `int` | `2` | Number of hierarchy levels |
| `experts_per_level` | `list[int]` | `[8, 16]` | Experts per level |
| `top_k` | `int` | `2` | Active experts per level |

## Related Pages

- [MoE](MoE.md) — Mixture of Experts
- [HybridMoETransformer](HybridMoETransformer.md) — Hybrid MoE Transformer
