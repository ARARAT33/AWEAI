# Topology

Network topology management for cluster interconnects.

## Usage

```bash
# View cluster topology
aweai cluster topology view

# Configure topology
aweai cluster topology configure --type fat-tree
```

## Topologies

| Topology | Description |
|----------|-------------|
| `fat-tree` | Balanced tree topology |
| `torus` | 3D torus network |
| `dragonfly` | Dragonfly topology |
| `fully-connected` | Full mesh |

## Related Pages

- [NVLink](NVLink.md) — NVLink
- [InfiniBand](InfiniBand.md) — InfiniBand
- [RoCE](RoCE.md) — RoCE
