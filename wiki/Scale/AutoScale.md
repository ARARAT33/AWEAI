# AutoScale

Automatic scaling based on hardware detection.

## Overview

Automatic scaling based on hardware detection.

## Usage

```bash
aweai scale --strategy zero3 --model transformer-2t --cluster gpu-cluster
```

## Configuration

The scale engine automatically detects hardware and selects the optimal parallelism strategy.

## Supported Strategies

- ZeRO Stage 1/2/3
- FSDP
- Pipeline Parallelism
- Tensor Parallelism (1D/2D/2.5D)
- CPU/SSD/NVMe Offloading
- Mixed Precision
- Gradient Accumulation
- Activation Checkpointing
