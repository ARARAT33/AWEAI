# Overview

AWEAI Scale Engine — train models from 1 to 2T+ parameters.

## Overview

AWEAI Scale Engine — train models from 1 to 2T+ parameters.

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
