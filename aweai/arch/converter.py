from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class WeightTransfer:
    def __init__(self, source: nn.Module, target: nn.Module) -> None:
        self.source = source
        self.target = target

    def transfer(self) -> None:
        source_keys = dict(self.source.named_parameters())
        target_keys = dict(self.target.named_parameters())
        for name, param in target_keys.items():
            if name in source_keys:
                src = source_keys[name]
                if param.shape == src.shape:
                    param.data.copy_(src.data)
                elif src.dim() == param.dim() and param.dim() > 1:
                    param.data = self._adapt_weight(src.data, param.shape)
                else:
                    param.data = src.data.flatten()[: param.numel()].reshape(param.shape)

    @staticmethod
    def _adapt_weight(src: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        src = src.flatten()
        target = torch.zeros(target_shape, device=src.device, dtype=src.dtype)
        target = src[: target.numel()].reshape(target_shape)
        return target


class MorphStep:
    def __init__(self, from_family: str, to_family: str, params: Dict[str, int]) -> None:
        self.from_family = from_family
        self.to_family = to_family
        self.params = params

    def apply(self, model: nn.Module) -> nn.Module:
        if self.from_family == "Transformer" and self.to_family == "MoE":
            return self._morph_to_moe(model)
        elif self.from_family == "CNN" and self.to_family == "Transformer":
            return self._morph_to_transformer(model)
        elif self.from_family == "MoE" and self.to_family == "Transformer":
            return self._morph_to_dense(model)
        else:
            return model

    def _morph_to_moe(self, model: nn.Module) -> nn.Module:
        from .moe import MoELayer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "attn" not in name and "ffn" in name:
                parent_name = name.rsplit(".", 1)[0]
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, name.rsplit(".", 1)[1], MoELayer(
                    d_model=module.in_features,
                    num_experts=self.params.get("num_experts", 8),
                    expert_hidden=self.params.get("expert_hidden", module.out_features),
                    k=self.params.get("k", 2),
                ))
        return model

    def _morph_to_transformer(self, model: nn.Module) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                parent_name = name.rsplit(".", 1)[0]
                parent = dict(model.named_modules())[parent_name]
                linear = nn.Linear(module.in_features * module.kernel_size[0] * module.kernel_size[1], module.out_features)
                setattr(parent, name.rsplit(".", 1)[1], linear)
        return model

    def _morph_to_dense(self, model: nn.Module) -> nn.Module:
        for name, module in model.named_modules():
            if isinstance(module, MoELayer):
                parent_name = name.rsplit(".", 1)[0]
                parent = dict(model.named_modules())[parent_name]
                d_model = module.experts[0][0].in_features
                out = module.experts[0][2].out_features
                dense = nn.Linear(d_model, out)
                setattr(parent, name.rsplit(".", 1)[1], dense)
        return model


class ArchitectureConverter:
    def __init__(self, source_arch: str, target_arch: str) -> None:
        self.source_arch = source_arch
        self.target_arch = target_arch
        self.morph_steps: List[MorphStep] = []

    def add_morph(self, step: MorphStep) -> None:
        self.morph_steps.append(step)

    def convert(self, model: nn.Module) -> nn.Module:
        for step in self.morph_steps:
            model = step.apply(model)
        return model

    def transfer_weights(self, source_model: nn.Module, target_model: nn.Module) -> None:
        transfer = WeightTransfer(source_model, target_model)
        transfer.transfer()
