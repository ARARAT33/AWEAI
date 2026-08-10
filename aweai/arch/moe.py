from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKGating(nn.Module):
    def __init__(self, d_model: int, num_experts: int, k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.w_gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.w_gate(x)
        topk_logits, topk_indices = logits.topk(self.k, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1)
        return topk_probs, topk_indices, logits


class MoELayer(nn.Module):
    def __init__(self, d_model: int, num_experts: int, expert_hidden: int, k: int = 2, capacity_factor: float = 1.0, drop_tokens: bool = True) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        self.gate = TopKGating(d_model, num_experts, k)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, expert_hidden), nn.GELU(), nn.Linear(expert_hidden, d_model))
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        flat = x.reshape(-1, d_model)
        num_tokens = flat.shape[0]
        capacity = int(self.capacity_factor * num_tokens / self.num_experts)
        topk_probs, topk_indices, logits = self.gate(flat)
        output = torch.zeros_like(flat)
        counts = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.k):
            indices = topk_indices[:, i]
            probs = topk_probs[:, i]
            for e in range(self.num_experts):
                mask = (indices == e)
                if not mask.any():
                    continue
                selected = flat[mask]
                cap = min(capacity, selected.shape[0])
                if self.drop_tokens and selected.shape[0] > cap:
                    perm = torch.randperm(selected.shape[0], device=x.device)[:cap]
                    selected = selected[perm]
                    mask_indices = mask.nonzero(as_tuple=False).squeeze(1)[perm]
                    counts[e] = cap
                else:
                    mask_indices = mask.nonzero(as_tuple=False).squeeze(1)
                    counts[e] = selected.shape[0]
                expert_out = self.experts[e](selected)
                output[mask_indices] += probs[mask][:len(mask_indices)].unsqueeze(-1) * expert_out
        output = output.reshape(batch_size, seq_len, d_model)
        aux_loss = self._load_balance_loss(logits, topk_indices, num_tokens)
        return output, aux_loss

    def _load_balance_loss(self, logits: torch.Tensor, topk_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        expert_prob = probs.sum(dim=0)
        routing_probs = torch.zeros(self.num_experts, device=logits.device)
        for i in range(self.k):
            routing_probs.index_add_(0, topk_indices[:, i], torch.ones_like(topk_indices[:, i], dtype=torch.float))
        routing_probs = routing_probs / (num_tokens * self.k)
        loss = self.num_experts * (expert_prob * routing_probs).sum()
        return loss


class MoEBlock(nn.Module):
    def __init__(self, d_model: int, num_experts: int, expert_hidden: int, k: int = 2, capacity_factor: float = 1.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, num_experts, expert_hidden, k, capacity_factor)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(x)
        h, aux_loss = self.moe(h)
        return self.dropout(h + x), aux_loss


class SwitchTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_experts: int, expert_hidden: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, num_experts, expert_hidden, k=1)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + h)
        h, aux = self.moe(x)
        x = self.norm2(x + h)
        return x, aux


class GLaMBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_experts: int, expert_hidden: int, topk: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, num_experts, expert_hidden, k=topk, capacity_factor=1.25)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, aux = self.moe(self.norm(x))
        return h + x, aux


class MixtralMoEBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_experts: int, expert_hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, num_experts, expert_hidden, k=2, capacity_factor=1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, aux = self.moe(self.norm(x))
        return h + x, aux
