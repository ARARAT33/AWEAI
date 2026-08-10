from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWKVBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, layer_norm_epsilon: float = 1e-5) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.att = nn.Linear(d_model, d_model * 3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.time_mix_k = nn.Parameter(torch.randn(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, C = x.size()
        xx = self.ln1(x)
        xk, xr, xv = self.att(xx).split(C, dim=-1)
        r = torch.sigmoid(xr * self.time_mix_r + xx * (1 - self.time_mix_r))
        k = (xk * self.time_mix_k + xx * (1 - self.time_mix_k)).reshape(B, L, C, 1)
        v = xv.reshape(B, L, 1, C)
        wkv = torch.matmul(k, v).reshape(B, L, C)
        return x + self.ln2(self.out(r * wkv)), state


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: str = "auto") -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = dt_rank
        if dt_rank == "auto":
            self.dt_rank = d_model // 16
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.size()
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[..., :L]
        x = x.transpose(1, 2)
        x = self.act(x)
        A = -torch.exp(self.A.float())
        dt = self.dt_proj(self.x_proj(x)[..., :self.dt_rank])
        dA = torch.exp(torch.einsum("bld,dn->bdln", dt, A))
        x_dbl = self.x_proj(x)[..., self.dt_rank:]
        dx = torch.einsum("bdl,bln->bdn", dA, x_dbl)
        x = self.out_proj(x + dx)
        return self.norm(x + z)


class RetNetBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, double_v_dim: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, past_key_values: Optional[tuple] = None) -> tuple[torch.Tensor, tuple]:
        qkv = self.qkv(self.norm1(x)).reshape(x.size(0), x.size(1), 3, self.num_heads, self.head_dim).transpose(1, 3)
        q, k, v = qkv.unbind(2)
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(x.size(0), x.size(1), -1)
        x = x + self.proj(out)
        x = x + self.ffn(self.norm2(x))
        return x, past_key_values


class GriffinBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.local_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.local_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        return x + self.ffn(self.norm2(x))


class UniversalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_steps: int = 8) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.halt = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(x)
        for t in range(self.max_steps):
            h = h + self.attn(h, h, h, attn_mask=mask)[0]
            h = h + self.ffn(self.norm(h))
        return h, torch.zeros(x.size(0), 1, device=x.device)


class HybridMoETransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_experts: int, expert_hidden: int, num_layers: int = 6) -> None:
        super().__init__()
        from .moe import MoEBlock, MoELayer
        layers = []
        for i in range(num_layers):
            if i % 2 == 1:
                layers.append(nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=expert_hidden))
            else:
                layers.append(MoEBlock(d_model, num_experts, expert_hidden))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aux_loss = 0.0
        for layer in self.layers:
            if hasattr(layer, "moe"):
                x, aux = layer(x)
                aux_loss = aux_loss + aux
            else:
                x = layer(x)
        return self.norm(x), aux_loss
