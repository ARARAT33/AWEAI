from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


class RoPEPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin", emb.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        return self._apply_rotary_pos_emb(x[..., :seq_len, :], self.cos[:, :, :seq_len, :], self.sin[:, :, seq_len, :])

    @staticmethod
    def _apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class ALiBiPositionalEmbedding(nn.Module):
    def __init__(self, num_heads: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        slopes = torch.tensor([2 ** (-8 * (i + 1) / num_heads) for i in range(num_heads)])
        self.register_buffer("bias", -torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0) * slopes.unsqueeze(1))

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        seq_len = attn.size(-1)
        return attn + self.bias[:, :, :seq_len, :seq_len]


class FlashAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).transpose(1, 3)
        q, k, v = qkv.unbind(dim=2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.proj(out)


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, d_model * 2, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(x).reshape(B, L, 2, 1, self.head_dim).transpose(1, 3)
        k, v = kv.unbind(dim=2)
        k = k.expand(-1, self.num_heads, -1, -1)
        v = v.expand(-1, self.num_heads, -1, -1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.proj(out)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_groups: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert num_heads % num_groups == 0
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, d_model * 2, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(x).reshape(B, L, 2, self.num_groups, self.head_dim).transpose(1, 3)
        k, v = kv.unbind(dim=2)
        k = k.unsqueeze(1).expand(-1, self.heads_per_group, -1, -1, -1).flatten(1, 2)
        v = v.unsqueeze(1).expand(-1, self.heads_per_group, -1, -1, -1).flatten(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        return self.proj(out)


class TransformerXLBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = FlashAttention(d_model, num_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, mask))
        return self.norm2(x + self.ffn(x))


class ReformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, lsh_num_buckets: int = 8) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = FlashAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.lsh_num_buckets = lsh_num_buckets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.ffn(x))


class PerformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, feature_map: str = "relu") -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiQueryAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.ffn(x))


class LinformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, seq_len: int, k: int = 256) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.k_proj = nn.Linear(seq_len, k, bias=False)
        self.v_proj = nn.Linear(seq_len, k, bias=False)
        self.attn = FlashAttention(d_model, num_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.ffn(x))
