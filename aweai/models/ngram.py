"""N-gram language model with fixed tuple-key serialization."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from aweai.models.base import BaseModel
from aweai.utils import serialize_ngram_key, tokenize


class NGramLM(BaseModel):
    model_type = "ngram"

    def __init__(self, n: int = 3, **params):
        super().__init__(n=n, **params)
        self.n = int(n)
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.vocab: List[str] = []
        self.total = 0

    def fit(self, texts, y=None, **kw):
        if isinstance(texts, str): texts = [texts]
        tokens_all: List[str] = []
        for t in texts:
            toks = tokenize(str(t)); tokens_all.extend(["<s>"] * (self.n - 1) + toks + ["<e>"])
        self.vocab = sorted(set(tokens_all))
        for i in range(len(tokens_all) - self.n + 1):
            ctx = tuple(tokens_all[i:i + self.n - 1]); nxt = tokens_all[i + self.n - 1]
            self.counts[serialize_ngram_key(ctx)][nxt] += 1; self.total += 1
        self.trained = True; self.metrics["tokens"] = len(tokens_all); self.metrics["contexts"] = len(self.counts)
        return self

    def next_token(self, ctx: Sequence[str]) -> str:
        dist = self.counts.get(serialize_ngram_key(tuple(ctx)))
        if not dist: return random.choice(self.vocab) if self.vocab else "<e>"
        items = list(dist.items()); r = random.uniform(0, sum(c for _, c in items)); acc = 0
        for tok, c in items:
            acc += c
            if r <= acc: return tok
        return items[-1][0]

    def generate(self, max_tokens: int = 20, seed: Optional[int] = None) -> str:
        random.seed(seed); ctx = ["<s>"] * (self.n - 1); out = []
        for _ in range(max_tokens):
            tok = self.next_token(ctx)
            if tok == "<e>": break
            if tok != "<s>": out.append(tok)
            ctx = (ctx + [tok])[-(self.n - 1):]
        return " ".join(out)

    def predict(self, X): return self.generate(max_tokens=10)
    def state_dict(self): return {"counts": {k: dict(v) for k, v in self.counts.items()}, "vocab": self.vocab, "total": self.total}
    def load_state(self, state):
        self.counts = defaultdict(lambda: defaultdict(int))
        for k, v in state.get("counts", {}).items(): self.counts[k].update(v)
        self.vocab = state.get("vocab", []); self.total = int(state.get("total", 0)); self.trained = True
    def export_json(self):
        data = self.to_dict(); data["state"] = self.state_dict(); return data


# Stable public compatibility name used by the smoke-test API.
NGramModel = NGramLM
