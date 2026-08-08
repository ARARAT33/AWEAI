"""Data augmentation: text, images (numpy), noise."""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

import numpy as np


def text_augment(text: str, n: int = 1, seed: Optional[int] = None) -> List[str]:
    rng = random.Random(seed)
    words = text.split()
    if not words:
        return [text]
    out: List[str] = []
    for _ in range(n):
        w = list(words)
        op = rng.choice(["shuffle", "drop", "repeat", "swap"])
        if op == "shuffle" and len(w) > 2:
            i, j = rng.sample(range(len(w)), 2)
            w[i], w[j] = w[j], w[i]
        elif op == "drop" and len(w) > 3:
            w.pop(rng.randrange(len(w)))
        elif op == "repeat":
            w.insert(rng.randrange(len(w) + 1), w[rng.randrange(len(w))])
        elif op == "swap" and len(w) > 1:
            i = rng.randrange(len(w))
            w[i] = w[i][::-1]
        out.append(" ".join(w))
    return out


def noise_augment(X: np.ndarray, noise_std: float = 0.01, n: int = 1, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    outs = [X]
    for _ in range(n):
        outs.append(X + rng.normal(0.0, noise_std, size=X.shape))
    return np.concatenate(outs, axis=0)


def image_augment_np(
    images: np.ndarray,
    n: int = 1,
    seed: Optional[int] = None,
    flip: bool = True,
    shift: int = 1,
    noise: float = 0.01,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N, D = images.shape
    H = int(round(D ** 0.5))
    W = D // H if H > 0 else D
    if H * W != D:
        outs = [images]
        for _ in range(n):
            outs.append(images + rng.normal(0.0, noise, size=images.shape))
        return np.concatenate(outs, axis=0)
    outs = [images]
    for _ in range(n):
        batch = images.reshape(N, H, W)
        for i in range(N):
            if flip and rng.random() < 0.5:
                batch[i] = batch[i, :, ::-1]
            if shift:
                dx = rng.integers(-shift, shift + 1)
                dy = rng.integers(-shift, shift + 1)
                batch[i] = np.roll(np.roll(batch[i], dx, axis=1), dy, axis=0)
            if noise:
                batch[i] = batch[i] + rng.normal(0.0, noise, size=(H, W))
        outs.append(np.clip(batch.reshape(N, D), 0.0, 1.0))
    return np.concatenate(outs, axis=0)


def augment(
    X: Optional[np.ndarray] = None,
    texts: Optional[List[str]] = None,
    images: Optional[np.ndarray] = None,
    n: int = 1,
    seed: Optional[int] = None,
) -> dict:
    out: dict = {}
    if X is not None:
        out["X"] = noise_augment(X, n=n, seed=seed)
    if texts is not None:
        out["texts"] = texts + [t for txt in texts for t in text_augment(txt, n=n, seed=seed)]
    if images is not None:
        out["images"] = image_augment_np(images, n=n, seed=seed)
    return out
