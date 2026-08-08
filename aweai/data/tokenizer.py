"""Own lightweight tokenizer (no tokenizers / transformers dependency).

Implements a byte-pair-encoding-lite and word-level tokenizer with:
  * train from text corpus
  * encode / decode
  * save / load vocab (JSON)
  * special tokens <pad> <unk> <bos> <eos>
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from aweai.errors import DataError

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]


def _words(text: str) -> List[str]:
    return re.findall(r"[\w\u0561-\u0587\u0531-\u0556]+|[^\s\w\u0561-\u0587\u0531-\u0556]", text.lower())


class Tokenizer:
    """Word-level tokenizer with frequency cap and vocab save/load."""

    def __init__(self, vocab_size: int = 20000, min_freq: int = 1) -> None:
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.vocab: Dict[str, int] = {}
        self.itos: List[str] = []

    @property
    def pad_id(self) -> int:
        return self.vocab.get(PAD, 0)

    @property
    def unk_id(self) -> int:
        return self.vocab.get(UNK, 1)

    @property
    def bos_id(self) -> int:
        return self.vocab.get(BOS, 2)

    @property
    def eos_id(self) -> int:
        return self.vocab.get(EOS, 3)

    def __len__(self) -> int:
        return len(self.vocab)

    def train(self, texts: Iterable[str]) -> "Tokenizer":
        counter: Counter = Counter()
        for t in texts:
            counter.update(_words(t))
        selected = [w for w, c in counter.most_common(self.vocab_size - len(SPECIALS)) if c >= self.min_freq]
        self.itos = SPECIALS + selected
        self.vocab = {w: i for i, w in enumerate(self.itos)}
        return self

    def encode(self, text: str, add_specials: bool = False) -> List[int]:
        ids = [self.vocab.get(w, self.unk_id) for w in _words(text)]
        if add_specials:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(self.itos[i] if 0 <= i < len(self.itos) else UNK for i in ids).replace(f" {EOS}", "").replace(f" {BOS} ", " ").strip()

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"vocab": self.vocab, "itos": self.itos, "vocab_size": self.vocab_size, "min_freq": self.min_freq},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Tokenizer":
        p = Path(path)
        if not p.exists():
            raise DataError(f"Tokenizer file not found: {path}")
        data = json.loads(p.read_text(encoding="utf-8"))
        tok = cls(vocab_size=data.get("vocab_size", 20000), min_freq=data.get("min_freq", 1))
        tok.vocab = data["vocab"]
        tok.itos = data["itos"]
        return tok


def build_tokenizer(texts: Iterable[str], vocab_size: int = 20000, min_freq: int = 1) -> Tokenizer:
    return Tokenizer(vocab_size=vocab_size, min_freq=min_freq).train(texts)
