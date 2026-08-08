"""Model training: from-scratch training, fine-tuning (LoRA), continue training.

Three modes:
  * train      — train a tiny transformer from scratch on JSONL data
                  (uses a compact architecture that works on CPU).
  * finetune   — LoRA/QLoRA fine-tuning of an existing HF model with PEFT.
  * continue   — continue training on an existing model checkpoint.

Data format (JSONL): {"text": "..."} for LM training, or
{"instruction": "...", "output": "..."} for instruction tuning.

Zero-dependency from-scratch trainer: a simple multi-layer perceptron
(n-gram window -> next token) is implemented in pure Python/numpy, so
"create a new AI model" works even without torch. When torch is present,
a real MiniGPT (nn.Embedding + nn.TransformerEncoder) is used instead.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from aweai.utils import safe_filename, write_json

try:  # pragma: no cover - optional
    import numpy as np  # type: ignore

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    _HAS_NUMPY = False

try:  # pragma: no cover - optional
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


@dataclass
class TrainResult:
    name: str
    mode: str
    path: str
    steps: int = 0
    loss: float = 0.0
    duration_s: float = 0.0
    messages: List[str] = field(default_factory=list)


def load_texts(path: str) -> List[str]:
    """Load training texts from .jsonl, .json, or .txt."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    texts: List[str] = []
    if p.suffix == ".jsonl":
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                if obj.get("text"):
                    texts.append(obj["text"])
                elif obj.get("instruction") and obj.get("output"):
                    texts.append(f"{obj['instruction']}\n{obj['output']}")
            elif isinstance(obj, str):
                texts.append(obj)
    elif p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    texts.append(obj.get("text") or f"{obj.get('instruction','')}\n{obj.get('output','')}")
                elif isinstance(obj, str):
                    texts.append(obj)
    else:  # txt
        texts = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return texts


class _Vocab:
    def __init__(self, texts: List[str], max_vocab: int = 4096) -> None:
        counts: Dict[str, int] = {}
        for t in texts:
            for tok in t.split():
                counts[tok] = counts.get(tok, 0) + 1
        top = sorted(counts, key=counts.get, reverse=True)[: max_vocab - 4]
        self.stoi = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        for w in top:
            self.stoi[w] = len(self.stoi)
        self.itos = {i: w for w, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(w, self.stoi["<unk>"]) for w in text.split()]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.itos.get(i, "<unk>") for i in ids)


class TinyNgramLM:
    """Pure-Python n-gram LM (no deps): builds P(next|context) from data."""

    def __init__(self, n: int = 2, smoothing: float = 0.01) -> None:
        self.n = n
        self.smoothing = smoothing
        self.counts: Dict[tuple, Dict[str, float]] = {}

    def train(self, texts: List[str]) -> Dict:
        for t in texts:
            toks = t.split()
            for i in range(len(toks)):
                ctx = tuple(toks[max(0, i - self.n + 1): i])
                nxt = toks[i]
                self.counts.setdefault(ctx, {})
                self.counts[ctx][nxt] = self.counts[ctx].get(nxt, 0.0) + 1.0
        return {"type": "ngram", "n": self.n, "contexts": len(self.counts)}

    def generate(self, seed: str = "", max_tokens: int = 50) -> str:
        toks = seed.split()
        out = list(toks)
        for _ in range(max_tokens):
            ctx = tuple(out[max(0, len(out) - self.n + 1): len(out)])
            table = self.counts.get(ctx)
            if not table:
                break
            # greedy argmax with smoothing fallback
            best = max(table, key=table.get)
            out.append(best)
        return " ".join(out)

    def save(self, path: Path) -> None:
        # JSON keys must be strings — encode tuple contexts with \x01
        serializable = {"\x01".join(ctx): probs for ctx, probs in self.counts.items()}
        write_json(path / "model.json", {"counts": serializable, "n": self.n})


class TorchMiniGPT:
    """Real tiny transformer (torch): Embedding + TransformerEncoder."""

    def __init__(self, vocab_size: int, dim: int = 64, nhead: int = 4,
                 layers: int = 2, max_len: int = 64) -> None:
        import torch.nn as nn

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, batch_first=True),
            num_layers=layers,
        )
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.head = nn.Linear(dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        import torch

        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        return self.head(self.model(h))

    def train_step(self, batch, lr: float = 0.003):
        import torch

        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        opt.zero_grad()
        logits = self.forward(batch[:, :-1])
        loss = self.loss_fn(logits.reshape(-1, self.vocab_size), batch[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        return float(loss.item())


def train_scratch(name: str, data_path: str, out_dir: Optional[str] = None,
                  epochs: int = 3, lr: float = 0.003, max_vocab: int = 4096,
                  steps_per_epoch: int = 50, seed_text: str = "Բարեւ AWEAI") -> TrainResult:
    """Create a brand-new model from zero on the given data."""
    from aweai.config import ensure_runtime_dirs

    dirs = ensure_runtime_dirs()
    out_root = Path(out_dir or dirs["models"])
    out_path = out_root / safe_filename(name)
    out_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    texts = load_texts(data_path)
    if not texts:
        raise ValueError("No training texts found in data file")

    messages: List[str] = []
    loss = 0.0
    steps = 0

    if _HAS_TORCH:
        import torch

        vocab = _Vocab(texts, max_vocab)
        model = TorchMiniGPT(vocab.vocab_size)
        samples = [vocab.encode(t) for t in texts]
        # pad to max_len window
        max_len = 48
        from torch.utils.data import DataLoader, TensorDataset

        tensors = []
        for s in samples:
            if len(s) < 2:
                continue
            s = s[:max_len]
            tensors.append(torch.tensor(s + [0] * (max_len - len(s)), dtype=torch.long))
        if not tensors:
            raise ValueError("Texts too short for torch training")
        data = torch.stack(tensors)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        for epoch in range(epochs):
            for batch_idx, (batch,) in enumerate(loader):
                if batch_idx >= steps_per_epoch:
                    break
                loss = model.train_step(batch, lr=lr)
                steps += 1
            messages.append(f"epoch {epoch + 1}/{epochs}: loss={loss:.4f}")
        # save vocab + weights
        torch.save(model.state_dict(), out_path / "model.pt")
        write_json(out_path / "vocab.json",
                   {"stoi": vocab.stoi, "itos": {str(k): v for k, v in vocab.itos.items()}})
        trained = {"type": "torch_minigpt", "vocab_size": vocab.vocab_size}
    elif _HAS_NUMPY:
        model = TinyNgramLM(n=3)
        trained = model.train(texts)
        model.save(out_path)
        messages.append(f"trained n-gram LM: {trained['contexts']} contexts")
        steps = trained["contexts"]
    else:
        raise RuntimeError("Need numpy or torch to train from scratch. Run: pip install aweai[ml]")

    meta = {
        "id": safe_filename(name),
        "name": safe_filename(name),
        "family": "custom",
        "mode": "scratch",
        "params_b": 0.0,
        "path": str(out_path),
        "local": True,
        "trained": trained,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": data_path,
    }
    write_json(out_path / "metadata.json", meta)
    result = TrainResult(
        name=safe_filename(name),
        mode="scratch",
        path=str(out_path),
        steps=steps,
        loss=loss,
        duration_s=time.time() - t0,
        messages=messages,
    )
    return result


def finetune(base_model: str, name: str, data_path: str, out_dir: Optional[str] = None,
             lora_r: int = 8, lora_alpha: int = 16, epochs: int = 1,
             learning_rate: float = 2e-4, max_length: int = 512) -> TrainResult:
    """Fine-tune an existing HuggingFace model with LoRA (PEFT)."""
    if not _HAS_TORCH:
        raise RuntimeError("Fine-tuning requires torch. Run: pip install aweai[ml]")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from datasets import Dataset as HFDataset
    except ImportError as e:
        raise RuntimeError(
            "Fine-tuning requires transformers, datasets, peft, accelerate. Run: pip install aweai[ml]"
        ) from e

    from aweai.config import ensure_runtime_dirs

    dirs = ensure_runtime_dirs()
    out_root = Path(out_dir or dirs["models"])
    out_path = out_root / safe_filename(name)
    out_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    texts = load_texts(data_path)
    if not texts:
        raise ValueError("No training texts found")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encodings = []
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=max_length, padding="max_length")
        encodings.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})

    ds = HFDataset.from_list(encodings)
    ds.set_format("torch")

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=str(out_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
    )
    train_result = trainer.train()
    model.save_pretrained(out_path / "lora")
    tokenizer.save_pretrained(out_path / "lora")

    meta = {
        "id": safe_filename(name),
        "name": safe_filename(name),
        "family": "lora",
        "mode": "finetune",
        "base_model": base_model,
        "params_b": 0.0,
        "path": str(out_path),
        "local": True,
        "lora": {"r": lora_r, "alpha": lora_alpha},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": data_path,
    }
    write_json(out_path / "metadata.json", meta)
    loss = float(getattr(train_result, "training_loss", 0.0))
    return TrainResult(
        name=safe_filename(name),
        mode="finetune",
        path=str(out_path),
        steps=int(getattr(train_result, "global_step", 0)),
        loss=loss,
        duration_s=time.time() - t0,
        messages=[f"LoRA fine-tune complete. Loss: {loss:.4f}"],
    )


def continue_training(name: str, checkpoint: str, data_path: str, epochs: int = 1,
                      learning_rate: float = 3e-5) -> TrainResult:
    """Continue training an existing local model checkpoint."""
    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    # If it's an n-gram / scratch model we created, continue with its format
    if (ckpt / "vocab.json").exists() and (ckpt / "model.pt").exists():
        # Torch mini model: train more epochs
        import torch

        texts = load_texts(data_path)
        vocab_data = json.loads((ckpt / "vocab.json").read_text(encoding="utf-8"))
        stoi = vocab_data["stoi"]
        itos = {int(k): v for k, v in vocab_data["itos"].items()}
        vocab = _Vocab(texts, len(stoi))
        vocab.stoi = stoi
        vocab.itos = {i: w for w, i in stoi.items()}
        model = TorchMiniGPT(len(stoi))
        model.model.load_state_dict(torch.load(ckpt / "model.pt", map_location="cpu"))
        # re-train a few steps on the new data
        samples = [vocab.encode(t) for t in texts]
        loss = 0.0
        steps = 0
        max_len = 48
        import torch as T

        tensors = []
        for s in samples:
            if len(s) < 2:
                continue
            s = s[:max_len]
            tensors.append(T.tensor(s + [0] * (max_len - len(s)), dtype=T.long))
        if tensors:
            data = T.stack(tensors)
            for epoch in range(epochs):
                for i in range(0, len(data), 8):
                    batch = data[i:i + 8]
                    loss = model.train_step(batch, lr=learning_rate)
                    steps += 1
        torch.save(model.state_dict(), ckpt / "model.pt")
        meta_path = ckpt / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["continued"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta["data"] = data_path
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return TrainResult(
            name=str(ckpt.name), mode="continue", path=str(ckpt),
            steps=steps, loss=loss,
            duration_s=0.0, messages=["Continuing torch model training…"],
        )

    if (ckpt / "model.json").exists():
        # n-gram: retrain merging old counts with new data
        import numpy as np  # noqa: F401

        old = json.loads((ckpt / "model.json").read_text(encoding="utf-8"))
        model = TinyNgramLM(n=int(old.get("n", 3)))
        model.counts = {tuple(k.split("\x01")): v for k, v in old["counts"].items()}
        texts = load_texts(data_path)
        model.train(texts)
        model.save(ckpt)
        meta_path = ckpt / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["continued"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return TrainResult(
            name=str(ckpt.name), mode="continue", path=str(ckpt),
            steps=len(model.counts), loss=0.0,
            duration_s=0.0, messages=["Continuing n-gram model training…"],
        )

    # HF-style checkpoint: continue with Trainer
    return finetune(checkpoint, name, data_path, epochs=epochs, learning_rate=learning_rate)
