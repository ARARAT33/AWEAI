"""Inference engine.

Two backends:
  * "local" — HuggingFace transformers pipeline (torch), quantized when GPU
    is weak; falls back gracefully.
  * "fallback" — a tiny built-in deterministic generator so AWEAI always
    works even without torch/transformers installed (lightweight, free).

The LLM class also supports API-based generation through aweai.models.apis.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from aweai.models import get_model
from aweai.hardware import detect


class TinyBrain:
    """A zero-dependency micro language model (rule + ngram based).

    Not a real neural net, but it gives meaningful echo / template answers
    for demo and offline use. When transformers are installed, prefer LocalLLM.
    """

    def __init__(self) -> None:
        self.responses = [
            "I'm AWEAI running in lightweight mode. Install the ML extras (`pip install aweai[ml]`) to enable real transformer models.",
            "This is the fallback brain. For full power: pip install aweai[all].",
        ]

    def generate(self, prompt: str, max_tokens: int = 128, **kwargs) -> str:
        p = prompt.strip().lower()
        if any(w in p for w in ("hello", "hi", "բարև", "բարի", "привет", "你好", "hola")):
            return "Hello! AWEAI is ready. Ask me anything or open the UI with `aweai serve`."
        if "name" in p:
            return "I am AWEAI — your universal AI toolbox."
        if "?" in prompt:
            base = prompt.strip().rstrip("?")[-160:]
            return f"Good question: {base}… (install `aweai[ml]` for a smarter model)"
        if len(prompt.strip()) > 10:
            return f"I understood your message: {prompt.strip()[:200]} (fallback mode)"
        return self.responses[0]


class LocalLLM:
    """Real local inference via HuggingFace transformers."""

    def __init__(self, model_id: str = "qwen2.5-0.5b", device: Optional[str] = None) -> None:
        self.model_id = model_id
        self.meta = get_model(model_id) or {"id": model_id, "family": "custom"}
        self.device = device or self._pick_device()
        self._pipeline = None

    def _pick_device(self) -> str:
        hw = detect()
        if hw.torch_cuda:
            return "cuda"
        if hw.torch_mps:
            return "mps"
        return "cpu"

    def _load(self):
        if self._pipeline is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(self.meta.get("hf", self.model_id))
        model = AutoModelForCausalLM.from_pretrained(
            self.meta.get("hf", self.model_id),
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
        if self.device != "cuda":
            model.to(self.device)
        self._pipeline = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device=self.device
        )

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, **kwargs) -> str:
        self._load()
        out = self._pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self._pipeline.tokenizer.eos_token_id,
        )
        text = out[0]["generated_text"]
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()


class LLM:
    """Unified facade: local or API depending on config."""

    def __init__(self, model_id: Optional[str] = None, backend: str = "auto") -> None:
        from aweai.config import get_config

        self.cfg = get_config()
        self.model_id = model_id or self.cfg.get("default_model")
        self.backend = backend or self.cfg.get("model_backend", "auto")
        self._engine = None

    def _resolve_engine(self):
        if self._engine is not None:
            return self._engine
        if self.backend == "api":
            from aweai.models.apis import APIManager

            self._engine = APIManager()
            return self._engine
        # try local transformers first
        try:
            from transformers import pipeline  # noqa: F401

            if not self.model_id:
                from aweai.models.selector import pick_best_model

                best = pick_best_model()
                self.model_id = best["id"] if best else "qwen2.5-0.5b"
            self._engine = LocalLLM(self.model_id)
            return self._engine
        except Exception:
            self._engine = TinyBrain()
            return self._engine

    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        return self._resolve_engine().generate(prompt, max_tokens=max_tokens, **kwargs)

    def chat(self, messages: List[Dict], max_tokens: int = 512, **kwargs) -> str:
        """Chat-style call; builds a prompt from messages."""
        if isinstance(self._resolve_engine(), (LocalLLM, TinyBrain)):
            prompt = "\n".join(
                f"{m.get('role','user')}: {m.get('content','')}" for m in messages
            )
            prompt += "\nassistant:"
            return self.generate(prompt, max_tokens=max_tokens, **kwargs)
        return self._engine.chat(messages, max_tokens=max_tokens, **kwargs)
