"""Natural-language action runner.

Supported intents (multilingual keywords):
  * train / create model / նոր մոդել / создать модель ...
        -> train a new model on the given data
  * finetune / fine-tune / дообучить ...
        -> fine-tune an existing model
  * rag / index / vectorize ...
        -> index documents for retrieval
  * agent / run agent / ագենտ ...
        -> run the ReAct agent on a task
  * hardware / resources / սարքավորում ...
        -> report hardware and best model
  * serve / ui / open ui / բացել ui ...
        -> start the browser UI

The runner matches intent keywords, extracts a data path / task text from
the sentence, executes the pipeline, and returns a structured report.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

INTENTS = {
    "train": {
        "en": ["train", "create model", "new model", "teach"],
        "hy": ["մարզել", "նոր մոդել", "ստեղծել մոդել", "սովորեցնել"],
        "ru": ["обучить", "создать модель", "новая модель"],
    },
    "finetune": {
        "en": ["finetune", "fine-tune", "tune"],
        "hy": ["fine-tuning", "ճշգրտել"],
        "ru": ["дообучить", "файнтюн"],
    },
    "rag": {
        "en": ["rag", "index", "vectorize", "knowledge base"],
        "hy": ["ռագ", "ինդեքսավորել"],
        "ru": ["раг", "индексировать"],
    },
    "agent": {
        "en": ["agent", "automate", "do task"],
        "hy": ["ագենտ", "ավտոմատացնել"],
        "ru": ["агент", "автоматизировать"],
    },
    "hardware": {
        "en": ["hardware", "resources", "what can i run"],
        "hy": ["սարքավորում", "ռեսուրսներ"],
        "ru": ["железо", "ресурсы"],
    },
    "serve": {
        "en": ["serve", "open ui", "web ui", "start server", "dashboard"],
        "hy": ["բացել ui", "վեբ ինտերֆեյս", "սերվեր"],
        "ru": ["открыть ui", "веб интерфейс", "сервер"],
    },
}


def _find_path(text: str) -> Optional[str]:
    """Extract a filesystem path from free text."""
    m = re.search(r"['\"`]([^'\"`]+)['\"`]", text)
    if m:
        return m.group(1)
    for token in text.split():
        token = token.rstrip(",.;")
        if "/" in token or token.endswith((".jsonl", ".json", ".txt", ".csv", ".md")):
            return token
        if token.startswith(("~/", "./", "../")) or token.startswith("/"):
            return token
    return None


def parse_action(text: str, lang: str = "en") -> Dict:
    """Parse a natural-language action into an intent + params."""
    lowered = text.lower()
    for intent, langs in INTENTS.items():
        for keywords in langs.values():
            for kw in keywords:
                if kw.lower() in lowered:
                    params: Dict[str, str] = {}
                    path = _find_path(text)
                    if path:
                        params["path"] = path
                    if intent == "train":
                        params["name"] = _extract_name(text, "model")
                    if intent == "finetune":
                        params["name"] = _extract_name(text, "tuned")
                        m = re.search(r"base[:\s]+([\w./-]+)", text)
                        if m:
                            params["base"] = m.group(1)
                    return {"intent": intent, "params": params, "language": lang}
    return {"intent": "chat", "params": {"text": text}, "language": lang}


def _extract_name(text: str, fallback: str) -> str:
    m = re.search(r"(?:named|called|կոչվում|называется)[:\s]+([\w-]+)", text)
    return m.group(1) if m else f"{fallback}_{int(time.time()) % 100000}"


class ActionsRunner:
    """Executes parsed intents and returns a structured report."""

    def __init__(self, lang: str = "en", verbose: bool = True) -> None:
        self.lang = lang
        self.verbose = verbose
        self.results: List[Dict] = []

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[actions] {msg}")

    def run(self, text: str) -> Dict:
        parsed = parse_action(text, self.lang)
        intent = parsed["intent"]
        params = parsed["params"]
        self._log(f"intent={intent} params={params}")

        if intent == "train":
            return self._run_train(params)
        if intent == "finetune":
            return self._run_finetune(params)
        if intent == "rag":
            return self._run_rag(params)
        if intent == "agent":
            return self._run_agent(params)
        if intent == "hardware":
            return self._run_hardware()
        if intent == "serve":
            return {"intent": intent, "status": "ok",
                    "message": "Run `aweai serve` to open the UI in your browser."}
        return {"intent": "chat", "status": "ok",
                "message": "Could not detect an action. Try: 'train a new model with this data'."}

    def _run_train(self, params: Dict) -> Dict:
        from aweai.models.trainer import train_scratch

        path = params.get("path")
        if not path:
            # generate a tiny sample dataset so the demo always works
            path = self._sample_data()
        name = params.get("name", f"model_{int(time.time()) % 100000}")
        self._log(f"training {name} on {path}")
        try:
            result = train_scratch(name, path)
            report = {
                "intent": "train", "status": "ok",
                "model": name, "path": result.path,
                "steps": result.steps, "loss": round(result.loss, 4),
                "duration_s": round(result.duration_s, 2),
                "messages": result.messages,
            }
            self.results.append(report)
            return report
        except Exception as e:
            return {"intent": "train", "status": "error", "error": str(e)}

    def _run_finetune(self, params: Dict) -> Dict:
        from aweai.models.trainer import finetune

        path = params.get("path")
        if not path:
            path = self._sample_data()
        base = params.get("base", "Qwen/Qwen2.5-0.5B-Instruct")
        name = params.get("name", f"tuned_{int(time.time()) % 100000}")
        self._log(f"fine-tuning {base} on {path}")
        try:
            result = finetune(base, name, path, epochs=1)
            return {
                "intent": "finetune", "status": "ok",
                "model": name, "path": result.path,
                "messages": result.messages,
            }
        except Exception as e:
            return {"intent": "finetune", "status": "error",
                    "error": str(e),
                    "hint": "Install ML extras: pip install aweai[ml]"}

    def _run_rag(self, params: Dict) -> Dict:
        from aweai.rag.engine import RAGEngine

        engine = RAGEngine()
        path = params.get("path")
        if path:
            p = Path(path)
            if p.is_dir():
                added = engine.index_directory(str(p))
            else:
                added = engine.index_file(str(p))
        else:
            # index sample docs
            sample = Path(tempfile.mkdtemp(prefix="aweai_rag_")) / "sample.txt"
            sample.write_text(
                "AWEAI is a universal AI toolbox. It supports RAG, agents, training and a 12-language UI.\n"
                "Armenia is a country in the South Caucasus. Yerevan is its capital.\n",
                encoding="utf-8",
            )
            added = engine.index_file(str(sample))
        return {
            "intent": "rag", "status": "ok",
            "added_chunks": added,
            "stats": engine.stats(),
        }

    def _run_agent(self, params: Dict) -> Dict:
        from aweai.agents.engine import AgentEngine

        task_text = params.get("path") or params.get("text") or "Summarize what AWEAI can do."
        agent = AgentEngine.create()
        result = agent.run(task_text, max_steps=3, verbose=False)
        return {
            "intent": "agent", "status": "ok",
            "task": task_text,
            "final": result["final"],
            "tool_calls": result["tool_calls"],
        }

    def _run_hardware(self) -> Dict:
        from aweai.hardware import detect
        from aweai.models.selector import pick_best_model, suggest_models

        hw = detect()
        best = pick_best_model(hw)
        return {
            "intent": "hardware", "status": "ok",
            "hardware": hw.to_dict(),
            "best_model": best["id"] if best else None,
            "suggestions": [m["id"] for m in suggest_models(hw, limit=3)],
        }

    def _sample_data(self) -> str:
        path = Path(tempfile.mkdtemp(prefix="aweai_train_")) / "sample.jsonl"
        lines = [
            {"text": "AWEAI is the universal AI toolbox created in Armenia."},
            {"text": "Բարեւ աշխարհ։ AWEAI-ը համընդհանուր AI գործիք է։"},
            {"text": "The capital of Armenia is Yerevan, one of the oldest cities in the world."},
            {"text": "AWEAI supports RAG, agents, fine-tuning and a 12-language interface."},
            {"text": "Machine learning is the study of algorithms that improve with data."},
        ]
        path.write_text(
            "\n".join(json.dumps(l, ensure_ascii=False) for l in lines), encoding="utf-8"
        )
        return str(path)
