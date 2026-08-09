"""AWEAI AI-agent tools — prompts, chains, memory, evals, RAG helpers.

Each tool has a unique purpose. Everything works offline with pure Python;
optional AI providers (BYOK) are wired through aweai.integrations.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from aweai.config import ensure_runtime_dirs
from aweai.tools.registry import tool


def _mem_path() -> Path:
    return ensure_runtime_dirs()["data"] / "aweai_agent_memory.json"


def _load_mem() -> Dict[str, Any]:
    p = _mem_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_mem(mem: Dict[str, Any]) -> None:
    _mem_path().write_text(json.dumps(mem, ensure_ascii=False, indent=2), encoding="utf-8")


@tool("prompt_templates", "aiagents", "List built-in prompt templates")
def prompt_templates() -> Dict[str, Any]:
    templates = {
        "system_default": "You are AWEAI, a powerful AI model factory assistant.",
        "system_expert": "You are an expert {domain} engineer with deep knowledge.",
        "summarize": "Summarize the following text concisely:\n\n{text}",
        "translate": "Translate the following text to {language}:\n\n{text}",
        "extract": "Extract key facts, entities and numbers from:\n\n{text}",
        "classify": "Classify the following text into one of {labels}:\n\n{text}",
        "generate_code": "Write {language} code that {task}. Include comments.",
        "explain": "Explain the following concept simply:\n\n{concept}",
        "review": "Review this code for bugs, style and security:\n\n{code}",
        "brainstorm": "Brainstorm {n} creative ideas about: {topic}",
    }
    return {"templates": templates, "count": len(templates)}


@tool("prompt_fill", "aiagents", "Fill a prompt template with variables")
def prompt_fill(template: str, variables: str = "{}") -> Dict[str, Any]:
    vars_data = json.loads(variables) if isinstance(variables, str) else variables
    try:
        filled = template.format(**vars_data)
    except KeyError as e:
        return {"error": f"missing variable {e}"}
    return {"filled": filled}


@tool("prompt_build", "aiagents", "Build a prompt from system + user + context parts")
def prompt_build(system: str = "", user: str = "", context: str = "") -> Dict[str, Any]:
    parts = []
    if system:
        parts.append(f"System: {system}")
    if context:
        parts.append(f"Context: {context}")
    if user:
        parts.append(f"User: {user}")
    return {"prompt": "\n\n".join(parts)}


@tool("agent_chat", "aiagents", "Chat with a configured AI provider (BYOK, falls back to echo)")
def agent_chat(provider: str = "openai", message: str = "hello") -> Dict[str, Any]:
    try:
        from aweai.integrations import chat

        return {"provider": provider, "message": message, "response": chat(provider, message)}
    except Exception:
        return {
            "provider": provider,
            "message": message,
            "response": f"[offline mode] echo: {message}",
            "offline": True,
        }


@tool("agent_chain", "aiagents", "Run a chain of prompts sequentially (JSON list of steps)")
def agent_chain(steps: str) -> Dict[str, Any]:
    step_list = json.loads(steps) if isinstance(steps, str) else steps
    if not isinstance(step_list, list):
        return {"error": "steps must be a JSON list"}
    outputs = []
    for i, step in enumerate(step_list, 1):
        prompt = step.get("prompt", step.get("message", str(step)))
        provider = step.get("provider", "openai")
        try:
            from aweai.integrations import chat

            resp = chat(provider, prompt)
        except Exception:
            resp = f"[offline] {prompt}"
        outputs.append({"step": i, "name": step.get("name", f"step{i}"), "response": resp})
    return {"chain": outputs, "count": len(outputs)}


@tool("memory_store", "aiagents", "Store a key-value pair in agent memory (JSON file)")
def memory_store(key: str, value: str) -> Dict[str, Any]:
    mem = _load_mem()
    mem[key] = {"value": value, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _save_mem(mem)
    return {"key": key, "stored": True}


@tool("memory_get", "aiagents", "Read a value from agent memory")
def memory_get(key: str) -> Dict[str, Any]:
    mem = _load_mem()
    if key in mem:
        return {"key": key, "value": mem[key]}
    return {"key": key, "value": None}


@tool("memory_list", "aiagents", "List all keys in agent memory")
def memory_list() -> Dict[str, Any]:
    mem = _load_mem()
    return {"keys": list(mem.keys()), "count": len(mem)}


@tool("memory_delete", "aiagents", "Delete a key from agent memory")
def memory_delete(key: str) -> Dict[str, Any]:
    mem = _load_mem()
    existed = key in mem
    mem.pop(key, None)
    _save_mem(mem)
    return {"key": key, "deleted": existed}


@tool("memory_clear", "aiagents", "Clear all agent memory")
def memory_clear() -> Dict[str, Any]:
    _save_mem({})
    return {"cleared": True}


@tool("eval_accuracy", "aiagents", "Evaluate a list of (predicted, expected) pairs for accuracy")
def eval_accuracy(pairs: str) -> Dict[str, Any]:
    data = json.loads(pairs) if isinstance(pairs, str) else pairs
    correct = sum(1 for p, e in data if p == e)
    return {"correct": correct, "total": len(data), "accuracy": correct / len(data) if data else 0}


@tool("eval_exact_match", "aiagents", "Exact-match score between generated and reference answers")
def eval_exact_match(generated: str, reference: str) -> Dict[str, Any]:
    match = generated.strip().lower() == reference.strip().lower()
    return {"exact_match": 1.0 if match else 0.0, "match": match}


@tool("eval_f1_words", "aiagents", "F1 score over word overlap between generated and reference")
def eval_f1_words(generated: str, reference: str) -> Dict[str, Any]:
    import re

    def words(s: str):
        return set(re.findall(r"[a-z0-9]+", s.lower()))

    g, r = words(generated), words(reference)
    if not g or not r:
        return {"f1": 0.0}
    inter = len(g & r)
    precision = inter / len(g)
    recall = inter / len(r)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4)}


@tool("rag_build", "aiagents", "Build a simple RAG index from documents (JSON list of text chunks)")
def rag_build(documents: str) -> Dict[str, Any]:
    docs = json.loads(documents) if isinstance(documents, str) else documents
    if not isinstance(docs, list):
        return {"error": "documents must be a JSON list"}
    index = {}
    for i, d in enumerate(docs):
        chunk = d.get("text", str(d)) if isinstance(d, dict) else str(d)
        index[str(i)] = {"id": str(i), "text": chunk, "tokens": len(chunk.split())}
    path = ensure_runtime_dirs()["data"] / "aweai_rag_index.json"
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"indexed": len(index), "path": str(path)}


@tool("rag_search", "aiagents", "Keyword-search the built RAG index (returns top-k chunks)")
def rag_search(query: str, top_k: int = 3) -> Dict[str, Any]:
    path = ensure_runtime_dirs()["data"] / "aweai_rag_index.json"
    if not path.exists():
        return {"error": "no RAG index found; run rag_build first"}
    index = json.loads(path.read_text(encoding="utf-8"))
    q = query.lower()
    scored = []
    for item in index.values():
        text = item.get("text", "")
        score = text.lower().count(q)
        if score:
            scored.append({"id": item["id"], "score": score, "text": text[:300]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"query": query, "results": scored[:top_k], "count": len(scored)}


@tool("agent_plan", "aiagents", "Generate a step-by-step plan for a goal (template-based)")
def agent_plan(goal: str, steps: int = 5) -> Dict[str, Any]:
    plan = [
        {"step": i + 1, "action": f"{goal} — phase {i + 1}: define, build, verify"}
        for i in range(steps)
    ]
    return {"goal": goal, "plan": plan, "count": len(plan)}


@tool("agent_decompose", "aiagents", "Decompose a task into subtasks (JSON list)")
def agent_decompose(task: str, parts: int = 3) -> Dict[str, Any]:
    return {"task": task, "subtasks": [{"id": i + 1, "name": f"subtask {i + 1} of {task}"} for i in range(parts)]}


@tool("context_pack", "aiagents", "Pack multiple context strings into one block (separated)")
def context_pack(parts: str, separator: str = "\n---\n") -> Dict[str, Any]:
    data = json.loads(parts) if isinstance(parts, str) else parts
    return {"packed": separator.join(str(p) for p in data)}


@tool("token_estimate", "aiagents", "Estimate token count of a text (chars/4 heuristic)")
def token_estimate(text: str) -> Dict[str, Any]:
    return {"chars": len(text), "words": len(text.split()), "tokens_estimate": max(1, len(text) // 4)}


@tool("classify_rule", "aiagents", "Classify text by simple keyword rules")
def classify_rule(text: str, rules: str = "{}") -> Dict[str, Any]:
    rule_map = json.loads(rules) if isinstance(rules, str) else rules
    text_l = text.lower()
    for label, keywords in rule_map.items():
        if any(k.lower() in text_l for k in keywords):
            return {"label": label, "matched": True}
    return {"label": "unknown", "matched": False}


@tool("sentiment_lexicon", "aiagents", "Simple lexicon-based sentiment score (-1 to 1)")
def sentiment_lexicon(text: str) -> Dict[str, Any]:
    pos = {"good", "great", "excellent", "amazing", "happy", "love", "awesome", "nice", "best", "cool"}
    neg = {"bad", "terrible", "awful", "hate", "worst", "sad", "poor", "ugly", "wrong", "fail"}
    words = text.lower().split()
    p = sum(1 for w in words if w.strip(".,!?") in pos)
    n = sum(1 for w in words if w.strip(".,!?") in neg)
    total = p + n
    score = (p - n) / total if total else 0
    return {"score": round(score, 3), "positive": p, "negative": n, "verdict": "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"}


@tool("intent_guess", "aiagents", "Guess intent of a user message from keywords")
def intent_guess(text: str) -> Dict[str, Any]:
    t = text.lower()
    intent = "chat"
    if any(k in t for k in ("train", "model")):
        intent = "train"
    elif any(k in t for k in ("eval", "evaluate", "test model")):
        intent = "evaluate"
    elif any(k in t for k in ("export", "convert")):
        intent = "export"
    elif any(k in t for k in ("delete", "remove")):
        intent = "delete"
    elif any(k in t for k in ("list", "show", "display")):
        intent = "list"
    elif any(k in t for k in ("help", "how", "what is")):
        intent = "help"
    return {"intent": intent, "confidence": "high" if intent != "chat" else "low"}


@tool("agent_echo", "aiagents", "Echo a message through a fake agent (offline test helper)")
def agent_echo(message: str) -> Dict[str, Any]:
    return {"agent": "aweai-offline", "message": message, "response": f"[agent] {message}"}


@tool("agent_think", "aiagents", "Structured 'thinking' output (chain-of-thought scaffold)")
def agent_think(question: str, depth: int = 3) -> Dict[str, Any]:
    thoughts = [f"thought {i + 1}: considering {question} from angle {i + 1}" for i in range(depth)]
    return {"question": question, "thoughts": thoughts, "conclusion": f"analysis of: {question}"}


@tool("agent_self_review", "aiagents", "Self-review checklist of an answer (critique)")
def agent_self_review(answer: str) -> Dict[str, Any]:
    checks = {
        "has_content": len(answer.strip()) > 0,
        "has_specifics": any(c.isdigit() for c in answer),
        "length_ok": 10 <= len(answer) <= 5000,
    }
    return {"checks": checks, "score": round(sum(checks.values()) / len(checks), 2)}


@tool("agent_router", "aiagents", "Route a request to a specialist agent by keywords")
def agent_router(request: str) -> Dict[str, Any]:
    t = request.lower()
    if any(k in t for k in ("python", "code", "bug", "refactor")):
        specialist = "codegen"
    elif any(k in t for k in ("security", "hash", "password", "scan")):
        specialist = "security"
    elif any(k in t for k in ("data", "stat", "ml", "model")):
        specialist = "datascience"
    elif any(k in t for k in ("docker", "ci", "deploy", "git")):
        specialist = "devops"
    elif any(k in t for k in ("image", "audio", "video", "media")):
        specialist = "media"
    else:
        specialist = "general"
    return {"request": request, "specialist": specialist}


__all__ = []
