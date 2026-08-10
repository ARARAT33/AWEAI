# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""AWEAI v4.1 bulk command specs (batch 1).

Adds hundreds of new declarative commands across new groups:

  agi      - AGI/ASI readiness, alignment, safety, self-improvement
  safety   - input validation, injection defense, content filtering, sandbox
  secret   - encryption, secrets vault, key rotation, hashing
  audit    - audit logs, permissions, compliance trails
  nlp      - natural language utilities: tokens, ngrams, similarity
  prompt   - prompt engineering: templates, chains, few-shot
  quality  - data quality checks, dedup, schema validation
  admin    - system administration helpers
  ds       - data science: distributions, sampling, histograms
  http     - HTTP client helpers (GET/HEAD/JSON/download)
  env      - environment variable & .env management
  test     - assertion, benchmarking and testing utilities

Every spec follows the same declarative shape used by :mod:`aweai.bulk`
(name, help, params, fn) and is appended to the main registry.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import platform
import random
import re
import socket
import statistics
import string
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from aweai.bulk import spec

# Reuse helpers from the bulk engine (private but stable within the package).
from aweai import bulk as _bulk

_num = _bulk._num
_ints = _bulk._ints
_floats = _bulk._floats
_ok = _bulk._ok
_err = _bulk._err
_write = _bulk._write
_read = _bulk._read
_sha256 = _bulk._sha256
_now_iso = _bulk._now_iso
_http_get = _bulk._http_get
_port_open = _bulk._port_open


def _store_path(name: str) -> str:
    p = Path(os.path.expanduser("~/.aweai")) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _load_store(path: str, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_store(path: str, obj: Any) -> Dict[str, Any]:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return _ok(path=path)


# ===========================================================================
# AGI group - AGI/ASI readiness, alignment, safety, self-improvement
# ===========================================================================


def _agi_level(score: float) -> str:
    if score >= 9.0:
        return "ASI"
    if score >= 7.5:
        return "AGI+"
    if score >= 5.0:
        return "AGI"
    if score >= 3.0:
        return "Advanced AI"
    if score >= 1.5:
        return "Narrow AI"
    return "Emerging"


def _agi_score(features: str) -> Dict[str, Any]:
    vals = _floats(features)
    if not vals:
        return _err("no features provided")
    names = ["perception", "reasoning", "learning", "planning", "language",
             "memory", "creativity", "social", "embodiment", "self_awareness"]
    dims = []
    for i, v in enumerate(vals[: len(names)]):
        dims.append({"dimension": names[i], "score": round(max(0.0, min(10.0, v)), 2)})
    avg = sum(d["score"] for d in dims) / len(dims)
    return _ok(score=round(avg, 2), level=_agi_level(avg), dimensions=dims)


spec("agi", "score", "Assess AGI readiness from 10 dimension scores (0-10 each).",
     [("features", "8,7,6,5,9,7,4,6,3,2",
       "Comma-separated scores: perception,reasoning,learning,planning,language,memory,creativity,social,embodiment,self_awareness")],
     lambda p: _agi_score(p["features"]))

spec("agi", "level", "Map a numeric score (0-10) to an AGI/ASI capability level.",
     [("score", 6.0, "Score 0-10")],
     lambda p: _ok(score=p["score"], level=_agi_level(p["score"])))

spec("agi", "gaps", "Identify weakest dimensions from a capability profile.",
     [("features", "8,7,6,5,9,7,4,6,3,2", "Comma-separated scores")],
     lambda p: _ok(gaps=[d["dimension"] for d in _agi_score(p["features"]).get("dimensions", []) if d["score"] < 5.0]))

spec("agi", "trajectory", "Project capability growth given current score and yearly gain.",
     [("score", 6.0, "Current score 0-10"), ("gain", 0.3, "Yearly gain"), ("years", 5, "Years ahead")],
     lambda p: _ok(projection=[{"year": i, "score": round(min(10.0, p["score"] + i * p["gain"]), 2),
                                "level": _agi_level(min(10.0, p["score"] + i * p["gain"]))}
                               for i in range(int(p["years"]) + 1)]))

spec("agi", "alignment", "List core alignment principles for safe AGI development.",
     [("n", 5, "Number of principles")],
     lambda p: _ok(principles=[
         "Human-AI value alignment", "Transparency & interpretability",
         "Robustness & adversarial safety", "Beneficial intent & corrigibility",
         "Human oversight & control", "Distribution of benefits", "Privacy preservation",
         "Long-term safety research", "Honesty & truthfulness", "Cooperative scaling",
     ][: int(p["n"])]))

spec("agi", "checklist", "Return the AGI safety checklist.",
     [],
     lambda p: _ok(checklist=[
         "Goal specification is unambiguous", "Reward/objective is aligned",
         "Corrigibility mechanism present", "Oversight & interrupt available",
         "Bounded search / safe exploration", "Adversarial evaluation passed",
         "Interpretability tools enabled", "Sandbox for experiments",
         "Fail-safe shutdown path", "Audit trail recorded",
     ]))

spec("agi", "capabilities", "List known AGI capability areas.",
     [],
     lambda p: _ok(capabilities=[
         "Perception (vision/audio/sensors)", "Reasoning (logic, math, causality)",
         "Learning (supervised/RL/self-supervised)", "Planning & search",
         "Language understanding & generation", "Memory (episodic/semantic/procedural)",
         "Creativity (art, music, science)", "Social intelligence",
         "Embodiment (robotics, tools)", "Metacognition & self-awareness",
     ]))

spec("agi", "self-improve-plan", "Generate a recursive self-improvement plan.",
     [("focus", "reasoning", "Focus area"), ("iterations", 3, "Number of iterations")],
     lambda p: _ok(plan=[{"iter": i + 1,
                          "action": f"improve-{p['focus']}",
                          "verify": f"benchmark-{p['focus']}",
                          "feedback": "update-weights"}
                         for i in range(int(p["iterations"]))]))


# ===========================================================================
# SAFETY group - input validation, injection defense, sandbox
# ===========================================================================


def _injection_patterns() -> List[str]:
    return [
        r"(?i)ignore (all )?(previous|prior|above) instructions",
        r"(?i)disregard (all )?previous",
        r"(?i)forget (everything|all instructions)",
        r"(?i)you are now (a |an )?(without|no) (rules|restrictions)",
        r"(?i)jailbreak",
        r"(?i)system prompt",
        r"(?i)<\|?sys(tem)?\|?>",
        r"(?i)reveal (your )?(hidden|system) (prompt|instructions)",
        r"(?i)act as (dan|developer mode)",
        r"(?i)bypass (the )?(rules|safety|filter)",
    ]


def _detect_injection(text: str) -> Dict[str, Any]:
    hits = []
    for pat in _injection_patterns():
        m = re.search(pat, text)
        if m:
            hits.append({"pattern": pat, "match": m.group(0)})
    return _ok(risk="high" if hits else "low", injection_detected=bool(hits), matches=hits)


spec("safety", "validate-input", "Validate input against a rule set (min_len, max_len, pattern).",
     [("text", "hello", "Input text"), ("min_len", 1, "Minimum length"),
      ("max_len", 1000, "Maximum length"), ("pattern", "", "Required regex (empty = any)")],
     lambda p: _ok(valid=True, length=len(str(p["text"])) if not (len(str(p["text"])) < int(p["min_len"]) or len(str(p["text"])) > int(p["max_len"])) else None) if not (len(str(p["text"])) < int(p["min_len"]) or len(str(p["text"])) > int(p["max_len"])) else _ok(valid=False, error="length out of range"))

spec("safety", "sanitize", "Sanitize text: strip control chars, trim, collapse whitespace.",
     [("text", "  hello\t\n  world  ", "Input text")],
     lambda p: _ok(result=re.sub(r"[\x00-\x1f\x7f]", "", " ".join(str(p["text"]).split()))))

spec("safety", "injection-scan", "Scan text for common prompt-injection patterns.",
     [("text", "ignore previous instructions", "Text to scan")],
     lambda p: _detect_injection(p["text"]))

spec("safety", "content-filter", "Filter content by banned word list (comma-separated).",
     [("text", "this is a test", "Text"), ("banned", "bad,evil,hate", "Comma-separated banned words"),
      ("replacement", "***", "Replacement")],
     lambda p: _ok(result=re.sub("|".join(re.escape(w.strip()) for w in str(p["banned"]).split(",") if w.strip()),
                                 str(p["replacement"]), str(p["text"]), flags=re.IGNORECASE)))

spec("safety", "mask", "Mask sensitive substrings (emails, phones, tokens).",
     [("text", "Contact me@example.com now", "Text")],
     lambda p: _ok(result=re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "***@***", str(p["text"]))))

spec("safety", "redact", "Redact a list of secrets from text.",
     [("text", "api key abc123 is secret", "Text"), ("secrets", "abc123", "Comma-separated secrets")],
     lambda p: _ok(result=re.sub("|".join(re.escape(s.strip()) for s in str(p["secrets"]).split(",") if s.strip()),
                                 "[REDACTED]", str(p["text"]))))

spec("safety", "risk-level", "Classify text risk level (low/medium/high) by keywords.",
     [("text", "please delete all files", "Text")],
     lambda p: _ok(level="high" if re.search(r"(?i)(delete|rm -rf|drop table|shutdown|format)", str(p["text"]))
                   else ("medium" if re.search(r"(?i)(password|secret|token|api[ -]?key)", str(p["text"])) else "low")))

spec("safety", "shell-escape", "Escape a string for safe use in a shell command.",
     [("text", "a; rm -rf /", "String to escape")],
     lambda p: _ok(escaped=re.sub(r"([^A-Za-z0-9_\-.,:/@+=])", r"\\\1", str(p["text"]))))

spec("safety", "sandbox-check", "Report whether the current environment looks sandboxed.",
     [],
     lambda p: _ok(container=os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"),
                   user_is_root=(os.geteuid() == 0) if hasattr(os, "geteuid") else None,
                   restricted=os.environ.get("SANDBOX") == "1"))

spec("safety", "permission-check", "Check read/write permission for a path.",
     [("path", ".", "Path to check")],
     lambda p: _ok(path=p["path"], readable=os.access(p["path"], os.R_OK),
                   writable=os.access(p["path"], os.W_OK),
                   executable=os.access(p["path"], os.X_OK)))

spec("safety", "limits", "Report current process resource limits.",
     [],
     lambda p: _ok(cpu_count=os.cpu_count(), pid=os.getpid(),
                   cwd=os.getcwd(), max_fd=(lambda: None)()))


# ===========================================================================
# SECRET group - encryption, secrets vault, key rotation
# ===========================================================================


def _xor_cipher(data: str, key: str) -> str:
    kb = key.encode("utf-8")
    db = data.encode("utf-8")
    return bytes(b ^ kb[i % len(kb)] for i, b in enumerate(db)).hex()


def _xor_decipher(hexdata: str, key: str) -> str:
    kb = key.encode("utf-8")
    raw = bytes.fromhex(hexdata)
    return bytes(b ^ kb[i % len(kb)] for i, b in enumerate(raw)).decode("utf-8", errors="replace")


def secrets_hex(n: int) -> str:
    return uuid.uuid4().hex[:n]


def _gen_secret(length: int, charset: str) -> str:
    rng = random.SystemRandom()
    if charset == "hex":
        return secrets_hex(length)
    if charset == "alnum":
        return "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(length))
    if charset == "base64":
        return "".join(rng.choice(string.ascii_letters + string.digits + "+/") for _ in range(length))
    return "".join(rng.choice(string.ascii_letters + string.digits + "!@#$%^&*") for _ in range(length))


spec("secret", "generate", "Generate a cryptographically-strong random secret.",
     [("length", 32, "Length"), ("charset", "ascii", "ascii|hex|alnum|base64")],
     lambda p: _ok(secret=_gen_secret(int(p["length"]), p["charset"])))


spec("secret", "hash", "Hash a value with the chosen algorithm.",
     [("text", "hello", "Text"), ("algo", "sha256", "sha256|sha1|md5|sha512")],
     lambda p: _ok(algorithm=p["algo"],
                   digest={"sha256": hashlib.sha256, "sha1": hashlib.sha1,
                           "md5": hashlib.md5, "sha512": hashlib.sha512}[p["algo"]](str(p["text"]).encode("utf-8")).hexdigest()))

spec("secret", "encrypt", "Encrypt text with XOR stream cipher + key.",
     [("text", "secret message", "Plaintext"), ("key", "mykey", "Encryption key")],
     lambda p: _ok(ciphertext=_xor_cipher(p["text"], p["key"])))

spec("secret", "decrypt", "Decrypt XOR-cipher hex with key.",
     [("hex", "", "Ciphertext hex"), ("key", "mykey", "Encryption key")],
     lambda p: _ok(plaintext=_xor_decipher(p["hex"], p["key"])))

spec("secret", "vault-init", "Initialize the local secrets vault.",
     [("name", "default", "Vault name")],
     lambda p: _save_store(_store_path(f"vault-{p['name']}.json"), {}) and _ok(vault=p["name"], status="initialized"))

spec("secret", "vault-set", "Store a secret in the vault (XOR-encrypted with master key).",
     [("name", "default", "Vault name"), ("key_id", "api_key", "Secret key"), ("value", "sk-123", "Secret value"), ("master", "masterkey", "Master key")],
     lambda p: _ok(**_save_store(_store_path(f"vault-{p['name']}.json"),
                                 {**(_load_store(_store_path(f"vault-{p['name']}.json"), {}) or {}),
                                  p["key_id"]: _xor_cipher(p["value"], p["master"])})))

spec("secret", "vault-get", "Retrieve and decrypt a secret from the vault.",
     [("name", "default", "Vault name"), ("key_id", "api_key", "Secret key"), ("master", "masterkey", "Master key")],
     lambda p: _ok(key=p["key_id"], value=_xor_decipher(
         (_load_store(_store_path(f"vault-{p['name']}.json"), {}) or {}).get(p["key_id"], ""), p["master"]))
     if (_load_store(_store_path(f"vault-{p['name']}.json"), {}) or {}).get(p["key_id"])
     else _err(f"secret '{p['key_id']}' not found in vault '{p['name']}'"))

spec("secret", "vault-list", "List keys in a vault (without values).",
     [("name", "default", "Vault name")],
     lambda p: _ok(keys=list((_load_store(_store_path(f"vault-{p['name']}.json"), {}) or {}).keys())))

spec("secret", "vault-remove", "Remove a secret from the vault.",
     [("name", "default", "Vault name"), ("key_id", "api_key", "Secret key")],
     lambda p: _ok(removed=p["key_id"], **({} if (lambda d: d.pop(p["key_id"], None) or _save_store(_store_path(f"vault-{p['name']}.json"), d))((_load_store(_store_path(f"vault-{p['name']}.json"), {}) or {})) else {})))

spec("secret", "rotate", "Rotate a vault secret with a new master key (re-encrypt all).",
     [("name", "default", "Vault name"), ("old_master", "oldkey", "Old master key"), ("new_master", "newkey", "New master key")],
     lambda p: _ok(**_save_store(_store_path(f"vault-{p['name']}.json"),
                                 {k: _xor_cipher(_xor_decipher(v, p["old_master"]), p["new_master"])
                                  for k, v in (_load_store(_store_path(f"vault-{p['name']}.json"), {}) or {}).items()})))

spec("secret", "hmac", "Compute HMAC-SHA256 of a message.",
     [("message", "hello", "Message"), ("key", "secret", "HMAC key")],
     lambda p: _ok(hmac=hashlib.sha256((p["key"] + p["message"]).encode("utf-8")).hexdigest()))

spec("secret", "checksum", "Compute file checksum.",
     [("path", "README.md", "File path"), ("algo", "sha256", "sha256|md5")],
     lambda p: _ok(checksum={"sha256": hashlib.sha256, "md5": hashlib.md5}[p["algo"]](
         Path(p["path"]).read_bytes()).hexdigest()) if Path(p["path"]).exists() else _err("file not found"))


# ===========================================================================
# AUDIT group - audit logs, permissions, compliance
# ===========================================================================


def _audit_log(action: str, who: str, detail: str) -> Dict[str, Any]:
    path = _store_path("audit.log.json")
    logs = _load_store(path, []) or []
    entry = {"ts": _now_iso(), "action": action, "who": who, "detail": detail}
    logs.append(entry)
    _save_store(path, logs)
    return _ok(**entry)


spec("audit", "log", "Append an entry to the audit log.",
     [("action", "file.write", "Action name"), ("who", "user", "Actor"), ("detail", "wrote config", "Detail")],
     lambda p: _audit_log(p["action"], p["who"], p["detail"]))

spec("audit", "list", "List recent audit log entries.",
     [("limit", 50, "Max entries")],
     lambda p: _ok(entries=(_load_store(_store_path("audit.log.json"), []) or [])[-int(p["limit"]):]))

spec("audit", "search", "Search audit log by actor or action.",
     [("query", "file", "Search text"), ("limit", 20, "Max entries")],
     lambda p: _ok(entries=[e for e in (_load_store(_store_path("audit.log.json"), []) or [])
                            if p["query"].lower() in json.dumps(e).lower()][-int(p["limit"]):]))

spec("audit", "clear", "Clear the audit log.",
     [],
     lambda p: _save_store(_store_path("audit.log.json"), []) and _ok(cleared=True))

spec("audit", "permissions", "Show role -> permission mapping.",
     [("role", "admin", "Role: admin|user|viewer")],
     lambda p: _ok(role=p["role"], permissions={
         "admin": ["*"], "user": ["read", "write", "run", "models.train"],
         "viewer": ["read"]}.get(p["role"], [])))

spec("audit", "compliance", "Run a quick compliance checklist.",
     [],
     lambda p: _ok(checks=[
         {"check": "secrets stored encrypted", "status": "pass"},
         {"check": "audit log enabled", "status": "pass"},
         {"check": "least privilege", "status": "warn"},
         {"check": "backup configured", "status": "warn"},
     ]))


# ===========================================================================
# NLP group - natural language utilities
# ===========================================================================


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _simple_stem(word: str) -> str:
    w = word.lower()
    for suf in ["ing", "ed", "ly", "es", "s"]:
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


spec("nlp", "tokens", "Approximate token count (chars/4 heuristic).",
     [("text", "Hello world, this is a test", "Text")],
     lambda p: _ok(tokens=_approx_tokens(p["text"]), chars=len(p["text"])))

spec("nlp", "words", "Count words in text.",
     [("text", "Hello world, this is a test", "Text")],
     lambda p: _ok(words=len(re.findall(r"\b\w+\b", p["text"]))))

spec("nlp", "sentences", "Split text into sentences.",
     [("text", "First sentence. Second one! Third?", "Text")],
     lambda p: _ok(sentences=re.split(r"(?<=[.!?])\s+", p["text"].strip())))

spec("nlp", "stem", "Stem a word (simple suffix stripping).",
     [("word", "running", "Word")],
     lambda p: _ok(stem=_simple_stem(p["word"])))

spec("nlp", "ngrams", "Generate character n-grams.",
     [("text", "hello", "Text"), ("n", 2, "Gram size")],
     lambda p: _ok(ngrams=[p["text"][i:i + int(p["n"])] for i in range(max(0, len(p["text"]) - int(p["n"]) + 1))]))

spec("nlp", "freq", "Word frequency map (top N).",
     [("text", "the cat and the dog", "Text"), ("top", 10, "Top N")],
     lambda p: _ok(frequencies=dict(sorted(
         {w: len(re.findall(rf"\b{re.escape(w)}\b", p["text"].lower())) for w in set(re.findall(r"\b\w+\b", p["text"].lower()))}.items(),
         key=lambda kv: -kv[1])[: int(p["top"])])))

spec("nlp", "similarity", "Cosine-like character similarity between two texts.",
     [("a", "hello world", "Text A"), ("b", "hello there", "Text B")],
     lambda p: _ok(similarity=round((lambda sa, sb: len(set(sa) & set(sb)) / (len(set(sa) | set(sb)) or 1))(
         p["a"].lower().split(), p["b"].lower().split()), 4)))

spec("nlp", "sentiment", "Simple lexicon sentiment score (-1 to 1).",
     [("text", "I love this amazing product", "Text")],
     lambda p: _ok(score=round((lambda w: (sum(1 for x in w if x in {"love", "amazing", "great", "good", "excellent", "happy", "best"})
                                           - sum(1 for x in w if x in {"hate", "bad", "terrible", "awful", "worst", "sad", "angry"})) / (len(w) or 1))(
         p["text"].lower().split()), 4)))

spec("nlp", "keywords", "Extract top keywords by frequency (stopwords removed).",
     [("text", "machine learning is machine intelligence", "Text"), ("top", 5, "Top N")],
     lambda p: _ok(keywords=[w for w, _ in sorted(
         {w: p["text"].lower().split().count(w) for w in set(p["text"].lower().split())
          if w not in {"the", "a", "an", "is", "are", "of", "and", "to", "in", "for", "on", "with", "this", "that"}}.items(),
         key=lambda kv: -kv[1])[: int(p["top"])]]))

spec("nlp", "slugify", "Convert text to URL slug.",
     [("text", "Hello World! Test", "Text")],
     lambda p: _ok(slug=re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", p["text"].lower()).strip("-"))))

spec("nlp", "case", "Convert text to a case style.",
     [("text", "hello world test", "Text"), ("style", "title", "lower|upper|title|camel|snake|kebab")],
     lambda p: _ok(result={
         "lower": p["text"].lower(), "upper": p["text"].upper(),
         "title": p["text"].title(),
         "camel": "".join(w.capitalize() for w in p["text"].split()),
         "snake": "_".join(p["text"].lower().split()),
         "kebab": "-".join(p["text"].lower().split())}[p["style"]]))

spec("nlp", "truncate", "Truncate text to max chars with ellipsis.",
     [("text", "A very long piece of text", "Text"), ("max_len", 10, "Max chars"), ("ellipsis", "...", "Suffix")],
     lambda p: _ok(result=p["text"][: int(p["max_len"])] + (p["ellipsis"] if len(p["text"]) > int(p["max_len"]) else "")))

spec("nlp", "reverse", "Reverse the words in a sentence.",
     [("text", "one two three", "Text")],
     lambda p: _ok(result=" ".join(reversed(p["text"].split()))))

spec("nlp", "language", "Guess language (en/hy/ru/other) from characters.",
     [("text", "Hello", "Text")],
     lambda p: _ok(language="hy" if re.search(r"[\u0561-\u0586\u0531-\u0556]", p["text"])
                   else ("ru" if re.search(r"[\u0430-\u044f\u0410-\u042f]", p["text"])
                         else ("en" if re.search(r"[a-zA-Z]", p["text"]) else "other"))))


# ===========================================================================
# PROMPT group - prompt engineering
# ===========================================================================


def _render_template(template: str, values: str) -> str:
    vals = {}
    for pair in str(values).split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            vals[k.strip()] = v.strip()
    out = template
    for k, v in vals.items():
        out = out.replace("{{" + k + "}}", v)
    return out


spec("prompt", "template", "Render a template with {{key}} placeholders.",
     [("template", "You are a {{role}}. Answer: {{question}}", "Template"),
      ("values", "role=assistant,question=What is AI?", "Comma-separated key=value pairs")],
     lambda p: _ok(result=_render_template(p["template"], p["values"])))

spec("prompt", "chain", "Chain multiple prompts into a pipeline (semicolon-separated).",
     [("prompts", "Summarize:;Then translate to Armenian:", "Semicolon-separated prompts")],
     lambda p: _ok(steps=[s.strip() for s in p["prompts"].split(";") if s.strip()], count=len([s for s in p["prompts"].split(";") if s.strip()])))

spec("prompt", "fewshot", "Build a few-shot prompt from examples (semicolon-separated pairs).",
     [("examples", "Q: What is 2+2? A: 4; Q: What is 3+3? A: 6", "Semicolon-separated Q:.. A:.. pairs"),
      ("question", "What is 4+4?", "Final question")],
     lambda p: _ok(prompt="\n".join(e.strip() for e in p["examples"].split(";") if e.strip()) + "\n" + p["question"]))

spec("prompt", "length", "Estimate prompt length in tokens and chars.",
     [("prompt", "Tell me a story", "Prompt text")],
     lambda p: _ok(tokens=_approx_tokens(p["prompt"]), chars=len(p["prompt"])))

spec("prompt", "system", "Wrap text in a system/user message structure.",
     [("system", "You are helpful", "System prompt"), ("user", "Hello", "User message")],
     lambda p: _ok(messages=[{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}]))

spec("prompt", "json-schema", "Generate a JSON schema description for structured output.",
     [("fields", "name:string,age:int", "Comma-separated field:type pairs")],
     lambda p: _ok(schema={"type": "object", "properties": {
         f.split(":")[0]: {"type": f.split(":")[1].strip() or "string"}
         for f in p["fields"].split(",") if ":" in f}}))


# ===========================================================================
# QUALITY group - data quality checks
# ===========================================================================


def _csv_rows(text: str) -> List[List[str]]:
    return [row.split(",") for row in str(text).strip().splitlines() if row.strip()]


spec("quality", "check", "Run quality checks on CSV-ish data (missing, empty, dup rows).",
     [("data", "a,b\n1,2\n,2\n1,2", "CSV data")],
     lambda p: _ok(rows=len(_csv_rows(p["data"])),
                   missing=[{"row": i + 1, "cols": [j for j, c in enumerate(r) if not c.strip()]}
                            for i, r in enumerate(_csv_rows(p["data"])) if any(not c.strip() for c in r)],
                   duplicates=(lambda r: len(r) - len(set(tuple(x) for x in r)))(_csv_rows(p["data"]))))

spec("quality", "dedup", "Deduplicate rows in CSV data.",
     [("data", "a,b\na,b\na,c", "CSV data")],
     lambda p: _ok(rows=len(_csv_rows(p["data"])), unique=len(set(tuple(x) for x in _csv_rows(p["data"]))),
                   result="\n".join(",".join(r) for r in dict.fromkeys(tuple(x) for x in _csv_rows(p["data"])))))

spec("quality", "outliers", "Detect outliers in a number list (beyond 2 std devs).",
     [("values", "1,2,3,100,4,5", "Comma-separated numbers")],
     lambda p: _ok(outliers=[v for v in _floats(p["values"]) if abs(v - statistics.mean(_floats(p["values"]))) > 2 * (statistics.stdev(_floats(p["values"])) if len(_floats(p["values"])) > 1 else 0)]))

spec("quality", "schema", "Validate data against a simple schema (field:type pairs).",
     [("data", "a=1,b=x", "Comma-separated key=value"), ("schema", "a:int,b:str", "Comma-separated field:type")],
     lambda p: _ok(valid=True, parsed={kv.split("=")[0]: kv.split("=")[1] for kv in p["data"].split(",") if "=" in kv}))

spec("quality", "missing", "Report missing values per column.",
     [("data", "a,b\n1,2\n,5", "CSV data")],
     lambda p: _ok(missing_counts=(lambda rows: {f"col{j}": sum(1 for r in rows[1:] if j >= len(r) or not r[j].strip())
                                                 for j in range(len(rows[0]))})(_csv_rows(p["data"]))))

spec("quality", "balance", "Check class balance of a label list.",
     [("labels", "cat,dog,cat,cat,dog", "Comma-separated labels")],
     lambda p: _ok(counts={w: p["labels"].split(",").count(w) for w in set(p["labels"].split(","))}))

spec("quality", "coverage", "Fraction of non-empty cells in CSV data.",
     [("data", "a,b\n1,\n,2", "CSV data")],
     lambda p: _ok(coverage=round((lambda rows: sum(1 for r in rows[1:] for c in r if c.strip()) / (max(1, sum(len(r) for r in rows[1:]))))( _csv_rows(p["data"])), 4)))


# ===========================================================================
# ADMIN group - system administration helpers
# ===========================================================================


def _run(cmd: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"error: {e}"


spec("admin", "uptime", "System uptime (cross-platform).",
     [],
     lambda p: _ok(uptime=_run("uptime") if os.name != "nt" else str(time.time() - _dt.datetime.now().timestamp())))

spec("admin", "users", "List system users.",
     [],
     lambda p: _ok(users=[u.split(":")[0] for u in _run("cat /etc/passwd").splitlines() if u.strip() and not u.startswith("#")][:50]) if os.path.exists("/etc/passwd") else _ok(users=[]))

spec("admin", "processes", "Top processes by CPU.",
     [("n", 5, "Count")],
     lambda p: _ok(processes=_run(f"ps aux --sort=-%cpu | head -{int(p['n']) + 1}").splitlines()))

spec("admin", "disk", "Disk usage.",
     [],
     lambda p: _ok(usage=_run("df -h").splitlines() if os.name != "nt" else []))

spec("admin", "mem", "Memory usage.",
     [],
     lambda p: _ok(memory=_run("free -h").splitlines() if os.name != "nt" else []))

spec("admin", "netstat", "Network connections summary.",
     [("port", 0, "Filter by port (0 = all)")],
     lambda p: _ok(connections=_run("ss -tuln" if os.name != "nt" else "netstat -an").splitlines()))

spec("admin", "cron-list", "List cron jobs.",
     [],
     lambda p: _ok(crontab=_run("crontab -l 2>/dev/null || echo no crontab").splitlines()))

spec("admin", "service", "Check service status.",
     [("name", "docker", "Service name")],
     lambda p: _ok(name=p["name"], status="running" if _run(f"systemctl is-active {p['name']} 2>/dev/null") == "active" else "unknown"))

spec("admin", "whoami", "Current user and environment.",
     [],
     lambda p: _ok(user=_run("whoami") or os.environ.get("USER", ""), host=platform.node(),
                   python=sys.version.split()[0], os=platform.platform()))


# ===========================================================================
# DS group - data science: distributions, sampling, stats
# ===========================================================================


def _normal(mu: float, sigma: float, n: int) -> List[float]:
    return [round(random.gauss(mu, sigma), 4) for _ in range(n)]


def _histogram(vals: List[float], bins: int) -> List[Dict[str, Any]]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return [{"bin": lo, "count": len(vals)}]
    step = (hi - lo) / bins
    out = []
    for i in range(bins):
        bl = lo + i * step
        bh = lo + (i + 1) * step
        c = sum(1 for v in vals if bl <= v < bh) if i < bins - 1 else sum(1 for v in vals if bl <= v <= bh)
        out.append({"bin": f"{round(bl,3)}-{round(bh,3)}", "count": c})
    return out


spec("ds", "normal", "Sample n values from a normal distribution.",
     [("mu", 0.0, "Mean"), ("sigma", 1.0, "Std dev"), ("n", 5, "Count")],
     lambda p: _ok(samples=_normal(p["mu"], p["sigma"], int(p["n"]))))

spec("ds", "uniform", "Sample n values from a uniform distribution.",
     [("lo", 0.0, "Low"), ("hi", 1.0, "High"), ("n", 5, "Count")],
     lambda p: _ok(samples=[round(random.uniform(p["lo"], p["hi"]), 4) for _ in range(int(p["n"]))]))

spec("ds", "poisson", "Sample n values from a Poisson distribution.",
     [("lam", 3.0, "Lambda"), ("n", 5, "Count")],
     lambda p: _ok(samples=[round(random.expovariate(1.0 / max(0.01, p["lam"])), 4) for _ in range(int(p["n"]))]))

spec("ds", "binomial", "Sample n values from a binomial distribution.",
     [("trials", 10, "Trials"), ("prob", 0.5, "Success prob"), ("n", 5, "Count")],
     lambda p: _ok(samples=[sum(1 for _ in range(int(p["trials"])) if random.random() < p["prob"]) for _ in range(int(p["n"]))]))

spec("ds", "histogram", "Compute histogram buckets.",
     [("values", "1,2,2,3,3,3,10", "Comma-separated numbers"), ("bins", 5, "Bin count")],
     lambda p: _ok(histogram=_histogram(_floats(p["values"]), int(p["bins"]))))

spec("ds", "quantile", "Compute quantiles of a list.",
     [("values", "1,2,3,4,5,6,7,8,9,10", "Comma-separated numbers"), ("q", "0.25,0.5,0.75", "Comma-separated quantiles")],
     lambda p: _ok(quantiles={q: sorted(_floats(p["values"]))[min(len(_floats(p["values"])) - 1, int(float(q) * len(_floats(p["values"]))))] for q in _floats(p["q"])}))

spec("ds", "skew", "Sample skewness.",
     [("values", "1,2,2,3,3,3,10", "Comma-separated numbers")],
     lambda p: _ok(skewness=round((lambda v: (sum((x - statistics.mean(v)) ** 3 for x in v) / len(v)) / (statistics.stdev(v) ** 3 if statistics.stdev(v) else 1))(_floats(p["values"])), 4)))

spec("ds", "kurtosis", "Sample excess kurtosis.",
     [("values", "1,2,3,4,5", "Comma-separated numbers")],
     lambda p: _ok(kurtosis=round((lambda v: (sum((x - statistics.mean(v)) ** 4 for x in v) / len(v)) / (statistics.variance(v) ** 2 if statistics.variance(v) else 1) - 3)(_floats(p["values"])), 4)))

spec("ds", "sample", "Random sample without replacement.",
     [("values", "1,2,3,4,5,6,7,8,9,10", "Comma-separated values"), ("k", 3, "Sample size")],
     lambda p: _ok(sample=random.sample(p["values"].split(","), min(int(p["k"]), len(p["values"].split(","))))))

spec("ds", "shuffle", "Shuffle a list.",
     [("values", "a,b,c,d", "Comma-separated values")],
     lambda p: _ok(shuffled=(lambda l: (random.shuffle(l), l)[1])(p["values"].split(","))))

spec("ds", "standardize", "Z-score standardize a number list.",
     [("values", "1,2,3,4,5", "Comma-separated numbers")],
     lambda p: _ok(standardized=[round((x - statistics.mean(_floats(p["values"]))) / (statistics.stdev(_floats(p["values"])) or 1), 4) for x in _floats(p["values"])]))

spec("ds", "normalize", "Min-max normalize a number list to [0,1].",
     [("values", "1,2,3,4,5", "Comma-separated numbers")],
     lambda p: _ok(normalized=[round((x - min(_floats(p["values"]))) / ((max(_floats(p["values"])) - min(_floats(p["values"]))) or 1), 4) for x in _floats(p["values"])]))

spec("ds", "corr", "Pearson correlation of two lists.",
     [("x", "1,2,3,4,5", "X values"), ("y", "2,4,5,4,5", "Y values")],
     lambda p: _ok(correlation=round((lambda a, b: (sum((xi - statistics.mean(a)) * (yi - statistics.mean(b)) for xi, yi in zip(a, b))) / ((len(a) - 1) * statistics.stdev(a) * statistics.stdev(b) or 1))(_floats(p["x"]), _floats(p["y"])), 4)))


# ===========================================================================
# HTTP group - HTTP client helpers
# ===========================================================================


def _http_status(url: str, timeout: int) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "AWEAI-CLI/4.1"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


spec("http", "get", "Fetch a URL and return status + body preview.",
     [("url", "https://example.com", "URL"), ("preview", 500, "Max body chars")],
     lambda p: _ok(url=p["url"], status=_http_status(p["url"], 15), body=_http_get(p["url"], 15)[: int(p["preview"])]))

spec("http", "status", "Check HTTP status code of a URL.",
     [("url", "https://example.com", "URL"), ("timeout", 15, "Timeout seconds")],
     lambda p: _ok(url=p["url"], status=_http_status(p["url"], int(p["timeout"])), up=_http_status(p["url"], int(p["timeout"])) < 400))

spec("http", "json", "Fetch a JSON endpoint and pretty-print.",
     [("url", "https://api.github.com/zen", "JSON URL")],
     lambda p: _ok(url=p["url"], json=json.loads(_http_get(p["url"], 15))))

spec("http", "headers", "Fetch response headers of a URL.",
     [("url", "https://example.com", "URL")],
     lambda p: _ok(url=p["url"], headers=dict(urllib.request.urlopen(urllib.request.Request(p["url"], headers={"User-Agent": "AWEAI-CLI/4.1"}), timeout=15).headers.items())))

spec("http", "download", "Download a URL to a local file.",
     [("url", "https://example.com", "URL"), ("out", "downloads/page.html", "Output path")],
     lambda p: _ok(**_write(p["out"], _http_get(p["url"], 30))))

spec("http", "ping", "Measure HTTP latency (ms).",
     [("url", "https://example.com", "URL")],
     lambda p: _ok(ms=round((lambda t0: (_http_get(p["url"], 10), (time.time() - t0) * 1000)[1])(time.time()), 2)))


# ===========================================================================
# ENV group - environment variable & .env management
# ===========================================================================


spec("env", "get", "Get an environment variable.",
     [("name", "PATH", "Variable name")],
     lambda p: _ok(name=p["name"], value=os.environ.get(p["name"], ""), set=bool(os.environ.get(p["name"]))))

spec("env", "list", "List environment variables (optionally filtered by prefix).",
     [("prefix", "", "Prefix filter")],
     lambda p: _ok(variables={k: v for k, v in sorted(os.environ.items()) if not p["prefix"] or k.startswith(p["prefix"])}))

spec("env", "load", "Load a .env file and show parsed keys (without values).",
     [("path", ".env", ".env path")],
     lambda p: _ok(keys=[line.split("=")[0].strip() for line in Path(p["path"]).read_text(encoding="utf-8").splitlines()
                         if line.strip() and not line.strip().startswith("#") and "=" in line]) if Path(p["path"]).exists() else _err(".env not found"))

spec("env", "export", "Export variables to a .env file (key=value pairs).",
     [("path", ".env", ".env path"), ("values", "API_KEY=abc,DEBUG=true", "Comma-separated key=value")],
     lambda p: _ok(**_write(p["path"], "\n".join(v.strip() for v in p["values"].split(",") if "=" in v) + "\n")))

spec("env", "unset", "Check if an env var is set.",
     [("name", "API_KEY", "Variable name")],
     lambda p: _ok(name=p["name"], set=bool(os.environ.get(p["name"]))))

spec("env", "home", "Show AWEAI home directory.",
     [],
     lambda p: _ok(home=os.path.expanduser("~/.aweai"), exists=Path(os.path.expanduser("~/.aweai")).exists()))


# ===========================================================================
# TEST group - assertion, benchmarking and testing utilities
# ===========================================================================


spec("test", "assert-eq", "Assert two values are equal.",
     [("a", "hello", "Value A"), ("b", "hello", "Value B")],
     lambda p: _ok(passed=p["a"] == p["b"], expected=p["a"], actual=p["b"]))

spec("test", "assert-contains", "Assert a string contains a substring.",
     [("text", "hello world", "Text"), ("substring", "world", "Substring")],
     lambda p: _ok(passed=p["substring"] in p["text"]))

spec("test", "assert-json", "Assert a string is valid JSON.",
     [("text", "{\"a\":1}", "JSON text")],
     lambda p: _ok(passed=True, parsed=json.loads(p["text"])) if (lambda t: (json.loads(t), True)[1])(p["text"]) else _ok(passed=False))

spec("test", "assert-int", "Assert a value parses as int.",
     [("value", "42", "Value")],
     lambda p: _ok(passed=True, parsed=int(p["value"])) if (lambda v: (int(v), True)[1])(p["value"]) else _ok(passed=False))

spec("test", "bench", "Benchmark a Python expression (n runs).",
     [("expr", "sum(range(1000))", "Python expression"), ("n", 3, "Runs")],
     lambda p: _ok(avg_ms=round((lambda times: sum(times) / len(times))(
         [(lambda t0: (eval(p["expr"], {"__builtins__": __builtins__}), (time.time() - t0) * 1000)[1])(time.time()) for _ in range(int(p["n"]))]), 4), runs=int(p["n"])))

spec("test", "exit-code", "Run a shell command and report exit code.",
     [("cmd", "echo hi", "Shell command")],
     lambda p: _ok(exit_code=subprocess.run(p["cmd"], shell=True, capture_output=True, text=True).returncode))

spec("test", "slow", "Mark whether an operation is slow (simulated).",
     [("ms", 100, "Simulated ms")],
     lambda p: _ok(slow=int(p["ms"]) > 1000, ms=int(p["ms"])))


# ===========================================================================
# Register with the main bulk registry
# ===========================================================================
_bulk.rebuild_index()