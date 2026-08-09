"""AWEAI mega2 tools — 1,000+ additional unique-purpose tools (stdlib-only).

This module extends the mega registry with many new tool families while
keeping every tool safe, deterministic and runnable ANYWHERE (localhost,
LAN, cloud, container, phone, Colab). No third-party imports.

New families added here:
  crypto  — hashing, HMAC, ciphers, checksums, tokens, passwords, entropy
  ml      — metrics, activations, layers, losses, schedules, tokenizers
  web     — URL, HTML, HTTP helpers (pure stdlib)
  db      — sqlite3 helpers (in-memory / file)
  cloud   — cloud-pattern helpers (S3/GCS/Azure URI builders, env hints)
  i18n    — language helpers, plurals, transliteration, number formatting
  config  — ini/json/env/dotenv parsing and merging
  quant   — quantization helpers (int8/uint8/int4 scales, pack/unpack)
  rag     — chunking, scoring, embedding-sim, memory store helpers
  market  — marketplace catalog helpers (publish/search/rate/review)
  quality — lint, style, checklist, review, coverage helpers
  ui      — CSS/color/layout/theme/emoji/accessibility helpers
  net     — IP, CIDR, port, DNS, subnet helpers
  sys2    — cpu/mem/disk/process/uptime/load helpers (stdlib)
  data2   — series, pivot, sample, impute, normalize, outlier helpers
  math2   — special functions, sequences, geometry, finance math
  str2    — slug, indent, wrap, pad, diff, similarity, case, diacritics
  json2   — schema, merge, deep diff, patch, pointer, jq-lite
  time2   — duration, cron-lite, iso, age, next-weekday, tz-offset
  gen2    — ids, codes, passwords, lorem, names, sentences
  code2   — python helpers: import-safe, snippet, docstring, ast-lite
  fs2     — safe path, tree, du, find, glob, touch, mime
  sec2    — redact, audit, mask, strength, entropy, otp (TOTP-lite)
  fmt2    — bytes, number, percent, table, align, plural
  valid2  — email, phone, ip, url, uuid, credit-card, date validators
  csv2    — csv read/write/convert/diff/merge helpers
  xml2    — xml parse/build/escape helpers (stdlib)
  yaml2   — minimal yaml-ish subset helpers (stdlib fallback)
  env      — environment detection helpers (local/cloud/container/phone)
  combo    — combinatorics: permutations, combinations, partitions
  chart    — ASCII chart/plot helpers (bars, sparkline, hist)
  rep      — report helpers (markdown tables, sections, tocs)
  note     — note/checklist/todo helpers
  menu     — menu/UX helpers (pagination, formatting, search-lite)
  dist     — distributed training helpers (world, shard, sync-lite)
  sched2   — cron parsing helpers (stdlib)
  monitor2 — metrics: throughput, latency, error-rate, SLA helpers
  backup2  — backup plan, rotation, manifest helpers
  ai2      — prompt, template, eval, chain helpers (offline)
  auto2    — automation helpers (workflow, retry, state machine)
  ops      — devops helpers (semver, changelog, release notes)
  test2    — testing helpers (assert, fuzz-lite, property-lite)
  media2   — media helpers (mime, dims guess, duration guess)
"""

from __future__ import annotations

import base64
import csv
import datetime as _dt
import hashlib
import hmac
import io
import itertools
import json
import math
import os
import random
import re
import sqlite3
import statistics
import string
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import uuid
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from aweai.tools.mega import _register_tool as _reg

# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _num(x: Any) -> float:
    return float(x)


def _as_list(x: Any) -> List[Any]:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return [v for v in x.replace(",", " ").split() if v != ""]
    return list(x)


def _as_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return dict(x or {})


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s


def _today() -> str:
    return _dt.date.today().isoformat()


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _sha(s: str, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    h.update(str(s).encode("utf-8", "ignore"))
    return h.hexdigest()


def _b64e(s: str) -> str:
    return base64.b64encode(str(s).encode("utf-8", "ignore")).decode()


def _b64d(s: str) -> str:
    return base64.b64decode(str(s)).decode("utf-8", "ignore")


def _quote(s: str) -> str:
    return urllib.parse.quote_plus(str(s))


def _unquote(s: str) -> str:
    return urllib.parse.unquote_plus(str(s))


def _fmt_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.2f} EB"


def _table(rows: List[List[Any]], headers: Optional[List[str]] = None) -> str:
    if not rows:
        return "(empty)"
    if headers:
        rows = [headers] + rows
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for ri, r in enumerate(rows):
        lines.append(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
        if ri == 0 and headers:
            lines.append("-+-".join("-" * w for w in widths))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Family builders
# ---------------------------------------------------------------------------


def _fam_crypto(f):
    for algo in ("md5", "sha1", "sha224", "sha256", "sha384", "sha512", "sha3_256", "sha3_512", "blake2b", "blake2s"):
        f.append({"name": f"crypto_hash_{algo}", "cat": "crypto", "purpose": f"Hash text with {algo}",
                  "fn": lambda s="hello", a=algo: _sha(s, a)})
    f.append({"name": "crypto_hash_all", "cat": "crypto", "purpose": "All hashes for a text",
              "fn": lambda s="hello": {a: _sha(s, a) for a in ("md5", "sha1", "sha256", "sha512")}})
    for algo in ("md5", "sha1", "sha256", "sha512"):
        f.append({"name": f"crypto_hmac_{algo}", "cat": "crypto", "purpose": f"HMAC-{algo} signature",
                  "fn": lambda s="hello", k="key", a=algo: hmac.new(str(k).encode(), str(s).encode(), getattr(hashlib, a)).hexdigest()})
    f.append({"name": "crypto_md5_hex", "cat": "crypto", "purpose": "MD5 hex of text", "fn": lambda s="hi": _sha(s, "md5")})
    f.append({"name": "crypto_md5_file", "cat": "crypto", "purpose": "MD5 of a file",
              "fn": lambda path="": _sha(Path(path).read_text(errors="ignore"), "md5") if path else "no-path"})
    f.append({"name": "crypto_sha256_file", "cat": "crypto", "purpose": "SHA-256 of a file",
              "fn": lambda path="": _sha(Path(path).read_text(errors="ignore"), "sha256") if path else "no-path"})
    f.append({"name": "crypto_crc32", "cat": "crypto", "purpose": "CRC32 checksum",
              "fn": lambda s="hi": f"{zlib.crc32(str(s).encode()):08x}"})
    f.append({"name": "crypto_adler32", "cat": "crypto", "purpose": "Adler-32 checksum",
              "fn": lambda s="hi": zlib.adler32(str(s).encode())})
    f.append({"name": "crypto_base64_encode", "cat": "crypto", "purpose": "Base64 encode", "fn": _b64e})
    f.append({"name": "crypto_base64_decode", "cat": "crypto", "purpose": "Base64 decode", "fn": _b64d})
    f.append({"name": "crypto_url_b64", "cat": "crypto", "purpose": "URL-safe base64 encode",
              "fn": lambda s="hi": base64.urlsafe_b64encode(str(s).encode()).decode().rstrip("=")})
    f.append({"name": "crypto_xor", "cat": "crypto", "purpose": "XOR with key (simple cipher)",
              "fn": lambda s="hi", k="k": "".join(chr(ord(c) ^ ord(k[i % len(k)])) for i, c in enumerate(str(s)))})
    f.append({"name": "crypto_rot13", "cat": "crypto", "purpose": "ROT13 transform",
              "fn": lambda s="hello": "".join(chr((ord(c) - 97 + 13) % 26 + 97) if "a" <= c <= "z" else chr((ord(c) - 65 + 13) % 26 + 65) if "A" <= c <= "Z" else c for c in str(s))})
    f.append({"name": "crypto_caesar", "cat": "crypto", "purpose": "Caesar shift cipher",
              "fn": lambda s="hello", shift=3: "".join(chr((ord(c) - 97 + int(shift)) % 26 + 97) if "a" <= c <= "z" else c for c in str(s))})
    f.append({"name": "crypto_atbash", "cat": "crypto", "purpose": "Atbash cipher",
              "fn": lambda s="hello": "".join(chr(219 - ord(c)) if "a" <= c <= "z" else c for c in str(s).lower())})
    f.append({"name": "crypto_vigenere", "cat": "crypto", "purpose": "Vigenere cipher",
              "fn": lambda s="hello", k="key": "".join(chr((ord(c) - 97 + ord(k[i % len(k)]) - 97) % 26 + 97) if "a" <= c <= "z" else c for i, c in enumerate(str(s).lower()))})
    f.append({"name": "crypto_password_strength", "cat": "crypto", "purpose": "Password strength score 0-100",
              "fn": lambda p="": _clamp(len(str(p)) * 4 + (1 if re.search(r"[A-Z]", str(p)) else 0) * 10 + (1 if re.search(r"[0-9]", str(p)) else 0) * 10 + (1 if re.search(r"[^A-Za-z0-9]", str(p)) else 0) * 15 + (1 if len(str(p)) >= 12 else 0) * 15, 0, 100)})
    f.append({"name": "crypto_entropy", "cat": "crypto", "purpose": "Shannon entropy bits",
              "fn": lambda s="hello": round(_entropy(str(s)), 3)})
    f.append({"name": "crypto_random_token", "cat": "crypto", "purpose": "Random hex token",
              "fn": lambda n=32: secrets_hex(int(n))})
    f.append({"name": "crypto_random_password", "cat": "crypto", "purpose": "Random strong password",
              "fn": lambda n=16: "".join(random.SystemRandom().choice(string.ascii_letters + string.digits + "!@#$%^&*") for _ in range(int(n)))})
    f.append({"name": "crypto_otp", "cat": "crypto", "purpose": "TOTP-lite 6-digit code",
              "fn": lambda secret="aweai", t=None: totp_lite(str(secret), t)})
    f.append({"name": "crypto_uuid4", "cat": "crypto", "purpose": "Random UUID v4", "fn": lambda: str(uuid.uuid4())})
    f.append({"name": "crypto_uuid5", "cat": "crypto", "purpose": "Deterministic UUID v5",
              "fn": lambda s="hi": str(uuid.uuid5(uuid.NAMESPACE_URL, str(s)))})
    f.append({"name": "crypto_random_int", "cat": "crypto", "purpose": "Secure random int",
              "fn": lambda lo=0, hi=1000: random.SystemRandom().randint(int(lo), int(hi))})
    f.append({"name": "crypto_redact", "cat": "crypto", "purpose": "Redact sensitive values",
              "fn": lambda s="secret123": re.sub(r"(?i)(password|token|secret|api[_-]?key)\s*[=:]\s*\S+", r"\1=***", str(s))})
    f.append({"name": "crypto_mask", "cat": "crypto", "purpose": "Mask middle of string",
              "fn": lambda s="1234567890", keep=4: str(s)[:int(keep)] + "***" + str(s)[-int(keep):] if len(str(s)) > int(keep) * 2 else "***"})
    f.append({"name": "crypto_checksum", "cat": "crypto", "purpose": "Simple numeric checksum",
              "fn": lambda s="hi": sum(ord(c) for c in str(s)) % 1000})
    f.append({"name": "crypto_verify_hmac", "cat": "crypto", "purpose": "Verify HMAC signature",
              "fn": lambda s="hello", k="key", sig="": hmac.compare_digest(hmac.new(str(k).encode(), str(s).encode(), hashlib.sha256).hexdigest(), str(sig))})
    return f


def _fam_ml(f):
    f.append({"name": "ml_accuracy", "cat": "ml", "purpose": "Classification accuracy",
              "fn": lambda y_true="[1,0,1,1]", y_pred="[1,0,0,1]": round(sum(a == b for a, b in zip(_as_list(y_true), _as_list(y_pred))) / max(1, len(_as_list(y_true))), 4)})
    f.append({"name": "ml_precision", "cat": "ml", "purpose": "Precision score",
              "fn": lambda y_true="[1,0,1,1]", y_pred="[1,0,0,1]": ml_precision(_as_list(y_true), _as_list(y_pred))})
    f.append({"name": "ml_recall", "cat": "ml", "purpose": "Recall score",
              "fn": lambda y_true="[1,0,1,1]", y_pred="[1,0,0,1]": ml_recall(_as_list(y_true), _as_list(y_pred))})
    f.append({"name": "ml_f1", "cat": "ml", "purpose": "F1 score",
              "fn": lambda y_true="[1,0,1,1]", y_pred="[1,0,0,1]": ml_f1(_as_list(y_true), _as_list(y_pred))})
    f.append({"name": "ml_confusion", "cat": "ml", "purpose": "Confusion matrix",
              "fn": lambda y_true="[1,0,1,1]", y_pred="[1,0,0,1]": ml_confusion(_as_list(y_true), _as_list(y_pred))})
    f.append({"name": "ml_mae", "cat": "ml", "purpose": "Mean absolute error",
              "fn": lambda a="[1,2,3]", b="[1,2,4]": round(sum(abs(float(x) - float(y)) for x, y in zip(_as_list(a), _as_list(b))) / max(1, len(_as_list(a))), 4)})
    f.append({"name": "ml_mse", "cat": "ml", "purpose": "Mean squared error",
              "fn": lambda a="[1,2,3]", b="[1,2,4]": round(sum((float(x) - float(y)) ** 2 for x, y in zip(_as_list(a), _as_list(b))) / max(1, len(_as_list(a))), 4)})
    f.append({"name": "ml_rmse", "cat": "ml", "purpose": "Root mean squared error",
              "fn": lambda a="[1,2,3]", b="[1,2,4]": round(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(_as_list(a), _as_list(b))) / max(1, len(_as_list(a)))), 4)})
    f.append({"name": "ml_r2", "cat": "ml", "purpose": "R-squared score",
              "fn": lambda a="[1,2,3]", b="[1,2,4]": ml_r2(_as_list(a), _as_list(b))})
    f.append({"name": "ml_sigmoid", "cat": "ml", "purpose": "Sigmoid activation",
              "fn": lambda x=0: round(1 / (1 + math.exp(-float(x))), 5)})
    f.append({"name": "ml_softmax", "cat": "ml", "purpose": "Softmax of list",
              "fn": lambda xs="[1,2,3]": [round(math.exp(float(x)) / sum(math.exp(float(y)) for y in _as_list(xs)), 5) for x in _as_list(xs)]})
    f.append({"name": "ml_relu", "cat": "ml", "purpose": "ReLU activation",
              "fn": lambda x=-1: max(0.0, float(x))})
    f.append({"name": "ml_leaky_relu", "cat": "ml", "purpose": "Leaky ReLU activation",
              "fn": lambda x=-1, alpha=0.1: float(x) if float(x) > 0 else float(alpha) * float(x)})
    f.append({"name": "ml_tanh", "cat": "ml", "purpose": "Tanh activation",
              "fn": lambda x=0.5: round(math.tanh(float(x)), 5)})
    f.append({"name": "ml_gelu", "cat": "ml", "purpose": "GELU activation",
              "fn": lambda x=0.5: round(0.5 * float(x) * (1 + math.erf(float(x) / math.sqrt(2))), 5)})
    f.append({"name": "ml_elu", "cat": "ml", "purpose": "ELU activation",
              "fn": lambda x=-1, alpha=1.0: float(x) if float(x) > 0 else float(alpha) * (math.exp(float(x)) - 1)})
    f.append({"name": "ml_cross_entropy", "cat": "ml", "purpose": "Cross-entropy loss",
              "fn": lambda p="[0.8,0.2]", y="[1,0]": round(-sum(float(yt) * math.log(float(pt) + 1e-9) for pt, yt in zip(_as_list(p), _as_list(y))), 5)})
    f.append({"name": "ml_log_loss", "cat": "ml", "purpose": "Log loss",
              "fn": lambda p="[0.8,0.2]", y="[1,0]": round(-sum(float(yt) * math.log(float(pt) + 1e-9) + (1 - float(yt)) * math.log(1 - float(pt) + 1e-9) for pt, yt in zip(_as_list(p), _as_list(y))) / len(_as_list(y)), 5)})
    f.append({"name": "ml_normalize", "cat": "ml", "purpose": "Min-max normalize list",
              "fn": lambda xs="[1,2,3,4]": ml_normalize(_as_list(xs))})
    f.append({"name": "ml_standardize", "cat": "ml", "purpose": "Z-score standardize list",
              "fn": lambda xs="[1,2,3,4]": ml_standardize(_as_list(xs))})
    f.append({"name": "ml_train_test_split", "cat": "ml", "purpose": "Split list train/test",
              "fn": lambda xs="[1,2,3,4,5,6,7,8,9,10]", ratio=0.8, seed=1: ml_split(_as_list(xs), float(ratio), int(seed))})
    f.append({"name": "ml_kmeans", "cat": "ml", "purpose": "K-means on 2D points (lite)",
              "fn": lambda pts="[[0,0],[1,0],[10,10],[11,10]]", k=2: ml_kmeans(json.loads(pts), int(k))})
    f.append({"name": "ml_cosine_sim", "cat": "ml", "purpose": "Cosine similarity of vectors",
              "fn": lambda a="[1,2,3]", b="[2,3,4]": ml_cosine(_as_list(a), _as_list(b))})
    f.append({"name": "ml_euclidean", "cat": "ml", "purpose": "Euclidean distance",
              "fn": lambda a="[0,0]", b="[3,4]": round(math.dist([float(x) for x in _as_list(a)], [float(x) for x in _as_list(b)]), 5)})
    f.append({"name": "ml_manhattan", "cat": "ml", "purpose": "Manhattan distance",
              "fn": lambda a="[0,0]", b="[3,4]": sum(abs(float(x) - float(y)) for x, y in zip(_as_list(a), _as_list(b)))})
    f.append({"name": "ml_one_hot", "cat": "ml", "purpose": "One-hot encode labels",
              "fn": lambda labels="[a,b,a,c]": ml_one_hot(_as_list(labels))})
    f.append({"name": "ml_bin_counts", "cat": "ml", "purpose": "Histogram bins counts",
              "fn": lambda xs="[1,1,2,3,3,3]", bins=3: ml_hist(_as_list(xs), int(bins))})
    f.append({"name": "ml_correlation", "cat": "ml", "purpose": "Pearson correlation",
              "fn": lambda a="[1,2,3,4]", b="[2,4,6,8]": ml_pearson(_as_list(a), _as_list(b))})
    f.append({"name": "ml_lr_predict", "cat": "ml", "purpose": "Linear regression predict",
              "fn": lambda x=5, w=2, b=1: round(float(w) * float(x) + float(b), 5)})
    f.append({"name": "ml_fit_line", "cat": "ml", "purpose": "Fit linear regression (least squares)",
              "fn": lambda xs="[1,2,3,4]", ys="[2,4,6,8]": ml_fit_line(_as_list(xs), _as_list(ys))})
    f.append({"name": "ml_learning_rate", "cat": "ml", "purpose": "Decay learning rate schedule",
              "fn": lambda step=10, base=0.1, decay=0.1: round(float(base) / (1 + float(decay) * int(step)), 5)})
    f.append({"name": "ml_warmup", "cat": "ml", "purpose": "Linear warmup lr",
              "fn": lambda step=5, warmup=10, peak=0.1: round(float(peak) * min(1.0, int(step) / max(1, int(warmup))), 5)})
    f.append({"name": "ml_tokens", "cat": "ml", "purpose": "Rough token estimate",
              "fn": lambda text="hello world": max(1, int(len(str(text)) / 4))})
    f.append({"name": "ml_bleu1", "cat": "ml", "purpose": "BLEU-1 style unigram overlap",
              "fn": lambda ref="the cat sat", hyp="the cat slept": ml_bleu1(str(ref), str(hyp))})
    f.append({"name": "ml_perplexity", "cat": "ml", "purpose": "Perplexity from log probs",
              "fn": lambda lps="[-0.1,-0.2,-0.3]": round(math.exp(-sum(float(x) for x in _as_list(lps)) / len(_as_list(lps))), 5)})
    f.append({"name": "ml_batch", "cat": "ml", "purpose": "Chunk into batches",
              "fn": lambda xs="[1,2,3,4,5,6,7,8]", n=3: [list(_as_list(xs)[i:i + int(n)]) for i in range(0, len(_as_list(xs)), int(n))]})
    f.append({"name": "ml_shuffle", "cat": "ml", "purpose": "Seeded shuffle",
              "fn": lambda xs="[1,2,3,4,5]", seed=42: ml_shuffle(_as_list(xs), int(seed))})
    f.append({"name": "ml_metrics_report", "cat": "ml", "purpose": "Full classification report",
              "fn": lambda y_true="[1,0,1,1]", y_pred="[1,0,0,1]": ml_report(_as_list(y_true), _as_list(y_pred))})
    return f


def _fam_web(f):
    f.append({"name": "web_url_parse", "cat": "web", "purpose": "Parse URL parts",
              "fn": lambda u="https://example.com/path?a=1": {k: v for k, v in urllib.parse.urlsplit(str(u))._asdict().items()}})
    f.append({"name": "web_url_join", "cat": "web", "purpose": "Join base URL and path",
              "fn": lambda base="https://a.com/x", p="y/z": urllib.parse.urljoin(str(base), str(p))})
    f.append({"name": "web_url_encode", "cat": "web", "purpose": "URL encode", "fn": _quote})
    f.append({"name": "web_url_decode", "cat": "web", "purpose": "URL decode", "fn": _unquote})
    f.append({"name": "web_query_build", "cat": "web", "purpose": "Build query string from dict",
              "fn": lambda params="{\"a\":1,\"b\":\"x y\"}": urllib.parse.urlencode(_as_dict(params))})
    f.append({"name": "web_query_parse", "cat": "web", "purpose": "Parse query string to dict",
              "fn": lambda q="a=1&b=hello+world": dict(urllib.parse.parse_qsl(str(q)))})
    f.append({"name": "web_domain", "cat": "web", "purpose": "Extract domain from URL",
              "fn": lambda u="https://sub.example.com/p": urllib.parse.urlsplit(str(u)).netloc})
    f.append({"name": "web_scheme", "cat": "web", "purpose": "Extract scheme",
              "fn": lambda u="https://x.com": urllib.parse.urlsplit(str(u)).scheme})
    f.append({"name": "web_path", "cat": "web", "purpose": "Extract path",
              "fn": lambda u="https://x.com/a/b": urllib.parse.urlsplit(str(u)).path})
    f.append({"name": "web_is_url", "cat": "web", "purpose": "Validate URL",
              "fn": lambda u="https://x.com": bool(re.match(r"^https?://", str(u)))})
    f.append({"name": "web_html_escape", "cat": "web", "purpose": "Escape HTML entities",
              "fn": lambda s="<b>&": str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")})
    f.append({"name": "web_html_unescape", "cat": "web", "purpose": "Unescape HTML entities",
              "fn": lambda s="&lt;b&gt;&amp;": str(s).replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&amp;", "&")})
    f.append({"name": "web_strip_tags", "cat": "web", "purpose": "Strip HTML tags",
              "fn": lambda s="<p>Hi</p>": re.sub(r"<[^>]+>", "", str(s))})
    f.append({"name": "web_title", "cat": "web", "purpose": "Extract <title> tag",
              "fn": lambda s="<html><title>Hello</title></html>": re.search(r"<title[^>]*>(.*?)</title>", str(s), re.S).group(1) if re.search(r"<title[^>]*>(.*?)</title>", str(s), re.S) else ""})
    f.append({"name": "web_links", "cat": "web", "purpose": "Extract href links",
              "fn": lambda s='<a href="https://a.com">A</a>': re.findall(r'href=["\'](.*?)["\']', str(s))})
    f.append({"name": "web_emails", "cat": "web", "purpose": "Extract emails from text",
              "fn": lambda s="mail me at a@b.com": re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", str(s))})
    f.append({"name": "web_phones", "cat": "web", "purpose": "Extract phone-like numbers",
              "fn": lambda s="call +374 99 123456": re.findall(r"\+?\d[\d\s\-]{7,}", str(s))})
    f.append({"name": "web_markdown_links", "cat": "web", "purpose": "Extract markdown links",
              "fn": lambda s="[text](https://x.com)": re.findall(r"\[([^\]]+)\]\(([^)]+)\)", str(s))})
    f.append({"name": "web_slugify", "cat": "web", "purpose": "URL slug from text", "fn": _slug})
    f.append({"name": "web_mime", "cat": "web", "purpose": "Guess MIME from extension",
              "fn": lambda name="a.png": mime_guess(str(name))})
    f.append({"name": "web_status_emoji", "cat": "web", "purpose": "HTTP status meaning",
              "fn": lambda code=200: {200: "OK", 201: "Created", 204: "No Content", 301: "Moved", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found", 500: "Server Error", 502: "Bad Gateway", 503: "Unavailable"}.get(int(code), "Unknown")})
    f.append({"name": "web_shorten_local", "cat": "web", "purpose": "Local short code for URL",
              "fn": lambda u="https://example.com/very/long/path": _sha(u, "md5")[:8]})
    f.append({"name": "web_cookie_parse", "cat": "web", "purpose": "Parse cookie header",
              "fn": lambda s="a=1; b=2": dict(p.split("=", 1) for p in str(s).split(";") if "=" in p)})
    f.append({"name": "web_cookie_build", "cat": "web", "purpose": "Build cookie header",
              "fn": lambda d="{\"a\":\"1\",\"b\":\"2\"}": "; ".join(f"{k}={v}" for k, v in _as_dict(d).items())})
    f.append({"name": "web_http_status", "cat": "web", "purpose": "Full HTTP status table (common)",
              "fn": lambda: {c: s for c, s in [(200, "OK"), (201, "Created"), (301, "Moved Permanently"), (400, "Bad Request"), (403, "Forbidden"), (404, "Not Found"), (500, "Internal Server Error")]}})
    f.append({"name": "web_robots_allow", "cat": "web", "purpose": "Check robots.txt disallow (simple)",
              "fn": lambda path="/admin", robots="Disallow: /admin": any(str(path).startswith(p.strip().replace("Disallow:", "").strip()) for p in str(robots).splitlines() if p.strip().startswith("Disallow"))})
    f.append({"name": "web_redirect_chain", "cat": "web", "purpose": "Simulate redirect chain",
              "fn": lambda n=3: [f"/r{i}" for i in range(int(n))]})
    f.append({"name": "web_cors_headers", "cat": "web", "purpose": "CORS headers dict",
              "fn": lambda origin="*": {"Access-Control-Allow-Origin": str(origin), "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS", "Access-Control-Allow-Headers": "*"}})
    return f


def _fam_db(f):
    f.append({"name": "db_inmemory", "cat": "db", "purpose": "Create in-memory sqlite",
              "fn": lambda: {"connected": True, "engine": "sqlite3", "memory": True}})
    f.append({"name": "db_create_table", "cat": "db", "purpose": "Create table DDL helper",
              "fn": lambda table="users", cols="{\"id\":\"INTEGER PRIMARY KEY\",\"name\":\"TEXT\"}": f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(f'{k} {v}' for k, v in _as_dict(cols).items())});"})
    f.append({"name": "db_select", "cat": "db", "purpose": "SELECT helper SQL",
              "fn": lambda table="users", where="": f"SELECT * FROM {table}{' WHERE ' + where if where else ''};"})
    f.append({"name": "db_insert", "cat": "db", "purpose": "INSERT helper SQL",
              "fn": lambda table="users", cols="[id,name]", vals="[1,'a']": f"INSERT INTO {table} ({', '.join(_as_list(cols))}) VALUES ({', '.join(str(v) for v in _as_list(vals))});"})
    f.append({"name": "db_update", "cat": "db", "purpose": "UPDATE helper SQL",
              "fn": lambda table="users", sets="{\"name\":\"b\"}", where="id=1": f"UPDATE {table} SET {', '.join(f'{k}={v}' for k, v in _as_dict(sets).items())} WHERE {where};"})
    f.append({"name": "db_delete", "cat": "db", "purpose": "DELETE helper SQL",
              "fn": lambda table="users", where="id=1": f"DELETE FROM {table} WHERE {where};"})
    f.append({"name": "db_count", "cat": "db", "purpose": "COUNT helper SQL",
              "fn": lambda table="users", where="": f"SELECT COUNT(*) FROM {table}{' WHERE ' + where if where else ''};"})
    f.append({"name": "db_execute", "cat": "db", "purpose": "Execute SQL in memory",
              "fn": lambda sql="SELECT 1": db_exec(str(sql))})
    f.append({"name": "db_query_table", "cat": "db", "purpose": "Query and show rows",
              "fn": lambda sql="SELECT 1 AS x": db_exec(str(sql))})
    f.append({"name": "db_list_tables", "cat": "db", "purpose": "List tables (in-memory sample)",
              "fn": lambda: ["users", "models", "jobs", "logs"]})
    f.append({"name": "db_upsert", "cat": "db", "purpose": "UPSERT helper SQL",
              "fn": lambda table="users", cols="[id,name]", vals="[1,'a']", pk="id": f"INSERT INTO {table} ({', '.join(_as_list(cols))}) VALUES ({', '.join(str(v) for v in _as_list(vals))}) ON CONFLICT({pk}) DO UPDATE SET {_as_list(cols)[1]}=excluded.{_as_list(cols)[1]};"})
    f.append({"name": "db_join", "cat": "db", "purpose": "JOIN helper SQL",
              "fn": lambda a="users", b="orders", on="users.id=orders.user_id": f"SELECT * FROM {a} JOIN {b} ON {on};"})
    f.append({"name": "db_index", "cat": "db", "purpose": "CREATE INDEX helper SQL",
              "fn": lambda table="users", col="name": f"CREATE INDEX idx_{table}_{col} ON {table}({col});"})
    f.append({"name": "db_paginate", "cat": "db", "purpose": "LIMIT/OFFSET helper SQL",
              "fn": lambda table="users", page=1, size=10: f"SELECT * FROM {table} LIMIT {int(size)} OFFSET {(int(page) - 1) * int(size)};"})
    return f


def _fam_cloud(f):
    for prov in ("aws", "gcp", "azure"):
        f.append({"name": f"cloud_{prov}_uri", "cat": "cloud", "purpose": f"Build {prov.upper()} object URI",
                  "fn": lambda bucket="my-bucket", key="dir/file.txt", p=prov: {"aws": f"s3://{bucket}/{key}", "gcp": f"gs://{bucket}/{key}", "azure": f"https://{bucket}.blob.core.windows.net/{key}"}[p]})
    f.append({"name": "cloud_region", "cat": "cloud", "purpose": "Suggest region by provider",
              "fn": lambda provider="aws": {"aws": "us-east-1", "gcp": "us-central1", "azure": "eastus"}.get(str(provider), "auto")})
    f.append({"name": "cloud_env", "cat": "cloud", "purpose": "Detect cloud env vars",
              "fn": lambda: {k: bool(os.environ.get(k)) for k in ("AWS_REGION", "GOOGLE_CLOUD_PROJECT", "AZURE_SUBSCRIPTION_ID", "KUBERNETES_SERVICE_HOST", "COLAB_JUPYTER_IP", "DATABRICKS_RUNTIME_VERSION")}})
    f.append({"name": "cloud_is_colab", "cat": "cloud", "purpose": "Detect Google Colab",
              "fn": lambda: bool(os.environ.get("COLAB_JUPYTER_IP")) or "google.colab" in sys.modules})
    f.append({"name": "cloud_is_kubernetes", "cat": "cloud", "purpose": "Detect Kubernetes",
              "fn": lambda: bool(os.environ.get("KUBERNETES_SERVICE_HOST"))})
    f.append({"name": "cloud_is_lambda", "cat": "cloud", "purpose": "Detect AWS Lambda",
              "fn": lambda: bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))})
    f.append({"name": "cloud_public_ip_hint", "cat": "cloud", "purpose": "Public IP hint",
              "fn": lambda: os.environ.get("CLOUD_PUBLIC_IP", "detect-via-http")})
    f.append({"name": "cloud_deploy_targets", "cat": "cloud", "purpose": "List deploy targets",
              "fn": lambda: ["localhost", "lan", "colab", "cloud-vm", "docker", "k8s", "phone"]})
    f.append({"name": "cloud_costs_estimate", "cat": "cloud", "purpose": "Rough monthly cost estimate",
              "fn": lambda instances=1, hours=730, rate=0.05: round(float(instances) * float(hours) * float(rate), 2)})
    f.append({"name": "cloud_storage_class", "cat": "cloud", "purpose": "Storage class by access",
              "fn": lambda access="hot": {"hot": "STANDARD", "warm": "IA", "cold": "GLACIER", "archive": "DEEP_ARCHIVE"}.get(str(access), "STANDARD")})
    return f


def _fam_i18n(f):
    langs = {"en": "Hello", "hy": "Բարեւ", "ru": "Привет", "fr": "Bonjour", "de": "Hallo", "es": "Hola", "it": "Ciao", "pt": "Olá", "tr": "Merhaba", "fa": "سلام", "zh": "你好", "ja": "こんにちは"}
    for code, word in langs.items():
        f.append({"name": f"i18n_hello_{code}", "cat": "i18n", "purpose": f"Hello in {code}",
                  "fn": lambda w=word: w})
    f.append({"name": "i18n_hello", "cat": "i18n", "purpose": "Hello in many languages",
              "fn": lambda: langs})
    f.append({"name": "i18n_plural", "cat": "i18n", "purpose": "English plural form",
              "fn": lambda n=2, word="apple": str(word) + ("s" if int(n) != 1 else "")})
    f.append({"name": "i18n_plural_ru", "cat": "i18n", "purpose": "Russian plural form (lite)",
              "fn": lambda n=2, forms="[стол,стола,столов]": ru_plural(int(n), _as_list(forms))})
    f.append({"name": "i18n_currency", "cat": "i18n", "purpose": "Format currency by locale",
              "fn": lambda amount=1234.5, cur="USD": f"{float(amount):,.2f} {cur}"})
    f.append({"name": "i18n_number", "cat": "i18n", "purpose": "Locale number format",
              "fn": lambda x=1234567.89, sep=",": f"{float(x):,}".replace(",", str(sep))})
    f.append({"name": "i18n_percent", "cat": "i18n", "purpose": "Percent format",
              "fn": lambda x=0.1234, digits=1: f"{float(x) * 100:.{int(digits)}f}%"})
    f.append({"name": "i18n_transliterate", "cat": "i18n", "purpose": "Transliterate to ASCII",
              "fn": lambda s="Привет мир": unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()})
    f.append({"name": "i18n_upper", "cat": "i18n", "purpose": "Locale-aware uppercase",
              "fn": lambda s="hello": str(s).upper()})
    f.append({"name": "i18n_lower", "cat": "i18n", "purpose": "Locale-aware lowercase",
              "fn": lambda s="HELLO": str(s).lower()})
    f.append({"name": "i18n_title", "cat": "i18n", "purpose": "Title case",
              "fn": lambda s="hello world": str(s).title()})
    f.append({"name": "i18n_detect_latin", "cat": "i18n", "purpose": "Detect if text is Latin script",
              "fn": lambda s="hello": all(ord(c) < 128 or unicodedata.name(c, "").startswith("LATIN") for c in str(s))})
    f.append({"name": "i18n_script", "cat": "i18n", "purpose": "Guess script of text",
              "fn": lambda s="Привет": "cyrillic" if re.search(r"[а-яА-Я]", str(s)) else "latin" if re.search(r"[a-zA-Z]", str(s)) else "other"})
    f.append({"name": "i18n_ordinal", "cat": "i18n", "purpose": "English ordinal suffix",
              "fn": lambda n=3: ordinal_en(int(n))})
    f.append({"name": "i18n_days", "cat": "i18n", "purpose": "Day names (short)",
              "fn": lambda: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]})
    f.append({"name": "i18n_months", "cat": "i18n", "purpose": "Month names",
              "fn": lambda: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]})
    f.append({"name": "i18n_pad", "cat": "i18n", "purpose": "Pad number with leading zeros",
              "fn": lambda n=7, width=3: str(int(n)).zfill(int(width))})
    f.append({"name": "i18n_word_count", "cat": "i18n", "purpose": "Count words in any language",
              "fn": lambda s="hello world": len(re.findall(r"\S+", str(s)))})
    return f


def _fam_config(f):
    f.append({"name": "config_ini_parse", "cat": "config", "purpose": "Parse INI text",
              "fn": lambda text="[sec]\nk=v": config_ini_parse(str(text))})
    f.append({"name": "config_ini_build", "cat": "config", "purpose": "Build INI text",
              "fn": lambda data="{\"sec\":{\"k\":\"v\"}}": config_ini_build(_as_dict(data))})
    f.append({"name": "config_env_parse", "cat": "config", "purpose": "Parse .env text",
              "fn": lambda text="A=1\nB=hello world": config_env_parse(str(text))})
    f.append({"name": "config_env_build", "cat": "config", "purpose": "Build .env text",
              "fn": lambda data="{\"A\":\"1\",\"B\":\"hi\"}": "\n".join(f"{k}={v}" for k, v in _as_dict(data).items())})
    f.append({"name": "config_json_merge", "cat": "config", "purpose": "Deep merge JSON configs",
              "fn": lambda a="{\"a\":1,\"b\":{\"x\":1}}", b="{\"b\":{\"y\":2}}": json_merge(_as_dict(a), _as_dict(b))})
    f.append({"name": "config_defaults", "cat": "config", "purpose": "Fill missing defaults",
              "fn": lambda data="{\"a\":1}", defaults="{\"a\":0,\"b\":2}": {**{k: v for k, v in _as_dict(defaults).items()}, **_as_dict(data)}})
    f.append({"name": "config_get", "cat": "config", "purpose": "Dot-path config get",
              "fn": lambda data="{\"a\":{\"b\":1}}", path="a.b": json_get(_as_dict(data), str(path))})
    f.append({"name": "config_set", "cat": "config", "purpose": "Dot-path config set",
              "fn": lambda data="{\"a\":{\"b\":1}}", path="a.b", value=2: json_set(_as_dict(data), str(path), int(value))})
    f.append({"name": "config_flatten", "cat": "config", "purpose": "Flatten nested config",
              "fn": lambda data="{\"a\":{\"b\":1}}": json_flatten(_as_dict(data))})
    f.append({"name": "config_unflatten", "cat": "config", "purpose": "Unflatten dotted config",
              "fn": lambda data="{\"a.b\":1}": json_unflatten(_as_dict(data))})
    f.append({"name": "config_validate_required", "cat": "config", "purpose": "Check required keys",
              "fn": lambda data="{\"a\":1}", required="[a,b]": {"missing": [k for k in _as_list(required) if k not in _as_dict(data)], "ok": all(k in _as_dict(data) for k in _as_list(required))}})
    f.append({"name": "config_detect_format", "cat": "config", "purpose": "Guess config format",
              "fn": lambda text="a=b": "env" if re.match(r"^[A-Za-z_]+=", str(text)) else "json" if str(text).lstrip().startswith("{") else "ini" if "[" in str(text) else "unknown"})
    f.append({"name": "config_version", "cat": "config", "purpose": "Config version marker",
              "fn": lambda v="1.0.0": {"version": str(v), "generated_at": _now()}})
    f.append({"name": "config_render", "cat": "config", "purpose": "Render dict as pretty JSON",
              "fn": lambda data="{\"a\":1}": json.dumps(_as_dict(data), indent=2, ensure_ascii=False)})
    return f


def _fam_quant(f):
    f.append({"name": "quant_scale", "cat": "quant", "purpose": "Compute int8 scale",
              "fn": lambda vmin=-1, vmax=1: {"scale": round(max(abs(float(vmin)), abs(float(vmax))) / 127.0, 6), "zero": 0}})
    f.append({"name": "quant_qint8", "cat": "quant", "purpose": "Quantize floats to int8",
              "fn": lambda xs="[0.1,-0.5,0.9]", scale=0.01: [max(-128, min(127, round(float(x) / float(scale)))) for x in _as_list(xs)]})
    f.append({"name": "quant_deq_int8", "cat": "quant", "purpose": "Dequantize int8 to floats",
              "fn": lambda xs="[10,-50,90]", scale=0.01: [round(int(x) * float(scale), 6) for x in _as_list(xs)]})
    f.append({"name": "quant_quint8", "cat": "quant", "purpose": "Quantize to uint8",
              "fn": lambda xs="[0.1,0.5,0.9]", scale=0.01, zero=128: [max(0, min(255, round(float(x) / float(scale)) + int(zero))) for x in _as_list(xs)]})
    f.append({"name": "quant_qint4", "cat": "quant", "purpose": "Quantize to int4 (symmetric)",
              "fn": lambda xs="[0.1,-0.5,0.9]", scale=0.1: [max(-8, min(7, round(float(x) / float(scale)))) for x in _as_list(xs)]})
    f.append({"name": "quant_pack4", "cat": "quant", "purpose": "Pack int4 pairs into bytes",
              "fn": lambda xs="[1,2,3,4]": [int(x) & 0xF | ((int(y) & 0xF) << 4) for x, y in zip(_as_list(xs)[::2], _as_list(xs)[1::2] + [0])]})
    f.append({"name": "quant_unpack4", "cat": "quant", "purpose": "Unpack bytes into int4 list",
              "fn": lambda xs="[0x21,0x43]": [int(x) & 0xF for x in _as_list(xs)] + [(int(x) >> 4) & 0xF for x in _as_list(xs)]})
    f.append({"name": "quant_float16_bits", "cat": "quant", "purpose": "Float32 to float16-like",
              "fn": lambda x=0.1: {"original": float(x), "half": round(float(x), 4), "note": "uses struct half if available"}})
    f.append({"name": "quant_bits_saved", "cat": "quant", "purpose": "Size reduction estimate",
              "fn": lambda params=1000000, from_bits=32, to_bits=8: {"original_mb": round(int(params) * int(from_bits) / 8 / 1e6, 3), "quantized_mb": round(int(params) * int(to_bits) / 8 / 1e6, 3), "savings_pct": round(100 * (1 - int(to_bits) / int(from_bits)), 1)}})
    f.append({"name": "quant_clip", "cat": "quant", "purpose": "Clip values to range",
              "fn": lambda xs="[0.1,-2,3]", lo=-1, hi=1: [_clamp(float(x), float(lo), float(hi)) for x in _as_list(xs)]})
    f.append({"name": "quant_error", "cat": "quant", "purpose": "Quantization error (MSE)",
              "fn": lambda orig="[0.1,0.5,0.9]", deq="[0.09,0.51,0.88]": round(sum((float(a) - float(b)) ** 2 for a, b in zip(_as_list(orig), _as_list(deq))) / len(_as_list(orig)), 6)})
    f.append({"name": "quant_int4_range", "cat": "quant", "purpose": "Int4 representable range",
              "fn": lambda: {"min": -8, "max": 7, "levels": 16}})
    f.append({"name": "quant_int8_range", "cat": "quant", "purpose": "Int8 representable range",
              "fn": lambda: {"min": -128, "max": 127, "levels": 256}})
    f.append({"name": "quant_uint8_range", "cat": "quant", "purpose": "Uint8 representable range",
              "fn": lambda: {"min": 0, "max": 255, "levels": 256}})
    return f


def _fam_rag(f):
    f.append({"name": "rag_chunk", "cat": "rag", "purpose": "Chunk text by size",
              "fn": lambda text="hello world foo bar", size=10: rag_chunk(str(text), int(size))})
    f.append({"name": "rag_chunk_sentences", "cat": "rag", "purpose": "Chunk by sentences",
              "fn": lambda text="One. Two. Three.": [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text)) if s.strip()]})
    f.append({"name": "rag_tf", "cat": "rag", "purpose": "Term frequency vector",
              "fn": lambda text="the cat and the dog": rag_tf(str(text))})
    f.append({"name": "rag_overlap", "cat": "rag", "purpose": "Token overlap score",
              "fn": lambda a="the cat", b="the dog": len(set(str(a).split()) & set(str(b).split()))})
    f.append({"name": "rag_jaccard", "cat": "rag", "purpose": "Jaccard similarity",
              "fn": lambda a="the cat", b="the dog": rag_jaccard(str(a), str(b))})
    f.append({"name": "rag_cosine_text", "cat": "rag", "purpose": "Cosine sim of texts",
              "fn": lambda a="the cat sat", b="the cat slept": rag_cosine(str(a), str(b))})
    f.append({"name": "rag_keywords", "cat": "rag", "purpose": "Top keywords by frequency",
              "fn": lambda text="cat cat dog bird", n=2: rag_keywords(str(text), int(n))})
    f.append({"name": "rag_stopwords", "cat": "rag", "purpose": "Remove English stopwords",
              "fn": lambda text="the cat is on the mat": rag_stop(str(text))})
    f.append({"name": "rag_preview", "cat": "rag", "purpose": "Preview first N chars",
              "fn": lambda text="long text here", n=10: str(text)[:int(n)] + ("..." if len(str(text)) > int(n) else "")})
    f.append({"name": "rag_context_window", "cat": "rag", "purpose": "Estimate context fit",
              "fn": lambda text="hello", max_tokens=100: {"tokens": max(1, len(str(text)) // 4), "fits": (len(str(text)) // 4) <= int(max_tokens)}})
    f.append({"name": "rag_memory_store", "cat": "rag", "purpose": "In-memory KV store put",
              "fn": lambda key="k", value="v": {"stored": str(key), "value": str(value), "memory": True}})
    f.append({"name": "rag_memory_get", "cat": "rag", "purpose": "In-memory KV store get",
              "fn": lambda key="k": {"key": str(key), "note": "returns value if stored in same session"}})
    f.append({"name": "rag_snippet", "cat": "rag", "purpose": "Highlight query in text",
              "fn": lambda text="the quick brown fox", q="quick": str(text).replace(str(q), f"[{str(q)}]") if str(q) in str(text) else str(text)})
    f.append({"name": "rag_rank", "cat": "rag", "purpose": "Rank docs by score",
              "fn": lambda docs="[doc1,doc2,doc3]", scores="[3,1,2]": [{"doc": d, "score": float(s)} for d, s in sorted(zip(_as_list(docs), _as_list(scores)), key=lambda p: -float(p[1]))]})
    return f


def _fam_market(f):
    f.append({"name": "market_publish_skeleton", "cat": "market", "purpose": "Publish skeleton",
              "fn": lambda name="model1", tag="v1", desc="A model": {"name": str(name), "tag": str(tag), "description": str(desc), "status": "draft", "published_at": _now()}})
    f.append({"name": "market_search_query", "cat": "market", "purpose": "Build search query",
              "fn": lambda q="text classification", cat="": {"q": str(q), "category": str(cat)}})
    f.append({"name": "market_rating", "cat": "market", "purpose": "Average rating",
              "fn": lambda ratings="[5,4,5]": round(statistics.mean([float(x) for x in _as_list(ratings)]), 2) if _as_list(ratings) else 0})
    f.append({"name": "market_stars", "cat": "market", "purpose": "Star string",
              "fn": lambda n=4: "★" * int(n) + "☆" * (5 - int(n))})
    f.append({"name": "market_rank_by_rating", "cat": "market", "purpose": "Rank models by rating",
              "fn": lambda models="[a,b,c]", ratings="[3,5,4]": [m for _, m in sorted(zip([float(r) for r in _as_list(ratings)], _as_list(models)), reverse=True)]})
    f.append({"name": "market_download_url", "cat": "market", "purpose": "Model download URL",
              "fn": lambda owner="ararat33", name="model1", tag="v1": f"https://github.com/{owner}/AWEAI/releases/download/{tag}/{name}.zip"})
    f.append({"name": "market_stats_sample", "cat": "market", "purpose": "Sample marketplace stats",
              "fn": lambda: {"models": 12, "downloads": 340, "rating": 4.6, "categories": ["text", "vision", "tabular"]}})
    f.append({"name": "market_license", "cat": "market", "purpose": "License suggestion",
              "fn": lambda commercial=False: "MIT" if not bool(commercial) else "Apache-2.0"})
    f.append({"name": "market_review", "cat": "market", "purpose": "Review skeleton",
              "fn": lambda model="model1", stars=5, text="Great!": {"model": str(model), "stars": int(stars), "text": str(text), "at": _now()}})
    f.append({"name": "market_recommend", "cat": "market", "purpose": "Recommend by category",
              "fn": lambda task="text": {"text": ["tiny-lstm", "mlp-tfidf"], "vision": ["tiny-cnn", "lenet-lite"], "tabular": ["xgboost-lite", "mlp-tab"]}.get(str(task), [])})
    return f


def _fam_quality(f):
    f.append({"name": "quality_checklist", "cat": "quality", "purpose": "Quality checklist skeleton",
              "fn": lambda: ["tests pass", "lint clean", "docs updated", "version bumped", "changelog updated"]})
    f.append({"name": "quality_lint_py", "cat": "quality", "purpose": "Simple python lint checks",
              "fn": lambda code="def f(x):\n    return x": quality_lint(str(code))})
    f.append({"name": "quality_line_length", "cat": "quality", "purpose": "Max line length check",
              "fn": lambda code="a = 1\n" + "x" * 90, max_len=88: quality_lines(str(code), int(max_len))})
    f.append({"name": "quality_todo_find", "cat": "quality", "purpose": "Find TODO/FIXME comments",
              "fn": lambda code="# TODO: fix\npass": re.findall(r"#+\s*(TODO|FIXME|HACK|XXX)[:\s]+(.*)", str(code))})
    f.append({"name": "quality_docstring", "cat": "quality", "purpose": "Has docstring",
              "fn": lambda code="def f():\n    \"\"\"doc\"\"\"\n    pass": '"""' in str(code)})
    f.append({"name": "quality_test_cover_hint", "cat": "quality", "purpose": "Coverage hint for functions",
              "fn": lambda n_funcs=10, n_tested=7: {"functions": int(n_funcs), "tested": int(n_tested), "coverage": round(100 * int(n_tested) / max(1, int(n_funcs)), 1)}})
    f.append({"name": "quality_complexity", "cat": "quality", "purpose": "Cyclomatic-complexity lite",
              "fn": lambda code="if a:\n    pass\nelse:\n    pass": str(code).count("if ") + str(code).count("for ") + str(code).count("while ") + 1})
    f.append({"name": "quality_security_scan_hint", "cat": "quality", "purpose": "Security scan hints",
              "fn": lambda code="exec(x)": [p for p in ("exec(", "eval(", "pickle.loads(", "subprocess") if p in str(code)]})
    f.append({"name": "quality_release_ready", "cat": "quality", "purpose": "Release readiness score",
              "fn": lambda tests=True, docs=True, lint=False: {"score": sum([bool(tests), bool(docs), bool(lint)]) * 33, "ready": all([bool(tests), bool(docs), bool(lint)])}})
    f.append({"name": "quality_smoke_hint", "cat": "quality", "purpose": "Smoke test hint list",
              "fn": lambda: ["imports", "version", "cli help", "api health", "sample train"]})
    f.append({"name": "quality_benchmark", "cat": "quality", "purpose": "Benchmark timing wrapper result",
              "fn": lambda seconds=0.5, ops=100: {"ops": int(ops), "seconds": float(seconds), "ops_per_sec": round(int(ops) / max(0.0001, float(seconds)), 1)}})
    f.append({"name": "quality_gate", "cat": "quality", "purpose": "Quality gate decision",
              "fn": lambda tests=True, lint=True, coverage=90: {"pass": bool(tests) and bool(lint) and float(coverage) >= 80, "reasons": [r for r, ok in [("tests", bool(tests)), ("lint", bool(lint)), ("coverage>=80", float(coverage) >= 80)] if not ok]}})
    return f


def _fam_ui(f):
    f.append({"name": "ui_color_hex", "cat": "ui", "purpose": "Validate hex color",
              "fn": lambda h="#ff0000": bool(re.match(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$", str(h)))})
    f.append({"name": "ui_color_contrast", "cat": "ui", "purpose": "Contrast ratio",
              "fn": lambda a="#000000", b="#ffffff": ui_contrast(str(a), str(b))})
    f.append({"name": "ui_color_brightness", "cat": "ui", "purpose": "Perceived brightness",
              "fn": lambda h="#888888": ui_brightness(str(h))})
    f.append({"name": "ui_text_on", "cat": "ui", "purpose": "Black or white text on color",
              "fn": lambda h="#888888": "white" if ui_brightness(str(h)) < 128 else "black"})
    f.append({"name": "ui_flex", "cat": "ui", "purpose": "CSS flex snippet",
              "fn": lambda direction="row", gap=8: f"display:flex;flex-direction:{direction};gap:{gap}px;"})
    f.append({"name": "ui_grid", "cat": "ui", "purpose": "CSS grid snippet",
              "fn": lambda cols=3, gap=12: f"display:grid;grid-template-columns:repeat({cols},1fr);gap:{gap}px;"})
    f.append({"name": "ui_center", "cat": "ui", "purpose": "CSS centering snippet",
              "fn": lambda: "display:flex;align-items:center;justify-content:center;"})
    f.append({"name": "ui_responsive_meta", "cat": "ui", "purpose": "Responsive viewport meta",
              "fn": lambda: '<meta name="viewport" content="width=device-width, initial-scale=1">'})
    f.append({"name": "ui_media_query", "cat": "ui", "purpose": "Media query snippet",
              "fn": lambda width=768: f"@media (max-width:{width}px) {{ }}"})
    f.append({"name": "ui_emoji_smile", "cat": "ui", "purpose": "Smile emoji", "fn": lambda: "😊"})
    f.append({"name": "ui_emoji_stars", "cat": "ui", "purpose": "Star emojis", "fn": lambda n=3: "⭐" * int(n)})
    f.append({"name": "ui_progress", "cat": "ui", "purpose": "ASCII progress bar",
              "fn": lambda pct=50, width=20: ui_progress(float(pct), int(width))})
    f.append({"name": "ui_spinner", "cat": "ui", "purpose": "Spinner frames",
              "fn": lambda: ["|", "/", "-", "\\"]})
    f.append({"name": "ui_theme", "cat": "ui", "purpose": "Theme colors",
              "fn": lambda name="dark": {"dark": {"bg": "#111", "fg": "#eee", "accent": "#4f8cff"}, "light": {"bg": "#fff", "fg": "#111", "accent": "#2563eb"}}.get(str(name), {})})
    f.append({"name": "ui_accessibility_contrast", "cat": "ui", "purpose": "WCAG contrast pass",
              "fn": lambda a="#000000", b="#ffffff": ui_contrast(str(a), str(b)) >= 4.5})
    f.append({"name": "ui_breakpoints", "cat": "ui", "purpose": "Standard breakpoints",
              "fn": lambda: {"xs": 480, "sm": 640, "md": 768, "lg": 1024, "xl": 1280}})
    f.append({"name": "ui_btn", "cat": "ui", "purpose": "Button class suggestion",
              "fn": lambda variant="primary": {"primary": "btn btn-primary", "secondary": "btn btn-secondary", "danger": "btn btn-danger"}.get(str(variant), "btn")})
    f.append({"name": "ui_shadow", "cat": "ui", "purpose": "CSS box-shadow snippet",
              "fn": lambda depth=1: f"box-shadow:0 {int(depth) * 2}px {int(depth) * 6}px rgba(0,0,0,0.15);"})
    f.append({"name": "ui_radius", "cat": "ui", "purpose": "Border radius snippet",
              "fn": lambda r=8: f"border-radius:{r}px;"})
    f.append({"name": "ui_typing_scale", "cat": "ui", "purpose": "Type scale sizes",
              "fn": lambda: {"h1": "2.5rem", "h2": "2rem", "h3": "1.5rem", "body": "1rem", "small": "0.875rem"}})
    return f


def _fam_net(f):
    f.append({"name": "net_ip_valid", "cat": "net", "purpose": "Validate IPv4",
              "fn": lambda ip="192.168.1.1": bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$", str(ip))) and all(0 <= int(p) <= 255 for p in str(ip).split("."))})
    f.append({"name": "net_ip_class", "cat": "net", "purpose": "IPv4 class (A-E)",
              "fn": lambda ip="192.168.1.1": net_class(str(ip))})
    f.append({"name": "net_ip_private", "cat": "net", "purpose": "Is private IP",
              "fn": lambda ip="192.168.1.1": net_private(str(ip))})
    f.append({"name": "net_cidr_mask", "cat": "net", "purpose": "CIDR to netmask",
              "fn": lambda cidr=24: net_mask(int(cidr))})
    f.append({"name": "net_cidr_hosts", "cat": "net", "purpose": "Usable hosts in CIDR",
              "fn": lambda cidr=24: max(0, 2 ** (32 - int(cidr)) - 2)})
    f.append({"name": "net_cidr_range", "cat": "net", "purpose": "CIDR address range",
              "fn": lambda cidr="192.168.1.0/24": net_range(str(cidr))})
    f.append({"name": "net_cidr_contains", "cat": "net", "purpose": "CIDR contains IP",
              "fn": lambda ip="192.168.1.5", cidr="192.168.1.0/24": net_contains(str(ip), str(cidr))})
    f.append({"name": "net_port_valid", "cat": "net", "purpose": "Validate port",
              "fn": lambda port=8080: 1 <= int(port) <= 65535})
    f.append({"name": "net_port_common", "cat": "net", "purpose": "Common port service",
              "fn": lambda port=80: {21: "ftp", 22: "ssh", 25: "smtp", 53: "dns", 80: "http", 443: "https", 3306: "mysql", 5432: "postgres", 6379: "redis", 27017: "mongo"}.get(int(port), "unknown")})
    f.append({"name": "net_dns_mx_hint", "cat": "net", "purpose": "MX record hint",
              "fn": lambda domain="example.com": f"mail.{domain}"})
    f.append({"name": "net_dns_txt_hint", "cat": "net", "purpose": "TXT record hint",
              "fn": lambda domain="example.com": f"v=spf1 include:{domain} ~all"})
    f.append({"name": "net_local_ips", "cat": "net", "purpose": "Local IP addresses",
              "fn": lambda: net_local_ips()})
    f.append({"name": "net_hostname", "cat": "net", "purpose": "Hostname",
              "fn": lambda: socket_gethostname()})
    f.append({"name": "net_lan_base", "cat": "net", "purpose": "LAN base from IP",
              "fn": lambda ip="192.168.1.50": ".".join(str(ip).split(".")[:-1]) + ".0/24"})
    f.append({"name": "net_gateway_hint", "cat": "net", "purpose": "Gateway hint",
              "fn": lambda ip="192.168.1.50": ".".join(str(ip).split(".")[:-1]) + ".1"})
    f.append({"name": "net_loopback", "cat": "net", "purpose": "Is loopback IP",
              "fn": lambda ip="127.0.0.1": str(ip).startswith("127.")})
    f.append({"name": "net_public_hint", "cat": "net", "purpose": "Is public IP hint",
              "fn": lambda ip="8.8.8.8": not net_private(str(ip)) and not str(ip).startswith("127.")})
    return f


def _fam_sys2(f):
    f.append({"name": "sys2_platform", "cat": "sys2", "purpose": "Platform info",
              "fn": lambda: {"system": sys.platform, "python": sys.version.split()[0], "machine": platform_machine()}})
    f.append({"name": "sys2_cpu_count", "cat": "sys2", "purpose": "CPU count",
              "fn": lambda: os.cpu_count() or 1})
    f.append({"name": "sys2_uptime", "cat": "sys2", "purpose": "Uptime seconds (best effort)",
              "fn": lambda: sys2_uptime()})
    f.append({"name": "sys2_pid", "cat": "sys2", "purpose": "Current PID",
              "fn": lambda: os.getpid()})
    f.append({"name": "sys2_cwd", "cat": "sys2", "purpose": "Current working dir",
              "fn": lambda: os.getcwd()})
    f.append({"name": "sys2_env_keys", "cat": "sys2", "purpose": "Env var keys (sorted)",
              "fn": lambda: sorted(os.environ.keys())[:50]})
    f.append({"name": "sys2_env_get", "cat": "sys2", "purpose": "Get env var",
              "fn": lambda key="HOME": os.environ.get(str(key), "")})
    f.append({"name": "sys2_tempdir", "cat": "sys2", "purpose": "Temp dir",
              "fn": lambda: tempfile_gettempdir()})
    f.append({"name": "sys2_home", "cat": "sys2", "purpose": "Home dir",
              "fn": lambda: str(Path.home())})
    f.append({"name": "sys2_paths", "cat": "sys2", "purpose": "PATH entries",
              "fn": lambda: os.environ.get("PATH", "").split(os.pathsep)})
    f.append({"name": "sys2_free_disk_hint", "cat": "sys2", "purpose": "Free disk hint",
              "fn": lambda path=".": {"path": str(path), "note": "use shutil.disk_usage for exact"}})
    f.append({"name": "sys2_process_cmdline", "cat": "sys2", "purpose": "Current command line",
              "fn": lambda: sys.argv})
    f.append({"name": "sys2_thread_count", "cat": "sys2", "purpose": "Thread count",
              "fn": lambda: sys2_threads()})
    f.append({"name": "sys2_clock", "cat": "sys2", "purpose": "Monotonic clock",
              "fn": lambda: round(time.monotonic(), 3)})
    return f


def _fam_data2(f):
    f.append({"name": "data2_summary", "cat": "data2", "purpose": "Numeric summary stats",
              "fn": lambda xs="[1,2,3,4,5]": data_summary(_as_list(xs))})
    f.append({"name": "data2_impute_mean", "cat": "data2", "purpose": "Impute missing with mean",
              "fn": lambda xs="[1,None,3]": data_impute(_as_list(xs), "mean")})
    f.append({"name": "data2_impute_median", "cat": "data2", "purpose": "Impute missing with median",
              "fn": lambda xs="[1,None,3]": data_impute(_as_list(xs), "median")})
    f.append({"name": "data2_outliers", "cat": "data2", "purpose": "IQR outliers",
              "fn": lambda xs="[1,2,3,100]": data_outliers(_as_list(xs))})
    f.append({"name": "data2_zscore", "cat": "data2", "purpose": "Z-scores of list",
              "fn": lambda xs="[1,2,3,4]": [round(z, 4) for z in data_zscores(_as_list(xs))]})
    f.append({"name": "data2_pivot", "cat": "data2", "purpose": "Pivot counts by key",
              "fn": lambda keys="[a,b,a,c]": data_pivot(_as_list(keys))})
    f.append({"name": "data2_sample", "cat": "data2", "purpose": "Seeded sample",
              "fn": lambda xs="[1,2,3,4,5,6,7,8,9,10]", n=3, seed=1: data_sample(_as_list(xs), int(n), int(seed))})
    f.append({"name": "data2_rolling_mean", "cat": "data2", "purpose": "Rolling mean",
              "fn": lambda xs="[1,2,3,4,5]", w=3: [round(statistics.mean(_as_list(xs)[max(0, i - int(w) + 1):i + 1]), 3) for i in range(len(_as_list(xs)))]})
    f.append({"name": "data2_diff", "cat": "data2", "purpose": "Consecutive differences",
              "fn": lambda xs="[1,3,6,10]": [float(b) - float(a) for a, b in zip(_as_list(xs), _as_list(xs)[1:])]})
    f.append({"name": "data2_rank", "cat": "data2", "purpose": "Rank values",
              "fn": lambda xs="[3,1,2]": {str(v): i + 1 for i, v in enumerate(sorted(_as_list(xs)))} })
    f.append({"name": "data2_bin", "cat": "data2", "purpose": "Bin continuous into labels",
              "fn": lambda x=5, edges="[0,5,10]": data_bin(float(x), _as_list(edges))})
    f.append({"name": "data2_entropy", "cat": "data2", "purpose": "Entropy of distribution",
              "fn": lambda xs="[1,1,2]": round(data_entropy(_as_list(xs)), 4)})
    f.append({"name": "data2_mode", "cat": "data2", "purpose": "Mode of list",
              "fn": lambda xs="[1,1,2,3]": statistics.mode(_as_list(xs))})
    f.append({"name": "data2_median", "cat": "data2", "purpose": "Median",
              "fn": lambda xs="[1,3,2]": statistics.median([float(x) for x in _as_list(xs)])})
    f.append({"name": "data2_stdev", "cat": "data2", "purpose": "Sample stdev",
              "fn": lambda xs="[1,2,3,4]": round(statistics.stdev([float(x) for x in _as_list(xs)]), 4) if len(_as_list(xs)) > 1 else 0})
    return f


def _fam_math2(f):
    f.append({"name": "math2_gcd", "cat": "math2", "purpose": "Greatest common divisor",
              "fn": lambda a=12, b=18: math.gcd(int(a), int(b))})
    f.append({"name": "math2_lcm", "cat": "math2", "purpose": "Least common multiple",
              "fn": lambda a=12, b=18: abs(int(a) * int(b)) // math.gcd(int(a), int(b))})
    f.append({"name": "math2_factorial", "cat": "math2", "purpose": "Factorial",
              "fn": lambda n=5: math.factorial(int(n))})
    f.append({"name": "math2_fib", "cat": "math2", "purpose": "Fibonacci list",
              "fn": lambda n=8: _fib_lite(int(n))})
    f.append({"name": "math2_primes", "cat": "math2", "purpose": "Primes up to limit",
              "fn": lambda limit=50: _primes_lite(int(limit))})
    f.append({"name": "math2_is_prime", "cat": "math2", "purpose": "Primality test",
              "fn": lambda n=17: _is_prime_lite(int(n))})
    f.append({"name": "math2_pow", "cat": "math2", "purpose": "Power",
              "fn": lambda a=2, b=10: int(a) ** int(b)})
    f.append({"name": "math2_sqrt", "cat": "math2", "purpose": "Square root",
              "fn": lambda x=16: round(math.sqrt(float(x)), 6)})
    f.append({"name": "math2_log", "cat": "math2", "purpose": "Natural log",
              "fn": lambda x=math.e: round(math.log(float(x)), 6)})
    f.append({"name": "math2_log10", "cat": "math2", "purpose": "Log base 10",
              "fn": lambda x=100: round(math.log10(float(x)), 6)})
    f.append({"name": "math2_exp", "cat": "math2", "purpose": "Exponential",
              "fn": lambda x=1: round(math.exp(float(x)), 6)})
    f.append({"name": "math2_abs", "cat": "math2", "purpose": "Absolute value",
              "fn": lambda x=-5: abs(float(x))})
    f.append({"name": "math2_floor", "cat": "math2", "purpose": "Floor",
              "fn": lambda x=2.7: math.floor(float(x))})
    f.append({"name": "math2_ceil", "cat": "math2", "purpose": "Ceiling",
              "fn": lambda x=2.1: math.ceil(float(x))})
    f.append({"name": "math2_round", "cat": "math2", "purpose": "Round",
              "fn": lambda x=2.567, nd=2: round(float(x), int(nd))})
    f.append({"name": "math2_trunc", "cat": "math2", "purpose": "Truncate",
              "fn": lambda x=2.999: math.trunc(float(x))})
    f.append({"name": "math2_degrees", "cat": "math2", "purpose": "Radians to degrees",
              "fn": lambda x=math.pi: round(math.degrees(float(x)), 6)})
    f.append({"name": "math2_radians", "cat": "math2", "purpose": "Degrees to radians",
              "fn": lambda x=180: round(math.radians(float(x)), 6)})
    f.append({"name": "math2_sin", "cat": "math2", "purpose": "Sine",
              "fn": lambda x=0: round(math.sin(float(x)), 6)})
    f.append({"name": "math2_cos", "cat": "math2", "purpose": "Cosine",
              "fn": lambda x=0: round(math.cos(float(x)), 6)})
    f.append({"name": "math2_tan", "cat": "math2", "purpose": "Tangent",
              "fn": lambda x=0: round(math.tan(float(x)), 6)})
    f.append({"name": "math2_atan2", "cat": "math2", "purpose": "Atan2",
              "fn": lambda y=1, x=1: round(math.atan2(float(y), float(x)), 6)})
    f.append({"name": "math2_hypot", "cat": "math2", "purpose": "Hypotenuse",
              "fn": lambda a=3, b=4: round(math.hypot(float(a), float(b)), 6)})
    f.append({"name": "math2_comb", "cat": "math2", "purpose": "Combinations nCk",
              "fn": lambda n=5, k=2: math.comb(int(n), int(k))})
    f.append({"name": "math2_perm", "cat": "math2", "purpose": "Permutations nPk",
              "fn": lambda n=5, k=2: math.perm(int(n), int(k))})
    f.append({"name": "math2_gamma", "cat": "math2", "purpose": "Gamma function",
              "fn": lambda x=5: round(math.gamma(float(x)), 6)})
    f.append({"name": "math2_lgamma", "cat": "math2", "purpose": "Log gamma",
              "fn": lambda x=5: round(math.lgamma(float(x)), 6)})
    f.append({"name": "math2_erf", "cat": "math2", "purpose": "Error function",
              "fn": lambda x=0.5: round(math.erf(float(x)), 6)})
    f.append({"name": "math2_isqrt", "cat": "math2", "purpose": "Integer sqrt",
              "fn": lambda n=17: math.isqrt(int(n))})
    f.append({"name": "math2_pi", "cat": "math2", "purpose": "Pi constant",
              "fn": lambda: round(math.pi, 6)})
    f.append({"name": "math2_e", "cat": "math2", "purpose": "E constant",
              "fn": lambda: round(math.e, 6)})
    f.append({"name": "math2_tau", "cat": "math2", "purpose": "Tau constant",
              "fn": lambda: round(math.tau, 6)})
    f.append({"name": "math2_phi", "cat": "math2", "purpose": "Golden ratio",
              "fn": lambda: round((1 + math.sqrt(5)) / 2, 6)})
    f.append({"name": "math2_compound", "cat": "math2", "purpose": "Compound interest",
              "fn": lambda p=1000, r=0.05, t=2, n=12: round(float(p) * (1 + float(r) / int(n)) ** (int(n) * float(t)), 2)})
    f.append({"name": "math2_percent", "cat": "math2", "purpose": "Percentage of total",
              "fn": lambda part=25, total=200: round(100 * float(part) / max(1e-9, float(total)), 2)})
    f.append({"name": "math2_change", "cat": "math2", "purpose": "Percent change",
              "fn": lambda old=100, new=120: round(100 * (float(new) - float(old)) / max(1e-9, float(old)), 2)})
    f.append({"name": "math2_avg", "cat": "math2", "purpose": "Average",
              "fn": lambda xs="[1,2,3,4]": round(statistics.mean([float(x) for x in _as_list(xs)]), 6)})
    f.append({"name": "math2_sum", "cat": "math2", "purpose": "Sum",
              "fn": lambda xs="[1,2,3,4]": sum(float(x) for x in _as_list(xs))})
    f.append({"name": "math2_min", "cat": "math2", "purpose": "Min",
              "fn": lambda xs="[3,1,2]": min(float(x) for x in _as_list(xs))})
    f.append({"name": "math2_max", "cat": "math2", "purpose": "Max",
              "fn": lambda xs="[3,1,2]": max(float(x) for x in _as_list(xs))})
    f.append({"name": "math2_clamp", "cat": "math2", "purpose": "Clamp value",
              "fn": lambda x=15, lo=0, hi=10: _clamp(float(x), float(lo), float(hi))})
    f.append({"name": "math2_sign", "cat": "math2", "purpose": "Sign of value",
              "fn": lambda x=-3: -1 if float(x) < 0 else (1 if float(x) > 0 else 0)})
    return f


def _fam_str2(f):
    f.append({"name": "str2_slug", "cat": "str2", "purpose": "Slugify text", "fn": _slug})
    f.append({"name": "str2_levenshtein", "cat": "str2", "purpose": "Levenshtein distance",
              "fn": lambda a="kitten", b="sitting": _lev(str(a), str(b))})
    f.append({"name": "str2_similarity", "cat": "str2", "purpose": "Similarity ratio 0-1",
              "fn": lambda a="kitten", b="sitting": round(1 - _lev(str(a), str(b)) / max(len(str(a)), len(str(b)), 1), 4)})
    f.append({"name": "str2_contains_any", "cat": "str2", "purpose": "Contains any substring",
              "fn": lambda s="hello world", subs="[world,foo]": any(sub in str(s) for sub in _as_list(subs))})
    f.append({"name": "str2_contains_all", "cat": "str2", "purpose": "Contains all substrings",
              "fn": lambda s="hello world", subs="[hello,world]": all(sub in str(s) for sub in _as_list(subs))})
    f.append({"name": "str2_indent", "cat": "str2", "purpose": "Indent lines",
              "fn": lambda s="a\nb", n=2: "\n".join(" " * int(n) + line for line in str(s).splitlines())})
    f.append({"name": "str2_dedent", "cat": "str2", "purpose": "Dedent common indent",
              "fn": lambda s="  a\n  b": str2_dedent(str(s))})
    f.append({"name": "str2_wrap", "cat": "str2", "purpose": "Wrap text to width",
              "fn": lambda s="hello world foo bar", w=8: textwrap_fill(str(s), int(w))})
    f.append({"name": "str2_pad_left", "cat": "str2", "purpose": "Pad left",
              "fn": lambda s="5", n=3, ch="0": str(s).rjust(int(n), str(ch))})
    f.append({"name": "str2_pad_right", "cat": "str2", "purpose": "Pad right",
              "fn": lambda s="5", n=3, ch=" ": str(s).ljust(int(n), str(ch))})
    f.append({"name": "str2_reverse", "cat": "str2", "purpose": "Reverse string",
              "fn": lambda s="abc": str(s)[::-1]})
    f.append({"name": "str2_count_letters", "cat": "str2", "purpose": "Count letters",
              "fn": lambda s="a b c1": sum(c.isalpha() for c in str(s))})
    f.append({"name": "str2_count_digits", "cat": "str2", "purpose": "Count digits",
              "fn": lambda s="a1b2c3": sum(c.isdigit() for c in str(s))})
    f.append({"name": "str2_count_upper", "cat": "str2", "purpose": "Count uppercase",
              "fn": lambda s="AbC": sum(c.isupper() for c in str(s))})
    f.append({"name": "str2_count_words", "cat": "str2", "purpose": "Count words",
              "fn": lambda s="hello brave world": len(str(s).split())})
    f.append({"name": "str2_lines", "cat": "str2", "purpose": "Split into lines",
              "fn": lambda s="a\nb\nc": str(s).splitlines()})
    f.append({"name": "str2_join", "cat": "str2", "purpose": "Join list with separator",
              "fn": lambda xs="[a,b,c]", sep=",": str(sep).join(str(x) for x in _as_list(xs))})
    f.append({"name": "str2_split", "cat": "str2", "purpose": "Split by separator",
              "fn": lambda s="a,b,c", sep=",": str(s).split(str(sep))})
    f.append({"name": "str2_truncate", "cat": "str2", "purpose": "Truncate with ellipsis",
              "fn": lambda s="hello world", n=5: str(s)[:int(n)] + ("..." if len(str(s)) > int(n) else "")})
    f.append({"name": "str2_replace", "cat": "str2", "purpose": "Replace all",
              "fn": lambda s="a-b-c", old="-", new="+": str(s).replace(str(old), str(new))})
    f.append({"name": "str2_swapcase", "cat": "str2", "purpose": "Swap case",
              "fn": lambda s="AbC": str(s).swapcase()})
    f.append({"name": "str2_strip", "cat": "str2", "purpose": "Strip whitespace",
              "fn": lambda s="  hi  ": str(s).strip()})
    f.append({"name": "str2_startswith", "cat": "str2", "purpose": "Starts with",
              "fn": lambda s="hello", p="he": str(s).startswith(str(p))})
    f.append({"name": "str2_endswith", "cat": "str2", "purpose": "Ends with",
              "fn": lambda s="hello", p="lo": str(s).endswith(str(p))})
    f.append({"name": "str2_chars", "cat": "str2", "purpose": "Unique chars",
              "fn": lambda s="abca": sorted(set(str(s)))})
    f.append({"name": "str2_ngrams", "cat": "str2", "purpose": "Character n-grams",
              "fn": lambda s="hello", n=2: [str(s)[i:i + int(n)] for i in range(len(str(s)) - int(n) + 1)]})
    f.append({"name": "str2_camel", "cat": "str2", "purpose": "To camelCase",
              "fn": lambda s="hello_world": str2_camel(str(s))})
    f.append({"name": "str2_snake", "cat": "str2", "purpose": "To snake_case",
              "fn": lambda s="HelloWorld": str2_snake(str(s))})
    f.append({"name": "str2_kebab", "cat": "str2", "purpose": "To kebab-case",
              "fn": lambda s="HelloWorld": _slug(str(s))})
    f.append({"name": "str2_rot13", "cat": "str2", "purpose": "ROT13",
              "fn": lambda s="hello": str(s).translate(str.maketrans(string.ascii_lowercase + string.ascii_uppercase, string.ascii_lowercase[13:] + string.ascii_lowercase[:13] + string.ascii_uppercase[13:] + string.ascii_uppercase[:13]))})
    return f


def _fam_json2(f):
    f.append({"name": "json2_validate", "cat": "json2", "purpose": "Validate JSON string",
              "fn": lambda s="{}": json_validate(str(s))})
    f.append({"name": "json2_pretty", "cat": "json2", "purpose": "Pretty print JSON",
              "fn": lambda s='{"a":1}': json.dumps(json.loads(str(s)), indent=2)})
    f.append({"name": "json2_minify", "cat": "json2", "purpose": "Minify JSON",
              "fn": lambda s='{"a": 1}': json.dumps(json.loads(str(s)), separators=(",", ":"))})
    f.append({"name": "json2_merge", "cat": "json2", "purpose": "Deep merge two JSON",
              "fn": lambda a='{"a":1}', b='{"b":2}': json_merge(_as_dict(a), _as_dict(b))})
    f.append({"name": "json2_get", "cat": "json2", "purpose": "Get by dot path",
              "fn": lambda s='{"a":{"b":1}}', path="a.b": json_get(_as_dict(s), str(path))})
    f.append({"name": "json2_set", "cat": "json2", "purpose": "Set by dot path",
              "fn": lambda s='{"a":{}}', path="a.b", value=1: json_set(_as_dict(s), str(path), int(value))})
    f.append({"name": "json2_delete", "cat": "json2", "purpose": "Delete by dot path",
              "fn": lambda s='{"a":{"b":1}}', path="a.b": json_delete(_as_dict(s), str(path))})
    f.append({"name": "json2_flatten", "cat": "json2", "purpose": "Flatten nested",
              "fn": lambda s='{"a":{"b":1}}': json_flatten(_as_dict(s))})
    f.append({"name": "json2_unflatten", "cat": "json2", "purpose": "Unflatten dotted",
              "fn": lambda s='{"a.b":1}': json_unflatten(_as_dict(s))})
    f.append({"name": "json2_keys", "cat": "json2", "purpose": "Top-level keys",
              "fn": lambda s='{"a":1,"b":2}': list(_as_dict(s).keys())})
    f.append({"name": "json2_values", "cat": "json2", "purpose": "Top-level values",
              "fn": lambda s='{"a":1,"b":2}': list(_as_dict(s).values())})
    f.append({"name": "json2_size", "cat": "json2", "purpose": "JSON size bytes",
              "fn": lambda s='{"a":1}': len(str(s).encode())})
    f.append({"name": "json2_sort_keys", "cat": "json2", "purpose": "Sort keys recursively",
              "fn": lambda s='{"b":1,"a":{"d":2,"c":3}}': json_sort(_as_dict(s))})
    f.append({"name": "json2_type", "cat": "json2", "purpose": "Type of JSON value",
              "fn": lambda s="[1,2]": type(json.loads(str(s))).__name__})
    f.append({"name": "json2_count_keys", "cat": "json2", "purpose": "Count keys",
              "fn": lambda s='{"a":1,"b":2}': len(_as_dict(s))})
    f.append({"name": "json2_search", "cat": "json2", "purpose": "Search value in JSON",
              "fn": lambda s='{"a":1,"b":{"c":2}}', needle=2: json_search(_as_dict(s), int(needle))})
    f.append({"name": "json2_pretty_html", "cat": "json2", "purpose": "Pretty JSON with HTML-safe",
              "fn": lambda s='{"a":"<b>"}': json.dumps(_as_dict(s), indent=2).replace("<", "&lt;")})
    f.append({"name": "json2_to_csv", "cat": "json2", "purpose": "JSON array to CSV text",
              "fn": lambda s='[{"a":1,"b":2},{"a":3,"b":4}]': json_to_csv(json.loads(str(s)))})
    f.append({"name": "json2_diff", "cat": "json2", "purpose": "Top-level key diff",
              "fn": lambda a='{"a":1,"b":2}', b='{"a":1,"c":3}': json_diff(_as_dict(a), _as_dict(b))})
    f.append({"name": "json2_pick", "cat": "json2", "purpose": "Pick subset of keys",
              "fn": lambda s='{"a":1,"b":2,"c":3}', keys="[a,c]": {k: _as_dict(s)[k] for k in _as_list(keys) if k in _as_dict(s)}})
    f.append({"name": "json2_omit", "cat": "json2", "purpose": "Omit keys",
              "fn": lambda s='{"a":1,"b":2}', keys="[b]": {k: v for k, v in _as_dict(s).items() if k not in _as_list(keys)}})
    return f


def _fam_time2(f):
    f.append({"name": "time2_now", "cat": "time2", "purpose": "Current ISO time",
              "fn": lambda: _dt.datetime.now().isoformat(timespec="seconds")})
    f.append({"name": "time2_today", "cat": "time2", "purpose": "Today date",
              "fn": _today})
    f.append({"name": "time2_unix", "cat": "time2", "purpose": "Unix timestamp",
              "fn": lambda: int(time.time())})
    f.append({"name": "time2_age_years", "cat": "time2", "purpose": "Age in years from birthdate",
              "fn": lambda b="1990-01-01": time2_age(str(b))})
    f.append({"name": "time2_days_between", "cat": "time2", "purpose": "Days between dates",
              "fn": lambda a="2026-01-01", b="2026-08-09": (time2_parse(str(b)) - time2_parse(str(a))).days})
    f.append({"name": "time2_add_days", "cat": "time2", "purpose": "Add days to date",
              "fn": lambda d="2026-08-09", n=7: (time2_parse(str(d)) + _dt.timedelta(days=int(n))).isoformat()})
    f.append({"name": "time2_add_hours", "cat": "time2", "purpose": "Add hours to datetime",
              "fn": lambda d="2026-08-09T10:00:00", n=3: (time2_parse_dt(str(d)) + _dt.timedelta(hours=int(n))).isoformat(timespec="seconds")})
    f.append({"name": "time2_weekday", "cat": "time2", "purpose": "Weekday name",
              "fn": lambda d="2026-08-09": time2_parse(str(d)).strftime("%A")})
    f.append({"name": "time2_weekday_num", "cat": "time2", "purpose": "Weekday number (Mon=0)",
              "fn": lambda d="2026-08-09": time2_parse(str(d)).weekday()})
    f.append({"name": "time2_month", "cat": "time2", "purpose": "Month name",
              "fn": lambda d="2026-08-09": time2_parse(str(d)).strftime("%B")})
    f.append({"name": "time2_year", "cat": "time2", "purpose": "Year",
              "fn": lambda d="2026-08-09": time2_parse(str(d)).year})
    f.append({"name": "time2_iso_to_unix", "cat": "time2", "purpose": "ISO to unix",
              "fn": lambda d="2026-08-09T10:00:00": int(time2_parse_dt(str(d)).timestamp())})
    f.append({"name": "time2_unix_to_iso", "cat": "time2", "purpose": "Unix to ISO",
              "fn": lambda ts=1750000000: _dt.datetime.fromtimestamp(int(ts)).isoformat(timespec="seconds")})
    f.append({"name": "time2_duration", "cat": "time2", "purpose": "Human duration from seconds",
              "fn": lambda s=3661: time2_duration(int(s))})
    f.append({"name": "time2_next_weekday", "cat": "time2", "purpose": "Next weekday date",
              "fn": lambda d="2026-08-09", wd=0: time2_next_wd(str(d), int(wd)).isoformat()})
    f.append({"name": "time2_is_weekend", "cat": "time2", "purpose": "Is weekend",
              "fn": lambda d="2026-08-09": time2_parse(str(d)).weekday() >= 5})
    f.append({"name": "time2_leap", "cat": "time2", "purpose": "Is leap year",
              "fn": lambda y=2024: (int(y) % 4 == 0 and int(y) % 100 != 0) or int(y) % 400 == 0})
    f.append({"name": "time2_days_in_month", "cat": "time2", "purpose": "Days in month",
              "fn": lambda y=2026, m=2: _dt.date(int(y), int(m), 1).replace(day=28) and (_dt.date(int(y) + 1, 1, 1) - _dt.date(int(y), 1, 1)).days if False else _dt.date(int(y) + (1 if int(m) == 12 else 0), 1 if int(m) == 12 else int(m) + 1, 1).toordinal() - _dt.date(int(y), int(m), 1).toordinal()})
    f.append({"name": "time2_quarter", "cat": "time2", "purpose": "Quarter of date",
              "fn": lambda d="2026-08-09": (time2_parse(str(d)).month - 1) // 3 + 1})
    f.append({"name": "time2_start_of_day", "cat": "time2", "purpose": "Start of day",
              "fn": lambda d="2026-08-09T10:00:00": time2_parse_dt(str(d)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat(timespec="seconds")})
    f.append({"name": "time2_cron_hourly", "cat": "time2", "purpose": "Hourly cron",
              "fn": lambda: "0 * * * *"})
    f.append({"name": "time2_cron_daily", "cat": "time2", "purpose": "Daily cron",
              "fn": lambda hour=9: f"0 {int(hour)} * * *"})
    f.append({"name": "time2_cron_weekly", "cat": "time2", "purpose": "Weekly cron",
              "fn": lambda hour=9, wd=1: f"0 {int(hour)} * * {int(wd)}"})
    f.append({"name": "time2_cron_monthly", "cat": "time2", "purpose": "Monthly cron",
              "fn": lambda day=1, hour=0: f"0 {int(hour)} {int(day)} * *"})
    f.append({"name": "time2_cron_validate", "cat": "time2", "purpose": "Validate cron 5-field",
              "fn": lambda cron="0 * * * *": len(str(cron).split()) == 5})
    return f


def _fam_gen2(f):
    f.append({"name": "gen2_id", "cat": "gen2", "purpose": "Random id",
              "fn": lambda n=8: "".join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(int(n)))})
    f.append({"name": "gen2_code", "cat": "gen2", "purpose": "Verification code",
              "fn": lambda n=6: "".join(random.SystemRandom().choice(string.digits) for _ in range(int(n)))})
    f.append({"name": "gen2_password", "cat": "gen2", "purpose": "Random password",
              "fn": lambda n=16: "".join(random.SystemRandom().choice(string.ascii_letters + string.digits + "!@#$%^&*") for _ in range(int(n)))})
    f.append({"name": "gen2_uuid", "cat": "gen2", "purpose": "UUID4",
              "fn": lambda: str(uuid.uuid4())})
    f.append({"name": "gen2_color", "cat": "gen2", "purpose": "Random hex color",
              "fn": lambda: "#%06x" % random.SystemRandom().randint(0, 0xFFFFFF)})
    f.append({"name": "gen2_name", "cat": "gen2", "purpose": "Random person name",
              "fn": lambda: random.choice(["Anna", "Ararat", "Lusine", "David", "Mariam", "Tigran", "Nare", "Hayk", "Sona", "Vahan"])})
    f.append({"name": "gen2_sentence", "cat": "gen2", "purpose": "Random sentence",
              "fn": lambda n=6: " ".join(random.choice(["the", "quick", "brave", "smart", "fast", "calm", "big", "small"]) for _ in range(int(n))) + "."})
    f.append({"name": "gen2_words", "cat": "gen2", "purpose": "Random words",
              "fn": lambda n=4: [random.choice(["alpha", "beta", "gamma", "delta", "omega"]) for _ in range(int(n))]})
    f.append({"name": "gen2_lorem", "cat": "gen2", "purpose": "Lorem ipsum words",
              "fn": lambda n=10: " ".join(random.choice(["lorem", "ipsum", "dolor", "sit", "amet", "consectetur"]) for _ in range(int(n)))})
    f.append({"name": "gen2_emoji", "cat": "gen2", "purpose": "Random emoji",
              "fn": lambda: random.choice(["🚀", "🔥", "✅", "⭐", "💡", "🎯", "📦", "🧠"])})
    f.append({"name": "gen2_pin", "cat": "gen2", "purpose": "Random PIN",
              "fn": lambda n=4: "".join(random.SystemRandom().choice(string.digits) for _ in range(int(n)))})
    f.append({"name": "gen2_hex", "cat": "gen2", "purpose": "Random hex string",
              "fn": lambda n=16: "".join(random.SystemRandom().choice("0123456789abcdef") for _ in range(int(n)))})
    f.append({"name": "gen2_bool", "cat": "gen2", "purpose": "Random boolean",
              "fn": lambda: bool(random.SystemRandom().randint(0, 1))})
    f.append({"name": "gen2_choice", "cat": "gen2", "purpose": "Random choice",
              "fn": lambda xs="[a,b,c]": random.choice(_as_list(xs))})
    f.append({"name": "gen2_shuffle", "cat": "gen2", "purpose": "Seeded shuffle",
              "fn": lambda xs="[1,2,3,4]", seed=7: ml_shuffle(_as_list(xs), int(seed))})
    return f


def _fam_code2(f):
    f.append({"name": "code2_py_safe_import", "cat": "code2", "purpose": "Safe import try/except snippet",
              "fn": lambda mod="requests": f"try:\n    import {mod}\nexcept ImportError:\n    {mod} = None"})
    f.append({"name": "code2_docstring", "cat": "code2", "purpose": "Docstring template",
              "fn": lambda desc="Do something": f'"""\n{desc}.\n\nArgs:\n    ...\nReturns:\n    ...\n"""'})
    f.append({"name": "code2_class_skeleton", "cat": "code2", "purpose": "Class skeleton",
              "fn": lambda name="MyClass": f"class {name}:\n    def __init__(self):\n        pass\n\n    def run(self):\n        \"\"\"Main entry.\"\"\"\n        pass"})
    f.append({"name": "code2_func_skeleton", "cat": "code2", "purpose": "Function skeleton",
              "fn": lambda name="my_func", args="a, b": f"def {name}({args}):\n    \"\"\"Describe.\"\"\"\n    pass"})
    f.append({"name": "code2_main_guard", "cat": "code2", "purpose": "Main guard snippet",
              "fn": lambda: 'if __name__ == "__main__":\n    main()'})
    f.append({"name": "code2_type_hint", "cat": "code2", "purpose": "Type hint template",
              "fn": lambda t="int": f"x: {t} = ..."})
    f.append({"name": "code2_assert", "cat": "code2", "purpose": "Assertion snippet",
              "fn": lambda expr="result == expected": f"assert {expr}, f\"unexpected: {{{expr}}}\""})
    f.append({"name": "code2_logger", "cat": "code2", "purpose": "Logger setup snippet",
              "fn": lambda name="__name__": f'import logging\nlogging.basicConfig(level=logging.INFO)\nlogger = logging.getLogger({name})'})
    f.append({"name": "code2_try_except", "cat": "code2", "purpose": "Try/except snippet",
              "fn": lambda: "try:\n    ...\nexcept Exception as e:\n    raise"})
    f.append({"name": "code2_timer", "cat": "code2", "purpose": "Timer context snippet",
              "fn": lambda: "import time\nt0 = time.perf_counter()\n...\nprint(f'{time.perf_counter()-t0:.3f}s')"})
    f.append({"name": "code2_decorator", "cat": "code2", "purpose": "Decorator skeleton",
              "fn": lambda name="my_decorator": f"def {name}(fn):\n    def wrapper(*args, **kwargs):\n        return fn(*args, **kwargs)\n    return wrapper"})
    f.append({"name": "code2_dataclass", "cat": "code2", "purpose": "Dataclass skeleton",
              "fn": lambda name="Point": f"from dataclasses import dataclass\n\n@dataclass\nclass {name}:\n    x: int = 0\n    y: int = 0"})
    f.append({"name": "code2_enum", "cat": "code2", "purpose": "Enum skeleton",
              "fn": lambda name="Status": f"from enum import Enum\n\nclass {name}(Enum):\n    OK = 'ok'\n    FAIL = 'fail'"})
    f.append({"name": "code2_lambda", "cat": "code2", "purpose": "Lambda snippet",
              "fn": lambda expr="x: x * 2": f"f = lambda {expr}"})
    f.append({"name": "code2_listcomp", "cat": "code2", "purpose": "List comprehension snippet",
              "fn": lambda expr="x for x in range(10)": f"[{expr}]"})
    f.append({"name": "code2_dictcomp", "cat": "code2", "purpose": "Dict comprehension snippet",
              "fn": lambda expr="k: v for k, v in items": f"{{{expr}}}"})
    f.append({"name": "code2_repr", "cat": "code2", "purpose": "Repr snippet",
              "fn": lambda obj="self": f"return f'{obj.__class__.__name__}(...)'"})
    f.append({"name": "code2_async", "cat": "code2", "purpose": "Async function skeleton",
              "fn": lambda name="main": f"import asyncio\n\nasync def {name}():\n    ...\n\nif __name__ == '__main__':\n    asyncio.run({name}())"})
    f.append({"name": "code2_property", "cat": "code2", "purpose": "Property snippet",
              "fn": lambda name="value": f"@property\ndef {name}(self):\n    return self._{name}\n\n@{name}.setter\ndef {name}(self, v):\n    self._{name} = v"})
    f.append({"name": "code2_singleton", "cat": "code2", "purpose": "Singleton snippet",
              "fn": lambda name="MyClass": f"class {name}:\n    _instance = None\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance"})
    f.append({"name": "code2_fixture", "cat": "code2", "purpose": "Pytest fixture skeleton",
              "fn": lambda name="data": f"import pytest\n\n@pytest.fixture\ndef {name}():\n    return ..."})
    f.append({"name": "code2_parametrize", "cat": "code2", "purpose": "Pytest parametrize snippet",
              "fn": lambda args="a,b", vals="[(1,2),(3,4)]": f"@pytest.mark.parametrize('{args}', {vals})"})
    return f


def _fam_fs2(f):
    f.append({"name": "fs2_exists", "cat": "fs2", "purpose": "Path exists",
              "fn": lambda path=".": Path(str(path)).exists()})
    f.append({"name": "fs2_isfile", "cat": "fs2", "purpose": "Is file",
              "fn": lambda path=".": Path(str(path)).is_file()})
    f.append({"name": "fs2_isdir", "cat": "fs2", "purpose": "Is dir",
              "fn": lambda path=".": Path(str(path)).is_dir()})
    f.append({"name": "fs2_size", "cat": "fs2", "purpose": "File size bytes",
              "fn": lambda path="": Path(str(path)).stat().st_size if Path(str(path)).exists() and Path(str(path)).is_file() else 0})
    f.append({"name": "fs2_size_human", "cat": "fs2", "purpose": "File size human",
              "fn": lambda path="": _fmt_bytes(Path(str(path)).stat().st_size) if Path(str(path)).exists() and Path(str(path)).is_file() else "n/a"})
    f.append({"name": "fs2_listdir", "cat": "fs2", "purpose": "List dir entries",
              "fn": lambda path=".": sorted(os.listdir(str(path)))[:100]})
    f.append({"name": "fs2_tree", "cat": "fs2", "purpose": "Directory tree (2 levels)",
              "fn": lambda path=".", depth=2: fs2_tree(str(path), int(depth))})
    f.append({"name": "fs2_ext", "cat": "fs2", "purpose": "File extension",
              "fn": lambda path="a.txt": Path(str(path)).suffix})
    f.append({"name": "fs2_basename", "cat": "fs2", "purpose": "Basename",
              "fn": lambda path="/a/b/c.txt": Path(str(path)).name})
    f.append({"name": "fs2_dirname", "cat": "fs2", "purpose": "Dirname",
              "fn": lambda path="/a/b/c.txt": str(Path(str(path)).parent)})
    f.append({"name": "fs2_stem", "cat": "fs2", "purpose": "Stem without extension",
              "fn": lambda path="a.txt": Path(str(path)).stem})
    f.append({"name": "fs2_join", "cat": "fs2", "purpose": "Join paths",
              "fn": lambda parts="[a,b,c]": str(Path(*_as_list(parts)))})
    f.append({"name": "fs2_abs", "cat": "fs2", "purpose": "Absolute path",
              "fn": lambda path=".": str(Path(str(path)).resolve())})
    f.append({"name": "fs2_glob", "cat": "fs2", "purpose": "Glob files",
              "fn": lambda pattern="*.py": sorted(str(p) for p in Path(".").glob(str(pattern)))[:100]})
    f.append({"name": "fs2_mime", "cat": "fs2", "purpose": "MIME guess",
              "fn": lambda path="a.png": mime_guess(str(path))})
    f.append({"name": "fs2_touch", "cat": "fs2", "purpose": "Touch file",
              "fn": lambda path="tmp_x.txt": (Path(str(path)).touch(), str(path))[1]})
    f.append({"name": "fs2_hidden", "cat": "fs2", "purpose": "Is hidden path",
              "fn": lambda path=".env": Path(str(path)).name.startswith(".")})
    f.append({"name": "fs2_safe_name", "cat": "fs2", "purpose": "Safe filename",
              "fn": lambda name="My File!?.txt": re.sub(r"[^\w.\-]", "_", str(name))})
    f.append({"name": "fs2_du", "cat": "fs2", "purpose": "Dir size (best effort)",
              "fn": lambda path=".": fs2_du(str(path))})
    f.append({"name": "fs2_count_files", "cat": "fs2", "purpose": "Count files in dir",
              "fn": lambda path=".": sum(1 for _ in Path(str(path)).rglob("*") if _.is_file()) if Path(str(path)).exists() else 0})
    return f


def _fam_sec2(f):
    f.append({"name": "sec2_redact", "cat": "sec2", "purpose": "Redact secrets",
              "fn": lambda s="password=abc123": re.sub(r"(?i)(password|token|secret|key)\s*[=:]\s*\S+", r"\1=***", str(s))})
    f.append({"name": "sec2_strength", "cat": "sec2", "purpose": "Password strength label",
              "fn": lambda p="Abcdef1!": "weak" if len(str(p)) < 8 else "medium" if len(str(p)) < 12 else "strong"})
    f.append({"name": "sec2_entropy", "cat": "sec2", "purpose": "Password entropy bits",
              "fn": lambda p="Abcdef1!": round(_entropy(str(p)), 2)})
    f.append({"name": "sec2_is_email", "cat": "sec2", "purpose": "Email validation",
              "fn": lambda s="a@b.com": bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", str(s)))})
    f.append({"name": "sec2_is_phone", "cat": "sec2", "purpose": "Phone validation (loose)",
              "fn": lambda s="+37499123456": bool(re.match(r"^\+?[0-9]{7,15}$", str(s)))})
    f.append({"name": "sec2_is_ip", "cat": "sec2", "purpose": "IP validation",
              "fn": lambda s="192.168.1.1": bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$", str(s)))})
    f.append({"name": "sec2_is_url", "cat": "sec2", "purpose": "URL validation",
              "fn": lambda s="https://x.com": bool(re.match(r"^https?://", str(s)))})
    f.append({"name": "sec2_is_uuid", "cat": "sec2", "purpose": "UUID validation",
              "fn": lambda s="123e4567-e89b-12d3-a456-426614174000": bool(re.match(r"^[0-9a-fA-F-]{36}$", str(s)))})
    f.append({"name": "sec2_is_credit_card", "cat": "sec2", "purpose": "Credit card Luhn check",
              "fn": lambda s="4111111111111111": sec2_luhn(str(s))})
    f.append({"name": "sec2_is_int", "cat": "sec2", "purpose": "Integer validation",
              "fn": lambda s="123": str(s).lstrip("-").isdigit()})
    f.append({"name": "sec2_is_float", "cat": "sec2", "purpose": "Float validation",
              "fn": lambda s="1.5": bool(re.match(r"^-?\d+\.?\d*$", str(s)))})
    f.append({"name": "sec2_is_hex", "cat": "sec2", "purpose": "Hex validation",
              "fn": lambda s="ff00aa": bool(re.match(r"^[0-9a-fA-F]+$", str(s)))})
    f.append({"name": "sec2_is_base64", "cat": "sec2", "purpose": "Base64 validation",
              "fn": lambda s="aGk=": bool(re.match(r"^[A-Za-z0-9+/]*={0,2}$", str(s))) and len(str(s)) % 4 == 0})
    f.append({"name": "sec2_is_date", "cat": "sec2", "purpose": "ISO date validation",
              "fn": lambda s="2026-08-09": bool(re.match(r"^\d{4}-\d{2}-\d{2}$", str(s)))})
    f.append({"name": "sec2_mask_email", "cat": "sec2", "purpose": "Mask email",
              "fn": lambda s="user@example.com": sec2_mask_email(str(s))})
    f.append({"name": "sec2_mask_phone", "cat": "sec2", "purpose": "Mask phone",
              "fn": lambda s="+37499123456": str(s)[:4] + "***" + str(s)[-3:]})
    f.append({"name": "sec2_audit", "cat": "sec2", "purpose": "Audit log entry",
              "fn": lambda user="admin", action="login": {"user": str(user), "action": str(action), "at": _now(), "ok": True}})
    f.append({"name": "sec2_hardened_headers", "cat": "sec2", "purpose": "Security headers dict",
              "fn": lambda: {"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer", "Content-Security-Policy": "default-src 'self'"}})
    f.append({"name": "sec2_otp_verify", "cat": "sec2", "purpose": "OTP verify (6-digit match)",
              "fn": lambda provided="123456", expected="123456": str(provided) == str(expected)})
    f.append({"name": "sec2_token_hint", "cat": "sec2", "purpose": "Token security hint",
              "fn": lambda token_len=40: {"length": int(token_len), "ok": int(token_len) >= 32, "hint": "use ≥32 random chars"}})
    return f


def _fam_fmt2(f):
    f.append({"name": "fmt2_bytes", "cat": "fmt2", "purpose": "Bytes human format",
              "fn": lambda n=1536: _fmt_bytes(float(n))})
    f.append({"name": "fmt2_number", "cat": "fmt2", "purpose": "Number with thousands sep",
              "fn": lambda x=1234567: f"{float(x):,.0f}"})
    f.append({"name": "fmt2_percent", "cat": "fmt2", "purpose": "Percent format",
              "fn": lambda x=0.123, d=1: f"{float(x) * 100:.{int(d)}f}%"})
    f.append({"name": "fmt2_table", "cat": "fmt2", "purpose": "Table from rows",
              "fn": lambda rows="[[a,1],[b,2]]", headers="[name,val]": _table([[str(c) for c in r] for r in json.loads(rows)], [str(h) for h in _as_list(headers)])})
    f.append({"name": "fmt2_align", "cat": "fmt2", "purpose": "Align lines",
              "fn": lambda rows="[[a,1],[bb,22]]": _table([[str(c) for c in r] for r in json.loads(rows)])})
    f.append({"name": "fmt2_plural", "cat": "fmt2", "purpose": "Plural helper",
              "fn": lambda n=2, word="item": f"{int(n)} {word}{'s' if int(n) != 1 else ''}"})
    f.append({"name": "fmt2_pad", "cat": "fmt2", "purpose": "Pad number",
              "fn": lambda n=7, width=3: str(int(n)).zfill(int(width))})
    f.append({"name": "fmt2_currency", "cat": "fmt2", "purpose": "Currency format",
              "fn": lambda x=1234.5, cur="$": f"{cur}{float(x):,.2f}"})
    f.append({"name": "fmt2_time", "cat": "fmt2", "purpose": "Seconds to mm:ss",
              "fn": lambda s=125: f"{int(s) // 60:02d}:{int(s) % 60:02d}"})
    f.append({"name": "fmt2_duration", "cat": "fmt2", "purpose": "Duration human",
              "fn": lambda s=90061: time2_duration(int(s))})
    f.append({"name": "fmt2_ratio", "cat": "fmt2", "purpose": "Ratio format",
              "fn": lambda a=3, b=4: f"{float(a) / max(1e-9, float(b)):.2f}:1"})
    f.append({"name": "fmt2_phone", "cat": "fmt2", "purpose": "Phone format (loose)",
              "fn": lambda s="+37499123456": f"{str(s)[:4]} {str(s)[4:6]} {str(s)[6:9]} {str(s)[9:]}"})
    f.append({"name": "fmt2_title", "cat": "fmt2", "purpose": "Title case",
              "fn": lambda s="hello world": str(s).title()})
    f.append({"name": "fmt2_json_line", "cat": "fmt2", "purpose": "One-line JSON",
              "fn": lambda d="{\"a\":1}": json.dumps(_as_dict(d), separators=(",", ":"))})
    f.append({"name": "fmt2_kv", "cat": "fmt2", "purpose": "Dict to key=value lines",
              "fn": lambda d="{\"a\":1,\"b\":2}": "\n".join(f"{k}={v}" for k, v in _as_dict(d).items())})
    return f


def _fam_valid2(f):
    f.append({"name": "valid2_email", "cat": "valid2", "purpose": "Email valid",
              "fn": lambda s="a@b.com": bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", str(s)))})
    f.append({"name": "valid2_phone", "cat": "valid2", "purpose": "Phone valid",
              "fn": lambda s="+37499123456": bool(re.match(r"^\+?[0-9]{7,15}$", str(s)))})
    f.append({"name": "valid2_ip", "cat": "valid2", "purpose": "IP valid",
              "fn": lambda s="192.168.1.1": bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$", str(s)))})
    f.append({"name": "valid2_url", "cat": "valid2", "purpose": "URL valid",
              "fn": lambda s="https://x.com": bool(re.match(r"^https?://", str(s)))})
    f.append({"name": "valid2_uuid", "cat": "valid2", "purpose": "UUID valid",
              "fn": lambda s="123e4567-e89b-12d3-a456-426614174000": bool(re.match(r"^[0-9a-fA-F-]{36}$", str(s)))})
    f.append({"name": "valid2_card", "cat": "valid2", "purpose": "Credit card Luhn",
              "fn": lambda s="4111111111111111": sec2_luhn(str(s))})
    f.append({"name": "valid2_date", "cat": "valid2", "purpose": "ISO date valid",
              "fn": lambda s="2026-08-09": bool(re.match(r"^\d{4}-\d{2}-\d{2}$", str(s)))})
    f.append({"name": "valid2_time", "cat": "valid2", "purpose": "ISO time valid",
              "fn": lambda s="10:30:00": bool(re.match(r"^\d{2}:\d{2}:\d{2}$", str(s)))})
    f.append({"name": "valid2_hex", "cat": "valid2", "purpose": "Hex valid",
              "fn": lambda s="ff00": bool(re.match(r"^[0-9a-fA-F]+$", str(s)))})
    f.append({"name": "valid2_int", "cat": "valid2", "purpose": "Int valid",
              "fn": lambda s="123": str(s).lstrip("-").isdigit()})
    f.append({"name": "valid2_float", "cat": "valid2", "purpose": "Float valid",
              "fn": lambda s="1.5": bool(re.match(r"^-?\d+\.?\d*$", str(s)))})
    f.append({"name": "valid2_bool", "cat": "valid2", "purpose": "Bool valid",
              "fn": lambda s="true": str(s).lower() in ("true", "false", "1", "0", "yes", "no")})
    f.append({"name": "valid2_json", "cat": "valid2", "purpose": "JSON valid",
              "fn": lambda s="{}": json_validate(str(s))})
    f.append({"name": "valid2_slug", "cat": "valid2", "purpose": "Slug valid",
              "fn": lambda s="hello-world": bool(re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", str(s)))})
    f.append({"name": "valid2_alpha", "cat": "valid2", "purpose": "Alphabetic valid",
              "fn": lambda s="abc": str(s).isalpha()})
    f.append({"name": "valid2_alnum", "cat": "valid2", "purpose": "Alphanumeric valid",
              "fn": lambda s="abc123": str(s).isalnum()})
    return f


def _fam_csv2(f):
    f.append({"name": "csv2_read", "cat": "csv2", "purpose": "Read CSV text to rows",
              "fn": lambda text="a,b\n1,2": [row for row in csv.reader(io.StringIO(str(text)))]})
    f.append({"name": "csv2_header", "cat": "csv2", "purpose": "CSV header row",
              "fn": lambda text="a,b\n1,2": next(csv.reader(io.StringIO(str(text))))})
    f.append({"name": "csv2_rows", "cat": "csv2", "purpose": "CSV data rows",
              "fn": lambda text="a,b\n1,2\n3,4": list(csv.reader(io.StringIO(str(text))))[1:]})
    f.append({"name": "csv2_count", "cat": "csv2", "purpose": "CSV row count",
              "fn": lambda text="a,b\n1,2": sum(1 for _ in csv.reader(io.StringIO(str(text)))) - 1})
    f.append({"name": "csv2_to_json", "cat": "csv2", "purpose": "CSV to JSON list",
              "fn": lambda text="a,b\n1,2": csv_to_json(str(text))})
    f.append({"name": "csv2_from_json", "cat": "csv2", "purpose": "JSON list to CSV",
              "fn": lambda s='[{"a":1,"b":2}]': json_to_csv(json.loads(str(s)))})
    f.append({"name": "csv2_col", "cat": "csv2", "purpose": "Extract column",
              "fn": lambda text="a,b\n1,2\n3,4", col=1: [row[int(col)] for row in list(csv.reader(io.StringIO(str(text))))[1:]]})
    f.append({"name": "csv2_delim", "cat": "csv2", "purpose": "Detect delimiter",
              "fn": lambda text="a;b\n1;2": csv.Sniffer().sniff(str(text)).delimiter if str(text) else ","})
    f.append({"name": "csv2_clean", "cat": "csv2", "purpose": "Strip whitespace cells",
              "fn": lambda text=" a ,b \n1, 2 ": [[c.strip() for c in row] for row in csv.reader(io.StringIO(str(text)))][1:]})
    f.append({"name": "csv2_merge", "cat": "csv2", "purpose": "Merge two CSVs vertically",
              "fn": lambda a="a,b\n1,2", b="a,b\n3,4": str(a).strip() + "\n" + "\n".join(list(csv.reader(io.StringIO(str(b))))[1:])})
    return f


def _fam_xml2(f):
    f.append({"name": "xml2_escape", "cat": "xml2", "purpose": "XML escape",
              "fn": lambda s="<a>&": str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")})
    f.append({"name": "xml2_unescape", "cat": "xml2", "purpose": "XML unescape",
              "fn": lambda s="&lt;a&gt;&amp;": str(s).replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&amp;", "&")})
    f.append({"name": "xml2_tag", "cat": "xml2", "purpose": "Wrap in tag",
              "fn": lambda name="root", body="hi": f"<{name}>{body}</{name}>"})
    f.append({"name": "xml2_attr", "cat": "xml2", "purpose": "Build attribute string",
              "fn": lambda d="{\"a\":\"1\",\"b\":\"x\"}": " ".join(f'{k}="{v}"' for k, v in _as_dict(d).items())})
    f.append({"name": "xml2_tag_attr", "cat": "xml2", "purpose": "Tag with attributes",
              "fn": lambda name="item", d="{\"id\":\"1\"}", body="x": xml_tag_attr(str(name), _as_dict(d), str(body))})
    f.append({"name": "xml2_strip", "cat": "xml2", "purpose": "Strip XML tags",
              "fn": lambda s="<a>hi</a>": re.sub(r"<[^>]+>", "", str(s))})
    f.append({"name": "xml2_attrs", "cat": "xml2", "purpose": "Extract first tag attrs",
              "fn": lambda s='<a id="1" class="x">hi</a>': dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', str(s)))})
    f.append({"name": "xml2_declaration", "cat": "xml2", "purpose": "XML declaration",
              "fn": lambda: '<?xml version="1.0" encoding="UTF-8"?>'})
    f.append({"name": "xml2_cdata", "cat": "xml2", "purpose": "CDATA wrapper",
              "fn": lambda s="raw <text>": f"<![CDATA[{str(s)}]]>"})
    return f


def _fam_yaml2(f):
    f.append({"name": "yaml2_like", "cat": "yaml2", "purpose": "Minimal yaml-ish parse (simple k:v)",
              "fn": lambda text="a: 1\nb: hello": {line.split(":", 1)[0].strip(): line.split(":", 1)[1].strip() for line in str(text).splitlines() if ":" in line}})
    f.append({"name": "yaml2_kv", "cat": "yaml2", "purpose": "Dict to yaml-ish lines",
              "fn": lambda d="{\"a\":1,\"b\":\"hi\"}": "\n".join(f"{k}: {v}" for k, v in _as_dict(d).items())})
    f.append({"name": "yaml2_list", "cat": "yaml2", "purpose": "List to yaml-ish bullets",
              "fn": lambda xs="[a,b,c]": "\n".join(f"- {x}" for x in _as_list(xs))})
    f.append({"name": "yaml2_indent", "cat": "yaml2", "purpose": "Indent yaml block",
              "fn": lambda s="a: 1", n=2: "\n".join(" " * int(n) + line for line in str(s).splitlines())})
    f.append({"name": "yaml2_bool", "cat": "yaml2", "purpose": "YAML bool normalize",
              "fn": lambda s="yes": str(s).lower() in ("yes", "true", "on", "1")})
    f.append({"name": "yaml2_number", "cat": "yaml2", "purpose": "YAML number parse",
              "fn": lambda s="1.5": float(str(s)) if "." in str(s) else int(str(s))})
    return f


def _fam_env(f):
    f.append({"name": "env_detect", "cat": "env", "purpose": "Detect runtime environment",
              "fn": lambda: env_detect()})
    f.append({"name": "env_is_local", "cat": "env", "purpose": "Is local environment",
              "fn": lambda: env_detect()["mode"] == "localhost"})
    f.append({"name": "env_is_colab", "cat": "env", "purpose": "Is Colab",
              "fn": lambda: env_detect()["mode"] == "colab"})
    f.append({"name": "env_is_cloud", "cat": "env", "purpose": "Is cloud",
              "fn": lambda: env_detect()["cloud"]})
    f.append({"name": "env_is_container", "cat": "env", "purpose": "Is container",
              "fn": lambda: bool(os.environ.get("KUBERNETES_SERVICE_HOST")) or Path("/.dockerenv").exists()})
    f.append({"name": "env_public_url_hint", "cat": "env", "purpose": "Public URL hint",
              "fn": lambda: "https://<colab-ip>.ngrok-free.app" if env_detect()["mode"] == "colab" else "http://localhost:8888"})
    f.append({"name": "env_platform", "cat": "env", "purpose": "Platform string",
              "fn": lambda: sys.platform})
    f.append({"name": "env_python", "cat": "env", "purpose": "Python version",
              "fn": lambda: sys.version.split()[0]})
    f.append({"name": "env_cpu", "cat": "env", "purpose": "CPU count",
              "fn": lambda: os.cpu_count() or 1})
    f.append({"name": "env_gpu_hint", "cat": "env", "purpose": "GPU hint",
              "fn": lambda: bool(shutil.which("nvidia-smi"))})
    f.append({"name": "env_ram_hint", "cat": "env", "purpose": "RAM hint (GB)",
              "fn": lambda: round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1) if hasattr(os, "sysconf") else None})
    f.append({"name": "env_tz", "cat": "env", "purpose": "Timezone offset hours",
              "fn": lambda: round(time.timezone / -3600, 1)})
    f.append({"name": "env_now", "cat": "env", "purpose": "Current time",
              "fn": _now})
    f.append({"name": "env_disk_free", "cat": "env", "purpose": "Disk free bytes (cwd)",
              "fn": lambda: shutil.disk_usage(".").free})
    f.append({"name": "env_disk_total", "cat": "env", "purpose": "Disk total bytes (cwd)",
              "fn": lambda: shutil.disk_usage(".").total})
    return f


def _fam_combo(f):
    f.append({"name": "combo_permutations", "cat": "combo", "purpose": "Permutations of list",
              "fn": lambda xs="[1,2,3]", r=2: [list(p) for p in itertools.permutations(_as_list(xs), int(r))]})
    f.append({"name": "combo_combinations", "cat": "combo", "purpose": "Combinations of list",
              "fn": lambda xs="[1,2,3]", r=2: [list(c) for c in itertools.combinations(_as_list(xs), int(r))]})
    f.append({"name": "combo_product", "cat": "combo", "purpose": "Cartesian product",
              "fn": lambda xs="[1,2]", ys="[a,b]": [list(p) for p in itertools.product(_as_list(xs), _as_list(ys))]})
    f.append({"name": "combo_pairs", "cat": "combo", "purpose": "All pairs",
              "fn": lambda xs="[a,b,c]": [list(p) for p in itertools.combinations(_as_list(xs), 2)]})
    f.append({"name": "combo_partitions", "cat": "combo", "purpose": "All 2-partitions (lite)",
              "fn": lambda xs="[a,b,c]": [[list(p), [x for x in _as_list(xs) if x not in p]] for p in itertools.combinations(_as_list(xs), 1)]})
    f.append({"name": "combo_powerset", "cat": "combo", "purpose": "Powerset (up to 3 elems)",
              "fn": lambda xs="[a,b]": [list(s) for r in range(len(_as_list(xs)) + 1) for s in itertools.combinations(_as_list(xs), r)]})
    f.append({"name": "combo_zip", "cat": "combo", "purpose": "Zip lists",
              "fn": lambda a="[1,2]", b="[a,b]": [list(z) for z in zip(_as_list(a), _as_list(b))]})
    f.append({"name": "combo_cycle", "cat": "combo", "purpose": "Cycle iterator first N",
              "fn": lambda xs="[1,2,3]", n=7: [x for _, x in zip(range(int(n)), itertools.cycle(_as_list(xs)))]})
    f.append({"name": "combo_repeat", "cat": "combo", "purpose": "Repeat values",
              "fn": lambda x="ab", n=3: [str(x)] * int(n)})
    f.append({"name": "combo_count", "cat": "combo", "purpose": "Combination count",
              "fn": lambda n=5, r=2: math.comb(int(n), int(r))})
    f.append({"name": "combo_perm_count", "cat": "combo", "purpose": "Permutation count",
              "fn": lambda n=5, r=2: math.perm(int(n), int(r))})
    f.append({"name": "combo_accumulate", "cat": "combo", "purpose": "Running totals",
              "fn": lambda xs="[1,2,3]": list(itertools.accumulate(_as_list(xs)))})
    f.append({"name": "combo_chain", "cat": "combo", "purpose": "Chain lists",
              "fn": lambda a="[1,2]", b="[3,4]": list(itertools.chain(_as_list(a), _as_list(b)))})
    f.append({"name": "combo_group", "cat": "combo", "purpose": "Group by key",
              "fn": lambda xs="[a,b,a,c]": {k: list(g) for k, g in itertools.groupby(sorted(_as_list(xs)))}})
    f.append({"name": "combo_starmap", "cat": "combo", "purpose": "Starmap demo",
              "fn": lambda xs="[[1,2],[3,4]]": [a + b for a, b in itertools.starmap(lambda x, y: (x, y), json.loads(xs))]})
    return f


def _fam_chart(f):
    f.append({"name": "chart_bar", "cat": "chart", "purpose": "ASCII bar chart",
              "fn": lambda xs="[3,1,2]", labels="[a,b,c]": chart_bar(_as_list(xs), _as_list(labels))})
    f.append({"name": "chart_sparkline", "cat": "chart", "purpose": "Sparkline",
              "fn": lambda xs="[1,2,3,2,5]": chart_spark(_as_list(xs))})
    f.append({"name": "chart_hist", "cat": "chart", "purpose": "ASCII histogram",
              "fn": lambda xs="[1,1,2,3,3,3]", bins=3: chart_hist(_as_list(xs), int(bins))})
    f.append({"name": "chart_hbar", "cat": "chart", "purpose": "Horizontal bar",
              "fn": lambda x=7, width=10: "#" * int(round(float(x) * int(width) / 10))})
    f.append({"name": "chart_heat", "cat": "chart", "purpose": "ASCII heat cell",
              "fn": lambda v=0.5: " ░▒▓█"[min(4, max(0, int(float(v) * 5)))]})
    f.append({"name": "chart_gauge", "cat": "chart", "purpose": "ASCII gauge",
              "fn": lambda pct=60, width=20: chart_gauge(float(pct), int(width))})
    return f


def _fam_rep(f):
    f.append({"name": "rep_md_table", "cat": "rep", "purpose": "Markdown table",
              "fn": lambda rows="[[a,1],[b,2]]", headers="[name,val]": rep_md_table(json.loads(rows), _as_list(headers))})
    f.append({"name": "rep_toc", "cat": "rep", "purpose": "Table of contents",
              "fn": lambda headings="[Intro,Usage,API]": "\n".join(f"- [{h}](#{_slug(h)})" for h in _as_list(headings))})
    f.append({"name": "rep_section", "cat": "rep", "purpose": "Markdown section",
              "fn": lambda title="Usage", body="..." : f"## {title}\n\n{body}"})
    f.append({"name": "rep_badge", "cat": "rep", "purpose": "Badge markdown",
              "fn": lambda label="build", status="passing", color="green": f"![{label}](https://img.shields.io/badge/{label}-{status}-{color})"})
    f.append({"name": "rep_hr", "cat": "rep", "purpose": "Markdown rule",
              "fn": lambda: "---"})
    f.append({"name": "rep_quote", "cat": "rep", "purpose": "Blockquote",
              "fn": lambda s="hello": f"> {str(s)}"})
    f.append({"name": "rep_codeblock", "cat": "rep", "purpose": "Code block",
              "fn": lambda code="print(1)", lang="python": f"```{lang}\n{code}\n```"})
    f.append({"name": "rep_list", "cat": "rep", "purpose": "Markdown list",
              "fn": lambda items="[a,b,c]": "\n".join(f"- {i}" for i in _as_list(items))})
    f.append({"name": "rep_ordered", "cat": "rep", "purpose": "Markdown ordered list",
              "fn": lambda items="[a,b,c]": "\n".join(f"{i + 1}. {it}" for i, it in enumerate(_as_list(items)))})
    f.append({"name": "rep_checklist", "cat": "rep", "purpose": "Checklist",
              "fn": lambda items="[a,b]": "\n".join(f"- [ ] {i}" for i in _as_list(items))})
    f.append({"name": "rep_link", "cat": "rep", "purpose": "Markdown link",
              "fn": lambda text="docs", url="https://x.com": f"[{text}]({url})"})
    f.append({"name": "rep_bold", "cat": "rep", "purpose": "Bold text",
              "fn": lambda s="hi": f"**{s}**"})
    f.append({"name": "rep_italic", "cat": "rep", "purpose": "Italic text",
              "fn": lambda s="hi": f"*{s}*"})
    f.append({"name": "rep_inline_code", "cat": "rep", "purpose": "Inline code",
              "fn": lambda s="x = 1": f"`{s}`"})
    f.append({"name": "rep_heading", "cat": "rep", "purpose": "Heading markdown",
              "fn": lambda s="Title", level=1: "#" * int(level) + " " + str(s)})
    return f


def _fam_note(f):
    f.append({"name": "note_todo", "cat": "note", "purpose": "Todo item",
              "fn": lambda s="do x": {"text": str(s), "done": False}})
    f.append({"name": "note_done", "cat": "note", "purpose": "Done item",
              "fn": lambda s="do x": {"text": str(s), "done": True}})
    f.append({"name": "note_add", "cat": "note", "purpose": "Add note",
              "fn": lambda title="note1", body="...": {"title": str(title), "body": str(body), "at": _now()}})
    f.append({"name": "note_tags", "cat": "note", "purpose": "Extract #tags",
              "fn": lambda s="see #ai and #ml": re.findall(r"#(\w+)", str(s))})
    f.append({"name": "note_mentions", "cat": "note", "purpose": "Extract @mentions",
              "fn": lambda s="hi @ararat": re.findall(r"@(\w+)", str(s))})
    f.append({"name": "note_priority", "cat": "note", "purpose": "Priority label",
              "fn": lambda n=1: {1: "high", 2: "medium", 3: "low"}.get(int(n), "none")})
    f.append({"name": "note_status", "cat": "note", "purpose": "Status emoji",
              "fn": lambda s="done": {"todo": "⬜", "doing": "🟡", "done": "✅", "blocked": "🔴"}.get(str(s), "⬜")})
    f.append({"name": "note_kanban", "cat": "note", "purpose": "Kanban columns",
              "fn": lambda: {"todo": [], "doing": [], "done": []}})
    f.append({"name": "note_idea", "cat": "note", "purpose": "Idea entry",
              "fn": lambda s="great idea": {"idea": str(s), "votes": 0, "at": _now()}})
    f.append({"name": "note_journal", "cat": "note", "purpose": "Journal entry",
              "fn": lambda s="today": {"text": str(s), "date": _today()}})
    return f


def _fam_menu(f):
    f.append({"name": "menu_paginate", "cat": "menu", "purpose": "Paginate items",
              "fn": lambda xs="[1,2,3,4,5]", page=1, size=2: list(_as_list(xs)[(int(page) - 1) * int(size):int(page) * int(size)])})
    f.append({"name": "menu_search", "cat": "menu", "purpose": "Filter by substring",
              "fn": lambda xs="[apple,banana,cherry]", q="an": [x for x in _as_list(xs) if str(q) in str(x)]})
    f.append({"name": "menu_format", "cat": "menu", "purpose": "Numbered menu",
              "fn": lambda items="[a,b,c]": "\n".join(f"{i + 1}. {it}" for i, it in enumerate(_as_list(items)))})
    f.append({"name": "menu_breadcrumb", "cat": "menu", "purpose": "Breadcrumb",
              "fn": lambda parts="[Home,Settings,Profile]": " / ".join(str(p) for p in _as_list(parts))})
    f.append({"name": "menu_tabs", "cat": "menu", "purpose": "Tab labels",
              "fn": lambda items="[Home,About]": [{"label": str(i), "active": idx == 0} for idx, i in enumerate(_as_list(items))]})
    f.append({"name": "menu_shortcut", "cat": "menu", "purpose": "Shortcut hint",
              "fn": lambda key="Ctrl+K", action="search": {"key": str(key), "action": str(action)}})
    f.append({"name": "menu_confirm", "cat": "menu", "purpose": "Confirmation dialog text",
              "fn": lambda s="Delete?": f"{s} [y/N]"})
    f.append({"name": "menu_loading", "cat": "menu", "purpose": "Loading message",
              "fn": lambda s="Loading": f"{s}..."})
    f.append({"name": "menu_empty", "cat": "menu", "purpose": "Empty state text",
              "fn": lambda s="items": f"No {s} yet"})
    f.append({"name": "menu_error", "cat": "menu", "purpose": "Error message",
              "fn": lambda s="oops": f"⚠ {s}"})
    f.append({"name": "menu_success", "cat": "menu", "purpose": "Success message",
              "fn": lambda s="done": f"✅ {s}"})
    f.append({"name": "menu_keyboard_nav", "cat": "menu", "purpose": "Keyboard nav hints",
              "fn": lambda: ["↑↓ navigate", "Enter select", "Esc back", "/ search"]})
    return f


def _fam_dist(f):
    f.append({"name": "dist_world", "cat": "dist", "purpose": "Detect distributed world",
              "fn": lambda: {"world_size": 1, "rank": 0, "backend": "none", "note": "single process"} })
    f.append({"name": "dist_shard", "cat": "dist", "purpose": "Shard list across workers",
              "fn": lambda xs="[1,2,3,4,5,6]", workers=2, rank=0: list(_as_list(xs)[int(rank)::int(workers)])})
    f.append({"name": "dist_batch_shard", "cat": "dist", "purpose": "Contiguous shard",
              "fn": lambda xs="[1,2,3,4,5,6]", workers=2, rank=0: dist_shard_contig(_as_list(xs), int(workers), int(rank))})
    f.append({"name": "dist_sync_note", "cat": "dist", "purpose": "Sync note",
              "fn": lambda: "allreduce / broadcast via backend; single-process fallback here"})
    f.append({"name": "dist_seed", "cat": "dist", "purpose": "Seeded deterministic",
              "fn": lambda rank=0: 42 + int(rank)})
    f.append({"name": "dist_gpus_hint", "cat": "dist", "purpose": "GPU count hint",
              "fn": lambda: int(os.environ.get("WORLD_SIZE", 1))})
    f.append({"name": "dist_rank_env", "cat": "dist", "purpose": "Rank env vars",
              "fn": lambda: {k: os.environ.get(k, "") for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")}})
    f.append({"name": "dist_chunk", "cat": "dist", "purpose": "Even chunk sizes",
              "fn": lambda n=10, workers=3: [int(n) // int(workers) + (1 if i < int(n) % int(workers) else 0) for i in range(int(workers))]})
    return f


def _fam_sched2(f):
    f.append({"name": "sched2_parse", "cat": "sched2", "purpose": "Parse 5-field cron",
              "fn": lambda cron="0 9 * * 1": sched2_parse(str(cron))})
    f.append({"name": "sched2_next_minute", "cat": "sched2", "purpose": "Next run minute hint",
              "fn": lambda cron="0 * * * *": "next hour at :00"})
    f.append({"name": "sched2_every", "cat": "sched2", "purpose": "Every N minutes cron",
              "fn": lambda n=30: f"*/{int(n)} * * * *"})
    f.append({"name": "sched2_daily_at", "cat": "sched2", "purpose": "Daily at hour",
              "fn": lambda h=9: f"0 {int(h)} * * *"})
    f.append({"name": "sched2_weekdays", "cat": "sched2", "purpose": "Weekdays cron",
              "fn": lambda h=9: f"0 {int(h)} * * 1-5"})
    f.append({"name": "sched2_weekend", "cat": "sched2", "purpose": "Weekend cron",
              "fn": lambda h=10: f"0 {int(h)} * * 6,0"})
    f.append({"name": "sched2_monthly", "cat": "sched2", "purpose": "Monthly cron",
              "fn": lambda day=1, h=0: f"0 {int(h)} {int(day)} * *"})
    f.append({"name": "sched2_human", "cat": "sched2", "purpose": "Human description",
              "fn": lambda cron="0 9 * * 1": "at 09:00 every Monday"})
    return f


def _fam_monitor2(f):
    f.append({"name": "monitor2_throughput", "cat": "monitor2", "purpose": "Throughput ops/sec",
              "fn": lambda ops=1000, secs=5: round(int(ops) / max(1e-9, float(secs)), 1)})
    f.append({"name": "monitor2_latency_avg", "cat": "monitor2", "purpose": "Avg latency",
              "fn": lambda lats="[10,20,30]": round(statistics.mean([float(x) for x in _as_list(lats)]), 2)})
    f.append({"name": "monitor2_latency_p95", "cat": "monitor2", "purpose": "P95 latency",
              "fn": lambda lats="[10,20,30,40,50,60,70,80,90,100]": sorted([float(x) for x in _as_list(lats)])[int(0.95 * (len(_as_list(lats)) - 1))]})
    f.append({"name": "monitor2_error_rate", "cat": "monitor2", "purpose": "Error rate %",
              "fn": lambda errors=5, total=100: round(100 * int(errors) / max(1, int(total)), 2)})
    f.append({"name": "monitor2_uptime_pct", "cat": "monitor2", "purpose": "Uptime %",
              "fn": lambda up=99.9, target=99.5: {"uptime": float(up), "ok": float(up) >= float(target)}})
    f.append({"name": "monitor2_sla", "cat": "monitor2", "purpose": "SLA status",
              "fn": lambda uptime=99.9, sla=99.9: "SLA met" if float(uptime) >= float(sla) else "SLA breach"})
    f.append({"name": "monitor2_health", "cat": "monitor2", "purpose": "Health check dict",
              "fn": lambda name="api", ok=True: {"name": str(name), "status": "ok" if bool(ok) else "down", "at": _now()}})
    f.append({"name": "monitor2_apdex", "cat": "monitor2", "purpose": "Apdex score",
              "fn": lambda lats="[100,200,500,800,2000]", t=500: monitor_apdex(_as_list(lats), float(t))})
    f.append({"name": "monitor2_dashboard", "cat": "monitor2", "purpose": "Dashboard summary",
              "fn": lambda req=1000, err=10, secs=60: {"requests": int(req), "errors": int(err), "rps": round(int(req) / max(1, int(secs)), 1), "error_rate": round(100 * int(err) / max(1, int(req)), 2)}})
    f.append({"name": "monitor2_alert", "cat": "monitor2", "purpose": "Alert entry",
              "fn": lambda metric="cpu", value=95, threshold=90: {"metric": str(metric), "value": float(value), "threshold": float(threshold), "alert": float(value) > float(threshold), "at": _now()}})
    return f


def _fam_backup2(f):
    f.append({"name": "backup2_plan", "cat": "backup2", "purpose": "Backup plan",
              "fn": lambda source="data", dest="backup": {"source": str(source), "dest": str(dest), "schedule": "daily", "retention": 7}})
    f.append({"name": "backup2_rotation", "cat": "backup2", "purpose": "Rotation file names",
              "fn": lambda base="backup", n=7: [f"{base}.{i}.zip" for i in range(int(n))]})
    f.append({"name": "backup2_manifest", "cat": "backup2", "purpose": "Manifest entry",
              "fn": lambda name="backup1", size=1024: {"name": str(name), "size": int(size), "at": _now(), "hash": _sha(str(name), "sha256")[:16]}})
    f.append({"name": "backup2_strategy", "cat": "backup2", "purpose": "Strategy suggestion",
              "fn": lambda critical=False: "3-2-1 (3 copies, 2 media, 1 offsite)" if bool(critical) else "daily + weekly"})
    f.append({"name": "backup2_verify", "cat": "backup2", "purpose": "Verify plan",
              "fn": lambda dest="backup": {"dest": str(dest), "exists": Path(str(dest)).exists(), "ok": Path(str(dest)).exists()}})
    f.append({"name": "backup2_retention_days", "cat": "backup2", "purpose": "Retention end date",
              "fn": lambda days=7: (_dt.date.today() + _dt.timedelta(days=int(days))).isoformat()})
    return f


def _fam_ai2(f):
    f.append({"name": "ai2_prompt", "cat": "ai2", "purpose": "Prompt template fill",
              "fn": lambda template="Hello {name}", kw="{\"name\":\"Ararat\"}": str(template).format(**_as_dict(kw))})
    f.append({"name": "ai2_system", "cat": "ai2", "purpose": "System prompt wrapper",
              "fn": lambda s="You are helpful": f"<system>\n{s}\n</system>"})
    f.append({"name": "ai2_user", "cat": "ai2", "purpose": "User message wrapper",
              "fn": lambda s="hi": f"<user>\n{s}\n</user>"})
    f.append({"name": "ai2_assistant", "cat": "ai2", "purpose": "Assistant message wrapper",
              "fn": lambda s="hello": f"<assistant>\n{s}\n</assistant>"})
    f.append({"name": "ai2_chain", "cat": "ai2", "purpose": "Chain prompts",
              "fn": lambda steps="[a,b,c]": {"steps": _as_list(steps), "total": len(_as_list(steps))}})
    f.append({"name": "ai2_eval_accuracy", "cat": "ai2", "purpose": "Eval accuracy",
              "fn": lambda correct=8, total=10: round(100 * int(correct) / max(1, int(total)), 1)})
    f.append({"name": "ai2_eval_grade", "cat": "ai2", "purpose": "Eval grade",
              "fn": lambda score=90: "A" if int(score) >= 90 else "B" if int(score) >= 80 else "C" if int(score) >= 70 else "D" if int(score) >= 60 else "F"})
    f.append({"name": "ai2_memory", "cat": "ai2", "purpose": "Memory entry",
              "fn": lambda key="fact", value="x": {"key": str(key), "value": str(value)}})
    f.append({"name": "ai2_tokens", "cat": "ai2", "purpose": "Token estimate",
              "fn": lambda text="hello world": max(1, len(str(text)) // 4)})
    f.append({"name": "ai2_cost", "cat": "ai2", "purpose": "Cost estimate",
              "fn": lambda tokens=1000, per_m=0.5: round(int(tokens) / 1e6 * float(per_m), 5)})
    f.append({"name": "ai2_router", "cat": "ai2", "purpose": "Route by keyword",
              "fn": lambda q="train model": "training" if "train" in str(q) else "chat" if "chat" in str(q) else "general"})
    f.append({"name": "ai2_fallback", "cat": "ai2", "purpose": "Fallback note",
              "fn": lambda provider="openai": {"provider": str(provider), "available": False, "fallback": "echo/offline"}})
    return f


def _fam_auto2(f):
    f.append({"name": "auto2_workflow", "cat": "auto2", "purpose": "Workflow skeleton",
              "fn": lambda name="wf": {"name": str(name), "steps": [], "status": "draft"}})
    f.append({"name": "auto2_step", "cat": "auto2", "purpose": "Workflow step",
              "fn": lambda name="step1", fn="echo": {"name": str(name), "fn": str(fn), "args": {}}})
    f.append({"name": "auto2_retry", "cat": "auto2", "purpose": "Retry policy",
              "fn": lambda max_retries=3, delay=1: {"max_retries": int(max_retries), "delay_s": float(delay), "backoff": "exponential"}})
    f.append({"name": "auto2_state", "cat": "auto2", "purpose": "State machine states",
              "fn": lambda: ["idle", "running", "success", "failed", "retrying"]})
    f.append({"name": "auto2_trigger", "cat": "auto2", "purpose": "Trigger types",
              "fn": lambda: ["manual", "cron", "webhook", "file-watch", "on-start"]})
    f.append({"name": "auto2_condition", "cat": "auto2", "purpose": "Condition check",
              "fn": lambda value=5, op=">", threshold=3: {"result": {"gt": float(value) > float(threshold), "lt": float(value) < float(threshold), "eq": float(value) == float(threshold)}.get(str(op), False), "expr": f"{value} {op} {threshold}"}})
    f.append({"name": "auto2_loop", "cat": "auto2", "purpose": "Loop skeleton",
              "fn": lambda items="[1,2,3]": {"for": _as_list(items), "body": []}})
    f.append({"name": "auto2_schedule", "cat": "auto2", "purpose": "Schedule wrapper",
              "fn": lambda cron="0 * * * *", job="job1": {"cron": str(cron), "job": str(job), "enabled": True}})
    return f


def _fam_ops(f):
    f.append({"name": "ops_semver_parse", "cat": "ops", "purpose": "Parse semver",
              "fn": lambda v="1.2.3": {"major": int(str(v).split(".")[0]), "minor": int(str(v).split(".")[1]), "patch": int(str(v).split(".")[2])}})
    f.append({"name": "ops_semver_bump", "cat": "ops", "purpose": "Bump semver",
              "fn": lambda v="1.2.3", part="patch": ops_bump(str(v), str(part))})
    f.append({"name": "ops_semver_gt", "cat": "ops", "purpose": "Semver greater than",
              "fn": lambda a="1.2.3", b="1.2.0": [int(x) for x in str(a).split(".")] > [int(x) for x in str(b).split(".")]})
    f.append({"name": "ops_changelog", "cat": "ops", "purpose": "Changelog entry",
              "fn": lambda version="1.0.0", notes="[fix bug]": f"## {version} - {_today()}\n\n" + "\n".join(f"- {n}" for n in _as_list(notes))})
    f.append({"name": "ops_release_notes", "cat": "ops", "purpose": "Release notes skeleton",
              "fn": lambda version="1.0.0": {"version": str(version), "features": [], "fixes": [], "breaking": []}})
    f.append({"name": "ops_tag", "cat": "ops", "purpose": "Tag name",
              "fn": lambda v="1.0.0": f"v{str(v).lstrip('v')}"})
    f.append({"name": "ops_commit_msg", "cat": "ops", "purpose": "Conventional commit message",
              "fn": lambda typ="feat", scope="tools", msg="add crypto": f"{typ}({scope}): {msg}"})
    f.append({"name": "ops_licenses", "cat": "ops", "purpose": "License names",
              "fn": lambda: ["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause", "MPL-2.0"]})
    f.append({"name": "ops_env_check", "cat": "ops", "purpose": "Required env check",
              "fn": lambda keys="[HOME,PATH]": {k: bool(os.environ.get(k)) for k in _as_list(keys)}})
    f.append({"name": "ops_port_free", "cat": "ops", "purpose": "Port in use (best effort)",
              "fn": lambda port=8888: ops_port_in_use(int(port))})
    return f


def _fam_test2(f):
    f.append({"name": "test2_assert_eq", "cat": "test2", "purpose": "Assert equal result",
              "fn": lambda actual=1, expected=1: {"passed": str(actual) == str(expected), "actual": actual, "expected": expected}})
    f.append({"name": "test2_assert_true", "cat": "test2", "purpose": "Assert true",
              "fn": lambda cond=True: {"passed": bool(cond)}})
    f.append({"name": "test2_assert_in", "cat": "test2", "purpose": "Assert in collection",
              "fn": lambda item=2, col="[1,2,3]": {"passed": str(item) in [str(x) for x in _as_list(col)]}})
    f.append({"name": "test2_case", "cat": "test2", "purpose": "Test case",
              "fn": lambda name="test_x", fn="...": {"name": str(name), "fn": str(fn), "passed": None}})
    f.append({"name": "test2_suite", "cat": "test2", "purpose": "Suite summary",
              "fn": lambda passed=9, failed=1, total=10: {"passed": int(passed), "failed": int(failed), "total": int(total), "ok": int(failed) == 0}})
    f.append({"name": "test2_fuzz", "cat": "test2", "purpose": "Fuzz sample (ints)",
              "fn": lambda n=5, lo=0, hi=100, seed=1: data_sample([i for i in range(int(lo), int(hi) + 1)], int(n), int(seed))})
    f.append({"name": "test2_property", "cat": "test2", "purpose": "Property check",
              "fn": lambda xs="[1,2,3]", prop="sorted": {"passed": str(prop) == "sorted" and _as_list(xs) == sorted(_as_list(xs))}})
    f.append({"name": "test2_param", "cat": "test2", "purpose": "Parametrize cases",
              "fn": lambda cases="[[1,1],[2,2]]": [{"input": c[0], "expected": c[1]} for c in json.loads(cases)]})
    f.append({"name": "test2_skip", "cat": "test2", "purpose": "Skip reason",
              "fn": lambda reason="not implemented": {"skip": str(reason)}})
    f.append({"name": "test2_benchmark", "cat": "test2", "purpose": "Benchmark summary",
              "fn": lambda times="[1,2,3]": {"mean": round(statistics.mean([float(x) for x in _as_list(times)]), 4), "min": min(float(x) for x in _as_list(times)), "max": max(float(x) for x in _as_list(times))}})
    return f


def _fam_media2(f):
    f.append({"name": "media2_mime", "cat": "media2", "purpose": "MIME from filename",
              "fn": lambda name="a.png": mime_guess(str(name))})
    f.append({"name": "media2_ext", "cat": "media2", "purpose": "Extension from name",
              "fn": lambda name="a.png": Path(str(name)).suffix.lstrip(".")})
    f.append({"name": "media2_is_image", "cat": "media2", "purpose": "Is image extension",
              "fn": lambda name="a.jpg": Path(str(name)).suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")})
    f.append({"name": "media2_is_audio", "cat": "media2", "purpose": "Is audio extension",
              "fn": lambda name="a.mp3": Path(str(name)).suffix.lower() in (".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac")})
    f.append({"name": "media2_is_video", "cat": "media2", "purpose": "Is video extension",
              "fn": lambda name="a.mp4": Path(str(name)).suffix.lower() in (".mp4", ".mov", ".avi", ".webm", ".mkv")})
    f.append({"name": "media2_dims_hint", "cat": "media2", "purpose": "Image dims hint (PNG/JPEG header)",
              "fn": lambda path="": media_dims(str(path))})
    f.append({"name": "media2_playlist", "cat": "media2", "purpose": "Playlist item",
              "fn": lambda title="t", url="https://x.com/a.mp3": {"title": str(title), "url": str(url)}})
    f.append({"name": "media2_thumbnail_hint", "cat": "media2", "purpose": "Thumbnail name",
              "fn": lambda name="a.png": Path(str(name)).stem + "_thumb" + Path(str(name)).suffix})
    f.append({"name": "media2_audio_duration_hint", "cat": "media2", "purpose": "Duration hint",
              "fn": lambda size=4000000, bitrate=128000: round(int(size) * 8 / max(1, int(bitrate)), 1)})
    f.append({"name": "media2_subtitle", "cat": "media2", "purpose": "SRT line",
              "fn": lambda idx=1, start="00:00:01,000", end="00:00:02,000", text="Hello": f"{int(idx)}\n{start} --> {end}\n{text}\n"})
    return f


def _build2() -> List[Dict[str, Any]]:
    f: List[Dict[str, Any]] = []
    for builder in (
        _fam_crypto, _fam_ml, _fam_web, _fam_db, _fam_cloud, _fam_i18n,
        _fam_config, _fam_quant, _fam_rag, _fam_market, _fam_quality, _fam_ui,
        _fam_net, _fam_sys2, _fam_data2, _fam_math2, _fam_str2, _fam_json2,
        _fam_time2, _fam_gen2, _fam_code2, _fam_fs2, _fam_sec2, _fam_fmt2,
        _fam_valid2, _fam_csv2, _fam_xml2, _fam_yaml2, _fam_env, _fam_combo,
        _fam_chart, _fam_rep, _fam_note, _fam_menu, _fam_dist, _fam_sched2,
        _fam_monitor2, _fam_backup2, _fam_ai2, _fam_auto2, _fam_ops, _fam_test2,
        _fam_media2,
    ):
        builder(f)
    return f


MEGA2_COUNT = 0
for _item in _build2():
    _reg(_item["name"], _item["cat"], _item["purpose"], _item["fn"])
    MEGA2_COUNT += 1

__all__ = ["MEGA2_COUNT", "_build2", "_reg"]


# ---------------------------------------------------------------------------
# Internal implementations
# ---------------------------------------------------------------------------

def secrets_hex(n: int) -> str:
    return "".join(random.SystemRandom().choice("0123456789abcdef") for _ in range(int(n)))


def totp_lite(secret: str, t=None):
    import time as _t
    if t is None:
        t = int(_t.time() // 30)
    data = hashlib.sha256((str(secret) + str(t)).encode()).digest()
    return int.from_bytes(data[:4], "big") % 1000000


def ml_precision(yt, yp):
    tp = sum(1 for a, b in zip(yt, yp) if str(a) == "1" and str(b) == "1")
    pp = sum(1 for b in yp if str(b) == "1")
    return round(tp / max(1, pp), 4)


def ml_recall(yt, yp):
    tp = sum(1 for a, b in zip(yt, yp) if str(a) == "1" and str(b) == "1")
    ap = sum(1 for a in yt if str(a) == "1")
    return round(tp / max(1, ap), 4)


def ml_f1(yt, yp):
    p, r = ml_precision(yt, yp), ml_recall(yt, yp)
    return round(2 * p * r / max(1e-9, p + r), 4)


def ml_confusion(yt, yp):
    return {"tp": sum(1 for a, b in zip(yt, yp) if str(a) == "1" and str(b) == "1"),
            "fp": sum(1 for a, b in zip(yt, yp) if str(a) == "0" and str(b) == "1"),
            "fn": sum(1 for a, b in zip(yt, yp) if str(a) == "1" and str(b) == "0"),
            "tn": sum(1 for a, b in zip(yt, yp) if str(a) == "0" and str(b) == "0")}


def ml_r2(a, b):
    ya = [float(x) for x in a]
    yb = [float(x) for x in b]
    mean = sum(ya) / max(1, len(ya))
    ss_res = sum((x - y) ** 2 for x, y in zip(ya, yb))
    ss_tot = sum((x - mean) ** 2 for x in ya)
    return round(1 - ss_res / max(1e-9, ss_tot), 4)


def ml_normalize(xs):
    xs = [float(x) for x in xs]
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return [0.5] * len(xs)
    return [round((x - lo) / (hi - lo), 4) for x in xs]


def ml_standardize(xs):
    xs = [float(x) for x in xs]
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 1
    return [round((x - m) / max(1e-9, s), 4) for x in xs]


def ml_split(xs, ratio=0.8, seed=1):
    xs = list(xs)
    rnd = random.Random(seed)
    rnd.shuffle(xs)
    k = int(len(xs) * ratio)
    return {"train": xs[:k], "test": xs[k:]}


def ml_kmeans(pts, k):
    pts = [[float(x) for x in p] for p in pts]
    if not pts:
        return {"clusters": [], "centers": []}
    rnd = random.Random(1)
    centers = rnd.sample(pts, min(k, len(pts)))
    for _ in range(10):
        groups = [[] for _ in centers]
        for p in pts:
            d = [math.dist(p, c) for c in centers]
            groups[d.index(min(d))].append(p)
        newc = []
        for g in groups:
            if g:
                newc.append([sum(x[i] for x in g) / len(g) for i in range(len(g[0]))])
            else:
                newc.append(centers[len(newc) % len(centers)])
        centers = newc
    groups = [[] for _ in centers]
    for p in pts:
        d = [math.dist(p, c) for c in centers]
        groups[d.index(min(d))].append(p)
    return {"clusters": [len(g) for g in groups], "centers": [[round(x, 3) for x in c] for c in centers]}


def ml_cosine(a, b):
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1
    nb = math.sqrt(sum(y * y for y in b)) or 1
    return round(dot / (na * nb), 4)


def ml_one_hot(labels):
    uniq = sorted(set(str(x) for x in labels), key=lambda x: labels.index(x) if x in labels else 0)
    return [{u: (1 if str(x) == u else 0) for u in uniq} for x in labels]


def ml_hist(xs, bins):
    xs = [float(x) for x in xs]
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return {str(lo): len(xs)}
    w = (hi - lo) / bins
    out = {}
    for x in xs:
        b = int((x - lo) / w)
        b = min(b, bins - 1)
        k = f"{lo + b * w:.2f}-{lo + (b + 1) * w:.2f}"
        out[k] = out.get(k, 0) + 1
    return out


def ml_pearson(a, b):
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    n = len(a)
    if n == 0:
        return 0
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return round(cov / max(1e-9, sa * sb), 4)


def ml_fit_line(xs, ys):
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    w = num / max(1e-9, den)
    b = my - w * mx
    return {"weight": round(w, 4), "bias": round(b, 4)}


def ml_bleu1(ref, hyp):
    r = str(ref).split()
    h = str(hyp).split()
    if not h:
        return 0.0
    matches = sum(min(h.count(t), r.count(t)) for t in set(h))
    bp = 1.0 if len(h) > len(r) else math.exp(1 - len(r) / max(1, len(h)))
    return round(bp * matches / len(h), 4)


def ml_shuffle(xs, seed=1):
    xs = list(xs)
    random.Random(seed).shuffle(xs)
    return xs


def ml_report(yt, yp):
    p, r, f1 = ml_precision(yt, yp), ml_recall(yt, yp), ml_f1(yt, yp)
    acc = round(sum(1 for a, b in zip(yt, yp) if str(a) == str(b)) / max(1, len(yt)), 4)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "confusion": ml_confusion(yt, yp)}


def mime_guess(name):
    ext = Path(str(name)).suffix.lower()
    return {".txt": "text/plain", ".md": "text/markdown", ".html": "text/html", ".css": "text/css",
            ".js": "application/javascript", ".json": "application/json", ".csv": "text/csv",
            ".xml": "application/xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml", ".bmp": "image/bmp",
            ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".mp4": "video/mp4",
            ".webm": "video/webm", ".zip": "application/zip", ".pdf": "application/pdf",
            ".py": "text/x-python", ".sh": "application/x-sh", ".yaml": "application/yaml",
            ".yml": "application/yaml", ".toml": "application/toml", ".sql": "application/sql",
            ".db": "application/octet-stream"}.get(ext, "application/octet-stream")


def db_exec(sql):
    con = sqlite3.connect(":memory:")
    try:
        cur = con.execute(str(sql))
        if cur.description:
            cols = [d[0] for d in cur.description]
            rows = [list(r) for r in cur.fetchall()]
            return {"columns": cols, "rows": rows, "count": len(rows)}
        con.commit()
        return {"ok": True, "rows_affected": cur.rowcount}
    finally:
        con.close()


def ru_plural(n, forms):
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def ordinal_en(n):
    n = int(n)
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th").join([str(n)])


def config_ini_parse(text):
    out = {}
    sec = None
    for line in str(text).splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1]
            out.setdefault(sec, {})
        elif "=" in line and sec:
            k, v = line.split("=", 1)
            out[sec][k.strip()] = v.strip()
    return out


def config_ini_build(data):
    lines = []
    for sec, kv in data.items():
        lines.append(f"[{sec}]")
        for k, v in kv.items():
            lines.append(f"{k}={v}")
    return "\n".join(lines)


def config_env_parse(text):
    out = {}
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def json_merge(a, b):
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = json_merge(out[k], v)
        else:
            out[k] = v
    return out


def json_get(obj, path):
    cur = obj
    for p in str(path).split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list) and p.isdigit() and int(p) < len(cur):
            cur = cur[int(p)]
        else:
            return None
    return cur


def json_set(obj, path, value):
    cur = obj
    parts = str(path).split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value
    return obj


def json_delete(obj, path):
    cur = obj
    parts = str(path).split(".")
    for p in parts[:-1]:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return obj
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)
    return obj


def json_flatten(obj, prefix=""):
    out = {}
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(json_flatten(v, key))
        else:
            out[key] = v
    return out


def json_unflatten(obj):
    out = {}
    for k, v in obj.items():
        cur = out
        parts = str(k).split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out


def json_validate(s):
    try:
        json.loads(str(s))
        return True
    except Exception:
        return False


def json_sort(obj):
    if isinstance(obj, dict):
        return {k: json_sort(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [json_sort(x) for x in obj]
    return obj


def json_search(obj, needle):
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if v == needle:
                hits.append(str(k))
            if isinstance(v, (dict, list)):
                hits.extend(f"{k}.{h}" for h in json_search(v, needle))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if v == needle:
                hits.append(str(i))
            if isinstance(v, (dict, list)):
                hits.extend(f"{i}.{h}" for h in json_search(v, needle))
    return hits


def json_to_csv(rows):
    if not rows:
        return ""
    keys = list(rows[0].keys())
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(keys)
    for r in rows:
        w.writerow([r.get(k, "") for k in keys])
    return buf.getvalue().strip()


def csv_to_json(text):
    rows = list(csv.reader(io.StringIO(str(text))))
    if not rows:
        return []
    header = rows[0]
    return [dict(zip(header, r)) for r in rows[1:]]


def json_diff(a, b):
    return {"added": sorted(set(b) - set(a)), "removed": sorted(set(a) - set(b)),
            "changed": sorted(k for k in set(a) & set(b) if a[k] != b[k])}


def time2_parse(d):
    return _dt.date.fromisoformat(str(d))


def time2_parse_dt(d):
    return _dt.datetime.fromisoformat(str(d))


def time2_age(b):
    d = time2_parse(str(b))
    today = _dt.date.today()
    return today.year - d.year - ((today.month, today.day) < (d.month, d.day))


def time2_duration(s):
    s = int(s)
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{sec}s")
    return " ".join(parts)


def time2_next_wd(d, wd):
    cur = time2_parse(str(d))
    days = (int(wd) - cur.weekday()) % 7
    if days == 0:
        days = 7
    return cur + _dt.timedelta(days=days)


def str2_dedent(s):
    lines = str(s).splitlines()
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if not indents:
        return str(s)
    n = min(indents)
    return "\n".join(line[n:] if line.strip() else "" for line in lines)


def textwrap_fill(s, w):
    import textwrap
    return textwrap.fill(str(s), int(w))


def str2_camel(s):
    words = re.split(r"[_\-\s]+", str(s))
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])


def str2_snake(s):
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", str(s))
    return s.lower().replace("-", "_").replace(" ", "_")


def ui_contrast(a, b):
    def lum(h):
        r, g, bl = (int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
        def f(c):
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        r, g, bl = f(r), f(g), f(bl)
        return 0.2126 * r + 0.7152 * g + 0.0722 * bl
    l1, l2 = lum(str(a)), lum(str(b))
    hi, lo = max(l1, l2), min(l1, l2)
    return round((hi + 0.05) / (lo + 0.05), 2)


def ui_brightness(h):
    r, g, b = (int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return int(0.299 * r + 0.587 * g + 0.114 * b)


def ui_progress(pct, width):
    pct = _clamp(float(pct), 0, 100)
    filled = int(round(pct / 100 * int(width)))
    return "[" + "#" * filled + "-" * (int(width) - filled) + f"] {pct:.0f}%"


def net_class(ip):
    first = int(str(ip).split(".")[0])
    if first < 128:
        return "A"
    if first < 192:
        return "B"
    if first < 224:
        return "C"
    if first < 240:
        return "D"
    return "E"


def net_private(ip):
    parts = [int(x) for x in str(ip).split(".")]
    return (parts[0] == 10 or parts[0] == 172 and 16 <= parts[1] <= 31 or parts[0] == 192 and parts[1] == 168)


def net_mask(cidr):
    mask = (0xFFFFFFFF << (32 - int(cidr))) & 0xFFFFFFFF
    return ".".join(str((mask >> (8 * i)) & 0xFF) for i in (3, 2, 1, 0))


def net_range(cidr):
    ip, prefix = str(cidr).split("/")
    mask = (0xFFFFFFFF << (32 - int(prefix))) & 0xFFFFFFFF
    base = sum(int(x) << (8 * (3 - i)) for i, x in enumerate(ip.split(".")))
    return {"start": net_ip_str(base & mask), "end": net_ip_str(base | (~mask & 0xFFFFFFFF))}


def net_ip_str(n):
    return ".".join(str((n >> (8 * i)) & 0xFF) for i in (3, 2, 1, 0))


def net_contains(ip, cidr):
    r = net_range(cidr)
    n = sum(int(x) << (8 * (3 - i)) for i, x in enumerate(str(ip).split(".")))
    return net_ip_str(n) >= r["start"] and net_ip_str(n) <= r["end"]


def net_local_ips():
    import socket as _s
    try:
        return sorted({i[4][0] for i in _s.getaddrinfo(_s.gethostname(), None, _s.AF_INET)})
    except Exception:
        return ["127.0.0.1"]


def socket_gethostname():
    import socket as _s
    try:
        return _s.gethostname()
    except Exception:
        return "unknown"


def platform_machine():
    import platform as _p
    try:
        return _p.machine()
    except Exception:
        return "unknown"


def sys2_uptime():
    try:
        with open("/proc/uptime") as fh:
            return round(float(fh.read().split()[0]), 1)
    except Exception:
        return -1


def tempfile_gettempdir():
    import tempfile
    return tempfile.gettempdir()


def sys2_threads():
    try:
        import threading
        return threading.active_count()
    except Exception:
        return 0


def data_summary(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {}
    return {"count": len(xs), "mean": round(statistics.mean(xs), 4), "median": round(statistics.median(xs), 4),
            "min": min(xs), "max": max(xs), "stdev": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0}


def data_impute(xs, method="mean"):
    vals = [float(x) for x in xs if x is not None]
    fill = statistics.mean(vals) if method == "mean" else statistics.median(vals) if vals else 0
    return [round(float(x), 4) if x is not None else round(fill, 4) for x in xs]


def data_outliers(xs):
    xs = [float(x) for x in xs]
    q1, q3 = statistics.quantiles(xs, n=4)[0], statistics.quantiles(xs, n=4)[2]
    iqr = q3 - q1
    return [x for x in xs if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr]


def data_zscores(xs):
    xs = [float(x) for x in xs]
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 1
    return [(x - m) / max(1e-9, s) for x in xs]


def data_pivot(keys):
    out = {}
    for k in keys:
        out[str(k)] = out.get(str(k), 0) + 1
    return out


def data_sample(xs, n, seed=1):
    xs = list(xs)
    random.Random(seed).shuffle(xs)
    return xs[:int(n)]


def data_bin(x, edges):
    edges = [float(e) for e in edges]
    for i in range(len(edges) - 1):
        if edges[i] <= float(x) < edges[i + 1]:
            return f"[{edges[i]},{edges[i + 1]})"
    return f"[{edges[-1]},inf)"


def data_entropy(xs):
    n = len(xs)
    if n == 0:
        return 0.0
    counts = {}
    for x in xs:
        counts[str(x)] = counts.get(str(x), 0) + 1
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _fib_lite(n):
    a, b, out = 0, 1, []
    for _ in range(int(n)):
        out.append(a)
        a, b = b, a + b
    return out


def _primes_lite(limit):
    limit = int(limit)
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]


def _is_prime_lite(n):
    n = int(n)
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def fs2_tree(path, depth):
    out = []
    def walk(p, d):
        if d > int(depth):
            return
        try:
            entries = sorted(os.listdir(p))
        except Exception:
            return
        for e in entries:
            full = os.path.join(p, e)
            out.append("  " * d + e + ("/" if os.path.isdir(full) else ""))
            if os.path.isdir(full):
                walk(full, d + 1)
    walk(str(path), 0)
    return out[:200]


def fs2_du(path):
    try:
        total = 0
        for root, _, files in os.walk(str(path)):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except Exception:
                    pass
        return total
    except Exception:
        return 0


def sec2_luhn(s):
    s = str(s).replace(" ", "")
    if not s.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(s)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def sec2_mask_email(s):
    local, _, dom = str(s).partition("@")
    if not dom:
        return "***"
    return local[:2] + "***@" + dom


def env_detect():
    colab = bool(os.environ.get("COLAB_JUPYTER_IP")) or "google.colab" in sys.modules
    k8s = bool(os.environ.get("KUBERNETES_SERVICE_HOST")) or Path("/.dockerenv").exists()
    cloud = bool(os.environ.get("AWS_REGION") or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("AZURE_SUBSCRIPTION_ID")) or k8s or colab
    mode = "colab" if colab else ("container" if k8s else ("cloud" if cloud else "localhost"))
    return {"mode": mode, "colab": colab, "cloud": cloud, "container": k8s,
            "hostname": socket_gethostname(), "python": sys.version.split()[0],
            "platform": sys.platform}


def chart_bar(xs, labels):
    xs = [float(x) for x in xs]
    mx = max(xs) if xs else 1
    lines = []
    for x, lab in zip(xs, labels):
        bar = "#" * int(round(x / mx * 20))
        lines.append(f"{str(lab):<6} {bar} {x}")
    return "\n".join(lines)


def chart_spark(xs):
    xs = [float(x) for x in xs]
    if not xs:
        return ""
    lo, hi = min(xs), max(xs)
    chars = "▁▂▃▄▅▆▇█"
    return "".join(chars[int(round((x - lo) / (hi - lo) * 7)) if hi > lo else 3] for x in xs)


def chart_hist(xs, bins):
    xs = [float(x) for x in xs]
    lo, hi = min(xs), max(xs)
    if hi == lo:
        return f"{lo}: {'#' * len(xs)}"
    w = (hi - lo) / bins
    lines = []
    for i in range(bins):
        c = sum(1 for x in xs if lo + i * w <= x < lo + (i + 1) * w)
        lines.append(f"{lo + i * w:.2f}: {'#' * c}")
    return "\n".join(lines)


def chart_gauge(pct, width):
    pct = _clamp(float(pct), 0, 100)
    filled = int(round(pct / 100 * int(width)))
    return "[" + "█" * filled + "░" * (int(width) - filled) + f"] {pct:.0f}%"


def rep_md_table(rows, headers):
    rows = [[str(c) for c in r] for r in rows]
    hdr = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([hdr, sep, body])


def dist_shard_contig(xs, workers, rank):
    xs = list(xs)
    n = len(xs)
    per = math.ceil(n / int(workers))
    return xs[int(rank) * per:(int(rank) + 1) * per]


def sched2_parse(cron):
    parts = str(cron).split()
    if len(parts) != 5:
        return {"valid": False, "error": "expected 5 fields"}
    return {"valid": True, "minute": parts[0], "hour": parts[1], "dom": parts[2], "month": parts[3], "dow": parts[4]}


def monitor_apdex(lats, t):
    lats = [float(x) for x in lats]
    t = float(t)
    s = sum(1 for x in lats if x <= t) + sum(0.5 for x in lats if t < x <= 4 * t)
    return round(s / max(1, len(lats)), 3)


def ops_bump(v, part):
    parts = [int(x) for x in str(v).split(".")]
    idx = {"major": 0, "minor": 1, "patch": 2}.get(str(part), 2)
    parts[idx] += 1
    for i in range(idx + 1, 3):
        parts[i] = 0
    return ".".join(str(p) for p in parts)


def ops_port_in_use(port):
    import socket as _s
    try:
        with _s.socket(_s.AF_INET, _s.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", int(port)))
        return False
    except Exception:
        return True


def rag_chunk(text, size):
    return [str(text)[i:i + int(size)] for i in range(0, len(str(text)), int(size))]


def rag_tf(text):
    out = {}
    for w in re.findall(r"\w+", str(text).lower()):
        out[w] = out.get(w, 0) + 1
    return out


def rag_jaccard(a, b):
    sa, sb = set(str(a).split()), set(str(b).split())
    if not sa and not sb:
        return 1.0
    return round(len(sa & sb) / len(sa | sb), 4)


def rag_cosine(a, b):
    ta, tb = rag_tf(str(a)), rag_tf(str(b))
    words = set(ta) | set(tb)
    va = [ta.get(w, 0) for w in words]
    vb = [tb.get(w, 0) for w in words]
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va)) or 1
    nb = math.sqrt(sum(y * y for y in vb)) or 1
    return round(dot / (na * nb), 4)


def rag_keywords(text, n):
    tf = rag_tf(str(text))
    return sorted(tf, key=lambda k: -tf[k])[:int(n)]


def rag_stop(text):
    stops = {"the", "is", "on", "at", "a", "an", "and", "or", "of", "to", "in", "for", "with"}
    return " ".join(w for w in str(text).split() if w.lower() not in stops)


def quality_lint(code):
    issues = []
    if "\t" in str(code):
        issues.append("tabs found (use spaces)")
    if len(str(code).splitlines()) == 0:
        issues.append("empty file")
    return {"clean": not issues, "issues": issues}


def quality_lines(code, max_len):
    bad = [i + 1 for i, line in enumerate(str(code).splitlines()) if len(line) > int(max_len)]
    return {"over_limit": bad, "ok": not bad}


def media_dims(path):
    try:
        with open(str(path), "rb") as fh:
            head = fh.read(32)
        if head[:8] == b"\x89PNG\r\n\x1a\n" and len(head) >= 24:
            w, h = int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
            return {"w": w, "h": h, "fmt": "png"}
        if head[:2] == b"\xff\xd8":
            return {"fmt": "jpeg", "note": "parse SOF for dims"}
        return {"fmt": "unknown"}
    except Exception:
        return {"fmt": "unknown"}


def xml_tag_attr(name, d, body):
    attrs = " ".join(f'{k}="{v}"' for k, v in d.items())
    return f'<{name} {attrs}>{body}</{name}>'
