"""AWEAI data-science tools — statistics, ML helpers, transforms, metrics.

Each tool has a unique purpose and works with pure Python + numpy (optional).
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import median, mode, pstdev, pvariance, stdev, variance
from typing import Any, Dict, List, Optional, Sequence

from aweai.tools.registry import tool


def _num_list(values: Any) -> List[float]:
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            values = [float(x) for x in values.replace(",", " ").split()]
    return [float(x) for x in values]


@tool("stats_describe", "datascience", "Descriptive statistics of a numeric list (min, max, mean, median, std)")
def stats_describe(values: Any) -> Dict[str, Any]:
    xs = _num_list(values)
    if not xs:
        return {"error": "empty list"}
    n = len(xs)
    mean = sum(xs) / n
    return {
        "count": n,
        "min": min(xs),
        "max": max(xs),
        "mean": mean,
        "median": median(xs),
        "stdev": stdev(xs) if n > 1 else 0.0,
        "variance": variance(xs) if n > 1 else 0.0,
        "sum": sum(xs),
        "range": max(xs) - min(xs),
    }


@tool("stats_histogram", "datascience", "Build a histogram (buckets + counts) of a numeric list")
def stats_histogram(values: Any, bins: int = 10) -> Dict[str, Any]:
    xs = _num_list(values)
    if not xs:
        return {"error": "empty list"}
    lo, hi = min(xs), max(xs)
    if lo == hi:
        return {"bins": [{"lo": lo, "hi": hi, "count": len(xs)}]}
    width = (hi - lo) / bins
    buckets = [0] * bins
    for x in xs:
        idx = min(int((x - lo) / width), bins - 1)
        buckets[idx] += 1
    out = []
    for i, c in enumerate(buckets):
        out.append({"lo": round(lo + i * width, 4), "hi": round(lo + (i + 1) * width, 4), "count": c})
    return {"bins": out}


@tool("stats_correlation", "datascience", "Pearson correlation between two numeric lists")
def stats_correlation(a: Any, b: Any) -> Dict[str, Any]:
    xs = _num_list(a)
    ys = _num_list(b)
    if len(xs) != len(ys) or len(xs) < 2:
        return {"error": "lists must be equal length >= 2"}
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return {"correlation": 0.0}
    return {"correlation": num / (dx * dy)}


@tool("stats_percentile", "datascience", "Compute percentiles of a numeric list (50th = median)")
def stats_percentile(values: Any, percentile: float = 50.0) -> Dict[str, Any]:
    xs = sorted(_num_list(values))
    if not xs:
        return {"error": "empty list"}
    k = (len(xs) - 1) * percentile / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return {"percentile": percentile, "value": xs[int(k)]}
    return {"percentile": percentile, "value": xs[f] * (c - k) + xs[c] * (k - f)}


@tool("stats_quantiles", "datascience", "Compute quartiles (Q1, Q2, Q3) of a numeric list")
def stats_quantiles(values: Any) -> Dict[str, Any]:
    xs = sorted(_num_list(values))
    if not xs:
        return {"error": "empty list"}
    q = lambda p: stats_percentile(xs, p)["value"]  # noqa: E731
    return {"q0": xs[0], "q1": q(25), "q2": q(50), "q3": q(75), "q4": xs[-1]}


@tool("stats_mode", "datascience", "Find the most frequent value(s) of a list")
def stats_mode(values: Any) -> Dict[str, Any]:
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            values = values.split()
    counts = Counter(values)
    if not counts:
        return {"error": "empty list"}
    top = max(counts.values())
    return {"mode": [k for k, v in counts.items() if v == top], "count": top}


@tool("stats_unique", "datascience", "Count unique values and their frequencies")
def stats_unique(values: Any) -> Dict[str, Any]:
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            values = values.split()
    counts = Counter(values)
    return {"unique": len(counts), "frequencies": dict(counts.most_common())}


@tool("stats_normalize", "datascience", "Min-max normalize a numeric list to [0, 1]")
def stats_normalize(values: Any) -> Dict[str, Any]:
    xs = _num_list(values)
    if not xs:
        return {"error": "empty list"}
    lo, hi = min(xs), max(xs)
    if lo == hi:
        return {"normalized": [0.0] * len(xs)}
    return {"normalized": [(x - lo) / (hi - lo) for x in xs]}


@tool("stats_zscore", "datascience", "Standardize a numeric list to z-scores (mean 0, std 1)")
def stats_zscore(values: Any) -> Dict[str, Any]:
    xs = _num_list(values)
    n = len(xs)
    if n == 0:
        return {"error": "empty list"}
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / n)
    if sd == 0:
        return {"z": [0.0] * n}
    return {"z": [(x - mean) / sd for x in xs]}


@tool("stats_outliers", "datascience", "Detect outliers using the IQR rule")
def stats_outliers(values: Any) -> Dict[str, Any]:
    xs = _num_list(values)
    if len(xs) < 4:
        return {"error": "need at least 4 values"}
    q = lambda p: stats_percentile(xs, p)["value"]  # noqa: E731
    q1, q3 = q(25), q(75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = [x for x in xs if x < lo or x > hi]
    return {"q1": q1, "q3": q3, "iqr": iqr, "bounds": [lo, hi], "outliers": outliers, "count": len(outliers)}


@tool("ml_train_test_split", "datascience", "Split a list into train/test subsets with a ratio and seed")
def ml_train_test_split(values: Any, ratio: float = 0.8, seed: int = 0) -> Dict[str, Any]:
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            values = values.split()
    xs = list(values)
    rng = random.Random(seed)
    xs = list(xs)
    rng.shuffle(xs)
    n = int(len(xs) * ratio)
    return {"train": xs[:n], "test": xs[n:], "train_count": n, "test_count": len(xs) - n}


@tool("ml_confusion_matrix", "datascience", "Build a confusion matrix from actual/predicted labels")
def ml_confusion_matrix(actual: Any, predicted: Any) -> Dict[str, Any]:
    if isinstance(actual, str):
        actual = json.loads(actual)
    if isinstance(predicted, str):
        predicted = json.loads(predicted)
    labels = sorted(set(actual) | set(predicted))
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for a, p in zip(actual, predicted):
        matrix[a][p] = matrix[a].get(p, 0) + 1
    return {"labels": labels, "matrix": matrix}


@tool("ml_accuracy", "datascience", "Compute classification accuracy")
def ml_accuracy(actual: Any, predicted: Any) -> Dict[str, Any]:
    if isinstance(actual, str):
        actual = json.loads(actual)
    if isinstance(predicted, str):
        predicted = json.loads(predicted)
    if len(actual) != len(predicted) or not actual:
        return {"error": "lists must match and be non-empty"}
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return {"accuracy": correct / len(actual), "correct": correct, "total": len(actual)}


@tool("ml_precision_recall", "datascience", "Compute precision and recall for a binary classification")
def ml_precision_recall(actual: Any, predicted: Any, positive: Any = 1) -> Dict[str, Any]:
    if isinstance(actual, str):
        actual = json.loads(actual)
    if isinstance(predicted, str):
        predicted = json.loads(predicted)
    tp = sum(1 for a, p in zip(actual, predicted) if a == positive and p == positive)
    fp = sum(1 for a, p in zip(actual, predicted) if a != positive and p == positive)
    fn = sum(1 for a, p in zip(actual, predicted) if a == positive and p != positive)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


@tool("ml_rmse", "datascience", "Compute RMSE between actual and predicted numeric values")
def ml_rmse(actual: Any, predicted: Any) -> Dict[str, Any]:
    a = _num_list(actual)
    p = _num_list(predicted)
    if len(a) != len(p) or not a:
        return {"error": "lists must match and be non-empty"}
    mse = sum((x - y) ** 2 for x, y in zip(a, p)) / len(a)
    return {"rmse": math.sqrt(mse), "mse": mse}


@tool("ml_mae", "datascience", "Compute mean absolute error between actual and predicted values")
def ml_mae(actual: Any, predicted: Any) -> Dict[str, Any]:
    a = _num_list(actual)
    p = _num_list(predicted)
    if len(a) != len(p) or not a:
        return {"error": "lists must match and be non-empty"}
    return {"mae": sum(abs(x - y) for x, y in zip(a, p)) / len(a)}


@tool("ml_r2", "datascience", "Compute R-squared coefficient of determination")
def ml_r2(actual: Any, predicted: Any) -> Dict[str, Any]:
    a = _num_list(actual)
    p = _num_list(predicted)
    if len(a) != len(p) or not a:
        return {"error": "lists must match and be non-empty"}
    mean = sum(a) / len(a)
    ss_res = sum((x - y) ** 2 for x, y in zip(a, p))
    ss_tot = sum((x - mean) ** 2 for x in a)
    if ss_tot == 0:
        return {"r2": 0.0}
    return {"r2": 1 - ss_res / ss_tot}


@tool("data_linspace", "datascience", "Generate evenly spaced numbers over an interval")
def data_linspace(lo: float = 0.0, hi: float = 1.0, n: int = 10) -> Dict[str, Any]:
    if n < 2:
        return {"error": "n must be >= 2"}
    step = (hi - lo) / (n - 1)
    return {"values": [round(lo + i * step, 6) for i in range(n)]}


@tool("data_arange", "datascience", "Generate a numeric range (like range() but floats supported)")
def data_arange(lo: float = 0.0, hi: float = 10.0, step: float = 1.0) -> Dict[str, Any]:
    vals = []
    x = lo
    while x < hi:
        vals.append(round(x, 6))
        x += step
    return {"values": vals}


@tool("data_zeros", "datascience", "Create a list of zeros of length n")
def data_zeros(n: int = 10) -> Dict[str, Any]:
    return {"values": [0.0] * n}


@tool("data_ones", "datascience", "Create a list of ones of length n")
def data_ones(n: int = 10) -> Dict[str, Any]:
    return {"values": [1.0] * n}


@tool("data_random_normal", "datascience", "Generate samples from a normal distribution")
def data_random_normal(mean: float = 0.0, stdev: float = 1.0, n: int = 100, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    return {"values": [round(rng.gauss(mean, stdev), 6) for _ in range(n)]}


@tool("data_random_uniform", "datascience", "Generate samples from a uniform distribution")
def data_random_uniform(lo: float = 0.0, hi: float = 1.0, n: int = 100, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    return {"values": [round(rng.uniform(lo, hi), 6) for _ in range(n)]}


@tool("data_sample", "datascience", "Randomly sample k items from a list")
def data_sample(values: Any, k: int = 5, seed: int = 0) -> Dict[str, Any]:
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            values = values.split()
    rng = random.Random(seed)
    k = min(k, len(values))
    return {"sample": rng.sample(list(values), k)}


@tool("data_shuffle", "datascience", "Shuffle a list deterministically with a seed")
def data_shuffle(values: Any, seed: int = 0) -> Dict[str, Any]:
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            values = values.split()
    xs = list(values)
    random.Random(seed).shuffle(xs)
    return {"shuffled": xs}


@tool("data_dedup", "datascience", "Remove duplicates from a list preserving order")
def data_dedup(values: Any) -> Dict[str, Any]:
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            values = values.split()
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return {"deduped": out, "removed": len(values) - len(out)}


@tool("data_flatten", "datascience", "Flatten a nested list one level deep")
def data_flatten(values: Any) -> Dict[str, Any]:
    if isinstance(values, str):
        values = json.loads(values)
    out = []
    for v in values:
        if isinstance(v, list):
            out.extend(v)
        else:
            out.append(v)
    return {"flattened": out}


@tool("data_chunk", "datascience", "Split a list into chunks of size n")
def data_chunk(values: Any, size: int = 10) -> Dict[str, Any]:
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            values = values.split()
    if size <= 0:
        return {"error": "size must be > 0"}
    chunks = [values[i:i + size] for i in range(0, len(values), size)]
    return {"chunks": chunks, "count": len(chunks)}


@tool("data_reverse", "datascience", "Reverse a list")
def data_reverse(values: Any) -> Dict[str, Any]:
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            values = values.split()
    return {"reversed": list(reversed(values))}


@tool("data_sort", "datascience", "Sort a list (numeric-aware)")
def data_sort(values: Any, descending: bool = False) -> Dict[str, Any]:
    if isinstance(values, str):
        try:
            values = json.loads(values)
        except Exception:
            values = values.split()
    try:
        xs = sorted(values, key=lambda v: float(v))
    except (ValueError, TypeError):
        xs = sorted(values)
    if descending:
        xs = list(reversed(xs))
    return {"sorted": xs}


@tool("data_summary_csv", "datascience", "Summarize a CSV file (columns, rows, dtypes)")
def data_summary_csv(path: str) -> Dict[str, Any]:
    import csv

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = 0
        for _ in reader:
            rows += 1
    return {"path": path, "columns": cols, "column_count": len(cols), "rows": rows}


@tool("data_convert_csv_json", "datascience", "Convert a CSV file to a JSON array")
def data_convert_csv_json(path: str, output: str = "") -> Dict[str, Any]:
    import csv

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    if output:
        Path(output).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"path": path, "rows": len(rows), "output": output}
    return {"path": path, "rows": len(rows), "data": rows[:50]}


@tool("text_tokenize", "datascience", "Tokenize text into words (lowercase, punctuation stripped)")
def text_tokenize(text: str) -> Dict[str, Any]:
    import re

    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return {"tokens": tokens, "count": len(tokens)}


@tool("text_word_freq", "datascience", "Word frequency table of a text")
def text_word_freq(text: str, top: int = 20) -> Dict[str, Any]:
    import re
    from collections import Counter

    words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    counts = Counter(words)
    return {"top": dict(counts.most_common(top)), "unique": len(counts), "total": len(words)}


@tool("text_ngrams", "datascience", "Extract n-grams from text")
def text_ngrams(text: str, n: int = 2) -> Dict[str, Any]:
    import re

    words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    ngrams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return {"ngrams": ngrams, "count": len(ngrams)}


@tool("text_sentences", "datascience", "Split text into sentences")
def text_sentences(text: str) -> Dict[str, Any]:
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return {"sentences": [s for s in parts if s], "count": len(parts)}


@tool("text_slugify", "datascience", "Convert text to a URL-friendly slug")
def text_slugify(text: str) -> Dict[str, Any]:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return {"slug": slug}


@tool("text_truncate", "datascience", "Truncate text to max characters with ellipsis")
def text_truncate(text: str, max_chars: int = 100) -> Dict[str, Any]:
    if len(text) <= max_chars:
        return {"text": text, "truncated": False}
    return {"text": text[:max_chars].rstrip() + "...", "truncated": True}


@tool("text_wrap", "datascience", "Wrap text to a given line width")
def text_wrap(text: str, width: int = 80) -> Dict[str, Any]:
    import textwrap

    return {"wrapped": textwrap.fill(text, width=width)}


@tool("text_contains", "datascience", "Check whether text contains a substring (case-insensitive)")
def text_contains(text: str, needle: str) -> Dict[str, Any]:
    return {"contains": needle.lower() in text.lower()}


@tool("text_replace", "datascience", "Replace all occurrences of a substring")
def text_replace(text: str, old: str, new: str) -> Dict[str, Any]:
    return {"text": text.replace(old, new), "occurrences": text.count(old)}


@tool("text_upper", "datascience", "Convert text to UPPERCASE")
def text_upper(text: str) -> Dict[str, Any]:
    return {"text": text.upper()}


@tool("text_lower", "datascience", "Convert text to lowercase")
def text_lower(text: str) -> Dict[str, Any]:
    return {"text": text.lower()}


@tool("text_title", "datascience", "Convert text to Title Case")
def text_title(text: str) -> Dict[str, Any]:
    return {"text": text.title()}


@tool("text_strip", "datascience", "Strip whitespace from both ends of text")
def text_strip(text: str) -> Dict[str, Any]:
    return {"text": text.strip()}


@tool("text_reverse", "datascience", "Reverse a string")
def text_reverse(text: str) -> Dict[str, Any]:
    return {"text": text[::-1]}


@tool("text_lines", "datascience", "Split text into lines")
def text_lines(text: str) -> Dict[str, Any]:
    return {"lines": text.splitlines(), "count": len(text.splitlines())}


@tool("math_fibonacci", "datascience", "Generate the first n Fibonacci numbers")
def math_fibonacci(n: int = 10) -> Dict[str, Any]:
    if n <= 0:
        return {"error": "n must be > 0"}
    a, b = 0, 1
    out = []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return {"fib": out}


@tool("math_factorial", "datascience", "Compute factorial of n")
def math_factorial(n: int = 5) -> Dict[str, Any]:
    return {"factorial": math.factorial(n)}


@tool("math_primes", "datascience", "Generate prime numbers up to a limit (sieve)")
def math_primes(limit: int = 100) -> Dict[str, Any]:
    if limit < 2:
        return {"primes": []}
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return {"primes": [i for i, v in enumerate(sieve) if v]}


@tool("math_gcd", "datascience", "Greatest common divisor of two numbers")
def math_gcd(a: int = 12, b: int = 18) -> Dict[str, Any]:
    return {"gcd": math.gcd(a, b)}


@tool("math_lcm", "datascience", "Least common multiple of two numbers")
def math_lcm(a: int = 4, b: int = 6) -> Dict[str, Any]:
    return {"lcm": abs(a * b) // math.gcd(a, b)}


@tool("math_pow", "datascience", "Raise a number to a power")
def math_pow(base: float = 2.0, exponent: float = 10.0) -> Dict[str, Any]:
    return {"result": base ** exponent}


@tool("math_log", "datascience", "Natural logarithm of a number")
def math_log(value: float = 1.0) -> Dict[str, Any]:
    return {"log": math.log(value)}


@tool("math_sqrt", "datascience", "Square root of a number")
def math_sqrt(value: float = 16.0) -> Dict[str, Any]:
    return {"sqrt": math.sqrt(value)}


@tool("math_round", "datascience", "Round a number to n decimal places")
def math_round(value: float = 3.14159, ndigits: int = 2) -> Dict[str, Any]:
    return {"value": round(value, ndigits)}


__all__ = []
