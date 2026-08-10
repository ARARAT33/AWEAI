# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Data management commands: datasets, pipelines, preprocessing,
tokenization and embeddings."""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from aweai.cmd.common import err, jdump, ok

app = typer.Typer(help="Data management: datasets, pipelines, preprocessing, tokenization, embedding")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
@app.command("inspect")
def inspect(
    path: str = typer.Argument(..., help="JSONL/CSV/text file"),
    rows: int = typer.Option(5, "--rows", "-n"),
):
    """Inspect the first rows of a dataset."""
    try:
        p = Path(path)
        if p.suffix == ".jsonl":
            lines = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            typer.echo(jdump({"total": len(lines), "preview": lines[:rows]}))
        elif p.suffix == ".csv":
            import csv

            with open(path, encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
            typer.echo(jdump({"total": len(reader), "preview": reader[:rows]}))
        else:
            text = p.read_text(encoding="utf-8")
            lines = text.splitlines()
            typer.echo(jdump({"total": len(lines), "chars": len(text),
                              "preview": [ln[:200] for ln in lines[:rows]]}))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("split")
def split(
    path: str = typer.Argument(..., help="JSONL dataset"),
    ratio: float = typer.Option(0.8, "--ratio", "-r", help="Train ratio"),
    seed: int = typer.Option(1, "--seed"),
    out_dir: str = typer.Option("data/split", "--out-dir", "-o"),
):
    """Split a JSONL dataset into train/valid/test."""
    try:
        rows = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        rng = random.Random(seed)
        rng.shuffle(rows)
        n_train = int(len(rows) * ratio)
        n_valid = (len(rows) - n_train) // 2
        parts = {"train": rows[:n_train], "valid": rows[n_train:n_train + n_valid], "test": rows[n_train + n_valid:]}
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for name, part in parts.items():
            with open(f"{out_dir}/{name}.jsonl", "w", encoding="utf-8") as f:
                for row in part:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        typer.echo(jdump(ok(out_dir=out_dir, **{k: len(v) for k, v in parts.items()})))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("merge")
def merge(
    files: str = typer.Argument(..., help="Comma-separated JSONL files"),
    out: str = typer.Option("data/merged.jsonl", "--out", "-o"),
):
    """Merge multiple JSONL files."""
    try:
        total = 0
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fo:
            for fn in files.split(","):
                fn = fn.strip()
                if not fn:
                    continue
                for ln in Path(fn).read_text(encoding="utf-8").splitlines():
                    if ln.strip():
                        fo.write(ln.strip() + "\n")
                        total += 1
        typer.echo(jdump(ok(rows=total, out=out)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("filter")
def filter_rows(
    path: str = typer.Argument(..., help="JSONL file"),
    key: str = typer.Option(..., "--key", "-k", help="Field key"),
    op: str = typer.Option("==", "--op", help="==,!=,>,<,>=,<=,contains"),
    value: str = typer.Option(..., "--value", "-v", help="Value"),
    out: str = typer.Option("data/filtered.jsonl", "--out", "-o"),
):
    """Filter rows by a field comparison."""
    try:
        kept = []
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            row = json.loads(ln)
            actual = row.get(key)
            match = _compare(actual, op, value)
            if match:
                kept.append(row)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        typer.echo(jdump(ok(kept=len(kept), out=out)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


def _compare(actual: Any, op: str, value: str) -> bool:
    if op == "contains":
        return value in str(actual)
    try:
        a = float(actual)
        v = float(value)
    except Exception:
        a, v = str(actual), value
    if op == "==":
        return a == v
    if op == "!=":
        return a != v
    if op == ">":
        return a > v
    if op == "<":
        return a < v
    if op == ">=":
        return a >= v
    if op == "<=":
        return a <= v
    return False


@app.command("map")
def map_rows(
    path: str = typer.Argument(..., help="JSONL file"),
    expr: str = typer.Option(..., "--expr", "-e", help="Python expression, `row` in scope, e.g. row['age'] * 2"),
    key: str = typer.Option(..., "--key", "-k", help="Output field name"),
    out: str = typer.Option("data/mapped.jsonl", "--out", "-o"),
):
    """Add a computed field to every row."""
    try:
        out_rows = []
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            row = json.loads(ln)
            row[key] = eval(expr, {"row": row, "math": math, "len": len, "str": str, "int": int, "float": float})
            out_rows.append(row)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        typer.echo(jdump(ok(rows=len(out_rows), out=out)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
@app.command("normalize")
def normalize(
    path: str = typer.Argument(..., help="JSONL/CSV file"),
    columns: str = typer.Option(..., "--columns", "-c", help="Comma-separated numeric columns"),
    method: str = typer.Option("minmax", "--method", "-m", help="minmax|zscore"),
    out: str = typer.Option("data/normalized.jsonl", "--out", "-o"),
):
    """Normalize numeric columns (min-max or z-score)."""
    try:
        rows = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        cols = [c.strip() for c in columns.split(",")]
        for col in cols:
            vals = [float(r[col]) for r in rows if col in r]
            lo, hi = min(vals), max(vals)
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            sd = math.sqrt(var) if var else 1.0
            for r in rows:
                if col in r:
                    v = float(r[col])
                    r[col] = round((v - lo) / (hi - lo), 6) if method == "minmax" and hi != lo else round((v - mean) / sd, 6)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        typer.echo(jdump(ok(rows=len(rows), method=method, out=out)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("onehot")
def onehot(
    path: str = typer.Argument(..., help="JSONL file"),
    key: str = typer.Option(..., "--key", "-k", help="Categorical field"),
    out: str = typer.Option("data/onehot.jsonl", "--out", "-o"),
):
    """One-hot encode a categorical field."""
    try:
        rows = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        cats = sorted({str(r.get(key)) for r in rows})
        for r in rows:
            for c in cats:
                r[f"{key}__{c}"] = 1 if str(r.get(key)) == c else 0
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        typer.echo(jdump(ok(rows=len(rows), categories=cats, out=out)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
@app.command("tokenize")
def tokenize(
    path: str = typer.Argument(..., help="Text file"),
    method: str = typer.Option("word", "--method", "-m", help="word|char|bpe-ish"),
    vocab_size: int = typer.Option(20000, "--vocab-size"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Save tokens as JSONL"),
):
    """Tokenize a text file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        if method == "char":
            tokens = list(text)
        elif method == "bpe-ish":
            tokens = _bpe_like(text, vocab_size)
        else:
            tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
        stats = {"tokens": len(tokens), "unique": len(set(tokens)), "method": method}
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                for t in tokens:
                    f.write(json.dumps({"token": t}) + "\n")
            stats["out"] = out
        typer.echo(jdump(ok(**stats, preview=tokens[:50])))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


def _bpe_like(text: str, vocab_size: int) -> List[str]:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    # character-level fallback merge (simple byte-pair style)
    vocab = {}
    for w in words:
        chars = list(w) + ["</w>"]
        for i in range(len(chars) - 1):
            pair = chars[i] + chars[i + 1]
            vocab[pair] = vocab.get(pair, 0) + 1
    # pick top pairs
    top = sorted(vocab.items(), key=lambda kv: -kv[1])[:max(1, vocab_size // 10)]
    tokens = []
    for w in words:
        chars = list(w) + ["</w>"]
        merged = "".join(chars)
        for pair, _ in top[:2000]:
            merged = merged.replace(pair, pair)
        tokens.append(merged)
    return tokens


# ---------------------------------------------------------------------------
# Embeddings (local, hash/tf-idf based — no external models)
# ---------------------------------------------------------------------------
@app.command("embed")
def embed(
    path: str = typer.Argument(..., help="Text file or JSONL with 'text' field"),
    dim: int = typer.Option(64, "--dim", "-d", help="Embedding dimension"),
    method: str = typer.Option("hash", "--method", "-m", help="hash|tfidf"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Save vectors JSONL"),
):
    """Create local embeddings (hash/tfidf) for documents."""
    try:
        p = Path(path)
        if p.suffix == ".jsonl":
            docs = [json.loads(ln)["text"] for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        else:
            docs = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        vecs = []
        for doc in docs:
            vec = _embed_doc(doc, dim, method, docs)
            vecs.append(vec)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                for doc, vec in zip(docs, vecs):
                    f.write(json.dumps({"text": doc, "vector": vec}) + "\n")
        typer.echo(jdump(ok(docs=len(docs), dim=dim, method=method, out=out)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


def _embed_doc(doc: str, dim: int, method: str, corpus: List[str]) -> List[float]:
    import hashlib

    vec = [0.0] * dim
    words = re.findall(r"[A-Za-z0-9']+", doc.lower())
    if method == "tfidf":
        # idf
        N = max(1, len(corpus))
        word_docs = {}
        for d in corpus:
            for w in set(re.findall(r"[A-Za-z0-9']+", d.lower())):
                word_docs[w] = word_docs.get(w, 0) + 1
        counts: Dict[str, int] = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        for w, c in counts.items():
            idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % dim
            idf = math.log((1 + N) / (1 + word_docs.get(w, 0))) + 1
            vec[idx] += c * idf
    else:
        for w in words:
            idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % dim
            vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


@app.command("similarity")
def similarity(
    a: str = typer.Option(..., "--a", help="Text A"),
    b: str = typer.Option(..., "--b", help="Text B"),
    dim: int = typer.Option(64, "--dim", "-d"),
):
    """Cosine similarity between two texts (hash embeddings)."""
    try:
        va = _embed_doc(a, dim, "hash", [a, b])
        vb = _embed_doc(b, dim, "hash", [a, b])
        dot = sum(x * y for x, y in zip(va, vb))
        typer.echo(jdump(ok(similarity=round(dot, 6))))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Pipelines (declarative step runner)
# ---------------------------------------------------------------------------
@app.command("pipeline")
def pipeline(
    path: str = typer.Argument(..., help="JSON pipeline file (list of steps)"),
    input: str = typer.Option(..., "--input", "-i", help="Input JSONL"),
    out: str = typer.Option("data/pipeline_out.jsonl", "--out", "-o"),
):
    """Run a declarative pipeline: [{\"op\":\"filter\",\"key\":\"age\",\"op2\":\">\",\"value\":\"18\"}, {\"op\":\"map\",\"key\":\"double\",\"expr\":\"row['age']*2\"}]"""
    try:
        steps = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = [json.loads(ln) for ln in Path(input).read_text(encoding="utf-8").splitlines() if ln.strip()]
        for step in steps:
            op = step.get("op")
            if op == "filter":
                rows = [r for r in rows if _compare(r.get(step.get("key")), step.get("op2", "=="), str(step.get("value", "")))]
            elif op == "map":
                key = step["key"]
                for r in rows:
                    r[key] = eval(step["expr"], {"row": r, "math": math})
            elif op == "dedupe":
                seen, kept = set(), []
                for r in rows:
                    k = str(r)
                    if k not in seen:
                        seen.add(k)
                        kept.append(r)
                rows = kept
            elif op == "drop":
                key = step["key"]
                for r in rows:
                    r.pop(key, None)
            elif op == "rename":
                for r in rows:
                    if step["from"] in r:
                        r[step["to"]] = r.pop(step["from"])
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        typer.echo(jdump(ok(steps=len(steps), rows=len(rows), out=out)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)
