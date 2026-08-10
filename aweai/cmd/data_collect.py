# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Data collection commands: scraping, crawling, import/export, cleaning,
augmentation and synthetic data generation."""

from __future__ import annotations

import csv
import json
import random
import re
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from aweai.cmd.common import err, jdump, ok

app = typer.Typer(help="Data collection: scraping, crawling, import/export, cleaning, augmentation, synthetic data")


# ---------------------------------------------------------------------------
# Scraping / crawling
# ---------------------------------------------------------------------------
@app.command("scrape")
def scrape(
    url: str = typer.Argument(..., help="URL to scrape"),
    selector: Optional[str] = typer.Option(None, "--selector", "-s", help="CSS-ish pattern filter (simple substring)"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Output file (.txt/.json)"),
):
    """Scrape a URL and extract readable text."""
    try:
        from aweai.bulk import _http_get

        html = _http_get(url)
        text = _html_to_text(html)
        if selector:
            text = "\n".join(ln for ln in text.splitlines() if selector in ln)
        if out:
            Path(out).write_text(text, encoding="utf-8")
            typer.echo(jdump(ok(url=url, path=out, chars=len(text))))
        else:
            typer.echo(text[:20000])
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("crawl")
def crawl(
    urls: str = typer.Argument(..., help="Comma-separated URLs"),
    out_dir: str = typer.Option("data/crawl", "--out-dir", "-o", help="Output directory"),
    max_pages: int = typer.Option(50, "--max-pages", "-m", help="Max pages to fetch"),
):
    """Crawl a list of seed URLs, extract text, save each page."""
    try:
        from aweai.bulk import _http_get

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fetched = 0
        pages = []
        for u in urls.split(",")[:max_pages]:
            u = u.strip()
            if not u:
                continue
            try:
                html = _http_get(u)
                text = _html_to_text(html)
                name = re.sub(r"[^A-Za-z0-9_-]+", "_", u.split("//")[-1][:60])
                (Path(out_dir) / f"{name}.txt").write_text(text, encoding="utf-8")
                pages.append({"url": u, "chars": len(text), "file": f"{name}.txt"})
                fetched += 1
            except Exception as e:
                pages.append({"url": u, "error": str(e)})
        (Path(out_dir) / "index.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(jdump(ok(pages=fetched, total=len(pages), out_dir=out_dir)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("fetch-json")
def fetch_json(
    url: str = typer.Argument(..., help="URL returning JSON"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Save to file"),
):
    """Fetch JSON from an API/URL."""
    try:
        from aweai.bulk import _http_get

        data = json.loads(_http_get(url))
        if out:
            Path(out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            typer.echo(jdump(ok(url=url, path=out, keys=list(data.keys()) if isinstance(data, dict) else len(data))))
        else:
            typer.echo(jdump(data))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("rss")
def rss(
    url: str = typer.Argument(..., help="RSS/Atom feed URL"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="Save items JSON"),
):
    """Parse an RSS/Atom feed into entries."""
    import xml.etree.ElementTree as ET

    try:
        from aweai.bulk import _http_get

        xml_text = _http_get(url)
        root = ET.fromstring(xml_text)
        items = []
        for item in root.iter():
            tag = item.tag.split("}")[-1].lower()
            if tag in ("item", "entry"):
                entry = {}
                for child in item:
                    ct = child.tag.split("}")[-1].lower()
                    if ct in ("title", "link", "description", "summary", "pubdate", "published", "guid", "id"):
                        entry[ct] = (child.text or "").strip()
                        if ct == "link" and not entry.get("link"):
                            entry["link"] = child.get("href", "")
                if entry:
                    items.append(entry)
        if out:
            Path(out).write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        typer.echo(jdump(ok(feed=url, items=len(items), out=out)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Import / export
# ---------------------------------------------------------------------------
@app.command("import-csv")
def import_csv(
    path: str = typer.Argument(..., help="CSV file"),
    out: str = typer.Option("data/imported.jsonl", "--out", "-o", help="Output JSONL"),
    text_columns: Optional[str] = typer.Option(None, "--text-columns", help="Comma-separated text columns"),
):
    """Import CSV into JSONL (one row per line)."""
    try:
        rows = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        typer.echo(jdump(ok(rows=len(rows), out=out, columns=list(rows[0].keys()) if rows else [])))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("import-json")
def import_json(
    path: str = typer.Argument(..., help="JSON file (array or object)"),
    out: str = typer.Option("data/imported.jsonl", "--out", "-o", help="Output JSONL"),
):
    """Import JSON (array) into JSONL."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        typer.echo(jdump(ok(rows=len(data), out=out)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("export-csv")
def export_csv(
    path: str = typer.Argument(..., help="JSONL input"),
    out: str = typer.Option("data/export.csv", "--out", "-o", help="Output CSV"),
):
    """Export JSONL to CSV (union of keys)."""
    try:
        rows = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not rows:
            raise ValueError("no rows")
        cols: List[str] = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
        typer.echo(jdump(ok(rows=len(rows), out=out, columns=cols)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("dedupe")
def dedupe(
    path: str = typer.Argument(..., help="JSONL file"),
    key: Optional[str] = typer.Option(None, "--key", "-k", help="Key to dedupe on (default whole line)"),
    out: str = typer.Option("data/deduped.jsonl", "--out", "-o", help="Output"),
):
    """Remove duplicate rows from JSONL."""
    try:
        seen = set()
        kept = []
        removed = 0
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            row = json.loads(ln)
            k = str(row.get(key, json.dumps(row, sort_keys=True))) if key else ln.strip()
            if k in seen:
                removed += 1
                continue
            seen.add(k)
            kept.append(row)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        typer.echo(jdump(ok(kept=len(kept), removed=removed, out=out)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("sample")
def sample(
    path: str = typer.Argument(..., help="JSONL file"),
    count: int = typer.Option(100, "--count", "-n"),
    seed: int = typer.Option(1, "--seed"),
    out: str = typer.Option("data/sample.jsonl", "--out", "-o", help="Output"),
):
    """Randomly sample rows from JSONL."""
    try:
        rows = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        picked = random.Random(seed).sample(rows, min(count, len(rows)))
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for row in picked:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        typer.echo(jdump(ok(sampled=len(picked), total=len(rows), out=out)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("shuffle")
def shuffle(
    path: str = typer.Argument(..., help="JSONL file"),
    seed: int = typer.Option(1, "--seed"),
    out: str = typer.Option("data/shuffled.jsonl", "--out", "-o", help="Output"),
):
    """Shuffle rows of a JSONL file."""
    try:
        rows = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        random.Random(seed).shuffle(rows)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        typer.echo(jdump(ok(rows=len(rows), out=out)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
@app.command("clean")
def clean(
    path: str = typer.Argument(..., help="Text file"),
    out: str = typer.Option("data/clean.txt", "--out", "-o", help="Output"),
    strip_html: bool = typer.Option(False, "--strip-html", help="Remove HTML tags"),
    lowercase: bool = typer.Option(False, "--lowercase"),
    collapse: bool = typer.Option(True, "--collapse/--no-collapse", help="Collapse whitespace"),
    remove_urls: bool = typer.Option(False, "--remove-urls"),
    remove_emails: bool = typer.Option(False, "--remove-emails"),
    min_len: int = typer.Option(0, "--min-len", help="Drop lines shorter than this"),
):
    """Clean a text file (HTML strip, whitespace, URLs, emails)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        lines = text.splitlines()
        if strip_html:
            lines = [re.sub(r"<[^>]+>", " ", ln) for ln in lines]
        if remove_urls:
            lines = [re.sub(r"https?://\S+", "", ln) for ln in lines]
        if remove_emails:
            lines = [re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "", ln) for ln in lines]
        if collapse:
            lines = [re.sub(r"\s+", " ", ln).strip() for ln in lines]
        if lowercase:
            lines = [ln.lower() for ln in lines]
        if min_len:
            lines = [ln for ln in lines if len(ln) >= min_len]
        text = "\n".join(ln for ln in lines if ln)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        typer.echo(jdump(ok(out=out, lines=len(lines), chars=len(text))))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
@app.command("augment-text")
def augment_text(
    text: str = typer.Argument(..., help="Text to augment"),
    count: int = typer.Option(3, "--count", "-n"),
    seed: int = typer.Option(1, "--seed"),
):
    """Augment text with synonym swaps, shuffles and noise (simple)."""
    try:
        rng = random.Random(seed)
        words = text.split()
        variants = []
        for _ in range(count):
            v = list(words)
            if len(v) > 2 and rng.random() < 0.7:
                i, j = rng.sample(range(len(v)), 2)
                v[i], v[j] = v[j], v[i]
            if rng.random() < 0.3:
                v.append(rng.choice(["indeed", "naturally", "of course", "additionally"]))
            variants.append(" ".join(v))
        typer.echo(jdump(ok(variants=variants)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
@app.command("synthetic")
def synthetic(
    kind: str = typer.Argument("text", help="text|jsonl|numbers|csv"),
    rows: int = typer.Option(100, "--rows", "-n"),
    out: str = typer.Option("data/synthetic.jsonl", "--out", "-o", help="Output"),
    seed: int = typer.Option(1, "--seed"),
    template: Optional[str] = typer.Option(None, "--template", help="JSONL: JSON template with {field} placeholders"),
):
    """Generate synthetic data (text/jsonl/numbers/csv)."""
    rng = random.Random(seed)
    first = ["Aram", "Lusine", "Tigran", "Anna", "David", "Nare", "Hovhannes", "Sona", "Vardan", "Mariam"]
    last = ["Petrosyan", "Grigoryan", "Sargsyan", "Hakobyan", "Harutyunyan", "Martirosyan"]
    cities = ["Yerevan", "Gyumri", "Vanadzor", "Vagharshapat", "Hrazdan", "Abovyan"]
    words = ["neural", "network", "model", "data", "train", "learn", "inference", "attention", "gradient", "token"]
    try:
        if kind == "text":
            lines = []
            for _ in range(rows):
                n = rng.randint(8, 30)
                lines.append(" ".join(rng.choice(words) for _ in range(n)))
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text("\n".join(lines), encoding="utf-8")
        elif kind == "numbers":
            lines = [str(rng.randint(0, 100)) for _ in range(rows)]
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text("\n".join(lines), encoding="utf-8")
        elif kind == "csv":
            lines = ["name,age,city,label"]
            for _ in range(rows):
                lines.append(f"{rng.choice(first)} {rng.choice(last)},{rng.randint(18, 70)},{rng.choice(cities)},{rng.randint(0, 1)}")
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text("\n".join(lines), encoding="utf-8")
        else:  # jsonl
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                for _ in range(rows):
                    if template:
                        row_text = template.format(
                            name=f"{rng.choice(first)} {rng.choice(last)}",
                            age=rng.randint(18, 70),
                            city=rng.choice(cities),
                            id=rng.randint(1000, 9999),
                            score=round(rng.uniform(0, 1), 3),
                            text=" ".join(rng.choice(words) for _ in range(rng.randint(5, 15))),
                        )
                        row = json.loads(row_text)
                    else:
                        row = {"id": rng.randint(1000, 9999), "name": f"{rng.choice(first)} {rng.choice(last)}",
                               "age": rng.randint(18, 70), "city": rng.choice(cities),
                               "label": rng.randint(0, 1), "text": " ".join(rng.choice(words) for _ in range(rng.randint(5, 15)))}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        typer.echo(jdump(ok(kind=kind, rows=rows, out=out)))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


@app.command("stats")
def stats(path: str = typer.Argument(..., help="JSONL or text file")):
    """Show basic dataset stats (rows, chars, unique)."""
    try:
        p = Path(path)
        if p.suffix == ".jsonl":
            rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            keys: Dict[str, int] = {}
            for r in rows:
                for k in r:
                    keys[k] = keys.get(k, 0) + 1
            typer.echo(jdump(ok(rows=len(rows), keys=keys, chars=p.stat().st_size)))
        else:
            text = p.read_text(encoding="utf-8")
            words = text.split()
            typer.echo(jdump(ok(lines=len(text.splitlines()), words=len(words), chars=len(text),
                                unique_words=len(set(words)))))
    except Exception as e:
        typer.echo(jdump(err(str(e))))
        raise typer.Exit(code=1)


def _html_to_text(html: str) -> str:
    """Minimal HTML to text conversion."""
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
    html = re.sub(r"<[^>]+>", " ", html)
    html = urllib.parse.unquote(html)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in html.splitlines()]
    return "\n".join(ln for ln in lines if ln)
