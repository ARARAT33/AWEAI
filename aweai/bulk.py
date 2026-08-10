# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""AWEAI bulk command engine.

Declarative command definitions used to mass-produce hundreds of useful
CLI commands without boilerplate. Each spec describes one command; the
registry turns specs into real Typer commands grouped under sub-apps.

The command space intentionally covers AI/ASI/AGI engineering: math,
strings, JSON, files, networking, time, crypto, ML helpers, text, image,
audio, video, databases, cloud, LLM utilities, RL, neuro and knowledge.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import shutil
import socket
import statistics
import string
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Small pure helpers used by many specs
# ---------------------------------------------------------------------------


def _num(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _ints(x: Any) -> List[int]:
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    s = str(x or "").strip()
    if not s:
        return []
    return [int(v) for v in re.split(r"[,\s]+", s) if v != ""]


def _floats(x: Any) -> List[float]:
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    s = str(x or "").strip()
    if not s:
        return []
    return [float(v) for v in re.split(r"[,\s]+", s) if v != ""]


def _ok(**kw: Any) -> Dict[str, Any]:
    return {"ok": True, **kw}


def _err(msg: str) -> Dict[str, Any]:
    return {"ok": False, "error": msg}


def _write(path: str, text: str) -> Dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return _ok(path=str(p), bytes=len(text.encode("utf-8")), lines=text.count("\n") + 1)


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Spec type: Dict with keys:
#   name, help, params: List[Tuple[name, default, help]], fn: Callable[[Dict], Any]
# ---------------------------------------------------------------------------
S: List[Dict[str, Any]] = []


def spec(group: str, name: str, help: str, params: List[Tuple[str, Any, str]], fn: Callable[[Dict[str, Any]], Any]) -> None:
    S.append({"group": group, "name": name, "help": help, "params": params, "fn": fn})


# ===========================================================================
# MATH group
# ===========================================================================
spec("math", "add", "Add numbers.", [("values", "1,2,3", "Comma-separated numbers")],
     lambda p: _ok(result=sum(_floats(p["values"]))))
spec("math", "sub", "Subtract numbers (first - rest).", [("values", "10,1,2", "Comma-separated numbers")],
     lambda p: _ok(result=_floats(p["values"])[0] - sum(_floats(p["values"])[1:])))
spec("math", "mul", "Multiply numbers.", [("values", "2,3,4", "Comma-separated numbers")],
     lambda p: _ok(result=math.prod(_floats(p["values"]))))
spec("math", "div", "Divide first by rest.", [("values", "100,4", "Comma-separated numbers")],
     lambda p: _ok(result=_floats(p["values"])[0] / (math.prod(_floats(p["values"])[1:]) or 1)))
spec("math", "pow", "Raise base to exponent.", [("base", 2.0, "Base"), ("exp", 10.0, "Exponent")],
     lambda p: _ok(result=p["base"] ** p["exp"]))
spec("math", "sqrt", "Square root.", [("x", 9.0, "Number")], lambda p: _ok(result=math.sqrt(p["x"])))
spec("math", "abs", "Absolute value.", [("x", -5.0, "Number")], lambda p: _ok(result=abs(p["x"])))
spec("math", "floor", "Floor of number.", [("x", 3.7, "Number")], lambda p: _ok(result=math.floor(p["x"])))
spec("math", "ceil", "Ceil of number.", [("x", 3.2, "Number")], lambda p: _ok(result=math.ceil(p["x"])))
spec("math", "round", "Round number to decimals.", [("x", 3.14159, "Number"), ("digits", 2, "Digits")],
     lambda p: _ok(result=round(p["x"], p["digits"])))
spec("math", "log", "Natural logarithm.", [("x", 2.718281828, "Number")], lambda p: _ok(result=math.log(p["x"])))
spec("math", "log10", "Base-10 logarithm.", [("x", 1000.0, "Number")], lambda p: _ok(result=math.log10(p["x"])))
spec("math", "log2", "Base-2 logarithm.", [("x", 8.0, "Number")], lambda p: _ok(result=math.log2(p["x"])))
spec("math", "exp", "Exponential e**x.", [("x", 1.0, "Number")], lambda p: _ok(result=math.exp(p["x"])))
spec("math", "sin", "Sine (radians).", [("x", 0.0, "Radians")], lambda p: _ok(result=math.sin(p["x"])))
spec("math", "cos", "Cosine (radians).", [("x", 0.0, "Radians")], lambda p: _ok(result=math.cos(p["x"])))
spec("math", "tan", "Tangent (radians).", [("x", 0.0, "Radians")], lambda p: _ok(result=math.tan(p["x"])))
spec("math", "asin", "Arcsine (result radians).", [("x", 0.5, "Number in [-1,1]")], lambda p: _ok(result=math.asin(p["x"])))
spec("math", "acos", "Arccosine (result radians).", [("x", 0.5, "Number in [-1,1]")], lambda p: _ok(result=math.acos(p["x"])))
spec("math", "atan", "Arctangent (result radians).", [("x", 1.0, "Number")], lambda p: _ok(result=math.atan(p["x"])))
spec("math", "atan2", "Arctangent of y/x.", [("y", 1.0, "Y"), ("x", 1.0, "X")], lambda p: _ok(result=math.atan2(p["y"], p["x"])))
spec("math", "sinh", "Hyperbolic sine.", [("x", 1.0, "Number")], lambda p: _ok(result=math.sinh(p["x"])))
spec("math", "cosh", "Hyperbolic cosine.", [("x", 1.0, "Number")], lambda p: _ok(result=math.cosh(p["x"])))
spec("math", "tanh", "Hyperbolic tangent.", [("x", 0.5, "Number")], lambda p: _ok(result=math.tanh(p["x"])))
spec("math", "deg", "Radians to degrees.", [("x", 1.57079632679, "Radians")], lambda p: _ok(result=math.degrees(p["x"])))
spec("math", "rad", "Degrees to radians.", [("x", 90.0, "Degrees")], lambda p: _ok(result=math.radians(p["x"])))
spec("math", "gcd", "Greatest common divisor.", [("values", "12,18", "Comma-separated ints")],
     lambda p: _ok(result=math.gcd(*_ints(p["values"]))))
spec("math", "lcm", "Least common multiple.", [("values", "4,6", "Comma-separated ints")],
     lambda p: _ok(result=math.lcm(*_ints(p["values"]))))
spec("math", "factorial", "Factorial of n.", [("n", 5, "Integer")], lambda p: _ok(result=math.factorial(int(p["n"]))))
spec("math", "comb", "Combinations n choose k.", [("n", 5, "n"), ("k", 2, "k")],
     lambda p: _ok(result=math.comb(int(p["n"]), int(p["k"]))))
spec("math", "perm", "Permutations P(n,k).", [("n", 5, "n"), ("k", 2, "k")],
     lambda p: _ok(result=math.perm(int(p["n"]), int(p["k"]))))
spec("math", "mod", "Modulo.", [("a", 17, "a"), ("b", 5, "b")], lambda p: _ok(result=int(p["a"]) % int(p["b"])))
spec("math", "min", "Minimum of numbers.", [("values", "3,1,2", "Comma-separated")],
     lambda p: _ok(result=min(_floats(p["values"]))))
spec("math", "max", "Maximum of numbers.", [("values", "3,1,2", "Comma-separated")],
     lambda p: _ok(result=max(_floats(p["values"]))))
spec("math", "clamp", "Clamp value into [lo, hi].", [("x", 5.0, "Value"), ("lo", 0.0, "Low"), ("hi", 3.0, "High")],
     lambda p: _ok(result=max(p["lo"], min(p["hi"], p["x"]))))
spec("math", "lerp", "Linear interpolation a -> b by t.", [("a", 0.0, "Start"), ("b", 10.0, "End"), ("t", 0.5, "t in [0,1]")],
     lambda p: _ok(result=p["a"] + (p["b"] - p["a"]) * p["t"]))
spec("math", "mean", "Arithmetic mean.", [("values", "1,2,3,4", "Comma-separated")],
     lambda p: _ok(result=statistics.mean(_floats(p["values"]))))
spec("math", "median", "Median.", [("values", "1,2,3,4", "Comma-separated")],
     lambda p: _ok(result=statistics.median(_floats(p["values"]))))
spec("math", "mode", "Mode.", [("values", "1,2,2,3", "Comma-separated")],
     lambda p: _ok(result=statistics.mode(_floats(p["values"]))))
spec("math", "stdev", "Sample standard deviation.", [("values", "1,2,3,4,5", "Comma-separated")],
     lambda p: _ok(result=statistics.stdev(_floats(p["values"]))))
spec("math", "variance", "Sample variance.", [("values", "1,2,3,4,5", "Comma-separated")],
     lambda p: _ok(result=statistics.variance(_floats(p["values"]))))
spec("math", "sum", "Sum of numbers.", [("values", "1,2,3", "Comma-separated")],
     lambda p: _ok(result=sum(_floats(p["values"]))))
spec("math", "prod", "Product of numbers.", [("values", "1,2,3,4", "Comma-separated")],
     lambda p: _ok(result=math.prod(_floats(p["values"]))))
spec("math", "percent", "Percent of total.", [("part", 25.0, "Part"), ("total", 200.0, "Total")],
     lambda p: _ok(result=p["part"] * 100.0 / p["total"]))
spec("math", "fib", "First n Fibonacci numbers.", [("n", 10, "Count")],
     lambda p: _ok(result=_fib(int(p["n"]))))
spec("math", "prime", "Check primality.", [("n", 97, "Integer")], lambda p: _ok(result=_is_prime(int(p["n"]))))
spec("math", "primes", "First n primes.", [("n", 10, "Count")], lambda p: _ok(result=_primes(int(p["n"]))))
spec("math", "isqrt", "Integer square root.", [("n", 50, "Integer")], lambda p: _ok(result=math.isqrt(int(p["n"]))))
spec("math", "hypot", "Hypotenuse sqrt(a^2+b^2).", [("a", 3.0, "a"), ("b", 4.0, "b")], lambda p: _ok(result=math.hypot(p["a"], p["b"])))
spec("math", "sign", "Sign of number (-1/0/1).", [("x", -7.0, "Number")], lambda p: _ok(result=(p["x"] > 0) - (p["x"] < 0)))
spec("math", "fraction", "Simplify fraction a/b.", [("a", 8, "Numerator"), ("b", 12, "Denominator")],
     lambda p: _ok(**(_simplify_frac(int(p["a"]), int(p["b"])))))
spec("math", "interest", "Compound interest A = P(1+r/n)^(nt).", [("p", 1000.0, "Principal"), ("r", 0.05, "Rate"), ("n", 12, "Times per year"), ("t", 5.0, "Years")],
     lambda p: _ok(result=p["p"] * (1 + p["r"] / p["n"]) ** (p["n"] * p["t"])))
spec("math", "angle_deg", "Convert radians to degrees (alias).", [("x", 3.14159, "Radians")], lambda p: _ok(result=math.degrees(p["x"])))


def _fib(n: int) -> List[int]:
    a, b = 0, 1
    out = []
    for _ in range(max(0, n)):
        out.append(a)
        a, b = b, a + b
    return out


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _primes(n: int) -> List[int]:
    out: List[int] = []
    cand = 2
    while len(out) < n:
        if _is_prime(cand):
            out.append(cand)
        cand += 1
    return out


def _simplify_frac(a: int, b: int) -> Dict[str, Any]:
    g = math.gcd(abs(a), abs(b))
    return {"num": a // g, "den": b // g}


# ===========================================================================
# STRING group
# ===========================================================================
spec("string", "upper", "Uppercase text.", [("text", "hello", "Input text")],
     lambda p: _ok(result=p["text"].upper()))
spec("string", "lower", "Lowercase text.", [("text", "HELLO", "Input text")],
     lambda p: _ok(result=p["text"].lower()))
spec("string", "title", "Title case.", [("text", "hello world", "Input text")],
     lambda p: _ok(result=p["text"].title()))
spec("string", "capitalize", "Capitalize first letter.", [("text", "hello world", "Input text")],
     lambda p: _ok(result=p["text"].capitalize()))
spec("string", "swapcase", "Swap letter cases.", [("text", "Hello World", "Input text")],
     lambda p: _ok(result=p["text"].swapcase()))
spec("string", "strip", "Strip whitespace.", [("text", "  hi  ", "Input text")],
     lambda p: _ok(result=p["text"].strip()))
spec("string", "lstrip", "Strip left whitespace.", [("text", "  hi", "Input text")],
     lambda p: _ok(result=p["text"].lstrip()))
spec("string", "rstrip", "Strip right whitespace.", [("text", "hi  ", "Input text")],
     lambda p: _ok(result=p["text"].rstrip()))
spec("string", "length", "Length of text.", [("text", "hello", "Input text")],
     lambda p: _ok(result=len(p["text"])))
spec("string", "reverse", "Reverse text.", [("text", "abc", "Input text")],
     lambda p: _ok(result=p["text"][::-1]))
spec("string", "split", "Split text by delimiter.", [("text", "a,b,c", "Input text"), ("delim", ",", "Delimiter")],
     lambda p: _ok(result=p["text"].split(p["delim"])))
spec("string", "join", "Join list with delimiter.", [("values", "a,b,c", "Comma-separated values"), ("delim", "-", "Delimiter")],
     lambda p: _ok(result=p["delim"].join(p["values"].split(","))))
spec("string", "replace", "Replace substring.", [("text", "hello world", "Input text"), ("old", "world", "Old"), ("new", "there", "New")],
     lambda p: _ok(result=p["text"].replace(p["old"], p["new"])))
spec("string", "find", "Find index of substring.", [("text", "hello world", "Input text"), ("sub", "world", "Substring")],
     lambda p: _ok(index=p["text"].find(p["sub"])))
spec("string", "count", "Count substring occurrences.", [("text", "ab ab ab", "Input text"), ("sub", "ab", "Substring")],
     lambda p: _ok(result=p["text"].count(p["sub"])))
spec("string", "startswith", "Check prefix.", [("text", "hello", "Input text"), ("prefix", "he", "Prefix")],
     lambda p: _ok(result=p["text"].startswith(p["prefix"])))
spec("string", "endswith", "Check suffix.", [("text", "hello", "Input text"), ("suffix", "lo", "Suffix")],
     lambda p: _ok(result=p["text"].endswith(p["suffix"])))
spec("string", "contains", "Check substring presence.", [("text", "hello world", "Input text"), ("sub", "world", "Substring")],
     lambda p: _ok(result=p["sub"] in p["text"]))
spec("string", "slice", "Slice text [start:end].", [("text", "hello world", "Input text"), ("start", 0, "Start"), ("end", 5, "End")],
     lambda p: _ok(result=p["text"][int(p["start"]):int(p["end"])]))
spec("string", "repeat", "Repeat text n times.", [("text", "ab", "Input text"), ("n", 3, "Times")],
     lambda p: _ok(result=p["text"] * int(p["n"])))
spec("string", "pad", "Pad text to width with char.", [("text", "42", "Input text"), ("width", 6, "Width"), ("char", "0", "Pad char")],
     lambda p: _ok(result=p["text"].rjust(int(p["width"]), p["char"])))
spec("string", "truncate", "Truncate text to n chars + ellipsis.", [("text", "a very long sentence", "Input text"), ("n", 7, "Max chars")],
     lambda p: _ok(result=(p["text"][: max(0, int(p["n"]))] + ("..." if len(p["text"]) > int(p["n"]) else ""))))
spec("string", "words", "Split into words.", [("text", "hello brave world", "Input text")],
     lambda p: _ok(result=p["text"].split()))
spec("string", "word_count", "Count words.", [("text", "one two three", "Input text")],
     lambda p: _ok(result=len(p["text"].split())))
spec("string", "chars", "List characters.", [("text", "abc", "Input text")],
     lambda p: _ok(result=list(p["text"])))
spec("string", "unique_chars", "Unique characters.", [("text", "abca", "Input text")],
     lambda p: _ok(result=sorted(set(p["text"]))))
spec("string", "alnum", "Keep alphanumeric only.", [("text", "a b!c1", "Input text")],
     lambda p: _ok(result="".join(ch for ch in p["text"] if ch.isalnum())))
spec("string", "digits", "Extract digits.", [("text", "abc123def456", "Input text")],
     lambda p: _ok(result="".join(ch for ch in p["text"] if ch.isdigit())))
spec("string", "letters", "Extract letters.", [("text", "a1b2c3", "Input text")],
     lambda p: _ok(result="".join(ch for ch in p["text"] if ch.isalpha())))
spec("string", "is_digit", "Check all digits.", [("text", "123", "Input text")],
     lambda p: _ok(result=p["text"].isdigit()))
spec("string", "is_alpha", "Check all letters.", [("text", "abc", "Input text")],
     lambda p: _ok(result=p["text"].isalpha()))
spec("string", "is_alnum", "Check alphanumeric.", [("text", "abc123", "Input text")],
     lambda p: _ok(result=p["text"].isalnum()))
spec("string", "is_lower", "Check lowercase.", [("text", "abc", "Input text")],
     lambda p: _ok(result=p["text"].islower()))
spec("string", "is_upper", "Check uppercase.", [("text", "ABC", "Input text")],
     lambda p: _ok(result=p["text"].isupper()))
spec("string", "is_space", "Check whitespace.", [("text", "   ", "Input text")],
     lambda p: _ok(result=p["text"].isspace()))
spec("string", "is_palindrome", "Check palindrome.", [("text", "racecar", "Input text")],
     lambda p: _ok(result=p["text"].lower().replace(" ", "") == p["text"][::-1].lower().replace(" ", "")))
spec("string", "slugify", "Make URL slug.", [("text", "Hello World! Foo", "Input text")],
     lambda p: _ok(result=re.sub(r"[^a-z0-9]+", "-", p["text"].lower()).strip("-")))
spec("string", "camel", "To camelCase.", [("text", "hello world", "Input text")],
     lambda p: _ok(result=(lambda w: w[0] + "".join(x.title() for x in w[1:]))(re.split(r"[^a-zA-Z0-9]+", p["text"].lower()))))
spec("string", "snake", "To snake_case.", [("text", "Hello World", "Input text")],
     lambda p: _ok(result=re.sub(r"[^a-z0-9]+", "_", p["text"].lower()).strip("_")))
spec("string", "kebab", "To kebab-case.", [("text", "Hello World", "Input text")],
     lambda p: _ok(result=re.sub(r"[^a-z0-9]+", "-", p["text"].lower()).strip("-")))
spec("string", "constant", "To CONSTANT_CASE.", [("text", "Hello World", "Input text")],
     lambda p: _ok(result=re.sub(r"[^A-Z0-9]+", "_", p["text"].upper()).strip("_")))
spec("string", "initials", "Initials of words.", [("text", "Artificial General Intelligence", "Input text")],
     lambda p: _ok(result="".join(w[0].upper() for w in p["text"].split() if w)))
spec("string", "leetspeak", "Convert to leetspeak.", [("text", "hello", "Input text")],
     lambda p: _ok(result=p["text"].translate(str.maketrans({"e": "3", "o": "0", "l": "1", "a": "4", "s": "5", "t": "7", "i": "!"}))))
spec("string", "ascii", "ASCII codes of text.", [("text", "abc", "Input text")],
     lambda p: _ok(result=[ord(c) for c in p["text"]]))
spec("string", "chr", "Character from code.", [("code", 65, "ASCII code")], lambda p: _ok(result=chr(int(p["code"]))))
spec("string", "ord", "Code of first char.", [("text", "A", "Input text")], lambda p: _ok(result=ord(p["text"][0])))
spec("string", "sha256", "SHA-256 hash of text.", [("text", "hello", "Input text")],
     lambda p: _ok(result=_sha256(p["text"])))
spec("string", "md5", "MD5 hash of text.", [("text", "hello", "Input text")],
     lambda p: _ok(result=hashlib.md5(p["text"].encode()).hexdigest()))
spec("string", "sha1", "SHA-1 hash of text.", [("text", "hello", "Input text")],
     lambda p: _ok(result=hashlib.sha1(p["text"].encode()).hexdigest()))
spec("string", "base64_encode", "Base64 encode.", [("text", "hello", "Input text")],
     lambda p: _ok(result=__import__("base64").b64encode(p["text"].encode()).decode()))
spec("string", "base64_decode", "Base64 decode.", [("text", "aGVsbG8=", "Base64 text")],
     lambda p: _ok(result=__import__("base64").b64decode(p["text"]).decode(errors="replace")))
spec("string", "url_encode", "URL-encode.", [("text", "a b&c=d", "Input text")],
     lambda p: _ok(result=urllib.parse.quote(p["text"])))
spec("string", "url_decode", "URL-decode.", [("text", "a%20b%26c%3Dd", "Input text")],
     lambda p: _ok(result=urllib.parse.unquote(p["text"])))
spec("string", "html_escape", "Escape HTML entities.", [("text", "<b>&</b>", "Input text")],
     lambda p: _ok(result=p["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")))
spec("string", "html_unescape", "Unescape HTML entities.", [("text", "&lt;b&gt;&amp;&lt;/b&gt;", "Input text")],
     lambda p: _ok(result=p["text"].replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("&quot;", '"')))
spec("string", "rot13", "ROT13 cipher.", [("text", "hello", "Input text")],
     lambda p: _ok(result=p["text"].translate(str.maketrans(
         "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
         "nopqrstuvwxyzabcdefghijklmNOPQRSTUVWXYZABCDEFGHIJKLM"))))
spec("string", "anagram", "Sort letters (anagram key).", [("text", "listen", "Input text")],
     lambda p: _ok(result="".join(sorted(p["text"]))))
spec("string", "shuffle", "Shuffle characters.", [("text", "abcdef", "Input text"), ("seed", 1, "Random seed")],
     lambda p: _ok(result=(lambda L: (random.Random(int(p["seed"])).shuffle(L), "".join(L))[1])(list(p["text"]))))


# ===========================================================================
# JSON / data group
# ===========================================================================
spec("json", "validate", "Validate JSON string.", [("text", '{"a":1}', "JSON text")],
     lambda p: _json_validate(p["text"]))
spec("json", "parse", "Parse JSON to pretty dict.", [("text", '{"a":1,"b":[1,2]}', "JSON text")],
     lambda p: _ok(result=json.loads(p["text"])))
spec("json", "stringify", "Serialize object to JSON.", [("text", '{"a":1}', "JSON text"), ("indent", 2, "Indent")],
     lambda p: _ok(result=json.dumps(json.loads(p["text"]), indent=int(p["indent"]), ensure_ascii=False)))
spec("json", "pretty", "Pretty-print JSON.", [("text", '{"a":1,"b":2}', "JSON text"), ("indent", 2, "Indent")],
     lambda p: _ok(result=json.dumps(json.loads(p["text"]), indent=int(p["indent"]), ensure_ascii=False)))
spec("json", "minify", "Minify JSON.", [("text", '{ "a" : 1 }', "JSON text")],
     lambda p: _ok(result=json.dumps(json.loads(p["text"]), separators=(",", ":"), ensure_ascii=False)))
spec("json", "keys", "Top-level keys of JSON object.", [("text", '{"a":1,"b":2}', "JSON text")],
     lambda p: _ok(result=list(json.loads(p["text"]).keys())))
spec("json", "values", "Top-level values of JSON object.", [("text", '{"a":1,"b":2}', "JSON text")],
     lambda p: _ok(result=list(json.loads(p["text"]).values())))
spec("json", "merge", "Merge two JSON objects.", [("a", '{"x":1}', "First JSON"), ("b", '{"y":2}', "Second JSON")],
     lambda p: _ok(result={**json.loads(p["a"]), **json.loads(p["b"])}))
spec("json", "get", "Get dotted path from JSON.", [("text", '{"a":{"b":42}}', "JSON text"), ("path", "a.b", "Dotted path")],
     lambda p: _ok(result=_json_get(json.loads(p["text"]), p["path"])))
spec("json", "set", "Set dotted path in JSON.", [("text", '{"a":{"b":1}}', "JSON text"), ("path", "a.c", "Dotted path"), ("value", "hello", "Value")],
     lambda p: _ok(result=_json_set(json.loads(p["text"]), p["path"], p["value"])))
spec("json", "type", "Type of JSON value.", [("text", "42", "JSON text")],
     lambda p: _ok(result=type(json.loads(p["text"])).__name__))
spec("json", "length", "Length of JSON array/object.", [("text", "[1,2,3]", "JSON text")],
     lambda p: _ok(result=len(json.loads(p["text"]))))
spec("json", "flatten", "Flatten nested JSON to dotted keys.", [("text", '{"a":{"b":1},"c":2}', "JSON text")],
     lambda p: _ok(result=_json_flatten(json.loads(p["text"]))))
spec("json", "sort_keys", "Sort JSON keys alphabetically.", [("text", '{"b":2,"a":1}', "JSON text")],
     lambda p: _ok(result=json.dumps(json.loads(p["text"]), sort_keys=True, indent=2, ensure_ascii=False)))
spec("json", "unique", "Unique values of JSON array.", [("text", "[1,2,2,3]", "JSON array")],
     lambda p: _ok(result=list(dict.fromkeys(json.loads(p["text"])))))
spec("json", "sort", "Sort JSON array of numbers.", [("text", "[3,1,2]", "JSON array"), ("reverse", False, "Descending")],
     lambda p: _ok(result=sorted(json.loads(p["text"]), reverse=bool(p["reverse"]))))


def _json_get(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return None
    return cur


def _json_validate(text: str) -> Dict[str, Any]:
    try:
        json.loads(text)
        return _ok(valid=True)
    except Exception as e:
        return _ok(valid=False, error=str(e))


def _json_set(obj: Any, path: str, value: Any) -> Any:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        nxt = cur.get(part) if isinstance(cur, dict) else None
        if not isinstance(nxt, (dict, list)):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value
    return obj


def _json_flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                out.update(_json_flatten(v, key))
            else:
                out[key] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            if isinstance(v, (dict, list)):
                out.update(_json_flatten(v, key))
            else:
                out[key] = v
    else:
        out[prefix] = obj
    return out


# ===========================================================================
# FILE group
# ===========================================================================
spec("file", "read", "Read file content.", [("path", "data.txt", "File path")],
     lambda p: _ok(content=_read(p["path"])))
spec("file", "write", "Write text to file.", [("path", "out.txt", "File path"), ("text", "hello", "Content")],
     lambda p: _write(p["path"], p["text"]))
spec("file", "append", "Append text to file.", [("path", "out.txt", "File path"), ("text", "more", "Content")],
     lambda p: (Path(p["path"]).parent.mkdir(parents=True, exist_ok=True),
                Path(p["path"]).open("a", encoding="utf-8").write(p["text"] + "\n"),
                _ok(path=p["path"]))[2])
spec("file", "exists", "Check file exists.", [("path", "data.txt", "File path")],
     lambda p: _ok(exists=Path(p["path"]).exists()))
spec("file", "size", "File size in bytes.", [("path", "data.txt", "File path")],
     lambda p: _ok(size=Path(p["path"]).stat().st_size))
spec("file", "lines", "Count lines in file.", [("path", "data.txt", "File path")],
     lambda p: _ok(lines=len(_read(p["path"]).splitlines())))
spec("file", "words", "Count words in file.", [("path", "data.txt", "File path")],
     lambda p: _ok(words=len(_read(p["path"]).split())))
spec("file", "bytes", "Count bytes in file.", [("path", "data.txt", "File path")],
     lambda p: _ok(bytes=Path(p["path"]).stat().st_size))
spec("file", "type", "File type by magic bytes.", [("path", "data.txt", "File path")],
     lambda p: _ok(type=_file_magic(p["path"])))
spec("file", "copy", "Copy file.", [("src", "a.txt", "Source"), ("dst", "b.txt", "Destination")],
     lambda p: (Path(p["dst"]).parent.mkdir(parents=True, exist_ok=True), shutil.copy2(p["src"], p["dst"]), _ok(dst=p["dst"]))[2])
spec("file", "move", "Move file.", [("src", "a.txt", "Source"), ("dst", "b.txt", "Destination")],
     lambda p: (Path(p["dst"]).parent.mkdir(parents=True, exist_ok=True), shutil.move(p["src"], p["dst"]), _ok(dst=p["dst"]))[2])
spec("file", "delete", "Delete file.", [("path", "tmp.txt", "File path")],
     lambda p: (Path(p["path"]).unlink(missing_ok=True), _ok(deleted=p["path"]))[1])
spec("file", "touch", "Create empty file / update mtime.", [("path", "new.txt", "File path")],
     lambda p: (Path(p["path"]).parent.mkdir(parents=True, exist_ok=True), Path(p["path"]).touch(), _ok(path=p["path"]))[2])
spec("file", "mkdir", "Create directory (recursive).", [("path", "dir/sub", "Directory path")],
     lambda p: (Path(p["path"]).mkdir(parents=True, exist_ok=True), _ok(path=p["path"]))[1])
spec("file", "list", "List directory entries.", [("path", ".", "Directory path")],
     lambda p: _ok(entries=sorted(os.listdir(p["path"]))))
spec("file", "tree", "Recursive file tree.", [("path", ".", "Directory path"), ("depth", 2, "Max depth")],
     lambda p: _ok(tree=_tree(p["path"], int(p["depth"]))))
spec("file", "glob", "Find files by pattern.", [("pattern", "*.py", "Glob pattern"), ("path", ".", "Base dir")],
     lambda p: _ok(files=[str(x) for x in sorted(Path(p["path"]).glob(p["pattern"]))]))
spec("file", "find", "Find files by name.", [("name", "readme.md", "File name"), ("path", ".", "Base dir")],
     lambda p: _ok(files=[str(x) for x in sorted(Path(p["path"]).rglob(p["name"]))]))
spec("file", "ext", "File extension.", [("path", "archive.tar.gz", "File path")],
     lambda p: _ok(ext=Path(p["path"]).suffix))
spec("file", "basename", "Base name of path.", [("path", "/a/b/c.txt", "File path")],
     lambda p: _ok(result=os.path.basename(p["path"])))
spec("file", "dirname", "Directory of path.", [("path", "/a/b/c.txt", "File path")],
     lambda p: _ok(result=os.path.dirname(p["path"])))
spec("file", "join_path", "Join path parts.", [("parts", "a,b,c", "Comma-separated parts")],
     lambda p: _ok(result=str(Path(*p["parts"].split(",")))))
spec("file", "abs_path", "Absolute path.", [("path", "data.txt", "File path")],
     lambda p: _ok(result=str(Path(p["path"]).resolve())))
spec("file", "is_dir", "Check directory.", [("path", ".", "Path")], lambda p: _ok(result=Path(p["path"]).is_dir()))
spec("file", "is_file", "Check file.", [("path", "data.txt", "Path")], lambda p: _ok(result=Path(p["path"]).is_file()))
spec("file", "tail", "Last n lines of file.", [("path", "data.txt", "File path"), ("n", 10, "Lines")],
     lambda p: _ok(lines=_read(p["path"]).splitlines()[-int(p["n"]):]))
spec("file", "head", "First n lines of file.", [("path", "data.txt", "File path"), ("n", 10, "Lines")],
     lambda p: _ok(lines=_read(p["path"]).splitlines()[:int(p["n"]):]))
spec("file", "grep", "Search lines matching pattern.", [("path", "data.txt", "File path"), ("pattern", "hello", "Regex")],
     lambda p: _ok(matches=[ln for ln in _read(p["path"]).splitlines() if re.search(p["pattern"], ln)]))
spec("file", "hash", "SHA-256 of file.", [("path", "data.txt", "File path")],
     lambda p: _ok(sha256=hashlib.sha256(Path(p["path"]).read_bytes()).hexdigest()))
spec("file", "checksum", "MD5 of file.", [("path", "data.txt", "File path")],
     lambda p: _ok(md5=hashlib.md5(Path(p["path"]).read_bytes()).hexdigest()))
spec("file", "zip", "Create zip archive.", [("path", "out.zip", "Zip path"), ("sources", "a.txt,b.txt", "Comma-separated files")],
     lambda p: _zip_files(p["path"], p["sources"].split(",")))
spec("file", "unzip", "Extract zip archive.", [("path", "out.zip", "Zip path"), ("dest", "extracted", "Destination dir")],
     lambda p: _unzip(p["path"], p["dest"]))
spec("file", "gzip", "Gzip-compress file.", [("path", "data.txt", "File path")],
     lambda p: _gzip_file(p["path"]))
spec("file", "gunzip", "Gzip-decompress file.", [("path", "data.txt.gz", "File path")],
     lambda p: _gunzip_file(p["path"]))
spec("file", "rename", "Rename file.", [("src", "a.txt", "Old name"), ("dst", "b.txt", "New name")],
     lambda p: (os.rename(p["src"], p["dst"]), _ok(dst=p["dst"]))[1])
spec("file", "du", "Directory size.", [("path", ".", "Directory path")],
     lambda p: _ok(bytes=sum(f.stat().st_size for f in Path(p["path"]).rglob("*") if f.is_file())))
spec("file", "mime", "Guess MIME type.", [("path", "data.txt", "File path")],
     lambda p: _ok(mime=__import__("mimetypes").guess_type(p["path"])[0]))


def _file_magic(path: str) -> str:
    try:
        head = Path(path).read_bytes()[:8]
    except Exception:
        return "unknown"
    if head[:4] == b"\x89PNG":
        return "png"
    if head[:3] == b"GIF":
        return "gif"
    if head[:2] == b"\xff\xd8":
        return "jpeg"
    if head[:4] == b"%PDF":
        return "pdf"
    if head[:4] == b"PK\x03\x04":
        return "zip"
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    if head[:4] == b"RIFF":
        return "riff"
    return "text" if all(32 <= b < 127 or b in (9, 10, 13) for b in head[:32] if b) else "binary"


def _tree(path: str, depth: int, prefix: str = "") -> List[str]:
    out: List[str] = []
    try:
        entries = sorted(Path(path).iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except Exception:
        return out
    for i, p in enumerate(entries):
        last = i == len(entries) - 1
        out.append(prefix + ("\u2514\u2500\u2500 " if last else "\u251c\u2500\u2500 ") + p.name + ("/" if p.is_dir() else ""))
        if p.is_dir() and depth > 0:
            out.extend(_tree(str(p), depth - 1, prefix + ("    " if last else "\u2502   ")))
    return out


def _zip_files(zip_path: str, sources: List[str]) -> Dict[str, Any]:
    import zipfile

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src in sources:
            sp = Path(src)
            if sp.is_file():
                zf.write(sp, sp.name)
            elif sp.is_dir():
                for f in sp.rglob("*"):
                    if f.is_file():
                        zf.write(f, str(f))
    return _ok(archive=zip_path, files=len(sources))


def _unzip(zip_path: str, dest: str) -> Dict[str, Any]:
    import zipfile

    Path(dest).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return _ok(dest=dest, files=len(zipfile.ZipFile(zip_path).namelist()))


def _gzip_file(path: str) -> Dict[str, Any]:
    data = Path(path).read_bytes()
    out = path + ".gz"
    Path(out).write_bytes(__import__("gzip").compress(data))
    return _ok(path=out, bytes=len(data), compressed=len(Path(out).read_bytes()))


def _gunzip_file(path: str) -> Dict[str, Any]:
    data = __import__("gzip").decompress(Path(path).read_bytes())
    out = path[:-3] if path.endswith(".gz") else path + ".out"
    Path(out).write_bytes(data)
    return _ok(path=out, bytes=len(data))


# ===========================================================================
# NETWORK group
# ===========================================================================
spec("net", "ip", "Public IP address.", [], lambda p: _ok(ip=_http_get("https://api.ipify.org").strip()))
spec("net", "hostname", "Hostname of this machine.", [], lambda p: _ok(hostname=socket.gethostname()))
spec("net", "resolve", "Resolve hostname to IP.", [("host", "example.com", "Hostname")],
     lambda p: _ok(ip=socket.gethostbyname(p["host"])))
spec("net", "reverse_dns", "Reverse DNS lookup.", [("ip", "8.8.8.8", "IP address")],
     lambda p: _ok(host=socket.gethostbyaddr(p["ip"])[0]))
spec("net", "port_open", "Check TCP port open.", [("host", "example.com", "Host"), ("port", 80, "Port")],
     lambda p: _ok(open=_port_open(p["host"], int(p["port"]))))
spec("net", "http_get", "GET a URL and return text.", [("url", "https://example.com", "URL"), ("timeout", 15, "Timeout seconds")],
     lambda p: _ok(status=200, body=_http_get(p["url"], int(p["timeout"]))))
spec("net", "http_head", "HEAD request (headers).", [("url", "https://example.com", "URL"), ("timeout", 15, "Timeout")],
     lambda p: _ok(headers=_http_head(p["url"], int(p["timeout"]))))
spec("net", "download", "Download URL to file.", [("url", "https://example.com", "URL"), ("out", "page.html", "Output path"), ("timeout", 60, "Timeout")],
     lambda p: _net_download(p["url"], p["out"], int(p["timeout"])))
spec("net", "ping", "Ping host (ICMP via system ping).", [("host", "example.com", "Host"), ("count", 3, "Count")],
     lambda p: _ok(output=_cmd(f"ping -c {int(p['count'])} -W 3 {p['host']}") if sys.platform != "win32" else _cmd(f"ping -n {int(p['count'])} {p['host']}")))
spec("net", "dns", "DNS TXT/A records via socket.", [("host", "example.com", "Host")],
     lambda p: _ok(ip=socket.gethostbyname_ex(p["host"])))
spec("net", "whois", "Whois lookup (if whois installed).", [("domain", "example.com", "Domain")],
     lambda p: _ok(output=_cmd(f"whois {p['domain']}") or "whois not available"))
spec("net", "url_parts", "Parse URL components.", [("url", "https://user:pass@example.com:8080/path?q=1#frag", "URL")],
     lambda p: _ok(parts={k: v for k, v in urllib.parse.urlsplit(p["url"])._asdict().items()}))
spec("net", "shorten", "Shorten URL (tinyurl).", [("url", "https://example.com", "URL")],
     lambda p: _ok(short=_http_get(f"https://tinyurl.com/api-create.php?url={urllib.parse.quote(p['url'])}").strip()))


def _http_get(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "AWEAI-CLI/4.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _http_head(url: str, timeout: int = 15) -> Dict[str, str]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "AWEAI-CLI/4.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return {k: v for k, v in r.headers.items()}


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except Exception:
        return False


def _net_download(url: str, out: str, timeout: int) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Agent": "AWEAI-CLI/4.0"})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    Path(out).write_bytes(data)
    return _ok(path=out, bytes=len(data))


def _cmd(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


# ===========================================================================
# TIME group
# ===========================================================================
spec("time", "now", "Current timestamp.", [], lambda p: _ok(iso=_now_iso(), unix=time.time()))
spec("time", "date", "Current date.", [], lambda p: _ok(date=_dt.date.today().isoformat()))
spec("time", "utc", "Current UTC time.", [], lambda p: _ok(utc=_dt.datetime.now(_dt.timezone.utc).isoformat()))
spec("time", "unix", "Current Unix epoch.", [], lambda p: _ok(unix=int(time.time())))
spec("time", "from_unix", "Convert Unix epoch to ISO.", [("unix", 1700000000, "Epoch seconds")],
     lambda p: _ok(iso=_dt.datetime.fromtimestamp(float(p["unix"])).isoformat()))
spec("time", "to_unix", "Convert ISO time to Unix.", [("iso", "2026-08-10T00:00:00", "ISO datetime")],
     lambda p: _ok(unix=int(_dt.datetime.fromisoformat(p["iso"]).timestamp())))
spec("time", "sleep", "Sleep n seconds.", [("seconds", 1, "Seconds")],
     lambda p: (time.sleep(float(p["seconds"])), _ok(slept=float(p["seconds"])))[1])
spec("time", "diff", "Difference between two ISO times.", [("a", "2026-08-10T00:00:00", "Start ISO"), ("b", "2026-08-10T02:30:00", "End ISO")],
     lambda p: _ok(seconds=(_dt.datetime.fromisoformat(p["b"]) - _dt.datetime.fromisoformat(p["a"])).total_seconds()))
spec("time", "add", "Add seconds to ISO time.", [("iso", "2026-08-10T00:00:00", "ISO datetime"), ("seconds", 3600, "Seconds")],
     lambda p: _ok(result=(_dt.datetime.fromisoformat(p["iso"]) + _dt.timedelta(seconds=float(p["seconds"]))).isoformat()))
spec("time", "age", "Age in years from birthdate.", [("birth", "1990-01-01", "Birth date YYYY-MM-DD")],
     lambda p: _ok(years=_dt.date.today().year - _dt.date.fromisoformat(p["birth"]).year))
spec("time", "weekday", "Weekday of date.", [("date", "2026-08-10", "Date YYYY-MM-DD")],
     lambda p: _ok(weekday=_dt.date.fromisoformat(p["date"]).strftime("%A")))
spec("time", "days_until", "Days until date.", [("date", "2027-01-01", "Target date")],
     lambda p: _ok(days=(_dt.date.fromisoformat(p["date"]) - _dt.date.today()).days))
spec("time", "format", "Format ISO time with strftime.", [("iso", "2026-08-10T13:00:00", "ISO datetime"), ("fmt", "%Y/%m/%d %H:%M", "Format")],
     lambda p: _ok(result=_dt.datetime.fromisoformat(p["iso"]).strftime(p["fmt"])))
spec("time", "timezone", "Current timezone info.", [], lambda p: _ok(tz=_dt.datetime.now().astimezone().tzname(), offset=str(_dt.datetime.now().astimezone().utcoffset())))
spec("time", "timer", "Measure command wall time (demo).", [("seconds", 1, "Seconds to wait")],
     lambda p: (lambda t0: (time.sleep(float(p["seconds"])), _ok(elapsed_s=round(time.time() - t0, 3)))[1])(time.time()))
spec("time", "stopwatch", "Elapsed since given epoch.", [("start", 1700000000, "Start epoch")],
     lambda p: _ok(elapsed_s=time.time() - float(p["start"])))


# ===========================================================================
# CRYPTO / random group
# ===========================================================================
spec("crypto", "sha256", "SHA-256 of text.", [("text", "hello", "Input")], lambda p: _ok(result=_sha256(p["text"])))
spec("crypto", "sha512", "SHA-512 of text.", [("text", "hello", "Input")], lambda p: _ok(result=hashlib.sha512(p["text"].encode()).hexdigest()))
spec("crypto", "md5", "MD5 of text.", [("text", "hello", "Input")], lambda p: _ok(result=hashlib.md5(p["text"].encode()).hexdigest()))
spec("crypto", "sha1", "SHA-1 of text.", [("text", "hello", "Input")], lambda p: _ok(result=hashlib.sha1(p["text"].encode()).hexdigest()))
spec("crypto", "hmac", "HMAC-SHA256.", [("key", "secret", "Key"), ("text", "message", "Message")],
     lambda p: _ok(result=__import__("hmac").new(p["key"].encode(), p["text"].encode(), hashlib.sha256).hexdigest()))
spec("crypto", "uuid", "Generate UUID.", [("version", 4, "UUID version 1/4")],
     lambda p: _ok(uuid=str(uuid.uuid4() if int(p["version"]) == 4 else uuid.uuid1())))
spec("crypto", "uuid_many", "Generate many UUIDs.", [("count", 5, "Count")],
     lambda p: _ok(uuids=[str(uuid.uuid4()) for _ in range(int(p["count"]))]))
spec("crypto", "rand_int", "Random integer in [lo, hi].", [("lo", 1, "Low"), ("hi", 100, "High"), ("seed", 1, "Seed")],
     lambda p: _ok(result=random.Random(int(p["seed"])).randint(int(p["lo"]), int(p["hi"]))))
spec("crypto", "rand_float", "Random float in [0,1).", [("seed", 1, "Seed")],
     lambda p: _ok(result=random.Random(int(p["seed"])).random()))
spec("crypto", "rand_bytes", "Random hex bytes.", [("n", 16, "Bytes"), ("seed", 1, "Seed")],
     lambda p: _ok(result=random.Random(int(p["seed"])).randbytes(int(p["n"])).hex()))
spec("crypto", "rand_choice", "Random choice from list.", [("values", "a,b,c", "Comma-separated"), ("seed", 1, "Seed")],
     lambda p: _ok(result=random.Random(int(p["seed"])).choice(p["values"].split(","))))
spec("crypto", "rand_password", "Random password.", [("length", 16, "Length"), ("seed", 1, "Seed")],
     lambda p: _ok(password="".join(random.Random(int(p["seed"])).choice(string.ascii_letters + string.digits + "!@#$%^&*") for _ in range(int(p["length"])))))
spec("crypto", "xor", "XOR bytes with key.", [("text", "hello", "Plaintext"), ("key", "k", "Key")],
     lambda p: _ok(result="".join(chr(ord(c) ^ ord(p["key"][i % len(p["key"])])) for i, c in enumerate(p["text"]))))
spec("crypto", "caesar", "Caesar cipher shift.", [("text", "hello", "Plaintext"), ("shift", 3, "Shift")],
     lambda p: _ok(result="".join(chr((ord(c) - 97 + int(p["shift"])) % 26 + 97) if "a" <= c <= "z" else chr((ord(c) - 65 + int(p["shift"])) % 26 + 65) if "A" <= c <= "Z" else c for c in p["text"])))
spec("crypto", "crc32", "CRC32 checksum.", [("text", "hello", "Input")],
     lambda p: _ok(result=zlib.crc32(p["text"].encode())))
spec("crypto", "entropy", "Estimate Shannon entropy (bits/char).", [("text", "aaaaaaaa", "Input")],
     lambda p: _ok(bits_per_char=_entropy(p["text"])))
spec("crypto", "token", "URL-safe random token.", [("bytes", 32, "Bytes"), ("seed", 1, "Seed")],
     lambda p: _ok(token=__import__("base64").urlsafe_b64encode(random.Random(int(p["seed"])).randbytes(int(p["bytes"]))).rstrip(b"=").decode()))


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    import collections

    cnt = collections.Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in cnt.values())


# ===========================================================================
# ML helpers group
# ===========================================================================
spec("ml", "accuracy", "Classification accuracy.", [("y_true", "0,1,1,0", "True labels"), ("y_pred", "0,1,0,0", "Pred labels")],
     lambda p: _ok(accuracy=(lambda a, b: sum(x == y for x, y in zip(a, b)) / len(a))(_ints(p["y_true"]), _ints(p["y_pred"]))))
spec("ml", "mse", "Mean squared error.", [("y_true", "1,2,3", "True"), ("y_pred", "1.5,2,3.5", "Pred")],
     lambda p: _ok(mse=statistics.mean((a - b) ** 2 for a, b in zip(_floats(p["y_true"]), _floats(p["y_pred"])))))
spec("ml", "mae", "Mean absolute error.", [("y_true", "1,2,3", "True"), ("y_pred", "1.5,2,3.5", "Pred")],
     lambda p: _ok(mae=statistics.mean(abs(a - b) for a, b in zip(_floats(p["y_true"]), _floats(p["y_pred"])))))
spec("ml", "rmse", "Root mean squared error.", [("y_true", "1,2,3", "True"), ("y_pred", "1.5,2,3.5", "Pred")],
     lambda p: _ok(rmse=math.sqrt(statistics.mean((a - b) ** 2 for a, b in zip(_floats(p["y_true"]), _floats(p["y_pred"]))))))
spec("ml", "r2", "R-squared score.", [("y_true", "1,2,3,4", "True"), ("y_pred", "1.1,2.1,2.9,4.1", "Pred")],
     lambda p: _ok(r2=_r2(_floats(p["y_true"]), _floats(p["y_pred"]))))
spec("ml", "confusion", "Confusion matrix counts.", [("y_true", "0,0,1,1", "True"), ("y_pred", "0,1,1,1", "Pred")],
     lambda p: _ok(matrix=_confusion(_ints(p["y_true"]), _ints(p["y_pred"]))))
spec("ml", "precision", "Precision (macro).", [("y_true", "0,0,1,1", "True"), ("y_pred", "0,1,1,1", "Pred")],
     lambda p: _ok(precision=_precision_recall_f1(_ints(p["y_true"]), _ints(p["y_pred"]))["precision"]))
spec("ml", "recall", "Recall (macro).", [("y_true", "0,0,1,1", "True"), ("y_pred", "0,1,1,1", "Pred")],
     lambda p: _ok(recall=_precision_recall_f1(_ints(p["y_true"]), _ints(p["y_pred"]))["recall"]))
spec("ml", "f1", "F1 score (macro).", [("y_true", "0,0,1,1", "True"), ("y_pred", "0,1,1,1", "Pred")],
     lambda p: _ok(f1=_precision_recall_f1(_ints(p["y_true"]), _ints(p["y_pred"]))["f1"]))
spec("ml", "normalize", "Min-max normalize list.", [("values", "1,2,3,4,5", "Numbers")],
     lambda p: _ok(normalized=_minmax(_floats(p["values"]))))
spec("ml", "standardize", "Z-score standardize list.", [("values", "1,2,3,4,5", "Numbers")],
     lambda p: _ok(standardized=_zscore(_floats(p["values"]))))
spec("ml", "onehot", "One-hot encode labels.", [("labels", "a,b,a,c", "Comma-separated")],
     lambda p: _ok(onehot=_onehot(p["labels"].split(","))))
spec("ml", "label_encode", "Encode labels to ints.", [("labels", "a,b,a,c", "Comma-separated")],
     lambda p: _ok(encoded=_label_encode(p["labels"].split(","))))
spec("ml", "correlation", "Pearson correlation.", [("x", "1,2,3,4,5", "X values"), ("y", "2,4,6,8,10", "Y values")],
     lambda p: _ok(r=_corr(_floats(p["x"]), _floats(p["y"]))))
spec("ml", "dot", "Dot product.", [("a", "1,2,3", "Vector a"), ("b", "4,5,6", "Vector b")],
     lambda p: _ok(result=sum(x * y for x, y in zip(_floats(p["a"]), _floats(p["b"])))))
spec("ml", "norm", "L2 norm of vector.", [("v", "3,4", "Vector")],
     lambda p: _ok(norm=math.sqrt(sum(x * x for x in _floats(p["v"])))))
spec("ml", "cosine", "Cosine similarity.", [("a", "1,0", "Vector a"), ("b", "1,0", "Vector b")],
     lambda p: _ok(similarity=_cosine(_floats(p["a"]), _floats(p["b"]))))
spec("ml", "euclidean", "Euclidean distance.", [("a", "0,0", "Point a"), ("b", "3,4", "Point b")],
     lambda p: _ok(distance=math.sqrt(sum((x - y) ** 2 for x, y in zip(_floats(p["a"]), _floats(p["b"]))))))
spec("ml", "manhattan", "Manhattan distance.", [("a", "0,0", "Point a"), ("b", "3,4", "Point b")],
     lambda p: _ok(distance=sum(abs(x - y) for x, y in zip(_floats(p["a"]), _floats(p["b"])))))
spec("ml", "softmax", "Softmax of scores.", [("scores", "1,2,3", "Logits")],
     lambda p: _ok(probs=_softmax(_floats(p["scores"]))))
spec("ml", "sigmoid", "Sigmoid of value.", [("x", 0.0, "Value")], lambda p: _ok(result=1.0 / (1.0 + math.exp(-p["x"]))))
spec("ml", "shuffle_data", "Shuffle aligned lists.", [("x", "1,2,3,4", "X"), ("y", "a,b,c,d", "Y"), ("seed", 1, "Seed")],
     lambda p: _ok(shuffled=_shuffle_aligned(p["x"].split(","), p["y"].split(","), int(p["seed"]))))
spec("ml", "split_ratio", "Split count into train/test.", [("n", 100, "Total"), ("ratio", 0.8, "Train ratio")],
     lambda p: _ok(train=int(p["n"] * p["ratio"]), test=int(p["n"]) - int(p["n"] * p["ratio"])))
spec("ml", "bins", "Histogram bins of values.", [("values", "1,1,2,3,3,3", "Numbers"), ("bins", 3, "Bins")],
     lambda p: _ok(hist=_hist(_floats(p["values"]), int(p["bins"]))))
spec("ml", "entropy", "Entropy of label distribution.", [("labels", "a,a,b", "Labels")],
     lambda p: _ok(entropy=_entropy(p["labels"].replace(",", ""))))


def _r2(y_true: List[float], y_pred: List[float]) -> float:
    mean_y = statistics.mean(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean_y) ** 2 for a in y_true)
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


def _confusion(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    labels = sorted(set(y_true + y_pred))
    mat = {str(a): {str(b): 0 for b in labels} for a in labels}
    for t, p in zip(y_true, y_pred):
        mat[str(t)][str(p)] = mat[str(t)].get(str(p), 0) + 1
    return mat


def _precision_recall_f1(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    labels = sorted(set(y_true + y_pred))
    precs, recs = [], []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        precs.append(tp / (tp + fp) if tp + fp else 0.0)
        recs.append(tp / (tp + fn) if tp + fn else 0.0)
    precision = statistics.mean(precs)
    recall = statistics.mean(recs)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _minmax(vals: List[float]) -> List[float]:
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return [0.0] * len(vals)
    return [(x - lo) / (hi - lo) for x in vals]


def _zscore(vals: List[float]) -> List[float]:
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 1.0
    return [(x - mean) / sd for x in vals]


def _onehot(labels: List[str]) -> Dict[str, Any]:
    uniq = sorted(set(labels))
    idx = {u: i for i, u in enumerate(uniq)}
    rows = [[1 if idx[x] == i else 0 for i in range(len(uniq))] for x in labels]
    return {"labels": uniq, "rows": rows}


def _label_encode(labels: List[str]) -> Dict[str, Any]:
    uniq = sorted(set(labels))
    mapping = {u: i for i, u in enumerate(uniq)}
    return {"mapping": mapping, "encoded": [mapping[x] for x in labels]}


def _corr(x: List[float], y: List[float]) -> float:
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else 0.0


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _softmax(scores: List[float]) -> List[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def _shuffle_aligned(x: List[str], y: List[str], seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    pairs = list(zip(x, y))
    rng.shuffle(pairs)
    return {"x": [a for a, _ in pairs], "y": [b for _, b in pairs]}


def _hist(vals: List[float], bins: int) -> Dict[str, Any]:
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return {str(lo): len(vals)}
    width = (hi - lo) / bins
    out = {}
    for i in range(bins):
        start = lo + i * width
        end = lo + (i + 1) * width if i < bins - 1 else hi + 1e-9
        label = f"[{start:.2f},{end:.2f})"
        out[label] = sum(1 for v in vals if start <= v < end)
    return out


# ===========================================================================
# TEXT processing group
# ===========================================================================
spec("text", "tokenize", "Split text into words/tokens.", [("text", "hello brave world", "Text")],
     lambda p: _ok(tokens=re.findall(r"[A-Za-z0-9']+", p["text"].lower())))
spec("text", "ngrams", "Generate n-grams.", [("text", "the quick brown fox", "Text"), ("n", 2, "N")],
     lambda p: _ok(ngrams=_ngrams(re.findall(r"[A-Za-z0-9']+", p["text"].lower()), int(p["n"]))))
spec("text", "freq", "Word frequencies.", [("text", "the cat and the dog", "Text")],
     lambda p: _ok(frequencies=_freq(re.findall(r"[A-Za-z0-9']+", p["text"].lower()))))
spec("text", "top_words", "Top frequent words.", [("text", "a a b b b c", "Text"), ("n", 2, "Count")],
     lambda p: _ok(top=sorted(_freq(re.findall(r"[A-Za-z0-9']+", p["text"].lower())).items(), key=lambda kv: -kv[1])[: int(p["n"])]))
spec("text", "stopwords", "Remove common stopwords.", [("text", "the quick brown fox jumps", "Text")],
     lambda p: _ok(result=" ".join(w for w in p["text"].split() if w.lower() not in _STOPWORDS)))
spec("text", "sentences", "Split into sentences.", [("text", "Hello. World! How are you?", "Text")],
     lambda p: _ok(sentences=re.split(r"(?<=[.!?])\s+", p["text"].strip())))
spec("text", "wrap", "Wrap text to width.", [("text", "The quick brown fox jumps over the lazy dog", "Text"), ("width", 20, "Width")],
     lambda p: _ok(lines=__import__("textwrap").wrap(p["text"], int(p["width"]))))
spec("text", "dedent", "Remove common leading whitespace.", [("text", "  hello\n  world", "Text")],
     lambda p: _ok(result=__import__("textwrap").dedent(p["text"])))
spec("text", "align", "Pad lines to width.", [("text", "a\nbb\nccc", "Text"), ("width", 6, "Width")],
     lambda p: _ok(result="\n".join(ln.ljust(int(p["width"])) for ln in p["text"].splitlines())))
spec("text", "csv", "Parse CSV text to rows.", [("text", "a,b\n1,2", "CSV text")],
     lambda p: _ok(rows=list(csv.reader(io.StringIO(p["text"])))))
spec("text", "tsv", "Parse TSV text to rows.", [("text", "a\tb\n1\t2", "TSV text")],
     lambda p: _ok(rows=list(csv.reader(io.StringIO(p["text"]), delimiter="\t"))))
spec("text", "table", "Render simple table.", [("rows", "Name,Age\nAlice,30\nBob,25", "CSV rows")],
     lambda p: _ok(table=_render_table([list(r) for r in csv.reader(io.StringIO(p["rows"]))])))
spec("text", "diff", "Simple line diff (similarity).", [("a", "hello world", "Text A"), ("b", "hello there", "Text B")],
     lambda p: _ok(similarity=_seq_sim(p["a"], p["b"])))
spec("text", "indent", "Indent each line.", [("text", "a\nb", "Text"), ("spaces", 4, "Spaces")],
     lambda p: _ok(result="\n".join(" " * int(p["spaces"]) + ln for ln in p["text"].splitlines())))
spec("text", "unindent", "Remove up to n leading spaces per line.", [("text", "    a\n    b", "Text"), ("spaces", 4, "Spaces")],
     lambda p: _ok(result="\n".join(ln[int(p["spaces"]):] if ln.startswith(" " * int(p["spaces"])) else ln for ln in p["text"].splitlines())))
spec("text", "case_count", "Count letter cases.", [("text", "Hello World", "Text")],
     lambda p: _ok(upper=sum(c.isupper() for c in p["text"]), lower=sum(c.islower() for c in p["text"])))
spec("text", "punct", "Extract punctuation.", [("text", "Hi! How are you?", "Text")],
     lambda p: _ok(punctuation=[c for c in p["text"] if c in string.punctuation]))
spec("text", "numbers", "Extract numbers from text.", [("text", "price is 12.5 and 3", "Text")],
     lambda p: _ok(numbers=[float(m) for m in re.findall(r"-?\d+\.?\d*", p["text"])]))
spec("text", "emails", "Extract email addresses.", [("text", "mail a@b.com and c@d.io", "Text")],
     lambda p: _ok(emails=re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", p["text"])))
spec("text", "urls", "Extract URLs.", [("text", "see https://a.com and http://b.org", "Text")],
     lambda p: _ok(urls=re.findall(r"https?://[^\s]+", p["text"])))
spec("text", "phones", "Extract phone-like numbers.", [("text", "call +374 91 123456", "Text")],
     lambda p: _ok(phones=re.findall(r"\+?\d[\d\s\-]{6,}", p["text"])))
spec("text", "hashtags", "Extract hashtags.", [("text", "love #AI and #AGI", "Text")],
     lambda p: _ok(hashtags=re.findall(r"#\w+", p["text"])))
spec("text", "mentions", "Extract @mentions.", [("text", "hello @ararat and @bob", "Text")],
     lambda p: _ok(mentions=re.findall(r"@\w+", p["text"])))
spec("text", "acronyms", "Extract uppercase acronyms.", [("text", "AI and NLP and CNN are acronyms", "Text")],
     lambda p: _ok(acronyms=re.findall(r"\b[A-Z]{2,}\b", p["text"])))
spec("text", "censored", "Mask emails and phones.", [("text", "mail a@b.com phone 37491123456", "Text")],
     lambda p: _ok(result=re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", re.sub(r"\+?\d[\d\s\-]{6,}", "[PHONE]", p["text"]))))
spec("text", "detect_latin", "Detect if text contains Latin script.", [("text", "hello", "Text")],
     lambda p: _ok(latin=any("a" <= c.lower() <= "z" for c in p["text"])))
spec("text", "detect_armenian", "Detect Armenian script.", [("text", "\u0562\u0561\u0580\u0565\u0582 \u0561\u0577\u056d\u0561\u0580\u0570", "Text")],
     lambda p: _ok(armenian=any("\u0531" <= c <= "\u0586" for c in p["text"])))
spec("text", "transliterate_hy", "Transliterate Armenian to Latin.", [("text", "\u0532\u0561\u0580\u0565\u0582", "Armenian text")],
     lambda p: _ok(result=_translit_hy(p["text"])))
spec("text", "transliterate_to_hy", "Transliterate Latin to Armenian.", [("text", "Barev", "Latin text")],
     lambda p: _ok(result=_translit_to_hy(p["text"])))
spec("text", "language", "Guess language (heuristic).", [("text", "Hello world", "Text")],
     lambda p: _ok(language=_guess_lang(p["text"])))
spec("text", "length_stats", "Character/word/line stats.", [("text", "hello world\nfoo", "Text")],
     lambda p: _ok(chars=len(p["text"]), words=len(p["text"].split()), lines=len(p["text"].splitlines())))
spec("text", "bom", "UTF-8 BOM strip.", [("text", "\ufeffhello", "Text")],
     lambda p: _ok(result=p["text"].lstrip("\ufeff")))
spec("text", "compress", "zlib-compress text (base64).", [("text", "hello hello hello", "Text")],
     lambda p: _ok(compressed=__import__("base64").b64encode(zlib.compress(p["text"].encode())).decode()))
spec("text", "decompress", "zlib-decompress base64.", [("text", "eJxLyUhKLElMAClFApE=", "Base64 zlib")],
     lambda p: _ok(result=zlib.decompress(__import__("base64").b64decode(p["text"])).decode(errors="replace")))


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "to", "in",
    "on", "at", "by", "for", "with", "about", "as", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "can", "could", "should", "may", "might", "must", "this", "that",
    "these", "those", "it", "its", "from", "up", "down", "out", "off", "over",
}


def _ngrams(tokens: List[str], n: int) -> List[List[str]]:
    return [tokens[i:i + n] for i in range(max(0, len(tokens) - n + 1))]


def _freq(tokens: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in tokens:
        out[t] = out.get(t, 0) + 1
    return out


def _render_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for ri, row in enumerate(rows):
        lines.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
        if ri == 0:
            lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    return "\n".join(lines)


def _seq_sim(a: str, b: str) -> float:
    # simple bigram Jaccard
    def bigrams(s: str) -> set:
        return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}
    ga, gb = bigrams(a), bigrams(b)
    if not ga and not gb:
        return 1.0
    return len(ga & gb) / len(ga | gb)


_HY_LATIN = {
    "\u0561": "a", "\u0562": "b", "\u0563": "g", "\u0564": "d", "\u0565": "e", "\u0566": "z", "\u0567": "e",
    "\u0568": "y", "\u0569": "t", "\u056a": "zh", "\u056b": "i", "\u056c": "l", "\u056d": "kh", "\u056e": "ts",
    "\u056f": "k", "\u0570": "h", "\u0571": "dz", "\u0572": "gh", "\u0573": "ch", "\u0574": "m", "\u0575": "y",
    "\u0576": "n", "\u0577": "sh", "\u0578": "o", "\u0579": "ch", "\u057a": "p", "\u057b": "j", "\u057c": "r",
    "\u057d": "s", "\u057e": "v", "\u057f": "t", "\u0580": "r", "\u0581": "ts", "\u0578\u0582": "u", "\u0583": "p",
    "\u0584": "q", "\u0587": "ev", "\u0585": "o", "\u0586": "f",
}
_LATIN_HY = {v: k for k, v in _HY_LATIN.items()}


def _translit_hy(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        c = text[i]
        low = c.lower()
        two = text[i:i + 2].lower()
        if two == "\u0578\u0582":
            out.append("u" if c.islower() else "U")
            i += 2
            continue
        if low in _HY_LATIN:
            mapped = _HY_LATIN[low]
            out.append(mapped.capitalize() if c.isupper() and not c.islower() else mapped)
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _translit_to_hy(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        c = text[i]
        low = c.lower()
        # two-letter mappings first (longest match)
        matched = False
        for ln in (2, 1):
            if ln == 2 and i + 1 < len(text):
                pair = text[i:i + 2].lower()
                if pair in _LATIN_HY:
                    out.append(_LATIN_HY[pair])
                    i += 2
                    matched = True
                    break
            elif ln == 1 and low in _LATIN_HY:
                out.append(_LATIN_HY[low])
                i += 1
                matched = True
                break
        if not matched:
            out.append(c)
            i += 1
    return "".join(out)


def _guess_lang(text: str) -> str:
    hy = sum(1 for c in text if "\u0531" <= c <= "\u0586")
    if hy > 0:
        return "hy (Armenian)"
    ru = sum(1 for c in text if "\u0410" <= c <= "\u044F")
    if ru > 0:
        return "ru (Russian)"
    # very rough Latin heuristic: English stopwords
    words = set(re.findall(r"[A-Za-z']+", text.lower()))
    en_stops = {"the", "and", "of", "to", "is", "in", "that", "for", "it", "with"}
    if words & en_stops:
        return "en (English)"
    return "unknown"


# ===========================================================================
# IMAGE / AUDIO / VIDEO group (pure metadata, no heavy deps)
# ===========================================================================
spec("image", "info", "Read image dimensions/type from file header.", [("path", "img.png", "Image path")],
     lambda p: _ok(**_image_info(p["path"])))
spec("image", "resize_hint", "Suggested resize dims preserving aspect.", [("width", 1920, "Width"), ("height", 1080, "Height"), ("max", 512, "Max side")],
     lambda p: _ok(dims=_resize_hint(int(p["width"]), int(p["height"]), int(p["max"]))))
spec("image", "histogram_text", "ASCII brightness histogram (requires PIL).", [("path", "img.png", "Image path")],
     lambda p: _ok(hist=_ascii_hist(p["path"])))
spec("audio", "info", "Audio file metadata via stdlib (WAV) or ffprobe.", [("path", "sound.wav", "Audio path")],
     lambda p: _ok(**_audio_info(p["path"])))
spec("audio", "tone", "Generate WAV sine tone.", [("path", "tone.wav", "Output path"), ("freq", 440.0, "Hz"), ("seconds", 2.0, "Seconds"), ("rate", 22050, "Sample rate")],
     lambda p: _ok(**_gen_tone(p["path"], p["freq"], p["seconds"], int(p["rate"]))))
spec("video", "info", "Video metadata via ffprobe (if available).", [("path", "movie.mp4", "Video path")],
     lambda p: _ok(**_video_info(p["path"])))


def _image_info(path: str) -> Dict[str, Any]:
    try:
        head = Path(path).read_bytes()[:64]
    except Exception as e:
        return {"error": str(e)}
    if head[:8] == b"\x89PNG\r\n\x1a\n" and len(head) >= 24:
        w, h = struct.unpack(">II", head[16:24])
        return {"type": "png", "width": w, "height": h}
    if head[:2] == b"\xff\xd8":
        # scan for SOF markers
        i = 2
        while i < len(head) - 8:
            if head[i] != 0xFF:
                i += 1
                continue
            marker = head[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", head[i + 5:i + 9])
                return {"type": "jpeg", "width": w, "height": h}
            i += 2
        return {"type": "jpeg", "width": None, "height": None}
    if head[:6] in (b"GIF87a", b"GIF89a") and len(head) >= 10:
        w, h = struct.unpack("<HH", head[6:10])
        return {"type": "gif", "width": w, "height": h}
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return {"type": "webp", "width": None, "height": None}
    if head[:4] == b"BM" and len(head) >= 26:
        w, h = struct.unpack("<ii", head[18:26])
        return {"type": "bmp", "width": w, "height": h}
    return {"type": "unknown", "width": None, "height": None}


def _resize_hint(w: int, h: int, max_side: int) -> Dict[str, int]:
    scale = min(1.0, max_side / max(w, h))
    return {"width": max(1, int(w * scale)), "height": max(1, int(h * scale))}


def _ascii_hist(path: str) -> Any:
    try:
        from PIL import Image
    except Exception:
        return {"error": "PIL not installed"}
    try:
        img = Image.open(path).convert("L")
        hist = img.histogram()
        # 10 buckets
        buckets = [0] * 10
        total = img.width * img.height
        for i, count in enumerate(hist):
            buckets[min(9, i * 10 // 256)] += count
        return {"buckets": [round(b / total * 100, 1) for b in buckets], "bars": ["#" * int(b / total * 50) for b in buckets]}
    except Exception as e:
        return {"error": str(e)}


def _audio_info(path: str) -> Dict[str, Any]:
    head = Path(path).read_bytes()[:44] if Path(path).exists() else b""
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE" and len(head) >= 44:
        channels, rate = struct.unpack("<HH", head[22:26])
        bits = struct.unpack("<H", head[34:36])[0]
        return {"type": "wav", "channels": channels, "sample_rate": rate, "bits": bits}
    probe = _cmd(f'ffprobe -v error -show_format -show_streams -of json "{path}"')
    if probe:
        try:
            data = json.loads(probe)
            streams = data.get("streams", [])
            fmt = data.get("format", {})
            return {"type": fmt.get("format_long_name", "media"),
                    "duration_s": fmt.get("duration"), "streams": len(streams)}
        except Exception:
            pass
    return {"type": "unknown"}


def _gen_tone(path: str, freq: float, seconds: float, rate: int) -> Dict[str, Any]:
    import wave

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = int(rate * seconds)
    frames = bytearray()
    for i in range(n):
        val = int(32767 * math.sin(2 * math.pi * freq * i / rate))
        frames += struct.pack("<h", val)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return {"path": path, "sample_rate": rate, "seconds": seconds, "samples": n}


def _video_info(path: str) -> Dict[str, Any]:
    probe = _cmd(f'ffprobe -v error -show_format -show_streams -of json "{path}"')
    if not probe:
        return {"error": "ffprobe not available"}
    try:
        data = json.loads(probe)
        streams = data.get("streams", [])
        v = next((s for s in streams if s.get("codec_type") == "video"), None)
        a = next((s for s in streams if s.get("codec_type") == "audio"), None)
        return {
            "duration_s": data.get("format", {}).get("duration"),
            "video_codec": v.get("codec_name") if v else None,
            "width": v.get("width") if v else None,
            "height": v.get("height") if v else None,
            "audio_codec": a.get("codec_name") if a else None,
        }
    except Exception as e:
        return {"error": str(e)}


# ===========================================================================
# SYSTEM / OS group
# ===========================================================================
spec("sys", "info", "System information.", [], lambda p: _ok(**_sys_info()))
spec("sys", "python", "Python version.", [], lambda p: _ok(version=sys.version.split()[0], executable=sys.executable))
spec("sys", "platform", "Platform details.", [], lambda p: _ok(system=platform.system(), release=platform.release(), machine=platform.machine()))
spec("sys", "cwd", "Current working directory.", [], lambda p: _ok(cwd=os.getcwd()))
spec("sys", "env", "Environment variable.", [("name", "PATH", "Var name")],
     lambda p: _ok(value=os.environ.get(p["name"], "")))
spec("sys", "env_all", "All environment variable names.", [], lambda p: _ok(names=sorted(os.environ.keys())))
spec("sys", "cpu_count", "CPU count.", [], lambda p: _ok(cpus=os.cpu_count()))
spec("sys", "memory", "Memory info (psutil if available).", [], lambda p: _ok(**_memory_info()))
spec("sys", "disk", "Disk usage of path.", [("path", ".", "Path")],
     lambda p: _ok(**_disk_usage(p["path"])))
spec("sys", "uptime", "System uptime (from /proc).", [], lambda p: _ok(seconds=_uptime()))
spec("sys", "users", "Current user.", [], lambda p: _ok(user=os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"))
spec("sys", "arch", "CPU architecture.", [], lambda p: _ok(arch=platform.machine()))
spec("sys", "python_path", "sys.path entries.", [], lambda p: _ok(path=sys.path))
spec("sys", "modules", "Loaded module count.", [], lambda p: _ok(count=len(sys.modules)))
spec("sys", "shell", "Default shell.", [], lambda p: _ok(shell=os.environ.get("SHELL", "unknown")))
spec("sys", "home", "Home directory.", [], lambda p: _ok(home=str(Path.home())))
spec("sys", "tempdir", "Temp directory.", [], lambda p: _ok(temp=__import__("tempfile").gettempdir()))


def _sys_info() -> Dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "hostname": socket.gethostname(),
    }


def _memory_info() -> Dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {"total": vm.total, "available": vm.available, "percent": vm.percent}
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                data = {}
                for line in f:
                    k, _, v = line.partition(":")
                    data[k] = int(v.strip().split()[0]) * 1024
            return {"total": data.get("MemTotal", 0), "available": data.get("MemAvailable", data.get("MemFree", 0))}
        except Exception:
            return {"error": "unavailable"}


def _disk_usage(path: str) -> Dict[str, Any]:
    try:
        import shutil

        total, used, free = shutil.disk_usage(path)
        return {"total": total, "used": used, "free": free, "percent": round(used / total * 100, 1) if total else 0}
    except Exception:
        return {"error": "unavailable"}


def _uptime() -> int:
    try:
        with open("/proc/uptime") as f:
            return int(float(f.read().split()[0]))
    except Exception:
        return 0


# ===========================================================================
# DB group (SQLite local)
# ===========================================================================
spec("db", "create", "Create SQLite table.", [("path", "db.sqlite", "DB path"), ("table", "items", "Table name"), ("schema", "id INTEGER PRIMARY KEY, name TEXT", "Column schema")],
     lambda p: _db_create(p["path"], p["table"], p["schema"]))
spec("db", "insert", "Insert row into SQLite.", [("path", "db.sqlite", "DB path"), ("table", "items", "Table"), ("values", "1,'hello'", "Values")],
     lambda p: _db_insert(p["path"], p["table"], p["values"]))
spec("db", "query", "Run SQL query.", [("path", "db.sqlite", "DB path"), ("sql", "SELECT * FROM items", "SQL")],
     lambda p: _db_query(p["path"], p["sql"]))
spec("db", "tables", "List SQLite tables.", [("path", "db.sqlite", "DB path")],
     lambda p: _db_tables(p["path"]))
spec("db", "schema", "Show table schema.", [("path", "db.sqlite", "DB path"), ("table", "items", "Table")],
     lambda p: _db_schema(p["path"], p["table"]))
spec("db", "count", "Count rows in table.", [("path", "db.sqlite", "DB path"), ("table", "items", "Table")],
     lambda p: _db_count(p["path"], p["table"]))
spec("db", "drop", "Drop table.", [("path", "db.sqlite", "DB path"), ("table", "items", "Table")],
     lambda p: _db_drop(p["path"], p["table"]))
spec("db", "delete", "Delete rows.", [("path", "db.sqlite", "DB path"), ("table", "items", "Table"), ("where", "id=1", "WHERE clause")],
     lambda p: _db_delete(p["path"], p["table"], p["where"]))
spec("db", "update", "Update rows.", [("path", "db.sqlite", "DB path"), ("table", "items", "Table"), ("set", "name='x'", "SET clause"), ("where", "id=1", "WHERE clause")],
     lambda p: _db_update(p["path"], p["table"], p["set"], p["where"]))


def _db_connect(path: str):
    import sqlite3

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _db_create(path: str, table: str, schema: str) -> Dict[str, Any]:
    conn = _db_connect(path)
    try:
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({schema})")
        conn.commit()
        return _ok(table=table, schema=schema)
    finally:
        conn.close()


def _db_insert(path: str, table: str, values: str) -> Dict[str, Any]:
    conn = _db_connect(path)
    try:
        cur = conn.execute(f"INSERT INTO {table} VALUES ({values})")
        conn.commit()
        return _ok(rowid=cur.lastrowid)
    finally:
        conn.close()


def _db_query(path: str, sql: str) -> Dict[str, Any]:
    conn = _db_connect(path)
    try:
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        return _ok(rows=rows, count=len(rows))
    finally:
        conn.close()


def _db_tables(path: str) -> Dict[str, Any]:
    conn = _db_connect(path)
    try:
        rows = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        return _ok(tables=rows)
    finally:
        conn.close()


def _db_schema(path: str, table: str) -> Dict[str, Any]:
    conn = _db_connect(path)
    try:
        rows = [dict(r) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        return _ok(columns=rows)
    finally:
        conn.close()


def _db_count(path: str, table: str) -> Dict[str, Any]:
    conn = _db_connect(path)
    try:
        return _ok(count=conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        conn.close()


def _db_drop(path: str, table: str) -> Dict[str, Any]:
    conn = _db_connect(path)
    try:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
        return _ok(dropped=table)
    finally:
        conn.close()


def _db_delete(path: str, table: str, where: str) -> Dict[str, Any]:
    conn = _db_connect(path)
    try:
        cur = conn.execute(f"DELETE FROM {table} WHERE {where}")
        conn.commit()
        return _ok(deleted=cur.rowcount)
    finally:
        conn.close()


def _db_update(path: str, table: str, set_clause: str, where: str) -> Dict[str, Any]:
    conn = _db_connect(path)
    try:
        cur = conn.execute(f"UPDATE {table} SET {set_clause} WHERE {where}")
        conn.commit()
        return _ok(updated=cur.rowcount)
    finally:
        conn.close()


# ===========================================================================
# CLOUD group (provider endpoints via public APIs)
# ===========================================================================
spec("cloud", "weather", "Weather for city (open-meteo, no key).", [("city", "Yerevan", "City name"), ("lat", None, "Latitude (optional)"), ("lon", None, "Longitude (optional)")],
     lambda p: _cloud_weather(p["city"], p["lat"], p["lon"]))
spec("cloud", "time", "Time in timezone (worldtimeapi).", [("zone", "Asia/Yerevan", "IANA timezone")],
     lambda p: _ok(result=_http_get(f"http://worldtimeapi.org/api/timezone/{p['zone']}") if _port_open("worldtimeapi.org", 80) else {"error": "offline"}))
spec("cloud", "geoip", "GeoIP of current connection.", [],
     lambda p: _ok(result=_http_get("http://ip-api.com/json/") if _port_open("ip-api.com", 80) else {"error": "offline"}))
spec("cloud", "ip", "Public IP (alias).", [], lambda p: _ok(ip=_http_get("https://api.ipify.org").strip()))
spec("cloud", "quote", "Random quote (dummyjson).", [],
     lambda p: _ok(result=json.loads(_http_get("https://dummyjson.com/quotes/random")) if _port_open("dummyjson.com", 443) else {"error": "offline"}))
spec("cloud", "users", "Random users (dummyjson).", [("count", 3, "Count")],
     lambda p: _ok(result=json.loads(_http_get(f"https://dummyjson.com/users?limit={int(p['count'])}")) if _port_open("dummyjson.com", 443) else {"error": "offline"}))
spec("cloud", "cat_fact", "Random cat fact.", [],
     lambda p: _ok(fact=json.loads(_http_get("https://catfact.ninja/fact"))["fact"] if _port_open("catfact.ninja", 443) else "offline"))


def _cloud_weather(city: str, lat: Optional[str], lon: Optional[str]) -> Dict[str, Any]:
    try:
        if not lat or not lon:
            geo = json.loads(_http_get(f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1"))
            results = geo.get("results")
            if not results:
                return _err(f"city not found: {city}")
            lat = str(results[0]["latitude"])
            lon = str(results[0]["longitude"])
        data = json.loads(_http_get(f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"))
        cur = data.get("current_weather", {})
        return _ok(city=city, temperature_c=cur.get("temperature"), windspeed_kmh=cur.get("windspeed"),
                   weather_code=cur.get("weathercode"), time=cur.get("time"))
    except Exception as e:
        return _err(str(e))


# ===========================================================================
# LLM group (BYOK adapters, no key = instructions)
# ===========================================================================
spec("llm", "providers", "List supported LLM providers.", [],
     lambda p: _ok(providers=["openai", "google", "microsoft", "anthropic", "huggingface", "ollama", "local"]))
spec("llm", "chat", "Chat with a provider (requires API key).", [("provider", "openai", "Provider"), ("message", "Hello", "Message"), ("model", None, "Model (optional)")],
     lambda p: _llm_chat(p["provider"], p["message"], p["model"]))
spec("llm", "prompt", "Print a reusable prompt template.", [("template", "chain_of_thought", "Template name")],
     lambda p: _ok(template=_PROMPT_TEMPLATES.get(p["template"], _PROMPT_TEMPLATES["chain_of_thought"])))
spec("llm", "templates", "List prompt templates.", [], lambda p: _ok(templates=sorted(_PROMPT_TEMPLATES.keys())))
spec("llm", "tokens_est", "Estimate tokens (chars/4).", [("text", "hello world", "Text")],
     lambda p: _ok(tokens=max(1, len(p["text"]) // 4), chars=len(p["text"])))


_PROMPT_TEMPLATES = {
    "chain_of_thought": "Solve the problem step by step, then give the final answer.\n\nProblem: {question}",
    "few_shot": "Examples:\n{examples}\n\nNow answer: {question}",
    "system": "You are a helpful assistant.\n\n{user}",
    "rag": "Context:\n{context}\n\nQuestion: {question}\nAnswer using only the context.",
    "summarize": "Summarize the following text concisely:\n\n{text}",
    "translate": "Translate the following text to {language}:\n\n{text}",
    "extract": "Extract key facts from the text as a list:\n\n{text}",
    "code_review": "Review this code for bugs, style and security:\n\n{code}",
    "explain": "Explain the following concept simply:\n\n{concept}",
    "brainstorm": "Brainstorm 10 creative ideas about: {topic}",
}


def _llm_chat(provider: str, message: str, model: Optional[str]) -> Dict[str, Any]:
    try:
        from aweai.integrations import chat
        return chat(provider, message, model=model)
    except Exception as e:
        return {"ok": False, "error": str(e),
                "hint": f"Set an API key for '{provider}' (aweai keys set {provider.upper()}_API_KEY) to use live chat."}


# ===========================================================================
# RL group (mini reinforcement learning utilities)
# ===========================================================================
spec("rl", "bandit_epsilon", "Epsilon-greedy bandit simulation.", [("arms", 3, "Arms"), ("steps", 100, "Steps"), ("epsilon", 0.1, "Exploration rate"), ("seed", 1, "Seed")],
     lambda p: _ok(result=_bandit(int(p["arms"]), int(p["steps"]), float(p["epsilon"]), int(p["seed"]))))
spec("rl", "q_learning", "Run Q-learning on a small grid (1D chain).", [("states", 5, "States"), ("episodes", 100, "Episodes"), ("lr", 0.1, "Learning rate"), ("gamma", 0.9, "Discount"), ("seed", 1, "Seed")],
     lambda p: _ok(result=_q_learning(int(p["states"]), int(p["episodes"]), float(p["lr"]), float(p["gamma"]), int(p["seed"]))))
spec("rl", "discount", "Discounted return of rewards.", [("rewards", "1,1,1", "Rewards"), ("gamma", 0.9, "Gamma")],
     lambda p: _ok(return_=_discounted_return(_floats(p["rewards"]), float(p["gamma"]))))
spec("rl", "exp_decay", "Exponential decay rate.", [("initial", 1.0, "Initial"), ("decay", 0.99, "Decay per step"), ("steps", 100, "Steps")],
     lambda p: _ok(result=p["initial"] * (p["decay"] ** int(p["steps"]))))


def _bandit(arms: int, steps: int, epsilon: float, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    true_means = [rng.random() for _ in range(arms)]
    counts = [0] * arms
    sums = [0.0] * arms
    rewards = []
    for _ in range(steps):
        if rng.random() < epsilon:
            a = rng.randrange(arms)
        else:
            a = max(range(arms), key=lambda i: sums[i] / counts[i] if counts[i] else 0.0)
        r = 1.0 if rng.random() < true_means[a] else 0.0
        counts[a] += 1
        sums[a] += r
        rewards.append(r)
    return {"true_means": true_means, "counts": counts,
            "estimated_means": [sums[i] / counts[i] if counts[i] else 0.0 for i in range(arms)],
            "total_reward": sum(rewards), "avg_reward": sum(rewards) / steps}


def _q_learning(states: int, episodes: int, lr: float, gamma: float, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    q = [[0.0, 0.0] for _ in range(states)]
    for _ in range(episodes):
        s = 0
        for _step in range(200):
            a = 0 if rng.random() < 0.5 else 1
            ns = max(0, min(states - 1, s + (1 if a == 1 else -1)))
            r = 1.0 if ns == states - 1 else 0.0
            q[s][a] += lr * (r + gamma * max(q[ns]) - q[s][a])
            s = ns
            if r > 0:
                break
    return {"states": states, "q_table": q, "policy": [int(q[s][1] > q[s][0]) for s in range(states)]}


def _discounted_return(rewards: List[float], gamma: float) -> float:
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
    return g


# ===========================================================================
# NEURO group (neural-net helpers, numpy-free)
# ===========================================================================
spec("neuro", "neuron", "Single neuron output (w*x + b, activation).", [("weights", "1,1", "Weights"), ("inputs", "2,3", "Inputs"), ("bias", 0.0, "Bias"), ("activation", "relu", "relu|sigmoid|tanh|none")],
     lambda p: _ok(output=_neuron(_floats(p["weights"]), _floats(p["inputs"]), float(p["bias"]), p["activation"])))
spec("neuro", "layer", "Dense layer forward pass.", [("weights", "1,0,0,1", "Weights (out x in)"), ("inputs", "1,2", "Inputs"), ("bias", "0,0", "Biases"), ("activation", "relu", "Activation")],
     lambda p: _ok(output=_layer(_floats(p["weights"]), _floats(p["inputs"]), _floats(p["bias"]), p["activation"])))
spec("neuro", "activation", "Apply activation function.", [("x", 0.0, "Value"), ("name", "relu", "relu|sigmoid|tanh|softplus|gelu|leaky_relu")],
     lambda p: _ok(result=_activate(float(p["x"]), p["name"])))
spec("neuro", "param_count", "Count parameters of layer shapes.", [("shapes", "784,128,128,10", "Layer sizes (comma)")],
     lambda p: _ok(params=_param_count(_ints(p["shapes"]))))
spec("neuro", "init", "Initialize weights (Xavier).", [("fan_in", 784, "Fan in"), ("fan_out", 128, "Fan out"), ("seed", 1, "Seed")],
     lambda p: _ok(limit=math.sqrt(6.0 / (int(p["fan_in"]) + int(p["fan_out"]))), note="Xavier uniform bound; sample U(-limit, limit)"))
spec("neuro", "learning_rate", "Schedule learning rate over steps.", [("base", 0.01, "Base LR"), ("steps", 1000, "Steps"), ("warmup", 100, "Warmup"), ("decay", "cosine", "cosine|linear")],
     lambda p: _ok(rate=_lr_schedule(float(p["base"]), int(p["steps"]), int(p["warmup"]), p["decay"])))


def _activate(x: float, name: str) -> float:
    name = name.lower()
    if name == "relu":
        return max(0.0, x)
    if name == "sigmoid":
        return 1.0 / (1.0 + math.exp(-x))
    if name == "tanh":
        return math.tanh(x)
    if name == "softplus":
        return math.log1p(math.exp(x))
    if name == "gelu":
        return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    if name == "leaky_relu":
        return x if x > 0 else 0.01 * x
    return x


def _neuron(weights: List[float], inputs: List[float], bias: float, activation: str) -> float:
    z = bias + sum(w * x for w, x in zip(weights, inputs))
    return _activate(z, activation)


def _layer(weights: List[float], inputs: List[float], bias: List[float], activation: str) -> List[float]:
    n_out = len(bias)
    n_in = len(inputs)
    out = []
    for o in range(n_out):
        z = bias[o]
        for i in range(n_in):
            z += weights[o * n_in + i] * inputs[i]
        out.append(_activate(z, activation))
    return out


def _param_count(shapes: List[int]) -> Dict[str, Any]:
    total = 0
    layers = []
    for i in range(len(shapes) - 1):
        w = shapes[i] * shapes[i + 1]
        b = shapes[i + 1]
        layers.append({"layer": i, "weights": w, "biases": b, "params": w + b})
        total += w + b
    return {"layers": layers, "total_params": total}


def _lr_schedule(base: float, steps: int, warmup: int, decay: str) -> float:
    # returns the final scheduled LR after `steps`
    if warmup > 0 and steps <= warmup:
        return base * steps / warmup
    t = max(0, steps - warmup)
    total = max(1, steps - warmup)
    if decay == "linear":
        return base * (1.0 - t / total)
    return base * 0.5 * (1.0 + math.cos(math.pi * t / total))


# ===========================================================================
# KNOWLEDGE group (AI/ASI/AGI encyclopedia shortcuts)
# ===========================================================================
spec("knowledge", "concepts", "Count knowledge-base entries.", [], lambda p: _ok(count=len(CONCEPTS)))
spec("knowledge", "timeline", "AI history timeline.", [], lambda p: _ok(timeline=TIMELINE))
spec("knowledge", "agi_levels", "AGI capability levels.", [], lambda p: _ok(levels=AGI_LEVELS))
spec("knowledge", "roadmap", "AI roadmap phases.", [], lambda p: _ok(roadmap=ROADMAP))
spec("knowledge", "self_improvement", "Self-improvement hooks.", [], lambda p: _ok(hooks=SELF_IMPROVEMENT_HOOKS))
spec("knowledge", "about", "Knowledge base stats.", [], lambda p: _ok(**about()))


# pull the AI knowledge base in for the knowledge group
from aweai.ai import CONCEPTS, TIMELINE, AGI_LEVELS, ROADMAP, SELF_IMPROVEMENT_HOOKS, about  # noqa: E402

# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------

GROUPS: List[str] = []
GROUP_INDEX: Dict[str, List[Dict[str, Any]]] = {}


def _build_index() -> None:
    global GROUPS, GROUP_INDEX
    GROUP_INDEX = {}
    for item in S:
        GROUP_INDEX.setdefault(item["group"], []).append(item)
    GROUPS = sorted(GROUP_INDEX.keys())


_build_index()


def count() -> int:
    return len(S)


def group_count() -> int:
    return len(GROUPS)

def rebuild_index() -> None:
    """Rebuild the group index after new specs are appended (used by bulk_extra)."""
    _build_index()

def specs_for(group: str) -> List[Dict[str, Any]]:
    return GROUP_INDEX.get(group, [])


def all_specs() -> List[Dict[str, Any]]:
    return S


def groups() -> List[str]:
    return GROUPS
