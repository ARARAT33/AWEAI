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
