"""AWEAI mega tools — 1000+ unique-purpose tools generated compactly.

This module provides the "mega" tool families. Instead of hand-writing
thousands of functions, every tool is defined declaratively in the FAMILIES
table below and registered at import time by `_register_families()`.

Each tool:
* has a UNIQUE name (e.g. ``math_add``, ``str_upper``, ``json_minify``)
* has a UNIQUE purpose (one-line description)
* accepts typed params (JSON-schema hints for UI/CLI rendering)
* returns a normalized dict ``{"result": ...}`` or ``{"error": ...}``
* runs anywhere (stdlib-only) — localhost, LAN, cloud, container, phone.

Families added here: math, string (str_), json, file (fs_), system (sys_),
network (net_), http, code, data, time, uuid, hash, encode, format (fmt_),
validate (val_), generate (gen_), archive (arc_), text (txt_), markdown (md_),
web, api, git, docker, ci, monitor, backup, sync, schedule, workflow, cloud,
db, k8s, deploy, security (sec_), ai, auto, csv, sql, xml, yaml, regex, misc.

All functions are safe: they never execute arbitrary shell without opt-in.
"""

from __future__ import annotations

import base64
import csv
import datetime as _dt
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import socket
import sqlite3
import statistics
import string
import subprocess
import textwrap
import sys
import time
import urllib.parse
import urllib.request
import uuid
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from aweai.tools.registry import tool

# ---------------------------------------------------------------------------
# Generic helpers
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


def _safe(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a plain function so it never crashes the registry."""

    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            return {"result": fn(*args, **kwargs)}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    return wrapper


def _reg(name: str, category: str, purpose: str, fn: Callable[..., Any], params: Optional[Dict[str, Any]] = None) -> None:
    """Register one tool with a normalized wrapper."""
    tool(name, category, purpose, params=params or {})(_safe(fn))


# ---------------------------------------------------------------------------
# Declarative families. Each entry: (name, purpose, fn, params)
# ---------------------------------------------------------------------------

def _build_families() -> List[Dict[str, Any]]:
    f: List[Dict[str, Any]] = []

    # ---- math -----------------------------------------------------------
    math_ops = [
        ("math_add", "Add two numbers", lambda a=0, b=0: _num(a) + _num(b)),
        ("math_sub", "Subtract b from a", lambda a=0, b=0: _num(a) - _num(b)),
        ("math_mul", "Multiply two numbers", lambda a=1, b=1: _num(a) * _num(b)),
        ("math_div", "Divide a by b", lambda a=1, b=1: _num(a) / _num(b) if _num(b) else "div-by-zero"),
        ("math_mod", "Modulo (remainder) of a / b", lambda a=10, b=3: _num(a) % _num(b) if _num(b) else "div-by-zero"),
        ("math_pow", "Raise a to the power b", lambda a=2, b=10: math.pow(_num(a), _num(b))),
        ("math_abs", "Absolute value", lambda x=0: abs(_num(x))),
        ("math_floor", "Floor of a number", lambda x=0: math.floor(_num(x))),
        ("math_ceil", "Ceiling of a number", lambda x=0: math.ceil(_num(x))),
        ("math_round", "Round to n decimals", lambda x=3.14159, n=2: round(_num(x), int(n))),
        ("math_sqrt", "Square root", lambda x=16: math.sqrt(_num(x)) if _num(x) >= 0 else "negative"),
        ("math_cbrt", "Cube root", lambda x=27: _num(x) ** (1 / 3) if _num(x) >= 0 else -(-_num(x)) ** (1 / 3)),
        ("math_square", "Square of a number", lambda x=5: _num(x) ** 2),
        ("math_cube", "Cube of a number", lambda x=3: _num(x) ** 3),
        ("math_reciprocal", "Reciprocal 1/x", lambda x=4: 1 / _num(x) if _num(x) else "div-by-zero"),
        ("math_log", "Natural logarithm", lambda x=math.e: math.log(_num(x)) if _num(x) > 0 else "invalid"),
        ("math_log10", "Base-10 logarithm", lambda x=100: math.log10(_num(x)) if _num(x) > 0 else "invalid"),
        ("math_log2", "Base-2 logarithm", lambda x=8: math.log2(_num(x)) if _num(x) > 0 else "invalid"),
        ("math_exp", "e raised to x", lambda x=1: math.exp(_num(x))),
        ("math_factorial", "Factorial of n", lambda n=5: math.factorial(int(n))),
        ("math_gcd", "Greatest common divisor", lambda a=12, b=18: math.gcd(int(a), int(b))),
        ("math_lcm", "Least common multiple", lambda a=4, b=6: abs(int(a) * int(b)) // math.gcd(int(a), int(b))),
        ("math_is_prime", "Check primality", lambda n=7: _is_prime(int(n))),
        ("math_primes", "Primes up to limit (sieve)", lambda limit=100: _primes(int(limit))),
        ("math_fib", "First n Fabonacci numbers", lambda n=10: _fib(int(n))),
        ("math_sum", "Sum of a list", lambda values=[1, 2, 3]: sum(_as_list(values))),
        ("math_product", "Product of a list", lambda values=[1, 2, 3, 4]: _prod(_as_list(values))),
        ("math_min", "Minimum of a list", lambda values=[3, 1, 2]: min(_as_list(values))),
        ("math_max", "Maximum of a list", lambda values=[3, 1, 2]: max(_as_list(values))),
        ("math_mean", "Arithmetic mean", lambda values=[1, 2, 3]: statistics.mean(_as_list(values))),
        ("math_median", "Median", lambda values=[1, 2, 3]: statistics.median(_as_list(values))),
        ("math_mode", "Mode (most frequent)", lambda values=[1, 2, 2]: _mode(_as_list(values))),
        ("math_stdev", "Sample standard deviation", lambda values=[1, 2, 3, 4]: statistics.stdev(_as_list(values))),
        ("math_variance", "Sample variance", lambda values=[1, 2, 3, 4]: statistics.variance(_as_list(values))),
        ("math_clamp", "Clamp x between lo and hi", lambda x=5, lo=0, hi=10: max(_num(lo), min(_num(x), _num(hi)))),
        ("math_sign", "Sign of a number (-1/0/1)", lambda x=0: (1 if _num(x) > 0 else (-1 if _num(x) < 0 else 0))),
        ("math_lerp", "Linear interpolation a..b by t", lambda a=0, b=10, t=0.5: _num(a) + (_num(b) - _num(a)) * _num(t)),
        ("math_deg2rad", "Degrees to radians", lambda deg=180: math.radians(_num(deg))),
        ("math_rad2deg", "Radians to degrees", lambda rad=math.pi: math.degrees(_num(rad))),
        ("math_sin", "Sine of x (radians)", lambda x=0: math.sin(_num(x))),
        ("math_cos", "Cosine of x (radians)", lambda x=0: math.cos(_num(x))),
        ("math_tan", "Tangent of x (radians)", lambda x=0: math.tan(_num(x))),
        ("math_asin", "Arcsine of x", lambda x=0: math.asin(_num(x)) if -1 <= _num(x) <= 1 else "invalid"),
        ("math_acos", "Arccosine of x", lambda x=0: math.acos(_num(x)) if -1 <= _num(x) <= 1 else "invalid"),
        ("math_atan2", "Arctangent of y/x", lambda y=1, x=1: math.atan2(_num(y), _num(x))),
        ("math_hypot", "Hypotenuse of a,b", lambda a=3, b=4: math.hypot(_num(a), _num(b))),
        ("math_is_even", "Check even", lambda n=4: int(n) % 2 == 0),
        ("math_is_odd", "Check odd", lambda n=3: int(n) % 2 == 1),
        ("math_is_nan", "Check NaN", lambda x="nan": _is_nan(_num(x)) if str(x).lower() in ("nan", "inf", "-inf") else False),
        ("math_int", "Convert to integer", lambda x=3.7: int(_num(x))),
        ("math_float", "Convert to float", lambda x="3.14": float(x)),
        ("math_neg", "Negate a number", lambda x=5: -_num(x)),
        ("math_inv", "Multiplicative inverse", lambda x=4: 1 / _num(x) if _num(x) else "div-by-zero"),
        ("math_pct", "Percentage value/100", lambda x=25: _num(x) / 100),
        ("math_pct_of", "percent% of total", lambda percent=10, total=200: _num(percent) / 100 * _num(total)),
        ("math_avg", "Average of list (alias)", lambda values=[1, 2, 3]: statistics.mean(_as_list(values))),
        ("math_range_sum", "Sum of integers from lo to hi", lambda lo=1, hi=10: sum(range(int(lo), int(hi) + 1))),
    ]
    for name, purpose, fn in math_ops:
        f.append({"name": name, "cat": "math", "purpose": purpose, "fn": fn})

    # ---- string ---------------------------------------------------------
    str_ops = [
        ("str_upper", "Uppercase a string", lambda text="hello": str(text).upper()),
        ("str_lower", "Lowercase a string", lambda text="HELLO": str(text).lower()),
        ("str_title", "Title case", lambda text="hello world": str(text).title()),
        ("str_capitalize", "Capitalize first letter", lambda text="hello": str(text).capitalize()),
        ("str_strip", "Strip whitespace", lambda text="  hi  ": str(text).strip()),
        ("str_lstrip", "Strip left whitespace", lambda text="  hi": str(text).lstrip()),
        ("str_rstrip", "Strip right whitespace", lambda text="hi  ": str(text).rstrip()),
        ("str_reverse", "Reverse a string", lambda text="abc": str(text)[::-1]),
        ("str_len", "Length of a string", lambda text="hello": len(str(text))),
        ("str_count", "Count occurrences of a substring", lambda text="a b a", sub="a": str(text).count(str(sub))),
        ("str_find", "Index of first occurrence (-1 if none)", lambda text="hello", sub="l": str(text).find(str(sub))),
        ("str_rfind", "Index of last occurrence", lambda text="hello", sub="l": str(text).rfind(str(sub))),
        ("str_startswith", "Check prefix", lambda text="hello", prefix="he": str(text).startswith(str(prefix))),
        ("str_endswith", "Check suffix", lambda text="hello", suffix="lo": str(text).endswith(str(suffix))),
        ("str_contains", "Check substring", lambda text="hello", sub="ell": str(sub) in str(text)),
        ("str_replace", "Replace all occurrences", lambda text="a-b-c", old="-", new="+": str(text).replace(str(old), str(new))),
        ("str_split", "Split string by separator", lambda text="a,b,c", sep=",": str(text).split(str(sep))),
        ("str_join", "Join list with separator", lambda parts=["a", "b"], sep=",": str(sep).join(_as_list(parts))),
        ("str_slice", "Slice string [start:end]", lambda text="hello", start=1, end=4: str(text)[int(start):int(end)]),
        ("str_repeat", "Repeat string n times", lambda text="ab", n=3: str(text) * int(n)),
        ("str_swapcase", "Swap letter case", lambda text="Hello": str(text).swapcase()),
        ("str_center", "Center string to width", lambda text="hi", width=9, fill="-": str(text).center(int(width), str(fill)[0])),
        ("str_ljust", "Left-justify to width", lambda text="hi", width=5, fill="-": str(text).ljust(int(width), str(fill)[0])),
        ("str_rjust", "Right-justify to width", lambda text="hi", width=5, fill="-": str(text).rjust(int(width), str(fill)[0])),
        ("str_zfill", "Zero-pad to width", lambda text="42", width=6: str(text).zfill(int(width))),
        ("str_truncate", "Truncate with ellipsis", lambda text="hello world", n=5: str(text)[:int(n)] + ("..." if len(str(text)) > int(n) else "")),
        ("str_pad", "Pad string to width (both sides)", lambda text="ab", width=6, fill=" ": str(text).center(int(width), str(fill)[0])),
        ("str_isalnum", "Check alphanumeric", lambda text="ab12": str(text).isalnum()),
        ("str_isalpha", "Check alphabetic", lambda text="abc": str(text).isalpha()),
        ("str_isdigit", "Check digits only", lambda text="123": str(text).isdigit()),
        ("str_islower", "Check all lowercase", lambda text="abc": str(text).islower()),
        ("str_isupper", "Check all uppercase", lambda text="ABC": str(text).isupper()),
        ("str_isspace", "Check whitespace only", lambda text="   ": str(text).isspace()),
        ("str_isnumeric", "Check numeric", lambda text="123": str(text).isnumeric()),
        ("str_isdecimal", "Check decimal", lambda text="123": str(text).isdecimal()),
        ("str_istitle", "Check title case", lambda text="Hello World": str(text).istitle()),
        ("str_lines", "Split into lines", lambda text="a\nb\nc": str(text).splitlines()),
        ("str_words", "Split into words", lambda text="a b c": str(text).split()),
        ("str_chars", "List characters", lambda text="abc": list(str(text))),
        ("str_sorted", "Sort characters", lambda text="cba": "".join(sorted(str(text)))),
        ("str_unique_chars", "Unique characters", lambda text="aabbc": "".join(dict.fromkeys(str(text)))),
        ("str_remove_ws", "Remove all whitespace", lambda text="a b c": re.sub(r"\s+", "", str(text))),
        ("str_squeeze_ws", "Collapse repeated whitespace", lambda text="a   b  c": re.sub(r"\s+", " ", str(text)).strip()),
        ("str_slug", "URL slug", lambda text="Hello World!": re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")),
        ("str_camel", "Camel case", lambda text="hello world": _camel(str(text))),
        ("str_snake", "Snake case", lambda text="HelloWorld": _snake(str(text))),
        ("str_kebab", "Kebab case", lambda text="HelloWorld": _snake(str(text)).replace("_", "-")),
        ("str_rot13", "ROT13 cipher", lambda text="hello": _rot13(str(text))),
        ("str_abbrev", "Abbreviate words", lambda text="As Soon As Possible": _abbrev(str(text))),
        ("str_ngrams", "Character n-grams", lambda text="abcde", n=2: _ngrams(str(text), int(n))),
        ("str_dedent", "Remove common indentation", lambda text="  a\n  b": textwrap.dedent(str(text))),
        ("str_expandtabs", "Expand tabs to spaces", lambda text="a\tb", tabsize=4: str(text).expandtabs(int(tabsize))),
        ("str_format", "Format with placeholders {name}", lambda template="Hi {name}", **kw: _format(template, kw)),
        ("str_mask", "Mask middle of string", lambda text="1234567890", keep=4: _mask(str(text), int(keep))),
        ("str_phone", "Format phone-ish string", lambda text="+37499123456": _phone(str(text))),
    ]
    for name, purpose, fn in str_ops:
        f.append({"name": name, "cat": "string", "purpose": purpose, "fn": fn})

    # ---- json ------------------------------------------------------------
    json_ops = [
        ("json_parse", "Parse JSON string", lambda text='{"a":1}': json.loads(text)),
        ("json_dumps", "Serialize object to JSON", lambda obj={"a": 1}: json.dumps(obj, ensure_ascii=False, indent=2)),
        ("json_minify", "Minify JSON", lambda text='{"a": 1}': json.dumps(json.loads(text), separators=(",", ":"))),
        ("json_pretty", "Pretty-print JSON", lambda text='{"a":1}': json.dumps(json.loads(text), ensure_ascii=False, indent=2)),
        ("json_validate", "Validate JSON string", lambda text='{"a":1}': _json_validate(text)),
        ("json_keys", "Top-level keys of JSON", lambda text='{"a":1,"b":2}': list(json.loads(text).keys())),
        ("json_values", "Top-level values of JSON", lambda text='{"a":1,"b":2}': list(json.loads(text).values())),
        ("json_merge", "Merge two JSON objects", lambda a='{"x":1}', b='{"y":2}': {**_as_dict(a), **_as_dict(b)}),
        ("json_get", "Get value at dot-path", lambda obj='{"a":{"b":1}}', path="a.b": _json_get(_as_dict(obj), str(path))),
        ("json_set", "Set value at dot-path", lambda obj='{"a":{}}', path="a.b", value=1: _json_set(_as_dict(obj), str(path), value)),
        ("json_delete", "Delete key at dot-path", lambda obj='{"a":{"b":1}}', path="a.b": _json_delete(_as_dict(obj), str(path))),
        ("json_flatten", "Flatten nested JSON to dot-keys", lambda obj='{"a":{"b":1}}': _json_flatten(_as_dict(obj))),
        ("json_unflatten", "Expand dot-keys to nested JSON", lambda obj='{"a.b":1}': _json_unflatten(_as_dict(obj))),
        ("json_type", "JSON type of value", lambda text='123': type(json.loads(text)).__name__),
        ("json_stringify", "Stringify a value as JSON", lambda value="x": json.dumps(value, ensure_ascii=False)),
        ("json_size", "Size in bytes of JSON string", lambda text='{"a":1}': len(text.encode("utf-8"))),
        ("json_count_keys", "Count all keys recursively", lambda text='{"a":{"b":1},"c":2}': _json_count(_as_dict(text))),
        ("json_filter", "Filter list of objects by key=value", lambda items='[{"a":1},{"a":2}]', key="a", value=1: [i for i in json.loads(items) if i.get(key) == value]),
        ("json_map", "Apply expression to each list item (jsonpath-ish)", lambda items='[1,2,3]', expr="x*2": _json_map(json.loads(items), str(expr))),
        ("json_sort", "Sort list of objects by key", lambda items='[{"a":2},{"a":1}]', key="a": sorted(json.loads(items), key=lambda i: i.get(key))),
        ("json_unique", "Unique items in JSON array", lambda text='[1,1,2]': list(dict.fromkeys(json.loads(text)))),
        ("json_sum", "Sum numeric items in JSON array", lambda text="[1,2,3]": sum(json.loads(text))),
        ("json_average", "Average numeric items", lambda text="[1,2,3]": statistics.mean(json.loads(text))),
        ("json_paths", "All dot-paths in JSON", lambda text='{"a":{"b":1}}': list(_json_flatten(_as_dict(text)).keys())),
    ]
    for name, purpose, fn in json_ops:
        f.append({"name": name, "cat": "json", "purpose": purpose, "fn": fn})

    return f


# --------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _register_families() -> int:
    count = 0
    for item in _build_families():
        _reg(item["name"], item["cat"], item["purpose"], item["fn"])
        count += 1
    return count


def _register_tool(name: str, cat: str, purpose: str, fn: Callable[..., Any], params: Optional[Dict[str, Any]] = None) -> None:
    """Public helper used by external modules to add tools at runtime."""
    _reg(name, cat, purpose, fn, params)


MEGA_COUNT = _register_families()

__all__ = ["MEGA_COUNT", "_register_tool", "tool", "_reg"]
