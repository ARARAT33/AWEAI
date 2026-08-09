"""AWEAI testing tools — unit, coverage, fuzz, benchmarks, smoke checks.

Each tool has a unique purpose and works with pytest (optional).
"""

from __future__ import annotations

import json
import random
import string
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aweai.tools.registry import tool


@tool("test_run", "testing", "Run pytest on a path and report summary")
def test_run(path: str = "tests", verbose: bool = False) -> Dict[str, Any]:
    import subprocess

    flag = "-v" if verbose else "-q"
    try:
        res = subprocess.run(["python", "-m", "pytest", path, flag], capture_output=True, text=True, timeout=300)
        return {"path": path, "returncode": res.returncode, "output": (res.stdout + res.stderr)[-3000:]}
    except Exception as e:
        return {"path": path, "error": str(e)}


@tool("test_collect", "testing", "List tests collected by pytest (no run)")
def test_collect(path: str = "tests") -> Dict[str, Any]:
    import subprocess

    try:
        res = subprocess.run(["python", "-m", "pytest", path, "--collect-only", "-q"], capture_output=True, text=True, timeout=120)
        lines = [l for l in res.stdout.splitlines() if "::" in l]
        return {"tests": lines, "count": len(lines)}
    except Exception as e:
        return {"error": str(e)}


@tool("test_smoke", "testing", "Run a minimal smoke test (import + basic call)")
def test_smoke(module: str = "aweai") -> Dict[str, Any]:
    import importlib

    try:
        importlib.import_module(module)
        return {"module": module, "import_ok": True}
    except Exception as e:
        return {"module": module, "import_ok": False, "error": str(e)}


@tool("test_assert", "testing", "Evaluate a Python expression and return the assertion result")
def test_assert(expression: str) -> Dict[str, Any]:
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


@tool("fuzz_string", "testing", "Generate random string inputs (fuzzing corpus)")
def fuzz_string(n: int = 10, length: int = 20, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    corpus = []
    for _ in range(n):
        corpus.append("".join(rng.choice(string.ascii_letters + string.digits + " !@#$%^&*()_+=") for _ in range(length)))
    return {"corpus": corpus, "count": len(corpus)}


@tool("fuzz_numbers", "testing", "Generate random number inputs (fuzzing corpus)")
def fuzz_numbers(n: int = 10, lo: int = -100, hi: int = 100, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    return {"corpus": [rng.randint(lo, hi) for _ in range(n)], "count": n}


@tool("fuzz_json", "testing", "Generate random JSON-ish payloads (fuzzing corpus)")
def fuzz_json(n: int = 5, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    corpus = []
    for _ in range(n):
        corpus.append({
            "id": rng.randint(1, 9999),
            "name": "".join(rng.choice(string.ascii_lowercase) for _ in range(8)),
            "active": rng.choice([True, False]),
            "tags": [rng.choice(["a", "b", "c"]) for _ in range(rng.randint(0, 3))],
        })
    return {"corpus": corpus, "count": len(corpus)}


@tool("benchmark_time", "testing", "Time a Python expression (avg over runs)")
def benchmark_time(expression: str = "sum(range(1000))", runs: int = 5) -> Dict[str, Any]:
    import timeit

    try:
        t = timeit.timeit(expression, number=runs)
        return {"expression": expression, "runs": runs, "total_seconds": round(t, 5), "avg_seconds": round(t / runs, 6)}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


@tool("benchmark_memory", "testing", "Approximate memory usage of a Python expression (tracemalloc)")
def benchmark_memory(expression: str = "list(range(1000))") -> Dict[str, Any]:
    import tracemalloc

    tracemalloc.start()
    try:
        eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        current, peak = tracemalloc.get_traced_memory()
        return {"expression": expression, "current_bytes": current, "peak_bytes": peak}
    finally:
        tracemalloc.stop()


@tool("coverage_report", "testing", "Run coverage on a module (if coverage installed)")
def coverage_report(module: str = "aweai", path: str = "tests") -> Dict[str, Any]:
    import subprocess

    try:
        res = subprocess.run(
            ["python", "-m", "coverage", "run", "-m", "pytest", path, "-q"],
            capture_output=True, text=True, timeout=300,
        )
        report = subprocess.run(["python", "-m", "coverage", "report", "-m"], capture_output=True, text=True, timeout=60)
        return {"returncode": res.returncode, "report": report.stdout[-3000:]}
    except Exception as e:
        return {"error": str(e)}


@tool("assert_true", "testing", "Assert that a value is truthy and return pass/fail")
def assert_true(value: bool = True, message: str = "") -> Dict[str, Any]:
    ok = bool(value)
    return {"passed": ok, "message": message or ("ok" if ok else "assertion failed")}


@tool("assert_equal", "testing", "Assert that two values are equal")
def assert_equal(actual: Any = None, expected: Any = None) -> Dict[str, Any]:
    return {"passed": actual == expected, "actual": actual, "expected": expected}


@tool("assert_contains", "testing", "Assert that a string contains a substring")
def assert_contains(text: str, needle: str) -> Dict[str, Any]:
    return {"passed": needle in text, "text": text[:100], "needle": needle}


@tool("test_property", "testing", "Property-style check: run a function n times on random inputs")
def test_property(expression: str = "lambda x: x + 1 > x", n: int = 10, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    try:
        fn = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        failures = 0
        for _ in range(n):
            x = rng.randint(-1000, 1000)
            if not fn(x):
                failures += 1
        return {"expression": expression, "runs": n, "failures": failures, "passed": failures == 0}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


@tool("test_compare_perf", "testing", "Compare timing of two expressions")
def test_compare_perf(a: str = "sum(range(1000))", b: str = "sum(x*x for x in range(1000))", runs: int = 3) -> Dict[str, Any]:
    import timeit

    ta = timeit.timeit(a, number=runs)
    tb = timeit.timeit(b, number=runs)
    return {
        "a": {"expr": a, "seconds": round(ta, 5)},
        "b": {"expr": b, "seconds": round(tb, 5)},
        "winner": "a" if ta < tb else "b",
    }


@tool("test_verify_cli", "testing", "Run a CLI command and verify exit code is 0")
def test_verify_cli(command: str) -> Dict[str, Any]:
    import subprocess

    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        return {"command": command, "returncode": res.returncode, "passed": res.returncode == 0, "output": (res.stdout + res.stderr)[-1000:]}
    except Exception as e:
        return {"command": command, "error": str(e)}


__all__ = []
