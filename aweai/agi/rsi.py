from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class CapabilityMetric:
    name: str
    score: float
    previous_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SafetyGuardrails:
    def __init__(self) -> None:
        self._rules: List[Callable[[Dict[str, Any]], bool]] = []
        self._blocked_patterns: List[str] = []
        self._resource_limits: Dict[str, float] = {}

    def add_rule(self, rule: Callable[[Dict[str, Any]], bool]) -> None:
        self._rules.append(rule)

    def add_blocked_pattern(self, pattern: str) -> None:
        self._blocked_patterns.append(pattern)

    def set_resource_limit(self, resource: str, limit: float) -> None:
        self._resource_limits[resource] = limit

    def check(self, context: Dict[str, Any]) -> Dict[str, Any]:
        violations = []
        for pattern in self._blocked_patterns:
            if pattern in str(context.get("code", "")):
                violations.append(f"Blocked pattern: {pattern}")
        for rule in self._rules:
            if not rule(context):
                violations.append(f"Rule violation: {rule}")
        return {"safe": len(violations) == 0, "violations": violations}

    def enforce(self, context: Dict[str, Any]) -> Dict[str, Any]:
        check = self.check(context)
        if not check["safe"]:
            return {"allowed": False, "reason": "; ".join(check["violations"])}
        return {"allowed": True}


class AlignmentChecker:
    def __init__(self) -> None:
        self._principles: List[str] = []
        self._history: List[Dict[str, Any]] = []

    def add_principle(self, principle: str) -> None:
        self._principles.append(principle)

    def check_alignment(self, action: Dict[str, Any]) -> Dict[str, Any]:
        violations = []
        for principle in self._principles:
            if principle.lower() not in str(action.get("description", "")).lower():
                if "harm" in principle.lower() or "safe" in principle.lower():
                    violations.append(f"Potential misalignment with: {principle}")
        result = {"aligned": len(violations) == 0, "violations": violations, "principles_checked": len(self._principles)}
        self._history.append(result)
        return result

    def alignment_history(self) -> List[Dict[str, Any]]:
        return list(self._history)


class SandboxedExecutor:
    def __init__(self, timeout: float = 5.0, max_memory_mb: float = 512.0) -> None:
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb

    def execute(self, code: str, language: str = "python") -> Dict[str, Any]:
        if language != "python":
            return {"success": False, "error": f"Unsupported language: {language}", "output": ""}
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error: {e}", "output": ""}
        with tempfile.TemporaryDirectory() as tmpdir:
            script = f"{tmpdir}/script.py"
            with open(script, "w") as f:
                f.write(code)
            try:
                result = subprocess.run(
                    ["python3", script],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                return {
                    "success": result.returncode == 0,
                    "output": result.stdout[:10000],
                    "error": result.stderr[:5000],
                    "returncode": result.returncode,
                }
            except subprocess.TimeoutExpired:
                return {"success": False, "error": "Execution timeout", "output": ""}
            except Exception as e:
                return {"success": False, "error": str(e), "output": ""}

    def execute_safe(self, code: str, allowed_modules: Optional[List[str]] = None) -> Dict[str, Any]:
        if allowed_modules is None:
            allowed_modules = ["math", "json", "datetime", "collections", "itertools", "functools"]
        sandboxed = f"""
import sys
sys.path = []
allowed = {allowed_modules!r}
for mod in allowed:
    __import__(mod)
"""
        sandboxed += code
        return self.execute(sandboxed, language="python")


class RSILoop:
    def __init__(self, executor: Optional[SandboxedExecutor] = None, guardrails: Optional[SafetyGuardrails] = None) -> None:
        self.executor = executor or SandboxedExecutor()
        self.guardrails = guardrails or SafetyGuardrails()
        self._metrics: List[CapabilityMetric] = []
        self._iterations: int = 0
        self._improvements: List[Dict[str, Any]] = []

    def improve(self, current_code: str, goal: str, max_iterations: int = 5) -> Dict[str, Any]:
        best_code = current_code
        best_score = self._evaluate(best_code, goal)
        for i in range(max_iterations):
            self._iterations += 1
            improved = self._generate_improvement(best_code, goal)
            check = self.guardrails.check({"code": improved})
            if not check["safe"]:
                continue
            score = self._evaluate(improved, goal)
            metric = CapabilityMetric(name=f"iteration_{i}", score=score, previous_score=best_score)
            self._metrics.append(metric)
            if score > best_score:
                best_score = score
                best_code = improved
                self._improvements.append({"iteration": i, "score": score, "delta": score - best_score})
        return {"code": best_code, "score": best_score, "iterations": self._iterations, "improvements": self._improvements}

    def _generate_improvement(self, code: str, goal: str) -> str:
        suggestions = [
            "Add error handling",
            "Optimize loops",
            "Add type hints",
            "Improve documentation",
            "Add input validation",
        ]
        import random
        suggestion = random.choice(suggestions)
        return f"# {suggestion}\n{code}"

    def _evaluate(self, code: str, goal: str) -> float:
        result = self.executor.execute(code)
        if not result.get("success", False):
            return 0.0
        base = 0.5
        if result.get("output"):
            base += 0.3
        if "error" not in result or not result["error"]:
            base += 0.2
        return min(1.0, base)

    def capability_metrics(self) -> List[CapabilityMetric]:
        return list(self._metrics)

    def iterations(self) -> int:
        return self._iterations
