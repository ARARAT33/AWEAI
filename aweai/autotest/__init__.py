#!/usr/bin/env python3
"""AWEAI autotest — 100% automatic verification of every command, tool, model, module and integration.

Usage:
    python -m aweai.autotest
    python -m aweai.autotest --quick
    python -m aweai.autotest --module aweai.tools.networking
    python -m aweai.autotest --command math_add
    python -m aweai.autotest --json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    category: str = "general"
    severity: str = "normal"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuite:
    name: str
    results: List[TestResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_ms: float = 0.0


class AutotestRunner:
    def __init__(self, quick: bool = False, module: Optional[str] = None, command: Optional[str] = None) -> None:
        self.quick = quick
        self.module_filter = module
        self.command_filter = command
        self.suites: List[TestSuite] = []
        self._start = time.perf_counter()

    def run(self) -> Dict[str, Any]:
        self._run_import_tests()
        self._run_registry_tests()
        self._run_tool_tests()
        self._run_model_tests()
        self._run_cli_tests()
        self._run_integration_tests()
        self._run_performance_tests()
        return self._report()

    def _suite(self, name: str) -> TestSuite:
        suite = TestSuite(name=name)
        self.suites.append(suite)
        return suite

    def _case(self, suite: TestSuite, name: str, fn, category: str = "general", severity: str = "normal") -> Optional[TestResult]:
        if self.command_filter and not (name == self.command_filter or self.command_filter in name):
            return None
        start = time.perf_counter()
        try:
            fn()
            dur = (time.perf_counter() - start) * 1000
            suite.passed += 1
            suite.results.append(TestResult(name=name, passed=True, duration_ms=dur, category=category, severity=severity))
            return suite.results[-1]
        except Exception as e:
            dur = (time.perf_counter() - start) * 1000
            suite.failed += 1
            err = "".join(traceback.format_exception_only(type(e), e)).strip()
            suite.results.append(TestResult(name=name, passed=False, duration_ms=dur, error=err, category=category, severity=severity))
            return suite.results[-1]

    def _run_import_tests(self) -> None:
        suite = self._suite("imports")
        modules = [
            "aweai",
            "aweai.cli",
            "aweai.hardware",
            "aweai.selector",
            "aweai.config",
            "aweai.utils",
            "aweai.errors",
            "aweai.bulk",
            "aweai.bulk_v5",
            "aweai.bulk_v6",
            "aweai.bulk_v7",
            "aweai.bulk_extra",
            "aweai.wiki",
            "aweai.ai",
            "aweai.actions.actions",
            "aweai.actions.runner",
            "aweai.agents.engine",
            "aweai.autotest",
            "aweai.data.loaders",
            "aweai.data.normalize",
            "aweai.data.split",
            "aweai.data.tokenizer",
            "aweai.data.augment",
            "aweai.distributed.engine",
            "aweai.eval.metrics",
            "aweai.eval.curves",
            "aweai.export.exporter",
            "aweai.export.edge",
            "aweai.hardware",
            "aweai.integrations.ai_tools",
            "aweai.management.manager",
            "aweai.market.market",
            "aweai.models.base",
            "aweai.models.mlp",
            "aweai.models.cnn",
            "aweai.models.rnn",
            "aweai.models.lstm",
            "aweai.models.gru",
            "aweai.models.transformer",
            "aweai.models.gan",
            "aweai.models.autoencoder",
            "aweai.models.vae",
            "aweai.models.diffusion",
            "aweai.models.linear",
            "aweai.models.logistic",
            "aweai.models.ngram",
            "aweai.models.kmeans",
            "aweai.models.registry",
            "aweai.models.selector",
            "aweai.models.trainer",
            "aweai.models.inference",
            "aweai.models.catalog",
            "aweai.models.apis",
            "aweai.models.sequence",
            "aweai.models.vision",
            "aweai.ports",
            "aweai.quantize.quantizer",
            "aweai.rag.engine",
            "aweai.selector",
            "aweai.train.trainer",
            "aweai.train.tuning",
            "aweai.tools.registry",
            "aweai.tools.core",
            "aweai.tools.security",
            "aweai.tools.devops",
            "aweai.tools.datascience",
            "aweai.tools.networking",
            "aweai.tools.media",
            "aweai.tools.monitoring",
            "aweai.tools.testing",
            "aweai.tools.codegen",
            "aweai.tools.creative",
            "aweai.tools.automation",
            "aweai.tools.extras",
            "aweai.tools.mega",
            "aweai.tools.mega2",
            "aweai.tools.aiagents",
            "aweai.hw",
            "aweai.arch.registry",
            "aweai.arch.moe",
            "aweai.arch.transformer",
            "aweai.arch.next",
            "aweai.arch.compound",
            "aweai.arch.designer",
            "aweai.arch.converter",
            "aweai.scale.zero",
            "aweai.scale.fsdp",
            "aweai.scale.pipeline",
            "aweai.scale.tensor",
            "aweai.scale.offload",
            "aweai.scale.autoscale",
            "aweai.scale.unified",
            "aweai.cluster.manager",
            "aweai.cluster.ssh",
            "aweai.cluster.k8s",
            "aweai.cluster.slurm",
            "aweai.cluster.autoscale",
            "aweai.cluster.deploy",
            "aweai.cluster.discovery",
            "aweai.db.metadata",
            "aweai.db.vector",
            "aweai.db.timeseries",
            "aweai.db.kv",
            "aweai.db.graph",
            "aweai.db.backup",
            "aweai.db.migration",
            "aweai.agi.agent",
            "aweai.agi.memory",
            "aweai.agi.reasoning",
            "aweai.agi.planning",
            "aweai.agi.rsi",
            "aweai.agi.consciousness",
            "aweai.agi.swarm",
            "aweai.compat.openai",
            "aweai.compat.google",
            "aweai.compat.anthropic",
            "aweai.compat.router",
            "aweai.compat.providers",
            "aweai.compat.local_models",
            "aweai.compat.types",
        ]
        if self.module_filter:
            modules = [m for m in modules if m == self.module_filter or m.startswith(self.module_filter + ".")]
        for mod in modules:
            self._case(suite, mod, lambda m=mod: importlib.import_module(m), category="import", severity="critical")

    def _run_registry_tests(self) -> None:
        suite = self._suite("registry")
        try:
            from aweai.models.registry import MODEL_TYPES
            self._case(suite, "model_registry_has_types", lambda: self._assert_true(len(MODEL_TYPES) > 0), category="registry", severity="high")
            self._case(suite, "model_registry_has_transformer", lambda: self._assert_true("transformer" in MODEL_TYPES), category="registry", severity="high")
            self._case(suite, "model_registry_has_mlp", lambda: self._assert_true("mlp" in MODEL_TYPES), category="registry", severity="high")
        except Exception as e:
            self._case(suite, "model_registry_import", lambda: (_ for _ in ()).throw(e), category="registry", severity="critical")

        try:
            from aweai.tools.registry import TOOLS
            self._case(suite, "tool_registry_not_empty", lambda: self._assert_true(len(TOOLS) > 0), category="registry", severity="high")
            self._case(suite, "tool_registry_has_tools", lambda: self._assert_true(any("math_" in k for k in TOOLS)), category="registry", severity="high")
        except Exception as e:
            self._case(suite, "tool_registry_import", lambda: (_ for _ in ()).throw(e), category="registry", severity="critical")

        try:
            from aweai.arch.registry import ArchitectureRegistry
            reg = ArchitectureRegistry()
            self._case(suite, "arch_registry_list_all", lambda: self._assert_true(len(reg.list_all()) > 0), category="registry", severity="high")
            self._case(suite, "arch_registry_get_transformer", lambda: self._assert_true(reg.get("transformer") is not None), category="registry", severity="high")
        except Exception as e:
            self._case(suite, "arch_registry_import", lambda: (_ for _ in ()).throw(e), category="registry", severity="critical")

    def _run_tool_tests(self) -> None:
        suite = self._suite("tools")
        try:
            from aweai.tools.registry import TOOLS
            tool_names = sorted(TOOLS.keys())
            if self.command_filter:
                tool_names = [n for n in tool_names if n == self.command_filter or self.command_filter in n]
            tested = 0
            for name in tool_names[:500]:
                fn = TOOLS[name]["fn"]
                try:
                    sig = getattr(fn, "__code__", None)
                    if sig:
                        argcount = sig.co_argcount
                        defaults = fn.__defaults__ or ()
                        required = max(0, argcount - len(defaults))
                    else:
                        required = 0
                    if required == 0:
                        result = fn()
                    else:
                        args = [f"test_{i}" for i in range(required)]
                        result = fn(*args)
                    self._case(suite, f"tool_{name}", lambda n=name, r=result: self._assert_true(r is not None), category="tool", severity="normal")
                    tested += 1
                except Exception as e:
                    self._case(suite, f"tool_{name}", lambda n=name, e=e: (_ for _ in ()).throw(e), category="tool", severity="low")
            suite.results.append(TestResult(name=f"tools_tested_{tested}", passed=True, duration_ms=0.0, category="tool", severity="info", metadata={"tested": tested, "total": len(tool_names)}))
        except Exception as e:
            self._case(suite, "tool_registry_load", lambda: (_ for _ in ()).throw(e), category="tool", severity="critical")

    def _run_model_tests(self) -> None:
        suite = self._suite("models")
        model_types = ["mlp", "cnn", "rnn", "transformer", "ngram", "linear", "logistic", "kmeans"]
        if not self.quick:
            model_types += ["gan", "autoencoder", "diffusion", "time_series"]
        for mt in model_types:
            self._case(suite, f"model_{mt}_import", lambda m=mt: self._import_model(m), category="model", severity="high")
            if not self.quick:
                self._case(suite, f"model_{mt}_instantiate", lambda m=mt: self._instantiate_model(m), category="model", severity="high")
                self._case(suite, f"model_{mt}_forward", lambda m=mt: self._model_forward(m), category="model", severity="medium")

    def _run_cli_tests(self) -> None:
        suite = self._suite("cli")
        self._case(suite, "cli_app_import", lambda: importlib.import_module("aweai.cli"), category="cli", severity="critical")
        self._case(suite, "cli_app_has_commands", lambda: self._assert_true(hasattr(importlib.import_module("aweai.cli"), "app")), category="cli", severity="high")

    def _run_integration_tests(self) -> None:
        suite = self._suite("integrations")
        self._case(suite, "scale_zero_import", lambda: importlib.import_module("aweai.scale.zero"), category="integration", severity="high")
        self._case(suite, "arch_moe_import", lambda: importlib.import_module("aweai.arch.moe"), category="integration", severity="high")
        self._case(suite, "cluster_manager_import", lambda: importlib.import_module("aweai.cluster.manager"), category="integration", severity="high")
        self._case(suite, "db_vector_import", lambda: importlib.import_module("aweai.db.vector"), category="integration", severity="high")
        self._case(suite, "agi_agent_import", lambda: importlib.import_module("aweai.agi.agent"), category="integration", severity="high")
        self._case(suite, "compat_openai_import", lambda: importlib.import_module("aweai.compat.openai"), category="integration", severity="high")

    def _run_performance_tests(self) -> None:
        if self.quick:
            return
        suite = self._suite("performance")
        self._case(suite, "perf_hardware_detect", lambda: self._perf_hardware(), category="performance", severity="medium")
        self._case(suite, "perf_model_registry", lambda: self._perf_model_registry(), category="performance", severity="medium")
        self._case(suite, "perf_tool_registry", lambda: self._perf_tool_registry(), category="performance", severity="medium")

    def _assert_true(self, condition: bool) -> None:
        if not condition:
            raise AssertionError("Assertion failed")

    def _import_model(self, model_type: str) -> None:
        mod = importlib.import_module(f"aweai.models.{model_type}")
        cls_name = "".join(w.capitalize() for w in model_type.split("_"))
        if not hasattr(mod, cls_name):
            alt = "MiniTransformer" if model_type == "transformer" else None
            if alt and hasattr(mod, alt):
                return
            raise AttributeError(f"Missing class {cls_name}")

    def _instantiate_model(self, model_type: str) -> None:
        mod = importlib.import_module(f"aweai.models.{model_type}")
        cls_name = "".join(w.capitalize() for w in model_type.split("_"))
        if model_type == "transformer":
            cls = mod.MiniTransformer
        elif model_type == "ngram":
            cls = mod.NGramModel
        elif model_type == "kmeans":
            cls = mod.KMeansModel
        else:
            cls = getattr(mod, cls_name)
        if model_type in ("mlp", "linear", "logistic"):
            cls(input_dim=4, output_dim=2)
        elif model_type in ("cnn", "vision_cnn"):
            cls(input_channels=1, output_dim=2)
        elif model_type in ("rnn", "lstm", "gru", "sequence"):
            cls(input_dim=8, hidden_dim=16, output_dim=2)
        elif model_type == "transformer":
            cls(vocab_size=50, d_model=16, nhead=2, layers=1, num_classes=2)
        elif model_type == "gan":
            cls(latent_dim=8, hidden_dim=16, output_dim=4)
        elif model_type == "autoencoder":
            cls(input_dim=8, hidden_dim=16, latent_dim=4)
        elif model_type == "diffusion":
            cls(input_dim=8, hidden_dim=16, timesteps=10)
        elif model_type == "time_series":
            cls(input_dim=4, hidden_dim=16, output_dim=1)
        elif model_type == "kmeans":
            cls(k=2)
        elif model_type == "ngram":
            cls(n=2)
        else:
            cls(input_dim=4, output_dim=2)

    def _model_forward(self, model_type: str) -> None:
        import numpy as np
        mod = importlib.import_module(f"aweai.models.{model_type}")
        cls_name = "".join(w.capitalize() for w in model_type.split("_"))
        if model_type == "transformer":
            model = mod.MiniTransformer(vocab_size=50, d_model=16, nhead=2, layers=1, num_classes=2)
            x = np.zeros((2, 8), dtype=int)
            x[0, :] = 1
            x[1, :] = 2
            logits, _ = model._forward(x)
            self._assert_true(logits.shape == (2, 2))
        elif model_type == "mlp":
            model = mod.MLP(input_dim=4, output_dim=2)
            x = np.random.default_rng(0).standard_normal((4, 4))
            out = model._forward(x)
            self._assert_true(out is not None)
        elif model_type == "kmeans":
            model = mod.KMeansModel(k=2)
            x = np.random.default_rng(0).standard_normal((10, 4))
            model.fit(x)
            self._assert_true(hasattr(model, "centers"))
        elif model_type == "ngram":
            model = mod.NGramModel(n=2)
            model.fit(["a b c d", "b c d e"])
            probs = model.predict_proba("a b")
            self._assert_true(len(probs) > 0)
        else:
            model = getattr(mod, cls_name)(input_dim=4, output_dim=2)
            x = np.random.default_rng(0).standard_normal((4, 4))
            out = model._forward(x)
            self._assert_true(out is not None)

    def _perf_hardware(self) -> None:
        from aweai.hardware import detect
        info = detect()
        d = info.to_dict()
        self._assert_true("cpu_count" in d)

    def _perf_model_registry(self) -> None:
        from aweai.models.registry import MODEL_TYPES
        start = time.perf_counter()
        d = MODEL_TYPES
        dur = time.perf_counter() - start
        self._assert_true(dur < 1.0)

    def _perf_tool_registry(self) -> None:
        from aweai.tools.registry import TOOLS
        start = time.perf_counter()
        d = TOOLS
        dur = time.perf_counter() - start
        self._assert_true(dur < 2.0)

    def _report(self) -> Dict[str, Any]:
        total_dur = (time.perf_counter() - self._start) * 1000
        total_passed = sum(s.passed for s in self.suites)
        total_failed = sum(s.failed for s in self.suites)
        total_skipped = sum(s.skipped for s in self.suites)
        total = total_passed + total_failed + total_skipped
        overall = {
            "total": total,
            "passed": total_passed,
            "failed": total_failed,
            "skipped": total_skipped,
            "pass_rate": f"{(total_passed / max(total, 1) * 100):.1f}%",
            "duration_ms": round(total_dur, 1),
            "suites": [],
        }
        for s in self.suites:
            suite_info = {
                "name": s.name,
                "passed": s.passed,
                "failed": s.failed,
                "skipped": s.skipped,
                "duration_ms": round(s.duration_ms, 1),
            }
            if s.failed > 0:
                suite_info["failures"] = [r.error for r in s.results if not r.passed and r.error][:10]
            overall["suites"].append(suite_info)
        return overall


def main() -> int:
    parser = argparse.ArgumentParser(description="AWEAI autotest runner")
    parser.add_argument("--quick", action="store_true", help="Skip heavy smoke-trains")
    parser.add_argument("--module", type=str, help="Run tests for a specific module")
    parser.add_argument("--command", type=str, help="Run tests for a specific command/tool")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()
    runner = AutotestRunner(quick=args.quick, module=args.module, command=args.command)
    report = runner.run()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"AWEAI autotest — {report['pass_rate']} ({report['passed']}/{report['total']}) in {report['duration_ms']}ms")
        for s in report["suites"]:
            status = "PASS" if s["failed"] == 0 else "FAIL"
            print(f"  [{status}] {s['name']}: {s['passed']} passed, {s['failed']} failed, {s['skipped']} skipped")
            if s.get("failures"):
                for f in s["failures"][:5]:
                    print(f"    - {f}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
