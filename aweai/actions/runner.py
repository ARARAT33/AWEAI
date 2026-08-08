"""Action runner + pipeline execution for the model factory."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aweai.config import ensure_runtime_dirs
from aweai.actions.actions import parse_action
from aweai.errors import ActionError
from aweai.utils import read_json, write_json


def _execute(action: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if action == "train":
        from aweai.train import train

        mtype = kwargs.get("model_type", "mlp")
        name = kwargs.get("name", f"auto_{int(time.time())}")
        res = train(mtype, name, X=kwargs.get("X"), y=kwargs.get("y"),
                    data_path=kwargs.get("data_path"), params=kwargs.get("params"))
        if isinstance(res, dict):
            out = dict(res)
            out.setdefault("model", name)
            return {"result": out}
        return {"result": {"model": name, "model_type": mtype, "trained": True}}
    if action == "hardware":
        from aweai.hardware import detect
        from aweai.models.selector import pick_best_model

        hw = detect()
        best = pick_best_model(hw)
        return {"result": {"hardware": hw.to_dict(), "best_model": (best or {}).get("id")}}
    if action == "eval":
        from aweai.management import load_model
        from aweai.eval import classification_report

        model, meta = load_model(kwargs["name"])
        X = kwargs.get("X")
        if X is None:
            return {"result": {"metrics": meta.get("metrics", {})}}
        y = kwargs.get("y")
        pred = model.predict(X)
        report = classification_report(y, pred) if y is not None else {"pred": pred.tolist()}
        return {"result": report}
    if action == "export":
        from aweai.management import export_model

        return {"result": export_model(kwargs["name"], fmt=kwargs.get("fmt", "json"))}
    if action == "delete":
        from aweai.management import delete_model

        return {"result": {"deleted": delete_model(kwargs["name"])}}
    if action == "list":
        from aweai.management import list_models

        return {"result": list_models()}
    if action == "recommend":
        from aweai.selector import recommend

        return {"result": recommend(kwargs.get("task", "classification"))}
    if action == "load_data":
        from aweai.data import load_any

        ds = load_any(kwargs["path"])
        return {"result": ds.to_dict()}
    if action == "rag_index":
        from aweai.rag import RAGEngine

        eng = RAGEngine()
        return {"result": eng.index_directory(kwargs["path"])}
    if action == "rag_ask":
        from aweai.rag import RAGEngine

        eng = RAGEngine()
        return {"result": eng.ask(kwargs["query"])}
    raise ActionError(f"Unknown action: {action}")


def run_action(text: str, **kwargs) -> Dict[str, Any]:
    parsed = parse_action(text)
    parsed["kwargs"].update(kwargs)
    return _execute(parsed["action"], parsed["kwargs"])


def save_pipeline(name: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    pdir = ensure_runtime_dirs()["pipelines"]
    path = pdir / f"{name}.json"
    write_json(path, {"name": name, "steps": steps, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return {"name": name, "path": str(path), "steps": len(steps)}


def list_pipelines() -> List[Dict[str, Any]]:
    pdir = ensure_runtime_dirs()["pipelines"]
    out = []
    for f in sorted(pdir.glob("*.json")):
        data = read_json(f, {})
        out.append({"name": data.get("name", f.stem), "steps": len(data.get("steps", [])), "path": str(f)})
    return out


def run_pipeline(name: str) -> Dict[str, Any]:
    pdir = ensure_runtime_dirs()["pipelines"]
    data = read_json(pdir / f"{name}.json")
    if data is None:
        raise ActionError(f"Pipeline '{name}' not found")
    results = []
    for step in data.get("steps", []):
        action = step.get("action")
        if not action:
            raise ActionError(f"Pipeline step missing 'action': {step}")
        res = _execute(action, step.get("kwargs", {}))
        results.append({"step": step.get("name", action), "result": res})
    return {"pipeline": name, "steps": results, "count": len(results)}


def run_batch(actions: List[str]) -> Dict[str, Any]:
    results = []
    for a in actions:
        try:
            results.append({"action": a, "result": run_action(a)})
        except Exception as e:
            results.append({"action": a, "error": str(e)})
    return {"results": results, "count": len(results)}


class ActionsRunner:
    """Object-oriented action runner.

    Usage:
        runner = ActionsRunner(verbose=False)
        result = runner.run("hardware")          # -> {"status": "ok", ...}
        result = runner.run("train an mlp model named demo ...")
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def run(self, text: str, **kwargs) -> Dict[str, Any]:
        try:
            res = run_action(text, **kwargs)
            payload = res.get("result", res) if isinstance(res, dict) else res
            if isinstance(payload, dict):
                return {"status": "ok", **payload}
            return {"status": "ok", "result": payload}
        except Exception as e:  # pragma: no cover - defensive
            return {"status": "error", "error": str(e)}
