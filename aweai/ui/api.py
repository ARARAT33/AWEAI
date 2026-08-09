"""AWEAI UI — powerful browser interface for the model factory.

Endpoints:
  /                 SPA (index.html)
  /api/health       health check
  /api/hardware     hardware + recommendation
  /api/model-types  list model types
  /api/models       list zoo models
  /api/models/train create+train a model
  /api/models/eval  evaluate a model
  /api/models/export export a model
  /api/models/delete delete a model
  /api/data/load    load dataset info
  /api/data/augment augment texts
  /api/rag/index    index documents
  /api/rag/ask      ask RAG
  /api/actions/run  run natural-language action
  /api/autotest     run autotest (the Autotest button)
  /api/languages    list languages
  /api/config       get/set config
  /api/quantize     quantize a model
  /api/export/edge  edge export
  /api/edge/footprint  edge footprint estimate
  /api/market       marketplace (publish/search/list/info/download/rate/stats)
  /api/integrations AI-tool integrations list
  /api/integrations/chat  chat via a provider
  /api/terminal     in-app terminal
  /api/allc         10,000+ command catalog
  /api/autoallc     10,000+ automation catalog
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aweai import __version__
from aweai.ports import resolve_port

STATIC_DIR = Path(__file__).parent / "static"


class TrainRequest(BaseModel):
    model_type: str = "mlp"
    name: str = "model_1"
    data_path: Optional[str] = None
    target: Optional[str] = None
    text_path: Optional[str] = None
    params: Dict[str, Any] = {}
    normalize: Optional[str] = None


class EvalRequest(BaseModel):
    name: str
    data_path: Optional[str] = None
    target: Optional[str] = None


class ExportRequest(BaseModel):
    name: str
    fmt: str = "json"


class DeleteRequest(BaseModel):
    name: str


class DataRequest(BaseModel):
    path: str
    target: Optional[str] = None


class AugmentRequest(BaseModel):
    texts: List[str]
    n: int = 1


class RagIndexRequest(BaseModel):
    path: Optional[str] = None
    texts: Optional[List[str]] = None


class RagAskRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class ActionRequest(BaseModel):
    text: str


class ConfigRequest(BaseModel):
    key: str
    value: Any = None


class QuantizeRequest(BaseModel):
    name: str
    fmt: str = "int8"


class EdgeExportRequest(BaseModel):
    name: str
    fmt: str = "onnx"
    quantize: Optional[str] = None


class MarketRequest(BaseModel):
    action: str
    arg: Optional[str] = None
    tag: Optional[str] = None
    description: Optional[str] = None
    stars: Optional[float] = None


class TerminalRequest(BaseModel):
    line: str


class IntegrationChatRequest(BaseModel):
    provider: str
    message: str


def create_app() -> FastAPI:
    app = FastAPI(title="AWEAI — AI Model Factory", version=__version__, docs_url="/docs", redoc_url="/redoc")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__, "name": "AWEAI Model Factory"}

    @app.get("/api/hardware")
    def api_hardware():
        from aweai.hardware import detect
        from aweai.selector import recommend

        hw = detect()
        return {"hardware": hw.to_dict(), "recommendation": recommend("classification", hw)}

    @app.get("/api/model-types")
    def api_model_types():
        from aweai.models.registry import MODEL_TYPES

        return {"types": [{"name": k, "task": v["task"], "desc": v["desc"]} for k, v in MODEL_TYPES.items()]}

    @app.get("/api/models")
    def api_models():
        from aweai.management import list_models

        return {"models": list_models()}

    @app.post("/api/models/train")
    def api_train(req: TrainRequest):
        from aweai.train import train

        try:
            res = train(
                req.model_type, req.name, data_path=req.data_path, text_path=req.text_path,
                target=req.target, params=dict(req.params), normalize=req.normalize,
            )
            return {"ok": True, "result": res}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/models/eval")
    def api_eval(req: EvalRequest):
        from aweai.data import load_any
        from aweai.eval import classification_report
        from aweai.management import load_model

        try:
            model, meta = load_model(req.name)
            if req.data_path:
                ds = load_any(req.data_path, target_column=req.target or None)
                pred = model.predict(ds.X if ds.X is not None else ds.texts)
                report = classification_report(ds.y, pred) if ds.y is not None else {"pred": pred.tolist()}
            else:
                report = {"metrics": meta.get("metrics", {})}
            return {"ok": True, "result": report}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/models/export")
    def api_export(req: ExportRequest):
        from aweai.management import export_model

        try:
            return {"ok": True, "result": export_model(req.name, fmt=req.fmt)}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/models/delete")
    def api_delete(req: DeleteRequest):
        from aweai.management import delete_model

        try:
            return {"ok": True, "result": {"deleted": delete_model(req.name)}}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/data/load")
    def api_data_load(req: DataRequest):
        from aweai.data import load_any

        try:
            ds = load_any(req.path, target_column=req.target or None)
            return {"ok": True, "result": ds.to_dict()}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/data/augment")
    def api_data_augment(req: AugmentRequest):
        from aweai.data import text_augment

        try:
            out = [t for txt in req.texts for t in text_augment(txt, n=req.n)]
            return {"ok": True, "result": {"augmented": out}}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/rag/index")
    def api_rag_index(req: RagIndexRequest):
        from aweai.rag import RAGEngine

        try:
            eng = RAGEngine()
            if req.path:
                res = eng.index_directory(req.path)
            else:
                res = eng.index_documents(req.texts or [])
            return {"ok": True, "result": res}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/rag/ask")
    def api_rag_ask(req: RagAskRequest):
        from aweai.rag import RAGEngine

        try:
            eng = RAGEngine()
            return {"ok": True, "result": eng.ask(req.query, top_k=req.top_k)}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/actions/run")
    def api_actions(req: ActionRequest):
        from aweai.actions import run_action

        try:
            return {"ok": True, "result": run_action(req.text)}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/autotest")
    def api_autotest(quick: bool = False, no_ui: bool = False):
        from aweai.autotest import run_autotest

        report = run_autotest(quick=quick, no_ui=no_ui, verbose=False)
        return report

    @app.get("/api/languages")
    def api_languages():
        from aweai.i18n import language_names

        return {"languages": language_names()}

    @app.get("/api/config")
    def api_config_get():
        from aweai.config import get_config

        return {"config": get_config().all()}

    @app.post("/api/config")
    def api_config_set(req: ConfigRequest):
        from aweai.config import get_config

        try:
            get_config().set(req.key, req.value)
            return {"ok": True}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/quantize")
    def api_quantize(req: QuantizeRequest):
        from aweai.quantize import quantize_model

        try:
            return {"ok": True, "result": quantize_model(req.name, fmt=req.fmt)}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/export/edge")
    def api_export_edge(req: EdgeExportRequest):
        from aweai.export import export_edge

        try:
            return {"ok": True, "result": export_edge(req.name, fmt=req.fmt, quantize=req.quantize)}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/edge/footprint")
    def api_edge_footprint(name: str):
        from aweai.export import estimate_edge_footprint

        try:
            return {"ok": True, "result": estimate_edge_footprint(name)}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/market")
    def api_market(req: MarketRequest):
        from aweai import market as mkt

        try:
            if req.action == "publish":
                return {"ok": True, "result": mkt.publish(req.arg or "", tag=req.tag or "v1", description=req.description or "")}
            if req.action == "search":
                return {"ok": True, "result": mkt.search(req.arg or "")}
            if req.action == "list":
                return {"ok": True, "result": mkt.list_listings()}
            if req.action == "info":
                return {"ok": True, "result": mkt.info(req.arg or "")}
            if req.action == "download":
                return {"ok": True, "result": mkt.download(req.arg or "")}
            if req.action == "rate":
                parts = (req.arg or "").split()
                if len(parts) < 2:
                    raise ValueError("rate requires <id> <stars>")
                return {"ok": True, "result": mkt.rate(parts[0], float(parts[1]))}
            if req.action == "stats":
                return {"ok": True, "result": mkt.stats()}
            raise ValueError(f"Unknown market action: {req.action}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/integrations")
    def api_integrations():
        from aweai.integrations import list_tools

        return {"ok": True, "result": list_tools()}

    @app.post("/api/integrations/chat")
    def api_integrations_chat(req: IntegrationChatRequest):
        from aweai.integrations import chat

        return {"ok": True, "result": chat(req.provider, req.message)}

    @app.post("/api/terminal")
    def api_terminal(req: TerminalRequest):
        from aweai.terminal import run as term_run

        return {"ok": True, "result": term_run(req.line)}

    @app.get("/api/allc")
    def api_allc(search: str = "", category: str = "", count: int = 50):
        from aweai.menus import build_catalog, search_catalog

        items = build_catalog(min_count=10000)
        if search or category:
            items = search_catalog(items, query=search, category=category)
        return {"ok": True, "total": len(items), "items": items[:count]}

    @app.get("/api/autoallc")
    def api_autoallc(search: str = "", category: str = "", count: int = 50):
        from aweai.menus import build_automations, search_catalog

        items = build_automations(min_count=10000)
        if search or category:
            items = search_catalog(items, query=search, category=category)
        return {"ok": True, "total": len(items), "items": items[:count]}

    return app


def serve(port: int = 8888, host: str = "127.0.0.1", open_browser: bool = True) -> None:
    import threading
    import webbrowser

    import uvicorn

    resolved = resolve_port(port, host)
    print(f"AWEAI Model Factory UI → http://{host}:{resolved}  (docs at /docs)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{resolved}")).start()
    uvicorn.run(create_app(), host=host, port=resolved, log_level="info")
