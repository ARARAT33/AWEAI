"""FastAPI application and REST API for the AWEAI web UI."""

from __future__ import annotations

import json
import threading
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aweai.config import get_config
from aweai.i18n import LANGUAGES, available_languages
from aweai.ports import resolve_port


class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []
    model: Optional[str] = None


class TrainRequest(BaseModel):
    name: str
    data: str
    mode: str = "scratch"  # scratch | finetune | continue
    base_model: Optional[str] = None
    epochs: int = 1


class RagIndexRequest(BaseModel):
    path: str = ""


class RagAskRequest(BaseModel):
    query: str
    top_k: int = 4


class AgentRunRequest(BaseModel):
    task: str
    max_steps: int = 5


class ActionRequest(BaseModel):
    text: str
    lang: str = "en"


class ConfigUpdateRequest(BaseModel):
    values: Dict


def create_app() -> FastAPI:
    app = FastAPI(title="AWEAI", version="2.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
    cfg = get_config()

    # ---------- system ----------
    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "2.0.0"}

    @app.get("/api/languages")
    def languages():
        return {"languages": available_languages()}

    @app.get("/api/config")
    def get_cfg():
        return {"config": cfg.all()}

    @app.post("/api/config")
    def set_cfg(req: ConfigUpdateRequest):
        cfg.update(req.values)
        return {"config": cfg.all()}

    # ---------- hardware & models ----------
    @app.get("/api/hardware")
    def hardware():
        from aweai.hardware import detect

        return detect().to_dict()

    @app.get("/api/models")
    def models():
        from aweai.models.registry import ModelRegistry

        reg = ModelRegistry()
        return {"catalog": reg.catalog(), "installed": reg.installed()}

    @app.get("/api/models/recommended")
    def recommended():
        from aweai.hardware import detect
        from aweai.models.selector import pick_best_model, suggest_models

        hw = detect()
        best = pick_best_model(hw)
        return {
            "hardware": hw.to_dict(),
            "best": best,
            "suggestions": suggest_models(hw, limit=5),
        }

    # ---------- chat ----------
    @app.post("/api/chat")
    def chat(req: ChatRequest):
        from aweai.models.inference import LLM

        try:
            llm = LLM(model_id=req.model)
            messages = req.history + [{"role": "user", "content": req.message}]
            reply = llm.chat(messages)
            return {"reply": reply, "model": llm.model_id or "auto"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ---------- training ----------
    @app.post("/api/train")
    def train(req: TrainRequest):
        from aweai.models.trainer import train_scratch, finetune, continue_training

        try:
            if req.mode == "finetune":
                if not req.base_model:
                    raise HTTPException(status_code=400, detail="base_model required for finetune")
                result = finetune(req.base_model, req.name, req.data, epochs=req.epochs)
            elif req.mode == "continue":
                if not req.base_model:
                    raise HTTPException(status_code=400, detail="checkpoint path required for continue")
                result = continue_training(req.name, req.base_model, req.data, epochs=req.epochs)
            else:
                result = train_scratch(req.name, req.data, epochs=req.epochs)
            return {
                "status": "ok",
                "name": result.name,
                "path": result.path,
                "steps": result.steps,
                "loss": round(result.loss, 4),
                "duration_s": round(result.duration_s, 2),
                "messages": result.messages,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ---------- RAG ----------
    @app.get("/api/rag/stats")
    def rag_stats():
        from aweai.rag.engine import RAGEngine

        return RAGEngine().stats()

    @app.post("/api/rag/index")
    def rag_index(req: RagIndexRequest):
        from aweai.rag.engine import RAGEngine

        try:
            engine = RAGEngine()
            if req.path:
                p = Path(req.path)
                if p.is_dir():
                    added = engine.index_directory(str(p))
                else:
                    added = engine.index_file(str(p))
            else:
                added = 0
            return {"status": "ok", "added": added, "stats": engine.stats()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/rag/ask")
    def rag_ask(req: RagAskRequest):
        from aweai.rag.engine import RAGEngine

        try:
            engine = RAGEngine()
            result = engine.ask(req.query, top_k=req.top_k)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ---------- agents ----------
    @app.post("/api/agent/run")
    def agent_run(req: AgentRunRequest):
        from aweai.agents.engine import AgentEngine

        try:
            agent = AgentEngine.create()
            result = agent.run(req.task, max_steps=req.max_steps, verbose=False)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ---------- actions (automation studio) ----------
    @app.post("/api/actions/run")
    def actions_run(req: ActionRequest):
        from aweai.actions.runner import ActionsRunner

        try:
            runner = ActionsRunner(lang=req.lang, verbose=False)
            return runner.run(req.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ---------- static SPA ----------
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


def serve(port: Optional[int] = None, host: str = "127.0.0.1",
          open_browser: Optional[bool] = None, debug: bool = False) -> None:
    """Start the UI server with smart port selection (8888, then +1)."""
    import uvicorn

    cfg = get_config()
    preferred = port or int(cfg.get("port", 8888))
    actual = resolve_port(preferred)
    if actual != preferred:
        print(f"[aweai] Port {preferred} is busy, using {actual} instead.")
    cfg.set("port", actual)

    url = f"http://{host}:{actual}"
    print(f"[aweai] AWEAI UI running at {url}")
    print(f"[aweai] API docs: {url}/api/docs")
    print("[aweai] Press Ctrl+C to stop.")

    if open_browser is None:
        open_browser = bool(cfg.get("auto_open_browser", True))
    if open_browser and host in ("127.0.0.1", "localhost"):
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app = create_app()
    uvicorn.run(app, host=host, port=actual, log_level="debug" if debug else "info")
