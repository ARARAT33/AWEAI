"""Google AI compatible layer — Gemini and PaLM style endpoints."""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

from aweai.compat.local_models import LocalModelServer
from aweai.compat.providers import GoogleProvider
from aweai.compat.router import RouteConfig, router
from aweai.compat.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CompletionChoice,
    EmbeddingResponse,
    ModelInfo,
    ModelsResponse,
    Usage,
    APIError,
    AuthenticationError,
    ValidationError,
    ModelNotFoundError,
)


class GoogleCompatibleServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 11435, model_dir: Optional[str] = None) -> None:
        self.host = host
        self.port = port
        self.model_dir = model_dir
        self.local_server = LocalModelServer(model_dir=model_dir)
        self.provider = GoogleProvider()
        router.register_provider("google", self.provider)
        router.register("google-default", RouteConfig(provider="google", model="gemini-2.0-flash"))
        self._routes: Dict[str, Callable] = {
            "/chat/completions": self._handle_chat,
            "/models": self._handle_models,
            "/embeddings": self._handle_embeddings,
        }

    def start(self) -> None:
        handler = _make_handler(self)
        server = HTTPServer((self.host, self.port), handler)
        server.serve_forever()

    def start_async(self) -> None:
        import threading
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        return thread

    def _handle_chat(self, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, str], Any]:
        try:
            request = self._parse_chat_request(body)
        except ValidationError as e:
            return 400, {"Content-Type": "application/json"}, e.to_dict()
        try:
            response = router.send(request)
        except APIError as e:
            return e.status_code, {"Content-Type": "application/json"}, e.to_dict()
        return 200, {"Content-Type": "application/json"}, response.to_dict()

    def _handle_embeddings(self, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, str], Any]:
        model = body.get("model", "local-model")
        texts = body.get("content", body.get("input", []))
        if not isinstance(texts, list):
            texts = [texts]
        try:
            response = self.local_server.embeddings(model, texts)
        except ModelNotFoundError as e:
            return 404, {"Content-Type": "application/json"}, e.to_dict()
        return 200, {"Content-Type": "application/json"}, response.to_dict()

    def _handle_models(self, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, str], Any]:
        response = self.provider.models()
        return 200, {"Content-Type": "application/json"}, response.to_dict()

    def _parse_chat_request(self, body: Dict[str, Any]) -> ChatRequest:
        model = body.get("model", "gemini-2.0-flash")
        messages = []
        for m in body.get("messages", []):
            messages.append(ChatMessage(role=m.get("role", "user"), content=m.get("content")))
        return ChatRequest(model=model, messages=messages, temperature=body.get("temperature"), max_tokens=body.get("max_tokens"))

    def gemini_chat(self, model: str, prompt: str, temperature: float = 0.7) -> ChatResponse:
        messages = [ChatMessage(role="user", content=prompt)]
        request = ChatRequest(model=model, messages=messages, temperature=temperature)
        return router.send(request)

    def palm_complete(self, model: str, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> ChatResponse:
        return self.gemini_chat(model, prompt, temperature)


class _GoogleHandler(BaseHTTPRequestHandler):
    server: GoogleCompatibleServer

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body_raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(body_raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        path = self.path.split("?")[0]
        handler = self.server._routes.get(path)
        if not handler:
            self.send_error(404, "Not Found")
            return
        status, headers_out, payload = handler(body, dict(self.headers))
        self.send_response(status)
        for k, v in headers_out.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/models":
            status, headers_out, payload = self.server._handle_models({}, dict(self.headers))
            self.send_response(status)
            for k, v in headers_out.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def log_message(self, format: str, *args: Any) -> None:
        pass


def _make_handler(server: GoogleCompatibleServer) -> type:
    class Handler(_GoogleHandler):
        pass
    Handler.server = server
    return Handler
