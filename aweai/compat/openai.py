"""OpenAI API compatible server — drop-in replacement using stdlib only."""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

from aweai.compat.local_models import LocalModelServer
from aweai.compat.providers import OpenAIProvider
from aweai.compat.router import RouteConfig, router
from aweai.compat.types import (
    ChatCompletionChunk,
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
    create_chunk_from_delta,
)


class OpenAICompatibleServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 11434, model_dir: Optional[str] = None) -> None:
        self.host = host
        self.port = port
        self.model_dir = model_dir
        self.local_server = LocalModelServer(model_dir=model_dir)
        self.provider = OpenAIProvider()
        router.register_provider("openai", self.provider)
        router.register("openai-default", RouteConfig(provider="openai", model="gpt-4o-mini"))
        self._routes: Dict[str, Callable] = {
            "/v1/chat/completions": self._handle_chat,
            "/v1/completions": self._handle_completions,
            "/v1/embeddings": self._handle_embeddings,
            "/v1/models": self._handle_models,
        }
        self._server: Optional[HTTPServer] = None

    def start(self) -> None:
        handler = _make_handler(self)
        self._server = HTTPServer((self.host, self.port), handler)
        self._server.serve_forever()

    def start_async(self) -> None:
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()

    def _handle_chat(self, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, str], Any]:
        try:
            request = self._parse_chat_request(body)
        except ValidationError as e:
            return 400, {"Content-Type": "application/json"}, e.to_dict()
        auth = headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            key = auth[7:]
            if not self.provider.api_key:
                self.provider.api_key = key
        try:
            response = router.send(request)
        except AuthenticationError as e:
            return 401, {"Content-Type": "application/json"}, e.to_dict()
        except ModelNotFoundError as e:
            return 404, {"Content-Type": "application/json"}, e.to_dict()
        except APIError as e:
            return e.status_code, {"Content-Type": "application/json"}, e.to_dict()
        return 200, {"Content-Type": "application/json"}, response.to_dict()

    def _handle_completions(self, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, str], Any]:
        model = body.get("model", "local-model")
        prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 128)
        temperature = body.get("temperature", 0.7)
        messages = [ChatMessage(role="user", content=prompt)]
        request = ChatRequest(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        try:
            response = router.send(request)
        except APIError as e:
            return e.status_code, {"Content-Type": "application/json"}, e.to_dict()
        result = {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"text": response.choices[0].text if response.choices else "", "index": 0, "finish_reason": response.choices[0].finish_reason if response.choices else "stop"}],
            "usage": response.usage.to_dict() if response.usage else None,
        }
        return 200, {"Content-Type": "application/json"}, result

    def _handle_embeddings(self, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, str], Any]:
        model = body.get("model", "local-model")
        input_texts = body.get("input", [])
        if not isinstance(input_texts, list):
            input_texts = [input_texts]
        try:
            response = self.local_server.embeddings(model, input_texts)
        except ModelNotFoundError as e:
            return 404, {"Content-Type": "application/json"}, e.to_dict()
        return 200, {"Content-Type": "application/json"}, response.to_dict()

    def _handle_models(self, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, str], Any]:
        models = router.providers.get("openai")
        if models:
            try:
                response = models.models()
                return 200, {"Content-Type": "application/json"}, response.to_dict()
            except Exception:
                pass
        response = self.local_server.list_models()
        return 200, {"Content-Type": "application/json"}, response.to_dict()

    def _parse_chat_request(self, body: Dict[str, Any]) -> ChatRequest:
        model = body.get("model")
        if not model:
            raise ValidationError("Model is required")
        messages = []
        for m in body.get("messages", []):
            messages.append(ChatMessage(
                role=m.get("role", "user"),
                content=m.get("content"),
                name=m.get("name"),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            ))
        return ChatRequest(
            model=model,
            messages=messages,
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            max_tokens=body.get("max_tokens"),
            max_completion_tokens=body.get("max_completion_tokens"),
            stop=body.get("stop"),
            stream=body.get("stream", False),
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
            response_format=body.get("response_format"),
            user=body.get("user"),
            seed=body.get("seed"),
        )

    def _stream_response(self, request: ChatRequest, start_event: Any) -> Any:
        for chunk in self.local_server.generate_stream(request):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"


class _Handler(BaseHTTPRequestHandler):
    server: OpenAICompatibleServer

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
        if isinstance(payload, (dict, list)):
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        elif hasattr(payload, "__iter__"):
            for item in payload:
                self.wfile.write(item.encode("utf-8") if isinstance(item, str) else item)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/v1/models":
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


def _make_handler(server: OpenAICompatibleServer) -> type:
    class Handler(_Handler):
        pass
    Handler.server = server
    return Handler
