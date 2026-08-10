"""Anthropic Claude compatible layer — drop-in replacement for /v1/messages."""

from __future__ import annotations

import base64
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

from aweai.compat.local_models import LocalModelServer
from aweai.compat.providers import AnthropicProvider
from aweai.compat.router import RouteConfig, router
from aweai.compat.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    CompletionChoice,
    ContentPart,
    ImageURL,
    MessageContentType,
    ModelInfo,
    ModelsResponse,
    Usage,
    APIError,
    AuthenticationError,
    ValidationError,
    ModelNotFoundError,
)


class AnthropicCompatibleServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 11436, model_dir: Optional[str] = None) -> None:
        self.host = host
        self.port = port
        self.model_dir = model_dir
        self.local_server = LocalModelServer(model_dir=model_dir)
        self.provider = AnthropicProvider()
        router.register_provider("anthropic", self.provider)
        router.register("anthropic-default", RouteConfig(provider="anthropic", model="claude-3-5-haiku-latest"))
        self._routes: Dict[str, Callable] = {
            "/v1/messages": self._handle_messages,
            "/v1/complete": self._handle_complete,
            "/models": self._handle_models,
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

    def _handle_messages(self, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, str], Any]:
        try:
            request = self._parse_messages_request(body)
        except ValidationError as e:
            return 400, {"Content-Type": "application/json"}, e.to_dict()
        auth = headers.get("x-api-key") or headers.get("Authorization", "")
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
        result = self._to_anthropic_format(response, body.get("system"))
        return 200, {"Content-Type": "application/json"}, result

    def _handle_complete(self, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, str], Any]:
        prompt = body.get("prompt", "")
        model = body.get("model", "claude-instant-1")
        max_tokens = body.get("max_tokens_to_sample", 256)
        messages = [ChatMessage(role="user", content=prompt)]
        request = ChatRequest(model=model, messages=messages, max_tokens=max_tokens, temperature=body.get("temperature"))
        try:
            response = router.send(request)
        except APIError as e:
            return e.status_code, {"Content-Type": "application/json"}, e.to_dict()
        result = {
            "completion": response.choices[0].text if response.choices else "",
            "stop_reason": response.choices[0].finish_reason if response.choices else "stop",
            "model": model,
            "usage": {"input_tokens": response.usage.prompt_tokens if response.usage else 0, "output_tokens": response.usage.completion_tokens if response.usage else 0},
        }
        return 200, {"Content-Type": "application/json"}, result

    def _handle_models(self, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Dict[str, str], Any]:
        response = self.provider.models()
        return 200, {"Content-Type": "application/json"}, response.to_dict()

    def _parse_messages_request(self, body: Dict[str, Any]) -> ChatRequest:
        model = body.get("model")
        if not model:
            raise ValidationError("Model is required")
        system = body.get("system")
        messages = []
        if system:
            messages.append(ChatMessage(role="system", content=system if isinstance(system, str) else system.get("type", "text")))
        for m in body.get("messages", []):
            role = m.get("role", "user")
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append(ContentPart(type=MessageContentType.TEXT, text=part.get("text", "")))
                    elif part.get("type") == "image":
                        source = part.get("source", {})
                        data = source.get("data", "")
                        parts.append(ContentPart(type=MessageContentType.INPUT_IMAGE, data=data))
                messages.append(ChatMessage(role=role, content=parts))
            else:
                messages.append(ChatMessage(role=role, content=content))
        return ChatRequest(
            model=model,
            messages=messages,
            max_tokens=body.get("max_tokens", 1024),
            temperature=body.get("temperature", 0.7),
            stop=body.get("stop_sequences"),
            metadata={"anthropic_version": body.get("anthropic_version", "2023-06-01")},
        )

    def _to_anthropic_format(self, response: ChatResponse, system: Optional[Any] = None) -> Dict[str, Any]:
        text = response.choices[0].text if response.choices else ""
        result: Dict[str, Any] = {
            "id": response.id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": response.model,
            "stop_reason": response.choices[0].finish_reason if response.choices else "stop",
            "usage": {
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }
        return result


class _AnthropicHandler(BaseHTTPRequestHandler):
    server: AnthropicCompatibleServer

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


def _make_handler(server: AnthropicCompatibleServer) -> type:
    class Handler(_AnthropicHandler):
        pass
    Handler.server = server
    return Handler
