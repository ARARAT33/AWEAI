"""Shared types, schemas, and streaming primitives for AWEAI compatibility layer."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    FUNCTION_CALL = "function_call"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class ErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request_error"
    AUTHENTICATION = "authentication_error"
    NOT_FOUND = "not_found_error"
    RATE_LIMIT = "rate_limit_error"
    SERVER = "server_error"
    OVERLOADED = "overloaded_error"
    TIMEOUT = "timeout_error"


class MessageContentType(str, Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"
    INPUT_IMAGE = "input_image"


class FunctionCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ImageURL:
    url: str
    detail: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return {"url": self.url, "detail": self.detail}


@dataclass
class ContentPart:
    type: MessageContentType
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None
    data: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"type": self.type.value}
        if self.text is not None:
            result["text"] = self.text
        if self.image_url is not None:
            result["image_url"] = self.image_url.to_dict()
        if self.data is not None:
            result["data"] = self.data
        return result


@dataclass
class ChatMessage:
    role: str
    content: Optional[Union[str, List[ContentPart]]] = None
    name: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"role": self.role}
        if isinstance(self.content, str):
            result["content"] = self.content
        elif isinstance(self.content, list):
            result["content"] = [c.to_dict() for c in self.content]
        if self.name is not None:
            result["name"] = self.name
        if self.function_call is not None:
            result["function_call"] = self.function_call.to_dict()
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass
class FunctionDefinition:
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"name": self.name}
        if self.description is not None:
            result["description"] = self.description
        if self.parameters is not None:
            result["parameters"] = self.parameters
        return result


@dataclass
class ToolDefinition:
    type: str = "function"
    function: Optional[FunctionDefinition] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "function": self.function.to_dict() if self.function else None}


@dataclass
class ChatRequest:
    model: str
    messages: List[ChatMessage] = field(default_factory=list)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    functions: Optional[List[FunctionDefinition]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"model": self.model, "messages": [m.to_dict() for m in self.messages]}
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.top_p is not None:
            result["top_p"] = self.top_p
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
        if self.max_completion_tokens is not None:
            result["max_completion_tokens"] = self.max_completion_tokens
        if self.stop is not None:
            result["stop"] = self.stop
        if self.stream:
            result["stream"] = self.stream
        if self.functions is not None:
            result["functions"] = [f.to_dict() for f in self.functions]
        if self.function_call is not None:
            result["function_call"] = self.function_call
        if self.tools is not None:
            result["tools"] = [t.to_dict() for t in self.tools]
        if self.tool_choice is not None:
            result["tool_choice"] = self.tool_choice
        if self.response_format is not None:
            result["response_format"] = self.response_format
        if self.presence_penalty is not None:
            result["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            result["frequency_penalty"] = self.frequency_penalty
        if self.logit_bias is not None:
            result["logit_bias"] = self.logit_bias
        if self.user is not None:
            result["user"] = self.user
        if self.seed is not None:
            result["seed"] = self.seed
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class CompletionChoice:
    index: int
    text: str
    finish_reason: str
    logprobs: Optional[Dict[str, Any]] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "index": self.index,
            "text": self.text,
            "finish_reason": self.finish_reason,
        }
        if self.logprobs is not None:
            result["logprobs"] = self.logprobs
        if self.function_call is not None:
            result["function_call"] = self.function_call.to_dict()
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        return result


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ChatResponse:
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = "local-model"
    choices: List[CompletionChoice] = field(default_factory=list)
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [c.to_dict() for c in self.choices],
        }
        if self.usage is not None:
            result["usage"] = self.usage.to_dict()
        if self.system_fingerprint is not None:
            result["system_fingerprint"] = self.system_fingerprint
        if self.service_tier is not None:
            result["service_tier"] = self.service_tier
        return result


@dataclass
class ChatCompletionChunk:
    id: str
    object: str = "chat.completion.chunk"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = "local-model"
    choices: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": self.choices,
        }


@dataclass
class EmbeddingResponse:
    object: str = "list"
    data: List[Dict[str, Any]] = field(default_factory=list)
    model: str = "local-model"
    usage: Optional[Usage] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "object": self.object,
            "data": self.data,
            "model": self.model,
        }
        if self.usage is not None:
            result["usage"] = self.usage.to_dict()
        return result


@dataclass
class ModelInfo:
    id: str
    object: str = "model"
    created: int = field(default_factory=lambda: int(time.time()))
    owned_by: str = "aweai"
    permission: Optional[List[Dict[str, Any]]] = None
    root: Optional[str] = None
    parent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "owned_by": self.owned_by,
        }
        if self.permission is not None:
            result["permission"] = self.permission
        if self.root is not None:
            result["root"] = self.root
        if self.parent is not None:
            result["parent"] = self.parent
        return result


@dataclass
class ModelsResponse:
    object: str = "list"
    data: List[ModelInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"object": self.object, "data": [m.to_dict() for m in self.data]}


class APIError(Exception):
    def __init__(self, message: str, code: ErrorCode, status_code: int = 400, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "error": {
                "message": self.message,
                "type": self.code.value,
                "code": self.code.value,
            }
        }
        if self.details:
            result["error"]["details"] = self.details
        return result


class AuthenticationError(APIError):
    def __init__(self, message: str = "Invalid API key") -> None:
        super().__init__(message, ErrorCode.AUTHENTICATION, 401)


class RateLimitError(APIError):
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None) -> None:
        super().__init__(message, ErrorCode.RATE_LIMIT, 429, {"retry_after": retry_after or 60})


class ModelNotFoundError(APIError):
    def __init__(self, message: str = "Model not found") -> None:
        super().__init__(message, ErrorCode.NOT_FOUND, 404)


class ServerError(APIError):
    def __init__(self, message: str = "Internal server error") -> None:
        super().__init__(message, ErrorCode.SERVER, 500)


class ValidationError(APIError):
    def __init__(self, message: str = "Invalid request parameters") -> None:
        super().__init__(message, ErrorCode.INVALID_REQUEST, 400)


def create_chunk_from_delta(model: str, delta: Dict[str, Any], finish_reason: Optional[str] = None) -> ChatCompletionChunk:
    choices = [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    return ChatCompletionChunk(id=chunk_id, model=model, choices=choices)
