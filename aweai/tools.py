"""Tool registry for AWEAI.

Tools are plain async functions decorated with ``@tool``.  The registry
provides:

* declarative tool schemas (name, description, JSON-schema parameters),
* a callable dispatcher with argument validation,
* a convenience decorator.

Tools let an agent perform actions (web search, math, file ops, ...).
A couple of built-in tools are included as examples.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

# ----------------------------------------------------------------------
# Tool abstraction
# ----------------------------------------------------------------------


@dataclass
class Tool:
    """A registered tool."""

    name: str
    description: str
    fn: Callable[..., Awaitable[Any]]
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)

    def schema(self) -> Dict[str, Any]:
        """Return the tool in OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


def tool(
    name: Optional[str] = None,
    *,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    required: Optional[List[str]] = None,
) -> Callable:
    """Decorator that turns an async function into a :class:`Tool`.

    Supports both bare (``@tool``) and parameterized (``@tool(name=...)``)
    usage.  The tool name, description and parameter JSON-schema can be
    given explicitly, or inferred from the function docstring / signature.

    Example::

        @tool
        async def add(a: float, b: float) -> float:
            "Add two numbers."
            return a + b
    """

    def decorator(fn: Callable[..., Awaitable[Any]]) -> Tool:
        tool_name = name or fn.__name__
        tool_desc = description or (inspect.getdoc(fn) or "").strip() or fn.__name__
        sig = inspect.signature(fn)
        params: Dict[str, Any] = dict(parameters or {})
        req: List[str] = list(required or [])
        for pname, param in sig.parameters.items():
            if pname in ("self", "ctx", "context"):
                continue
            if pname not in params:
                params[pname] = _param_schema(param)
            if param.default is inspect.Parameter.empty and pname not in req:
                req.append(pname)
        return Tool(
            name=tool_name,
            description=tool_desc,
            fn=fn,
            parameters=params,
            required=req,
        )

    if callable(name):
        # Used as a bare decorator: @tool
        fn = name
        name = None
        return decorator(fn)
    return decorator


def _param_schema(param: inspect.Parameter) -> Dict[str, Any]:
    """Best-effort JSON schema inference from a Python parameter."""
    annotation = param.annotation
    mapping = {
        "str": {"type": "string"},
        "int": {"type": "integer"},
        "float": {"type": "number"},
        "bool": {"type": "boolean"},
        "list": {"type": "array"},
        "dict": {"type": "object"},
    }
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}
    if isinstance(annotation, type):
        return mapping.get(annotation.__name__, {"type": "string"})
    return {"type": "string"}


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


class ToolRegistry:
    """Holds and dispatches tools."""

    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool_obj: Tool) -> None:
        if tool_obj.name in self._tools:
            raise ValueError(f"Tool already registered: {tool_obj.name}")
        self._tools[tool_obj.name] = tool_obj

    def add(self, fn: Callable[..., Awaitable[Any]], **kwargs: Any) -> Tool:
        """Register an async function directly (decorator sugar)."""
        t = tool(**kwargs)(fn)
        self.register(t)
        return t

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return sorted(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    async def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Validate arguments against the JSON schema, then invoke."""
        t = self._tools.get(name)
        if t is None:
            raise KeyError(f"Unknown tool: {name}")

        missing = [r for r in t.required if r not in arguments]
        if missing:
            raise ValueError(
                f"Tool '{name}' missing required argument(s): {missing}"
            )
        return await t.fn(**arguments)


# ----------------------------------------------------------------------
# Built-in example tools
# ----------------------------------------------------------------------


@tool
async def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


@tool
async def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b


@tool
async def now_utc() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


@tool
async def echo_json(data: Dict[str, Any]) -> str:
    """Echo input data as pretty-printed JSON (debug tool)."""
    return json.dumps(data, indent=2, ensure_ascii=False)


def default_registry() -> ToolRegistry:
    """A registry pre-populated with the built-in example tools."""
    return ToolRegistry([add, multiply, now_utc, echo_json])
