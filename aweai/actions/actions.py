"""Natural-language action parsing: map free text to factory operations."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# Patterns: (regex, action name, default kwargs)
ACTION_PATTERNS = [
    # "train a new model [named X] with [this] data PATH"
    (re.compile(r"train\s+(?:a\s+)?new\s+model\s+(?:named\s+([\w-]+)\s+)?(?:with\s+(?:this\s+)?data\s+(.+))?", re.I), "train", {}),
    (re.compile(r"train\s+(?:an?\s+)?(mlp|linear|logistic|kmeans|ngram|autoencoder|gan|rnn|lstm|cnn|transformer)\s+(?:model)?\s*(?:named\s+)?([\w-]+)?", re.I), "train", {}),
    (re.compile(r"evaluate\s+(?:the\s+)?model\s+([\w-]+)", re.I), "eval", {}),
    (re.compile(r"export\s+(?:the\s+)?model\s+([\w-]+)\s+(?:to\s+)?(json|raw|onnx|torchscript)", re.I), "export", {}),
    (re.compile(r"delete\s+(?:the\s+)?model\s+([\w-]+)", re.I), "delete", {}),
    (re.compile(r"list\s+(?:all\s+)?models", re.I), "list", {}),
    (re.compile(r"recommend\s+(?:a\s+)?model\s+(?:for\s+)?([\w_]+)?", re.I), "recommend", {}),
    (re.compile(r"(?:load|read)\s+(?:data|dataset)\s+from\s+(.+)", re.I), "load_data", {}),
    (re.compile(r"^(?:hardware|hw)$", re.I), "hardware", {}),
    (re.compile(r"(?:ինդեքսավորել|index)\s+(?:փաստաթղթերը?|փաստաթուղթը?|documents?)?\s*(?:in\s+)?(.+)", re.I), "rag_index", {}),
    (re.compile(r"index\s+(?:documents?)?\s*(?:in\s+)?(.+)", re.I), "rag_index", {}),
    (re.compile(r"ask\s+(.+)", re.I), "rag_ask", {}),
]

# Map parsed actions to coarse intents (used by the compatibility API).
_INTENT_MAP = {
    "train": "train",
    "eval": "evaluate",
    "export": "export",
    "delete": "delete",
    "list": "list",
    "recommend": "recommend",
    "load_data": "load_data",
    "rag_index": "rag",
    "rag_ask": "rag",
}


def parse_action(text: str) -> Dict[str, Any]:
    """Parse a natural-language instruction into an action dict.

    Returns a dict that supports BOTH API shapes:
      {"action": str, "kwargs": {...}}          (canonical)
      {"intent": str, "params": {...}}          (compatibility alias)
    or raises ValueError.
    """
    text = text.strip()
    for pattern, action, defaults in ACTION_PATTERNS:
        m = pattern.search(text)
        if m:
            groups = [g for g in m.groups() if g]
            kwargs = dict(defaults)
            if action == "train":
                if groups:
                    # new-model form: [name, data_path] OR [model_type, name]
                    if m.group(1) and m.group(2) and not re.match(r"^(mlp|linear|logistic|kmeans|ngram|autoencoder|gan|rnn|lstm|cnn|transformer)$", m.group(1), re.I):
                        kwargs["name"] = m.group(1)
                        kwargs["data_path"] = m.group(2).strip()
                    elif len(groups) >= 1 and re.match(r"^(mlp|linear|logistic|kmeans|ngram|autoencoder|gan|rnn|lstm|cnn|transformer)$", groups[0], re.I):
                        kwargs["model_type"] = groups[0].lower()
                        if len(groups) > 1:
                            kwargs["name"] = groups[1]
                    elif len(groups) >= 1:
                        # "train a new model with this data PATH"
                        kwargs["data_path"] = groups[0].strip()
            elif action in ("eval", "export", "delete"):
                if groups:
                    kwargs["name"] = groups[0]
                if action == "export" and len(groups) > 1:
                    kwargs["fmt"] = groups[1]
            elif action == "recommend":
                if groups:
                    kwargs["task"] = groups[0]
            elif action in ("load_data", "rag_index"):
                if groups:
                    kwargs["path"] = groups[0]
            elif action == "rag_ask":
                if groups:
                    kwargs["query"] = groups[0]
            params = dict(kwargs)
            if "data_path" in params:
                params.setdefault("path", params["data_path"])
            return {
                "action": action,
                "kwargs": kwargs,
                "intent": _INTENT_MAP.get(action, action),
                "params": params,
            }
    raise ValueError(f"Could not parse action: {text}")
