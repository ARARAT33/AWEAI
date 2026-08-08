"""Natural-language action parsing: map free text to factory operations."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# Patterns: (regex, action name, default kwargs)
ACTION_PATTERNS = [
    (re.compile(r"train\s+(?:an?\s+)?(mlp|linear|logistic|kmeans|ngram|autoencoder|gan|rnn|lstm|cnn|transformer)\s+(?:model)?\s*(?:named\s+)?([\w-]+)?", re.I), "train", {}),
    (re.compile(r"evaluate\s+(?:the\s+)?model\s+([\w-]+)", re.I), "eval", {}),
    (re.compile(r"export\s+(?:the\s+)?model\s+([\w-]+)\s+(?:to\s+)?(json|raw|onnx|torchscript)", re.I), "export", {}),
    (re.compile(r"delete\s+(?:the\s+)?model\s+([\w-]+)", re.I), "delete", {}),
    (re.compile(r"list\s+(?:all\s+)?models", re.I), "list", {}),
    (re.compile(r"recommend\s+(?:a\s+)?model\s+(?:for\s+)?([\w_]+)?", re.I), "recommend", {}),
    (re.compile(r"(?:load|read)\s+(?:data|dataset)\s+from\s+(.+)", re.I), "load_data", {}),
    (re.compile(r"index\s+(?:documents?\s+)?(?:in\s+)?(.+)", re.I), "rag_index", {}),
    (re.compile(r"ask\s+(.+)", re.I), "rag_ask", {}),
]


def parse_action(text: str) -> Dict[str, Any]:
    """Parse a natural-language instruction into an action dict.

    Returns {"action": str, "kwargs": {...}} or raises ValueError.
    """
    text = text.strip()
    for pattern, action, defaults in ACTION_PATTERNS:
        m = pattern.search(text)
        if m:
            groups = [g for g in m.groups() if g]
            kwargs = dict(defaults)
            if action == "train":
                if groups:
                    kwargs["model_type"] = groups[0].lower()
                if len(groups) > 1:
                    kwargs["name"] = groups[1]
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
            return {"action": action, "kwargs": kwargs}
    raise ValueError(f"Could not parse action: {text}")
