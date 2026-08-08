"""Smart port resolution: 8888 → 8889 → … until a free port is found."""

from __future__ import annotations

import socket
from typing import Optional


def is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def resolve_port(preferred: int = 8888, host: str = "127.0.0.1", max_tries: int = 50) -> int:
    """Return the first free port starting at `preferred`, auto-incrementing."""
    for port in range(preferred, preferred + max_tries):
        if is_free(port, host):
            return port
    raise RuntimeError(f"No free port found in range {preferred}..{preferred + max_tries}")


def current_port() -> Optional[int]:
    """Best-effort: return the configured port if it is free, else resolved."""
    from aweai.config import get_config

    cfg = get_config()
    preferred = int(cfg.get("port", 8888))
    return resolve_port(preferred)
