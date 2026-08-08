"""Smart port selection: start at 8888, increment on conflict."""

from __future__ import annotations

import socket
from typing import Optional


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if the given port is not bound (TCP)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


def find_free_port(start: int = 8888, max_tries: int = 50, host: str = "0.0.0.0") -> int:
    """Return the first free port >= start; increment on conflicts.

    Binds on 0.0.0.0 so it also detects ports used on other interfaces.
    """
    for port in range(start, start + max_tries):
        if is_port_free(port, host):
            return port
    # fall back to an OS-assigned port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def resolve_port(preferred: Optional[int] = None) -> int:
    """Resolve the actual port to serve on.

    - preferred: the requested port (default 8888)
    - returns preferred if free, otherwise preferred+1, +2, ...
    """
    start = preferred or 8888
    return find_free_port(start)
