"""AWEAI Anywhere — one-command universal deployment.

Innovative "work from anywhere" layer:
  * environment auto-detection (local / LAN / cloud / Colab / container / phone)
  * dependency-free QR code generator (pure stdlib, byte mode, ECC-L, v1-v7)
    rendered as Unicode blocks in the terminal — scan with any phone camera
  * zero-dependency public tunnel attempts: cloudflared -> ngrok -> localtunnel
    -> serveo -> ssh -R, each wrapped in try/except so it never crashes
  * ``aweai anywhere`` launches the UI on 0.0.0.0 and prints every reachable URL
    (localhost, LAN IP, public tunnel) plus a scannable QR code.

Nothing here requires third-party packages; optional tunnels degrade gracefully.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

__all__ = [
    "detect_environment",
    "lan_ip",
    "is_online",
    "make_qr",
    "qr_to_text",
    "print_qr",
    "open_tunnel",
    "public_url",
    "anywhere_report",
    "print_report",
]

#------------------------------------------------------------------------ environment

def lan_ip() -> str:
    ""Best-effort LAN IPv4 of this machine (no deps)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"
