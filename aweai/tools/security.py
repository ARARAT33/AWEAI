# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""AWEAI security tools — hashing, checksums, crypto, auditing, scanning, watermarking.

Each tool has a unique purpose. All use only the Python standard library so
they work in every environment (localhost, cloud, containers, offline).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import string
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from aweai.tools.registry import tool
from aweai.watermark import (
    embed_watermark,
    extract_watermark,
    get_watermark_status,
    inspect_watermark,
    verify_watermark,
)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

@tool("hash_sha256", "security", "Compute SHA-256 hash of a string")
def hash_sha256(text: str) -> Dict[str, Any]:
    return {"algorithm": "sha256", "hash": hashlib.sha256(text.encode("utf-8")).hexdigest()}


@tool("hash_sha1", "security", "Compute SHA-1 hash of a string (deprecated, informational)")
def hash_sha1(text: str) -> Dict[str, Any]:
    return {"algorithm": "sha1", "hash": hashlib.sha1(text.encode("utf-8")).hexdigest()}


@tool("hash_md5", "security", "Compute MD5 hash of a string (not for security use)")
def hash_md5(text: str) -> Dict[str, Any]:
    return {"algorithm": "md5", "hash": hashlib.md5(text.encode("utf-8")).hexdigest()}


@tool("hash_sha512", "security", "Compute SHA-512 hash of a string")
def hash_sha512(text: str) -> Dict[str, Any]:
    return {"algorithm": "sha512", "hash": hashlib.sha512(text.encode("utf-8")).hexdigest()}


@tool("hash_blake2b", "security", "Compute BLAKE2b hash of a string")
def hash_blake2b(text: str) -> Dict[str, Any]:
    return {"algorithm": "blake2b", "hash": hashlib.blake2b(text.encode("utf-8")).hexdigest()}


@tool("hash_file_sha256", "security", "Compute SHA-256 of a file in streaming mode")
def hash_file_sha256(path: str) -> Dict[str, Any]:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return {"path": path, "sha256": h.hexdigest()}


@tool("hash_file_md5", "security", "Compute MD5 of a file in streaming mode")
def hash_file_md5(path: str) -> Dict[str, Any]:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return {"path": path, "md5": h.hexdigest()}


@tool("checksum_crc32", "security", "Compute CRC32 checksum of a string or file")
def checksum_crc32(text: str = "", path: str = "") -> Dict[str, Any]:
    if path:
        data = Path(path).read_bytes()
        return {"path": path, "crc32": f"{zlib.crc32(data):08x}"}
    return {"crc32": f"{zlib.crc32(text.encode('utf-8')):08x}"}


@tool("hash_compare", "security", "Compare a computed hash against an expected hash (constant-time)")
def hash_compare(actual: str, expected: str) -> Dict[str, Any]:
    match = hmac.compare_digest(actual.lower(), expected.lower())
    return {"match": match}


@tool("hash_all", "security", "Compute all common hashes of a string (md5, sha1, sha256, sha512, blake2b)")
def hash_all(text: str) -> Dict[str, Any]:
    data = text.encode("utf-8")
    return {
        "md5": hashlib.md5(data).hexdigest(),
        "sha1": hashlib.sha1(data).hexdigest(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "sha512": hashlib.sha512(data).hexdigest(),
        "blake2b": hashlib.blake2b(data).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Encoding / decoding
# ---------------------------------------------------------------------------

@tool("encode_base64", "security", "Encode a string to Base64")
def encode_base64(text: str) -> Dict[str, Any]:
    return {"encoded": base64.b64encode(text.encode("utf-8")).decode("ascii")}


@tool("decode_base64", "security", "Decode a Base64 string to text")
def decode_base64(encoded: str) -> Dict[str, Any]:
    return {"decoded": base64.b64decode(encoded).decode("utf-8", errors="replace")}


@tool("encode_url", "security", "URL-encode a string")
def encode_url(text: str) -> Dict[str, Any]:
    import urllib.parse

    return {"encoded": urllib.parse.quote(text, safe="")}


@tool("decode_url", "security", "URL-decode a string")
def decode_url(encoded: str) -> Dict[str, Any]:
    import urllib.parse

    return {"decoded": urllib.parse.unquote(encoded)}


@tool("encode_hex", "security", "Encode a string to hexadecimal")
def encode_hex(text: str) -> Dict[str, Any]:
    return {"encoded": text.encode("utf-8").hex()}


@tool("decode_hex", "security", "Decode a hexadecimal string to text")
def decode_hex(encoded: str) -> Dict[str, Any]:
    return {"decoded": bytes.fromhex(encoded).decode("utf-8", errors="replace")}


@tool("rot13", "security", "Apply ROT13 cipher to a string")
def rot13(text: str) -> Dict[str, Any]:
    return {"rot13": text.translate(str.maketrans(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm",
    ))}


@tool("caesar_cipher", "security", "Apply a Caesar cipher shift to a string")
def caesar_cipher(text: str, shift: int = 3) -> Dict[str, Any]:
    out = []
    for ch in text:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            out.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            out.append(ch)
    return {"shift": shift, "result": "".join(out)}


# ---------------------------------------------------------------------------
# Password / secrets
# ---------------------------------------------------------------------------

@tool("password_generate", "security", "Generate a strong random password")
def password_generate(length: int = 16) -> Dict[str, Any]:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()-_=+"
    return {"password": "".join(secrets.choice(alphabet) for _ in range(length))}


@tool("password_strength", "security", "Estimate password strength (score 0-100 + verdict)")
def password_strength(password: str) -> Dict[str, Any]:
    score = 0
    checks = {
        "length>=8": len(password) >= 8,
        "lowercase": bool(re.search(r"[a-z]", password)),
        "uppercase": bool(re.search(r"[A-Z]", password)),
        "digits": bool(re.search(r"\d", password)),
        "symbols": bool(re.search(r"[^A-Za-z0-9]", password)),
    }
    for v in checks.values():
        if v:
            score += 20
    score = min(100, score + max(0, (len(password) - 12) * 2))
    verdict = "weak" if score < 40 else "medium" if score < 70 else "strong"
    return {"score": score, "verdict": verdict, "checks": checks}


@tool("token_generate", "security", "Generate a URL-safe random token")
def token_generate(bytes_n: int = 32) -> Dict[str, Any]:
    return {"token": secrets.token_urlsafe(bytes_n)}


@tool("secret_hex", "security", "Generate a random secret as hex")
def secret_hex(bytes_n: int = 32) -> Dict[str, Any]:
    return {"secret": secrets.token_hex(bytes_n)}


@tool("api_key_generate", "security", "Generate a fake API key in the sk-... style")
def api_key_generate(prefix: str = "aweai") -> Dict[str, Any]:
    return {"api_key": f"{prefix}_{secrets.token_hex(24)}"}


# ---------------------------------------------------------------------------
# HMAC / crypto (std-lib only)
# ---------------------------------------------------------------------------

@tool("hmac_sign", "security", "Compute an HMAC-SHA256 signature for a message")
def hmac_sign(message: str, key: str) -> Dict[str, Any]:
    sig = hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()
    return {"signature": sig, "algorithm": "HMAC-SHA256"}


@tool("hmac_verify", "security", "Verify an HMAC-SHA256 signature (constant-time)")
def hmac_verify(message: str, key: str, signature: str) -> Dict[str, Any]:
    expected = hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()
    return {"valid": hmac.compare_digest(expected, signature)}


@tool("xorshift", "security", "XOR a message with a repeated key (simple obfuscation)")
def xorshift(message: str, key: str) -> Dict[str, Any]:
    kb = key.encode("utf-8")
    out = bytes(b ^ kb[i % len(kb)] for i, b in enumerate(message.encode("utf-8")))
    return {"obfuscated": base64.b64encode(out).decode("ascii")}


@tool("xorshift_decode", "security", "Decode a base64 message obfuscated with XOR key")
def xorshift_decode(encoded: str, key: str) -> Dict[str, Any]:
    kb = key.encode("utf-8")
    data = base64.b64decode(encoded)
    out = bytes(b ^ kb[i % len(kb)] for i, b in enumerate(data))
    return {"decoded": out.decode("utf-8", errors="replace")}


# ---------------------------------------------------------------------------
# Auditing / scanning
# ---------------------------------------------------------------------------

@tool("scan_deps", "security", "Scan requirements files for known risky pin patterns (== / no pin)")
def scan_deps(path: str = "requirements.txt") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"error": f"{path} not found", "findings": []}
    findings = []
    for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[\]]", line)[0].strip()
        if "==" not in line:
            findings.append({"line": i, "pkg": name, "issue": "not pinned with =="})
    return {"path": path, "findings": findings, "count": len(findings)}


@tool("scan_secrets", "security", "Scan a directory for potential secrets/API keys in text files")
def scan_secrets(root: str = ".", max_files: int = 200) -> Dict[str, Any]:
    patterns = [
        (r"(?i)api[_-]?key\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{16,}", "api_key"),
        (r"(?i)secret\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{16,}", "secret"),
        (r"(?i)password\s*[:=]\s*[\"']?[^\s\"']{8,}", "password"),
        (r"(?i)token\s*[:=]\s*[\"']?[A-Za-z0-9_\-\.]{16,}", "token"),
        (r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "private_key"),
    ]
    findings = []
    count = 0
    for p in Path(root).rglob("*"):
        if count >= max_files:
            break
        if not p.is_file():
            continue
        if p.suffix.lower() in {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".gz"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        count += 1
        for pat, kind in patterns:
            for m in re.finditer(pat, text):
                snippet = text[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
                findings.append({"file": str(p), "kind": kind, "snippet": snippet})
                break
    return {"root": root, "findings": findings, "count": len(findings)}


@tool("audit_file_perms", "security", "List files with world-writable permissions under a directory")
def audit_file_perms(root: str = ".") -> Dict[str, Any]:
    findings = []
    for p in Path(root).rglob("*"):
        try:
            mode = p.stat().st_mode & 0o777
            if mode & 0o002:
                findings.append({"path": str(p), "mode": oct(mode)})
        except Exception:
            continue
    return {"root": root, "world_writable": findings, "count": len(findings)}


@tool("password_check_common", "security", "Check whether a password is in a small common-password list")
def password_check_common(password: str) -> Dict[str, Any]:
    common = {
        "password", "123456", "12345678", "qwerty", "abc123", "monkey",
        "letmein", "dragon", "111111", "admin", "welcome", "password1",
        "123456789", "1234567", "123123", "iloveyou", "sunshine", "princess",
        "football", "shadow", "superman", "michael", "batman", "trustno1",
    }
    return {"common": password.lower() in common}


@tool("entropy_estimate", "security", "Estimate entropy (bits) of a string")
def entropy_estimate(text: str) -> Dict[str, Any]:
    from collections import Counter

    if not text:
        return {"bits": 0}
    counts = Counter(text)
    length = len(text)
    entropy = -sum((c / length) * (c / length and __import__("math").log2(c / length)) for c in counts.values())
    return {"bits": round(entropy * length, 2), "per_char": round(entropy, 4)}


@tool("url_validate", "security", "Validate a URL string and return parsed components")
def url_validate(url: str) -> Dict[str, Any]:
    import urllib.parse

    parts = urllib.parse.urlparse(url)
    if parts.scheme not in {"http", "https"}:
        return {"valid": False, "reason": "scheme must be http/https", "url": url}
    return {"valid": True, "scheme": parts.scheme, "host": parts.netloc, "path": parts.path}


@tool("email_validate", "security", "Validate an email address format")
def email_validate(email: str) -> Dict[str, Any]:
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return {"valid": bool(re.match(pattern, email)), "email": email}


@tool("is_private_ip", "security", "Check whether an IP address is private/rfc1918")
def is_private_ip(ip: str) -> Dict[str, Any]:
    import ipaddress

    try:
        addr = ipaddress.ip_address(ip)
        return {"ip": ip, "private": addr.is_private, "loopback": addr.is_loopback}
    except ValueError:
        return {"ip": ip, "error": "invalid IP"}


@tool("is_valid_ip", "security", "Check whether a string is a valid IPv4/IPv6 address")
def is_valid_ip(ip: str) -> Dict[str, Any]:
    import ipaddress

    try:
        addr = ipaddress.ip_address(ip)
        return {"valid": True, "version": addr.version}
    except ValueError:
        return {"valid": False}


@tool("is_valid_port", "security", "Check whether a number is a valid TCP/UDP port")
def is_valid_port(port: int) -> Dict[str, Any]:
    return {"valid": 0 <= port <= 65535, "port": port}


@tool("check_tls", "security", "Check TLS certificate of a host:port (best effort)")
def check_tls(host: str, port: int = 443, timeout: int = 10) -> Dict[str, Any]:
    import ssl
    import socket as _socket

    ctx = ssl.create_default_context()
    try:
        with _socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                return {
                    "host": host,
                    "port": port,
                    "tls_version": ssock.version(),
                    "cipher": ssock.cipher(),
                    "subject": dict(x[0] for x in cert.get("subject", [])),
                    "not_after": cert.get("notAfter"),
                }
    except Exception as e:
        return {"host": host, "port": port, "error": str(e)}


@tool("text_anonymize", "security", "Mask emails and phone numbers in text (anonymization)")
def text_anonymize(text: str) -> Dict[str, Any]:
    masked = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    masked = re.sub(r"(\+?\d[\d\s\-\(\)]{7,}\d)", "[PHONE]", masked)
    return {"masked": masked, "masked_count": text != masked}


@tool("safe_filename", "security", "Sanitize a string into a safe filename")
def safe_filename(name: str) -> Dict[str, Any]:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    safe = safe.strip("._")
    return {"safe": safe or "untitled"}


@tool("redact", "security", "Redact a substring from a text (replace with ***)")
def redact(text: str, secret: str) -> Dict[str, Any]:
    if not secret:
        return {"text": text, "redacted": 0}
    count = text.count(secret)
    return {"text": text.replace(secret, "***"), "redacted": count}


@tool("hash_chain", "security", "Compute a hash chain (repeat hashing n times)")
def hash_chain(text: str, rounds: int = 10) -> Dict[str, Any]:
    h = text.encode("utf-8")
    for _ in range(rounds):
        h = hashlib.sha256(h).digest()
    return {"rounds": rounds, "final": h.hexdigest()}


@tool("salt_generate", "security", "Generate a random salt (hex)")
def salt_generate(bytes_n: int = 16) -> Dict[str, Any]:
    return {"salt": secrets.token_hex(bytes_n)}


@tool("pbkdf2_hash", "security", "Derive a PBKDF2-HMAC-SHA256 hash from a password")
def pbkdf2_hash(password: str, salt: str = "", iterations: int = 100_000) -> Dict[str, Any]:
    salt_b = salt.encode("utf-8") if salt else secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_b, iterations)
    return {
        "salt_hex": salt_b.hex(),
        "iterations": iterations,
        "hash_hex": dk.hex(),
    }


@tool("secure_compare", "security", "Constant-time comparison of two strings")
def secure_compare(a: str, b: str) -> Dict[str, Any]:
    return {"equal": hmac.compare_digest(a, b)}


# ---------------------------------------------------------------------------
# Watermarking tools
# ---------------------------------------------------------------------------

@tool("watermark_embed", "security", "Embed multi-layer visible/invisible steganographic watermark into text, dict, or file")
def watermark_embed_tool(target: str, payload: str = "") -> Dict[str, Any]:
    res = embed_watermark(target, payload=payload or None)
    return {"result": res if isinstance(res, (dict, str)) else str(res)}


@tool("watermark_verify", "security", "Verify watermark integrity and tamper detection in text, dict, or file")
def watermark_verify_tool(target: str) -> Dict[str, Any]:
    return verify_watermark(target)


@tool("watermark_extract", "security", "Extract steganographic payload from text or JSON")
def watermark_extract_tool(target: str) -> Dict[str, Any]:
    return extract_watermark(target)


@tool("watermark_inspect", "security", "Inspect watermark layers and status")
def watermark_inspect_tool(target: str = "") -> Dict[str, Any]:
    if target:
        return inspect_watermark(target)
    return get_watermark_status()


__all__ = []
