# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""AWEAI Watermark Engine — Indelible, Multi-Layer Visible & Invisible Watermarking.

Provides cryptographic signature, zero-width unicode steganography in text/json,
floating-point array weight perturbation, file integrity checking, and tamper detection.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# Default root secret key for indelible signature hashing
DEFAULT_WATERMARK_SECRET = "ARARAT33_AWEAI_PRODUCTION_WATERMARK_KEY_2026"
DEFAULT_WATERMARK_TEXT = "Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved."

# Steganographic zero-width unicode character set
ZWC_START = "\u200d"  # Zero Width Joiner
ZWC_ZERO = "\u200b"   # Zero Width Space (bit 0)
ZWC_ONE = "\u200c"    # Zero Width Non-Joiner (bit 1)
ZWC_SEP = "\ufeff"    # Zero Width No-Break Space (separator)


def _compute_hmac(payload: str, secret: str = DEFAULT_WATERMARK_SECRET) -> str:
    """Compute HMAC-SHA256 digest for payload."""
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def text_to_zwc(payload: str) -> str:
    """Convert payload text into invisible zero-width unicode steganographic string."""
    encoded_bytes = payload.encode("utf-8")
    bit_stream = "".join(f"{b:08b}" for b in encoded_bytes)
    zwc_body = "".join(ZWC_ONE if bit == "1" else ZWC_ZERO for bit in bit_stream)
    return f"{ZWC_START}{zwc_body}{ZWC_START}"


def zwc_to_text(zwc_str: str) -> Optional[str]:
    """Extract and decode zero-width unicode steganographic string to text payload."""
    if ZWC_START not in zwc_str:
        return None
    try:
        parts = zwc_str.split(ZWC_START)
        for segment in parts:
            if not segment:
                continue
            bits = [ "1" if c == ZWC_ONE else "0" for c in segment if c in (ZWC_ZERO, ZWC_ONE) ]
            if not bits or len(bits) % 8 != 0:
                continue
            bit_str = "".join(bits)
            byte_list = [int(bit_str[i : i + 8], 2) for i in range(0, len(bit_str), 8)]
            decoded = bytes(byte_list).decode("utf-8", errors="ignore")
            if decoded:
                return decoded
    except Exception:
        pass
    return None


class AWEAIWatermarkEngine:
    """Multi-layer indelible watermarking engine for text, dicts, arrays, models and files."""

    def __init__(self, secret_key: str = DEFAULT_WATERMARK_SECRET, owner: str = "ARARAT33"):
        self.secret_key = secret_key
        self.owner = owner

    def embed_text(
        self,
        text: str,
        payload: Optional[str] = None,
        visible_header: bool = True,
        visible_footer: bool = True,
    ) -> str:
        """Embed visible headers/footers and invisible steganographic zero-width watermark into text."""
        watermark_payload = payload or f"{DEFAULT_WATERMARK_TEXT} [Owner: {self.owner}]"
        sig = _compute_hmac(watermark_payload, self.secret_key)[:16]
        zwc_mark = text_to_zwc(f"{watermark_payload}|SIG:{sig}")

        header = f"<!-- AWEAI WATERMARK: {self.owner} | SIG:{sig} -->\n" if visible_header else ""
        footer = f"\n<!-- Based on AWEAI - Copyright (c) 2026 {self.owner} -->" if visible_footer else ""

        # Inject steganographic mark invisibly in the middle or end of text
        if len(text) > 10:
            mid = len(text) // 2
            text_with_stego = text[:mid] + zwc_mark + text[mid:]
        else:
            text_with_stego = text + zwc_mark

        return f"{header}{text_with_stego}{footer}"

    def extract_text(self, text: str) -> Dict[str, Any]:
        """Extract visible and invisible watermark data from text."""
        stego_payload = zwc_to_text(text)
        has_visible = self.owner in text or "AWEAI WATERMARK" in text or "Based on AWEAI" in text

        sig_valid = False
        extracted_sig = None
        extracted_payload = None

        if stego_payload and "|SIG:" in stego_payload:
            parts = stego_payload.rsplit("|SIG:", 1)
            extracted_payload = parts[0]
            extracted_sig = parts[1]
            expected_sig = _compute_hmac(extracted_payload, self.secret_key)[:16]
            sig_valid = (extracted_sig == expected_sig)

        return {
            "has_watermark": bool(stego_payload or has_visible),
            "steganographic": bool(stego_payload),
            "visible": has_visible,
            "signature_valid": sig_valid,
            "extracted_payload": extracted_payload or stego_payload,
            "extracted_sig": extracted_sig,
        }

    def embed_dict(
        self,
        data: Dict[str, Any],
        payload: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Embed visible metadata, steganographic string, and HMAC signature into a dictionary."""
        out = dict(data)
        watermark_payload = payload or f"{DEFAULT_WATERMARK_TEXT} [Owner: {self.owner}]"
        sig = _compute_hmac(watermark_payload, self.secret_key)
        zwc_mark = text_to_zwc(f"{watermark_payload}|SIG:{sig[:16]}")

        out["_watermark"] = DEFAULT_WATERMARK_TEXT
        out["_watermark_owner"] = self.owner
        out["_watermark_stego"] = zwc_mark
        out["_watermark_sig"] = sig[:16]

        # Compute hash over core fields
        core_keys = sorted([k for k in out.keys() if not k.startswith("_watermark")])
        canonical_str = json.dumps({k: out[k] for k in core_keys}, sort_keys=True, default=str)
        out["_stego_hash"] = _compute_hmac(canonical_str, self.secret_key)

        return out

    def verify_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify dictionary watermark validity and check for tampering."""
        if not isinstance(data, dict):
            return {"valid": False, "tampered": True, "reason": "not a dict"}

        has_vis = "_watermark" in data
        has_owner = data.get("_watermark_owner") == self.owner
        zwc_val = data.get("_watermark_stego", "")
        stego_info = self.extract_text(str(zwc_val)) if zwc_val else {"signature_valid": False}

        # Verify HMAC hash integrity
        tampered = False
        stego_hash = data.get("_stego_hash")
        if stego_hash:
            core_keys = sorted([k for k in data.keys() if not k.startswith("_watermark") and k != "_stego_hash"])
            canonical_str = json.dumps({k: data[k] for k in core_keys}, sort_keys=True, default=str)
            expected_hash = _compute_hmac(canonical_str, self.secret_key)
            if stego_hash != expected_hash:
                tampered = True

        valid = bool(has_vis and stego_info["signature_valid"] and not tampered)

        return {
            "valid": valid,
            "tampered": tampered,
            "visible_mark": has_vis,
            "owner_matches": has_owner,
            "signature_valid": stego_info.get("signature_valid", False),
            "payload": stego_info.get("extracted_payload"),
        }

    def embed_array(
        self,
        arr: np.ndarray,
        payload: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Embed micro-perturbation steganographic watermark into floating-point array."""
        arr_copy = np.array(arr, copy=True)
        watermark_payload = payload or f"{DEFAULT_WATERMARK_TEXT} [{self.owner}]"
        sig = _compute_hmac(watermark_payload, self.secret_key)[:16]

        if arr_copy.size > 0 and np.issubdtype(arr_copy.dtype, np.floating):
            flat = arr_copy.ravel()
            seed = int(hashlib.md5(sig.encode("utf-8")).hexdigest()[:8], 16) % (2**31 - 1)
            rng = np.random.RandomState(seed)
            indices = rng.choice(flat.size, min(16, flat.size), replace=False)
            # Subtle perturbation (1e-7) preserving numerical behavior
            for idx in indices:
                flat[idx] += 1e-7

        array_hash = hashlib.sha256(arr_copy.tobytes()).hexdigest()[:16]
        meta = {
            "owner": self.owner,
            "payload": watermark_payload,
            "sig": sig,
            "array_hash": array_hash,
            "stego_applied": True,
        }
        return arr_copy, meta

    def verify_array(self, arr: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Verify watermark metadata against floating-point array."""
        if not isinstance(meta, dict) or "sig" not in meta:
            return {"valid": False, "tampered": True, "reason": "missing meta"}

        expected_sig = _compute_hmac(meta.get("payload", ""), self.secret_key)[:16]
        sig_ok = (meta.get("sig") == expected_sig)
        curr_hash = hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()[:16]
        hash_ok = (meta.get("array_hash") == curr_hash)

        return {
            "valid": bool(sig_ok and hash_ok),
            "tampered": not hash_ok,
            "signature_valid": sig_ok,
            "hash_match": hash_ok,
        }

    def embed_file(self, filepath: Union[str, Path], out_filepath: Optional[Union[str, Path]] = None) -> str:
        """Apply visible and steganographic watermark to file on disk."""
        path = Path(filepath)
        out_path = Path(out_filepath) if out_filepath else path

        content = path.read_text(encoding="utf-8")
        if path.suffix in (".json", ".geojson"):
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    watermarked_dict = self.embed_dict(data)
                    out_path.write_text(json.dumps(watermarked_dict, indent=2, ensure_ascii=False), encoding="utf-8")
                    return str(out_path)
            except Exception:
                pass

        watermarked_text = self.embed_text(content)
        out_path.write_text(watermarked_text, encoding="utf-8")
        return str(out_path)

    def verify_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Verify watermark in file on disk."""
        path = Path(filepath)
        if not path.exists():
            return {"valid": False, "tampered": True, "reason": "file not found"}

        content = path.read_text(encoding="utf-8")
        if path.suffix in (".json", ".geojson"):
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    return self.verify_dict(data)
            except Exception:
                pass

        return self.extract_text(content)


# Global default engine instance
_default_engine = AWEAIWatermarkEngine()


def _is_existing_file(target: Any) -> bool:
    if not isinstance(target, (str, Path)):
        return False
    target_str = str(target)
    if len(target_str) > 255 or "\n" in target_str or "\r" in target_str:
        return False
    try:
        return Path(target_str).exists() and Path(target_str).is_file()
    except Exception:
        return False


def embed_watermark(
    target: Any,
    payload: Optional[str] = None,
    secret_key: str = DEFAULT_WATERMARK_SECRET,
) -> Any:
    """Universal function to watermark text, dict, array, or file path."""
    engine = AWEAIWatermarkEngine(secret_key=secret_key)
    if _is_existing_file(target):
        return engine.embed_file(target)
    elif isinstance(target, str):
        return engine.embed_text(target, payload=payload)
    elif isinstance(target, dict):
        return engine.embed_dict(target, payload=payload)
    elif isinstance(target, np.ndarray):
        return engine.embed_array(target, payload=payload)
    else:
        raise TypeError(f"Unsupported target type for watermarking: {type(target)}")


def verify_watermark(
    target: Any,
    secret_key: str = DEFAULT_WATERMARK_SECRET,
) -> Dict[str, Any]:
    """Universal function to verify watermark in text, dict, array, or file path."""
    engine = AWEAIWatermarkEngine(secret_key=secret_key)
    if _is_existing_file(target):
        return engine.verify_file(target)
    elif isinstance(target, str):
        return engine.extract_text(target)
    elif isinstance(target, dict):
        return engine.verify_dict(target)
    else:
        raise TypeError(f"Unsupported target type for watermark verification: {type(target)}")


def extract_watermark(text_or_dict: Any) -> Dict[str, Any]:
    """Extract steganographic watermark from text or dict."""
    return _default_engine.extract_text(str(text_or_dict)) if not isinstance(text_or_dict, dict) else _default_engine.verify_dict(text_or_dict)


def inspect_watermark(target: Any) -> Dict[str, Any]:
    """Comprehensive inspection of watermark layers across target."""
    return verify_watermark(target)


def get_watermark_status() -> Dict[str, Any]:
    """Return watermark engine status and system configuration."""
    return {
        "engine": "AWEAIWatermarkEngine",
        "owner": "ARARAT33",
        "steganography": "Zero-Width Unicode (ZWC)",
        "hash_algorithm": "HMAC-SHA256",
        "layers": ["visible_header", "visible_footer", "zwc_steganography", "hmac_signature", "array_perturbation"],
        "status": "active",
    }


__all__ = [
    "AWEAIWatermarkEngine",
    "DEFAULT_WATERMARK_SECRET",
    "DEFAULT_WATERMARK_TEXT",
    "text_to_zwc",
    "zwc_to_text",
    "embed_watermark",
    "verify_watermark",
    "extract_watermark",
    "inspect_watermark",
    "get_watermark_status",
]
