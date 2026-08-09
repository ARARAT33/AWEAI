"""AWEAI media tools — image, audio, video and OCR helpers.

Each tool has a unique purpose. Heavy dependencies (Pillow, ffmpeg) are
optional: tools degrade gracefully with clear error messages.
"""

from __future__ import annotations

import base64
import io
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from aweai.tools.registry import tool


def _pil():
    try:
        from PIL import Image  # type: ignore
        return Image
    except ImportError:
        return None


@tool("image_info", "media", "Read basic image metadata (size, mode, format)")
def image_info(path: str) -> Dict[str, Any]:
    Image = _pil()
    if Image is None:
        return {"error": "Pillow not installed"}
    img = Image.open(path)
    info = {"path": path, "width": img.width, "height": img.height, "mode": img.mode, "format": img.format}
    img.close()
    return info


@tool("image_resize", "media", "Resize an image to a new size (optionally save)")
def image_resize(path: str, width: int = 256, height: int = 256, output: str = "") -> Dict[str, Any]:
    Image = _pil()
    if Image is None:
        return {"error": "Pillow not installed"}
    img = Image.open(path)
    img2 = img.resize((width, height))
    if output:
        img2.save(output)
        return {"path": path, "width": width, "height": height, "output": output}
    buf = io.BytesIO()
    img2.save(buf, format="PNG")
    img.close()
    return {"path": path, "width": width, "height": height, "png_base64_len": len(buf.getvalue())}


@tool("image_grayscale", "media", "Convert an image to grayscale")
def image_grayscale(path: str, output: str = "") -> Dict[str, Any]:
    Image = _pil()
    if Image is None:
        return {"error": "Pillow not installed"}
    img = Image.open(path).convert("L")
    if output:
        img.save(output)
        return {"path": path, "output": output, "mode": "L"}
    return {"path": path, "mode": "L", "converted": True}


@tool("image_rotate", "media", "Rotate an image by degrees")
def image_rotate(path: str, degrees: float = 90.0, output: str = "") -> Dict[str, Any]:
    Image = _pil()
    if Image is None:
        return {"error": "Pillow not installed"}
    img = Image.open(path).rotate(degrees, expand=True)
    if output:
        img.save(output)
        return {"path": path, "degrees": degrees, "output": output}
    return {"path": path, "degrees": degrees, "rotated": True}


@tool("image_flip", "media", "Flip an image horizontally or vertically")
def image_flip(path: str, direction: str = "horizontal", output: str = "") -> Dict[str, Any]:
    Image = _pil()
    if Image is None:
        return {"error": "Pillow not installed"}
    img = Image.open(path)
    img2 = img.transpose(Image.FLIP_LEFT_RIGHT if direction == "horizontal" else Image.FLIP_TOP_BOTTOM)
    if output:
        img2.save(output)
        return {"path": path, "direction": direction, "output": output}
    return {"path": path, "direction": direction, "flipped": True}


@tool("image_crop", "media", "Crop an image to a box (left, top, right, bottom)")
def image_crop(path: str, left: int = 0, top: int = 0, right: int = 100, bottom: int = 100, output: str = "") -> Dict[str, Any]:
    Image = _pil()
    if Image is None:
        return {"error": "Pillow not installed"}
    img = Image.open(path)
    img2 = img.crop((left, top, right, bottom))
    if output:
        img2.save(output)
        return {"path": path, "box": [left, top, right, bottom], "output": output}
    return {"path": path, "box": [left, top, right, bottom], "cropped": True}


@tool("image_convert", "media", "Convert an image to another format (png/jpeg/webp)")
def image_convert(path: str, fmt: str = "png", output: str = "") -> Dict[str, Any]:
    Image = _pil()
    if Image is None:
        return {"error": "Pillow not installed"}
    img = Image.open(path)
    fmt = fmt.lower()
    if output:
        img.save(output, format=fmt.upper())
        return {"path": path, "format": fmt, "output": output}
    buf = io.BytesIO()
    img.save(buf, format=fmt.upper())
    return {"path": path, "format": fmt, "bytes": len(buf.getvalue())}


@tool("image_to_base64", "media", "Encode an image file to a base64 data string")
def image_to_base64(path: str) -> Dict[str, Any]:
    data = Path(path).read_bytes()
    return {"path": path, "base64": base64.b64encode(data).decode("ascii")[:2000], "bytes": len(data)}


@tool("image_thumbnail", "media", "Create a thumbnail of an image (max dimension)")
def image_thumbnail(path: str, max_size: int = 128, output: str = "") -> Dict[str, Any]:
    Image = _pil()
    if Image is None:
        return {"error": "Pillow not installed"}
    img = Image.open(path)
    img.thumbnail((max_size, max_size))
    if output:
        img.save(output)
        return {"path": path, "max_size": max_size, "output": output, "size": img.size}
    return {"path": path, "max_size": max_size, "size": img.size}


@tool("image_pixel_stats", "media", "Compute basic pixel statistics of an image (mean, extrema)")
def image_pixel_stats(path: str) -> Dict[str, Any]:
    Image = _pil()
    if Image is None:
        return {"error": "Pillow not installed"}
    img = Image.open(path).convert("RGB")
    hist = img.histogram()
    total = sum(hist)
    return {
        "path": path,
        "size": img.size,
        "bins_per_channel": 256,
        "total_pixels": total,
        "brightness_mean": sum(i * v for i, v in enumerate(hist)) / total if total else 0,
    }


@tool("audio_info", "media", "Probe audio file metadata via ffprobe (if available)")
def audio_info(path: str) -> Dict[str, Any]:
    import subprocess

    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
            capture_output=True, text=True, timeout=30,
        ).stdout
        return {"path": path, "probe": json.loads(out)}
    except FileNotFoundError:
        return {"error": "ffprobe not installed"}
    except Exception as e:
        return {"error": str(e)}


@tool("video_info", "media", "Probe video file metadata via ffprobe (if available)")
def video_info(path: str) -> Dict[str, Any]:
    return audio_info(path)


@tool("media_duration", "media", "Get media duration in seconds (via ffprobe)")
def media_duration(path: str) -> Dict[str, Any]:
    import subprocess

    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
            capture_output=True, text=True, timeout=30,
        ).stdout
        data = json.loads(out)
        return {"path": path, "duration_seconds": float(data.get("format", {}).get("duration", 0))}
    except FileNotFoundError:
        return {"error": "ffprobe not installed"}
    except Exception as e:
        return {"error": str(e)}


@tool("video_extract_frame", "media", "Extract a frame from a video at a timestamp (via ffmpeg)")
def video_extract_frame(path: str, at_seconds: float = 0.0, output: str = "frame.png") -> Dict[str, Any]:
    import subprocess

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "quiet", "-ss", str(at_seconds), "-i", path, "-frames:v", "1", output],
            check=True, timeout=60,
        )
        return {"path": path, "at_seconds": at_seconds, "output": output}
    except FileNotFoundError:
        return {"error": "ffmpeg not installed"}
    except Exception as e:
        return {"error": str(e)}


@tool("audio_convert", "media", "Convert audio format (via ffmpeg, e.g. mp3->wav)")
def audio_convert(path: str, output: str = "out.wav") -> Dict[str, Any]:
    import subprocess

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "quiet", "-i", path, output],
            check=True, timeout=120,
        )
        return {"path": path, "output": output}
    except FileNotFoundError:
        return {"error": "ffmpeg not installed"}
    except Exception as e:
        return {"error": str(e)}


@tool("video_compress", "media", "Compress a video (via ffmpeg, crf 28)")
def video_compress(path: str, output: str = "compressed.mp4", crf: int = 28) -> Dict[str, Any]:
    import subprocess

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "quiet", "-i", path, "-vcodec", "libx264", "-crf", str(crf), output],
            check=True, timeout=300,
        )
        return {"path": path, "output": output, "crf": crf}
    except FileNotFoundError:
        return {"error": "ffmpeg not installed"}
    except Exception as e:
        return {"error": str(e)}


@tool("png_metadata", "media", "Read PNG chunk metadata (IHDR dimensions) without Pillow")
def png_metadata(path: str) -> Dict[str, Any]:
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return {"error": "not a PNG file"}
    width, height = struct.unpack(">II", data[16:24])
    return {"path": path, "width": width, "height": height, "bit_depth": data[24], "color_type": data[25]}


@tool("svg_check", "media", "Check whether a file is a valid SVG (has <svg root)")
def svg_check(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return {"is_svg": "<svg" in text.lower(), "bytes": len(text)}


@tool("image_brightness", "media", "Estimate average brightness of an image (0-255)")
def image_brightness(path: str) -> Dict[str, Any]:
    Image = _pil()
    if Image is None:
        return {"error": "Pillow not installed"}
    img = Image.open(path).convert("L")
    hist = img.histogram()
    total = sum(hist)
    brightness = sum(i * v for i, v in enumerate(hist)) / total if total else 0
    return {"path": path, "brightness": round(brightness, 2)}


@tool("image_contrast", "media", "Estimate image contrast (std of grayscale pixels)")
def image_contrast(path: str) -> Dict[str, Any]:
    Image = _pil()
    if Image is None:
        return {"error": "Pillow not installed"}
    img = Image.open(path).convert("L")
    hist = img.histogram()
    total = sum(hist)
    mean = sum(i * v for i, v in enumerate(hist)) / total if total else 0
    var = sum(v * (i - mean) ** 2 for i, v in enumerate(hist)) / total if total else 0
    return {"path": path, "contrast_std": round(math.sqrt(var), 2)}


@tool("media_file_type", "media", "Detect media file type from magic bytes")
def media_file_type(path: str) -> Dict[str, Any]:
    data = Path(path).read_bytes()[:16]
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return {"type": "png"}
    if data[:3] == b"\xff\xd8\xff":
        return {"type": "jpeg"}
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return {"type": "gif"}
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return {"type": "webp"}
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return {"type": "wav"}
    if data[:4] == b"\x1aE\xdf\xa3":
        return {"type": "webm/mkv"}
    if data[:4] == b"ftyp":
        return {"type": "mp4"}
    return {"type": "unknown", "hex": data.hex()}


@tool("image_dominant", "media", "Approximate dominant color of an image (downsampled)")
def image_dominant(path: str) -> Dict[str, Any]:
    Image = _pil()
    if Image is None:
        return {"error": "Pillow not installed"}
    img = Image.open(path).convert("RGB")
    img.thumbnail((32, 32))
    pixels = list(img.getdata())
    from collections import Counter

    counts = Counter(pixels)
    top = counts.most_common(3)
    return {"path": path, "dominant": [{"rgb": list(rgb), "count": c} for rgb, c in top]}


@tool("image_histogram_json", "media", "Return the grayscale histogram of an image as JSON")
def image_histogram_json(path: str) -> Dict[str, Any]:
    Image = _pil()
    if Image is None:
        return {"error": "Pillow not installed"}
    img = Image.open(path).convert("L")
    return {"path": path, "histogram": img.histogram()}


__all__ = []
