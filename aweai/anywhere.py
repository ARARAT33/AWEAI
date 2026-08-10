"""AWEAI Anywhere — one-command universal deployment.

Innovative "work from anywhere" layer:
  * environment auto-detection (local / LAN / cloud / Colab / container / phone)
  * dependency-free QR code generator (pure stdlib, byte mode, ECC-L, v1-v7)
    rendered as Unicode blocks in the terminal — scan with any phone camera
  * zero-dependency public tunnel attempts: cloudflared -> ngrok -> localtunnel
    -> serveo -> ssh -R, each wrapped in try/except so it never crashes
  * `aweai anywhere` launches the UI on 0.0.0.0 and prints every reachable URL
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

# ---------------------------------------------------------------- environment

def lan_ip() -> str:
    """Best-effort LAN IPv4 of this machine (no deps)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def is_online(timeout: float = 2.0) -> bool:
    try:
        urllib.request.urlopen("https://api.github.com", timeout=timeout)
        return True
    except Exception:
        return False


def detect_environment() -> dict:
    """Return a dict describing where this process is running."""
    env = {}
    env["platform"] = sys.platform
    env["python"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    env["hostname"] = socket.gethostname()
    env["host_ips"] = _host_ips()
    env["container"] = os.path.exists("/.dockerenv") or os.path.exists("/proc/1/cgroup")
    env["colab"] = "COLAB_GPU" in os.environ or "COLAB_JUPYTER_IP" in os.environ
    env["kaggle"] = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    env["codespaces"] = "CODESPACES" in os.environ
    env["github_actions"] = "GITHUB_ACTIONS" in os.environ
    env["online"] = is_online()
    env["offline_fallback"] = True
    env["cors"] = "*"
    env["bind"] = "0.0.0.0"
    env["port_hint"] = 8888
    hostname = env["hostname"].lower()
    if env["colab"]:
        env["mode"] = "colab"
    elif env["kaggle"]:
        env["mode"] = "kaggle"
    elif env["codespaces"]:
        env["mode"] = "codespaces"
    elif env["github_actions"]:
        env["mode"] = "ci"
    elif env["container"]:
        env["mode"] = "container"
    elif any(k in hostname for k in ("localhost", "desktop", "laptop", "thinkpad", "macbook", "surface")):
        env["mode"] = "local"
    else:
        env["mode"] = "server"
    return env


def _host_ips() -> list:
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and not ip.startswith("127."):
                ips.append(ip)
    except Exception:
        pass
    if not ips:
        try:
            ips.append(lan_ip())
        except Exception:
            pass
    return ips


# ------------------------------------------------------------ QR (pure stdlib)

_QR_ALIGN = {1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34], 7: [6, 22, 38]}
# (data codewords, ec codewords) per block, ECC level L
_QR_ECL = {
    1: [(19, 7)],
    2: [(34, 10)],
    3: [(55, 15)],
    4: [(80, 20)],
    5: [(108, 26)],
    6: [(68, 18), (68, 18)],
    7: [(78, 20), (78, 20)],
}
_QR_CAP = {v: sum(d for d, _ in blocks) - 2 for v, blocks in _QR_ECL.items()}


def _format_bits(mask: int) -> int:
    data = (0b01 << 3) | mask  # ECC L = 01
    v = data << 10
    g = 0x537
    for i in range(14, 9, -1):
        if v & (1 << (i + 1)):
            v ^= g << (i - 9)
    return ((data << 10) | v) ^ 0x5412

_QR_FORMAT_L = {m: _format_bits(m) for m in range(8)}


def _gf_mul(a: int, b: int) -> int:
    r = 0
    for _ in range(8):
        if b & 1:
            r ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= 0x11D
        b >>= 1
    return r


def _rs_generator(n: int) -> list:
    g = [1]
    for i in range(n):
        nxt = [0] * (len(g) + 1)
        for j, c in enumerate(g):
            nxt[j] ^= _gf_mul(c, 1)
            nxt[j + 1] ^= c
        g = nxt
    return g


def _rs_encode(data: list, ec_count: int) -> list:
    gen = _rs_generator(ec_count)
    res = data + [0] * ec_count
    for i in range(len(data)):
        coef = res[i]
        if coef:
            for j, g in enumerate(gen[1:]):
                res[i + j + 1] ^= _gf_mul(g, coef)
    return res[-ec_count:]


def _choose_version(text: str) -> tuple:
    data = text.encode("utf-8")
    for v in sorted(_QR_CAP):
        if len(data) <= _QR_CAP[v]:
            return v, data
    raise ValueError(f"QR payload too long ({len(data)} bytes; max {max(_QR_CAP.values())})")


def _build_matrix(version: int, data: list) -> tuple:
    size = 17 + 4 * version
    m = [[0] * size for _ in range(size)]

    def setf(r, c, v):
        if 0 <= r < size and 0 <= c < size:
            m[r][c] = v

    # finder patterns + separators
    for (fr, fc) in ((0, 0), (0, size - 7), (size - 7, 0)):
        for r in range(-1, 8):
            for c in range(-1, 8):
                rr, cc = fr + r, fc + c
                if 0 <= rr < size and 0 <= cc < size:
                    v = 1 if (max(abs(r), abs(c)) != 2 and 0 <= r <= 6 and 0 <= c <= 6) else 0
                    setf(rr, cc, v)
    # timing
    for i in range(8, size - 8):
        setf(6, i, i % 2 == 0)
        setf(i, 6, i % 2 == 0)
    # dark module
    setf(size - 8, 8, 1)
    # alignment
    for r in _QR_ALIGN[version]:
        for c in _QR_ALIGN[version]:
            if m[r][c] == 1 or (r == size - 7 and c == size - 7) or (r == 6 and c == 6) or (r == 6 and c == size - 7) or (r == size - 7 and c == 6):
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    setf(r + dr, c + dc, 1 if max(abs(dr), abs(dc)) != 1 else 0)

    # collect format cells (values filled after mask choice) — 15 + 15
    format_cells = [
        (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 7),
        (8, 8), (7, 8), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8),
        (size - 1, 8), (size - 2, 8), (size - 3, 8), (size - 4, 8),
        (size - 5, 8), (size - 6, 8), (size - 7, 8),
        (8, size - 8), (8, size - 7), (8, size - 6), (8, size - 5),
        (8, size - 4), (8, size - 3), (8, size - 2), (8, size - 1),
    ]

    # version info (v>=7)
    if version >= 7:
        vi = _version_bits(version)
        for i in range(6):
            for j in range(3):
                bit = (vi >> (i * 3 + j)) & 1
                setf(size - 11 + i, j, bit)
                setf(j, size - 11 + i, bit)

    # place data bits with zigzag
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    idx = 0
    format_set = set(format_cells)
    col = size - 1
    direction = 1
    while col > 0:
        if col == 6:
            col -= 1
        for _ in range(2):
            c = col
            if direction == 1:
                riter = range(size - 1, -1, -1)
            else:
                riter = range(0, size)
            for r in riter:
                if _is_function(m, r, c, size, version) or (r, c) in format_set:
                    continue
                if idx < len(bits):
                    m[r][c] = bits[idx]
                    idx += 1
                else:
                    m[r][c] = 0
            col -= 1
            if col < 0:
                break
        direction *= -1

    return m, format_cells


def _version_bits(version: int) -> int:
    v = version << 12
    g = 0x1F25
    for i in range(17, 11, -1):
        if v & (1 << (i + 1)):
            v ^= g << (i - 11)
    return (version << 12) | v


def _is_function(m, r, c, size, version):
    # conservative: skip finder/timing/alignment/format/version areas
    if r < 9 and c < 9 and (r < 7 or c < 7):
        return True
    if r < 9 and c >= size - 8:
        return True
    if c < 9 and r >= size - 8:
        return True
    if r == 6 or c == 6:
        return True
    for a in _QR_ALIGN[version]:
        for b in _QR_ALIGN[version]:
            if abs(r - a) <= 2 and abs(c - b) <= 2:
                return True
    return False


def _apply_mask(m, mask: int, format_cells, size, version):
    for r in range(size):
        for c in range(size):
            if (r, c) in format_cells or _is_function(m, r, c, size, version):
                continue
            v = m[r][c]
            cond = {
                0: (r + c) % 2 == 0,
                1: r % 2 == 0,
                2: c % 3 == 0,
                3: (r + c) % 3 == 0,
                4: (r // 2 + c // 3) % 2 == 0,
                5: (r * c) % 2 + (r * c) % 3 == 0,
                6: ((r * c) % 2 + (r * c) % 3) % 2 == 0,
                7: ((r + c) % 2 + (r * c) % 3) % 2 == 0,
            }[mask]
            if cond:
                m[r][c] = 1 - v
    bits = _QR_FORMAT_L[mask]
    for i in range(15):
        r, c = format_cells[i]
        m[r][c] = (bits >> (14 - i)) & 1
    for i in range(15):
        r, c = format_cells[15 + i]
        m[r][c] = (bits >> i) & 1
    return m


def _penalty(m):
    size = len(m)
    score = 0
    for r in range(size):
        run = 1
        for c in range(1, size):
            if m[r][c] == m[r][c - 1]:
                run += 1
            else:
                if run >= 5:
                    score += run - 2
                run = 1
        if run >= 5:
            score += run - 2
    for c in range(size):
        run = 1
        for r in range(1, size):
            if m[r][c] == m[r - 1][c]:
                run += 1
            else:
                if run >= 5:
                    score += run - 2
                run = 1
        if run >= 5:
            score += run - 2
    for r in range(size - 1):
        for c in range(size - 1):
            if m[r][c] == m[r + 1][c] == m[r][c + 1] == m[r + 1][c + 1]:
                score += 3
    dark = sum(sum(row) for row in m)
    total = size * size
    pct = dark * 100 // total
    prev = pct // 5 * 5
    score += min(abs(prev - 50) // 5, abs(prev + 5 - 50) // 5) * 10
    return score


def make_qr(text: str) -> list:
    """Return QR matrix (list of rows of 0/1) for `text` (ECC-L, v1-v7)."""
    version, payload = _choose_version(text)
    blocks = _QR_ECL[version]
    data_codewords = sum(d for d, _ in blocks)
    # build bit stream
    bits = []
    mode = 0b0100
    count = len(payload)
    for i in range(3, -1, -1):
        bits.append((mode >> i) & 1)
    for i in range(7, -1, -1):
        bits.append((count >> i) & 1)
    for b in payload:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    # terminator + pad to byte boundary
    cap_bits = data_codewords * 8
    bits.extend([0] * min(4, cap_bits - len(bits)))
    while len(bits) % 8:
        bits.append(0)
    # pad codewords
    pad = [0xEC, 0x11]
    while len(bits) < cap_bits:
        for p in pad:
            if len(bits) >= cap_bits:
                break
            for i in range(7, -1, -1):
                bits.append((p >> i) & 1)
    # split into blocks, compute ECC
    codewords = [int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, cap_bits, 8)]
    all_data, all_ec = [], []
    pos = 0
    for d, e in blocks:
        blk = codewords[pos:pos + d]
        pos += d
        all_data.append(blk)
        all_ec.append(_rs_encode(blk, e))
    # interleave
    inter = []
    max_d = max(len(b) for b in all_data)
    for i in range(max_d):
        for b in all_data:
            if i < len(b):
                inter.append(b[i])
    max_e = max(len(e) for e in all_ec)
    for i in range(max_e):
        for e in all_ec:
            if i < len(e):
                inter.append(e[i])
    matrix, fmt = _build_matrix(version, inter)
    best = None
    best_score = None
    for mask in range(8):
        trial = [row[:] for row in matrix]
        _apply_mask(trial, mask, fmt, len(matrix), version)
        s = _penalty(trial)
        if best_score is None or s < best_score:
            best_score = s
            best = trial
    return best


def qr_to_text(matrix: list, dark: str = "\u2588\u2588", light: str = "  ") -> str:
    lines = []
    for row in matrix:
        lines.append("".join(dark if v else light for v in row))
    return "\n".join(lines)


def print_qr(text: str) -> None:
    """Render a scannable QR code in the terminal (pure stdlib)."""
    m = make_qr(text)
    border = "\u2588\u2588" * (len(m) + 2)
    out = [border]
    for row in m:
        out.append("\u2588\u2588" + qr_to_text([row]).replace("\n", "") + "\u2588\u2588")
    out.append(border)
    print("\n".join(out))


# ------------------------------------------------------------------- tunnels

def _run(cmd, timeout=12):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout or r.stderr).strip()
    except Exception as e:
        return False, str(e)


def open_tunnel(port: int, timeout: float = 30.0) -> dict:
    """Try public tunnels in order; return first working public URL."""
    attempts = []
    started = time.time()

    def remaining():
        return max(1.0, timeout - (time.time() - started))

    # 1) cloudflared (single binary, no account)
    if shutil.which("cloudflared"):
        try:
            proc = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            url = None
            while time.time() - started < remaining():
                line = proc.stdout.readline()
                if "trycloudflare.com" in line:
                    for tok in line.split():
                        if "trycloudflare.com" in tok:
                            url = tok.strip()
                            break
                if url:
                    break
            if url:
                return {"provider": "cloudflared", "url": url, "attempts": attempts + ["cloudflared"]}
            proc.kill()
            attempts.append("cloudflared (no URL)")
        except Exception as e:
            attempts.append(f"cloudflared ({e})")

    # 2) ngrok
    if shutil.which("ngrok"):
        try:
            time.sleep(2)
            ok, body = _run(["curl", "-s", "http://127.0.0.1:4040/api/tunnels"], timeout=6)
            if ok and body:
                data = json.loads(body)
                tunnels = data.get("tunnels", [])
                if tunnels:
                    return {"provider": "ngrok", "url": tunnels[0]["public_url"], "attempts": attempts + ["ngrok"]}
            attempts.append("ngrok (no tunnel)")
        except Exception as e:
            attempts.append(f"ngrok ({e})")

    # 3) localtunnel via npx (no account for short URLs)
    if shutil.which("npx"):
        try:
            proc = subprocess.Popen(
                ["npx", "-y", "localtunnel", "--port", str(port)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            url = None
            while time.time() - started < remaining():
                line = proc.stdout.readline()
                if "loca.lt" in line:
                    for tok in line.split():
                        if "loca.lt" in tok:
                            url = tok.strip()
                            break
                if url:
                    break
            if url:
                return {"provider": "localtunnel", "url": url, "attempts": attempts + ["localtunnel"]}
            proc.kill()
            attempts.append("localtunnel (no URL)")
        except Exception as e:
            attempts.append(f"localtunnel ({e})")

    # 4) serveo / ssh remote forward (last resort, no deps)
    if shutil.which("ssh"):
        try:
            proc = subprocess.Popen(
                ["ssh", "-o", "StrictHostKeyChecking=no", "-R", f"80:127.0.0.1:{port}", "serveo.net"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            url = None
            while time.time() - started < remaining():
                line = proc.stdout.readline()
                if "serveo.net" in line:
                    for tok in line.split():
                        if "serveo.net" in tok:
                            url = tok.strip()
                            break
                if url:
                    break
            if url:
                return {"provider": "serveo", "url": url, "attempts": attempts + ["serveo"]}
            proc.kill()
            attempts.append("serveo (no URL)")
        except Exception as e:
            attempts.append(f"serveo ({e})")

    return {"provider": None, "url": None, "attempts": attempts}


def public_url(port: int, timeout: float = 25.0) -> str:
    """Best-effort public URL for the running server ('' if none found)."""
    try:
        res = open_tunnel(port, timeout=timeout)
        return res.get("url") or ""
    except Exception:
        return ""


# ------------------------------------------------------------------ anywhere

def anywhere_report(port: int = 8888, with_tunnel: bool = True) -> dict:
    """Full 'anywhere' status: environment + URLs + optional public tunnel."""
    env = detect_environment()
    report = {"environment": env}
    urls = {"local": f"http://127.0.0.1:{port}", "lan": f"http://{lan_ip()}:{port}"}
    if env.get("colab"):
        try:
            urls["colab"] = f"https://{socket.gethostname()}-{port}.colab.research.google.com"
        except Exception:
            pass
    if env.get("codespaces") and os.environ.get("CODESPACE_NAME"):
        urls["codespaces"] = f"https://{os.environ['CODESPACE_NAME']}-{port}.app.github.dev"
    tunnel = None
    if with_tunnel:
        tunnel = open_tunnel(port)
        if tunnel.get("url"):
            urls["public"] = tunnel["url"]
    report["urls"] = urls
    report["tunnel"] = tunnel or {"provider": None, "url": None, "attempts": []}
    return report


def print_report(report: dict) -> None:
    env = report["environment"]
    print("=" * 60)
    print("  AWEAI Anywhere")
    print("=" * 60)
    print(f"  mode        : {env.get('mode')}")
    print(f"  platform    : {env.get('platform')} / py{env.get('python')}")
    print(f"  online      : {env.get('online')}")
    print(f"  bind        : {env.get('bind')}  (CORS {env.get('cors')})")
    print("-" * 60)
    for kind, url in report["urls"].items():
        print(f"  {kind:<9}: {url}")
    tunnel = report.get("tunnel") or {}
    if tunnel.get("url"):
        print(f"  tunnel      : {tunnel['provider']} -> {tunnel['url']}")
        print()
        print("  Scan to open on any device:")
        try:
            print_qr(tunnel["url"])
        except Exception:
            print(f"  (QR unavailable: {tunnel['url']})")
    else:
        if tunnel.get("attempts"):
            print(f"  tunnel      : none ({', '.join(tunnel['attempts'])}")
        print("  Tip: install cloudflared for a free public URL, or use ngrok.")
    print("=" * 60)


if __name__ == "__main__":  # pragma: no cover
    print_report(anywhere_report(port=int(sys.argv[1]) if len(sys.argv) > 1 else 8888, with_tunnel=False))
