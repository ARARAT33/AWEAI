"""AWEAI networking tools — DNS, HTTP, ping, ports, proxy and web utilities.

Each tool has a unique purpose and works with the standard library only.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import ssl
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from aweai.tools.registry import tool


@tool("http_get", "networking", "Fetch a URL over HTTP(S) and return status, headers and body")
def http_get(url: str, timeout: int = 15, headers: str = "{}") -> Dict[str, Any]:
    hdrs = json.loads(headers) if headers else {}
    req = urllib.request.Request(url, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(8000).decode("utf-8", errors="replace")
            return {
                "url": url,
                "status": r.status,
                "headers": dict(r.headers),
                "body": body,
            }
    except urllib.error.HTTPError as e:
        return {"url": url, "status": e.code, "error": str(e.reason), "body": e.read(2000).decode("utf-8", errors="replace")}
    except Exception as e:
        return {"url": url, "error": str(e)}


@tool("http_head", "networking", "Fetch only HTTP headers of a URL")
def http_head(url: str, timeout: int = 10) -> Dict[str, Any]:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"url": url, "status": r.status, "headers": dict(r.headers)}
    except urllib.error.HTTPError as e:
        return {"url": url, "status": e.code, "headers": dict(e.headers)}
    except Exception as e:
        return {"url": url, "error": str(e)}


@tool("http_post", "networking", "POST JSON to a URL and return the response")
def http_post(url: str, data: str = "{}", timeout: int = 15) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=data.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"url": url, "status": r.status, "body": r.read(8000).decode("utf-8", errors="replace")}
    except urllib.error.HTTPError as e:
        return {"url": url, "status": e.code, "body": e.read(2000).decode("utf-8", errors="replace")}
    except Exception as e:
        return {"url": url, "error": str(e)}


@tool("dns_lookup", "networking", "Resolve a hostname to IPv4 addresses")
def dns_lookup(host: str) -> Dict[str, Any]:
    try:
        ips = sorted({i[4][0] for i in socket.getaddrinfo(host, None, socket.AF_INET)})
        return {"host": host, "ips": ips}
    except Exception as e:
        return {"host": host, "error": str(e)}


@tool("dns_lookup_all", "networking", "Resolve a hostname to all address families (A + AAAA)")
def dns_lookup_all(host: str) -> Dict[str, Any]:
    try:
        results = []
        for i in socket.getaddrinfo(host, None):
            results.append({"family": i[0].name, "address": i[4][0]})
        return {"host": host, "records": results}
    except Exception as e:
        return {"host": host, "error": str(e)}


@tool("dns_reverse", "networking", "Reverse DNS lookup of an IP address")
def dns_reverse(ip: str) -> Dict[str, Any]:
    try:
        return {"ip": ip, "hostname": socket.gethostbyaddr(ip)[0]}
    except Exception as e:
        return {"ip": ip, "error": str(e)}


@tool("ping_host", "networking", "Ping a host (ICMP echo via system ping)")
def ping_host(host: str, count: int = 4, timeout: int = 30) -> Dict[str, Any]:
    try:
        out = subprocess.run(
            ["ping", "-c", str(count), "-W", "3", host],
            capture_output=True, text=True, timeout=timeout,
        ).stdout
        return {"host": host, "output": out[-2000:]}
    except FileNotFoundError:
        return {"host": host, "error": "ping not available"}
    except Exception as e:
        return {"host": host, "error": str(e)}


@tool("port_scan", "networking", "Scan a range of TCP ports on a host (fast connect check)")
def port_scan(host: str, ports: str = "80,443,8080", timeout: float = 1.0) -> Dict[str, Any]:
    open_ports = []
    try:
        port_list = [int(p.strip()) for p in ports.split(",") if p.strip()]
    except ValueError:
        return {"host": host, "error": "invalid ports list"}
    for port in port_list:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            if s.connect_ex((host, port)) == 0:
                open_ports.append(port)
        finally:
            s.close()
    return {"host": host, "ports": port_list, "open": open_ports, "open_count": len(open_ports)}


@tool("port_check", "networking", "Check whether a single TCP port is open")
def port_check(host: str, port: int, timeout: float = 3.0) -> Dict[str, Any]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return {"host": host, "port": port, "open": True}
    except Exception:
        return {"host": host, "port": port, "open": False}
    finally:
        s.close()


@tool("traceroute", "networking", "Run traceroute to a host (via system traceroute)")
def traceroute(host: str, max_hops: int = 20) -> Dict[str, Any]:
    try:
        out = subprocess.run(
            ["traceroute", "-m", str(max_hops), host],
            capture_output=True, text=True, timeout=60,
        ).stdout
        return {"host": host, "output": out[-3000:]}
    except FileNotFoundError:
        return {"host": host, "error": "traceroute not available"}
    except Exception as e:
        return {"host": host, "error": str(e)}


@tool("my_public_ip", "networking", "Discover the public IP via an external service")
def my_public_ip(timeout: int = 10) -> Dict[str, Any]:
    for svc in ("https://api.ipify.org", "https://ifconfig.me/ip", "https://icanhazip.com"):
        try:
            with urllib.request.urlopen(svc, timeout=timeout) as r:
                ip = r.read(100).decode("utf-8", errors="replace").strip()
                return {"ip": ip, "via": svc}
        except Exception:
            continue
    return {"error": "could not determine public IP (offline?)"}


@tool("whois_domain", "networking", "Run a whois query for a domain (if whois is installed)")
def whois_domain(domain: str) -> Dict[str, Any]:
    try:
        out = subprocess.run(["whois", domain], capture_output=True, text=True, timeout=30).stdout
        return {"domain": domain, "output": out[:4000]}
    except FileNotFoundError:
        return {"domain": domain, "error": "whois not installed"}
    except Exception as e:
        return {"domain": domain, "error": str(e)}


@tool("tls_cert", "networking", "Fetch the TLS certificate of a host (subject, issuer, expiry)")
def tls_cert(host: str, port: int = 443, timeout: int = 10) -> Dict[str, Any]:
    ctx = ssl.create_default_context()
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                return {
                    "host": host,
                    "version": ssock.version(),
                    "cipher": ssock.cipher(),
                    "subject": dict(x[0] for x in cert.get("subject", [])),
                    "issuer": dict(x[0] for x in cert.get("issuer", [])),
                    "not_before": cert.get("notBefore"),
                    "not_after": cert.get("notAfter"),
                    "serial": cert.get("serialNumber"),
                }
    except Exception as e:
        return {"host": host, "error": str(e)}


@tool("tcp_echo", "networking", "Send a TCP payload and read the response (raw socket)")
def tcp_echo(host: str, port: int, payload: str, timeout: float = 5.0) -> Dict[str, Any]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.sendall(payload.encode("utf-8"))
        data = s.recv(4096).decode("utf-8", errors="replace")
        return {"host": host, "port": port, "response": data}
    except Exception as e:
        return {"host": host, "port": port, "error": str(e)}
    finally:
        s.close()


@tool("udp_send", "networking", "Send a UDP datagram to a host:port")
def udp_send(host: str, port: int, payload: str) -> Dict[str, Any]:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.sendto(payload.encode("utf-8"), (host, port))
        return {"host": host, "port": port, "sent_bytes": len(payload)}
    finally:
        s.close()


@tool("ip_details", "networking", "Detailed classification of an IP address (private, loopback, multicast)")
def ip_details(ip: str) -> Dict[str, Any]:
    try:
        addr = ipaddress.ip_address(ip)
        return {
            "ip": ip,
            "version": addr.version,
            "private": addr.is_private,
            "loopback": addr.is_loopback,
            "multicast": addr.is_multicast,
            "link_local": addr.is_link_local,
            "global": addr.is_global,
        }
    except ValueError:
        return {"ip": ip, "error": "invalid IP"}


@tool("subnet_info", "networking", "Show network info for a CIDR subnet (address count, hosts)")
def subnet_info(cidr: str) -> Dict[str, Any]:
    try:
        net = ipaddress.ip_network(cidr, strict=False)
        return {
            "cidr": cidr,
            "network": str(net.network_address),
            "broadcast": str(net.broadcast_address),
            "prefixlen": net.prefixlen,
            "num_addresses": net.num_addresses,
            "hosts": [str(h) for h in list(net.hosts())[:5]],
        }
    except ValueError as e:
        return {"cidr": cidr, "error": str(e)}


@tool("url_parse", "networking", "Parse a URL into components")
def url_parse(url: str) -> Dict[str, Any]:
    parts = urllib.parse.urlparse(url)
    return {
        "url": url,
        "scheme": parts.scheme,
        "netloc": parts.netloc,
        "hostname": parts.hostname,
        "port": parts.port,
        "path": parts.path,
        "query": parts.query,
        "fragment": parts.fragment,
    }


@tool("url_build", "networking", "Build a URL from components")
def url_build(scheme: str = "https", host: str = "example.com", path: str = "", port: str = "", query: str = "") -> Dict[str, Any]:
    netloc = host if not port else f"{host}:{port}"
    url = f"{scheme}://{netloc}/{path.lstrip('/')}"
    if query:
        url += f"?{query}"
    return {"url": url}


@tool("url_encode", "networking", "Percent-encode a URL component")
def url_encode(text: str) -> Dict[str, Any]:
    return {"encoded": urllib.parse.quote(text, safe="")}


@tool("url_decode", "networking", "Percent-decode a URL component")
def url_decode(text: str) -> Dict[str, Any]:
    return {"decoded": urllib.parse.unquote(text)}


@tool("http_status_meaning", "networking", "Explain an HTTP status code")
def http_status_meaning(code: int) -> Dict[str, Any]:
    meanings = {
        200: "OK", 201: "Created", 202: "Accepted", 204: "No Content",
        301: "Moved Permanently", 302: "Found", 304: "Not Modified",
        400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
        404: "Not Found", 405: "Method Not Allowed", 408: "Request Timeout",
        409: "Conflict", 410: "Gone", 418: "I'm a teapot",
        422: "Unprocessable Entity", 429: "Too Many Requests",
        500: "Internal Server Error", 501: "Not Implemented",
        502: "Bad Gateway", 503: "Service Unavailable", 504: "Gateway Timeout",
    }
    return {"code": code, "meaning": meanings.get(code, "Unknown status code")}


@tool("net_iface", "networking", "List network interfaces and their IPs (via socket/ifconfig)")
def net_iface() -> Dict[str, Any]:
    try:
        out = subprocess.run(["ip", "-o", "addr"], capture_output=True, text=True, timeout=10).stdout
        return {"interfaces": out.strip()[:4000]}
    except Exception:
        return {"interfaces": ""}


@tool("net_bandwidth", "networking", "Approximate download bandwidth by fetching a known file (best effort)")
def net_bandwidth(url: str = "https://speed.cloudflare.com/__down?bytes=1000000", timeout: int = 30) -> Dict[str, Any]:
    import time

    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            n = 0
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                n += len(chunk)
        elapsed = time.time() - start
        mbps = n * 8 / elapsed / 1_000_000 if elapsed else 0
        return {"bytes": n, "seconds": round(elapsed, 3), "mbps": round(mbps, 2)}
    except Exception as e:
        return {"error": str(e)}


@tool("proxy_check", "networking", "Check environment proxy variables")
def proxy_check() -> Dict[str, Any]:
    import os

    proxies = {k: v for k, v in os.environ.items() if "proxy" in k.lower()}
    return {"proxies": proxies, "count": len(proxies)}


@tool("local_ips", "networking", "List all local IPv4 addresses on this machine")
def local_ips() -> Dict[str, Any]:
    ips = set()
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=10).stdout
        ips.update(out.split())
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return {"ips": sorted(ips)}


@tool("gateway_ip", "networking", "Find the default gateway IP (via ip route)")
def gateway_ip() -> Dict[str, Any]:
    try:
        out = subprocess.run(["ip", "route"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if line.startswith("default"):
                parts = line.split()
                if "via" in parts:
                    return {"gateway": parts[parts.index("via") + 1]}
        return {"gateway": None}
    except Exception:
        return {"gateway": None}


__all__ = []
