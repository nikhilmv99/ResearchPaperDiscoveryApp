"""Download a PDF from a URL that came from a search result, i.e. from someone we do not trust.

Without this, a paper record could point the server at itself, the local network or a cloud metadata
address (SSRF), or at an endless file. Every redirect hop is checked again, and the connection goes to
the exact IP address that was checked, so DNS cannot change in between.
"""
import ipaddress
import socket
import time

import httpx

from . import config

MAX_REDIRECTS = 5
TOTAL_SECONDS = 60
PORTS = {"http": 80, "https": 443}
_NAT64 = ipaddress.ip_network("64:ff9b::/96")


class UnsafeURL(ValueError):
    pass


def _is_public(addr: str) -> bool:
    ip = ipaddress.ip_address(addr.split("%")[0])
    if ip.version == 6:
        if ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        elif ip in _NAT64:
            ip = ipaddress.ip_address(int(ip) & 0xFFFFFFFF)
        elif ip.sixtofour or ip.teredo:
            return False
    return ip.is_global and not ip.is_multicast


def resolve(url: httpx.URL) -> str:
    """The public IP address to connect to, or UnsafeURL."""
    if url.scheme not in PORTS:
        raise UnsafeURL("only http and https are allowed")
    if url.userinfo:
        raise UnsafeURL("login details in the URL are not allowed")
    if url.port not in (None, PORTS[url.scheme]):
        raise UnsafeURL(f"port {url.port} is not allowed")
    host = url.raw_host.decode("ascii")
    if not host:
        raise UnsafeURL("no host in the URL")
    try:
        infos = socket.getaddrinfo(host, PORTS[url.scheme], type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as e:
        raise UnsafeURL(f"cannot resolve {host}") from e
    addrs = [i[4][0] for i in infos]
    if not addrs or not all(_is_public(a) for a in addrs):
        raise UnsafeURL(f"{host} is not a public address")
    # IPv4 first: many machines have no IPv6 route
    return next((a for a in addrs if ":" not in a), addrs[0])


def fetch_pdf(url: str) -> bytes:
    limit = config.GUARD_PDF_MB * 1024 * 1024
    deadline = time.monotonic() + TOTAL_SECONDS
    current = httpx.URL(url)
    # trust_env=False: a proxy would do its own DNS lookup and undo the check
    with httpx.Client(timeout=20, follow_redirects=False, trust_env=False) as client:
        for _ in range(MAX_REDIRECTS + 1):
            ip = resolve(current)
            req = client.build_request(
                "GET", current.copy_with(host=ip),
                headers={"Host": current.netloc.decode("ascii"), "Accept": "application/pdf",
                         "Accept-Encoding": "identity", "User-Agent": "ResearchSearchAgent/1.0"},
                extensions={"sni_hostname": current.raw_host.decode("ascii")})  # TLS still checks the real name
            r = client.send(req, stream=True)
            try:
                if r.has_redirect_location:
                    current = current.join(r.headers["location"])
                    client.cookies.clear()
                    continue
                r.raise_for_status()
                data = bytearray()
                for chunk in r.iter_raw():
                    data += chunk
                    if len(data) > limit:
                        raise UnsafeURL(f"file is larger than {config.GUARD_PDF_MB} MB")
                    if time.monotonic() > deadline:
                        raise UnsafeURL("download took too long")
            finally:
                r.close()
            if b"%PDF" not in data[:1024]:
                raise UnsafeURL("not a PDF")
            return bytes(data)
    raise UnsafeURL("too many redirects")