"""Per-client rate limits on expensive endpoints (ADR-0035): an in-process token bucket per client IP.

One API process per VM (ADR-0018, ADR-0034), so in-memory state is enough; a shared store (Redis) replaces
`TokenBuckets` if the API ever runs as several processes. The client IP comes from X-Forwarded-For only
when the request arrives from a trusted proxy (Caddy), otherwise from the connection.
"""

import ipaddress
import math
import threading
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request

from xcrs.config import get_settings


@dataclass
class Bucket:
    tokens: float
    updated: float


class TokenBuckets:
    """`rate` tokens a minute, at most `burst` saved up; one bucket per key."""

    def __init__(self, rate_per_minute: float, burst: int, clock=time.monotonic, max_keys: int = 50_000):
        self.rate, self.burst, self.clock, self.max_keys = rate_per_minute / 60.0, burst, clock, max_keys
        self._buckets: dict[str, Bucket] = {}
        self._lock = threading.Lock()

    def take(self, key: str) -> float:
        """0 if allowed, else the seconds until a token is available."""
        now = self.clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self.max_keys:  # don't let many addresses grow memory without bound
                    self._buckets.clear()
                bucket = self._buckets[key] = Bucket(float(self.burst), now)
            bucket.tokens = min(self.burst, bucket.tokens + (now - bucket.updated) * self.rate)
            bucket.updated = now
            if bucket.tokens >= 1:
                bucket.tokens -= 1
                return 0.0
            return (1 - bucket.tokens) / self.rate


def _trusted(address: str, networks: list) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(ip in net for net in networks)


def client_ip(request: Request, trusted: list) -> str:
    peer = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded and _trusted(peer, trusted):
        return forwarded.split(",")[-1].strip()
    return peer


class Limiter:
    def __init__(self):
        settings = get_settings()
        self.enabled = settings.rate_limits_enabled
        self.trusted = [ipaddress.ip_network(n.strip()) for n in settings.trusted_proxies.split(",") if n.strip()]
        self.buckets = {
            "heavy": TokenBuckets(settings.rate_heavy_per_minute, settings.rate_heavy_burst),
            "light": TokenBuckets(settings.rate_light_per_minute, settings.rate_light_burst),
        }

    def check(self, request: Request, kind: str) -> None:
        if not self.enabled:
            return
        wait = self.buckets[kind].take(client_ip(request, self.trusted))
        if wait:
            raise HTTPException(
                429,
                "Too many requests; please wait a moment.",
                headers={"Retry-After": str(max(1, math.ceil(wait)))},
            )


_limiter: Limiter | None = None


def limiter() -> Limiter:
    global _limiter
    if _limiter is None:
        _limiter = Limiter()
    return _limiter


def heavy(request: Request) -> None:
    """FastAPI dependency for LLM- or embedding-backed endpoints."""
    limiter().check(request, "heavy")


def light(request: Request) -> None:
    limiter().check(request, "light")
