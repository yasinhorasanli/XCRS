"""Abuse protection (ADR-0035): token buckets, the client IP behind a trusted proxy, 429 on the endpoints,
and the cap on new LLM matches per request."""

import ipaddress

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from xcrs.api import limits
from xcrs.catalog import model
from xcrs.domain.skill_matching import LexicalIndex
from xcrs.services.skill_matching import SkillMatcher


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def request(peer: str, forwarded: str | None = None) -> Request:
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return Request({"type": "http", "client": (peer, 1234), "headers": headers, "method": "POST", "path": "/"})


def test_a_bucket_allows_a_burst_then_refills_at_the_rate():
    clock = Clock()
    buckets = limits.TokenBuckets(rate_per_minute=6, burst=2, clock=clock)
    assert buckets.take("a") == 0 and buckets.take("a") == 0
    assert buckets.take("a") == pytest.approx(10.0)  # one token every 10 s
    assert buckets.take("b") == 0  # other clients are unaffected
    clock.now = 10.0
    assert buckets.take("a") == 0


def test_forwarded_for_counts_only_from_a_trusted_proxy():
    trusted = [ipaddress.ip_network("172.16.0.0/12")]
    assert limits.client_ip(request("172.18.0.5", "198.51.100.7"), trusted) == "198.51.100.7"
    assert limits.client_ip(request("203.0.113.9", "198.51.100.7"), trusted) == "203.0.113.9"  # spoofed header
    # Funnel → Docker → Caddy: the visitor is the rightmost entry that isn't one of our proxies
    chain = "203.0.113.50, 172.18.0.1"
    assert limits.client_ip(request("172.18.0.5", chain), trusted) == "203.0.113.50"
    assert limits.client_ip(request("172.18.0.5", "172.18.0.1"), trusted) == "172.18.0.1"  # only proxies: first


def test_an_exhausted_bucket_answers_429_with_retry_after():
    limiter = limits.Limiter()
    limiter.enabled = True
    limiter.buckets["heavy"] = limits.TokenBuckets(rate_per_minute=1, burst=1, clock=Clock())
    limiter.check(request("203.0.113.9"), "heavy")
    with pytest.raises(HTTPException) as caught:
        limiter.check(request("203.0.113.9"), "heavy")
    assert caught.value.status_code == 429 and caught.value.headers["Retry-After"] == "60"


def test_only_the_first_new_phrases_of_a_request_go_to_the_llm():
    cat = model.load_catalog()
    index = LexicalIndex.build((s.id, s.name, s.onet) for s in cat.skills.values())

    class Store:
        catalog_checksum = "c"

        def similarities(self, vector):
            return {"python": 0.9}

        def cached(self, key, picker):
            return None

        def save(self, *args):
            pass

    class Picker:
        prompt_version, model, calls = "p", "m", 0

        def pick(self, phrase):
            Picker.calls += 1
            return ["python"]

    class Embedder:
        def embed_query(self, texts):
            return [[1.0]]

    matcher = SkillMatcher(index, Store(), Embedder(), Picker(), max_new=2)
    results = matcher.match(["snake language one", "snake language two", "snake language three"])
    assert Picker.calls == 2
    assert [r.method for r in results] == ["llm", "llm", "embedding"]


def test_every_write_endpoint_has_a_rate_limit():
    """Security audit 2026-10-26 (#1): feedback and the dev profile run had none. /internal/sign-in is for the
    Nuxt server only (its secret), and limiting it would limit every visitor's sign-in behind one address."""
    from fastapi.routing import APIRoute

    from xcrs.api.app import app

    def calls(dependant):
        yield dependant.call
        for d in dependant.dependencies:
            yield from calls(d)

    unlimited = [
        (sorted(r.methods), r.path)
        for r in app.routes
        if isinstance(r, APIRoute)
        and r.methods & {"POST", "PUT", "PATCH", "DELETE"}
        and not r.path.startswith("/internal/")
        and not {limits.heavy, limits.light} & set(calls(r.dependant))
    ]
    assert unlimited == []


def test_ipv6_clients_share_a_bucket_per_64_prefix():
    """Security audit 2026-10-26 (#18): one host usually has a whole /64 to rotate through."""
    limiter = limits.Limiter()
    limiter.enabled = True
    limiter.buckets["heavy"] = limits.TokenBuckets(rate_per_minute=1, burst=1, clock=Clock())
    limiter.check(request("2001:db8:1:2::1"), "heavy")
    with pytest.raises(HTTPException):
        limiter.check(request("2001:db8:1:2:ffff::9"), "heavy")
    limiter.check(request("2001:db8:1:3::1"), "heavy")  # another /64
    limiter.check(request("203.0.113.9"), "heavy")
    with pytest.raises(HTTPException):
        limiter.check(request("203.0.113.9"), "heavy")
    limiter.check(request("203.0.113.10"), "heavy")  # IPv4 addresses stay one bucket each
    assert limits.client_key("::ffff:203.0.113.9") == "203.0.113.9" and limits.client_key("unknown") == "unknown"
