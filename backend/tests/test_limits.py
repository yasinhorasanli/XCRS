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
