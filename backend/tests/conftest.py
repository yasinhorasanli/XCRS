"""Shared test setup: rate limits (ADR-0035) are off for API tests, which all come from one client;
tests/test_limits.py exercises the limiter directly."""

import pytest

from xcrs.api import limits


@pytest.fixture(autouse=True)
def no_rate_limits():
    limiter = limits.limiter()
    enabled, limiter.enabled = limiter.enabled, False
    yield
    limiter.enabled = enabled
