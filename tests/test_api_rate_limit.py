"""Tests that protect the per-minute Redis rate limit contract of `api/rate_limit.py`.

api_spec.md mục 5: cửa sổ cố định theo phút qua Redis, vượt ngưỡng -> 429 kèm
``Retry-After``; Redis lỗi -> fail-open (log warning, không chặn request).
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from production_legal_qa_rag.api.rate_limit import enforce_rate_limit
from production_legal_qa_rag.api.schemas import ApiError


class _FakeRedis:
    """In-memory stand-in for the two Redis calls ``enforce_rate_limit`` makes."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.expired: dict[str, int] = {}

    async def incr(self, key: str) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key: str, ttl: int) -> None:
        self.expired[key] = ttl


class _FailingRedis:
    """Every call raises, simulating a dead Redis connection."""

    async def incr(self, key: str) -> int:
        raise ConnectionError("redis unreachable")

    async def expire(self, key: str, ttl: int) -> None:
        raise ConnectionError("redis unreachable")


def test_requests_within_limit_do_not_raise() -> None:
    """Up to (and including) the limit must be allowed."""
    redis = _FakeRedis()

    async def scenario() -> None:
        for _ in range(5):
            await enforce_rate_limit(redis, "user-1", limit_per_minute=5)

    asyncio.run(scenario())
    assert redis.counts[list(redis.counts)[0]] == 5


def test_request_over_limit_raises_429_with_retry_after() -> None:
    """The request over the threshold is rejected with a positive Retry-After."""
    redis = _FakeRedis()

    async def scenario() -> ApiError:
        for _ in range(2):
            await enforce_rate_limit(redis, "user-1", limit_per_minute=2)
        with pytest.raises(ApiError) as excinfo:
            await enforce_rate_limit(redis, "user-1", limit_per_minute=2)
        return excinfo.value

    error = asyncio.run(scenario())
    assert error.status_code == 429
    assert error.error_type == "rate_limit_error"
    assert error.code == "rate_limit_exceeded"
    assert error.retry_after_seconds is not None
    assert error.retry_after_seconds > 0


def test_first_increment_sets_expiry_but_later_ones_do_not() -> None:
    """Only the first ``INCR`` in a minute bucket should set the TTL (mục 5)."""
    redis = _FakeRedis()

    async def scenario() -> None:
        await enforce_rate_limit(redis, "user-1", limit_per_minute=5)
        await enforce_rate_limit(redis, "user-1", limit_per_minute=5)

    asyncio.run(scenario())
    assert len(redis.expired) == 1
    assert next(iter(redis.expired.values())) > 60


def test_different_users_have_independent_counters() -> None:
    """One user hitting the limit must not affect another user's quota."""
    redis = _FakeRedis()

    async def scenario() -> None:
        for _ in range(3):
            await enforce_rate_limit(redis, "user-a", limit_per_minute=3)
        # user-b's first request must still be allowed.
        await enforce_rate_limit(redis, "user-b", limit_per_minute=3)

    asyncio.run(scenario())  # no ApiError raised


def test_redis_failure_fails_open(caplog: pytest.LogCaptureFixture) -> None:
    """A dead Redis must never block chat traffic — only log a warning."""
    caplog.set_level(logging.WARNING, logger="production_legal_qa_rag.api.rate_limit")

    asyncio.run(enforce_rate_limit(_FailingRedis(), "user-1", limit_per_minute=1))

    assert "fail-open" in caplog.text
