"""Khoá Redis best-effort chống cache stampede cho answer generation."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

from redis.asyncio import Redis

from production_legal_qa_rag.cache.keys import lock_key
from production_legal_qa_rag.cache.models import CachedAnswer
from production_legal_qa_rag.cache.store import AnswerCache

logger = logging.getLogger(__name__)

LOCK_TTL_SECONDS: Final = 60
FOLLOWER_WAIT_SECONDS: Final = 45
FOLLOWER_POLL_SECONDS: Final = 0.2
_RELEASE_LOCK_SCRIPT: Final = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('del', KEYS[1]) end return 0"
)


class SingleFlight:
    """Phối hợp request cùng query qua Redis, lỗi Redis vẫn để request tự chạy."""

    def __init__(self, redis: Redis, answer_cache: AnswerCache) -> None:
        self._redis = redis
        self._answer_cache = answer_cache

    @asynccontextmanager
    async def acquire(
        self, standalone_query: str, request_id: str | None = None
    ) -> AsyncIterator[Flight]:
        """Lấy khoá cho query và trả một flight leader/follower.

        Args:
            standalone_query: Câu hỏi độc lập cần sinh answer.
            request_id: ID request từ API; tự sinh khi caller không có sẵn.

        Yields:
            Flight báo vai trò hiện tại; follower có thể chờ answer cache.
        """
        request_id = request_id or uuid.uuid4().hex
        key = lock_key(self._answer_cache.key_for(standalone_query))
        acquired = await self._try_acquire(key, request_id)
        flight = Flight(
            redis=self._redis,
            answer_cache=self._answer_cache,
            standalone_query=standalone_query,
            lock_key=key,
            request_id=request_id,
            is_leader=acquired,
            owns_lock=acquired,
        )
        try:
            yield flight
        finally:
            if flight.owns_lock:
                await self._release(key, request_id)

    async def _try_acquire(self, key: str, request_id: str) -> bool:
        try:
            return bool(
                await self._redis.set(key, request_id, nx=True, ex=LOCK_TTL_SECONDS)
            )
        except Exception:
            logger.warning(
                "Không lấy được single-flight lock; request sẽ tự xử lý.",
                exc_info=True,
            )
            return True

    async def _release(self, key: str, request_id: str) -> None:
        try:
            await self._redis.eval(_RELEASE_LOCK_SCRIPT, 1, key, request_id)
        except Exception:
            logger.warning(
                "Không nhả được single-flight lock; chờ TTL hết hạn.", exc_info=True
            )


class Flight:
    """Trạng thái một request trong nhóm single-flight."""

    def __init__(
        self,
        *,
        redis: Redis,
        answer_cache: AnswerCache,
        standalone_query: str,
        lock_key: str,
        request_id: str,
        is_leader: bool,
        owns_lock: bool,
    ) -> None:
        self._redis = redis
        self._answer_cache = answer_cache
        self._standalone_query = standalone_query
        self._lock_key = lock_key
        self._request_id = request_id
        self.is_leader = is_leader
        self.owns_lock = owns_lock

    async def wait_for_answer(self) -> CachedAnswer | None:
        """Chờ leader ghi cache, hoặc trở thành leader khi lock biến mất.

        Returns:
            Cached answer nếu leader hoàn thành; ``None`` khi caller cần tự generate.
        """
        deadline = asyncio.get_running_loop().time() + FOLLOWER_WAIT_SECONDS
        while True:
            hit = await self._answer_cache.get(self._standalone_query)
            if hit is not None:
                return hit
            if await self._try_take_over():
                return None
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                if await self._try_take_over():
                    return None
                return None
            await asyncio.sleep(min(FOLLOWER_POLL_SECONDS, remaining))

    async def _try_take_over(self) -> bool:
        try:
            acquired = await self._redis.set(
                self._lock_key,
                self._request_id,
                nx=True,
                ex=LOCK_TTL_SECONDS,
            )
        except Exception:
            logger.warning(
                "Không kiểm tra được single-flight lock; follower sẽ tự xử lý.",
                exc_info=True,
            )
            self.is_leader = True
            return True
        if not acquired:
            return False
        self.is_leader = True
        self.owns_lock = True
        return True
