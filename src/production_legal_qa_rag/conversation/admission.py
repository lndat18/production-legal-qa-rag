"""Quota theo ngày và giới hạn đồng thời cho phần tốn LLM (conversation_spec.md mục 8).

Quota lưu ở Redis (fail-open khi Redis lỗi); semaphore đồng thời nằm trong
process nên chỉ đúng với 1 worker. Muốn nhiều replica thì chuyển semaphore
sang Redis sau cùng interface `AdmissionController`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Final, Literal

from redis.asyncio import Redis

from production_legal_qa_rag.config import AdmissionSettings
from production_legal_qa_rag.retrieval.loop_bound import LoopBoundClient

logger = logging.getLogger(__name__)

_QUOTA_TTL_SECONDS: Final = 48 * 3600
# Ước lượng thô thời gian một câu trả lời (~vài chục giây) để client thử lại.
OVERLOADED_RETRY_AFTER_SECONDS: Final = 10.0

type DenialKind = Literal["user_quota", "global_budget", "overloaded"]


class AdmissionDenied(Exception):
    """Request bị từ chối trước khi tốn LLM."""

    def __init__(
        self, kind: DenialKind, retry_after_seconds: float | None = None
    ) -> None:
        super().__init__(kind)
        self.kind: DenialKind = kind
        self.retry_after_seconds = retry_after_seconds


class AdmissionTicket:
    """Vé của một request đã vào; cho phép hoàn quota khi lỗi hệ thống."""

    def __init__(self) -> None:
        self.refund_requested = False

    def request_refund(self) -> None:
        """Đánh dấu hoàn 1 đơn vị quota khi thoát slot (lỗi hệ thống, chưa có token)."""
        self.refund_requested = True


class AdmissionController:
    """Kiểm soát quota user/ngày, ngân sách toàn cục/ngày và đồng thời."""

    def __init__(
        self,
        settings: AdmissionSettings | None = None,
        redis_client: Redis | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._settings = settings or AdmissionSettings()  # type: ignore[call-arg]
        self._redis = LoopBoundClient(self._create_redis, redis_client)
        self._clock = clock
        self._semaphore = asyncio.Semaphore(self._settings.max_concurrent_answers)
        self._waiting = 0

    def _create_redis(self) -> Redis:
        return Redis.from_url(self._settings.redis_url)

    @asynccontextmanager
    async def slot(self, user_id: str) -> AsyncIterator[AdmissionTicket]:
        """Giữ một chỗ trả lời LLM; chỉ dùng khi cache miss.

        Args:
            user_id: Định danh người dùng đã xác thực.

        Yields:
            Vé để caller yêu cầu hoàn quota nếu luồng lỗi trước khi có token.

        Raises:
            AdmissionDenied: Vượt quota user, ngân sách toàn cục hoặc hàng đợi.
        """
        day = self._clock().strftime("%Y%m%d")
        charged = await self._charge_quota(user_id, day)
        ticket = AdmissionTicket()
        try:
            await self._acquire_slot()
        except AdmissionDenied, asyncio.CancelledError:
            # Huỷ khi đang chờ hàng đợi cũng chưa tốn LLM nên hoàn quota.
            await self._refund(charged)
            raise
        try:
            yield ticket
        finally:
            self._semaphore.release()
            if ticket.refund_requested:
                await asyncio.shield(self._refund(charged))

    async def _charge_quota(self, user_id: str, day: str) -> list[str]:
        """Tăng quota user rồi toàn cục; trả các khoá đã tính để hoàn lại được."""
        settings = self._settings
        checks: tuple[tuple[str, int, DenialKind], ...] = (
            (
                f"quota:user:{user_id}:{day}",
                settings.user_daily_llm_answers,
                "user_quota",
            ),
            (f"quota:global:{day}", settings.global_daily_llm_answers, "global_budget"),
        )
        charged: list[str] = []
        for key, limit, kind in checks:
            count = await self._increment(key)
            if count is None:  # Redis lỗi: fail-open.
                continue
            charged.append(key)
            if count > limit:
                await self._refund(charged)
                raise AdmissionDenied(kind)
        return charged

    async def _acquire_slot(self) -> None:
        if self._semaphore.locked() and self._waiting >= self._settings.max_waiting:
            raise AdmissionDenied(
                "overloaded", retry_after_seconds=OVERLOADED_RETRY_AFTER_SECONDS
            )
        self._waiting += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._waiting -= 1

    async def _increment(self, key: str) -> int | None:
        try:
            async with self._redis.get().pipeline(transaction=True) as pipe:
                pipe.incr(key)
                pipe.expire(key, _QUOTA_TTL_SECONDS)
                results = await pipe.execute()
            return int(results[0])
        except Exception:
            logger.warning("Redis quota lỗi, bỏ qua giới hạn.", exc_info=True)
            return None

    async def _refund(self, keys: list[str]) -> None:
        for key in keys:
            try:
                await self._redis.get().decr(key)
            except Exception:
                logger.warning("Không hoàn được quota trên Redis.", exc_info=True)
