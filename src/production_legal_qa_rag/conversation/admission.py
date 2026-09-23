"""Giới hạn đồng thời cho phần tốn LLM của luồng hội thoại.

Semaphore nằm trong process nên chỉ đúng với một worker. Khi cần nhiều replica,
thay implementation này bằng cơ chế phân tán sau cùng interface
``AdmissionController``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final, Literal

from production_legal_qa_rag.config import AdmissionSettings

# Ước lượng thô thời gian một câu trả lời (~vài chục giây) để client thử lại.
OVERLOADED_RETRY_AFTER_SECONDS: Final = 10.0

type DenialKind = Literal["overloaded"]


class AdmissionDenied(Exception):
    """Request bị từ chối vì hàng đợi trả lời đã quá tải."""

    def __init__(
        self, kind: DenialKind, retry_after_seconds: float | None = None
    ) -> None:
        super().__init__(kind)
        self.kind: DenialKind = kind
        self.retry_after_seconds = retry_after_seconds


class AdmissionController:
    """Giới hạn số request đồng thời và số request chờ trong process."""

    def __init__(self, settings: AdmissionSettings | None = None) -> None:
        self._settings = settings or AdmissionSettings()
        self._semaphore = asyncio.Semaphore(self._settings.max_concurrent_answers)
        self._waiting = 0

    @asynccontextmanager
    async def slot(self, _user_id: str) -> AsyncIterator[None]:
        """Giữ một chỗ trả lời LLM; chỉ dùng khi cache miss.

        Args:
            _user_id: Định danh người dùng đã xác thực. Giữ trong contract để
                interface có thể chuyển sang admission phân tán trong tương lai.

        Yields:
            ``None`` khi request đã vào được slot.

        Raises:
            AdmissionDenied: Hàng đợi đã vượt giới hạn cấu hình.
        """
        await self._acquire_slot()
        try:
            yield
        finally:
            self._semaphore.release()

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
