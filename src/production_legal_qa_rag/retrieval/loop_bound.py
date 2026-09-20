"""Giữ client async gắn với event loop đang chạy, tạo lại khi loop đổi.

`AsyncGroq` và `httpx.AsyncClient` bám vào event loop đầu tiên dùng chúng;
dùng lại sau khi `asyncio.run` kết thúc gây `RuntimeError: Event loop is
closed`. `LoopBoundClient` tạo client lazy trong loop hiện tại và tạo lại khi
gặp loop khác, để một pipeline dùng được qua nhiều `asyncio.run`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable


class LoopBoundClient[ClientT]:
    """Cấp client đúng với event loop hiện tại.

    Client do caller inject (`fixed_client`) được dùng nguyên, không tạo lại:
    caller chịu trách nhiệm về vòng đời của nó.
    """

    def __init__(
        self, factory: Callable[[], ClientT], fixed_client: ClientT | None = None
    ) -> None:
        self._factory = factory
        self._fixed_client = fixed_client
        self._client: ClientT | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def get(self) -> ClientT:
        """Trả về client gắn với loop đang chạy (phải gọi từ trong coroutine).

        Không thread-safe: một instance chỉ được dùng trong một thread.
        Client cũ khi loop đổi không được đóng tường minh (loop của nó thường
        đã đóng nên không thể `await close()`), chỉ được GC thu gom.
        """
        if self._fixed_client is not None:
            return self._fixed_client
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            self._client = self._factory()
            self._loop = loop
        return self._client
