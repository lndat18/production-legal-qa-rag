"""Điều phối stream chat và ghi chatlog không chặn response.

Module này giữ phần lifecycle độc lập với serialisation OpenAI/SSE để cùng một
contract được dùng cho cả stream và non-stream route. Một generator turn luôn
tạo đúng một ``TurnTrace`` và lên lịch đúng một lần ghi sau khi kết thúc.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Protocol

from production_legal_qa_rag.chatlog.models import (
    ChatLogMetadata,
    TurnRecord,
    from_trace,
)
from production_legal_qa_rag.conversation.models import (
    ChatMessage,
    RequestContext,
    TurnTrace,
)
from production_legal_qa_rag.generation.models import GenerationEvent

_logger = logging.getLogger(__name__)
_SHUTDOWN_DRAIN_SECONDS = 5.0


class ChatLogRepositoryPort(Protocol):
    """Phần repository mà HTTP lifecycle cần để ghi chatlog."""

    async def record(self, turn: TurnRecord) -> None:
        """Ghi một turn mà không làm lỗi response lan ra ngoài."""


class ChatOrchestratorPort(Protocol):
    """Giao diện stream của conversation orchestrator dùng bởi API."""

    def stream(
        self,
        messages: list[ChatMessage],
        ctx: RequestContext,
        trace: TurnTrace,
    ) -> AsyncIterator[GenerationEvent]:
        """Phát event và điền trace của lượt hiện tại."""


class ChatLogTaskManager:
    """Giữ task ghi chatlog đến khi hoàn tất hoặc API tắt.

    Task được giữ tham chiếu mạnh để không bị garbage collector dọn giữa chừng.
    Một lỗi không mong đợi từ task chỉ tạo warning có ``request_id``; tuyệt đối
    không đưa nội dung câu hỏi/câu trả lời vào log ứng dụng.
    """

    def __init__(
        self,
        repository: ChatLogRepositoryPort,
        metadata: ChatLogMetadata,
    ) -> None:
        self._repository = repository
        self._metadata = metadata
        self._tasks: set[asyncio.Task[None]] = set()

    def schedule(self, trace: TurnTrace, context: RequestContext) -> None:
        """Lên lịch ghi đúng một turn mà không chờ Postgres.

        Args:
            trace: Vết đã hoàn tất của stream.
            context: Danh tính request, chỉ id nội bộ.
        """
        try:
            turn = from_trace(trace, context, metadata=self._metadata)
            task = asyncio.create_task(self._repository.record(turn))
        except Exception:
            _logger.warning(
                "Không thể lên lịch ghi chatlog (request_id=%s)",
                context.request_id,
                exc_info=True,
            )
            return
        self._tasks.add(task)
        task.add_done_callback(
            lambda completed: self._handle_completed_task(completed, context.request_id)
        )

    async def drain(self) -> None:
        """Chờ các task ghi tối đa năm giây khi API shutdown."""
        if not self._tasks:
            return
        _, pending = await asyncio.wait(
            self._tasks.copy(), timeout=_SHUTDOWN_DRAIN_SECONDS
        )
        for task in pending:
            task.cancel()
        if pending:
            _logger.warning(
                "Hết thời gian chờ %d task ghi chatlog khi shutdown", len(pending)
            )

    def _handle_completed_task(
        self,
        task: asyncio.Task[None],
        request_id: str,
    ) -> None:
        """Bỏ task đã xong và ghi warning an toàn nếu nó bị lỗi."""
        self._tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            _logger.warning("Task ghi chatlog bị huỷ (request_id=%s)", request_id)
        except Exception:
            _logger.warning(
                "Task ghi chatlog thất bại (request_id=%s)",
                request_id,
                exc_info=True,
            )


async def stream_chat_turn(
    orchestrator: ChatOrchestratorPort,
    messages: list[ChatMessage],
    context: RequestContext,
    chatlog_tasks: ChatLogTaskManager,
) -> AsyncIterator[GenerationEvent]:
    """Phát một lượt orchestrator và luôn lên lịch ghi chatlog sau cùng.

    Args:
        orchestrator: Orchestrator đã được khởi tạo một lần ở API lifespan.
        messages: Messages đã parse từ request HTTP.
        context: Request context xác thực ở tầng HTTP.
        chatlog_tasks: Manager giữ background task và metadata runtime.

    Yields:
        Các event generation từ orchestrator.
    """
    trace = TurnTrace()
    try:
        async for event in orchestrator.stream(messages, context, trace):
            yield event
    except asyncio.CancelledError:
        trace.outcome = "client_disconnected"
        raise
    finally:
        chatlog_tasks.schedule(trace, context)
