"""Route chat/models/health và điều phối stream chat + ghi chatlog.

Module này giữ phần lifecycle độc lập với serialisation OpenAI/SSE để cùng một
contract được dùng cho cả stream và non-stream route. Một generator turn luôn
tạo đúng một ``TurnTrace`` và lên lịch đúng một lần ghi sau khi kết thúc.

Thứ tự xử lý của ``POST /v1/chat/completions`` (api_spec.md mục 7):

1. Xác thực (``auth.authenticate``) -> ``RequestContext``; rate limit theo
   phút (``rate_limit.enforce_rate_limit``) — cả hai chạy **trước** khi tạo
   ``StreamingResponse`` vì Starlette gửi status HTTP ngay khi bắt đầu stream
   (trước khi generator được duyệt lần đầu); không thể trả 401/429 sau đó.
2. ``messages`` hợp lệ? (``build_window`` trực tiếp ở đây, tách khỏi bước 3 vì
   lý do tương tự — ``InvalidConversationError`` phải thành 422 trước stream).
3. ``trace = TurnTrace()``; ``events = stream_chat_turn(...)``.
4. ``stream=true``: ``StreamingResponse(sse_stream(events))``; ``stream=false``:
   gom toàn bộ event thành một JSON (``build_completion``). Cả hai ghi chatlog
   trong ``finally`` của ``stream_chat_turn``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from production_legal_qa_rag.api.auth import authenticate
from production_legal_qa_rag.api.openai_format import build_completion, sse_stream
from production_legal_qa_rag.api.rate_limit import enforce_rate_limit
from production_legal_qa_rag.api.schemas import (
    MODEL_ID,
    MODEL_OWNED_BY,
    ApiError,
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
    ModelListResponse,
    ModelObject,
)
from production_legal_qa_rag.chatlog.models import (
    ChatLogMetadata,
    TurnRecord,
    from_trace,
)
from production_legal_qa_rag.config import ApiSettings
from production_legal_qa_rag.conversation.history import (
    InvalidConversationError,
    build_window,
)
from production_legal_qa_rag.conversation.models import (
    ChatMessage,
    RequestContext,
    TurnTrace,
)
from production_legal_qa_rag.generation.models import GenerationEvent

_logger = logging.getLogger(__name__)
_SHUTDOWN_DRAIN_SECONDS = 5.0

router = APIRouter()


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
        except Exception:  # noqa: BLE001 - logging must not affect the response path.
            _logger.warning(
                "Không thể lên lịch ghi chatlog (request_id=%s)",
                context.request_id,
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
        except Exception:  # noqa: BLE001 - task failures cannot affect the response path.
            _logger.warning(
                "Task ghi chatlog thất bại (request_id=%s)",
                request_id,
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


def _to_chat_messages(
    messages: list[ChatCompletionRequestMessage],
) -> list[ChatMessage]:
    """Giữ lại message ``user``/``assistant``; bỏ ``system`` (mục 9)."""
    chat_messages: list[ChatMessage] = []
    for message in messages:
        if message.role == "user":
            chat_messages.append(ChatMessage(role="user", content=message.content))
        elif message.role == "assistant":
            chat_messages.append(ChatMessage(role="assistant", content=message.content))
    return chat_messages


def _validate_messages(messages: list[ChatMessage]) -> None:
    """Kiểm tra ``messages`` hợp lệ trước khi mở stream (mục 7 bước 2).

    Raises:
        ApiError: 422 khi ``InvalidConversationError``.
    """
    try:
        build_window(messages)
    except InvalidConversationError as exc:
        raise ApiError(
            422, str(exc), "invalid_request_error", "invalid_messages"
        ) from exc


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    ctx: Annotated[RequestContext, Depends(authenticate)],
) -> StreamingResponse | JSONResponse:
    """Endpoint chính, tương thích ``POST /v1/chat/completions`` OpenAI."""
    settings: ApiSettings = request.app.state.api_settings
    redis: Redis = request.app.state.redis
    await enforce_rate_limit(redis, ctx.user_id, settings.rate_limit_per_minute)

    chat_messages = _to_chat_messages(payload.messages)
    _validate_messages(chat_messages)

    orchestrator = request.app.state.orchestrator
    chatlog_tasks: ChatLogTaskManager = request.app.state.chatlog_tasks
    events = stream_chat_turn(orchestrator, chat_messages, ctx, chatlog_tasks)
    include_usage = bool(
        payload.stream_options and payload.stream_options.include_usage
    )
    headers = {"X-Request-Id": ctx.request_id}

    if payload.stream:
        body = sse_stream(
            events,
            model=MODEL_ID,
            include_usage=include_usage,
            keepalive_seconds=settings.keepalive_seconds,
        )
        return StreamingResponse(body, media_type="text/event-stream", headers=headers)

    response = await build_completion(
        events, model=MODEL_ID, include_usage=include_usage
    )
    return JSONResponse(response.model_dump(exclude_none=True), headers=headers)


@router.get("/v1/models")
async def list_models(
    ctx: Annotated[RequestContext, Depends(authenticate)],
) -> ModelListResponse:
    """Luôn trả đúng một model để OpenWebUI không cho chọn model khác (mục 2)."""
    return ModelListResponse(data=[ModelObject(id=MODEL_ID, owned_by=MODEL_OWNED_BY)])


@router.get("/healthz")
async def healthz() -> JSONResponse:
    """Liveness: process còn sống -> 200, không kiểm tra dependency."""
    return JSONResponse({"status": "ok"})


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    """Readiness: ping Redis + Postgres; lỗi -> 503 (không ping Groq/Pinecone)."""
    redis: Redis = request.app.state.redis
    engine: AsyncEngine = request.app.state.database_engine
    ready = True
    try:
        await redis.ping()
    except Exception:
        _logger.warning("readyz: Redis không sẵn sàng.", exc_info=True)
        ready = False
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception:
        _logger.warning("readyz: Postgres không sẵn sàng.", exc_info=True)
        ready = False
    status_code = 200 if ready else 503
    return JSONResponse(
        {"status": "ready" if ready else "not_ready"}, status_code=status_code
    )
