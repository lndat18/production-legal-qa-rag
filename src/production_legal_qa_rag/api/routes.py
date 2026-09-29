"""Route chat/models/health và điều phối stream chat + trace Langfuse.

Module này giữ phần lifecycle độc lập với serialisation OpenAI/SSE để cùng một
contract được dùng cho cả stream và non-stream route. Một generator turn luôn
tạo đúng một ``TurnTrace`` và đúng một root trace Langfuse ``chat_turn``,
được cập nhật output/metadata/tags sau khi kết thúc (observability_spec.md 4.5).

Thứ tự xử lý của ``POST /v1/chat/completions`` (api_spec.md mục 7):

1. Xác thực (``auth.authenticate``) -> ``RequestContext``; rate limit theo
   phút (``rate_limit.enforce_rate_limit``) — cả hai chạy **trước** khi tạo
   ``StreamingResponse`` vì Starlette gửi status HTTP ngay khi bắt đầu stream
   (trước khi generator được duyệt lần đầu); không thể trả 401/429 sau đó.
2. ``messages`` hợp lệ? (``build_window`` trực tiếp ở đây, tách khỏi bước 3 vì
   lý do tương tự — ``InvalidConversationError`` phải thành 422 trước stream).
3. ``trace = TurnTrace()``; ``events = stream_chat_turn(...)``.
4. ``stream=true``: ``StreamingResponse(sse_stream(events))``; ``stream=false``:
   gom toàn bộ event thành một JSON (``build_completion``). Cả hai cập nhật
   trace và metrics trong ``finally`` của ``stream_chat_turn``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from redis.asyncio import Redis

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
from production_legal_qa_rag.observability import metrics, tracing
from production_legal_qa_rag.observability.turn_trace import (
    RuntimeVersions,
    update_turn_trace,
)

_logger = logging.getLogger(__name__)

router = APIRouter()


class ChatOrchestratorPort(Protocol):
    """Giao diện stream của conversation orchestrator dùng bởi API."""

    def stream(
        self,
        messages: list[ChatMessage],
        ctx: RequestContext,
        trace: TurnTrace,
    ) -> AsyncIterator[GenerationEvent]:
        """Phát event và điền trace của lượt hiện tại."""


async def stream_chat_turn(
    orchestrator: ChatOrchestratorPort,
    messages: list[ChatMessage],
    context: RequestContext,
    versions: RuntimeVersions,
) -> AsyncIterator[GenerationEvent]:
    """Phát một lượt orchestrator và luôn cập nhật root trace Langfuse sau cùng.

    Bọc root span Langfuse ``chat_turn`` quanh toàn bộ lượt (observability_spec.md
    mục 4.4): mọi span/generation con tạo bên trong ``orchestrator.stream`` tự
    lồng đúng vị trí qua OTel context. Mọi nhánh (trả lời, từ chối, lỗi, cache
    hit, ngắt kết nối) đều đi qua cùng một ``finally`` nên có đúng một trace với
    output/metadata/tags đủ (mục 4.5). ``metrics.record_turn`` chạy cùng chỗ vì
    cùng dùng ``trace`` đã điền đầy đủ; cả hai đều fail-safe.

    Args:
        orchestrator: Orchestrator đã được khởi tạo một lần ở API lifespan.
        messages: Messages đã parse từ request HTTP.
        context: Request context xác thực ở tầng HTTP.
        versions: Phiên bản prompt/corpus/model của runtime, gắn vào trace.

    Yields:
        Các event generation từ orchestrator.
    """
    trace = TurnTrace()
    root_query = messages[-1].content if messages else ""
    try:
        with tracing.span(
            "chat_turn",
            input=root_query,
            metadata={"request_id": context.request_id},
            user_id=context.user_id,
            session_id=context.chat_id,
        ) as root_span:
            try:
                async for event in orchestrator.stream(messages, context, trace):
                    yield event
            except asyncio.CancelledError:
                trace.outcome = "client_disconnected"
                raise
            finally:
                update_turn_trace(
                    root_span,
                    trace,
                    request_id=context.request_id,
                    versions=versions,
                )
    finally:
        metrics.record_turn(trace)


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
    versions: RuntimeVersions = request.app.state.runtime_versions
    events = stream_chat_turn(orchestrator, chat_messages, ctx, versions)
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
    """Readiness: ping Redis; lỗi -> 503 (không ping Groq/Pinecone)."""
    redis: Redis = request.app.state.redis
    try:
        await redis.ping()
    except Exception:
        _logger.warning("readyz: Redis không sẵn sàng.", exc_info=True)
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return JSONResponse({"status": "ready"})
