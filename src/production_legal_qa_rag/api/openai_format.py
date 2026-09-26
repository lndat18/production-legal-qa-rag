"""Chuyển `GenerationEvent` sang định dạng OpenAI (api_spec.md mục 6).

Hai điểm vào công khai:

- :func:`sse_stream`: dùng cho ``stream=true``, phát bytes SSE kèm keep-alive.
- :func:`build_completion`: dùng cho ``stream=false``, gom toàn bộ event
  thành một ``ChatCompletionResponse`` duy nhất.

Cả hai dùng chung :class:`_AnswerState` + :func:`_content_delta` để đảm bảo
quy tắc nối "Nguồn"/cảnh báo/disclaimer giống hệt nhau giữa hai chế độ.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Final, Literal

from production_legal_qa_rag.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionMessageResponse,
    ChatCompletionResponse,
    UsageObject,
)
from production_legal_qa_rag.conversation.history import (
    DATA_SNAPSHOT_DISCLAIMER,
    SOURCES_FOOTER_MARKER,
)
from production_legal_qa_rag.generation.models import (
    Citation,
    CitationsEvent,
    DoneEvent,
    ErrorEvent,
    GenerationEvent,
    RefusalEvent,
    StatusEvent,
    TokenEvent,
    Usage,
    WarningEvent,
)

_KEEPALIVE_COMMENT: Final[bytes] = b": keep-alive\n\n"
_DONE_LINE: Final[bytes] = b"data: [DONE]\n\n"

# Chỉ 3 message mẫu nêu ở mục 6 cho 5 giá trị stage của StatusEvent; các stage
# hậu kỳ (drafting/verification/repairing) đều thuộc bước "soạn câu trả lời"
# dưới góc nhìn người dùng cuối (quyết định của developer, spec chưa liệt kê
# đủ 5 -> 3 message).
_STAGE_MESSAGES: Final[dict[str, str]] = {
    "guardrail": "Đang kiểm tra câu hỏi…",
    "retrieval": "Đang tra cứu văn bản pháp luật…",
    "drafting": "Đang soạn câu trả lời…",
    "verification": "Đang soạn câu trả lời…",
    "repairing": "Đang soạn câu trả lời…",
}


@dataclass
class _AnswerState:
    """Trạng thái tích luỹ khi duyệt qua một luồng ``GenerationEvent``."""

    has_content: bool = False
    citations_seen: bool = False
    finish_reason: Literal["stop", "length"] = "stop"


@dataclass
class _ChunkContext:
    """Giá trị cố định của một luồng SSE, dùng lại cho mọi chunk."""

    completion_id: str
    created: int
    model: str
    state: _AnswerState = field(default_factory=_AnswerState)
    role_sent: bool = False


def _new_chunk_context(model: str) -> _ChunkContext:
    return _ChunkContext(
        completion_id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model,
    )


def _sources_block(citations: list[Citation]) -> str:
    """Dựng khối "Nguồn" theo đúng định dạng mục 6."""
    lines = "\n".join(
        f"[{citation.n}] {citation.breadcrumb} — {citation.source_document}"
        for citation in citations
    )
    return f"{SOURCES_FOOTER_MARKER}{lines}"


def _status_message(stage: str) -> str:
    """Trả message ``reasoning_content`` cho một stage; fallback an toàn."""
    return _STAGE_MESSAGES.get(stage, "Đang xử lý câu hỏi…")


def _content_delta(event: GenerationEvent, state: _AnswerState) -> str | None:
    """Trả phần text cần nối thêm cho một event, cập nhật ``state`` kèm theo.

    Args:
        event: Event hiện tại của luồng generation.
        state: Trạng thái tích luỹ, được cập nhật in-place.

    Returns:
        Text cần nối vào câu trả lời, hoặc ``None`` nếu event này không sinh
        nội dung câu trả lời (``status``/``done``).
    """
    match event:
        case TokenEvent(text=text):
            state.has_content = True
            return text
        case RefusalEvent(message=message):
            state.has_content = True
            return message
        case CitationsEvent(citations=citations):
            state.has_content = True
            state.citations_seen = True
            return _sources_block(citations)
        case WarningEvent(code=code, message=message):
            state.has_content = True
            if code == "truncated":
                state.finish_reason = "length"
            return f"\n⚠️ {message}"
        case ErrorEvent(code=code, message=message, retry_after_seconds=retry_after):
            prefix = "\n⚠️ " if state.has_content else ""
            text = f"{prefix}{message}"
            if code == "rate_limited" and retry_after is not None:
                text += f" Thử lại sau {int(retry_after)} giây."
            state.has_content = True
            return text
        case _:
            return None


def _disclaimer_tail(state: _AnswerState) -> str | None:
    """Disclaimer chỉ áp dụng cho luồng có khối "Nguồn" thật sự (đã trả lời)."""
    return DATA_SNAPSHOT_DISCLAIMER if state.citations_seen else None


def _usage_object(usage: Usage) -> UsageObject:
    prompt_tokens = usage.prompt_tokens or 0
    completion_tokens = usage.completion_tokens or 0
    return UsageObject(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


# --- stream=true --------------------------------------------------------


def _delta_chunk(
    ctx: _ChunkContext,
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
) -> ChatCompletionChunk:
    delta = ChatCompletionChunkDelta(
        role="assistant" if not ctx.role_sent else None,
        content=content,
        reasoning_content=reasoning_content,
    )
    ctx.role_sent = True
    return ChatCompletionChunk(
        id=ctx.completion_id,
        created=ctx.created,
        model=ctx.model,
        choices=[ChatCompletionChunkChoice(delta=delta, finish_reason=None)],
    )


def _finish_chunk(ctx: _ChunkContext) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=ctx.completion_id,
        created=ctx.created,
        model=ctx.model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(), finish_reason=ctx.state.finish_reason
            )
        ],
    )


def _usage_chunk(ctx: _ChunkContext, usage: Usage) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=ctx.completion_id,
        created=ctx.created,
        model=ctx.model,
        choices=[],
        usage=_usage_object(usage),
    )


def _event_to_chunks(
    event: GenerationEvent, ctx: _ChunkContext, *, include_usage: bool
) -> list[ChatCompletionChunk]:
    """Chuyển một event thành 0..2 chunk OpenAI (mục 6)."""
    if isinstance(event, StatusEvent):
        return [_delta_chunk(ctx, reasoning_content=_status_message(event.stage))]
    if isinstance(event, DoneEvent):
        chunks: list[ChatCompletionChunk] = []
        tail = _disclaimer_tail(ctx.state)
        if tail is not None:
            chunks.append(_delta_chunk(ctx, content=tail))
        chunks.append(_finish_chunk(ctx))
        if include_usage and event.usage is not None:
            chunks.append(_usage_chunk(ctx, event.usage))
        return chunks
    delta_text = _content_delta(event, ctx.state)
    if delta_text is None:
        return []
    return [_delta_chunk(ctx, content=delta_text)]


def _encode_chunk(chunk: ChatCompletionChunk) -> bytes:
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode()


async def sse_stream(
    events: AsyncIterator[GenerationEvent],
    *,
    model: str,
    include_usage: bool,
    keepalive_seconds: float,
) -> AsyncIterator[bytes]:
    """Chuyển luồng ``GenerationEvent`` thành SSE OpenAI, có keep-alive.

    Dùng ``asyncio.wait`` (không phải ``wait_for``) để chờ event kế tiếp: nếu
    dùng ``wait_for``, hết giờ sẽ **hủy** coroutine ``__anext__()`` đang chạy —
    tương đương hủy luôn generator gốc giữa chừng (mất mọi event còn lại, kể
    cả token/citations/done thật sự sắp tới). Task chờ event được giữ nguyên
    qua nhiều vòng keep-alive, chỉ tạo task mới sau khi đã nhận được event.

    Args:
        events: Luồng event của một lượt chat (từ ``routes.stream_chat_turn``).
        model: Id model trả về trong mỗi chunk (luôn ``MODEL_ID``).
        include_usage: Có phát chunk ``usage`` cuối cùng hay không.
        keepalive_seconds: Ngưỡng chờ trước khi gửi dòng comment keep-alive.

    Yields:
        Bytes theo đúng khung ``text/event-stream``, kết thúc bằng
        ``data: [DONE]\\n\\n``.
    """
    ctx = _new_chunk_context(model)
    iterator = events.__aiter__()
    pending: asyncio.Task[GenerationEvent] = asyncio.ensure_future(iterator.__anext__())
    try:
        while True:
            done, _pending_set = await asyncio.wait(
                {pending}, timeout=keepalive_seconds
            )
            if not done:
                yield _KEEPALIVE_COMMENT
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                break
            pending = asyncio.ensure_future(iterator.__anext__())
            for chunk in _event_to_chunks(event, ctx, include_usage=include_usage):
                yield _encode_chunk(chunk)
    finally:
        pending.cancel()
    yield _DONE_LINE


# --- stream=false --------------------------------------------------------


async def build_completion(
    events: AsyncIterator[GenerationEvent],
    *,
    model: str,
    include_usage: bool,
) -> ChatCompletionResponse:
    """Gom toàn bộ luồng event thành một ``chat.completion`` (mục 2).

    Args:
        events: Luồng event của một lượt chat.
        model: Id model trả về (luôn ``MODEL_ID``).
        include_usage: Có gắn ``usage`` vào response hay không.

    Returns:
        Response OpenAI không-stream tương đương nội dung đã stream.
    """
    state = _AnswerState()
    parts: list[str] = []
    usage: Usage | None = None
    async for event in events:
        if isinstance(event, DoneEvent):
            tail = _disclaimer_tail(state)
            if tail is not None:
                parts.append(tail)
            usage = event.usage
            continue
        delta_text = _content_delta(event, state)
        if delta_text is not None:
            parts.append(delta_text)
    usage_object = _usage_object(usage) if include_usage and usage is not None else None
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessageResponse(content="".join(parts)),
                finish_reason=state.finish_reason,
            )
        ],
        usage=usage_object,
    )
