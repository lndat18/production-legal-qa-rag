"""Điều phối guardrail, retrieval, generation stream và hậu kiểm đầu ra."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Final

from production_legal_qa_rag.generation.generator import AnswerGenerator
from production_legal_qa_rag.generation.guardrail import (
    INJECTION_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    InputGuardrail,
)
from production_legal_qa_rag.generation.models import (
    CitationsEvent,
    DoneEvent,
    ErrorEvent,
    GenerationEvent,
    GuardrailVerdict,
    RefusalEvent,
    StatusEvent,
    TokenEvent,
    WarningEvent,
)
from production_legal_qa_rag.generation.output_check import check_output
from production_legal_qa_rag.retrieval.models import RetrievalError, RetrievedChunk
from production_legal_qa_rag.retrieval.pipeline import retrieve as default_retrieve

_RATE_LIMIT_MESSAGE: Final = (
    "Dịch vụ tạo câu trả lời đang quá tải. Vui lòng thử lại sau."
)
_LLM_ERROR_MESSAGE: Final = "Không thể tạo câu trả lời lúc này. Vui lòng thử lại sau."
_RETRIEVAL_ERROR_MESSAGE: Final = (
    "Không thể tra cứu văn bản lúc này. Vui lòng thử lại sau."
)
_NO_CONTEXT_MESSAGE: Final = "Không tìm thấy văn bản phù hợp để trả lời câu hỏi này."
_EMPTY_CONTENT_MESSAGE: Final = "Mô hình không tạo được nội dung câu trả lời."

RetrieveCallable = Callable[[str], Awaitable[list[RetrievedChunk]]]


class GenerationPipeline:
    """Giữ các dependency dùng lại và phát event cho một câu hỏi stateless."""

    def __init__(
        self,
        *,
        guardrail: InputGuardrail | None = None,
        generator: AnswerGenerator | None = None,
        retrieve: RetrieveCallable = default_retrieve,
    ) -> None:
        self._guardrail = guardrail or InputGuardrail()
        self._generator = generator or AnswerGenerator()
        self._retrieve = retrieve

    async def answer_stream(self, query: str) -> AsyncIterator[GenerationEvent]:
        """Chạy một lượt hỏi đáp và luôn đóng luồng bằng event ``done``.

        Args:
            query: Câu hỏi tiếng Việt độc lập, không giữ state từ lượt trước.

        Yields:
            Event trạng thái, token/refusal/error và cuối cùng là ``done``.
        """
        yield StatusEvent(stage="guardrail")
        verdict = await self._guardrail.check_input(query)
        if verdict.verdict != "allow":
            yield _refusal_event(verdict)
            yield DoneEvent()
            return

        yield StatusEvent(stage="retrieval")
        try:
            chunks = await self._retrieve(query)
        except RetrievalError:
            yield ErrorEvent(code="retrieval_error", message=_RETRIEVAL_ERROR_MESSAGE)
            yield DoneEvent()
            return
        except Exception:  # noqa: BLE001 - external retrieval errors must not escape SSE.
            yield ErrorEvent(code="retrieval_error", message=_RETRIEVAL_ERROR_MESSAGE)
            yield DoneEvent()
            return

        if not chunks:
            yield ErrorEvent(code="no_context", message=_NO_CONTEXT_MESSAGE)
            yield DoneEvent()
            return

        yield StatusEvent(stage="generation")
        text_parts: list[str] = []
        finish_reason: str | None = None
        usage = None
        try:
            async for delta in self._generator.stream(query, chunks):
                if delta.text:
                    text_parts.append(delta.text)
                    yield TokenEvent(text=delta.text)
                finish_reason = delta.finish_reason or finish_reason
                usage = delta.usage or usage
        except Exception as error:  # noqa: BLE001 - external Groq errors must end the stream.
            is_rate_limited = _is_rate_limited(error)
            yield ErrorEvent(
                code="rate_limited" if is_rate_limited else "llm_error",
                message=_RATE_LIMIT_MESSAGE if is_rate_limited else _LLM_ERROR_MESSAGE,
                retry_after_seconds=_retry_after_seconds(error)
                if is_rate_limited
                else None,
            )
            yield DoneEvent(usage=usage)
            return

        text = "".join(text_parts)
        if not text:
            yield ErrorEvent(code="llm_error", message=_EMPTY_CONTENT_MESSAGE)
            yield DoneEvent(usage=usage)
            return

        result = check_output(text, chunks)
        yield CitationsEvent(citations=result.citations)
        if finish_reason == "length":
            yield WarningEvent(
                code="truncated",
                message="Câu trả lời có thể đã bị cắt do đạt giới hạn token.",
            )
        for warning in result.warnings:
            yield WarningEvent(
                code=warning.code,
                message=warning.message,
                detail=warning.detail,
            )
        yield DoneEvent(usage=usage)


def _refusal_event(verdict: GuardrailVerdict) -> RefusalEvent:
    """Chuyển verdict bị chặn thành message cố định, không do LLM sinh."""
    if verdict.verdict == "out_of_scope":
        return RefusalEvent(reason="out_of_scope", message=OUT_OF_SCOPE_MESSAGE)
    return RefusalEvent(reason="injection", message=INJECTION_MESSAGE)


def _is_rate_limited(error: Exception) -> bool:
    """Nhận diện lỗi 429 từ Groq hoặc fake client dùng trong kiểm thử."""
    return (
        getattr(error, "status_code", None) == 429
        or getattr(getattr(error, "response", None), "status_code", None) == 429
    )


def _retry_after_seconds(error: Exception) -> float | None:
    """Đọc header retry-after khi server cung cấp giá trị giây hợp lệ."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or getattr(error, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except TypeError, ValueError:
        return None


_default_pipeline: GenerationPipeline | None = None


async def answer_stream(query: str) -> AsyncIterator[GenerationEvent]:
    """Phát event generation qua pipeline mặc định dùng lại giữa các request.

    Args:
        query: Câu hỏi tiếng Việt độc lập.

    Yields:
        Event theo contract của package ``generation``.
    """
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = GenerationPipeline()
    async for event in _default_pipeline.answer_stream(query):
        yield event
