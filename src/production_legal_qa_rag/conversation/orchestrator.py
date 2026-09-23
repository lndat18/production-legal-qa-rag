"""Điều phối một lượt chat nhiều lượt (conversation_spec.md mục 7).

Luồng: guardrail ‖ condense -> answer cache -> single-flight -> admission ->
retrieval (có cache) -> generation. Generator chỉ thấy câu độc lập nên câu
trả lời cache được đúng; history chỉ vào condense và guardrail.

`cache/` mới có model (`cache/models.py`); các thành phần còn lại được inject
qua các Protocol bên dưới, bám interface trong `cache_spec.md`. Thiếu thành
phần nào thì bước đó bị bỏ qua (coi như miss / không khoá).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Final, Protocol

from production_legal_qa_rag.cache.models import CachedAnswer
from production_legal_qa_rag.conversation.admission import (
    AdmissionController,
    AdmissionDenied,
)
from production_legal_qa_rag.conversation.condenser import QueryCondenser
from production_legal_qa_rag.conversation.history import HistoryWindow, build_window
from production_legal_qa_rag.conversation.models import (
    ChatMessage,
    RequestContext,
    TurnTrace,
)
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
from production_legal_qa_rag.generation.pipeline import GenerationPipeline
from production_legal_qa_rag.retrieval.models import RetrievedChunk
from production_legal_qa_rag.retrieval.pipeline import retrieve as default_retrieve
from production_legal_qa_rag.retrieval.relevance import (
    MIN_RERANK_SCORE,
    is_low_relevance,
)

logger = logging.getLogger(__name__)

_NOT_FOUND_PHRASE: Final = "không tìm thấy quy định phù hợp"
_INTERNAL_ERROR_MESSAGE: Final = "Đã xảy ra lỗi. Vui lòng thử lại sau."
_OVERLOADED_MESSAGE: Final = "Hệ thống đang quá tải. Vui lòng thử lại sau."
_RETRIEVAL_ERROR_MESSAGE: Final = (
    "Không thể tra cứu văn bản lúc này. Vui lòng thử lại sau."
)
_NO_CONTEXT_MESSAGE: Final = "Không tìm thấy văn bản phù hợp để trả lời câu hỏi này."

type RetrieveCallable = Callable[[str], Awaitable[list[RetrievedChunk]]]
type ReplayCallable = Callable[[CachedAnswer], AsyncIterator[GenerationEvent]]


class AnswerCachePort(Protocol):
    """Interface `AnswerCache` của cache_spec.md mục 2 (không ném lỗi Redis)."""

    async def get(self, standalone_query: str) -> CachedAnswer | None: ...

    async def set(self, standalone_query: str, answer: CachedAnswer) -> None: ...


class RetrievalCachePort(Protocol):
    """Interface `RetrievalCache` của cache_spec.md mục 2."""

    async def get(self, standalone_query: str) -> list[RetrievedChunk] | None: ...

    async def set(
        self, standalone_query: str, chunks: list[RetrievedChunk]
    ) -> None: ...


class FlightPort(Protocol):
    """Kết quả ``SingleFlight.acquire``: mình là leader hay follower."""

    is_leader: bool

    async def wait_for_answer(self) -> CachedAnswer | None:
        """Follower poll answer cache; ``None`` khi hết hạn/leader không ghi."""
        ...


class SingleFlightPort(Protocol):
    """Interface `SingleFlight` (cache_spec.md mục 6), khoá suy ra từ câu hỏi."""

    def acquire(
        self, standalone_query: str, request_id: str
    ) -> AbstractAsyncContextManager[FlightPort]: ...


class ChatOrchestrator:
    """Điểm vào duy nhất của lớp API cho luồng chat của người dùng cuối."""

    def __init__(
        self,
        *,
        guardrail: InputGuardrail | None = None,
        condenser: QueryCondenser | None = None,
        generation: GenerationPipeline | None = None,
        admission: AdmissionController | None = None,
        retrieve: RetrieveCallable = default_retrieve,
        answer_cache: AnswerCachePort | None = None,
        retrieval_cache: RetrievalCachePort | None = None,
        single_flight: SingleFlightPort | None = None,
        replay: ReplayCallable | None = None,
    ) -> None:
        if answer_cache is not None and replay is None:
            raise ValueError("answer_cache cần đi kèm replay để phát lại câu trả lời.")
        self._guardrail = guardrail or InputGuardrail()
        self._condenser = condenser or QueryCondenser()
        self._generation = generation or GenerationPipeline(
            guardrail=self._guardrail, retrieve=retrieve
        )
        self._admission = admission or AdmissionController()
        self._retrieve = retrieve
        self._answer_cache = answer_cache
        self._retrieval_cache = retrieval_cache
        self._single_flight = single_flight
        self._replay = replay

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        ctx: RequestContext,
        trace: TurnTrace,
    ) -> AsyncIterator[GenerationEvent]:
        """Chạy một lượt chat, điền ``trace`` và luôn kết thúc bằng ``done``.

        Args:
            messages: Toàn bộ ``messages[]`` client gửi lên.
            ctx: Danh tính request do lớp API điền.
            trace: Đối tượng mutable để lớp API đọc sau khi stream xong.

        Yields:
            Event theo union ``GenerationEvent`` của ``generation/``.

        Raises:
            InvalidConversationError: ``messages`` không hợp lệ, ném ở lần
                duyệt đầu tiên, trước mọi event (API trả 422).
        """
        window = build_window(messages)
        trace.raw_query = window.query
        started = time.perf_counter()
        finished = False
        try:
            async for event in self._run(window, ctx, trace):
                _record_event(event, trace, started)
                finished = finished or isinstance(event, DoneEvent)
                yield event
        except Exception:
            logger.exception("Lỗi không lường trước trong luồng chat.")
            if not finished:
                for event in _internal_error_events():
                    _record_event(event, trace, started)
                    yield event
        finally:
            trace.latency_ms = _elapsed_ms(started)

    async def _run(
        self, window: HistoryWindow, ctx: RequestContext, trace: TurnTrace
    ) -> AsyncIterator[GenerationEvent]:
        yield StatusEvent(stage="guardrail")
        verdict, standalone = await self._guard_and_condense(window)
        trace.verdict = verdict
        trace.standalone_query = standalone
        if verdict.verdict != "allow":
            trace.outcome = "refused"
            yield _refusal_event(verdict)
            yield DoneEvent()
            return
        async for event in self._answer(standalone, ctx, trace):
            yield event

    async def _guard_and_condense(
        self, window: HistoryWindow
    ) -> tuple[GuardrailVerdict, str]:
        """Guardrail đọc câu gốc song song với condense (không đọc câu đã condense)."""
        guard = self._guardrail.check_input(window.query, window.recent_user_turns)
        if not window.has_history:
            return await guard, window.query
        verdict, standalone = await asyncio.gather(
            guard, self._condenser.condense(window.query, window.history)
        )
        return verdict, standalone

    async def _answer(
        self, standalone: str, ctx: RequestContext, trace: TurnTrace
    ) -> AsyncIterator[GenerationEvent]:
        hit = await self._cached_answer(standalone)
        if hit is not None:
            trace.cache_status = "answer_hit"
            async for event in self._replay_hit(hit):
                yield event
            return
        if self._single_flight is None:
            async for event in self._produce(standalone, ctx, trace):
                yield event
            return
        async with self._single_flight.acquire(standalone, ctx.request_id) as flight:
            if not flight.is_leader:
                hit = await flight.wait_for_answer()
                if hit is not None:
                    trace.cache_status = "answer_hit"
                    async for event in self._replay_hit(hit):
                        yield event
                    return
                # Leader lỗi/ngắt: tự chạy thay vì chờ vô hạn (cache_spec mục 6).
            async for event in self._produce(standalone, ctx, trace):
                yield event

    async def _cached_answer(self, standalone: str) -> CachedAnswer | None:
        if self._answer_cache is None:
            return None
        return await self._answer_cache.get(standalone)

    def _replay_hit(self, hit: CachedAnswer) -> AsyncIterator[GenerationEvent]:
        if self._replay is None:
            raise RuntimeError("Thiếu replay cho answer cache.")
        return self._replay(hit)

    async def _produce(
        self, standalone: str, ctx: RequestContext, trace: TurnTrace
    ) -> AsyncIterator[GenerationEvent]:
        """Cache miss: xin slot admission rồi retrieval + generation."""
        try:
            async with self._admission.slot(ctx.user_id):
                async for event in self._retrieve_and_generate(standalone, trace):
                    yield event
        except AdmissionDenied as denied:
            yield _denial_event(denied)
            yield DoneEvent()

    async def _retrieve_and_generate(
        self, standalone: str, trace: TurnTrace
    ) -> AsyncIterator[GenerationEvent]:
        yield StatusEvent(stage="retrieval")
        chunks, failure = await self._load_chunks(standalone, trace)
        if failure is not None:
            yield failure
            yield DoneEvent()
            return
        trace.chunk_ids = [chunk.chunk_id for chunk in chunks]
        async for event in self._generation.generate(standalone, chunks):
            if isinstance(event, DoneEvent):
                await self._store_answer(standalone, trace)
            yield event

    async def _load_chunks(
        self, standalone: str, trace: TurnTrace
    ) -> tuple[list[RetrievedChunk], ErrorEvent | None]:
        cached: list[RetrievedChunk] | None = None
        if self._retrieval_cache is not None:
            cached = await self._retrieval_cache.get(standalone)
        if cached:
            trace.cache_status = "retrieval_hit"
            chunks = cached
        else:
            try:
                chunks = await self._retrieve(standalone)
            except Exception:
                logger.warning("Retrieval lỗi.", exc_info=True)
                return [], ErrorEvent(
                    code="retrieval_error", message=_RETRIEVAL_ERROR_MESSAGE
                )
        # 18.2.2: chặn sớm khi 5 chunk quá ít liên quan (gate của conversation/,
        # retrieve() không lọc gì — retrieval_spec.md mục 16 điểm 8).
        if not chunks:
            return [], ErrorEvent(code="no_context", message=_NO_CONTEXT_MESSAGE)
        if is_low_relevance(chunks):
            # Quan sát (reviewer PR #41): log để theo dõi gate có chặn sát ngưỡng
            # hay không trong vận hành thật (không log nội dung câu hỏi, mục 12).
            scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
            logger.info(
                "Gate độ liên quan chặn no_context: max_rerank_score=%s "
                "ngưỡng=%s n_chunks=%d.",
                max(scores) if scores else None,
                MIN_RERANK_SCORE,
                len(chunks),
            )
            return [], ErrorEvent(code="no_context", message=_NO_CONTEXT_MESSAGE)
        if not cached and self._retrieval_cache is not None:
            await self._retrieval_cache.set(standalone, chunks)
        return chunks, None

    async def _store_answer(self, standalone: str, trace: TurnTrace) -> None:
        """Ghi answer cache chỉ khi luồng sạch (cache_spec.md mục 5).

        Gọi ngay trước khi phát ``done`` để không phụ thuộc client còn đọc tiếp.
        """
        if self._answer_cache is None or trace.error_code or trace.warnings:
            return
        if not trace.citations and _NOT_FOUND_PHRASE not in trace.answer_text.lower():
            return
        await self._answer_cache.set(
            standalone,
            CachedAnswer(
                text=trace.answer_text,
                citations=trace.citations,
                created_at=datetime.now(UTC),
            ),
        )


def _record_event(event: GenerationEvent, trace: TurnTrace, started: float) -> None:
    """Điền ``trace`` từ event đi qua; không phụ thuộc DB."""
    match event:
        case TokenEvent():
            if trace.time_to_first_token_ms is None:
                trace.time_to_first_token_ms = _elapsed_ms(started)
            trace.answer_text += event.text
        case CitationsEvent():
            trace.citations = event.citations
        case WarningEvent():
            trace.warnings.append(event)
        case ErrorEvent():
            trace.error_code = event.code
        case DoneEvent():
            trace.usage = event.usage
            if trace.outcome != "refused":
                trace.outcome = "error" if trace.error_code else "answered"
        case _:
            pass


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _refusal_event(verdict: GuardrailVerdict) -> RefusalEvent:
    """Message từ chối cố định, không do LLM sinh."""
    if verdict.verdict == "out_of_scope":
        return RefusalEvent(reason="out_of_scope", message=OUT_OF_SCOPE_MESSAGE)
    return RefusalEvent(reason="injection", message=INJECTION_MESSAGE)


def _denial_event(denied: AdmissionDenied) -> ErrorEvent:
    return ErrorEvent(
        code="rate_limited",
        message=_OVERLOADED_MESSAGE,
        retry_after_seconds=denied.retry_after_seconds,
    )


def _internal_error_events() -> list[GenerationEvent]:
    return [ErrorEvent(code="llm_error", message=_INTERNAL_ERROR_MESSAGE), DoneEvent()]
