"""Điều phối buffered draft, hard gate, Evidence Judge và event đã duyệt."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Final

from production_legal_qa_rag.generation.generator import (
    AnswerGenerator,
    GeneratedAnswer,
)
from production_legal_qa_rag.generation.guardrail import (
    INJECTION_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    InputGuardrail,
)
from production_legal_qa_rag.generation.judge import EvidenceJudge
from production_legal_qa_rag.generation.models import (
    CitationsEvent,
    DoneEvent,
    ErrorEvent,
    GenerationEvent,
    GuardrailVerdict,
    JudgeIssue,
    JudgeVerdict,
    RefusalEvent,
    StatusEvent,
    TokenEvent,
    Usage,
    VerificationIssue,
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
_INSUFFICIENT_EVIDENCE_MESSAGE: Final = (
    "Tôi chưa có đủ căn cứ pháp lý trong các văn bản hiện có để trả lời chính xác."
)
_UNABLE_TO_VERIFY_MESSAGE: Final = (
    "Tôi không thể xác minh đầy đủ câu trả lời lúc này. Vui lòng thử lại sau."
)

type RetrieveCallable = Callable[[str], Awaitable[list[RetrievedChunk]]]


class GenerationPipeline:
    """Giữ dependency và phát câu trả lời chỉ sau evidence verification."""

    def __init__(
        self,
        *,
        guardrail: InputGuardrail | None = None,
        generator: AnswerGenerator | None = None,
        judge: EvidenceJudge | None = None,
        retrieve: RetrieveCallable = default_retrieve,
    ) -> None:
        self._guardrail = guardrail or InputGuardrail()
        self._generator = generator or AnswerGenerator()
        self._judge = judge or EvidenceJudge()
        self._retrieve = retrieve

    async def answer_stream(self, query: str) -> AsyncIterator[GenerationEvent]:
        """Chạy guardrail, retrieve đúng một lần rồi generation verified.

        Args:
            query: Câu hỏi tiếng Việt độc lập, không giữ state từ lượt trước.

        Yields:
            Event trạng thái, answer/refusal/error và đúng một event ``done``.
        """
        yield StatusEvent(stage="guardrail")
        verdict = await self._guardrail.check_input(query)
        if verdict.verdict != "allow":
            yield _guardrail_refusal_event(verdict)
            yield DoneEvent()
            return

        yield StatusEvent(stage="retrieval")
        try:
            chunks = await self._retrieve(query)
        except RetrievalError:
            yield ErrorEvent(code="retrieval_error", message=_RETRIEVAL_ERROR_MESSAGE)
            yield DoneEvent()
            return
        except Exception:  # noqa: BLE001 - external retrieval errors must not escape.
            yield ErrorEvent(code="retrieval_error", message=_RETRIEVAL_ERROR_MESSAGE)
            yield DoneEvent()
            return

        if not chunks:
            yield ErrorEvent(code="no_context", message=_NO_CONTEXT_MESSAGE)
            yield DoneEvent()
            return

        async for event in self.generate(query, chunks):
            yield event

    async def generate(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> AsyncIterator[GenerationEvent]:
        """Sinh và kiểm chứng answer từ context caller đã cố định.

        Không gọi guardrail hoặc retrieval. Một repair budget dùng chung cho hard
        gate và Judge, nên mọi draft sau repair vẫn dùng đúng ``chunks`` ban đầu.

        Args:
            query: Câu hỏi độc lập đã được caller điều phối.
            chunks: Context rerank theo đúng thứ tự citation.

        Yields:
            Chỉ token answer đã pass, hoặc refusal/error, rồi đúng một ``done``.
        """
        yield StatusEvent(stage="drafting")
        draft, error_event = await self._draft(query, chunks)
        total_usage = draft.usage if draft is not None else None
        if error_event is not None:
            yield error_event
            yield DoneEvent(usage=total_usage)
            return
        if draft is None or not draft.text:
            yield ErrorEvent(code="llm_error", message=_EMPTY_CONTENT_MESSAGE)
            yield DoneEvent(usage=total_usage)
            return

        repair_used = False
        while True:
            hard_gate = check_output(
                draft.text, chunks, finish_reason=draft.finish_reason
            )
            if hard_gate.hard_issues:
                if repair_used:
                    yield _unable_to_verify_event()
                    yield DoneEvent(usage=total_usage)
                    return
                repair_used = True
                yield StatusEvent(stage="repairing")
                yield StatusEvent(stage="drafting")
                draft, error_event = await self._repair(
                    query, chunks, draft.text, hard_gate.hard_issues
                )
                total_usage = _merge_usage(
                    total_usage, draft.usage if draft is not None else None
                )
                if error_event is not None:
                    yield error_event
                    yield DoneEvent(usage=total_usage)
                    return
                if draft is None or not draft.text:
                    yield ErrorEvent(code="llm_error", message=_EMPTY_CONTENT_MESSAGE)
                    yield DoneEvent(usage=total_usage)
                    return
                continue

            yield StatusEvent(stage="verification")
            try:
                judge_verdict = await self._judge.judge(
                    query, chunks, draft.text, hard_gate.citations
                )
                _validate_judge_verdict(judge_verdict, len(chunks))
            except Exception:  # noqa: BLE001 - every Judge failure must fail closed.
                yield _unable_to_verify_event()
                yield DoneEvent(usage=total_usage)
                return

            if judge_verdict.verdict == "insufficient_evidence":
                yield RefusalEvent(
                    reason="insufficient_evidence",
                    message=_INSUFFICIENT_EVIDENCE_MESSAGE,
                )
                yield DoneEvent(usage=total_usage)
                return
            if judge_verdict.verdict == "repair":
                if repair_used:
                    yield _unable_to_verify_event()
                    yield DoneEvent(usage=total_usage)
                    return
                repair_used = True
                yield StatusEvent(stage="repairing")
                yield StatusEvent(stage="drafting")
                draft, error_event = await self._repair(
                    query,
                    chunks,
                    draft.text,
                    [_to_verification_issue(issue) for issue in judge_verdict.issues],
                )
                total_usage = _merge_usage(
                    total_usage, draft.usage if draft is not None else None
                )
                if error_event is not None:
                    yield error_event
                    yield DoneEvent(usage=total_usage)
                    return
                if draft is None or not draft.text:
                    yield ErrorEvent(code="llm_error", message=_EMPTY_CONTENT_MESSAGE)
                    yield DoneEvent(usage=total_usage)
                    return
                continue

            for fragment in draft.fragments:
                yield TokenEvent(text=fragment)
            yield CitationsEvent(citations=hard_gate.citations)
            for warning in hard_gate.warnings:
                yield WarningEvent(
                    code=warning.code,
                    message=warning.message,
                    detail=warning.detail,
                )
            yield DoneEvent(usage=total_usage)
            return

    async def _draft(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> tuple[GeneratedAnswer | None, ErrorEvent | None]:
        """Buffer draft đầu tiên, không phát bất kỳ delta nào cho caller."""
        try:
            return await self._generator.draft(query, chunks), None
        except Exception as error:  # noqa: BLE001 - provider errors become public events.
            return None, _generator_error_event(error)

    async def _repair(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        draft: str,
        issues: list[VerificationIssue],
    ) -> tuple[GeneratedAnswer | None, ErrorEvent | None]:
        """Buffer toàn bộ repair với cùng context; không có đường retrieve lại."""
        try:
            return await self._generator.repair(query, chunks, draft, issues), None
        except Exception as error:  # noqa: BLE001 - injected/provider failure is safe error.
            return None, _generator_error_event(error)


def _validate_judge_verdict(verdict: JudgeVerdict, context_size: int) -> None:
    """Từ chối schema hợp lệ về mặt JSON nhưng mâu thuẫn policy verification."""
    if verdict.verdict == "pass" and verdict.issues:
        raise ValueError("Judge pass không được mang issue")
    if verdict.verdict == "repair" and not verdict.issues:
        raise ValueError("Judge repair phải nêu issue")
    if verdict.verdict == "insufficient_evidence" and not verdict.issues:
        raise ValueError("Judge insufficient_evidence phải nêu issue")
    for issue in verdict.issues:
        if any(
            number < 1 or number > context_size for number in issue.evidence_numbers
        ):
            raise ValueError("Judge tham chiếu evidence ngoài context")


def _to_verification_issue(issue: JudgeIssue) -> VerificationIssue:
    """Chuyển JudgeIssue đã validate sang input repair không có dữ liệu thừa."""
    return VerificationIssue.model_validate(issue.model_dump())


def _merge_usage(left: Usage | None, right: Usage | None) -> Usage | None:
    """Cộng usage draft và repair để done phản ánh toàn bộ request."""
    if left is None:
        return right
    if right is None:
        return left
    return Usage(
        prompt_tokens=_sum_optional(left.prompt_tokens, right.prompt_tokens),
        completion_tokens=_sum_optional(
            left.completion_tokens, right.completion_tokens
        ),
        reasoning_tokens=_sum_optional(left.reasoning_tokens, right.reasoning_tokens),
    )


def _sum_optional(left: int | None, right: int | None) -> int | None:
    """Cộng hai số token, chỉ None khi cả hai phía đều không có usage."""
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


def _guardrail_refusal_event(verdict: GuardrailVerdict) -> RefusalEvent:
    """Chuyển verdict guardrail bị chặn thành message cố định."""
    if verdict.verdict == "out_of_scope":
        return RefusalEvent(reason="out_of_scope", message=OUT_OF_SCOPE_MESSAGE)
    return RefusalEvent(reason="injection", message=INJECTION_MESSAGE)


def _unable_to_verify_event() -> RefusalEvent:
    """Trả message cố định khi evidence pipeline không thể phát answer an toàn."""
    return RefusalEvent(reason="unable_to_verify", message=_UNABLE_TO_VERIFY_MESSAGE)


def _generator_error_event(error: Exception) -> ErrorEvent:
    """Map lỗi generator/provider thành event công khai không lộ implementation."""
    is_rate_limited = _is_rate_limited(error)
    return ErrorEvent(
        code="rate_limited" if is_rate_limited else "llm_error",
        message=_RATE_LIMIT_MESSAGE if is_rate_limited else _LLM_ERROR_MESSAGE,
        retry_after_seconds=_retry_after_seconds(error) if is_rate_limited else None,
    )


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
        Event theo contract evidence-verified của package ``generation``.
    """
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = GenerationPipeline()
    async for event in _default_pipeline.answer_stream(query):
        yield event
