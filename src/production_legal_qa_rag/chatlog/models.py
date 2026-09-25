"""Model dữ liệu cho chatlog — ánh xạ TurnTrace + RequestContext thành TurnRecord.

TurnRecord là đơn vị ghi một lượt hỏi–đáp vào bảng ``chat_turns`` (Postgres).
Module này không phụ thuộc vào SQLAlchemy để có thể dùng độc lập khi test.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from production_legal_qa_rag.conversation.models import RequestContext, TurnTrace


class TurnRecord(BaseModel):
    """Một hàng dữ liệu ghi vào bảng ``chat_turns``.

    Dựng từ :func:`from_trace`. Không ném ngoại lệ — lớp gọi phải đảm bảo
    giá trị hợp lệ trước khi truyền vào.

    Args:
        id: UUID sinh ở tầng app, là PK của bảng.
        created_at: Thời điểm ghi (UTC); sinh tự động nếu không truyền.
        request_id: Tương ứng header/log ứng dụng.
        user_id: Id nội bộ OpenWebUI, không phải email/tên.
        chat_id: Id cuộc hội thoại của OpenWebUI, nullable.
        raw_query: Câu người dùng vừa gửi.
        standalone_query: Câu sau condense; ``None`` nếu bị chặn trước đó.
        outcome: Kết quả của lượt hỏi đáp.
        verdict: Nhãn guardrail.
        error_code: Mã lỗi nội bộ, nullable.
        cache_status: Trạng thái cache.
        chunk_ids: Danh sách chunk_id theo thứ tự context.
        answer_text: Văn bản trả lời; rỗng nếu refused/error.
        citations: Danh sách citation dạng JSON.
        warnings: Danh sách warning dạng JSON.
        usage: Thông tin token; nullable.
        time_to_first_token_ms: Thời gian tới token đầu tiên (ms); nullable.
        latency_ms: Tổng độ trễ (ms).
        prompt_version: Phiên bản prompt đang dùng.
        corpus_version: Phiên bản corpus đang dùng.
        model_name: Tên model LLM.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    user_id: str
    chat_id: str | None = None
    raw_query: str
    standalone_query: str | None = None
    outcome: str  # answered | refused | error | client_disconnected
    verdict: str  # allow | out_of_scope | injection
    error_code: str | None = None
    cache_status: str  # answer_hit | retrieval_hit | miss | bypass
    chunk_ids: list[str] = Field(default_factory=list)
    answer_text: str = ""
    citations: list[dict] = Field(default_factory=list)
    warnings: list[dict] = Field(default_factory=list)
    usage: dict | None = None
    time_to_first_token_ms: int | None = None
    latency_ms: int = 0
    prompt_version: str = ""
    corpus_version: str = ""
    model_name: str = ""


def from_trace(
    trace: TurnTrace,
    ctx: RequestContext,
    *,
    prompt_version: str = "",
    corpus_version: str = "",
    model_name: str = "",
) -> TurnRecord:
    """Dựng ``TurnRecord`` từ ``TurnTrace`` và ``RequestContext``.

    Args:
        trace: Vết lượt hỏi–đáp do orchestrator điền.
        ctx: Danh tính request do lớp API cung cấp.
        prompt_version: Phiên bản prompt đang dùng.
        corpus_version: Phiên bản corpus đang dùng.
        model_name: Tên model LLM.

    Returns:
        TurnRecord sẵn sàng ghi vào bảng ``chat_turns``.
    """
    verdict_str = trace.verdict.verdict if trace.verdict is not None else "allow"
    return TurnRecord(
        request_id=ctx.request_id,
        user_id=ctx.user_id,
        chat_id=ctx.chat_id,
        raw_query=trace.raw_query,
        standalone_query=trace.standalone_query,
        outcome=trace.outcome,
        verdict=verdict_str,
        error_code=trace.error_code,
        cache_status=trace.cache_status,
        chunk_ids=trace.chunk_ids,
        answer_text=trace.answer_text,
        citations=[c.model_dump() for c in trace.citations],
        warnings=[w.model_dump() for w in trace.warnings],
        usage=trace.usage.model_dump() if trace.usage is not None else None,
        time_to_first_token_ms=trace.time_to_first_token_ms,
        latency_ms=trace.latency_ms,
        prompt_version=prompt_version,
        corpus_version=corpus_version,
        model_name=model_name,
    )
