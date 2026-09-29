"""Cập nhật cuối lượt lên root trace Langfuse (observability_spec.md mục 4.5).

Langfuse là nơi duy nhất lưu nhật ký từng lượt hỏi-đáp (thay bảng `chat_turns`
của `chatlog/` đã gỡ): `update_turn_trace` gắn output, metadata và tags từ
`TurnTrace` đã điền đủ lên root span `chat_turn`. Tách khỏi `tracing.py` để
`tracing.py` (được `generation/`, `retrieval/` import) không phụ thuộc
`conversation/`.
"""

from __future__ import annotations

import logging
from typing import Any

from langfuse import propagate_attributes
from pydantic import BaseModel, Field

from production_legal_qa_rag.conversation.models import TurnTrace

_logger = logging.getLogger(__name__)


class RuntimeVersions(BaseModel):
    """Phiên bản prompt/corpus/model của API runtime, gắn vào mỗi trace lượt hỏi.

    Tạo một lần ở API lifespan, cùng thời điểm các dependency xử lý câu hỏi
    được khởi tạo (mục 4.5). Không cho phép giá trị rỗng để mỗi trace luôn gắn
    được với prompt, corpus và model đã thực sự phục vụ nó — dùng để so sánh
    trước/sau khi đổi cấu hình.
    """

    prompt_version: str = Field(min_length=1)
    corpus_version: str = Field(min_length=1)
    model_name: str = Field(min_length=1)


def update_turn_trace(
    root_span: Any,
    trace: TurnTrace,
    *,
    request_id: str,
    versions: RuntimeVersions,
) -> None:
    """Điền output/metadata/tags cuối lượt lên root span `chat_turn` (mục 4.5).

    Gọi khi `root_span` còn là span đang active (trước khi thoát `span`), dùng
    `trace` đã điền đủ. Lỗi ở đây chỉ log warning, không lan ra ngoài — nhật ký
    không bao giờ được làm hỏng câu trả lời (mục 1). Không đưa nội dung câu
    hỏi/trả lời vào log ứng dụng.

    Args:
        root_span: Observation trả về từ `span("chat_turn", ...)`.
        trace: Vết lượt hỏi đã hoàn tất (hoặc bị ngắt giữa chừng).
        request_id: Id request (đã có ở metadata lúc mở span, lặp để chắc chắn).
        versions: Phiên bản prompt/corpus/model của runtime.
    """
    try:
        root_span.update(
            output=trace.answer_text,
            metadata=_turn_metadata(trace, request_id, versions),
        )
        with propagate_attributes(tags=_turn_tags(trace)):
            pass
    except Exception as exc:  # noqa: BLE001 - trace update must never affect the response.
        # Chỉ log tên lỗi, không kèm message/traceback: message có thể chứa nội dung
        # câu hỏi/câu trả lời (riêng tư).
        _logger.warning(
            "Không thể cập nhật trace Langfuse cho lượt hỏi (request_id=%s, error=%s)",
            request_id,
            type(exc).__name__,
        )


def _turn_metadata(
    trace: TurnTrace, request_id: str, versions: RuntimeVersions
) -> dict[str, Any]:
    """Metadata root của trace: mọi trường chatlog cũ ngoài input/output/user/session."""
    return {
        "request_id": request_id,
        "outcome": trace.outcome,
        "error_code": trace.error_code,
        "cache_status": trace.cache_status,
        "verdict": trace.verdict.verdict if trace.verdict is not None else "allow",
        "standalone_query": trace.standalone_query,
        "chunk_ids": trace.chunk_ids,
        "citations": [citation.model_dump() for citation in trace.citations],
        "warnings": [warning.model_dump() for warning in trace.warnings],
        "usage": trace.usage.model_dump() if trace.usage is not None else None,
        "time_to_first_token_ms": trace.time_to_first_token_ms,
        "latency_ms": trace.latency_ms,
        "prompt_version": versions.prompt_version,
        "corpus_version": versions.corpus_version,
        "model_name": versions.model_name,
    }


def _turn_tags(trace: TurnTrace) -> list[str]:
    """Tag lọc nhanh trên UI Langfuse: `outcome:<x>`, `cache:<y>`."""
    return [f"outcome:{trace.outcome}", f"cache:{trace.cache_status}"]
