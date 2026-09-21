"""Model dữ liệu của `conversation/` (conversation_spec.md mục 2).

`EvaluationResult` thuộc phase RAGAS sau, chưa định nghĩa ở đây.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from production_legal_qa_rag.cache.models import CacheStatus
from production_legal_qa_rag.generation.models import (
    Citation,
    GuardrailVerdict,
    Usage,
    WarningEvent,
)


class ChatMessage(BaseModel):
    """Một message trong ``messages[]`` do client gửi (dữ liệu không tin cậy)."""

    role: Literal["user", "assistant"]
    content: str


class RequestContext(BaseModel):
    """Danh tính request do lớp API điền; orchestrator không biết HTTP."""

    user_id: str
    chat_id: str | None = None
    request_id: str


class TurnTrace(BaseModel):
    """Vết một lượt hỏi đáp, orchestrator điền dần để lớp API ghi `chatlog/`."""

    raw_query: str = ""
    standalone_query: str | None = None
    verdict: GuardrailVerdict | None = None
    cache_status: CacheStatus = "miss"
    # Mặc định "error": luồng bị huỷ giữa chừng không được ghi nhầm là đã trả lời.
    outcome: Literal["answered", "refused", "error"] = "error"
    error_code: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)
    answer_text: str = ""
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[WarningEvent] = Field(default_factory=list)
    usage: Usage | None = None
    time_to_first_token_ms: int | None = None
    latency_ms: int = 0
