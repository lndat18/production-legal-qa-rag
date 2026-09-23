"""Các contract Pydantic cho generation, verification và event stream."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class GuardrailVerdict(BaseModel):
    """Kết quả phân loại an toàn của câu hỏi đầu vào."""

    verdict: Literal["allow", "out_of_scope", "injection"]
    reason: str


class Citation(BaseModel):
    """Nguồn tương ứng với một ký hiệu trích dẫn ``[n]`` trong câu trả lời."""

    n: int
    chunk_id: str
    source_document: str
    breadcrumb: str


class Usage(BaseModel):
    """Thông tin token Groq trả về sau khi hoàn tất stream, nếu có."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None


class VerificationIssue(BaseModel):
    """Một lỗi có thể gửi cho generator để viết lại toàn bộ câu trả lời."""

    code: Literal[
        "truncated",
        "invalid_citation",
        "unverified_sensitive_number",
        "unsupported_claim",
        "citation_mismatch",
        "missing_material_condition",
        "context_insufficient",
    ]
    claim: str = ""
    detail: str
    evidence_numbers: list[int] = Field(default_factory=list)


class OutputWarning(BaseModel):
    """Tín hiệu mềm do code hard gate phát hiện nhưng không cần chặn answer."""

    code: Literal["unverified_number"]
    message: str
    detail: str = ""


class HardGateResult(BaseModel):
    """Kết quả deterministic gate trước khi gửi draft cho Evidence Judge."""

    citations: list[Citation] = Field(default_factory=list)
    hard_issues: list[VerificationIssue] = Field(default_factory=list)
    warnings: list[OutputWarning] = Field(default_factory=list)


class JudgeIssue(BaseModel):
    """Một nhận định evidence-level của Judge, không chứa chain-of-thought."""

    code: Literal[
        "unsupported_claim",
        "citation_mismatch",
        "missing_material_condition",
        "context_insufficient",
    ]
    claim: str
    detail: str
    evidence_numbers: list[int] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    """Phán quyết có cấu trúc của Evidence Judge."""

    verdict: Literal["pass", "repair", "insufficient_evidence"]
    issues: list[JudgeIssue] = Field(default_factory=list)


class StatusEvent(BaseModel):
    """Báo hiệu bắt đầu một bước xử lý để giao diện cập nhật trạng thái."""

    type: Literal["status"] = "status"
    stage: Literal[
        "guardrail",
        "retrieval",
        "drafting",
        "verification",
        "repairing",
    ]


class TokenEvent(BaseModel):
    """Một mẩu nội dung câu trả lời được sinh theo stream."""

    type: Literal["token"] = "token"
    text: str


class RefusalEvent(BaseModel):
    """Từ chối câu hỏi bị guardrail chặn."""

    type: Literal["refusal"] = "refusal"
    reason: Literal[
        "out_of_scope",
        "injection",
        "insufficient_evidence",
        "unable_to_verify",
    ]
    message: str


class CitationsEvent(BaseModel):
    """Tập nguồn hợp lệ thực sự xuất hiện trong nội dung đã stream."""

    type: Literal["citations"] = "citations"
    citations: list[Citation]


class WarningEvent(BaseModel):
    """Cảnh báo hậu kiểm không thể thu hồi nội dung đã gửi."""

    type: Literal["warning"] = "warning"
    code: Literal["truncated", "invalid_citation", "unverified_number"]
    message: str
    detail: str = ""


class ErrorEvent(BaseModel):
    """Lỗi khiến generation không thể tiếp tục một cách an toàn."""

    type: Literal["error"] = "error"
    code: Literal[
        "rate_limited",
        "llm_error",
        "retrieval_error",
        "no_context",
        "quota_exceeded",
    ]
    message: str
    retry_after_seconds: float | None = None


class DoneEvent(BaseModel):
    """Sự kiện cuối cùng, luôn được phát cho mọi luồng."""

    type: Literal["done"] = "done"
    usage: Usage | None = None


type GenerationEvent = Annotated[
    StatusEvent
    | TokenEvent
    | RefusalEvent
    | CitationsEvent
    | WarningEvent
    | ErrorEvent
    | DoneEvent,
    Field(discriminator="type"),
]
