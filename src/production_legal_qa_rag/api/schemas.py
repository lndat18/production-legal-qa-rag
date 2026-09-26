"""Schema request/response OpenAI (api_spec.md mục 2, 6, 11, 12).

Các model ở đây theo pydantic v2 và chỉ giữ đủ trường mà OpenWebUI thực sự
dùng — tham số OpenAI khác (``temperature``, ``top_p``…) bị bỏ qua nhờ
``model_config = ConfigDict(extra="ignore")``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Chatbot chỉ phục vụ đúng một "model" ảo cho OpenWebUI (api_spec.md mục 2).
MODEL_ID = "legal-qa"
MODEL_OWNED_BY = "production-legal-qa-rag"


class ApiError(Exception):
    """Lỗi trước khi stream bắt đầu; ``app.py`` chuyển thành body OpenAI.

    Args:
        status_code: Mã HTTP trả về.
        message: Thông điệp con người đọc được, không lộ nội dung nội bộ.
        error_type: Trường ``error.type`` trong body OpenAI.
        code: Trường ``error.code`` trong body OpenAI.
        retry_after_seconds: Khi có, ``app.py`` set thêm header ``Retry-After``.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str,
        code: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code
        self.retry_after_seconds = retry_after_seconds


class ErrorDetail(BaseModel):
    """Nội dung lỗi theo đúng 3 trường OpenAI cần (mục 2)."""

    message: str
    type: str
    code: str


class ErrorResponse(BaseModel):
    """Body lỗi OpenAI-compatible: ``{"error": {...}}``."""

    error: ErrorDetail


class ChatCompletionRequestMessage(BaseModel):
    """Một message thô trong ``messages[]`` của client.

    ``role`` giữ nguyên dạng ``str`` (không giới hạn Literal) vì OpenWebUI có
    thể gửi ``system`` — bị bỏ qua ở tầng chuyển đổi (mục 9), không phải lỗi.
    """

    role: str
    content: str = ""


class StreamOptions(BaseModel):
    """Tuỳ chọn stream OpenAI; chỉ ``include_usage`` được dùng (mục 2)."""

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """Body ``POST /v1/chat/completions``; tham số khác ngoài mục 2 bị bỏ qua."""

    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatCompletionRequestMessage]
    stream: bool = False
    stream_options: StreamOptions | None = None


class ModelObject(BaseModel):
    """Một model trong ``GET /v1/models`` (đúng một phần tử, mục 2)."""

    id: str
    object: Literal["model"] = "model"
    owned_by: str


class ModelListResponse(BaseModel):
    """Body ``GET /v1/models``."""

    object: Literal["list"] = "list"
    data: list[ModelObject]


class ChatCompletionChunkDelta(BaseModel):
    """Delta của một chunk stream; chỉ field khác ``None`` mới được serialize."""

    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning_content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """Một choice trong chunk stream (luôn đúng một choice, index 0)."""

    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Literal["stop", "length"] | None = None


class UsageObject(BaseModel):
    """Usage OpenAI tối giản, suy từ ``generation.models.Usage`` khi có."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChunk(BaseModel):
    """Một dòng ``data:`` của luồng SSE ``chat.completion.chunk``."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice] = Field(default_factory=list)
    usage: UsageObject | None = None


class ChatCompletionMessageResponse(BaseModel):
    """Message hoàn chỉnh trong response không-stream."""

    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """Choice duy nhất của response không-stream."""

    index: int = 0
    message: ChatCompletionMessageResponse
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(BaseModel):
    """Body ``chat.completion`` khi ``stream=false`` (mục 2)."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageObject | None = None
