"""Tests that protect the OpenAI request/response Pydantic contract (`api/schemas.py`).

api_spec.md mục 2: ``model``/``messages`` are required; any other OpenAI
parameter (``temperature``, ``top_p``...) must be silently ignored rather than
rejected, since OpenWebUI may send them even though the backend does not use
them.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from production_legal_qa_rag.api.schemas import (
    ApiError,
    ChatCompletionRequest,
    ChatCompletionRequestMessage,
    StreamOptions,
)


def test_chat_completion_request_requires_model_and_messages() -> None:
    """Both fields are mandatory; missing either is a validation failure."""
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[{"role": "user", "content": "hỏi"}])

    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="legal-qa")


def test_chat_completion_request_ignores_unknown_openai_parameters() -> None:
    """``temperature``/``top_p``/... must not raise (mục 2: "bị bỏ qua")."""
    request = ChatCompletionRequest.model_validate(
        {
            "model": "legal-qa",
            "messages": [{"role": "user", "content": "hỏi"}],
            "temperature": 0.9,
            "top_p": 0.5,
            "tools": [{"type": "function"}],
        }
    )

    assert request.model == "legal-qa"
    assert not hasattr(request, "temperature")


def test_chat_completion_request_defaults() -> None:
    """``stream`` defaults false and ``stream_options`` is optional."""
    request = ChatCompletionRequest(
        model="legal-qa", messages=[{"role": "user", "content": "hỏi"}]
    )

    assert request.stream is False
    assert request.stream_options is None


def test_chat_completion_request_message_defaults_empty_content() -> None:
    """A message without ``content`` must not fail validation (mục 2)."""
    message = ChatCompletionRequestMessage(role="assistant")

    assert message.content == ""


def test_stream_options_default_excludes_usage() -> None:
    """``include_usage`` defaults to ``False`` when the client omits it."""
    options = StreamOptions()

    assert options.include_usage is False


def test_chat_completion_request_role_accepts_any_string() -> None:
    """``role`` is not restricted to a Literal so ``system`` messages parse fine.

    ``routes._to_chat_messages`` is the layer that later drops ``system``
    (mục 9); the schema itself must not reject it.
    """
    request = ChatCompletionRequest(
        model="legal-qa",
        messages=[
            {"role": "system", "content": "prompt mặc định"},
            {"role": "user", "content": "hỏi"},
        ],
    )

    assert [m.role for m in request.messages] == ["system", "user"]


def test_api_error_carries_optional_retry_after() -> None:
    """``ApiError`` without ``retry_after_seconds`` must default to ``None``."""
    error = ApiError(401, "sai key", "invalid_request_error", "invalid_api_key")

    assert error.retry_after_seconds is None
    assert error.status_code == 401


def test_api_error_with_retry_after_for_rate_limit() -> None:
    """429 errors must be able to carry a positive ``retry_after_seconds``."""
    error = ApiError(
        429,
        "quá nhiều yêu cầu",
        "rate_limit_error",
        "rate_limit_exceeded",
        retry_after_seconds=12.0,
    )

    assert error.retry_after_seconds == 12.0
