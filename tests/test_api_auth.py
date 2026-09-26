"""Tests that protect the Bearer auth and identity contract of `api/auth.py`.

api_spec.md mục 4: sai/thiếu key -> 401; danh tính chỉ lấy từ
``X-OpenWebUI-User-Id``/``-Chat-Id`` sau khi key hợp lệ; thiếu ``User-Id`` ->
``"anonymous"``; ``request_id`` sinh mới mỗi lần và được lưu vào
``request.state`` để exception handler đọc lại.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from production_legal_qa_rag.api.auth import ANONYMOUS_USER_ID, authenticate
from production_legal_qa_rag.api.schemas import ApiError
from production_legal_qa_rag.config import ApiSettings

_API_KEY = "test-chatbot-api-key"


def _fake_request() -> SimpleNamespace:
    """Build the minimal ``Request``-like object ``authenticate`` reads from."""
    settings = ApiSettings(chatbot_api_key=_API_KEY)
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(api_settings=settings)),
        state=SimpleNamespace(),
    )


def test_missing_authorization_header_raises_401() -> None:
    """No ``Authorization`` header at all must be rejected, not treated as anonymous."""
    request = _fake_request()

    with pytest.raises(ApiError) as excinfo:
        asyncio.run(authenticate(request, authorization=None))

    assert excinfo.value.status_code == 401
    assert excinfo.value.error_type == "invalid_request_error"
    assert excinfo.value.code == "invalid_api_key"


@pytest.mark.parametrize(
    "authorization",
    [
        "Bearer wrong-key",
        "wrong-scheme " + _API_KEY,
        _API_KEY,  # missing "Bearer " prefix entirely
        "Bearer ",
    ],
)
def test_wrong_or_malformed_bearer_raises_401(authorization: str) -> None:
    """Any deviation from ``Bearer <exact key>`` is a 401, never a partial match."""
    request = _fake_request()

    with pytest.raises(ApiError) as excinfo:
        asyncio.run(authenticate(request, authorization=authorization))

    assert excinfo.value.status_code == 401


def test_valid_bearer_without_user_header_falls_back_to_anonymous() -> None:
    """Missing ``X-OpenWebUI-User-Id`` groups the caller under one shared identity."""
    request = _fake_request()

    context = asyncio.run(authenticate(request, authorization=f"Bearer {_API_KEY}"))

    assert context.user_id == ANONYMOUS_USER_ID
    assert context.chat_id is None
    assert isinstance(context.request_id, str) and context.request_id


def test_valid_bearer_reads_identity_headers_and_sets_request_state() -> None:
    """User/chat id come straight from the OpenWebUI headers once the key is valid."""
    request = _fake_request()

    context = asyncio.run(
        authenticate(
            request,
            authorization=f"Bearer {_API_KEY}",
            x_openwebui_user_id="user-42",
            x_openwebui_chat_id="chat-7",
        )
    )

    assert context.user_id == "user-42"
    assert context.chat_id == "chat-7"
    # app.py's unexpected-error handler reads this back for logging (mục 12).
    assert request.state.request_id == context.request_id


def test_request_id_is_unique_per_call() -> None:
    """Each call must mint a fresh ``request_id`` even for the same identity."""
    request = _fake_request()

    first = asyncio.run(authenticate(request, authorization=f"Bearer {_API_KEY}"))
    second = asyncio.run(authenticate(request, authorization=f"Bearer {_API_KEY}"))

    assert first.request_id != second.request_id
