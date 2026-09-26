"""Xác thực Bearer key và dựng danh tính request (api_spec.md mục 4).

Backend không tự quản lý user — chỉ tin ``X-OpenWebUI-User-Id``/``-Chat-Id`` do
OpenWebUI gửi kèm, và chỉ sau khi Bearer key đã hợp lệ. Không đọc/lưu email hay
tên người dùng.
"""

from __future__ import annotations

import secrets
import uuid
from typing import Annotated

from fastapi import Header, Request

from production_legal_qa_rag.api.schemas import ApiError
from production_legal_qa_rag.config import ApiSettings
from production_legal_qa_rag.conversation.models import RequestContext

ANONYMOUS_USER_ID = "anonymous"
_BEARER_PREFIX = "Bearer "


async def authenticate(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_openwebui_user_id: Annotated[
        str | None, Header(alias="X-OpenWebUI-User-Id")
    ] = None,
    x_openwebui_chat_id: Annotated[
        str | None, Header(alias="X-OpenWebUI-Chat-Id")
    ] = None,
) -> RequestContext:
    """Xác thực Bearer key rồi dựng ``RequestContext`` từ header OpenWebUI.

    Args:
        request: Request hiện tại, dùng để đọc ``ApiSettings`` đã khởi tạo ở
            lifespan (``app.state.api_settings``).
        authorization: Header ``Authorization: Bearer <key>``.
        x_openwebui_user_id: Id nội bộ OpenWebUI; thiếu -> ``"anonymous"``.
        x_openwebui_chat_id: Id cuộc hội thoại OpenWebUI; có thể vắng mặt.

    Returns:
        Danh tính request cho lượt hiện tại, ``request_id`` sinh mới mỗi lần.

    Raises:
        ApiError: 401 khi thiếu hoặc sai key.
    """
    settings: ApiSettings = request.app.state.api_settings
    _verify_bearer(authorization, settings.chatbot_api_key.get_secret_value())
    context = RequestContext(
        user_id=x_openwebui_user_id or ANONYMOUS_USER_ID,
        chat_id=x_openwebui_chat_id,
        request_id=uuid.uuid4().hex,
    )
    # Cho exception handler chung đọc lại request_id khi lỗi xảy ra sau auth
    # (mục 12: log lỗi không lường trước phải kèm request_id).
    request.state.request_id = context.request_id
    return context


def _verify_bearer(authorization: str | None, expected_key: str) -> None:
    """So sánh Bearer key bằng ``secrets.compare_digest`` để tránh timing attack."""
    if authorization is None or not authorization.startswith(_BEARER_PREFIX):
        raise _unauthorized()
    provided = authorization[len(_BEARER_PREFIX) :]
    if not secrets.compare_digest(provided, expected_key):
        raise _unauthorized()


def _unauthorized() -> ApiError:
    return ApiError(
        401,
        "Thiếu hoặc sai khoá xác thực.",
        "invalid_request_error",
        "invalid_api_key",
    )
