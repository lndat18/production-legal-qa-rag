"""Rate limit theo phút qua Redis, fail-open khi Redis lỗi (api_spec.md mục 5).

Áp dụng cho **mọi** request tới ``/v1/chat/completions`` (kể cả cache hit) —
đây là hàng phòng thủ đầu tiên chống lạm dụng, trước cả quota ngày và hàng đợi
đồng thời của ``AdmissionController`` (chỉ tính khi thực sự gọi LLM).
"""

from __future__ import annotations

import logging
import time

from redis.asyncio import Redis

from production_legal_qa_rag.api.schemas import ApiError

logger = logging.getLogger(__name__)

# Cửa sổ cố định theo phút UNIX; TTL 70s (> 60s) để không xoá key trước khi
# phút kế tiếp bắt đầu đọc, tránh reset đếm sớm do lệch giờ.
_KEY_TTL_SECONDS = 70


async def enforce_rate_limit(redis: Redis, user_id: str, limit_per_minute: int) -> None:
    """Tăng bộ đếm rate limit của user trong phút hiện tại.

    Args:
        redis: Client Redis dùng chung của API.
        user_id: Danh tính đã xác thực (``"anonymous"`` nếu thiếu User-Id).
        limit_per_minute: Ngưỡng tối đa cho phép trong một phút.

    Raises:
        ApiError: 429 kèm ``retry_after_seconds`` khi vượt ngưỡng. Lỗi Redis
            không ném ra ngoài — coi như fail-open, chỉ log warning.
    """
    minute_bucket = int(time.time() // 60)
    key = f"rl:{user_id}:{minute_bucket}"
    try:
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, _KEY_TTL_SECONDS)
    except Exception:
        logger.warning(
            "Rate limit Redis lỗi; fail-open cho user_id=%s.", user_id, exc_info=True
        )
        return
    if count > limit_per_minute:
        raise ApiError(
            429,
            "Bạn đã gửi quá nhiều yêu cầu trong một phút. Vui lòng thử lại sau.",
            "rate_limit_error",
            "rate_limit_exceeded",
            retry_after_seconds=_seconds_to_next_minute(),
        )


def _seconds_to_next_minute() -> int:
    return 60 - int(time.time() % 60)
