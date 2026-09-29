"""Trace Langfuse cho một lượt hỏi (observability_spec.md mục 4).

`observability/` là hạ tầng cross-cutting (giống logging): được phép import
trực tiếp từ bất kỳ module gọi LLM nào, không cần đi qua tầng điều phối như
`chatlog/`/`cache/`. Thiếu `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` khiến
SDK tự chuyển sang chế độ disabled: mọi `span`/`generation` trở thành no-op,
không raise, không log rác — đây là toàn bộ cơ chế fail-safe (không có cờ
`OBSERVABILITY_ENABLED` riêng, mục 4.1).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse

from production_legal_qa_rag.config import LangfuseSettings

_client: Langfuse | None = None
# SDK giữ `_tracing_enabled=True` ngay cả khi disabled do thiếu key (chỉ tracer
# nội bộ đổi thành no-op) — tự theo dõi riêng để `current_trace_id()` không gọi
# `Langfuse.get_current_trace_id()` lúc disabled: bản thân SDK log WARNING
# "Context error: No active span..." mỗi lần gọi hàm đó khi không có span OTel
# thật đang active, tức lặp lại mỗi lượt hỏi — vi phạm "không log rác" (mục 4.1,
# nghiệm thu mục 9.6). Đây là điểm khác biệt giữa pseudo-code của spec (giả định
# SDK tự im lặng hoàn toàn) và hành vi thật của SDK v4 đã cài (4.15.x).
_client_enabled = False


def get_langfuse_client() -> Langfuse:
    """Singleton Langfuse client dựng từ `LangfuseSettings`.

    SDK tự no-op nếu thiếu `public_key`/`secret_key` (mục 4.1): các lời gọi
    `start_as_current_observation` sau đó không raise, không gửi dữ liệu đi.

    Returns:
        Instance `Langfuse` dùng chung cho toàn bộ process.
    """
    global _client, _client_enabled
    if _client is None:
        settings = LangfuseSettings()
        _client_enabled = (
            settings.public_key is not None and settings.secret_key is not None
        )
        _client = Langfuse(
            public_key=(
                settings.public_key.get_secret_value()
                if settings.public_key is not None
                else None
            ),
            secret_key=(
                settings.secret_key.get_secret_value()
                if settings.secret_key is not None
                else None
            ),
            base_url=settings.base_url,
        )
    return _client


@contextmanager
def span(name: str, **kwargs: Any) -> Iterator[Any]:
    """Span cho bước không gọi LLM (cache lookup, container bao ngoài).

    Args:
        name: Tên hiển thị trên cây trace Langfuse.
        **kwargs: Chuyển thẳng vào `start_as_current_observation` (`input`,
            `output`, `metadata`, ...).

    Yields:
        Observation đang active; gọi `.update(...)` để bổ sung output/metadata
        trước khi span đóng.
    """
    with get_langfuse_client().start_as_current_observation(
        name=name, as_type="span", **kwargs
    ) as observation:
        yield observation


@contextmanager
def generation(name: str, *, model: str, **kwargs: Any) -> Iterator[Any]:
    """Generation cho bước gọi LLM — Langfuse hiển thị riêng model/usage/cost.

    Args:
        name: Tên hiển thị trên cây trace Langfuse.
        model: Tên model đã dùng cho lời gọi này.
        **kwargs: Chuyển thẳng vào `start_as_current_observation` (`metadata`
            nên gồm `key_bucket` lấy từ `retrieval/llm_throttle.describe_bucket`,
            mục 4.3).

    Yields:
        Observation đang active; gọi `.update(...)` để bổ sung output/usage
        trước khi generation đóng.
    """
    with get_langfuse_client().start_as_current_observation(
        name=name, as_type="generation", model=model, **kwargs
    ) as observation:
        yield observation


def current_trace_id() -> str | None:
    """ID trace Langfuse đang active.

    Không gọi `Langfuse.get_current_trace_id()` khi client disabled: SDK log
    WARNING mỗi lần gọi hàm đó lúc không có span OTel thật (xem comment cạnh
    `_client_enabled`), gọi mỗi lượt hỏi sẽ thành log rác — trả `None` sớm để
    tránh hẳn lời gọi đó khi chưa cấu hình Langfuse.

    Returns:
        `None` nếu không có trace đang active (client disabled hoặc gọi ngoài
        một `span`/`generation`).
    """
    client = get_langfuse_client()
    if not _client_enabled:
        return None
    return client.get_current_trace_id()
