"""Trace Langfuse cho một lượt hỏi (observability_spec.md mục 4).

`observability/` là hạ tầng cross-cutting (giống logging): được phép import
trực tiếp từ bất kỳ module gọi LLM nào, không cần đi qua tầng điều phối như
`cache/`. Thiếu `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` khiến SDK tự chuyển
sang chế độ disabled: mọi `span`/`generation` trở thành no-op, không raise,
không log rác — đây là toàn bộ cơ chế fail-safe (không có cờ
`OBSERVABILITY_ENABLED` riêng, mục 4.1).

Việc điền output/metadata/tags cuối lượt lên root span nằm ở `turn_trace.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse, propagate_attributes

from production_legal_qa_rag.config import LangfuseSettings

_client: Langfuse | None = None


def get_langfuse_client() -> Langfuse:
    """Singleton Langfuse client dựng từ `LangfuseSettings`.

    SDK tự no-op nếu thiếu `public_key`/`secret_key` (mục 4.1): các lời gọi
    `start_as_current_observation` sau đó không raise, không gửi dữ liệu đi.

    Returns:
        Instance `Langfuse` dùng chung cho toàn bộ process.
    """
    global _client
    if _client is None:
        settings = LangfuseSettings()
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
def span(
    name: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    **kwargs: Any,
) -> Iterator[Any]:
    """Span cho bước không gọi LLM (cache lookup, container bao ngoài).

    Args:
        name: Tên hiển thị trên cây trace Langfuse.
        user_id: Người dùng của trace (thuộc tính trace chuẩn của Langfuse, lan
            xuống mọi observation con). `None` thì bỏ trống, không bịa giá trị.
        session_id: Phiên/cuộc hội thoại của trace, gom các lượt cùng session.
            `None` thì bỏ trống.
        **kwargs: Chuyển thẳng vào `start_as_current_observation` (`input`,
            `output`, `metadata`, ...).

    Yields:
        Observation đang active; gọi `.update(...)` để bổ sung output/metadata
        trước khi span đóng.
    """
    with get_langfuse_client().start_as_current_observation(
        name=name, as_type="span", **kwargs
    ) as observation:
        if user_id is None and session_id is None:
            yield observation
            return
        with propagate_attributes(user_id=user_id, session_id=session_id):
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
