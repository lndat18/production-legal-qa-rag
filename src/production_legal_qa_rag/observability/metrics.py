"""Metrics Prometheus mức vận hành (observability_spec.md mục 5).

Chỉ 2 nhóm, cố tình không đo lặp lại những gì Langfuse (`tracing.py`) đã trả
lời tốt hơn (per-model/key token, per-step latency chi tiết):

1. HTTP mặc định của `prometheus-fastapi-instrumentator` (`instrument_app`).
2. Custom mức lượt hỏi (`record_turn`), nguồn dữ liệu là `TurnTrace` đã điền
   đầy đủ — cùng điểm cuối lượt với việc cập nhật trace (`turn_trace.py`), không
   tính toán lại.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from production_legal_qa_rag.conversation.models import TurnTrace

_logger = logging.getLogger(__name__)

CHAT_TURNS_TOTAL = Counter(
    "chat_turns_total", "Tổng lượt hỏi", ["outcome", "cache_status"]
)
TURN_LATENCY_SECONDS = Histogram(
    "turn_latency_seconds",
    "Độ trễ toàn bộ 1 lượt hỏi (giây)",
    ["outcome"],
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 60, 120),
)
TIME_TO_FIRST_TOKEN_SECONDS = Histogram(
    "time_to_first_token_seconds",
    "Độ trễ tới token đầu tiên (giây)",
    buckets=(0.2, 0.5, 1, 2, 5, 10, 20),
)


def record_turn(trace: TurnTrace) -> None:
    """Cập nhật các metric mức lượt hỏi từ một `TurnTrace` đã hoàn tất.

    Không bao giờ để lỗi ghi metric lan ra ngoài (observability_spec.md mục 1,
    "thiếu/lỗi ... Prometheus thì chatbot vẫn chạy bình thường") — cùng tinh
    thần với `update_turn_trace` (`turn_trace.py`): bắt mọi lỗi tại đây, chỉ
    log warning không kèm nội dung câu hỏi/câu trả lời, không raise.

    Args:
        trace: Vết lượt hỏi đã điền đầy đủ, cùng nguồn dữ liệu với `update_turn_trace`.
    """
    try:
        CHAT_TURNS_TOTAL.labels(
            outcome=trace.outcome, cache_status=trace.cache_status
        ).inc()
        TURN_LATENCY_SECONDS.labels(outcome=trace.outcome).observe(
            trace.latency_ms / 1000
        )
        if trace.time_to_first_token_ms is not None:
            TIME_TO_FIRST_TOKEN_SECONDS.observe(trace.time_to_first_token_ms / 1000)
    except Exception:  # noqa: BLE001 - metrics failures cannot affect the response path.
        _logger.warning("Không thể ghi metrics cho lượt hỏi")


def instrument_app(app: FastAPI) -> None:
    """Bật metrics HTTP mặc định và expose `GET /metrics` (gọi 1 lần khi tạo app).

    Args:
        app: FastAPI application vừa được khởi tạo trong `create_app()`.
    """
    Instrumentator().instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False
    )
