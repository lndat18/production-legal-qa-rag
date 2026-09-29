"""Test `observability/metrics.py` (observability_spec.md mục 5).

`CHAT_TURNS_TOTAL`/`TURN_LATENCY_SECONDS`/`TIME_TO_FIRST_TOKEN_SECONDS` là Counter/
Histogram đăng ký 1 lần ở module-level trên `prometheus_client.REGISTRY` mặc định —
cộng dồn xuyên suốt cả phiên pytest. Mỗi test so sánh **độ tăng** (before/after)
thay vì giá trị tuyệt đối để không phụ thuộc thứ tự chạy test khác trong cùng
tiến trình.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from production_legal_qa_rag.conversation.models import TurnTrace
from production_legal_qa_rag.observability import metrics


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


# --------------------------------------------------------------- record_turn


def test_record_turn_tang_dem_theo_outcome_va_cache_status() -> None:
    labels = {"outcome": "answered", "cache_status": "answer_hit"}
    before = _sample("chat_turns_total", labels)

    metrics.record_turn(TurnTrace(outcome="answered", cache_status="answer_hit"))

    assert _sample("chat_turns_total", labels) == before + 1.0


def test_record_turn_quan_sat_do_tre_theo_giay_khong_phai_mili_giay() -> None:
    labels = {"outcome": "refused"}
    count_before = _sample("turn_latency_seconds_count", labels)
    sum_before = _sample("turn_latency_seconds_sum", labels)

    metrics.record_turn(
        TurnTrace(outcome="refused", cache_status="miss", latency_ms=2000)
    )

    assert _sample("turn_latency_seconds_count", labels) == count_before + 1.0
    assert _sample("turn_latency_seconds_sum", labels) == sum_before + 2.0


def test_record_turn_bo_qua_time_to_first_token_khi_none() -> None:
    before = _sample("time_to_first_token_seconds_count")

    metrics.record_turn(
        TurnTrace(outcome="refused", cache_status="miss", time_to_first_token_ms=None)
    )

    assert _sample("time_to_first_token_seconds_count") == before


def test_record_turn_quan_sat_time_to_first_token_khi_co_gia_tri() -> None:
    before = _sample("time_to_first_token_seconds_count")
    sum_before = _sample("time_to_first_token_seconds_sum")

    metrics.record_turn(
        TurnTrace(outcome="answered", cache_status="miss", time_to_first_token_ms=800)
    )

    assert _sample("time_to_first_token_seconds_count") == before + 1.0
    assert _sample("time_to_first_token_seconds_sum") == sum_before + 0.8


def test_record_turn_khong_lam_lan_loi_ra_ngoai_khi_metric_that_bai(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """observability_spec.md mục 1: lỗi Prometheus không bao giờ ảnh hưởng
    response path — tương tự `test_failed_background_record_warning_has_no_turn_content`
    (chatlog) nhưng cho `record_turn`."""
    raw_query = "PRIVATE_QUERY_DO_NOT_LOG"
    answer_text = "PRIVATE_ANSWER_DO_NOT_LOG"

    def _boom(*_: object, **__: object) -> None:
        raise RuntimeError(f"bad label: {raw_query} / {answer_text}")

    monkeypatch.setattr(metrics.CHAT_TURNS_TOTAL, "labels", _boom)

    caplog.set_level(
        logging.WARNING, logger="production_legal_qa_rag.observability.metrics"
    )
    metrics.record_turn(
        TurnTrace(
            outcome="answered",
            cache_status="miss",
            raw_query=raw_query,
            answer_text=answer_text,
        )
    )

    assert "Không thể ghi metrics" in caplog.text
    assert raw_query not in caplog.text
    assert answer_text not in caplog.text


# --------------------------------------------------------------- instrument_app


def test_instrument_app_expose_metrics_endpoint_voi_du_lieu_da_ghi() -> None:
    app = FastAPI()

    @app.get("/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    metrics.instrument_app(app)
    metrics.record_turn(TurnTrace(outcome="answered", cache_status="miss"))

    client = TestClient(app)
    client.get("/ping")
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text
    assert "chat_turns_total" in response.text


def test_instrument_app_khong_dua_metrics_vao_openapi_schema() -> None:
    """`include_in_schema=False` (spec mục 5) — `/metrics` không lộ ra OpenAPI docs."""
    app = FastAPI()
    metrics.instrument_app(app)

    schema = app.openapi()

    assert "/metrics" not in schema["paths"]
