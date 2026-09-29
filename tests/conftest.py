"""Thiết lập môi trường tối thiểu, không có secret, cho test suite."""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from langfuse import Langfuse


def pytest_configure() -> None:
    """Cấp token HF giả trước khi pytest thu thập test module.

    ``EmbeddingSettings`` bây giờ bắt buộc ``HF_TOKEN`` theo embedding spec.
    Một số test chunking import setting này khi module được thu thập, nên
    giá trị giả phải có trước collection; không test nào được gọi API thật.
    """
    os.environ.setdefault("HF_TOKEN", "test-hf-token")


def _clear_throttle_buckets() -> None:
    """Xoá cache bucket throttle nếu module đã được nạp bởi một test nào đó.

    Không import chủ động: `retrieval/__init__` kéo theo torch/pinecone, không nên
    trả giá đó cho các test không đụng tới retrieval. Module chưa nạp thì cache chưa
    có gì để xoá.
    """
    module = sys.modules.get("production_legal_qa_rag.retrieval.llm_throttle")
    if module is None:
        return
    # Test có thể đang thay `_get_bucket_throttle` bằng spy (monkeypatch chỉ hoàn tác
    # sau teardown của fixture này), spy không có `cache_clear`.
    cache_clear = getattr(module._get_bucket_throttle, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


@pytest.fixture(autouse=True)
def reset_llm_throttle_buckets(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Cô lập throttle ``gpt-oss-20b`` giữa các test (conversation_spec.md mục 12.1).

    Bucket ``(model, key)`` của ``get_throttle`` sống trong cả tiến trình và mỗi lời
    gọi condense/HyDE/Judge ước lượng hàng nghìn token, nên nhiều test dùng chung
    model/key giả sẽ tích lũy ngân sách và gặp ``ThrottleTimeout`` ngẫu nhiên theo
    thứ tự chạy. Fixture xoá cache bucket trước/sau mỗi test và nới giới hạn mặc định
    qua env để test không liên quan tới throttle không bị giãn thời gian; test về
    throttle tự đặt lại ``THROTTLE_*`` hoặc inject ``TokenWindowThrottle`` riêng.
    """
    monkeypatch.setenv("THROTTLE_TPM_LIMIT", "1000000000")
    monkeypatch.setenv("THROTTLE_RPM_LIMIT", "1000000")
    _clear_throttle_buckets()
    yield
    _clear_throttle_buckets()


def _reset_langfuse_client() -> None:
    """Xoá singleton Langfuse client nếu `observability/tracing` đã được nạp."""
    module = sys.modules.get("production_legal_qa_rag.observability.tracing")
    if module is not None:
        module._client = None


@pytest.fixture(autouse=True)
def langfuse_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Mặc định Langfuse ở chế độ disabled, không phụ thuộc `.env` của máy dev.

    `get_langfuse_client()` dựng client từ `LangfuseSettings` (đọc `.env` ở root);
    máy dev có khoá thật sẽ khiến mọi test đi qua `tracing.span`/`generation` thực sự
    gửi trace. Fixture tắt việc đọc `.env`, xoá khoá khỏi môi trường và reset singleton
    trước/sau mỗi test (observability_spec.md mục 4.1: thiếu khoá -> no-op).
    """
    from production_legal_qa_rag.config import LangfuseSettings

    monkeypatch.setattr(
        LangfuseSettings,
        "model_config",
        {**LangfuseSettings.model_config, "env_file": None},
    )
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    _reset_langfuse_client()
    yield
    _reset_langfuse_client()


@pytest.fixture
def in_memory_langfuse(monkeypatch: pytest.MonkeyPatch) -> Iterator[Langfuse]:
    """Langfuse client thật (enabled) nhưng hermetic: không gửi gì qua mạng.

    Khoá giả duy nhất mỗi test (resource manager của SDK là singleton theo
    `public_key`), `TracerProvider` riêng (không đụng provider toàn cục của OTel) và
    `InMemorySpanExporter` thay cho OTLP exporter nên không có kết nối nào tới
    `base_url` (cổng không lắng nghe). Client được gắn vào `tracing._client` để
    `tracing.span`/`generation` dùng nó; test đọc thuộc tính OTel qua fixture
    `observation_attributes`.
    """
    from langfuse import Langfuse as LangfuseClient
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from production_legal_qa_rag.observability import tracing

    for name in (
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_TRACING_ENABLED",
        "LANGFUSE_SAMPLE_RATE",
        "LANGFUSE_TRACING_ENVIRONMENT",
        "OTEL_SDK_DISABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    provider = TracerProvider()
    client = LangfuseClient(
        public_key=f"pk-lf-test-{uuid.uuid4().hex}",
        secret_key="sk-lf-test",
        base_url="http://127.0.0.1:9",
        tracer_provider=provider,
        span_exporter=InMemorySpanExporter(),
        sample_rate=1.0,
    )
    monkeypatch.setattr(tracing, "_client", client)
    yield client
    client.shutdown()
    provider.shutdown()


@pytest.fixture
def observation_attributes() -> Callable[[Any], dict[str, Any]]:
    """Đọc thuộc tính OTel của một observation Langfuse (span/generation).

    Đây là NƠI DUY NHẤT test truy cập `observation._otel_span` — thuộc tính nội bộ
    (private) của Langfuse SDK v3, không thuộc API công khai. Khi nâng phiên bản
    `langfuse`, rà lại helper này; các test khác chỉ gọi fixture, không đụng `_otel_span`.
    """

    def read(observation: Any) -> dict[str, Any]:
        return dict(observation._otel_span.attributes)

    return read
