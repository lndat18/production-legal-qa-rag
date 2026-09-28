"""Thiết lập môi trường tối thiểu, không có secret, cho test suite."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator

import pytest


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
    if module is not None:
        module._get_bucket_throttle.cache_clear()


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
