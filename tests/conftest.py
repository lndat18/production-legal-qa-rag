"""Thiết lập môi trường tối thiểu, không có secret, cho test suite."""

from __future__ import annotations

import os


def pytest_configure() -> None:
    """Cấp token HF giả trước khi pytest thu thập test module.

    ``EmbeddingSettings`` bây giờ bắt buộc ``HF_TOKEN`` theo embedding spec.
    Một số test chunking import setting này khi module được thu thập, nên
    giá trị giả phải có trước collection; không test nào được gọi API thật.
    """
    os.environ.setdefault("HF_TOKEN", "test-hf-token")
