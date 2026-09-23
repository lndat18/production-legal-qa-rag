"""Model dữ liệu công khai của `cache/` (cache_spec.md mục 2).

Chỉ có phần model mà `conversation/` cần; `store`, `singleflight`, `replay`
chưa được triển khai.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from production_legal_qa_rag.generation.models import Citation

type CacheStatus = Literal["answer_hit", "retrieval_hit", "miss", "bypass"]


class CachedAnswer(BaseModel):
    """Câu trả lời sạch (không warning/usage) đã lưu để phát lại."""

    text: str
    citations: list[Citation]
    created_at: datetime
