"""Maximal Marginal Relevance trên dense vector (mục 7)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from production_legal_qa_rag.retrieval.models import Candidate

MMR_LAMBDA = 0.5


def select(
    candidates: list[Candidate],
    query_embedding: Sequence[float],
    lambda_: float,
    top_n: int,
) -> list[Candidate]:
    """Chọn tuần tự `top_n` candidate tối đa hoá MMR.

    `MMR = λ·sim(d, query) − (1−λ)·max sim(d, d')` với `d'` đã chọn; `sim` là
    cosine. Candidate không có vector bị bỏ (chỉ xảy ra khi fetch bổ sung
    không trả về).

    Args:
        candidates: Candidate sau RRF, đã kèm `values`.
        query_embedding: Embedding của câu hỏi gốc.
        lambda_: Trọng số relevance so với đa dạng, trong [0, 1].
        top_n: Số candidate cần chọn.

    Returns:
        Candidate theo thứ tự được chọn.
    """
    pool = [c for c in candidates if c.values]
    relevance = {c.chunk_id: _cosine(_values(c), query_embedding) for c in pool}
    selected: list[Candidate] = []
    while pool and len(selected) < top_n:
        best = max(
            pool,
            key=lambda c: (
                lambda_ * relevance[c.chunk_id]
                - (1 - lambda_)
                * max((_cosine(_values(c), _values(s)) for s in selected), default=0.0)
            ),
        )
        selected.append(best)
        pool.remove(best)
    return selected


def _values(candidate: Candidate) -> list[float]:
    return candidate.values or []


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity; 0 nếu một vector có chuẩn bằng 0."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0
