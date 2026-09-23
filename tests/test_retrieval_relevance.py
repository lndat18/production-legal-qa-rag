"""Test `retrieval/relevance.py` — gate độ liên quan (conversation_spec.md mục 18.2.2)."""

from __future__ import annotations

from production_legal_qa_rag.retrieval.models import RetrievedChunk
from production_legal_qa_rag.retrieval.relevance import (
    MIN_RERANK_SCORE,
    is_low_relevance,
)


def _chunk(chunk_id: str, rerank_score: float | None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_document="d",
        breadcrumb="Điều 1",
        content="x",
        rerank_score=rerank_score,
    )


def test_empty_chunks_are_not_low_relevance() -> None:
    assert is_low_relevance([]) is False


def test_all_scores_none_keeps_old_behavior() -> None:
    """Rerank lỗi/fallback (mọi `rerank_score` là `None`) -> không gate mù."""
    chunks = [_chunk("c1", None), _chunk("c2", None)]
    assert is_low_relevance(chunks) is False


def test_all_scores_below_threshold_is_low_relevance() -> None:
    chunks = [
        _chunk("c1", MIN_RERANK_SCORE - 0.5),
        _chunk("c2", MIN_RERANK_SCORE - 1.0),
    ]
    assert is_low_relevance(chunks) is True


def test_one_high_score_is_enough_to_pass() -> None:
    """Chỉ cần 1 chunk đủ liên quan (max >= ngưỡng) là không gate."""
    chunks = [
        _chunk("c1", MIN_RERANK_SCORE - 3.0),
        _chunk("c2", MIN_RERANK_SCORE + 0.01),
        _chunk("c3", None),
    ]
    assert is_low_relevance(chunks) is False


def test_score_exactly_at_threshold_is_not_low_relevance() -> None:
    """`max(scores) < MIN_RERANK_SCORE`: bằng ngưỡng thì vẫn coi là đủ liên quan."""
    chunks = [_chunk("c1", MIN_RERANK_SCORE)]
    assert is_low_relevance(chunks) is False


def test_mix_of_none_and_low_scores_is_low_relevance() -> None:
    chunks = [_chunk("c1", None), _chunk("c2", MIN_RERANK_SCORE - 0.1)]
    assert is_low_relevance(chunks) is True
