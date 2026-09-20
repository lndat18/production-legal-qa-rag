"""Reciprocal Rank Fusion cho 2 danh sách của một nhánh (mục 6.3)."""

from __future__ import annotations

from production_legal_qa_rag.retrieval.models import Candidate, SearchHit

RRF_K = 60


def rrf(
    dense_hits: list[SearchHit], sparse_hits: list[SearchHit], *, k: int = RRF_K
) -> list[Candidate]:
    """Hợp nhất dense và sparse theo `chunk_id` bằng RRF.

    `rrf_score = Σ 1 / (k + rank)` (rank bắt đầu từ 1) cộng qua các danh sách
    mà chunk có mặt. Metadata/vector lấy từ hit dense nếu có (sparse index
    không lưu chúng).

    Args:
        dense_hits: Kết quả dense, đã sắp giảm dần theo độ liên quan.
        sparse_hits: Kết quả sparse, đã sắp giảm dần theo độ liên quan.
        k: Hằng số RRF.

    Returns:
        Candidate sắp giảm dần theo `rrf_score` (ổn định khi bằng điểm).
    """
    candidates: dict[str, Candidate] = {}
    for hits, carries_payload in ((dense_hits, True), (sparse_hits, False)):
        for rank, hit in enumerate(hits, start=1):
            contribution = 1.0 / (k + rank)
            existing = candidates.get(hit.chunk_id)
            if existing is None:
                candidates[hit.chunk_id] = Candidate(
                    chunk_id=hit.chunk_id,
                    rrf_score=contribution,
                    metadata=hit.metadata if carries_payload else None,
                    values=hit.values if carries_payload else None,
                )
            else:
                existing.rrf_score += contribution
    return sorted(candidates.values(), key=lambda candidate: -candidate.rrf_score)
