"""Tín hiệu độ liên quan dựa trên `rerank_score` (mục 16 điểm 8).

Module này **không** đổi hợp đồng `retrieve()` — `retrieve()` vẫn luôn trả top
`FINAL_TOP_K` chunk theo rerank, không lọc gì (`retrieval_spec.md` mục 1).
Quyết định "5 chunk không đủ liên quan -> từ chối" là chính sách của
`conversation/orchestrator.py` (`conversation_spec.md` mục 18.2.2); hàm ở đây
chỉ cung cấp tín hiệu, không tự gọi `no_context`.
"""

from __future__ import annotations

from typing import Final

from production_legal_qa_rag.retrieval.models import RetrievedChunk

# Logit thô (không qua sigmoid) của `AITeamVN/Vietnamese_Reranker`. Đo trên 12
# câu (4 viện dẫn Khoản + 8 câu hợp lệ khác không viện dẫn) và 2 biến thể ca 9
# (meta-request, biết chắc không liên quan) — xem `conversation_spec.md` mục
# 18.2.2 cho số liệu đầy đủ:
#   - min(max-score) của câu hợp lệ đã đo: -4.0964 ("làm thêm giờ")
#   - max(max-score) của ca 9 đã đo: -7.0697
# Chọn -6.0, thiên về bảo thủ (gần phía ca 9 hơn phía câu hợp lệ) theo đúng chỉ
# dẫn "thà bỏ sót còn hơn chặn oan" — margin ~1.9 so với câu hợp lệ thấp nhất đã
# đo, ~1.07 so với ca 9 cao nhất đã đo.
MIN_RERANK_SCORE: Final = -6.0


def is_low_relevance(chunks: list[RetrievedChunk]) -> bool:
    """True nếu không có chunk nào đủ liên quan (rerank_score thấp/không có)."""
    scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
    if not scores:
        return False  # rerank lỗi/fallback: giữ hành vi cũ, không gate mù
    return max(scores) < MIN_RERANK_SCORE
