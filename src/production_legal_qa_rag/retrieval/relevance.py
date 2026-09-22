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

# Logit thô (không qua sigmoid) của `AITeamVN/Vietnamese_Reranker`.
#
# REVISE (2026-09-22, PR #41 review): ngưỡng ban đầu -6.0 đo **1 lần/câu** —
# không tính tới nhiễu HyDE (`retrieval/hyde.py` gọi Groq với `temperature=0.2`,
# hypothetical document dao động giữa các lần gọi cùng câu hỏi). Đo lại **3
# lần/câu** (script dev tạm, không commit) trên 12 câu hợp lệ cũ + 2 biến thể
# ca 9 + 3 câu rìa corpus (bảo hiểm thất nghiệp, tai nạn lao động, kỷ luật lao
# động) — xem `conversation_spec.md` mục 18.2.2 "CẬP NHẬT" cho số liệu đầy đủ
# (khoảng dao động min-max từng câu):
#   - min(min-score qua 3 lần) của 12 câu hợp lệ cũ: -4.0964 ("làm thêm giờ",
#     ổn định, spread 0 qua 3 lần — an toàn)
#   - max(max-score qua 3 lần) của ca 9 (2 biến thể): -7.0371 (ổn định, spread
#     0 qua 3 lần)
#   - 2/3 câu rìa corpus (không có luật riêng trong `data/raw/`, xem mục
#     18.2.2) dao động MẠNH do nhiễu HyDE: "trợ cấp thất nghiệp" dao động
#     [-5.9692, -3.1486] (spread 2.82), "tai nạn lao động trên đường đi làm"
#     dao động [-6.2113, -4.0244] (spread 2.19) — xác nhận đúng quan sát của
#     reviewer. Câu rìa thứ 3 ("kỷ luật lao động", CÓ chương riêng trong Bộ
#     luật Lao động) ổn định ở -0.7616 — không phải ca biên, loại khỏi cân nhắc
#     ngưỡng.
# Khoảng an toàn thật giữa max(ca 9)=-7.0371 và min(2 câu rìa thực sự biên)=
# -6.2113 chỉ rộng ~0.83 — KHÔNG đạt biên độ an toàn mong muốn (>= 1.5-2 lần
# biên độ nhiễu quan sát ~2.2-2.8, tức cần margin >= ~3.3-5.6). Không tìm được
# ngưỡng nào an toàn tuyệt đối cho 2 câu rìa này; xem `conversation_spec.md`
# mục 18.2.2 phần "Rủi ro tồn đọng" cho phân tích đầy đủ và các phương án đã
# cân nhắc (đo nhiều lần rồi lấy trung bình, hạ temperature HyDE — cả hai đều
# NGOÀI phạm vi vòng sửa này).
# Chọn -6.8: nằm trong khoảng an toàn hẹp trên, thiên về phía không chặn oan
# câu hợp lệ (margin ~0.59-0.83 với 2 câu rìa) hơn là chặn chắc ca 9 (margin
# ~0.24 với ca 9b) — đúng tinh thần "thà bỏ sót còn hơn chặn oan" đã chốt.
# Margin với 12 câu hợp lệ cũ (không rìa) vẫn rất lớn (>= 2.70, "làm thêm giờ"
# là câu sát nhất). Ca 9 vẫn có 2 lớp phòng thủ độc lập khác (guardrail,
# GENERATION_SYSTEM_PROMPT quy tắc 12) nếu lọt gate này.
MIN_RERANK_SCORE: Final = -6.8


def is_low_relevance(chunks: list[RetrievedChunk]) -> bool:
    """True nếu không có chunk nào đủ liên quan (rerank_score thấp/không có)."""
    scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
    if not scores:
        return False  # rerank lỗi/fallback: giữ hành vi cũ, không gate mù
    return max(scores) < MIN_RERANK_SCORE
