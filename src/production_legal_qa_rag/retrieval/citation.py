"""Nhận diện câu hỏi viện dẫn Điều và chọn extras từ sparse (mục 8.1).

Với câu hỏi kiểu "Điều 36 khoản 2 quy định gì", chunk đáp án thường xếp cao ở
sparse nhưng thấp ở dense nên dễ bị RRF/MMR cắt; các extras đảm bảo top-k
sparse của nhánh B luôn vào union.
"""

from __future__ import annotations

import re
import unicodedata

from production_legal_qa_rag.retrieval.models import Candidate, SearchHit

CITATION_SPARSE_TOP_K = 10

# (?<!\w) để "điều" không dính vào cuối từ khác; số Điều là tín hiệu chính,
# "khoản N" đứng một mình không đủ.
_CITATION_PATTERN = re.compile(r"(?<!\w)điều\s+\d+", re.IGNORECASE)


def has_citation(query: str) -> bool:
    """Câu hỏi gốc có viện dẫn số Điều (vd. "Điều 36", "khoản 2 điều 36") không.

    Không khớp "điều kiện lao động", "trong 3 điều kiện" hay "khoản 2" đơn lẻ.

    Args:
        query: Câu hỏi gốc của người dùng (không phải hypothetical document).
    """
    return _CITATION_PATTERN.search(unicodedata.normalize("NFC", query)) is not None


def citation_extras(sparse_hits: list[SearchHit]) -> list[Candidate]:
    """Top `CITATION_SPARSE_TOP_K` kết quả sparse (đã sắp theo điểm) thành candidate.

    Candidate chưa có metadata; `fill_missing` bổ sung sau khi gộp vào union.
    """
    # rrf_score chỉ có nghĩa với kết quả RRF; extras không qua RRF (điểm BM25
    # của hit không cùng thang) và không ai đọc trường này sau bước fusion.
    return [
        Candidate(chunk_id=hit.chunk_id, rrf_score=0.0)
        for hit in sparse_hits[:CITATION_SPARSE_TOP_K]
    ]
