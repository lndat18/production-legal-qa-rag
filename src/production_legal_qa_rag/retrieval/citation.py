"""Nhận diện câu hỏi viện dẫn Điều và chọn extras từ sparse (mục 8.1).

Với câu hỏi kiểu "Điều 36 khoản 2 quy định gì", chunk đáp án thường xếp cao ở
sparse nhưng thấp ở dense nên dễ bị RRF/MMR cắt; các extras đảm bảo top-k
sparse luôn vào union. Nguyên tắc nhận diện: ưu tiên không false positive hơn
là bắt hết.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import NamedTuple

from production_legal_qa_rag.retrieval.models import Candidate, SearchHit

logger = logging.getLogger(__name__)

CITATION_SPARSE_TOP_K = 10
MAX_CITATION_ARTICLES = 3
CITATION_EXTRAS_BUDGET = 16

_FLAGS = re.IGNORECASE
# (?<!\w): "điều" không dính vào cuối từ khác (vd. "chiều 5"); \d{1,3}(?!\d):
# "điều 2024" không khớp. Chữ số bắt buộc ngay sau "điều"/"điều thứ" nên
# "điều kiện", "điều khoản", "điều hành" không khớp.
_ARTICLE = re.compile(r"(?<!\w)(điều\s*(?:thứ\s+)?)(\d{1,3})(?!\d)", _FLAGS)
# Số nối sau một Điều: "Điều 3, 5 và 7", "Điều 3 đến Điều 5". Số kèm phần thập
# phân/nghìn ("5.000") không tính.
_CONTINUATION = re.compile(
    r"\s*(?:,|;|(?<!\w)(?:và|hoặc|hay|đến|tới)(?!\w))\s*"
    r"(?:(điều\s*(?:thứ\s+)?))?(\d{1,3})(?!\d|[.,]\d)",
    _FLAGS,
)
# Số nối đứng trước đơn vị thì là đại lượng, không phải số Điều ("Điều 3 và 5 người").
_UNIT_AFTER = re.compile(
    r"\s*(?:(?:tháng|ngày|năm|tuổi|người|lần|đồng|triệu)(?!\w)|%)", _FLAGS
)


class _Mention(NamedTuple):
    """Một lần nhắc số Điều: giá trị và khoảng ký tự cần xoá khi lọc sub-query."""

    number: int
    start: int
    end: int


def _find_mentions(text: str) -> list[_Mention]:
    """Mọi lần nhắc số Điều theo thứ tự xuất hiện (chưa dedupe, chưa cắt trần)."""
    mentions: list[_Mention] = []
    position = 0
    while (match := _ARTICLE.search(text, position)) is not None:
        mentions.append(_Mention(int(match.group(2)), match.start(), match.end(2)))
        position = match.end(2)
        while (cont := _CONTINUATION.match(text, position)) is not None:
            if _UNIT_AFTER.match(text, cont.end()):
                break
            start = cont.start(1) if cont.group(1) else cont.start(2)
            mentions.append(_Mention(int(cont.group(2)), start, cont.end(2)))
            position = cont.end(2)
    return mentions


def _distinct_numbers(mentions: list[_Mention]) -> list[int]:
    return list(dict.fromkeys(mention.number for mention in mentions))


def extract_citation_numbers(query: str) -> list[int]:
    """Các số Điều được viện dẫn trong câu hỏi gốc, theo thứ tự xuất hiện.

    Hỗ trợ "Điều 36", "điều36", "Điều thứ 5", "Điều 36.2" (lấy 36), danh sách
    "Điều 3, 5 và 7" và khoảng "Điều 3 đến Điều 5" (chỉ hai đầu mút). Không hỗ
    trợ "Đ.3", số La Mã, số bằng chữ. Dedupe giữ thứ tự, tối đa
    `MAX_CITATION_ARTICLES` (phần dư bỏ qua và log info).

    Args:
        query: Câu hỏi gốc của người dùng (không phải hypothetical document).
    """
    numbers = _distinct_numbers(_find_mentions(unicodedata.normalize("NFC", query)))
    if len(numbers) > MAX_CITATION_ARTICLES:
        logger.info(
            "Câu hỏi nhắc %d Điều, chỉ dùng %d Điều đầu: %s",
            len(numbers),
            MAX_CITATION_ARTICLES,
            numbers[MAX_CITATION_ARTICLES:],
        )
        numbers = numbers[:MAX_CITATION_ARTICLES]
    return numbers


def has_citation(query: str) -> bool:
    """Câu hỏi gốc có viện dẫn số Điều không (xem `extract_citation_numbers`)."""
    return bool(extract_citation_numbers(query))


def build_article_queries(query: str, numbers: list[int]) -> list[str]:
    """Sub-query cho từng Điều: câu gốc bỏ số Điều của các Điều khác.

    Số Điều của Điều khác (kể cả Điều vượt trần) bị bỏ kèm "điều"/"điều thứ" đứng trước nếu có. "khoản Y" đi kèm
    không bị xoá (không phân định chắc "khoản" thuộc Điều nào) — giới hạn đã
    biết, chấp nhận.

    Args:
        query: Câu hỏi gốc.
        numbers: Số Điều từ `extract_citation_numbers`.

    Returns:
        Danh sách cùng thứ tự `numbers`.
    """
    text = unicodedata.normalize("NFC", query)
    # Cả Điều vượt trần cũng bị bỏ: chúng là "Điều khác" gây nhiễu như còn lại.
    mentions = _find_mentions(text)
    queries: list[str] = []
    for number in numbers:
        pieces: list[str] = []
        cursor = 0
        for mention in mentions:
            if mention.number == number:
                continue
            pieces.append(text[cursor : mention.start])
            cursor = mention.end
        pieces.append(text[cursor:])
        queries.append(" ".join("".join(pieces).split()))
    return queries


def citation_extras(
    numbers: list[int],
    branch_b_sparse_hits: list[SearchHit],
    article_hits: dict[int, list[SearchHit]] | None = None,
) -> list[Candidate]:
    """Chọn extras đưa vào union cho câu hỏi viện dẫn.

    - 1 Điều (hoặc mọi lượt sparse phụ lỗi): `CITATION_SPARSE_TOP_K` hit đầu
      của sparse thô nhánh B.
    - n >= 2 Điều: top `floor(CITATION_EXTRAS_BUDGET / n)` hit của mỗi Điều,
      xen kẽ theo Điều, dedupe theo `chunk_id` (giữ vị trí sớm nhất).

    Candidate chưa có metadata; `fill_missing` bổ sung sau khi gộp vào union.

    Args:
        numbers: Số Điều nhận diện được.
        branch_b_sparse_hits: Sparse hit thô của nhánh B (đã sắp theo điểm).
        article_hits: Hit của lượt sparse phụ theo số Điều; Điều thiếu (lượt
            lỗi) bị bỏ qua.
    """
    if not numbers:
        return []
    usable = {n: article_hits[n] for n in numbers if article_hits and n in article_hits}
    if len(numbers) == 1 or not usable:
        hits = branch_b_sparse_hits[:CITATION_SPARSE_TOP_K]
    else:
        quota = CITATION_EXTRAS_BUDGET // len(numbers)
        hits = [
            usable[number][rank]
            for rank in range(quota)
            for number in numbers
            if number in usable and rank < len(usable[number])
        ]
    unique = dict.fromkeys(hit.chunk_id for hit in hits)
    # rrf_score chỉ có nghĩa với kết quả RRF; extras không qua RRF (điểm BM25
    # không cùng thang) và không ai đọc trường này sau bước fusion.
    return [Candidate(chunk_id=chunk_id, rrf_score=0.0) for chunk_id in unique]
