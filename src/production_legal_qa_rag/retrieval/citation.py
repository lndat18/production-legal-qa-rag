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
from collections.abc import Sequence
from typing import NamedTuple

from production_legal_qa_rag.retrieval.models import (
    Candidate,
    RetrievedChunk,
    SearchHit,
)

logger = logging.getLogger(__name__)

CITATION_SPARSE_TOP_K = 10
MAX_CITATION_ARTICLES = 3
CITATION_EXTRAS_BUDGET = 16
MAX_CITATION_KHOANS = 3
PIN_PER_ARTICLE = 2

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
# Không gồm "người": "Điều 36 người lao động được quyền gì" là viện dẫn thật.
_UNIT_AFTER = re.compile(
    r"\s*(?:(?:tháng|ngày|năm|tuổi|lần|đồng|triệu)(?!\w)|%)", _FLAGS
)
_KHOAN = re.compile(r"(?<!\w)khoản\s*(\d{1,3})(?!\d)", _FLAGS)
_PHAN_SUFFIX = re.compile(r"\s*\(phần \d+/\d+\)\s*$")
_BREADCRUMB_DIEU = re.compile(r"Điều (\d+)(?:\.|$)")
_BREADCRUMB_KHOAN = re.compile(r"Khoản (\d+)$")


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
        position = match.end(2)
        # Số Điều ngay trước đơn vị ("điều 5 tháng") là đại lượng, không phải
        # viện dẫn; siết vì token cấu trúc và ghim làm nhận nhầm tốn kém hơn.
        if _UNIT_AFTER.match(text, position):
            continue
        mentions.append(_Mention(int(match.group(2)), match.start(), position))
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
    trợ "Đ.3", số La Mã, số bằng chữ. Số Điều ngay trước đơn vị (tháng, ngày,
    năm, tuổi, lần, %, đồng, triệu) không tính. Dedupe giữ thứ tự, tối đa
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


def extract_citation_khoans(query: str) -> list[int]:
    """Các số Khoản trong câu hỏi gốc ("khoản 2"), dedupe giữ thứ tự, tối đa
    `MAX_CITATION_KHOANS`.

    Chỉ có nghĩa khi câu hỏi đã có số Điều (token cấu trúc, ghim).
    """
    khoans = list(
        dict.fromkeys(
            int(m.group(1))
            for m in _KHOAN.finditer(unicodedata.normalize("NFC", query))
        )
    )
    if len(khoans) > MAX_CITATION_KHOANS:
        logger.info(
            "Câu hỏi nhắc %d Khoản, chỉ dùng %d Khoản đầu: %s",
            len(khoans),
            MAX_CITATION_KHOANS,
            khoans[MAX_CITATION_KHOANS:],
        )
        khoans = khoans[:MAX_CITATION_KHOANS]
    return khoans


# ------------------------------------------------------------- token cấu trúc
# Một bộ hàm định dạng duy nhất cho cả phía document (breadcrumb) và phía query
# để hai phía không thể lệch nhau. Hậu tố số nên không va chạm token pyvi
# (`điều_khoản`, ...).


def _dieu_term(number: int) -> str:
    return f"điều_{number}"


def _khoan_term(number: int) -> str:
    return f"khoản_{number}"


def _pair_term(dieu: int, khoan: int) -> str:
    return f"điều_{dieu}_khoản_{khoan}"


class BreadcrumbRef(NamedTuple):
    """Số Điều/Khoản parse từ breadcrumb; `None` khi breadcrumb không có."""

    dieu: int | None
    khoan: int | None


def parse_breadcrumb(breadcrumb: str) -> BreadcrumbRef:
    """Lấy Điều N và Khoản M từ breadcrumb `... - Điều N. Tên - Khoản M [- Điểm ...]`.

    Parse theo từng đoạn phân tách bằng " - " nên tên văn bản ("... MỘT SỐ ĐIỀU
    CỦA ...") không bị nhận nhầm. "(phần i/n)" cuối chuỗi và "Điểm ..." bị bỏ
    qua. Số Điều/Khoản có hậu tố chữ ("Điều 48a", "Khoản 3a") không lấy được
    vì câu hỏi chỉ nhận diện số thuần.
    """
    dieu: int | None = None
    khoan: int | None = None
    for segment in _PHAN_SUFFIX.sub("", breadcrumb).split(" - "):
        segment = segment.strip()
        if dieu is None and (m := _BREADCRUMB_DIEU.match(segment)):
            dieu = int(m.group(1))
        elif (
            dieu is not None
            and khoan is None
            and (m := _BREADCRUMB_KHOAN.match(segment))
        ):
            khoan = int(m.group(1))
    return BreadcrumbRef(dieu, khoan)


def breadcrumb_structural_terms(breadcrumb: str) -> list[str]:
    """Token cấu trúc phía document, chỉ những token có trong breadcrumb."""
    ref = parse_breadcrumb(breadcrumb)
    terms: list[str] = []
    if ref.dieu is not None:
        terms.append(_dieu_term(ref.dieu))
    if ref.khoan is not None:
        terms.append(_khoan_term(ref.khoan))
    if ref.dieu is not None and ref.khoan is not None:
        terms.append(_pair_term(ref.dieu, ref.khoan))
    return terms


def structural_terms(numbers: Sequence[int], khoans: Sequence[int]) -> list[str]:
    """Token cấu trúc phía query: mỗi Điều, mỗi Khoản, mỗi cặp (Điều, Khoản).

    "Khoản M" không kèm số Điều không sinh token. Tối đa 3 + 3 + 9 = 15 token.
    """
    if not numbers:
        return []
    return (
        [_dieu_term(n) for n in numbers]
        + [_khoan_term(m) for m in khoans]
        + [_pair_term(n, m) for n in numbers for m in khoans]
    )


# ------------------------------------------------------------- ghim khớp chính xác


def pin_exact_matches(
    ranked: Sequence[RetrievedChunk],
    numbers: Sequence[int],
    khoans: Sequence[int],
    *,
    final_top_k: int,
) -> list[RetrievedChunk]:
    """Đưa chunk khớp chính xác Điều/Khoản lên đầu danh sách đã xếp hạng (mục 8.2).

    Với mỗi Điều được hỏi: chunk có `Điều N.` trong breadcrumb; nếu câu hỏi có
    Khoản và tồn tại chunk khớp cả Khoản thì chỉ giữ chúng. Mỗi Điều tối đa
    `PIN_PER_ARTICLE` chunk, tổng tối đa `final_top_k - 1`, chọn xen kẽ theo
    Điều. Chunk ghim giữ thứ tự sẵn có của `ranked`; `rerank_score` không đổi.

    Args:
        ranked: Toàn bộ chunk đã xếp hạng (theo rerank hoặc fallback).
        numbers: Số Điều được hỏi.
        khoans: Số Khoản được hỏi.
        final_top_k: Số chunk trả về cuối cùng (để tính trần ghim).

    Returns:
        Danh sách cùng phần tử với `ranked`: chunk ghim trước, phần còn lại
        giữ thứ tự cũ.
    """
    refs = [parse_breadcrumb(chunk.breadcrumb) for chunk in ranked]
    per_article: list[list[int]] = []
    for number in numbers:
        indices = [i for i, ref in enumerate(refs) if ref.dieu == number]
        if khoans:
            exact = [i for i in indices if refs[i].khoan in khoans]
            indices = exact or indices
        per_article.append(indices)

    limit = final_top_k - 1
    pinned: set[int] = set()
    for rank in range(PIN_PER_ARTICLE):
        for indices in per_article:
            if len(pinned) < limit and rank < len(indices):
                pinned.add(indices[rank])
    order = sorted(pinned) + [i for i in range(len(ranked)) if i not in pinned]
    return [ranked[i] for i in order]
