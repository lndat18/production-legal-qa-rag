"""Nhận diện câu hỏi viện dẫn Điều, token cấu trúc và chọn extras từ sparse (mục 6.2, 8.1).

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

from production_legal_qa_rag.retrieval.models import Candidate, SearchHit

logger = logging.getLogger(__name__)

CITATION_SPARSE_TOP_K = 10
MAX_CITATION_KHOANS = 3

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


def _find_numbers(text: str) -> list[int]:
    """Mọi số Điều theo thứ tự xuất hiện (chưa dedupe)."""
    numbers: list[int] = []
    position = 0
    while (match := _ARTICLE.search(text, position)) is not None:
        position = match.end(2)
        # Số Điều ngay trước đơn vị ("điều 5 tháng") là đại lượng, không phải
        # viện dẫn; siết vì token cấu trúc và extras làm nhận nhầm tốn kém hơn.
        if _UNIT_AFTER.match(text, position):
            continue
        numbers.append(int(match.group(2)))
        while (cont := _CONTINUATION.match(text, position)) is not None:
            if _UNIT_AFTER.match(text, cont.end()):
                break
            numbers.append(int(cont.group(2)))
            position = cont.end(2)
    return numbers


def extract_citation_numbers(query: str) -> list[int]:
    """Các số Điều được viện dẫn trong câu hỏi gốc, theo thứ tự xuất hiện.

    Hỗ trợ "Điều 36", "điều36", "Điều thứ 5", "Điều 36.2" (lấy 36), danh sách
    "Điều 3, 5 và 7" và khoảng "Điều 3 đến Điều 5" (chỉ hai đầu mút). Không hỗ
    trợ "Đ.3", số La Mã, số bằng chữ. Số Điều ngay trước đơn vị (tháng, ngày,
    năm, tuổi, lần, %, đồng, triệu) không tính. Dedupe giữ thứ tự, không có
    trần (câu nhiều Điều nằm ngoài phạm vi tối ưu, mục 1).

    Args:
        query: Câu hỏi gốc của người dùng (không phải hypothetical document).
    """
    return list(dict.fromkeys(_find_numbers(unicodedata.normalize("NFC", query))))


def has_citation(query: str) -> bool:
    """Câu hỏi gốc có viện dẫn số Điều không (xem `extract_citation_numbers`)."""
    return bool(extract_citation_numbers(query))


def citation_extras(branch_b_sparse_hits: list[SearchHit]) -> list[Candidate]:
    """Extras cho câu hỏi viện dẫn: `CITATION_SPARSE_TOP_K` hit đầu sparse thô nhánh B.

    Áp cho mọi số Điều n >= 1 (không xử lý riêng từng Điều). Candidate chưa có
    metadata; `fill_missing` bổ sung sau khi gộp vào union.

    Args:
        branch_b_sparse_hits: Sparse hit thô của nhánh B (đã sắp theo điểm).
    """
    unique = dict.fromkeys(
        hit.chunk_id for hit in branch_b_sparse_hits[:CITATION_SPARSE_TOP_K]
    )
    # rrf_score chỉ có nghĩa với kết quả RRF; extras không qua RRF (điểm BM25
    # không cùng thang) và không ai đọc trường này sau bước fusion.
    return [Candidate(chunk_id=chunk_id, rrf_score=0.0) for chunk_id in unique]


def extract_citation_khoans(query: str) -> list[int]:
    """Các số Khoản trong câu hỏi gốc ("khoản 2"), dedupe giữ thứ tự, tối đa
    `MAX_CITATION_KHOANS`.

    Chỉ có nghĩa khi câu hỏi đã có số Điều (token cấu trúc).
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
# (`điều_khoản`, ...); token văn bản có tiền tố `vb_`.


def _dieu_term(number: int) -> str:
    return f"điều_{number}"


def _khoan_term(number: int) -> str:
    return f"khoản_{number}"


def _pair_term(dieu: int, khoan: int) -> str:
    return f"điều_{dieu}_khoản_{khoan}"


def _doc_term(key: str, dieu: int | None = None, khoan: int | None = None) -> str:
    """`vb_X`, `vb_X_điều_N`, `vb_X_điều_N_khoản_M`; tiền tố `vb_` tránh va chạm pyvi."""
    term = f"vb_{key}"
    if dieu is not None:
        term += f"_điều_{dieu}"
        if khoan is not None:
            term += f"_khoản_{khoan}"
    return term


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


def breadcrumb_structural_terms(
    breadcrumb: str, source_document: str | None = None
) -> list[str]:
    """Token cấu trúc phía document, chỉ những token có trong breadcrumb.

    Kèm token theo văn bản (`vb_X`, `vb_X_điều_N`, `vb_X_điều_N_khoản_M`) khi
    `source_document` có trong bảng `DOCUMENTS`.
    """
    ref = parse_breadcrumb(breadcrumb)
    terms: list[str] = []
    if ref.dieu is not None:
        terms.append(_dieu_term(ref.dieu))
    if ref.khoan is not None:
        terms.append(_khoan_term(ref.khoan))
    if ref.dieu is not None and ref.khoan is not None:
        terms.append(_pair_term(ref.dieu, ref.khoan))

    document = DOCUMENTS.get(source_document) if source_document else None
    if document is not None:
        terms.append(_doc_term(document.key))
        if ref.dieu is not None:
            terms.append(_doc_term(document.key, ref.dieu))
            if ref.khoan is not None:
                terms.append(_doc_term(document.key, ref.dieu, ref.khoan))
    return terms


def structural_terms(
    numbers: Sequence[int], khoans: Sequence[int], doc: str | None = None
) -> list[str]:
    """Token cấu trúc phía query: mỗi Điều, mỗi Khoản, mỗi cặp (Điều, Khoản).

    "Khoản M" không kèm số Điều không sinh token. Khi `doc` (key văn bản duy
    nhất trong câu hỏi) có mặt, thêm `vb_X`, `vb_X_điều_N` cho mỗi Điều và
    `vb_X_điều_N_khoản_M` cho mỗi cặp.
    """
    if not numbers:
        return []
    terms = (
        [_dieu_term(n) for n in numbers]
        + [_khoan_term(m) for m in khoans]
        + [_pair_term(n, m) for n in numbers for m in khoans]
    )
    if doc is not None:
        terms += (
            [_doc_term(doc)]
            + [_doc_term(doc, n) for n in numbers]
            + [_doc_term(doc, n, m) for n in numbers for m in khoans]
        )
    return terms


# ------------------------------------------------------------- văn bản (mục 6.2)


def _fold(text: str) -> str:
    """NFC + lowercase + bỏ dấu (kể cả đ -> d) + gộp khoảng trắng."""
    decomposed = unicodedata.normalize("NFD", unicodedata.normalize("NFC", text))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.lower().replace("đ", "d").split())


class DocumentEntry(NamedTuple):
    """Văn bản trong corpus: key ASCII làm token và các alias (đã chuẩn hoá)."""

    key: str
    aliases: tuple[str, ...]


def _entry(key: str, *aliases: str) -> DocumentEntry:
    return DocumentEntry(key, tuple(_fold(alias) for alias in aliases))


# `source_document` nguyên văn trong data/chunks -> key + alias. Alias chỉ gồm
# tên có tiền tố luật/bộ luật/nghị định hoặc chữ viết tắt; tên chủ đề trơn
# ("bảo hiểm xã hội", "lao động") không phải alias vì thường chỉ chủ đề. Thêm/đổi
# văn bản trong corpus phải cập nhật bảng này.
DOCUMENTS: dict[str, DocumentEntry] = {
    "BỘ LUẬT LAO ĐỘNG": _entry("blld", "bộ luật lao động", "luật lao động", "blld"),
    "LUẬT BẢO HIỂM XÃ HỘI": _entry("bhxh", "luật bảo hiểm xã hội", "luật bhxh"),
    "LUẬT BẢO HIỂM Y TẾ": _entry("bhyt", "luật bảo hiểm y tế", "luật bhyt"),
    "LUẬT THUẾ THU NHẬP CÁ NHÂN": _entry(
        "tncn",
        "luật thuế thu nhập cá nhân",
        "luật thuế tncn",
        "thuế thu nhập cá nhân",
        "thuế tncn",
    ),
    "NGHỊ ĐỊNH QUY ĐỊNH MỨC LƯƠNG TỐI THIỂU ĐỐI VỚI NGƯỜI LAO ĐỘNG LÀM VIỆC THEO HỢP ĐỒNG LAO ĐỘNG": _entry(
        "nd_luong",
        "nghị định lương tối thiểu",
        "nghị định mức lương tối thiểu",
        "nghị định quy định mức lương tối thiểu",
    ),
    "NGHỊ ĐỊNH QUY ĐỊNH CHI TIẾT VÀ HƯỚNG DẪN THI HÀNH MỘT SỐ ĐIỀU CỦA BỘ LUẬT LAO ĐỘNG VỀ ĐIỀU KIỆN LAO ĐỘNG VÀ QUAN HỆ LAO ĐỘNG": _entry(
        "nd_dkld",
        "nghị định điều kiện lao động",
        "nghị định quan hệ lao động",
        "nghị định hướng dẫn bộ luật lao động",
        "nghị định về điều kiện lao động và quan hệ lao động",
    ),
}

_ALIAS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (entry.key, re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)"))
    for entry in DOCUMENTS.values()
    for alias in entry.aliases
]


def detect_document(query: str) -> str | None:
    """Key của văn bản duy nhất được nêu tên trong câu hỏi gốc, hoặc `None`.

    So khớp alias sau khi chuẩn hoá (NFC, lowercase, bỏ dấu). Khi các khoảng
    khớp chồng lấn, alias dài thắng ("nghị định hướng dẫn bộ luật lao động"
    thắng "bộ luật lao động" nằm trong nó). Không nêu văn bản, hoặc nêu từ 2
    văn bản khác nhau (mơ hồ), trả `None`.
    """
    text = _fold(query)
    spans: list[tuple[int, int, str]] = []
    for key, pattern in _ALIAS_PATTERNS:
        spans.extend((m.start(), m.end(), key) for m in pattern.finditer(text))
    accepted: list[tuple[int, int, str]] = []
    for start, end, key in sorted(spans, key=lambda s: s[0] - s[1]):  # dài trước
        if all(end <= a_start or start >= a_end for a_start, a_end, _ in accepted):
            accepted.append((start, end, key))
    keys = {key for _, _, key in accepted}
    return next(iter(keys)) if len(keys) == 1 else None
