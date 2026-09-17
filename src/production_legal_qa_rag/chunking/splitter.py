"""Thuật toán cắt Khoản thành `Chunk` (mục 4, 5, 4.6).

`split_khoan()` là điểm vào cho 1 Khoản thật (cấu trúc Phần/Chương/Mục/Điều/
Khoản): áp dụng ngoại lệ bảng (mục 5.1) trước, sau đó tới thuật toán tách câu
dẫn + ghép "cận dưới" theo Điểm/câu (mục 4.2-4.5).

`split_implicit_khoan()` là điểm vào riêng cho 2 "Khoản ngầm định cấp văn
bản" (frontmatter/backmatter, mục 4.6) — không có Điểm, cắt bằng
`RecursiveCharacterTextSplitter` thay vì thuật toán Điểm/câu.

QUYẾT ĐỊNH THIẾT KẾ (spec không định nghĩa rạch ròi, xem báo cáo bàn giao):

- Mục 11 đòi hỏi literal "mọi chunk không có bảng đều <= MAX_TOKENS", không
  có ngoại lệ nào khác — chặt hơn "chấp nhận sai số nhỏ" của mục 4.4. Vì vậy
  khi 1 câu (đơn vị nhỏ nhất mục 4.4 định nghĩa) tự nó vẫn vượt `budget_hiệu_dụng`,
  thêm 1 tầng tách nữa theo dấu phẩy (`patterns.split_finer`, cùng thuật
  toán cận dưới + overlap) TRƯỚC khi chấp nhận vượt ngân sách — chỉ khi cả 2
  tầng đều không tách được nữa (đơn vị chỉ còn đúng 1 mệnh đề) mới thực sự
  giữ nguyên vượt ngân sách (fallback cuối cùng, cùng tinh thần ngoại lệ
  bảng ở mục 5.1: model tự truncate phần dư).
- Theo template breadcrumb literal ở mục 3 (không theo ví dụ rút gọn ngay
  sau đó), đoạn Điều luôn gồm cả tên điều ("Điều {n}. {tên điều}"); Chương/
  Mục/Phần chỉ gồm số hiệu, không kèm tên — vì template chỉ định nghĩa
  placeholder tên riêng cho Điều.
- Bảng công thức 1 hàng dạng HTML thô (``<table>``, xem `parser.py`/
  `tables.py`) được coi là bảng giống bảng pipe: `has_table=True`, giữ
  nguyên không cắt (mục 5.1) dù không phải bảng pipe literal.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from production_legal_qa_rag.chunking.models import Chunk, KhoanNode
from production_legal_qa_rag.chunking.patterns import (
    RE_DIEM,
    split_finer,
    split_sentences,
)
from production_legal_qa_rag.chunking.tables import standardize_table
from production_legal_qa_rag.chunking.tokenizer import count_tokens

# Các tầng tách dùng cho fallback mục 4.4 khi 1 đơn vị tự nó vượt ngân sách,
# thử lần lượt cho tới khi tách được > 1 phần (xem `_explode_oversized`).
_FALLBACK_SPLITTERS = (split_sentences, split_finer)

# ==========================================================================
# Đơn vị/nhóm nội bộ (mục 4.2-4.4)
# ==========================================================================


@dataclass
class _Unit:
    kind: str  # "point" | "sentence"
    label: str | None
    text: str
    is_fragment: bool = False  # True nếu là câu tách ra từ fallback mục 4.4


@dataclass
class _Group:
    text: str
    point_labels: list[str]


def _split_into_points(content: str) -> tuple[str, list[tuple[str, str]]]:
    """Tách nội dung Khoản thành (câu dẫn, [(nhãn Điểm, nội dung)]) (mục 4.2).

    Nếu Khoản không có Điểm gắn nhãn, danh sách điểm rỗng và toàn bộ nội
    dung nằm trong câu dẫn (khi đó không có khái niệm "câu dẫn" theo spec —
    caller phải tự kiểm tra `points` rỗng trước khi coi giá trị trả về đầu
    tiên là câu dẫn thật).
    """
    paragraphs = content.split("\n\n")
    preamble: list[str] = []
    points: list[tuple[str, str]] = []
    current_label: str | None = None
    current_parts: list[str] = []

    for paragraph in paragraphs:
        match = RE_DIEM.match(paragraph.strip())
        if match:
            if current_label is not None:
                points.append((current_label, "\n\n".join(current_parts)))
            current_label = match.group(1)
            current_parts = [match.group(2)]
        elif current_label is not None:
            current_parts.append(paragraph)
        else:
            preamble.append(paragraph)

    if current_label is not None:
        points.append((current_label, "\n\n".join(current_parts)))

    return "\n\n".join(preamble).strip(), points


def _make_group(units: list[_Unit]) -> _Group:
    use_space = bool(units) and (units[0].is_fragment or units[0].kind == "sentence")
    joiner = " " if use_space else "\n\n"
    text = joiner.join(unit.text for unit in units)
    labels = list(
        dict.fromkeys(
            unit.label for unit in units if unit.kind == "point" and unit.label
        )
    )
    return _Group(text=text, point_labels=labels)


def _pack_with_overlap(fragments: list[_Unit], budget: int, tier: int) -> list[_Group]:
    """Cận dưới ở cấp câu, cho phép overlap 1 câu cuối (mục 4.4).

    `tier` là tầng tách hiện tại trong `_FALLBACK_SPLITTERS` — truyền tiếp
    cho `_explode_oversized` khi 1 fragment vẫn tự nó vượt ngân sách, để thử
    tầng tách kế tiếp thay vì chấp nhận vượt ngân sách ngay.
    """
    groups: list[_Group] = []
    current: list[_Unit] = []
    current_tokens = 0

    for fragment in fragments:
        tokens = count_tokens(fragment.text)
        if tokens > budget:
            if current:
                groups.append(_make_group(current))
                current, current_tokens = [], 0
            groups.extend(_explode_oversized(fragment, budget, tier + 1))
            continue
        if current and current_tokens + tokens > budget:
            groups.append(_make_group(current))
            overlap = current[-1]
            overlap_tokens = count_tokens(overlap.text)
            if overlap_tokens + tokens <= budget:
                current = [overlap, fragment]
                current_tokens = overlap_tokens + tokens
            else:
                current = [fragment]
                current_tokens = tokens
            continue
        current.append(fragment)
        current_tokens += tokens

    if current:
        groups.append(_make_group(current))
    return groups


def _explode_oversized(unit: _Unit, budget: int, tier: int = 0) -> list[_Group]:
    """1 đơn vị tự nó vượt ngân sách: thử lần lượt các tầng tách (mục 4.4).

    Thử `_FALLBACK_SPLITTERS[tier]` (câu, rồi tới mệnh đề theo dấu phẩy);
    nếu tầng đó không tách được (trả về đúng 1 phần, tức đơn vị đã nguyên
    vẹn), thử tầng kế tiếp. Hết tầng mà vẫn không tách được thì chấp nhận
    giữ nguyên vượt ngân sách (không có "tầng dưới" nào được spec định
    nghĩa).
    """
    if tier >= len(_FALLBACK_SPLITTERS):
        return [_make_group([unit])]

    pieces = _FALLBACK_SPLITTERS[tier](unit.text)
    if len(pieces) <= 1:
        return _explode_oversized(unit, budget, tier + 1)

    fragments = [
        _Unit(kind=unit.kind, label=unit.label, text=piece, is_fragment=True)
        for piece in pieces
    ]
    return _pack_with_overlap(fragments, budget, tier)


def _pack_units(units: list[_Unit], budget: int) -> list[_Group]:
    """Thuật toán ghép "cận dưới" (mục 4.3), gồm cả fallback mục 4.4."""
    groups: list[_Group] = []
    current: list[_Unit] = []
    current_tokens = 0

    for unit in units:
        tokens = count_tokens(unit.text)
        if tokens > budget:
            if current:
                groups.append(_make_group(current))
                current, current_tokens = [], 0
            groups.extend(_explode_oversized(unit, budget))
            continue
        if current and current_tokens + tokens > budget:
            groups.append(_make_group(current))
            current, current_tokens = [unit], tokens
            continue
        current.append(unit)
        current_tokens += tokens

    if current:
        groups.append(_make_group(current))
    return groups


# ==========================================================================
# Breadcrumb & chunk_id
# ==========================================================================


def _khoan_base_breadcrumb(khoan: KhoanNode) -> str:
    if khoan.khoan_number:
        return f"{khoan.breadcrumb_prefix} - Khoản {khoan.khoan_number}"
    return khoan.breadcrumb_prefix


def _compose_split_breadcrumb(
    base: str, point_labels: list[str], index: int, total: int
) -> str:
    if total <= 1:
        return base
    if point_labels:
        return f"{base} - Điểm {', '.join(point_labels)} (phần {index}/{total})"
    return f"{base} (phần {index}/{total})"


def _make_chunk_id(source_document: str, breadcrumb: str) -> str:
    """Chunk id deterministic, sinh từ `source_document` + breadcrumb đầy đủ
    (mục 2: breadcrumb đã bao gồm cả nhãn Điểm và chỉ số "(phần i/n)" khi có,
    nên băm `source_document + breadcrumb` thoả đúng yêu cầu "sinh từ
    source_document + đường dẫn breadcrumb + chỉ số phần (nếu bị cắt)").
    """
    digest = hashlib.sha256(f"{source_document}::{breadcrumb}".encode())
    return digest.hexdigest()


# ==========================================================================
# Điểm vào — Khoản thật (mục 4, 5)
# ==========================================================================


def _build_table_chunk(khoan: KhoanNode, base: str, source_document: str) -> Chunk:
    """Khoản có bảng: giữ nguyên 1 chunk, không cắt (mục 5.1-5.4)."""
    standardization = standardize_table(khoan.raw_table or "")
    narrative = khoan.content.strip()
    content = f"{narrative}\n\n{standardization}" if narrative else standardization

    return Chunk(
        chunk_id=_make_chunk_id(source_document, base),
        source_document=source_document,
        breadcrumb=base,
        content=content,
        token_count=count_tokens(content),
        has_table=True,
        raw_table=khoan.raw_table,
        standardization_table=standardization,
        is_split=False,
    )


def split_khoan(
    khoan: KhoanNode, *, source_document: str, max_tokens: int
) -> list[Chunk]:
    """Cắt 1 `KhoanNode` (Khoản thật) thành 1 hoặc nhiều `Chunk` (mục 4, 5).

    Args:
        khoan: Khoản đã parse (kèm breadcrumb tới Điều, nội dung thô).
        source_document: Tên văn bản đầy đủ (mục 2) chứa Khoản.
        max_tokens: Ngân sách token tối đa cho 1 chunk (`EmbeddingSettings.max_tokens`).

    Returns:
        Danh sách `Chunk`, luôn có ít nhất 1 phần tử.
    """
    base = _khoan_base_breadcrumb(khoan)

    if khoan.has_table:
        return [_build_table_chunk(khoan, base, source_document)]

    full_tokens = count_tokens(khoan.content)
    if full_tokens <= max_tokens:
        return [
            Chunk(
                chunk_id=_make_chunk_id(source_document, base),
                source_document=source_document,
                breadcrumb=base,
                content=khoan.content,
                token_count=full_tokens,
                has_table=False,
                is_split=False,
            )
        ]

    preamble_text, points = _split_into_points(khoan.content)

    if points:
        # Câu dẫn (mục 4.2) = đoạn đứng trước nhãn Điểm đầu tiên, tách riêng
        # -- KHÔNG đưa vào danh sách đơn vị cần đóng gói (mục 4.3).
        preamble = preamble_text or None
        units = [_Unit(kind="point", label=label, text=text) for label, text in points]
    else:
        # Không có Điểm gắn nhãn: không có khái niệm câu dẫn -- toàn bộ nội
        # dung tách thẳng theo câu (mục 4.2).
        preamble = None
        units = [
            _Unit(kind="sentence", label=None, text=sentence)
            for sentence in split_sentences(khoan.content)
        ]

    budget = max_tokens - count_tokens(preamble) if preamble else max_tokens
    groups = _pack_units(units, budget)
    total = len(groups)
    is_split = total > 1

    chunks: list[Chunk] = []
    for index, group in enumerate(groups, start=1):
        breadcrumb = _compose_split_breadcrumb(base, group.point_labels, index, total)
        # Câu dẫn (nếu có) được ghép vào ĐẦU content của MỌI chunk con sinh ra
        # từ Khoản này (mục 4.5) -- kể cả khi chỉ có đúng 1 chunk con.
        content = f"{preamble}\n\n{group.text}" if preamble else group.text
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(source_document, breadcrumb),
                source_document=source_document,
                breadcrumb=breadcrumb,
                content=content,
                token_count=count_tokens(content),
                has_table=False,
                is_split=is_split,
                split_index=index if is_split else None,
                split_total=total if is_split else None,
            )
        )
    return chunks


# ==========================================================================
# Điểm vào — frontmatter/backmatter (mục 4.6)
# ==========================================================================


def split_implicit_khoan(
    content: str, *, breadcrumb_prefix: str, source_document: str, max_tokens: int
) -> list[Chunk]:
    """Cắt 1 "Khoản ngầm định cấp văn bản" (frontmatter/backmatter, mục 4.6).

    Không dùng thuật toán Điểm/câu mục 4.2-4.5 (không có cấu trúc Điểm, có
    thể trích dẫn xen kẽ nhiều luật khác nhau) -- dùng
    `RecursiveCharacterTextSplitter` với `length_function=count_tokens`
    (đếm token PhoBERT thật) khi vượt `max_tokens`.

    Args:
        content: Nội dung thô của vùng frontmatter/backmatter.
        breadcrumb_prefix: `{source_document}` (frontmatter) hoặc
            `{source_document} - Chú thích sửa đổi (cuối văn bản)` (backmatter).
        source_document: Tên văn bản đầy đủ (mục 2).
        max_tokens: Ngân sách token tối đa cho 1 chunk.

    Returns:
        Danh sách `Chunk`, luôn có ít nhất 1 phần tử.
    """
    content = content.strip()
    token_count = count_tokens(content)
    if token_count <= max_tokens:
        return [
            Chunk(
                chunk_id=_make_chunk_id(source_document, breadcrumb_prefix),
                source_document=source_document,
                breadcrumb=breadcrumb_prefix,
                content=content,
                token_count=token_count,
                has_table=False,
                is_split=False,
            )
        ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=0,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    pieces = [piece.strip() for piece in splitter.split_text(content) if piece.strip()]
    total = len(pieces)

    chunks: list[Chunk] = []
    for index, piece in enumerate(pieces, start=1):
        breadcrumb = (
            f"{breadcrumb_prefix} (phần {index}/{total})"
            if total > 1
            else breadcrumb_prefix
        )
        is_split = total > 1
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(source_document, breadcrumb),
                source_document=source_document,
                breadcrumb=breadcrumb,
                content=piece,
                token_count=count_tokens(piece),
                has_table=False,
                is_split=is_split,
                split_index=index if is_split else None,
                split_total=total if is_split else None,
            )
        )
    return chunks
