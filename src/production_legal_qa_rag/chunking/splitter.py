"""Thuật toán cắt Khoản thành `Chunk` (mục 4, 5).

`split_khoan()` là điểm vào duy nhất: áp dụng ngoại lệ bảng (mục 5.1) trước,
sau đó mới tới thuật toán cắt theo Điểm/câu (mục 4.2-4.4) và lan truyền câu
phủ định vào breadcrumb/`negation_note` (mục 4.5).

QUYẾT ĐỊNH THIẾT KẾ (spec không định nghĩa rạch ròi, xem báo cáo bàn giao):

- `negation_note` được tính cho **mọi** Khoản, kể cả khi Khoản không bị cắt
  (``token_count <= MAX_TOKENS``, mục 4.1 giữ nguyên 1 chunk). Đọc literal
  mục 4.5 thì luồng phát hiện phủ định chỉ nằm trong nhánh cắt (mục 4.2-4.5,
  cần "đơn vị #0"), nhưng ví dụ xác nhận thủ công ở mục 11
  (`Luật bảo hiểm y tế.md` dòng 58 — Khoản 3 Điều 1, 1 câu, không có Điểm,
  không vượt `MAX_TOKENS`) chỉ khớp được nếu tổng quát hoá: "đoạn mở đầu" =
  toàn bộ nội dung Khoản khi Khoản không có Điểm gắn nhãn, áp dụng cho mọi
  Khoản chứ không riêng luồng cắt.
- Breadcrumb chỉ được nối thêm "- Điểm ...", "(phần i/n)", câu phủ định khi
  Khoản THỰC SỰ sinh ra nhiều hơn 1 chunk (đúng theo literal mục 3: "Khi
  Khoản bị cắt nhỏ..."); Khoản giữ nguyên 1 chunk chỉ có `negation_note`
  được gán, breadcrumb không đổi.
- Theo template breadcrumb literal ở mục 3 (không theo ví dụ rút gọn ngay
  sau đó), đoạn Điều luôn gồm cả tên điều ("Điều {n}. {tên điều}"); Chương/
  Mục/Phần chỉ gồm số hiệu, không kèm tên — vì template chỉ định nghĩa
  placeholder tên riêng cho Điều.
- Mục 11 đòi hỏi literal "mọi chunk không có bảng đều <= MAX_TOKENS", không
  có ngoại lệ nào khác — chặt hơn "chấp nhận sai số nhỏ" của mục 4.4. Vì vậy
  khi 1 câu (đơn vị nhỏ nhất mục 4.4 định nghĩa) tự nó vẫn vượt `MAX_TOKENS`,
  thêm 1 tầng tách nữa theo dấu phẩy (`patterns.split_finer`, cùng thuật
  toán cận dưới + overlap) TRƯỚC khi chấp nhận vượt ngân sách — chỉ khi cả 2
  tầng đều không tách được nữa (đơn vị chỉ còn đúng 1 mệnh đề) mới thực sự
  giữ nguyên vượt ngân sách (fallback cuối cùng, cùng tinh thần ngoại lệ
  bảng ở mục 5.1: model tự truncate phần dư).
- Bảng công thức 1 hàng dạng HTML thô (``<table>``, xem `parser.py`/
  `tables.py`) được coi là bảng giống bảng pipe: `has_table=True`, giữ
  nguyên không cắt (mục 5.1) dù không phải bảng pipe literal.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from production_legal_qa_rag.chunking.models import Chunk, KhoanNode
from production_legal_qa_rag.chunking.patterns import (
    RE_DIEM,
    RE_NEGATION,
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
    kind: str  # "unit0" | "point" | "sentence"
    label: str | None
    text: str
    is_fragment: bool = False  # True nếu là câu tách ra từ fallback mục 4.4


@dataclass
class _Group:
    text: str
    point_labels: list[str]


def _split_into_points(content: str) -> tuple[str, list[tuple[str, str]]]:
    """Tách nội dung Khoản thành (đoạn mở đầu, [(nhãn Điểm, nội dung)]).

    Nếu Khoản không có Điểm gắn nhãn, danh sách điểm rỗng và toàn bộ nội
    dung nằm trong đoạn mở đầu (mục 4.2).
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


def _find_negation_sentence(preamble: str) -> str | None:
    """Trích câu chứa từ khoá phủ định trong đoạn mở đầu, nếu có (mục 4.5)."""
    if not preamble or RE_NEGATION.search(preamble) is None:
        return None
    for sentence in split_sentences(preamble):
        if RE_NEGATION.search(sentence):
            return sentence
    return preamble.strip()


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


def _pack_with_overlap(
    fragments: list[_Unit], max_tokens: int, tier: int
) -> list[_Group]:
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
        if tokens > max_tokens:
            if current:
                groups.append(_make_group(current))
                current, current_tokens = [], 0
            groups.extend(_explode_oversized(fragment, max_tokens, tier + 1))
            continue
        if current and current_tokens + tokens > max_tokens:
            groups.append(_make_group(current))
            overlap = current[-1]
            overlap_tokens = count_tokens(overlap.text)
            if overlap_tokens + tokens <= max_tokens:
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


def _explode_oversized(unit: _Unit, max_tokens: int, tier: int = 0) -> list[_Group]:
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
        return _explode_oversized(unit, max_tokens, tier + 1)

    fragments = [
        _Unit(kind=unit.kind, label=unit.label, text=piece, is_fragment=True)
        for piece in pieces
    ]
    return _pack_with_overlap(fragments, max_tokens, tier)


def _pack_units(units: list[_Unit], max_tokens: int) -> list[_Group]:
    """Thuật toán ghép "cận dưới" (mục 4.3), gồm cả fallback mục 4.4."""
    groups: list[_Group] = []
    current: list[_Unit] = []
    current_tokens = 0

    for unit in units:
        tokens = count_tokens(unit.text)
        if tokens > max_tokens:
            if current:
                groups.append(_make_group(current))
                current, current_tokens = [], 0
            groups.extend(_explode_oversized(unit, max_tokens))
            continue
        if current and current_tokens + tokens > max_tokens:
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
    base: str,
    point_labels: list[str],
    index: int,
    total: int,
    negation_sentence: str | None,
) -> str:
    if total <= 1:
        return base
    if point_labels:
        breadcrumb = f"{base} - Điểm {', '.join(point_labels)} (phần {index}/{total})"
    else:
        breadcrumb = f"{base} (phần {index}/{total})"
    if negation_sentence:
        breadcrumb = f"{breadcrumb} - {negation_sentence}"
    return breadcrumb


def _make_chunk_id(source_document: str, breadcrumb: str) -> str:
    """Chunk id deterministic, sinh từ `source_document` + breadcrumb đầy đủ.

    Breadcrumb đã bao gồm cả nhãn Điểm và chỉ số "(phần i/n)" khi có, nên
    băm `source_document + breadcrumb` thoả đúng yêu cầu mục 2: "sinh từ
    so_hieu + đường dẫn breadcrumb + chỉ số phần (nếu bị cắt)".
    """
    digest = hashlib.sha256(f"{source_document}::{breadcrumb}".encode())
    return digest.hexdigest()


# ==========================================================================
# Điểm vào
# ==========================================================================


def _build_table_chunk(khoan: KhoanNode, base: str, source_document: str) -> Chunk:
    """Khoản có bảng: giữ nguyên 1 chunk, không cắt (mục 5.1-5.4)."""
    standardization = standardize_table(khoan.raw_table or "")
    narrative = khoan.content.strip()
    content = f"{narrative}\n\n{standardization}" if narrative else standardization
    negation_sentence = _find_negation_sentence(narrative) if narrative else None

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
        negation_note=negation_sentence,
    )


def split_khoan(
    khoan: KhoanNode, *, source_document: str, max_tokens: int
) -> list[Chunk]:
    """Cắt 1 `KhoanNode` thành 1 hoặc nhiều `Chunk` (mục 4, 5).

    Args:
        khoan: Khoản đã parse (kèm breadcrumb tới Điều, nội dung thô).
        source_document: `so_hieu` (fallback tên file) của văn bản chứa Khoản.
        max_tokens: Ngân sách token tối đa cho 1 chunk (`EmbeddingSettings.max_tokens`).

    Returns:
        Danh sách `Chunk`, luôn có ít nhất 1 phần tử.
    """
    base = _khoan_base_breadcrumb(khoan)

    if khoan.has_table:
        return [_build_table_chunk(khoan, base, source_document)]

    preamble, points = _split_into_points(khoan.content)
    negation_sentence = _find_negation_sentence(preamble)

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
                negation_note=negation_sentence,
            )
        ]

    units: list[_Unit]
    if points:
        units = []
        if preamble:
            units.append(_Unit(kind="unit0", label=None, text=preamble))
        units.extend(
            _Unit(kind="point", label=label, text=text) for label, text in points
        )
    else:
        units = [
            _Unit(kind="sentence", label=None, text=sentence)
            for sentence in split_sentences(khoan.content)
        ]

    groups = _pack_units(units, max_tokens)
    total = len(groups)
    is_split = total > 1

    chunks: list[Chunk] = []
    for index, group in enumerate(groups, start=1):
        breadcrumb = _compose_split_breadcrumb(
            base,
            group.point_labels,
            index,
            total,
            negation_sentence if is_split else None,
        )
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(source_document, breadcrumb),
                source_document=source_document,
                breadcrumb=breadcrumb,
                content=group.text,
                token_count=count_tokens(group.text),
                has_table=False,
                is_split=is_split,
                split_index=index if is_split else None,
                split_total=total if is_split else None,
                negation_note=negation_sentence,
            )
        )
    return chunks
