"""Regex nhận diện cấu trúc heading Markdown, nhãn Điểm, dòng bảng và tách câu
dùng để cắt Khoản (chunking_spec.md mục 3, 4, 5, 10).

Heading mapping (Phần/Phụ Lục -> `#`, Chương -> `##`, Mục -> `###`,
Điều -> `####`, Khoản -> `#####`) do `formatting/` sinh ra theo
`formatting_spec.md` mục 3; các regex ở đây khớp lại đúng định dạng đó để
dựng `DocumentTree` trong `parser.py`. Không import từ `formatting/` — hai
package độc lập, chỉ tuân theo cùng 1 quy ước định dạng output/input.
"""

from __future__ import annotations

import re

# ==========================================================================
# HEADING CẤU TRÚC (Phần/Phụ Lục/Chương/Mục/Điều/Khoản)
# ==========================================================================

RE_HEADING = re.compile(r"^(#{1,5})\s+(.*)$")

ROMAN = r"[IVXLCDM]+"
# "Phần thứ nhất" dùng số thứ tự tiếng Việt — corpus hiện tại không có Phần
# nào, nhưng vẫn hỗ trợ để tổng quát theo formatting/patterns.py.
ORDINAL_WORD = r"nhất|hai|ba|tư|bốn|năm|sáu|bảy|tám|chín|mười"

RE_PHU_LUC = re.compile(r"^PHỤ\s+LỤC\b", re.IGNORECASE)
RE_PHAN = re.compile(
    rf"^Phần\s+(?:thứ\s+)?({ROMAN}|\d+|{ORDINAL_WORD})\b", re.IGNORECASE
)
RE_CHUONG = re.compile(rf"^Chương\s+({ROMAN}|\d+)\b", re.IGNORECASE)
RE_MUC = re.compile(r"^Mục\s+(\d+[a-zđ]?)\b", re.IGNORECASE)
RE_DIEU = re.compile(r"^Điều\s+(\d+[a-zđ]?)\s*\.\s*(.*)$")

# Heading Khoản chuẩn: "Khoản N" (không kèm nội dung).
RE_KHOAN_LABEL = re.compile(r"^Khoản\s+(\d+[a-zđ]?)$", re.IGNORECASE)

# Heading Khoản trong Phụ Lục, nội dung ngắn gộp thẳng vào heading:
# "N. Thành phố Hồ Chí Minh" — xem formatting/emitter.py `_emit_khoan`.
RE_KHOAN_MERGED = re.compile(r"^(\d+[a-zđ]?)\.\s+(.*)$")

# Điểm: chỉ dạng chữ cái + ")" — spec (mục 3, 10) chỉ định nghĩa nhãn chữ
# cái; dạng "-" mà formatting/ dùng riêng cho danh mục Phụ Lục KHÔNG được coi
# là ranh giới Điểm ở đây (quyết định thiết kế, xem báo cáo bàn giao).
RE_DIEM = re.compile(r"^([a-zđư])\)\s+(.*)$")


def is_structural_heading(level: int, text: str) -> bool:
    """Heading (đã bỏ `#`) có phải nhãn cấu trúc Phần/Phụ Lục/Chương/Mục/
    Điều/Khoản thật sự hay không, theo đúng cấp (`level`) của nó.

    Dùng để phân biệt heading `#` (H1) của **tên văn bản** trong frontmatter
    (chunking_spec.md mục 2, 4.6) — không khớp `RE_PHU_LUC`/`RE_PHAN` — với
    heading `#` thật của Phần/Phụ Lục cấu trúc. `parser.py` dùng hàm này để
    xác định biên frontmatter: block heading **đầu tiên** mà hàm này trả về
    `True` mới là điểm bắt đầu cấu trúc Phần/Chương/Mục/Điều/Khoản thật của
    văn bản (mục 4.6) — không phải "gặp `#` bất kỳ đầu tiên".
    """
    if level == 1:
        return bool(RE_PHU_LUC.match(text) or RE_PHAN.match(text))
    if level == 2:
        return bool(RE_CHUONG.match(text))
    if level == 3:
        return bool(RE_MUC.match(text))
    if level == 4:
        return bool(RE_DIEU.match(text))
    if level == 5:
        return bool(RE_KHOAN_LABEL.match(text) or RE_KHOAN_MERGED.match(text))
    return False


# ==========================================================================
# BẢNG
# ==========================================================================

# Dòng bắt đầu bằng "|" — bảng markdown GFM do formatting/tables.py sinh ra.
RE_TABLE_LINE = re.compile(r"^\s*\|")

# Bảng công thức 1 hàng (vd. "Tiền lương làm thêm giờ = ... x ...") được
# formatting/tables.py::_single_row_table_to_html render bằng HTML thô thay
# vì pipe table (bảng pipe sẽ đẩy nhầm số hạng đầu tiên lên làm header) —
# chunking_spec.md mục 5 không nhắc tới dạng này, coi là 1 loại bảng cần giữ
# nguyên tương tự bảng pipe (quyết định thiết kế, xem báo cáo bàn giao).
RE_HTML_TABLE_LINE = re.compile(r"^\s*<table\b", re.IGNORECASE)


# ==========================================================================
# MARKER TÁCH BACKMATTER (mục 4.6)
# ==========================================================================

# formatting/pipeline.py._compose_markdown ngăn cách back matter (nếu có)
# bằng đúng 1 dòng "---" đứng riêng (formatting_spec.md mục 1.1 "Ghép output
# cuối cùng" + mục 2, xác nhận trên toàn bộ 6 file data/markdown/*.md thật —
# mỗi file có ĐÚNG 1 dòng "^---$", luôn nằm ngay sau nội dung Khoản cuối
# cùng). Đây là marker thật parser.py dùng để tách backmatter — khác mô tả
# `**[n]**`/`footnotes.py::render_blockquote` trong chunking_spec.md mục 4.6
# (tài liệu cũ từ thiết kế footnotes.py trước đây, module này đã bị xoá khỏi
# `formatting/` từ bản redesign Groq 2026-09-16, xem báo cáo bàn giao
# developer). Không neo `\s*$` lỏng lẻo: dòng gạch ngang trang trí trong
# frontmatter dài hơn 3 ký tự (vd. "--------", "---------------") sẽ không
# khớp regex 3 ký tự đúng khít này.
RE_BACKMATTER_SEPARATOR = re.compile(r"^-{3}$")


# ==========================================================================
# TÁCH CÂU (mục 4.4)
# ==========================================================================

# Tách câu đơn giản trên dấu kết câu "." hoặc ";" theo sau bởi khoảng trắng —
# chấp nhận sai số nhỏ ở tầng fallback hiếm gặp này, không dùng NLP nặng.
_RE_SENTENCE_BOUNDARY = re.compile(r"(?<=[.;])\s+")


def split_sentences(text: str) -> list[str]:
    """Tách văn bản thành danh sách câu theo regex đơn giản (mục 4.4)."""
    return [part.strip() for part in _RE_SENTENCE_BOUNDARY.split(text) if part.strip()]


# Tầng tách thêm khi 1 câu (mục 4.4) tự nó vẫn vượt MAX_TOKENS (không được
# spec định nghĩa — mở rộng để đáp ứng tiêu chí mục 11 "token_count <=
# MAX_TOKENS cho mọi chunk không có bảng", xem báo cáo bàn giao).
_RE_CLAUSE_BOUNDARY = re.compile(r",\s+")


def split_finer(text: str) -> list[str]:
    """Tách nhỏ hơn nữa theo dấu phẩy, dùng khi `split_sentences` không đủ."""
    return [part.strip() for part in _RE_CLAUSE_BOUNDARY.split(text) if part.strip()]
