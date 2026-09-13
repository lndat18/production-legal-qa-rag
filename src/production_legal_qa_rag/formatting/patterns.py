"""Regex nhận diện cấu trúc văn bản pháp luật và tiện ích liên quan.

Toàn bộ regex dùng để nhận diện Phần/Phụ Lục/Chương/Mục/Điều/Khoản/Điểm, chú
thích sửa đổi, bảng theo nội dung và front matter được gom về một chỗ để dễ
tra cứu/khớp thứ tự ưu tiên. Cũng chứa ``normalize_text`` (chuẩn hóa văn bản
trước khi áp regex) và ``sort_key`` (so sánh số hiệu có hậu tố chữ), vì hai
hàm này được mọi module khác trong package dùng chung mà không phụ thuộc gì
thêm.
"""

from __future__ import annotations

import re
import unicodedata

# ==========================================================================
# CẤU HÌNH
# ==========================================================================

# Tăng thủ công mỗi khi rule nhận diện heading thay đổi. Bump trong cùng commit
# với thay đổi golden/tham chiếu: hai việc đó là một sự kiện.
PARSER_VERSION = "1.0.0"

NON_BREAKING_SPACE = " "


def normalize_text(raw: str) -> str:
    """Chuẩn hóa NFC, thay non-breaking space, gộp khoảng trắng, strip.

    Args:
        raw: Văn bản thô lấy từ DOCX.

    Returns:
        Văn bản đã chuẩn hóa, sẵn sàng để áp các regex nhận diện cấu trúc.
    """
    text = unicodedata.normalize("NFC", raw)
    text = text.replace(NON_BREAKING_SPACE, " ")
    return " ".join(text.split())


# ==========================================================================
# REGEX NHẬN DIỆN CẤU TRÚC
# ==========================================================================

ROMAN = r"[IVXLCDM]+"

# "Phần thứ nhất" dùng số thứ tự tiếng Việt, không phải chữ số La Mã. Corpus
# hiện tại không có Phần nào, nhưng Bộ luật Dân sự thì có.
ORDINAL_WORD = r"nhất|hai|ba|tư|bốn|năm|sáu|bảy|tám|chín|mười"

RE_PHU_LUC = re.compile(r"^PHỤ\s+LỤC\b", re.IGNORECASE)
RE_PHAN = re.compile(
    rf"^Phần\s+(?:thứ\s+)?({ROMAN}|\d+|{ORDINAL_WORD})\b", re.IGNORECASE
)
RE_CHUONG = re.compile(rf"^Chương\s+({ROMAN}|\d+)\b", re.IGNORECASE)
RE_MUC = re.compile(r"^Mục\s+(\d+[a-zđ]?)\b", re.IGNORECASE)

# Hỗ trợ số hiệu chữ: Điều 41a, Điều 7a, Điều 48b — phổ biến ở văn bản hợp nhất
# khi bổ sung điều mới vào giữa. Bắt buộc có dấu chấm sau số hiệu, nhờ vậy các
# dòng "Điều 2 của Luật số 46/2014/QH13 ... quy định như sau:" trong vùng chú
# thích không bị nhận nhầm thành heading.
RE_DIEU = re.compile(r"^Điều\s+(\d+[a-zđ]?)\s*\.\s*(.*)$")

# Khoản = bắt đầu bằng số. Không kèm điều kiện ngữ cảnh.
# Nhánh (?:\[\d+\])? giữ lại làm lớp phòng vệ thứ hai; lớp phòng vệ thật là
# bước gỡ marker ở strip_all, vì corpus có dạng "3.3[3]" mà nhánh này
# không cứu được.
RE_KHOAN = re.compile(r"^(\d+[a-zđ]?)\s*\.\s*(?:\[\d+\])?\s+(.*)$")

# Điểm KHÔNG bao giờ tạo heading. Chỉ dùng cho QC và nhận biết ngữ cảnh.
# Hai hình thức: chữ cái + ")" trong thân văn bản, dấu "-" trong Phụ lục.
RE_DIEM = re.compile(r"^(?:([a-zđư])\)|-)\s+(.*)$")

# Bảng chữ cái tiếng Việt dùng cho điểm (không có f, j, w, z).
VIETNAMESE_POINT_LETTERS = [
    "a",
    "b",
    "c",
    "d",
    "đ",
    "e",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "ư",
    "v",
    "x",
    "y",
]

# --- Chú thích sửa đổi ---------------------------------------------------

# Marker trong thân văn bản. Chữ số lặp có thể nằm hai bên: "3.3[3]", "[4]4".
# Chỉ nuốt pre/post khi nó BẰNG num, nếu không "10.[15]" sẽ mất số khoản 10.
RE_MARKER = re.compile(
    r"(?:(?P<pre>\d{1,3}))?\[(?P<num>\d{1,3})\](?:(?P<post>\d{1,3}))?"
)

# Dòng định nghĩa chú thích có 4 dạng trong corpus:
#   "[1] ..."   "3[3] ..."   "[4]4 ..."   và dạng trần "2 Điểm này được sửa..."
RE_FN_DEF_BRACKET = re.compile(r"^(?:(\d{1,3}))?\[(\d{1,3})\](?:(\d{1,3}))?\s*(.*)$")

# Dạng trần: KHÔNG có dấu chấm sau số. Đó là điều phân biệt nó với RE_KHOAN,
# nên dòng "1. Luật này có hiệu lực..." được trích dẫn bên trong chú thích
# không thể bị nhận nhầm thành một định nghĩa chú thích mới.
RE_FN_DEF_BARE = re.compile(r"^(\d{1,3})\s+(\S.*)$")

RE_FOOTNOTE_MARKER = re.compile(r"\[(\d+)\]")

# Sau khi gỡ marker dính liền, "a)Thành lập" và "1.Cá nhân" mất khoảng trắng.
RE_MISSING_SPACE = re.compile(r"^([a-zđư]\)|\d{1,3}\.)(?=\S)")

# --- Bảng ----------------------------------------------------------------

# Nhận diện bảng theo NỘI DUNG, không theo vị trí: bảng chữ ký của Nghị định
# 293 nằm ở giữa văn bản, toàn bộ Phụ lục nằm sau nó. Hai trong sáu bảng chữ ký
# lại có ô đầu rỗng nên riêng "Nơi nhận:" là không đủ.
# Không neo "^": các pattern này dò trên TOÀN BỘ bảng đã render, mà chuỗi đó bắt
# đầu bằng "<table>" hoặc "| ". Neo đầu chuỗi sẽ không bao giờ khớp.
RE_SIGNATURE_CELL = re.compile(
    r"Nơi\s+nhận:"
    r"|XÁC\s+THỰC\s+VĂN\s+BẢN\s+HỢP\s+NHẤT"
    r"|CHỦ\s+NHIỆM"
    r"|TM\.\s"
    r"|KT\.\s"
    r"|THỦ\s+TƯỚNG",
    re.IGNORECASE,
)
RE_QUOC_HIEU_CELL = re.compile(
    r"CỘNG\s+HÒA\s+XÃ\s+HỘI\s+CHỦ\s+NGHĨA\s+VIỆT\s+NAM", re.IGNORECASE
)
RE_ATTACHMENT_TABLE = re.compile(
    r"FILE\s+ĐƯỢC\s+ĐÍNH\s+KÈM\s+THEO\s+VĂN\s+BẢN", re.IGNORECASE
)

# --- Front matter --------------------------------------------------------

RE_SO_HIEU = re.compile(r"Số:\s*([^\s|<]+)")

# "Hà Nội, ngày 20 tháng 5 năm 2026" và "Hà Nội ngày 10 tháng 11 năm 2025"
# (Nghị định 293 thiếu dấu phẩy).
RE_NGAY = re.compile(
    r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE
)

RE_HIEU_LUC = re.compile(
    r"có\s+hiệu\s+lực(?:\s+thi\s+hành)?(?:\s+kể)?\s+từ\s+ngày\s+"
    r"(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
    re.IGNORECASE,
)

DOC_TYPE_ALT = r"Bộ luật|Luật|Nghị định|Nghị quyết|Thông tư|Quyết định"

RE_BAN_HANH = re.compile(
    rf"\bban\s+hành\s+((?:{DOC_TYPE_ALT})\s+.+?)\s*\.?\s*$", re.IGNORECASE
)
RE_TEN_SO = re.compile(rf"^((?:{DOC_TYPE_ALT})\s+.+?)\s+số\s+\d+/", re.IGNORECASE)
RE_DOC_TYPE = re.compile(
    r"^(BỘ\s+LUẬT|LUẬT|NGHỊ\s+ĐỊNH|NGHỊ\s+QUYẾT|QUYẾT\s+ĐỊNH|THÔNG\s+TƯ)$"
)

# Tiêu đề Phụ lục của Nghị định 293 dính cả phần "(Kèm theo ...)" dài ~90 ký tự.
RE_KEM_THEO = re.compile(r"\s*(\(\s*Kèm\s+theo\b.*)$", re.IGNORECASE | re.DOTALL)


def is_structural(text: str) -> bool:
    """Dòng có phải nhãn cấu trúc (Phụ lục/Phần/Chương/Mục/Điều) hay không."""
    return any(
        pattern.match(text)
        for pattern in (RE_PHU_LUC, RE_PHAN, RE_CHUONG, RE_MUC, RE_DIEU)
    )


def sort_key(number: str) -> tuple[int, str]:
    """Khóa so sánh số hiệu dạng "48a".

    ``int()`` trần sẽ báo giả trên chuỗi hợp lệ 48 -> 48a -> 48b -> 49 của Luật
    Bảo hiểm y tế, và 1 -> 1a -> 1b -> 1c của Bộ luật lao động.

    Args:
        number: Số hiệu dạng chuỗi, có thể kèm hậu tố chữ (vd. "41a").

    Returns:
        Cặp ``(phần số, phần chữ)`` dùng để so sánh thứ tự đúng.
    """
    digits = "".join(character for character in number if character.isdigit())
    suffix = number[len(digits) :]
    return (int(digits) if digits else 0, suffix)
