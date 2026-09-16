"""Regex nhận diện cấu trúc văn bản pháp luật và tiện ích liên quan.

Toàn bộ regex dùng để nhận diện Phần/Phụ Lục/Chương/Mục/Điều/Khoản/Điểm và
bảng theo nội dung (chữ ký, đính kèm) được gom về một chỗ để dễ tra cứu/khớp
thứ tự ưu tiên. Cũng chứa ``normalize_text`` (chuẩn hóa văn bản trước khi áp
regex) và ``sort_key`` (so sánh số hiệu có hậu tố chữ), vì hai hàm này được
mọi module khác trong package dùng chung mà không phụ thuộc gì thêm.

Không còn regex trích field front matter (số hiệu, ngày ban hành, loại văn
bản...) hay marker chú thích sửa đổi ``[n]`` -- cả hai đã bị bỏ hoàn toàn
theo thiết kế mới (formatting_spec.md mục 1.1): front matter/back matter
chuyển đổi nguyên khối bằng Gemini, không còn trích field/khôi phục inline.
"""

from __future__ import annotations

import re
import unicodedata

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
# dòng "Điều 2 của Luật số 46/2014/QH13 ... quy định như sau:" trong nội dung
# trích dẫn không bị nhận nhầm thành heading.
RE_DIEU = re.compile(r"^Điều\s+(\d+[a-zđ]?)\s*\.\s*(.*)$")

# Khoản = bắt đầu bằng số + dấu chấm. Không kèm điều kiện ngữ cảnh.
RE_KHOAN = re.compile(r"^(\d+[a-zđ]?)\s*\.\s+(.*)$")

# Điểm KHÔNG bao giờ tạo heading. Chỉ dùng cho QC và nhận biết ngữ cảnh.
# Hai hình thức: chữ cái + ")" trong thân văn bản, dấu "-" trong Phụ lục.
RE_DIEM = re.compile(r"^(?:([a-zđư])\)|-)\s+(.*)$")

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
RE_ATTACHMENT_TABLE = re.compile(
    r"FILE\s+ĐƯỢC\s+ĐÍNH\s+KÈM\s+THEO\s+VĂN\s+BẢN", re.IGNORECASE
)

# Tiêu đề Phụ lục của Nghị định 293 dính cả phần "(Kèm theo ...)" dài ~90 ký tự.
RE_KEM_THEO = re.compile(r"\s*(\(\s*Kèm\s+theo\b.*)$", re.IGNORECASE | re.DOTALL)


def is_structural(text: str) -> bool:
    """Dòng có phải nhãn cấu trúc (Phụ lục/Phần/Chương/Mục/Điều) hay không.

    Dùng để xác định biên front matter (formatting_spec.md mục 1.1): block
    đầu tiên khớp hàm này là block đầu tiên KHÔNG thuộc front matter.
    """
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
