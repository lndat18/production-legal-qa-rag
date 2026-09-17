"""Regex nhận diện cấu trúc văn bản pháp luật và tiện ích liên quan.

Toàn bộ regex dùng để nhận diện Phần/Phụ Lục/Chương/Mục/Điều/Khoản/Điểm và
bảng theo nội dung (chữ ký, đính kèm) được gom về một chỗ để dễ tra cứu/khớp
thứ tự ưu tiên. Cũng chứa ``normalize_text`` (chuẩn hóa văn bản trước khi áp
regex) và ``sort_key`` (so sánh số hiệu có hậu tố chữ), vì hai hàm này được
mọi module khác trong package dùng chung mà không phụ thuộc gì thêm.

Không còn regex trích field front matter (số hiệu, ngày ban hành, loại văn
bản...) -- đã bị bỏ hoàn toàn theo thiết kế mới (formatting_spec.md mục 1.1):
front matter/back matter chuyển đổi bằng Groq (theo chunk, mục 1.2), không
còn trích field/khôi phục inline chú thích về vị trí gốc.

Vẫn GIỮ regex + hàm ``strip_markers`` gỡ marker chú thích ``[n]`` dính liền
trong text -- khác với việc khôi phục *nội dung* chú thích (đã bỏ), đây chỉ
là bước làm sạch text thuộc vùng nội dung ở giữa, chạy trước khi áp regex
heading (``RE_KHOAN`` yêu cầu khoảng trắng ngay sau dấu chấm, marker dính
liền như ``"1.[2] Bảo hiểm..."`` sẽ phá regex nếu không strip trước). Xem
formatting_spec.md mục 1.1, "Lưu ý quan trọng — marker [n]...".
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

# --- Marker chú thích [n] (vùng nội dung ở giữa) --------------------------

# Marker chú thích dính liền trong text, vd. "1.[2] Bảo hiểm y tế là...",
# "Điều 7a. ...Xã hội[16]". Dùng để phát hiện marker còn sót lại (QC) VÀ để
# gỡ marker trong `strip_markers` bên dưới.
RE_FOOTNOTE_MARKER = re.compile(r"\[(\d{1,3})\]")

# Marker có thể có chữ số lặp ở một trong hai bên ("3.3[3]", "1.4[4]",
# "[4]4", "a)2[2]") -- chỉ nuốt chữ số lặp khi nó BẰNG số marker, nếu không
# "10.[15]" sẽ mất số khoản 10.
RE_MARKER = re.compile(
    r"(?:(?P<pre>\d{1,3}))?\[(?P<num>\d{1,3})\](?:(?P<post>\d{1,3}))?"
)

# Sau khi gỡ marker dính liền, "a)Thành lập" và "1.Cá nhân" mất khoảng trắng.
RE_MISSING_SPACE = re.compile(r"^([a-zđư]\)|\d{1,3}\.)(?=\S)")


def strip_markers(text: str) -> str:
    """Gỡ mọi marker chú thích ``[n]`` dính liền khỏi một dòng text.

    Áp dụng cho mọi block thuộc vùng nội dung ở giữa, TRƯỚC khi áp regex
    heading (``emitter.py``) -- nếu không, marker dính ngay sau số Khoản/Điều
    (vd. ``"1.[2] Bảo hiểm y tế..."``) sẽ phá ``RE_KHOAN``/``RE_DIEU`` (yêu
    cầu khoảng trắng ngay sau dấu chấm) và làm mất heading thật sự. Đây là
    bước làm sạch text độc lập với việc khôi phục *nội dung* chú thích (đã
    bỏ) -- xem formatting_spec.md mục 1.1.

    Args:
        text: Text thô của một block (paragraph) thuộc vùng nội dung ở giữa.

    Returns:
        Text đã gỡ sạch marker, khoảng trắng/dấu câu được vá lại quanh vị trí
        marker vừa gỡ.
    """

    def _replace(match: re.Match[str]) -> str:
        number = match.group("num")
        pre = match.group("pre")
        post = match.group("post")
        keep_pre = pre if pre and pre != number else ""
        keep_post = post if post and post != number else ""
        return keep_pre + keep_post

    stripped = RE_MARKER.sub(_replace, text)
    # "a)Thành lập" -> "a) Thành lập"; "1.Cá nhân" -> "1. Cá nhân"
    stripped = RE_MISSING_SPACE.sub(r"\1 ", stripped)
    # Marker giữa câu để lại khoảng trắng thừa trước dấu câu.
    stripped = stripped.replace(" ,", ",").replace(" ;", ";").replace(" .", ".")
    return " ".join(stripped.split())


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


# --- Dòng loại văn bản trong front matter (formatting_spec.md mục 1.1) ----

# Khớp TRỌN DÒNG danh sách tên loại văn bản đơn thuần (không kèm tên riêng),
# vd. dòng "NGHỊ ĐỊNH" đứng một mình trước dòng tên đầy đủ văn bản. Dùng bởi
# `frontmatter.find_title` để định vị dòng loại văn bản -- block tên văn bản
# là block paragraph ngay sau đó (không cần điều kiện bold/viết hoa riêng,
# đã verify trên corpus thật -- 2/6 file có dòng tên không bold).
RE_DOC_TYPE_ONLY = re.compile(
    r"^(?:THÔNG\s+TƯ\s+LIÊN\s+TỊCH|THÔNG\s+TƯ|NGHỊ\s+ĐỊNH|NGHỊ\s+QUYẾT"
    r"|BỘ\s+LUẬT|QUYẾT\s+ĐỊNH|PHÁP\s+LỆNH|CHỈ\s+THỊ|LUẬT)$",
    re.IGNORECASE,
)


# --- Bug setext heading (formatting_spec.md mục 1.1, "[MỚI 2026-09-17]") --

# Dòng thuần gạch ngang/gạch bằng (>= 3 ký tự) -- đúng ngưỡng CommonMark cho
# thematic break/setext-heading underline (vd. "--------", "========"). Nếu
# dòng này đứng ngay dưới một dòng text không rỗng (không có dòng trống ở
# giữa), CommonMark hiểu nhầm đó là cú pháp setext heading, biến dòng text
# phía trên thành heading cấp 1/2 ngoài ý muốn.
RE_SETEXT_UNDERLINE = re.compile(r"^[-=]{3,}\s*$")


def escape_setext_underline(markdown: str) -> str:
    """Chèn 1 dòng trống trước mọi dòng gạch ngang/gạch bằng đứng liền kề text.

    Áp dụng lên markdown front matter/back matter SAU KHI ghép xong các chunk
    từ Groq (``frontmatter.assemble``/``backmatter.assemble``), trước khi trả
    về cho ``pipeline.py`` -- Groq có thể trả về dòng gạch ngang trang trí
    (vd. dưới tên cơ quan trong bảng quốc hiệu) liền ngay dưới dòng text, vô
    tình tạo thành setext heading hợp lệ theo CommonMark (formatting_spec.md
    mục 1.1, xác nhận thực tế trên ``Luật bảo hiểm xã hội.md``). Không áp
    dụng cho phần nội dung ở giữa (``emitter.py`` tự sinh heading ATX ``#``,
    không có dòng gạch ngang trang trí nào).

    Args:
        markdown: Markdown đã ghép xong (front matter hoặc back matter).

    Returns:
        Markdown với 1 dòng trống được chèn thêm trước mỗi dòng khớp
        ``RE_SETEXT_UNDERLINE`` mà dòng ngay phía trên không phải dòng trống.
        Không xoá nội dung, không đổi dòng gạch ngang.
    """
    lines = markdown.split("\n")
    result: list[str] = []
    for line in lines:
        if RE_SETEXT_UNDERLINE.match(line) and result and result[-1].strip() != "":
            result.append("")
        result.append(line)
    return "\n".join(result)


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
