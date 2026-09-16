"""Duyệt Block theo thứ tự, áp heading mapping, sinh markdown thân văn bản.

Đây là bước lõi của formatting: mỗi Block được xét theo regex nhận diện của
CHÍNH level đó (không đọc style/bold gốc) để quyết định render thành heading
cấp nào, theo đúng bảng ánh xạ Phần/Phụ Lục → `#`, Chương → `##`, Mục → `###`,
Điều → `####`, Khoản → `#####` (chỉ chứa nhãn, trừ trường hợp Phụ lục có nội
dung ngắn), còn Điểm thì xuất nguyên văn như đoạn văn thường. Đầu vào là
Block thuộc vùng nội dung ở giữa — front matter/back matter đã được cắt và
xử lý riêng bởi ``frontmatter.py``/``backmatter.py`` (mục 1.1 spec), không
còn bước chèn inline chú thích vào giữa thân văn bản.
"""

from __future__ import annotations

from production_legal_qa_rag.formatting.docx_reader import Block
from production_legal_qa_rag.formatting.patterns import (
    RE_CHUONG,
    RE_DIEU,
    RE_KEM_THEO,
    RE_KHOAN,
    RE_MUC,
    RE_PHAN,
    RE_PHU_LUC,
    is_structural,
)

# Trong Phụ lục ghi "##### 28. Thành phố Hồ Chí Minh" thay cho "##### Khoản 28",
# để tên tỉnh/thành nằm trong breadcrumb của mọi chunk con.
PHU_LUC_HEADING_WITH_TITLE = True

# Chốt chặn cho tùy chọn trên: mục Phụ lục dài hơn ngưỡng này quay về dạng
# "##### Khoản N" để heading không phình.
PHU_LUC_HEADING_MAX_CHARS = 120


def _is_uppercase_title(text: str) -> bool:
    """Đoạn có phải dòng tiêu đề viết hoa hay không."""
    letters = "".join(character for character in text if character.isalpha())
    return len(letters) >= 2 and letters.isupper()


def _join_title(
    blocks: list[Block], index: int, separator: str
) -> tuple[str, str | None, int]:
    """Gộp nhãn cấu trúc với dòng tiêu đề viết hoa ngay sau nó.

    Trong corpus, "Chương I" và "NHỮNG QUY ĐỊNH CHUNG" là hai paragraph riêng,
    cần gộp lại thành "## Chương I. Những quy định chung".

    Trả về ``(tiêu_đề, phần_dư, số_block_đã_tiêu_thụ)``. Phần dư là đoạn
    "(Kèm theo ...)" tách khỏi tiêu đề Phụ lục — nếu không tách thì heading
    dài ra hàng trăm ký tự.
    """
    title = blocks[index].text
    consumed = 1
    extra: str | None = None

    following = index + 1
    if following < len(blocks) and blocks[following].kind == "paragraph":
        candidate = blocks[following].text
        # Tách "(Kèm theo ...)" TRƯỚC khi kiểm tra viết hoa: tiêu đề Phụ lục
        # là một paragraph duy nhất kết thúc bằng phần trong ngoặc có chữ
        # thường, nếu kiểm tra sau thì dòng này không bao giờ được gộp.
        match = RE_KEM_THEO.search(candidate)
        if match is not None:
            extra = match.group(1).strip()
            candidate = candidate[: match.start()].strip()

        if (
            candidate
            and _is_uppercase_title(candidate)
            and not is_structural(candidate)
        ):
            title = f"{title}{separator}{candidate}"
            consumed = 2
        else:
            extra = None

    return title, extra, consumed


def _emit_titled_heading(
    blocks: list[Block], index: int, parts: list[str], level: str, separator: str
) -> int:
    """Xuất heading có thể gộp tiêu đề dòng sau (Phụ lục/Phần/Chương).

    Dùng chung cho ba nhánh vì cả ba đều cần `_join_title` để gộp dòng tiêu đề
    viết hoa theo sau và tách phần "(Kèm theo ...)" nếu có.

    Returns:
        Số block đã tiêu thụ (1, hoặc 2 nếu gộp thêm dòng tiêu đề).
    """
    title, extra, consumed = _join_title(blocks, index, separator)
    parts.append(f"{level} {title}")
    if extra:
        parts.append(extra)
    return consumed


def _emit_phu_luc(blocks: list[Block], index: int, parts: list[str]) -> int:
    """Xuất heading Phụ lục (cấp `#`), đánh dấu đang ở trong Phụ lục."""
    return _emit_titled_heading(blocks, index, parts, "#", " — ")


def _emit_phan(blocks: list[Block], index: int, parts: list[str]) -> int:
    """Xuất heading Phần (cấp `#`), ra khỏi phạm vi Phụ lục nếu đang ở đó."""
    return _emit_titled_heading(blocks, index, parts, "#", ". ")


def _emit_chuong(blocks: list[Block], index: int, parts: list[str]) -> int:
    """Xuất heading Chương (cấp `##`), ra khỏi phạm vi Phụ lục nếu đang ở đó."""
    return _emit_titled_heading(blocks, index, parts, "##", ". ")


def _emit_khoan(number: str, content: str, parts: list[str], in_phu_luc: bool) -> None:
    """Xuất heading Khoản (cấp `#####`).

    Trong Phụ lục, nếu nội dung đủ ngắn thì gộp vào heading (tên tỉnh/thành là
    thông tin định danh quan trọng nhất của mục, cần có trong breadcrumb của
    chunk con); ngoài ra heading chỉ chứa nhãn, nội dung xuống dòng riêng.
    """
    if (
        in_phu_luc
        and PHU_LUC_HEADING_WITH_TITLE
        and len(content) <= PHU_LUC_HEADING_MAX_CHARS
    ):
        parts.append(f"##### {number}. {content}")
    else:
        parts.append(f"##### Khoản {number}")
        parts.append(content)


def emit(blocks: list[Block]) -> list[str]:
    """Sinh các đoạn markdown thân văn bản theo bảng ánh xạ heading mục 3 spec.

    Args:
        blocks: Block thuộc vùng nội dung ở giữa — đã cắt front/back matter
            (``frontmatter.find_boundary``, ``backmatter.split_backmatter``)
            và lọc bảng đính kèm (``tables.filter_middle_tables``).

    Returns:
        Danh sách các đoạn markdown, theo đúng thứ tự xuất hiện.
    """
    parts: list[str] = []
    in_phu_luc = False

    index = 0
    while index < len(blocks):
        block = blocks[index]

        if block.kind == "table":
            parts.append(block.text)
            index += 1
            continue

        text = block.text

        if RE_PHU_LUC.match(text):
            index += _emit_phu_luc(blocks, index, parts)
            in_phu_luc = True
            continue

        if RE_PHAN.match(text):
            index += _emit_phan(blocks, index, parts)
            in_phu_luc = False
            continue

        if RE_CHUONG.match(text):
            index += _emit_chuong(blocks, index, parts)
            in_phu_luc = False
            continue

        if RE_MUC.match(text):
            parts.append(f"### {text}")
            index += 1
            continue

        if RE_DIEU.match(text):
            # Giữ nguyên tiêu đề gốc của Điều.
            parts.append(f"#### {text}")
            index += 1
            continue

        khoan = RE_KHOAN.match(text)
        if khoan is not None:
            _emit_khoan(khoan.group(1), khoan.group(2), parts, in_phu_luc)
            index += 1
            continue

        # Còn lại, gồm điểm "a)" và "- ...": xuất nguyên văn.
        parts.append(text)
        index += 1

    return parts
