"""Duyệt Block theo thứ tự, áp heading mapping, sinh markdown cuối cùng.

Đây là bước lõi của formatting: mỗi Block được xét theo regex nhận diện của
CHÍNH level đó (không đọc style/bold gốc) để quyết định render thành heading
cấp nào, theo đúng bảng ánh xạ Phần/Phụ Lục → `#`, Chương → `##`, Mục → `###`,
Điều → `####`, Khoản → `#####` (chỉ chứa nhãn, trừ trường hợp Phụ lục có nội
dung ngắn), còn Điểm thì xuất nguyên văn như đoạn văn thường.
"""

from __future__ import annotations

from production_legal_qa_rag.formatting.docx_reader import Block
from production_legal_qa_rag.formatting.footnotes import (
    FOOTNOTE_INLINE_MAX_CHARS,
    Footnote,
    render_blockquote,
)
from production_legal_qa_rag.formatting.models import QcWarning
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


def find_preamble_end(blocks: list[Block]) -> int:
    """Chỉ số block của heading cấu trúc đầu tiên.

    Vùng trước đó là dẫn nhập và được xuất nguyên văn. Văn bản hợp nhất liệt kê
    luật sửa đổi bằng dòng đánh số ngay trong vùng này ("1. Luật Nhà giáo số
    73/2025/QH15..."); nếu chạy RE_KHOAN ở đây thì sinh ra "##### Khoản 1-4"
    giả trước cả Chương I.
    """
    for index, block in enumerate(blocks):
        if block.kind == "paragraph" and is_structural(block.text):
            return index
    return len(blocks)


def _join_title(
    blocks: list[Block],
    index: int,
    refs: dict[int, list[int]],
    separator: str,
) -> tuple[str, str | None, int, list[int]]:
    """Gộp nhãn cấu trúc với dòng tiêu đề viết hoa ngay sau nó.

    Trong corpus, "Chương I" và "NHỮNG QUY ĐỊNH CHUNG" là hai paragraph riêng,
    cần gộp lại thành "## Chương I. Những quy định chung".

    Trả về ``(tiêu_đề, phần_dư, số_block_đã_tiêu_thụ, refs_gộp)``. Phần dư là
    đoạn "(Kèm theo ...)" tách khỏi tiêu đề Phụ lục — nếu không tách thì
    heading dài ra hàng trăm ký tự.
    """
    title = blocks[index].text
    collected = list(refs.get(index, []))
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
            # Marker sửa đổi có thể nằm ở dòng thứ hai của tiêu đề, phải gộp theo.
            collected.extend(refs.get(following, []))
            consumed = 2
        else:
            extra = None

    return title, extra, consumed, collected


def emit(
    blocks: list[Block],
    refs: dict[int, list[int]],
    footnote_map: dict[int, Footnote],
    preamble_end: int,
) -> tuple[list[str], list[str], bool, list[QcWarning]]:
    """Sinh các khối markdown theo bảng ánh xạ heading của mục 3 spec.

    Args:
        blocks: Body đã tách bảng quốc hiệu, cắt vùng chú thích và gỡ marker.
        refs: Map chỉ số block -> danh sách số hiệu chú thích xuất hiện ở đó.
        footnote_map: Map số hiệu chú thích -> nội dung, dùng để chèn lại.
        preamble_end: Chỉ số block đầu tiên thuộc vùng cấu trúc (sau dẫn nhập).

    Returns:
        Bộ bốn ``(các đoạn markdown, chú thích dời xuống cuối, có đang ở
        trong Phụ lục hay không tại thời điểm kết thúc, cảnh báo QC)``.
    """
    warnings: list[QcWarning] = []
    parts: list[str] = []
    deferred: list[str] = []
    in_phu_luc = False
    used: set[int] = set()

    def flush(numbers: list[int]) -> None:
        for number in numbers:
            footnote = footnote_map.get(number)
            if footnote is None:
                warnings.append(QcWarning(code="orphan_footnote", detail=f"[{number}]"))
                continue
            used.add(number)
            inline, tail = render_blockquote(
                footnote, inline_max=FOOTNOTE_INLINE_MAX_CHARS
            )
            if inline:
                parts.append(inline)
            if tail is not None:
                deferred.append(tail)
                warnings.append(
                    QcWarning(code="long_footnote_deferred", detail=f"[{number}]")
                )

    index = 0
    while index < len(blocks):
        block = blocks[index]
        here = refs.get(index, [])

        if block.kind == "table":
            parts.append(block.text)
            flush(here)
            index += 1
            continue

        if index < preamble_end:
            parts.append(block.text)
            flush(here)
            index += 1
            continue

        text = block.text

        if RE_PHU_LUC.match(text):
            title, extra, consumed, merged = _join_title(blocks, index, refs, " — ")
            parts.append(f"# {title}")
            flush(merged)
            if extra:
                parts.append(extra)
            in_phu_luc = True
            index += consumed
            continue

        if RE_PHAN.match(text):
            title, extra, consumed, merged = _join_title(blocks, index, refs, ". ")
            parts.append(f"# {title}")
            flush(merged)
            if extra:
                parts.append(extra)
            in_phu_luc = False
            index += consumed
            continue

        if RE_CHUONG.match(text):
            title, extra, consumed, merged = _join_title(blocks, index, refs, ". ")
            parts.append(f"## {title}")
            flush(merged)
            if extra:
                parts.append(extra)
            in_phu_luc = False
            index += consumed
            continue

        if RE_MUC.match(text):
            parts.append(f"### {text}")
            flush(here)
            index += 1
            continue

        if RE_DIEU.match(text):
            # Giữ nguyên tiêu đề gốc của Điều.
            parts.append(f"#### {text}")
            flush(here)
            index += 1
            continue

        khoan = RE_KHOAN.match(text)
        if khoan is not None:
            number, content = khoan.group(1), khoan.group(2)
            if (
                in_phu_luc
                and PHU_LUC_HEADING_WITH_TITLE
                and len(content) <= PHU_LUC_HEADING_MAX_CHARS
            ):
                # Tên tỉnh/thành là thông tin định danh quan trọng nhất của mục,
                # giữ nó trên dòng heading để breadcrumb của chunk con có tên.
                parts.append(f"##### {number}. {content}")
            else:
                # Heading của Khoản chỉ chứa nhãn, nội dung xuống dòng riêng.
                parts.append(f"##### Khoản {number}")
                parts.append(content)
            flush(here)
            index += 1
            continue

        # Còn lại, gồm điểm "a)" và "- ...": xuất nguyên văn.
        parts.append(text)
        flush(here)
        index += 1

    for number in sorted(set(footnote_map) - used):
        warnings.append(QcWarning(code="unused_footnote", detail=f"[{number}]"))

    return parts, deferred, in_phu_luc, warnings
