"""Chú thích sửa đổi ở cuối văn bản: tìm vùng, đọc nội dung, render lại inline.

Văn bản hợp nhất/sửa đổi đánh dấu điều khoản bị thay đổi bằng marker
``[n]`` trong thân văn bản, rồi liệt kê nội dung sửa đổi tương ứng ở một
vùng riêng tại cuối file. ``find_region_start`` tìm **vị trí** vùng đó —
100% deterministic, không đổi. Đường chính đọc **nội dung** từng chú thích
trong vùng là LLM (``llm_client``, schema ``FootnoteExtraction`` —
formatting_spec.md mục 1, 6), qua ``resolve_region``; ``parse_region``
(regex) giữ nguyên logic, đổi vai trò thành baseline: fallback khi LLM lỗi,
và so sánh số lượng chú thích để phát ``QcWarning`` khi lệch nhau.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from production_legal_qa_rag.formatting import llm_client
from production_legal_qa_rag.formatting.docx_reader import Block
from production_legal_qa_rag.formatting.models import FootnoteExtraction, QcWarning
from production_legal_qa_rag.formatting.patterns import (
    RE_FN_DEF_BARE,
    RE_FN_DEF_BRACKET,
    RE_FOOTNOTE_MARKER,
    RE_MARKER,
    RE_MISSING_SPACE,
)

# Chú thích dài hơn ngưỡng này chỉ inline đoạn đầu, phần còn lại dời xuống cuối
# file dưới dòng in đậm **[n]** (không phải heading).
FOOTNOTE_INLINE_MAX_CHARS = 1200

_FOOTNOTE_PROMPT_TEMPLATE = """\
Bạn là trợ lý trích xuất chú thích sửa đổi trong văn bản pháp luật Việt Nam.

Vùng văn bản dưới đây liệt kê các chú thích sửa đổi theo thứ tự, mỗi chú \
thích thường mở đầu bằng số hiệu trong dấu ngoặc vuông (vd. "[1]", "[2]"), \
một số dòng không có dấu ngoặc nhưng vẫn mở đầu bằng số thứ tự. Với MỖI chú \
thích, trích số hiệu và toàn bộ nội dung của chú thích đó, giữ nguyên văn,
không tóm tắt hay bỏ sót đoạn nào.

Vùng chú thích:
{text}
"""


@dataclass(frozen=True, slots=True)
class Footnote:
    """Một chú thích sửa đổi ở cuối văn bản."""

    number: int
    paragraphs: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


def find_region_start(blocks: list[Block]) -> tuple[int | None, list[QcWarning]]:
    """Tìm chỉ số block bắt đầu vùng chú thích ở cuối văn bản.

    Mỗi văn bản có chú thích chỉ có ĐÚNG MỘT paragraph mở đầu bằng ``[1]``
    (chấp nhận chữ số lặp phía trước), và nó luôn nằm ngay sau bảng chữ ký.
    """
    warnings: list[QcWarning] = []
    hits: list[int] = []

    for index, block in enumerate(blocks):
        if block.kind != "paragraph":
            continue
        match = RE_FN_DEF_BRACKET.match(block.text)
        if match is None or match.group(2) != "1":
            continue
        # Loại trường hợp "7[1]" — chữ số lặp phải bằng số chú thích.
        if match.group(1) is not None and match.group(1) != "1":
            continue
        hits.append(index)

    if not hits:
        return None, warnings

    if len(hits) > 1:
        warnings.append(
            QcWarning(
                code="ambiguous_footnote_region",
                detail=f"tìm thấy {len(hits)} dòng mở đầu [1], lấy dòng cuối",
            )
        )

    return hits[-1], warnings


def parse_region(blocks: list[Block]) -> tuple[dict[int, Footnote], list[QcWarning]]:
    """Phân tích vùng chú thích thành map ``số hiệu -> nội dung``.

    Dùng bộ đếm kỳ vọng vì corpus có bốn dạng dòng định nghĩa, trong đó một
    dạng KHÔNG có ngoặc ("2 Điểm này được sửa đổi..." ở Bộ luật lao động).
    Parser ``^\\[(\\d+)\\]`` thuần sẽ mất hẳn chú thích đó.

    Dạng trần chỉ được chấp nhận khi số bằng đúng số kỳ vọng kế tiếp. Nhờ vậy
    nội dung trích dẫn bên trong chú thích không bị nhận nhầm: các dòng đó có
    dấu chấm ("1. Luật này có hiệu lực...") nên RE_FN_DEF_BARE không khớp, và
    kể cả khớp thì bộ đếm cũng đã vượt qua số đó.
    """
    warnings: list[QcWarning] = []
    collected: dict[int, list[str]] = {}
    expected = 1
    current: list[str] | None = None

    for block in blocks:
        if block.kind != "paragraph":
            if current is not None:
                current.append(block.text)
            continue

        text = block.text

        match = RE_FN_DEF_BRACKET.match(text)
        if match is not None and match.group(2):
            number = int(match.group(2))
            if number != expected:
                warnings.append(
                    QcWarning(
                        code="footnote_number_gap",
                        detail=f"kỳ vọng [{expected}], gặp [{number}]",
                    )
                )
            rest = match.group(4).strip()
            current = collected.setdefault(number, [])
            if rest:
                current.append(rest)
            expected = number + 1
            continue

        bare = RE_FN_DEF_BARE.match(text)
        if bare is not None and int(bare.group(1)) == expected:
            number = expected
            current = collected.setdefault(number, [])
            current.append(bare.group(2).strip())
            expected = number + 1
            continue

        if current is not None:
            current.append(text)
        else:
            # Rác đứng trước chú thích [1]; giữ lại để không mất nội dung im lặng.
            collected.setdefault(0, []).append(text)

    footnotes = {
        number: Footnote(number=number, paragraphs=tuple(paragraphs))
        for number, paragraphs in collected.items()
        if number > 0
    }
    return footnotes, warnings


def _region_text(region_blocks: list[Block]) -> str:
    return "\n".join(block.text for block in region_blocks if block.kind == "paragraph")


def extract_footnotes_llm(region_blocks: list[Block]) -> FootnoteExtraction | None:
    """Gọi LLM đọc nội dung toàn bộ chú thích trong vùng, làm đường chính.

    Trả ``None`` khi ``llm_client.extract_structured`` thất bại (lỗi/timeout/
    hết số lần thử) — caller (``resolve_region``) tự fallback baseline.
    """
    prompt = _FOOTNOTE_PROMPT_TEMPLATE.format(text=_region_text(region_blocks))
    result = llm_client.extract_structured(prompt, FootnoteExtraction)
    if not isinstance(result, FootnoteExtraction):
        return None
    return result


def resolve_region(
    region_blocks: list[Block],
) -> tuple[dict[int, Footnote], list[QcWarning]]:
    """Phân giải nội dung vùng chú thích: LLM là đường chính, ``parse_region``
    (regex) là baseline để fallback/so sánh (mục 1, 6, 7 spec).

    Cảnh báo từ ``parse_region`` (vd. ``footnote_number_gap``) luôn được giữ
    lại — đó là tín hiệu QC từ chính vùng văn bản, độc lập với việc chọn
    nguồn nội dung nào làm kết quả cuối cùng.
    """
    baseline_map, warnings = parse_region(region_blocks)

    extraction = extract_footnotes_llm(region_blocks)
    if extraction is None or not extraction.entries:
        warnings.append(QcWarning(code="llm_footnote_extraction_failed", detail=""))
        return baseline_map, warnings

    llm_map = {
        entry.number: Footnote(number=entry.number, paragraphs=(entry.content,))
        for entry in extraction.entries
    }

    if len(llm_map) != len(baseline_map):
        warnings.append(
            QcWarning(
                code="llm_footnote_count_mismatch",
                detail=f"llm={len(llm_map)} baseline={len(baseline_map)}",
            )
        )

    return llm_map, warnings


def strip_markers(text: str) -> tuple[str, list[int]]:
    """Gỡ mọi marker ``[n]`` khỏi một dòng, trả về dòng sạch và số hiệu đã gặp.

    Marker trong corpus có chữ số lặp ở một trong hai bên: "3.3[3]", "1.4[4]",
    "[4]4", "a)2[2]". Chỉ nuốt chữ số lặp khi nó BẰNG số chú thích — nếu không,
    "10.[15]" sẽ mất số khoản 10.
    """
    found: list[int] = []

    def _replace(match: re.Match[str]) -> str:
        number = match.group("num")
        pre = match.group("pre")
        post = match.group("post")
        found.append(int(number))
        keep_pre = pre if pre and pre != number else ""
        keep_post = post if post and post != number else ""
        return keep_pre + keep_post

    stripped = RE_MARKER.sub(_replace, text)
    # "a)Thành lập" -> "a) Thành lập"; "1.Cá nhân" -> "1. Cá nhân"
    stripped = RE_MISSING_SPACE.sub(r"\1 ", stripped)
    # Marker giữa câu để lại khoảng trắng thừa trước dấu câu.
    stripped = stripped.replace(" ,", ",").replace(" ;", ";").replace(" .", ".")
    return " ".join(stripped.split()), found


def strip_all(
    blocks: list[Block],
) -> tuple[list[Block], dict[int, list[int]], list[QcWarning]]:
    """Gỡ marker khỏi toàn bộ body, ghi nhớ vị trí theo chỉ số block.

    ``refs`` khóa theo chỉ số block là đủ mịn: đích chèn blockquote là "cuối
    khối tương ứng", không phải offset ký tự. Nhờ vậy marker nằm trên tiêu đề
    Điều, trên dòng dẫn nhập hay trên Khoản đều đi cùng một nhánh code.
    """
    warnings: list[QcWarning] = []
    cleaned: list[Block] = []
    refs: dict[int, list[int]] = {}

    for index, block in enumerate(blocks):
        if block.kind == "table":
            # Không bao giờ gỡ marker trong bảng — dây bẫy, không phải feature.
            if RE_FOOTNOTE_MARKER.search(block.text):
                warnings.append(
                    QcWarning(code="footnote_marker_in_table", detail=f"block {index}")
                )
            cleaned.append(block)
            continue

        text, numbers = strip_markers(block.text)
        if numbers:
            refs[index] = numbers
        cleaned.append(
            Block(kind="paragraph", text=text, style=block.style, is_bold=block.is_bold)
        )

    return cleaned, refs, warnings


def render_blockquote(footnote: Footnote, *, inline_max: int) -> tuple[str, str | None]:
    """Render chú thích thành blockquote.

    Trả về ``(inline, deferred)``. Chú thích ngắn thì chèn nguyên văn tại chỗ.
    Chú thích dài (trích nguyên cả Điều của luật khác) chỉ inline đoạn đầu,
    phần còn lại dời xuống cuối file để không làm phồng khối Khoản cha thêm
    nhiều KB.
    """
    paragraphs = [p for p in footnote.paragraphs if p.strip()]
    if not paragraphs:
        return "", None

    total = sum(len(p) for p in paragraphs)

    if total <= inline_max:
        body = "\n>\n".join(f"> {p}" for p in paragraphs)
        return f"> **Sửa đổi:** {body.removeprefix('> ')}", None

    head = paragraphs[0]
    inline = (
        f"> **Sửa đổi:** {head}\n>\n"
        f"> _(Trích dẫn đầy đủ: xem chú thích [{footnote.number}] ở cuối văn bản.)_"
    )
    deferred = "\n\n".join([f"**[{footnote.number}]**", *paragraphs])
    return inline, deferred
