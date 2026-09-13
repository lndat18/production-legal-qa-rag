"""QC trên markdown đã sinh: heading-level-skip và các bất thường khác.

Toàn bộ rule ở đây chỉ cảnh báo, không bao giờ làm fail file — kết quả trả
về là danh sách ``QcWarning`` để ``pipeline.py`` gom vào summary cuối cùng.
"""

from __future__ import annotations

from production_legal_qa_rag.formatting.models import QcWarning
from production_legal_qa_rag.formatting.patterns import (
    RE_DIEU,
    RE_FOOTNOTE_MARKER,
    RE_PHU_LUC,
    sort_key,
)

# Dòng heading dài hơn ngưỡng này bị cảnh báo suspicious_heading_length.
HEADING_MAX_LEN = 200


def _strip_front_matter(markdown: str) -> list[str]:
    """Bỏ khối YAML đầu file, trả về các dòng còn lại."""
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return lines
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return lines[index + 1 :]
    return lines


def _monotonic_breaks(numbers: list[str]) -> list[str]:
    """Các vị trí số hiệu không tăng dần, so sánh theo (số, hậu tố)."""
    breaks: list[str] = []
    previous: tuple[int, str] | None = None
    for number in numbers:
        current = sort_key(number)
        if previous is not None and current <= previous:
            breaks.append(number)
        previous = current
    return breaks


def validate(markdown: str) -> list[QcWarning]:
    """Chạy toàn bộ rule QC trên markdown."""
    warnings: list[QcWarning] = []
    lines = _strip_front_matter(markdown)

    in_phu_luc = False
    # seen_dieu là sticky trong thân văn bản: Chương/Mục không reset nó — level
    # trên bị bỏ trống là hợp lệ.
    seen_dieu = False
    reported_level_skip = False
    in_code_or_table = False

    heading_lines: list[tuple[int, str, str]] = []  # (số dòng, dấu #, nội dung)
    dieu_numbers: list[str] = []
    khoan_by_parent: dict[str, list[str]] = {}
    parent = "<none>"
    dieu_has_content: dict[str, bool] = {}
    current_dieu: str | None = None

    for line_number, raw in enumerate(lines, start=1):
        line = raw.rstrip()

        if line.startswith("<table>"):
            in_code_or_table = True
        if line.startswith("</table>"):
            in_code_or_table = False
            continue
        if in_code_or_table or line.startswith("|"):
            if current_dieu is not None:
                dieu_has_content[current_dieu] = True
            continue

        if not line.startswith("#"):
            if line.strip() and current_dieu is not None:
                dieu_has_content[current_dieu] = True
            continue

        hashes = line[: len(line) - len(line.lstrip("#"))]
        content = line[len(hashes) :].strip()
        heading_lines.append((line_number, hashes, content))

        if len(hashes) > 5:
            warnings.append(
                QcWarning(
                    code="heading_too_deep", detail=f"dòng {line_number}: {hashes}"
                )
            )

        if len(hashes) > len(content) and not content:
            continue

        if len(hashes) == 1:
            in_phu_luc = RE_PHU_LUC.match(content) is not None
            parent = "phu_luc" if in_phu_luc else f"phan:{content}"
            current_dieu = None
        elif len(hashes) == 4:
            seen_dieu = True
            current_dieu = content
            dieu_has_content.setdefault(content, False)
            match = RE_DIEU.match(content)
            if match is not None:
                dieu_numbers.append(match.group(1))
                parent = f"dieu:{match.group(1)}"
            else:
                parent = f"dieu:{content}"
        elif len(hashes) == 5:
            # Trong Phụ lục, "#####" nằm trực tiếp dưới "#" là hợp lệ — không có
            # ngoại lệ này thì mọi văn bản có Phụ lục đều báo cảnh báo giả.
            if not in_phu_luc and not seen_dieu and not reported_level_skip:
                warnings.append(
                    QcWarning(
                        code="heading_level_skip",
                        detail=f"dòng {line_number}: {content}",
                    )
                )
                reported_level_skip = True
            number = content.removeprefix("Khoản ").split(".")[0].strip()
            khoan_by_parent.setdefault(parent, []).append(number)
            if current_dieu is not None:
                dieu_has_content[current_dieu] = True

        if len(line) > HEADING_MAX_LEN:
            warnings.append(
                QcWarning(
                    code="suspicious_heading_length",
                    detail=f"dòng {line_number}: {len(line)} ký tự",
                )
            )

    if not heading_lines:
        warnings.append(QcWarning(code="no_heading", detail=""))

    for number in _monotonic_breaks(dieu_numbers):
        warnings.append(QcWarning(code="dieu_not_monotonic", detail=f"Điều {number}"))

    for parent_key, numbers in khoan_by_parent.items():
        for number in _monotonic_breaks(numbers):
            warnings.append(
                QcWarning(
                    code="khoan_not_monotonic", detail=f"{parent_key} -> khoản {number}"
                )
            )

    for dieu, has_content in dieu_has_content.items():
        if not has_content:
            warnings.append(QcWarning(code="empty_dieu", detail=dieu))

    for match in RE_FOOTNOTE_MARKER.finditer(markdown):
        # Marker còn sót ngoài blockquote/vùng dời là lỗi thật; trong hai vùng
        # đó thì "[n]" là nhãn cố ý.
        line_start = markdown.rfind("\n", 0, match.start()) + 1
        prefix = markdown[line_start : match.start()].lstrip()
        if prefix.startswith((">", "**")):
            continue
        warnings.append(QcWarning(code="orphan_footnote", detail=match.group(0)))

    return warnings
