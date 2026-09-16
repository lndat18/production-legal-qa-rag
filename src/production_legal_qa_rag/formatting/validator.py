"""QC trên markdown vùng nội dung ở giữa: heading-level-skip và bất thường khác.

Toàn bộ rule ở đây chỉ cảnh báo, không bao giờ làm fail file — kết quả trả
về là danh sách ``QcWarning`` để ``pipeline.py`` gom vào summary cuối cùng.
``validate()`` chỉ chạy trên markdown do ``emitter.emit`` sinh ra (vùng nội
dung ở giữa) — front matter/back matter do Gemini sinh không đi qua đây (mục
1.1, 3 spec), không còn khối YAML nào để bóc tách ở đầu file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from production_legal_qa_rag.formatting.models import QcWarning
from production_legal_qa_rag.formatting.patterns import (
    RE_DIEU,
    RE_FOOTNOTE_MARKER,
    RE_PHU_LUC,
    sort_key,
)

# Dòng heading dài hơn ngưỡng này bị cảnh báo suspicious_heading_length.
HEADING_MAX_LEN = 200


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


@dataclass
class _HeadingScan:
    """Dữ liệu thô thu được từ một lượt quét tuần tự các dòng heading.

    ``warnings`` chỉ gồm các cảnh báo BẮT BUỘC phát ngay tại dòng phát sinh
    (heading_too_deep, heading_level_skip, suspicious_heading_length) — thứ tự
    của chúng phụ thuộc vị trí dòng nên không thể tách rời khỏi lượt quét. Các
    trường còn lại là input cho những rule chạy sau khi quét xong (mục 3, 7
    spec: monotonic Điều/Khoản, empty_dieu).
    """

    warnings: list[QcWarning] = field(default_factory=list)
    heading_lines: list[tuple[int, str, str]] = field(default_factory=list)
    dieu_numbers: list[str] = field(default_factory=list)
    khoan_by_parent: dict[str, list[str]] = field(default_factory=dict)
    dieu_has_content: dict[str, bool] = field(default_factory=dict)


def _check_heading_too_deep(line_number: int, hashes: str) -> QcWarning | None:
    """Rule: heading sâu hơn cấp 5 (Khoản) là bất thường."""
    if len(hashes) <= 5:
        return None
    return QcWarning(code="heading_too_deep", detail=f"dòng {line_number}: {hashes}")


def _check_heading_level_skip(
    line_number: int, content: str, *, in_phu_luc: bool, seen_dieu: bool
) -> QcWarning | None:
    """Rule: heading cấp Khoản (`#####`) xuất hiện mà chưa qua Điều (`####`).

    Trong Phụ lục, "#####" nằm trực tiếp dưới "#" là hợp lệ — không có ngoại lệ
    này thì mọi văn bản có Phụ lục đều báo cảnh báo giả.
    """
    if in_phu_luc or seen_dieu:
        return None
    return QcWarning(code="heading_level_skip", detail=f"dòng {line_number}: {content}")


def _check_suspicious_heading_length(line_number: int, line: str) -> QcWarning | None:
    """Rule: dòng heading dài bất thường, dấu hiệu gộp nhầm nội dung."""
    if len(line) <= HEADING_MAX_LEN:
        return None
    return QcWarning(
        code="suspicious_heading_length",
        detail=f"dòng {line_number}: {len(line)} ký tự",
    )


def _scan_headings(lines: list[str]) -> _HeadingScan:
    """Quét tuần tự các dòng, thu thập dữ liệu thô cho toàn bộ rule QC.

    Một lượt quét duy nhất vì nhiều rule phụ thuộc trạng thái tuần tự: cấp
    heading hiện tại (``parent``), Điều đang mở (``current_dieu``), có đang ở
    trong Phụ lục hay không. ``seen_dieu`` là sticky trong thân văn bản: Chương
    /Mục không reset nó — level trên bị bỏ trống là hợp lệ.
    """
    scan = _HeadingScan()

    in_phu_luc = False
    seen_dieu = False
    reported_level_skip = False
    in_code_or_table = False
    parent = "<none>"
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
                scan.dieu_has_content[current_dieu] = True
            continue

        if not line.startswith("#"):
            if line.strip() and current_dieu is not None:
                scan.dieu_has_content[current_dieu] = True
            continue

        hashes = line[: len(line) - len(line.lstrip("#"))]
        content = line[len(hashes) :].strip()
        scan.heading_lines.append((line_number, hashes, content))

        too_deep = _check_heading_too_deep(line_number, hashes)
        if too_deep is not None:
            scan.warnings.append(too_deep)

        if len(hashes) > len(content) and not content:
            continue

        if len(hashes) == 1:
            in_phu_luc = RE_PHU_LUC.match(content) is not None
            parent = "phu_luc" if in_phu_luc else f"phan:{content}"
            current_dieu = None
        elif len(hashes) == 4:
            seen_dieu = True
            current_dieu = content
            scan.dieu_has_content.setdefault(content, False)
            match = RE_DIEU.match(content)
            if match is not None:
                scan.dieu_numbers.append(match.group(1))
                parent = f"dieu:{match.group(1)}"
            else:
                parent = f"dieu:{content}"
        elif len(hashes) == 5:
            if not reported_level_skip:
                skip_warning = _check_heading_level_skip(
                    line_number, content, in_phu_luc=in_phu_luc, seen_dieu=seen_dieu
                )
                if skip_warning is not None:
                    scan.warnings.append(skip_warning)
                    reported_level_skip = True
            number = content.removeprefix("Khoản ").split(".")[0].strip()
            scan.khoan_by_parent.setdefault(parent, []).append(number)
            if current_dieu is not None:
                scan.dieu_has_content[current_dieu] = True

        length_warning = _check_suspicious_heading_length(line_number, line)
        if length_warning is not None:
            scan.warnings.append(length_warning)

    return scan


def _check_no_heading(heading_lines: list[tuple[int, str, str]]) -> list[QcWarning]:
    """Rule: file không có heading nào là dấu hiệu parser lỗi nặng."""
    if heading_lines:
        return []
    return [QcWarning(code="no_heading", detail="")]


def _check_dieu_monotonic(dieu_numbers: list[str]) -> list[QcWarning]:
    """Rule: số hiệu Điều phải tăng dần xuyên suốt văn bản."""
    return [
        QcWarning(code="dieu_not_monotonic", detail=f"Điều {number}")
        for number in _monotonic_breaks(dieu_numbers)
    ]


def _check_khoan_monotonic(khoan_by_parent: dict[str, list[str]]) -> list[QcWarning]:
    """Rule: số hiệu Khoản phải tăng dần trong cùng một Điều/Phụ lục cha."""
    warnings: list[QcWarning] = []
    for parent_key, numbers in khoan_by_parent.items():
        for number in _monotonic_breaks(numbers):
            warnings.append(
                QcWarning(
                    code="khoan_not_monotonic", detail=f"{parent_key} -> khoản {number}"
                )
            )
    return warnings


def _check_empty_dieu(dieu_has_content: dict[str, bool]) -> list[QcWarning]:
    """Rule: Điều không có nội dung nào (chỉ trơ heading) là dấu hiệu cắt nhầm."""
    return [
        QcWarning(code="empty_dieu", detail=dieu)
        for dieu, has_content in dieu_has_content.items()
        if not has_content
    ]


def _check_orphan_footnotes(markdown: str) -> list[QcWarning]:
    """Rule: marker chú thích "[n]" còn sót lại trong vùng nội dung ở giữa.

    ``emitter._strip_block_markers`` đã gỡ marker khỏi mọi block paragraph
    trước khi sinh markdown (``patterns.strip_markers``) — nhưng KHÔNG áp
    dụng cho bảng (marker trong bảng là dấu hiệu bất thường, không phải nội
    dung cần làm sạch, xem ``emitter.py``). Nếu marker vẫn còn ở đây, đó là
    dấu hiệu ``strip_markers`` bỏ sót một dạng marker nào đó, hoặc marker nằm
    trong bảng — cả hai đều đáng rà lại thủ công.
    """
    return [
        QcWarning(code="orphan_footnote", detail=match.group(0))
        for match in RE_FOOTNOTE_MARKER.finditer(markdown)
    ]


def validate(markdown: str) -> list[QcWarning]:
    """Chạy toàn bộ rule QC trên markdown vùng nội dung ở giữa.

    Compose lại từ một lượt quét tuần tự (`_scan_headings`, bắt buộc vì nhiều
    rule phụ thuộc trạng thái vị trí) và các rule độc lập chạy sau đó, đúng
    theo thứ tự đã có trước khi tách hàm (heading-level rule trong lúc quét,
    rồi no_heading, dieu/khoan monotonic, empty_dieu, orphan_footnote).
    """
    scan = _scan_headings(markdown.splitlines())

    warnings = list(scan.warnings)
    warnings.extend(_check_no_heading(scan.heading_lines))
    warnings.extend(_check_dieu_monotonic(scan.dieu_numbers))
    warnings.extend(_check_khoan_monotonic(scan.khoan_by_parent))
    warnings.extend(_check_empty_dieu(scan.dieu_has_content))
    warnings.extend(_check_orphan_footnotes(markdown))
    return warnings
