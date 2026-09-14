"""Chuyển bảng markdown (`raw_table`) thành text chuẩn hoá để embed (mục 5.3).

Bảng gốc do `formatting/tables.py` render dạng GFM chuẩn (`| ... |`, dòng
phân cách `| --- | ... |`), tiêu đề cột có thể chứa `<br>` để xuống dòng.

`formatting/tables.py::_single_row_table_to_html` còn render riêng bảng công
thức 1 hàng (vd. "Tiền lương làm thêm giờ = ... x ...") bằng HTML thô
(``<table>...</table>``) thay vì pipe table — mục 5.3 của spec không mô tả
dạng này, `standardize_table` tự nhận diện và xử lý thêm (quyết định thiết
kế, xem báo cáo bàn giao): nối text các ô `<td>` theo đúng thứ tự bằng dấu
cách, giữ nguyên thứ tự đọc trái sang phải của công thức gốc.
"""

from __future__ import annotations

import html
import re

_RE_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_RE_HTML_CELL = re.compile(r"<td>(.*?)</td>", re.IGNORECASE | re.DOTALL)


def parse_pipe_table(markdown: str) -> list[list[str]]:
    """Tách bảng Markdown dạng pipe về lại ma trận ô, bỏ dòng phân cách.

    Hàng đầu tiên trả về là tiêu đề cột; các hàng sau là dữ liệu, giữ nguyên
    thứ tự xuất hiện.
    """
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= {"-", " ", ":"} and cell for cell in cells):
            continue  # dòng phân cách header (hỗ trợ cả ":---"/"---:" căn lề)
        rows.append(cells)
    return rows


def _standardize_html_table(raw_table: str) -> str:
    """Chuẩn hoá bảng công thức 1 hàng dạng HTML thô — nối các ô `<td>`."""
    parts: list[str] = []
    for cell in _RE_HTML_CELL.findall(raw_table):
        text = html.unescape(_RE_BR.sub(" ", cell)).strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def standardize_table(raw_table: str) -> str:
    """Chuyển `raw_table` thành text theo mục 5.3.

    Nối cột đầu tiên làm nhãn dòng, các cột còn lại nối theo dạng
    ``{tên cột}: {giá trị}``, các phần nối bằng " - "; mỗi dòng dữ liệu
    thành 1 dòng text, các dòng cách nhau bằng ký tự xuống dòng. Tiêu đề cột
    có `<br>` thì bỏ `<br>` (nối liền, không thêm khoảng trắng).

    Nếu `raw_table` là bảng công thức 1 hàng dạng HTML thô (không phải spec
    literal, xem docstring module), dùng `_standardize_html_table` thay thế.
    """
    if raw_table.lstrip().lower().startswith("<table"):
        return _standardize_html_table(raw_table)

    rows = parse_pipe_table(raw_table)
    if not rows:
        return ""

    header = [_RE_BR.sub("", cell) for cell in rows[0]]

    lines: list[str] = []
    for row in rows[1:]:
        if not row:
            continue
        sentence_parts = [row[0]]
        for index in range(1, len(row)):
            column_name = header[index] if index < len(header) else ""
            sentence_parts.append(f"{column_name}: {row[index]}")
        lines.append(" - ".join(sentence_parts))

    return "\n".join(lines)
