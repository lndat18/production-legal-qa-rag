"""Nhận diện loại bảng theo nội dung và render bảng sang Markdown/HTML.

Bảng trong văn bản pháp luật được phân loại theo NỘI DUNG, không theo vị trí:
bảng chữ ký của Nghị định 293 nằm ở giữa văn bản, toàn bộ Phụ lục nằm sau nó.
Module này vừa render bảng DOCX sang Markdown/HTML (``table_to_markdown``),
vừa tách bảng quốc hiệu/loại bỏ bảng nhiễu (``triage_tables``), vừa parse
ngược bảng Markdown dạng pipe về ma trận ô (``parse_pipe_table``) để
``frontmatter.py`` trích metadata từ bảng quốc hiệu.
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from docx.table import Table, _Cell

from production_legal_qa_rag.formatting.models import QcWarning
from production_legal_qa_rag.formatting.patterns import (
    RE_ATTACHMENT_TABLE,
    RE_QUOC_HIEU_CELL,
    RE_SIGNATURE_CELL,
    normalize_text,
)

if TYPE_CHECKING:
    # Chỉ dùng cho type hint: import thật sẽ tạo vòng lặp vì docx_reader gọi
    # ngược lại table_to_markdown khi đọc DOCX.
    from production_legal_qa_rag.formatting.docx_reader import Block

# Số block đầu văn bản còn được coi là vùng có thể chứa bảng quốc hiệu.
_QUOC_HIEU_SEARCH_LIMIT = 3


def _cell_lines(cell: _Cell) -> list[str]:
    """Lấy các dòng có nội dung trong một ô Word."""
    return [
        normalize_text(line)
        for paragraph in cell.paragraphs
        for line in paragraph.text.splitlines()
        if line.strip()
    ]


def _cell_to_markdown(cell: _Cell) -> str:
    """Chuyển nội dung một ô Word thành nội dung hợp lệ trong bảng Markdown."""
    return "<br>".join(_cell_lines(cell)).replace("|", r"\|")


def _single_row_table_to_html(table: Table) -> str:
    """Xuất bảng một hàng không có header bằng HTML hợp lệ trong Markdown.

    Sau khi bảng chữ ký bị loại ở bước triage, các bảng một hàng còn lại là
    bảng công thức (vd. "Tiền lương làm thêm giờ = ... x ..."). Bảng pipe sẽ
    đẩy số hạng đầu tiên lên làm header, nên dùng HTML.
    """
    cells = [
        f"      <td>{'<br>'.join(escape(line) for line in _cell_lines(cell))}</td>"
        for cell in table.rows[0].cells
    ]
    return "\n".join(
        [
            "<table>",
            "  <tbody>",
            "    <tr>",
            *cells,
            "    </tr>",
            "  </tbody>",
            "</table>",
        ]
    )


def table_to_markdown(table: Table) -> str:
    """Chuyển bảng DOCX sang cú pháp bảng phù hợp trong Markdown."""
    rows = [[_cell_to_markdown(cell) for cell in row.cells] for row in table.rows]

    if not rows:
        return ""

    if len(rows) == 1:
        return _single_row_table_to_html(table)

    column_count = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (column_count - len(row)) for row in rows]

    header = normalized_rows[0]
    body_rows = normalized_rows[1:]
    separator = ["---"] * column_count

    return "\n".join(
        f"| {' | '.join(row)} |" for row in (header, separator, *body_rows)
    )


def parse_pipe_table(markdown: str) -> list[list[str]]:
    """Tách bảng Markdown dạng pipe về lại ma trận ô."""
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= {"-", " "} and cell for cell in cells):
            continue  # dòng phân cách header
        rows.append(cells)
    return rows


def triage_tables(
    blocks: list[Block],
) -> tuple[Block | None, list[Block], list[QcWarning]]:
    """Tách bảng quốc hiệu và loại bảng nhiễu.

    Nhận diện theo NỘI DUNG, không theo vị trí: bảng chữ ký của Nghị định 293
    nằm ở block 39/170, toàn bộ Phụ lục nằm sau nó. Hai trong sáu bảng chữ ký
    lại có ô đầu rỗng nên riêng "Nơi nhận:" không đủ.

    Args:
        blocks: Toàn bộ Block đọc được từ DOCX, theo đúng thứ tự xuất hiện.

    Returns:
        Bộ ba ``(bảng quốc hiệu hoặc None, block còn giữ lại, cảnh báo QC)``.
    """
    warnings: list[QcWarning] = []
    quoc_hieu: Block | None = None
    kept: list[Block] = []

    for index, block in enumerate(blocks):
        if block.kind != "table":
            kept.append(block)
            continue

        if (
            quoc_hieu is None
            and index < _QUOC_HIEU_SEARCH_LIMIT
            and RE_QUOC_HIEU_CELL.search(block.text)
        ):
            quoc_hieu = block
            continue

        if RE_SIGNATURE_CELL.search(block.text):
            warnings.append(
                QcWarning(code="dropped_noi_nhan_table", detail=f"block {index}")
            )
            continue

        if RE_ATTACHMENT_TABLE.search(block.text):
            warnings.append(
                QcWarning(code="dropped_attachment_table", detail=f"block {index}")
            )
            continue

        kept.append(block)

    if quoc_hieu is None:
        warnings.append(QcWarning(code="missing_quoc_hieu_table", detail=""))

    return quoc_hieu, kept, warnings
