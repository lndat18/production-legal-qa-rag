"""Nhận diện loại bảng theo nội dung và render bảng sang Markdown/HTML.

Bảng trong văn bản pháp luật được phân loại theo NỘI DUNG, không theo vị trí:
bảng chữ ký của Nghị định 293 nằm ở giữa văn bản, toàn bộ Phụ lục nằm sau nó.
Module này render bảng DOCX sang Markdown/HTML (``table_to_markdown``), nhận
diện bảng chữ ký (``is_signature_table`` — dùng để xác định biên back matter,
``backmatter.py`` mục 1.1 spec) và loại bảng đính kèm khỏi vùng nội dung ở
giữa (``filter_middle_tables``). Không còn phân loại bảng "quốc hiệu" — bảng
đó giờ nằm trong vùng front matter, Gemini xử lý nguyên khối cùng các block
khác (mục 1.1, 6 spec).
"""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

from docx.table import Table, _Cell

from production_legal_qa_rag.formatting.models import QcWarning
from production_legal_qa_rag.formatting.patterns import (
    RE_ATTACHMENT_TABLE,
    RE_SIGNATURE_CELL,
    normalize_text,
)

if TYPE_CHECKING:
    # Chỉ dùng cho type hint: import thật sẽ tạo vòng lặp vì docx_reader gọi
    # ngược lại table_to_markdown khi đọc DOCX.
    from production_legal_qa_rag.formatting.docx_reader import Block


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

    Bảng một hàng trong corpus là bảng công thức (vd. "Tiền lương làm thêm
    giờ = ... x ..."). Bảng pipe sẽ đẩy số hạng đầu tiên lên làm header, nên
    dùng HTML.
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


def is_signature_table(block: Block) -> bool:
    """Bảng có phải bảng chữ ký hay không (nhận diện theo nội dung).

    Dùng để xác định biên back matter (``backmatter.find_boundary``, mục 1.1
    spec) — KHÔNG dùng để phân loại "quốc hiệu" (đã bỏ hoàn toàn).
    """
    return block.kind == "table" and bool(RE_SIGNATURE_CELL.search(block.text))


def filter_middle_tables(blocks: list[Block]) -> tuple[list[Block], list[QcWarning]]:
    """Loại bảng đính kèm khỏi vùng nội dung ở giữa.

    Chỉ xử lý bảng đính kèm ("FILE ĐƯỢC ĐÍNH KÈM..."): bảng chữ ký đã được
    cắt khỏi vùng này từ trước bởi ``backmatter.split_backmatter`` (mục 1.1
    spec) — không lặp lại việc nhận diện ở đây.

    Args:
        blocks: Block thuộc vùng nội dung ở giữa, đã cắt front/back matter.

    Returns:
        Cặp ``(block còn giữ lại, cảnh báo QC)``.
    """
    warnings: list[QcWarning] = []
    kept: list[Block] = []

    for index, block in enumerate(blocks):
        if block.kind == "table" and RE_ATTACHMENT_TABLE.search(block.text):
            warnings.append(
                QcWarning(code="dropped_attachment_table", detail=f"block {index}")
            )
            continue
        kept.append(block)

    return kept, warnings
