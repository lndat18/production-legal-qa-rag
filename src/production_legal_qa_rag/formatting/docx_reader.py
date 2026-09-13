"""Đọc `.docx` bằng python-docx và trích xuất Block theo đúng thứ tự.

Duyệt ``document.element.body`` chứ không duyệt ``doc.paragraphs`` rồi
``doc.tables`` riêng — cách sau làm mất thứ tự xen kẽ giữa đoạn văn và bảng.
Style/bold của đoạn văn chỉ được giữ lại như tín hiệu phụ: rule nhận diện
heading không bao giờ đọc các giá trị này, vì style trong corpus thực tế
không nhất quán.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from production_legal_qa_rag.formatting.patterns import normalize_text
from production_legal_qa_rag.formatting.tables import table_to_markdown


@dataclass(frozen=True, slots=True)
class Block:
    """Một phần tử nội dung lấy từ DOCX, giữ đúng thứ tự xuất hiện."""

    kind: Literal["paragraph", "table"]
    text: str  # với table: markdown (hoặc HTML) đã render sẵn
    style: str | None = None  # tên style gốc, chỉ là tín hiệu phụ
    is_bold: bool = False


def _paragraph_is_bold(paragraph: Paragraph) -> bool:
    """Đoạn có in đậm toàn bộ hay không.

    Chỉ là tín hiệu phụ. Rule nhận diện heading không bao giờ được đọc giá trị
    này: phần lớn paragraph trong corpus có style "Normal" và bold không phủ
    đủ toàn đoạn.
    """
    runs = [run for run in paragraph.runs if run.text.strip()]
    return bool(runs) and all(run.bold for run in runs)


def read_docx(path: str | Path) -> list[Block]:
    """Đọc DOCX, giữ đúng thứ tự xen kẽ giữa paragraph và table.

    Args:
        path: Đường dẫn tới file `.docx`.

    Returns:
        Danh sách Block theo đúng thứ tự xuất hiện trong tài liệu gốc.
    """
    document = Document(str(path))
    blocks: list[Block] = []

    for element in document.element.body.iterchildren():
        if isinstance(element, CT_P):
            paragraph = Paragraph(element, document)
            text = normalize_text(paragraph.text)
            if not text:
                continue
            style = paragraph.style.name if paragraph.style is not None else None
            blocks.append(
                Block(
                    kind="paragraph",
                    text=text,
                    style=style,
                    is_bold=_paragraph_is_bold(paragraph),
                )
            )
        elif isinstance(element, CT_Tbl):
            markdown_table = table_to_markdown(Table(element, document))
            if markdown_table:
                blocks.append(Block(kind="table", text=markdown_table))

    return blocks
