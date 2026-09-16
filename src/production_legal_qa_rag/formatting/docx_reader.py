"""Đọc `.docx` bằng python-docx và trích xuất Block theo đúng thứ tự.

Duyệt ``document.element.body`` chứ không duyệt ``doc.paragraphs`` rồi
``doc.tables`` riêng — cách sau làm mất thứ tự xen kẽ giữa đoạn văn và bảng.
Style/bold/nghiêng của đoạn văn được giữ lại như tín hiệu phụ: rule nhận
diện heading (mục 3 spec) không bao giờ đọc các giá trị này, vì style trong
corpus thực tế không nhất quán; tín hiệu này chỉ dùng để serialize block cho
Groq khi chuyển đổi front matter/back matter (``serialize_blocks_for_llm``,
mục 1.1 spec).

Cũng chứa ``chunk_blocks_for_llm`` — dồn Block tuần tự thành từng chunk theo
ước lượng token (heuristic ký tự, mục 1.2 spec), phục vụ cơ chế chunking +
rate limiting khi gửi front matter/back matter cho Groq free tier.
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
    is_italic: bool = False


def _paragraph_all_runs_have(paragraph: Paragraph, attribute: str) -> bool:
    """Mọi run (có chữ) của đoạn có bật thuộc tính ``attribute`` hay không.

    Dùng chung cho bold/italic. Chỉ là tín hiệu phụ. Rule nhận diện heading
    không bao giờ được đọc giá trị này: phần lớn paragraph trong corpus có
    style "Normal" và bold/italic không phủ đủ toàn đoạn.
    """
    runs = [run for run in paragraph.runs if run.text.strip()]
    return bool(runs) and all(getattr(run, attribute) for run in runs)


def _paragraph_is_bold(paragraph: Paragraph) -> bool:
    """Đoạn có in đậm toàn bộ hay không."""
    return _paragraph_all_runs_have(paragraph, "bold")


def _paragraph_is_italic(paragraph: Paragraph) -> bool:
    """Đoạn có in nghiêng toàn bộ hay không."""
    return _paragraph_all_runs_have(paragraph, "italic")


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
                    is_italic=_paragraph_is_italic(paragraph),
                )
            )
        elif isinstance(element, CT_Tbl):
            markdown_table = table_to_markdown(Table(element, document))
            if markdown_table:
                blocks.append(Block(kind="table", text=markdown_table))

    return blocks


# Hệ số ước lượng token theo mục 1.2 spec: bảo thủ hơn mức phổ biến 4 ký
# tự/token của tiếng Anh, vì tiếng Việt có dấu thường tách nhiều subword
# token hơn. Chỉ dùng để quyết định ranh giới chunk trước khi gửi, KHÔNG
# dùng để track budget rate-limit thật (xem `llm_client.py`, đọc
# `usage.total_tokens` từ response sau khi gọi thành công).
_CHARS_PER_TOKEN = 2.5


def _estimate_block_tokens(block: Block) -> float:
    """Ước lượng token của một Block bằng heuristic ký tự (mục 1.2 spec)."""
    return len(block.text) / _CHARS_PER_TOKEN


def chunk_blocks_for_llm(blocks: list[Block], token_limit: int) -> list[list[Block]]:
    """Dồn Block tuần tự thành từng chunk, không bao giờ cắt giữa 1 Block.

    Dùng để chia nhỏ front matter/back matter trước khi gửi từng chunk cho
    Groq (mục 1.2 spec) — free tier có giới hạn TPM thấp hơn nhiều so với
    kích thước back matter của một số văn bản. Mỗi Block được dồn tuần tự
    vào chunk hiện tại; khi thêm một Block khiến tổng ước lượng token của
    chunk vượt ``token_limit``, chunk hiện tại (không kèm Block đó) được
    chốt lại và Block đó mở đầu chunk kế tiếp. Một Block tự nó đã vượt
    ``token_limit`` vẫn được giữ nguyên trong 1 chunk riêng (không có cách
    nào chia nhỏ hơn mà không cắt giữa Block).

    Args:
        blocks: Danh sách Block theo đúng thứ tự gốc (front matter hoặc
            back matter, đã cắt biên từ trước).
        token_limit: Ngưỡng ước lượng token tối đa mỗi chunk.

    Returns:
        Danh sách các chunk (mỗi chunk là ``list[Block]``), giữ đúng thứ tự
        gốc. Rỗng nếu ``blocks`` rỗng.
    """
    if not blocks:
        return []

    chunks: list[list[Block]] = []
    current_chunk: list[Block] = []
    current_tokens = 0.0

    for block in blocks:
        block_tokens = _estimate_block_tokens(block)
        if current_chunk and current_tokens + block_tokens > token_limit:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0.0
        current_chunk.append(block)
        current_tokens += block_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def serialize_blocks_for_llm(blocks: list[Block]) -> str:
    """Render Block thành text kèm tín hiệu bold/nghiêng, dùng làm input cho Groq.

    Paragraph in đậm/nghiêng toàn dòng được bọc sẵn ``**``/``*`` (cả hai cùng
    lúc thì bọc ``***``) — Groq chỉ cần giữ nguyên định dạng khi chuyển đổi
    (mục 1.1 spec), không phải tự đoán từ văn bản thô. Bảng đã có sẵn
    markdown/HTML từ ``tables.table_to_markdown``, giữ nguyên văn.

    Args:
        blocks: Block front matter hoặc back matter, theo đúng thứ tự gốc.

    Returns:
        Text đã ghép, mỗi block cách nhau một dòng trống.
    """
    lines: list[str] = []
    for block in blocks:
        if block.kind == "table":
            lines.append(block.text)
            continue
        text = block.text
        if block.is_bold and block.is_italic:
            text = f"***{text}***"
        elif block.is_bold:
            text = f"**{text}**"
        elif block.is_italic:
            text = f"*{text}*"
        lines.append(text)
    return "\n\n".join(lines)
