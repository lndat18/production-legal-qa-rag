"""Pydantic models trao đổi giữa các module của `chunking/` (mục 2, 10).

`KhoanNode`/`DocumentTree` là kết quả trung gian của `parser.py`; `Chunk`/
`ChunkingResult` là kết quả cuối cùng ghi ra `data/chunks/*.jsonl`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class KhoanNode(BaseModel):
    """1 Khoản đã dựng từ heading Markdown, kèm nội dung thô (chưa cắt).

    ``khoan_number`` là ``None`` khi đây là nội dung nằm trực tiếp dưới 1
    Điều mà không có heading `##### Khoản N` riêng — Điều không chia Khoản
    (vd. Điều 46/47 `Luật bảo hiểm y tế.md`), hoặc đoạn mở đầu đứng trước
    Khoản đầu tiên của 1 Điều (vd. Điều 48a). Đây là "Khoản ngầm định" — spec
    không định nghĩa trường hợp này, quyết định thiết kế để không mất nội
    dung thật trong corpus (xem báo cáo bàn giao).
    """

    breadcrumb_prefix: str
    khoan_number: str | None = None
    content: str
    has_table: bool = False
    raw_table: str | None = None


class DocumentTree(BaseModel):
    """Cây breadcrumb -> Khoản của 1 file markdown đã parse (mục 10)."""

    source_document: str
    khoans: list[KhoanNode] = Field(default_factory=list)


class Chunk(BaseModel):
    """1 chunk sẵn sàng ghi ra `data/chunks/*.jsonl` (mục 2)."""

    chunk_id: str
    source_document: str
    breadcrumb: str
    content: str
    token_count: int
    has_table: bool = False
    raw_table: str | None = None
    standardization_table: str | None = None
    is_split: bool = False
    split_index: int | None = None
    split_total: int | None = None
    negation_note: str | None = None


class ChunkingResult(BaseModel):
    """Kết quả chunk hoá 1 file markdown."""

    source_path: str
    chunks: list[Chunk] = Field(default_factory=list)
    khoan_count: int = 0
    split_khoan_count: int = 0
