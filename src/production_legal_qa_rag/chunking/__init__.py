"""Bước chunking: chuyển Markdown có cấu trúc thành chunk cấp Khoản.

Xem `chunking_spec.md` trong package này để biết chi tiết thuật toán cắt
Khoản, xử lý bảng, đếm token và cấu trúc breadcrumb.
"""

from __future__ import annotations

from production_legal_qa_rag.chunking.models import Chunk, ChunkingResult, DocumentTree
from production_legal_qa_rag.chunking.pipeline import (
    convert_directory,
    convert_markdown_to_chunks,
)

__all__ = [
    "Chunk",
    "ChunkingResult",
    "DocumentTree",
    "convert_directory",
    "convert_markdown_to_chunks",
]
