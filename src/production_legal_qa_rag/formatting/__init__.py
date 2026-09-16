"""Bước formatting: chuyển văn bản pháp luật `.docx` sang Markdown có cấu trúc.

Xem `formatting_spec.md` trong package này để biết chi tiết heading mapping,
front matter/back matter (chuyển bằng Groq) và QC warnings.
"""

from __future__ import annotations

from production_legal_qa_rag.formatting.models import FormattingResult, QcWarning
from production_legal_qa_rag.formatting.pipeline import (
    convert_directory,
    convert_docx_to_markdown,
)

__all__ = [
    "FormattingResult",
    "QcWarning",
    "convert_directory",
    "convert_docx_to_markdown",
]
