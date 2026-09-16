"""Pydantic models trao đổi giữa các module của bước formatting.

Định nghĩa các kiểu dữ liệu công khai: ``QcWarning`` (một cảnh báo QC không
bao giờ làm fail file) và ``FormattingResult`` (kết quả cuối cùng của
``pipeline.convert_docx_to_markdown``, dùng bởi cả CLI lẫn package khác gọi
vào) — xem ``formatting_spec.md`` mục 1.1, 6.

Không còn ``FrontMatter``/``FrontMatterExtraction``/``BackMatterExtraction``:
thiết kế mới (mục 1.1) không trích field có cấu trúc, không sinh YAML — front
matter/back matter là text markdown thuần do Groq sinh, ghép trực tiếp vào
file `.md`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Toàn bộ mã QcWarning mà package này phát ra, gom một chỗ để mypy bắt được
# typo ngay lúc type-check thay vì phải chờ chạy test rồi so KNOWN_WARNING_CODES
# trong `tests/test_formatting.py`. Cập nhật danh sách này trong cùng lúc
# thêm rule QC mới ở bất kỳ module nào.
QcWarningCode = Literal[
    "dropped_noi_nhan_table",
    "dropped_attachment_table",
    "heading_too_deep",
    "heading_level_skip",
    "suspicious_heading_length",
    "no_heading",
    "dieu_not_monotonic",
    "khoan_not_monotonic",
    "empty_dieu",
    "orphan_footnote",
    "llm_frontmatter_conversion_failed",
    "llm_backmatter_conversion_failed",
]


class QcWarning(BaseModel):
    """Một cảnh báo QC. Không bao giờ làm fail file.

    Tên có tiền tố ``Qc`` là cố ý: ``Warning`` trần sẽ che builtin exception.
    """

    code: QcWarningCode
    detail: str = ""


class FormattingResult(BaseModel):
    """Kết quả chuyển đổi một DOCX sang Markdown có cấu trúc."""

    markdown: str
    warnings: list[QcWarning] = Field(default_factory=list)
