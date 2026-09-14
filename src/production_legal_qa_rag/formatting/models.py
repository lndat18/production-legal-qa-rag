"""Pydantic models trao đổi giữa các module của bước formatting.

Định nghĩa các kiểu dữ liệu công khai: ``QcWarning`` (một cảnh báo QC không
bao giờ làm fail file), ``FrontMatter`` (metadata cấp văn bản, render được
thành YAML), ``FormattingResult`` (kết quả cuối cùng của
``pipeline.convert_docx_to_markdown``, dùng bởi cả CLI lẫn package khác gọi
vào) và hai schema structured-output cho LLM (``FrontMatterExtraction``,
``FootnoteExtraction``) dùng bởi ``llm_client.extract_structured`` — xem
``formatting_spec.md`` mục 1, 6.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from production_legal_qa_rag.formatting.patterns import PARSER_VERSION

# Toàn bộ mã QcWarning mà package này phát ra, gom một chỗ để mypy bắt được
# typo (vd. gõ nhầm "orphan_foot_note") ngay lúc type-check thay vì phải chờ
# chạy test rồi so KNOWN_WARNING_CODES trong `tests/test_formatting.py`.
# Cập nhật danh sách này trong cùng lúc thêm rule QC mới ở bất kỳ module nào.
QcWarningCode = Literal[
    "orphan_footnote",
    "long_footnote_deferred",
    "unused_footnote",
    "dropped_noi_nhan_table",
    "dropped_attachment_table",
    "missing_quoc_hieu_table",
    "missing_frontmatter_field",
    "missing_optional_frontmatter_field",
    "ambiguous_footnote_region",
    "footnote_number_gap",
    "footnote_marker_in_table",
    "heading_too_deep",
    "heading_level_skip",
    "suspicious_heading_length",
    "no_heading",
    "dieu_not_monotonic",
    "khoan_not_monotonic",
    "empty_dieu",
    "llm_frontmatter_extraction_failed",
    "llm_footnote_extraction_failed",
    "llm_frontmatter_mismatch",
    "llm_footnote_count_mismatch",
]


class QcWarning(BaseModel):
    """Một cảnh báo QC. Không bao giờ làm fail file.

    Tên có tiền tố ``Qc`` là cố ý: ``Warning`` trần sẽ che builtin exception.
    """

    code: QcWarningCode
    detail: str = ""


class FrontMatter(BaseModel):
    """Metadata cấp văn bản, render thành YAML ở đầu file .md."""

    so_hieu: str | None = None
    loai_van_ban: str | None = None
    ten_van_ban: str | None = None
    co_quan_ban_hanh: str | None = None
    ngay_ban_hanh: str | None = None
    ngay_hieu_luc: str | None = None
    is_van_ban_hop_nhat: bool = False
    is_phu_luc: bool = False
    source_path: str | None = None
    parser_version: str = PARSER_VERSION

    def to_yaml(self) -> str:
        """Render YAML bằng tay để output ổn định theo byte.

        Không dùng ``yaml.safe_dump``: nó sắp lại khóa và escape tiếng Việt,
        làm output đối chiếu nhiễu mỗi lần nâng thư viện.
        """
        lines = ["---"]
        for key, value in (
            ("so_hieu", self.so_hieu),
            ("loai_van_ban", self.loai_van_ban),
            ("ten_van_ban", self.ten_van_ban),
            ("co_quan_ban_hanh", self.co_quan_ban_hanh),
            ("ngay_ban_hanh", self.ngay_ban_hanh),
            ("ngay_hieu_luc", self.ngay_hieu_luc),
            ("is_van_ban_hop_nhat", self.is_van_ban_hop_nhat),
            ("is_phu_luc", self.is_phu_luc),
            ("source_path", self.source_path),
            ("parser_version", self.parser_version),
        ):
            lines.append(f"{key}: {_yaml_scalar(value)}")
        lines.append("---")
        return "\n".join(lines)


def _yaml_scalar(value: str | bool | None) -> str:
    """Render một giá trị scalar Python thành cú pháp YAML tương ứng."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class FormattingResult(BaseModel):
    """Kết quả chuyển đổi một DOCX sang Markdown có cấu trúc."""

    markdown: str
    front_matter: FrontMatter
    warnings: list[QcWarning] = Field(default_factory=list)


class FrontMatterExtraction(BaseModel):
    """Structured output LLM đọc giá trị các field front matter.

    Đường chính (formatting_spec.md mục 1, 6) thay cho các hàm regex trong
    ``frontmatter.py`` (nay là baseline: fallback khi field ở đây là
    ``None``, và để so sánh phát ``QcWarning`` khi lệch nhau). ``None``
    nghĩa là LLM không tìm thấy giá trị trong văn bản, không phải lỗi gọi
    API — lỗi gọi API được ``llm_client.extract_structured`` bắt riêng và
    trả nguyên ``None`` cho cả object.
    """

    so_hieu: str | None = Field(
        default=None,
        description="Số hiệu văn bản, giữ nguyên định dạng gốc, vd. '293/2025/NĐ-CP'.",
    )
    loai_van_ban: str | None = Field(
        default=None,
        description="Loại văn bản, vd. 'Nghị định', 'Luật', 'Thông tư', 'Nghị quyết'.",
    )
    ten_van_ban: str | None = Field(default=None, description="Tên đầy đủ của văn bản.")
    co_quan_ban_hanh: str | None = Field(
        default=None,
        description="Tên cơ quan ban hành văn bản, giữ nguyên viết hoa gốc.",
    )
    ngay_ban_hanh: str | None = Field(
        default=None, description="Ngày ban hành văn bản, định dạng YYYY-MM-DD."
    )
    ngay_hieu_luc: str | None = Field(
        default=None,
        description="Ngày văn bản có hiệu lực thi hành, định dạng YYYY-MM-DD.",
    )


class FootnoteEntry(BaseModel):
    """Một chú thích sửa đổi riêng lẻ trong ``FootnoteExtraction.entries``."""

    number: int = Field(description="Số hiệu chú thích, vd. 1 cho '[1]'.")
    content: str = Field(description="Nội dung đầy đủ của chú thích, giữ nguyên văn.")


class FootnoteExtraction(BaseModel):
    """Structured output LLM đọc nội dung toàn bộ chú thích sửa đổi.

    Đường chính (formatting_spec.md mục 1, 6) thay cho ``parse_region``
    (regex, nay là baseline) để đọc **nội dung** từng chú thích trong vùng
    chú thích ở cuối văn bản; **vị trí** vùng chú thích vẫn do
    ``find_region_start`` xác định, LLM không tham gia bước đó.
    """

    entries: list[FootnoteEntry] = Field(
        default_factory=list,
        description=(
            "Danh sách chú thích tìm được trong vùng văn bản, mỗi chú thích "
            "một số hiệu và nội dung đầy đủ."
        ),
    )
