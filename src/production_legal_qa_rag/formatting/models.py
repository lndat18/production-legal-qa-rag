"""Pydantic models trao đổi giữa các module của bước formatting.

Định nghĩa ba kiểu dữ liệu công khai: ``QcWarning`` (một cảnh báo QC không
bao giờ làm fail file), ``FrontMatter`` (metadata cấp văn bản, render được
thành YAML) và ``FormattingResult`` (kết quả cuối cùng của
``pipeline.convert_docx_to_markdown``, dùng bởi cả CLI lẫn package khác gọi
vào).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from production_legal_qa_rag.formatting.patterns import PARSER_VERSION


class QcWarning(BaseModel):
    """Một cảnh báo QC. Không bao giờ làm fail file.

    Tên có tiền tố ``Qc`` là cố ý: ``Warning`` trần sẽ che builtin exception.
    """

    code: str
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
