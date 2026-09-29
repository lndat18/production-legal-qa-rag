"""Model dữ liệu cho golden testset và tiến độ sinh (evaluation_spec.md mục 5).

Chỉ phục vụ Phase 1 (sinh testset); không lẫn với contract của `retrieval/`
(`RetrievedChunk`) hay bất kỳ package nào khác trong pipeline chính.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class GoldenTestCase(BaseModel):
    """Một câu hỏi mẫu trong golden testset, sinh tự động bởi ragas.

    Giữ đúng 3 cột bắt buộc theo schema ``EvaluationDataset`` của ragas
    (``user_input``, ``reference``, ``reference_contexts``) để Phase 2 dùng
    thẳng không cần convert; ``synthesizer_name`` là cột phụ giữ lại để biết
    loại câu hỏi (single-hop/multi-hop) lúc review bằng mắt.
    ``source_document``/``source_section`` không do ragas trả: `generate` biết
    đang xử lý đơn vị nào nên tự gắn vào (mục 5).
    """

    user_input: str
    reference: str
    reference_contexts: list[str]
    synthesizer_name: str | None = None
    source_document: str | None = None  # tên file .md nguồn
    source_section: str | None = None  # tiêu đề Chương (hoặc các Chương gộp)

    def empty_required_fields(self) -> list[str]:
        """Tên các cột bắt buộc đang rỗng (câu hỏi/đáp án/ngữ cảnh trống hoặc chỉ khoảng trắng)."""
        missing = []
        if not self.user_input.strip():
            missing.append("user_input")
        if not self.reference.strip():
            missing.append("reference")
        if not any(context.strip() for context in self.reference_contexts):
            missing.append("reference_contexts")
        return missing


class UnitProgress(BaseModel):
    """Tiến độ của một đơn vị: đã sinh xong (`done`) hoặc dừng dở giữa chừng (`partial`).

    Các trường thêm sau (`status`, `skipped_samples`, `tokens`, `reasoning_tokens`) đều có
    mặc định để đọc được `generation_progress.json` của các đơn vị ghi từ trước
    (evaluation_spec.md mục 3.2 B, 3.3, 4.5).
    """

    title: str
    chars: int  # để phát hiện văn bản/quy tắc chia đã đổi so với lúc sinh
    estimated_tokens: int
    # single_hop / abstract / specific; cộng dồn khi --append và khi chạy tiếp đơn vị partial
    questions: dict[str, int]
    llm_calls: int
    seconds: float
    # `done`: lúc xong. `partial`: lúc dừng gần nhất (không phải lúc xong); đổi thành lúc xong khi
    # đơn vị chạy tiếp và chuyển `done`.
    completed_at: datetime
    # `partial`: đã có một phần câu trong raw nhưng dừng giữa chừng (hết quota ngày...), đơn vị
    # nằm trong danh sách chờ và chỉ sinh phần còn thiếu (mục 3.3).
    status: Literal["done", "partial"] = "done"
    skipped_samples: int = (
        0  # sample lỗi không phải hết quota (parse...) bị bỏ, cộng dồn
    )
    # Token thật từ `usage` của Groq (prompt + completion; reasoning là phần con của completion).
    # `None` = không có số đo (đơn vị ghi trước khi có tính năng đếm token).
    tokens: int | None = None
    reasoning_tokens: int | None = None


class UnitFailure(BaseModel):
    """Lần dừng gần nhất; không chứa nội dung câu hỏi/context."""

    unit: str  # khoá "<tên file>#<số thứ tự>"
    error: str
    at: datetime


class GenerationProgress(BaseModel):
    """Nội dung `generation_progress.json`: khoá đơn vị -> `UnitProgress`."""

    units: dict[str, UnitProgress] = Field(default_factory=dict)
    last_failure: UnitFailure | None = None
