"""Model dữ liệu cho golden testset sinh tự động bởi ragas (evaluation_spec.md mục 5).

Chỉ phục vụ Phase 1 (sinh testset); không lẫn với contract của `retrieval/`
(`RetrievedChunk`) hay bất kỳ package nào khác trong pipeline chính.
"""

from __future__ import annotations

from pydantic import BaseModel


class GoldenTestCase(BaseModel):
    """Một câu hỏi mẫu trong golden testset, sinh tự động bởi ragas.

    Giữ đúng 3 cột bắt buộc theo schema ``EvaluationDataset`` của ragas
    (``user_input``, ``reference``, ``reference_contexts``) để Phase 2 dùng
    thẳng không cần convert; ``synthesizer_name`` là cột phụ giữ lại để biết
    loại câu hỏi (single-hop/multi-hop) lúc review bằng mắt.
    """

    user_input: str
    reference: str
    reference_contexts: list[str]
    synthesizer_name: str | None = None
