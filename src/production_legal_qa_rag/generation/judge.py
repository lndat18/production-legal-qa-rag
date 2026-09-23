"""Evidence Judge kiểm tra draft dựa duy nhất trên query và context cố định."""

from __future__ import annotations

from typing import Final

from groq import AsyncGroq

from production_legal_qa_rag.config import JudgeSettings
from production_legal_qa_rag.generation.generator import build_context
from production_legal_qa_rag.generation.models import Citation, JudgeVerdict
from production_legal_qa_rag.retrieval.loop_bound import LoopBoundClient
from production_legal_qa_rag.retrieval.models import RetrievedChunk

JUDGE_PROMPT_VERSION: Final = "v1"
_REASONING_EFFORT: Final = "low"
_TEMPERATURE: Final = 0.0
_MAX_COMPLETION_TOKENS: Final = 1024

_SYSTEM_PROMPT: Final = """Bạn là Evidence Judge cho hệ thống tra cứu pháp luật.
Bạn chỉ đánh giá draft dựa trên Câu hỏi và Văn bản đánh số được cung cấp. Không dùng
kiến thức ngoài, không browse, không trả lời người dùng và không trình bày suy luận.

Kiểm tra mọi claim pháp lý trọng yếu: claim phải được Văn bản hỗ trợ, citation đứng
cạnh claim phải thực sự hỗ trợ claim, và draft không được bỏ điều kiện/ngoại lệ/phạm
vi làm thay đổi kết luận. Nếu context không đủ để trả lời, dùng insufficient_evidence.

Chỉ trả JSON đúng schema sau, không markdown hay trường khác:
{"verdict":"pass|repair|insufficient_evidence","issues":[{"code":"unsupported_claim|citation_mismatch|missing_material_condition|context_insufficient","claim":"claim ngắn","detail":"mô tả ngắn có thể hành động","evidence_numbers":[1]}]}

pass chỉ khi không có issue. repair khi có thể viết lại chỉ với context này.
insufficient_evidence khi context không đủ; đây không có nghĩa luật không tồn tại."""

_USER_TEMPLATE: Final = """Văn bản:
{context}

Câu hỏi: {query}

Draft:
{draft}

Citation hợp lệ trong draft: {citations}"""


class JudgeError(RuntimeError):
    """Judge không thể tạo verdict hợp lệ để pipeline fail-closed."""


class EvidenceJudge:
    """Sở hữu client Judge và chỉ trả JudgeVerdict đã parse thành công."""

    def __init__(
        self,
        settings: JudgeSettings | None = None,
        client: AsyncGroq | None = None,
    ) -> None:
        self._settings = settings
        self._client = LoopBoundClient(self._create_client, client)

    def _create_client(self) -> AsyncGroq:
        settings = self._get_settings()
        return AsyncGroq(
            api_key=settings.api_key,
            max_retries=settings.max_retries,
            timeout=float(settings.timeout_seconds),
        )

    def _get_settings(self) -> JudgeSettings:
        """Khởi tạo settings lười để lỗi config thành JudgeError ở pipeline."""
        if self._settings is None:
            self._settings = JudgeSettings()
        return self._settings

    async def judge(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        draft: str,
        citations: list[Citation],
    ) -> JudgeVerdict:
        """Đánh giá claim của draft bằng context đã chốt.

        Args:
            query: Câu hỏi độc lập của user.
            chunks: Context rerank cố định, tối đa năm chunk.
            draft: Draft đã pass deterministic hard gate.
            citations: Citation hợp lệ đã trích từ draft.

        Returns:
            Verdict có cấu trúc của Judge.

        Raises:
            JudgeError: Provider lỗi, trả rỗng, JSON sai hoặc schema không hợp lệ.
        """
        try:
            settings = self._get_settings()
            response = await self._client.get().chat.completions.create(
                model=settings.model_name,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _USER_TEMPLATE.format(
                            context=build_context(chunks),
                            query=query,
                            draft=draft,
                            citations=_render_citations(citations),
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                reasoning_effort=_REASONING_EFFORT,
                temperature=_TEMPERATURE,
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Judge trả về nội dung rỗng")
            return JudgeVerdict.model_validate_json(content)
        except Exception as error:
            raise JudgeError("Không thể xác minh evidence của câu trả lời.") from error


def _render_citations(citations: list[Citation]) -> str:
    """Render citation map nhỏ, tránh Judge hiểu một số citation ngoài context."""
    if not citations:
        return "(không có citation)"
    return ", ".join(f"[{citation.n}] {citation.breadcrumb}" for citation in citations)
