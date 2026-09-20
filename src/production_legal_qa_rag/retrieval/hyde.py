"""Sinh hypothetical document (HyDE) bằng Groq (mục 4)."""

from __future__ import annotations

import logging

from groq import AsyncGroq

from production_legal_qa_rag.config import LLMSettings
from production_legal_qa_rag.retrieval.loop_bound import LoopBoundClient

logger = logging.getLogger(__name__)

# Cấm nêu số Điều/Khoản/tên văn bản: sparse của nhánh A search trên breadcrumb
# (chứa số Điều/Khoản), viện dẫn bịa sai sẽ kéo chunk khớp số sai vào nhánh A.
HYDE_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam. Hãy viết một đoạn văn ngắn \
(3-5 câu) trả lời trực tiếp câu hỏi dưới đây, theo đúng văn phong của văn bản quy \
phạm pháp luật Việt Nam (dùng các thuật ngữ pháp lý như "người lao động", "người sử \
dụng lao động", "được hưởng", "có trách nhiệm"...).

Quy tắc bắt buộc:
- Chỉ viết nội dung quy định, KHÔNG nêu số Điều, số Khoản, số Điểm hay tên văn bản \
cụ thể (kể cả khi câu hỏi có nhắc tới).
- Không giải thích thêm, không mở đầu hay kết luận, chỉ trả về đoạn văn.

Câu hỏi: {query}"""

_MAX_COMPLETION_TOKENS = 2048


class HydeGenerator:
    """Sinh 1 đoạn văn giả định theo văn phong luật cho một câu hỏi."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        client: AsyncGroq | None = None,
    ) -> None:
        self._settings = settings or LLMSettings()
        self._client = LoopBoundClient(self._create_client, client)

    def _create_client(self) -> AsyncGroq:
        return AsyncGroq(
            api_key=self._settings.groq_api_key,
            max_retries=self._settings.max_retries,
            timeout=float(self._settings.timeout_seconds),
        )

    async def generate(self, query: str) -> str | None:
        """Sinh hypothetical document.

        Args:
            query: Câu hỏi gốc của người dùng.

        Returns:
            Đoạn văn đã strip, hoặc `None` khi Groq lỗi/timeout hoặc trả rỗng
            (vd. model reasoning dùng hết token cho reasoning) — pipeline sẽ
            bỏ nhánh A (mục 10).
        """
        try:
            response = await self._client.get().chat.completions.create(
                model=self._settings.model_name,
                messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
            )
        except Exception:
            logger.warning("Groq HyDE lỗi, bỏ nhánh A.", exc_info=True)
            return None

        content = (response.choices[0].message.content or "").strip()
        if not content:
            logger.warning("Groq HyDE trả về rỗng, bỏ nhánh A.")
            return None
        return content
