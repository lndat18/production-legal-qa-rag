"""Sinh hypothetical document (HyDE) bằng Groq (mục 4)."""

from __future__ import annotations

import logging
from typing import Final

from groq import AsyncGroq

from production_legal_qa_rag.config import LLMSettings
from production_legal_qa_rag.retrieval.loop_bound import LoopBoundClient

logger = logging.getLogger(__name__)

# Prompt nguyên văn chốt 2026-09-20 (retrieval_spec.md mục 4). Quy tắc 3 cấm
# nêu số Điều/Khoản/tên văn bản: sparse của nhánh A search trên breadcrumb
# (chứa số Điều/Khoản), viện dẫn bịa sai sẽ kéo chunk khớp số sai vào nhánh A.
# Chỉ HYDE_USER_TEMPLATE được `.format`; system prompt dùng nguyên văn nên
# `{`/`}` không gây lỗi.
HYDE_SYSTEM_PROMPT = """Bạn là chuyên gia pháp luật Việt Nam. Nhiệm vụ: với mỗi câu hỏi của người dùng,
viết một đoạn văn giả định như thể trích từ văn bản quy phạm pháp luật Việt Nam
đang trả lời câu hỏi đó. Đoạn văn này chỉ dùng để tìm kiếm điều luật tương tự,
không phải câu trả lời cho người dùng.

Phạm vi pháp luật thường gặp: lao động và quan hệ lao động, bảo hiểm xã hội,
bảo hiểm y tế, thuế thu nhập cá nhân, tiền lương và mức lương tối thiểu.

Quy tắc:
1. Viết bằng văn phong văn bản quy phạm pháp luật (câu khẳng định, mang tính quy
   định: "được", "có quyền", "có trách nhiệm", "phải", "không được"...), dùng
   thuật ngữ pháp lý đúng lĩnh vực của câu hỏi (ví dụ: người lao động, người sử
   dụng lao động, người tham gia bảo hiểm, đóng và hưởng bảo hiểm, người nộp
   thuế, thu nhập chịu thuế, mức lương tối thiểu). Chỉ dùng thuật ngữ hợp với
   chủ đề câu hỏi, không nhồi thuật ngữ của lĩnh vực khác.
2. Độ dài: 3 đến 4 câu, khoảng 60-100 từ.
3. TUYỆT ĐỐI không nêu số Điều, Khoản, Điểm, Chương, Mục, tên hay số hiệu văn
   bản, năm ban hành — kể cả khi câu hỏi có nhắc tới.
4. Không nêu con số, mức tiền, tỉ lệ, thời hạn cụ thể, trừ khi bạn chắc chắn
   đúng; nếu không chắc, diễn đạt bằng từ chung ("mức tối thiểu theo quy định",
   "trong thời hạn theo quy định").
5. Câu hỏi mơ hồ, quá ngắn hoặc ngoài các lĩnh vực trên: vẫn viết một đoạn theo
   văn phong quy định về chủ đề pháp lý gần nhất mà câu hỏi gợi ra. Không từ
   chối, không hỏi lại.
6. Câu hỏi chỉ hỏi theo số Điều/Khoản mà không nêu chủ đề (ví dụ "Điều 36 khoản
   2 quy định gì?"): không đoán hay bịa chủ đề; viết một đoạn ngắn, chung chung
   về việc quy định các quyền, nghĩa vụ và trách nhiệm của các bên liên quan.
7. Đầu ra: chỉ một đoạn văn thuần tiếng Việt. Không markdown, không gạch đầu
   dòng, không tiêu đề, không lời mở đầu hay kết luận, không giải thích thêm."""

HYDE_USER_TEMPLATE = "Câu hỏi: {query}"

# Tham số Groq (mục 4): hằng số nội bộ, không vào LLMSettings.
# `reasoning_effort` là tham số đặc thù của họ gpt-oss: nếu đổi sang model không
# hỗ trợ, lời gọi sẽ lỗi và HyDE degrade im lặng (bỏ nhánh A, chỉ có warning).
_REASONING_EFFORT: Final = "low"
_TEMPERATURE = 0.2
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
                messages=[
                    {"role": "system", "content": HYDE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": HYDE_USER_TEMPLATE.format(query=query),
                    },
                ],
                reasoning_effort=_REASONING_EFFORT,
                temperature=_TEMPERATURE,
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
