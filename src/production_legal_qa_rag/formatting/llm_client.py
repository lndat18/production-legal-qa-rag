"""Client gọi Groq API để trích xuất structured output bằng LLM.

Dùng chung bởi `frontmatter.py` (schema `FrontMatterExtraction`) và
`footnotes.py` (schema `FootnoteExtraction`) — hai module đó gọi
`extract_structured` làm đường chính đọc **nội dung**, tự fallback về giá
trị regex baseline khi hàm này trả `None` (formatting_spec.md mục 1, 5, 6).
Cấu hình (`model_name`, `max_retries`, `timeout_seconds`, `groq_api_key`) đọc
từ `LLMSettings` trong `config.py`, dùng chung với `chunking/`.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import instructor
from groq import Groq
from pydantic import BaseModel

from production_legal_qa_rag.config import LLMSettings

logger = logging.getLogger(__name__)

# Groq mặc định giới hạn completion khá thấp cho model này (~1024 token) —
# chú thích dài (trích nguyên cả Điều của luật khác) từng bị cắt giữa JSON,
# báo "max completion tokens reached before generating a valid document"
# dù input hoàn toàn hợp lệ. Nới rộng để structured output đủ chỗ sinh ra
# toàn bộ danh sách entries, đã xác nhận bằng gọi thật.
_MAX_COMPLETION_TOKENS = 4096


@lru_cache(maxsize=1)
def _client() -> instructor.Instructor:
    """Dựng Groq client bọc `instructor`, cache theo tiến trình.

    Dùng `Mode.JSON` — `Mode.TOOLS` (mặc định của `instructor.from_groq`) bị
    model `openai/gpt-oss-120b` trên Groq trả lỗi "Tool choice is required,
    but model did not call a tool" dù nội dung sinh ra hợp lệ, đã xác nhận
    bằng gọi thật.
    """
    settings = LLMSettings()
    return instructor.from_groq(
        Groq(api_key=settings.groq_api_key), mode=instructor.Mode.JSON
    )


def extract_structured(
    prompt: str,
    schema: type[BaseModel],
    *,
    max_retries: int | None = None,
) -> BaseModel | None:
    """Gọi Groq API để ép LLM trả structured output theo `schema`.

    Không bao giờ raise ra ngoài: mọi lỗi (thiếu API key, timeout, hết số
    lần thử, lỗi mạng, response không hợp lệ theo `schema`...) bị bắt và trả
    `None`. Caller (`frontmatter.py`, `footnotes.py`) tự fallback về giá trị
    regex baseline và phát `QcWarning` tương ứng — lỗi LLM không bao giờ
    được để thoát lên tới `pipeline.convert_directory` (mục 5, 7 spec).

    Args:
        prompt: Nội dung yêu cầu LLM trích xuất, đã kèm ngữ cảnh văn bản.
        schema: Pydantic model mô tả structured output kỳ vọng.
        max_retries: Số lần thử lại khi thất bại; mặc định đọc từ
            `LLMSettings.max_retries`.

    Returns:
        Instance của `schema` nếu gọi thành công, `None` nếu lỗi, timeout
        hoặc hết số lần thử.
    """
    try:
        settings = LLMSettings()
        client = _client()
        response = client.chat.completions.create(
            model=settings.model_name,
            max_retries=max_retries
            if max_retries is not None
            else settings.max_retries,
            timeout=settings.timeout_seconds,
            max_completion_tokens=_MAX_COMPLETION_TOKENS,
            response_model=schema,
            messages=[{"role": "user", "content": prompt}],
        )
    except (
        Exception
    ):  # lỗi LLM không bao giờ raise ra ngoài (BLE001 không phải rule đang bật)
        logger.warning(
            "llm_client.extract_structured thất bại, caller sẽ fallback baseline",
            exc_info=True,
        )
        return None
    return response
