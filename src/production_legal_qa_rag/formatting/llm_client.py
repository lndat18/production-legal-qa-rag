"""Client gọi Gemini API để chuyển đổi front matter/back matter sang markdown.

Dùng chung bởi ``frontmatter.py`` và ``backmatter.py`` — cả hai gọi
``convert_to_markdown`` với một khối text đã serialize sẵn (mục 1.1 spec).
Đây là text generation thuần, KHÔNG dùng ``response_schema``/structured
output: tác vụ ở đây là convert text sang markdown, không trích field. Cấu
hình (``model_name``, ``max_retries``, ``timeout_seconds``,
``gemini_api_key``) đọc từ ``LLMSettings`` trong ``config.py`` (mục 4 spec).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from google import genai
from google.genai import types

from production_legal_qa_rag.config import LLMSettings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    """Dựng Gemini client, cache theo tiến trình.

    Timeout mỗi request đọc từ ``LLMSettings.timeout_seconds`` (giây), SDK
    ``google-genai`` nhận giá trị theo mili-giây qua ``HttpOptions.timeout``.
    """
    settings = LLMSettings()
    return genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=settings.timeout_seconds * 1000),
    )


def convert_to_markdown(prompt: str, *, max_retries: int | None = None) -> str | None:
    """Gọi Gemini để chuyển đổi một khối text sang markdown thuần.

    Không bao giờ raise ra ngoài: mọi lỗi (thiếu API key, timeout, lỗi mạng,
    response rỗng...) bị bắt, thử lại tối đa ``max_retries`` lần rồi trả
    ``None``. Caller (``frontmatter.py``, ``backmatter.py``) tự bỏ qua phần
    tương ứng và phát ``QcWarning`` — lỗi Gemini không bao giờ được để thoát
    lên tới ``pipeline.convert_directory`` (mục 5, 7 spec), và không có
    fallback regex nào được thử.

    Args:
        prompt: Nội dung yêu cầu Gemini chuyển đổi, đã kèm ngữ cảnh văn bản.
        max_retries: Số lần thử; mặc định đọc từ ``LLMSettings.max_retries``.

    Returns:
        Markdown do Gemini sinh (đã strip khoảng trắng thừa hai đầu), hoặc
        ``None`` nếu hết số lần thử.
    """
    try:
        settings = LLMSettings()
    except Exception:
        logger.warning(
            "llm_client.convert_to_markdown: không đọc được LLMSettings "
            "(thiếu GEMINI_API_KEY?)",
            exc_info=True,
        )
        return None

    attempts = max_retries if max_retries is not None else settings.max_retries

    for attempt in range(1, attempts + 1):
        try:
            client = _client()
            response = client.models.generate_content(
                model=settings.model_name, contents=prompt
            )
            text = response.text
        except Exception:  # lỗi Gemini không bao giờ raise ra ngoài
            logger.warning(
                "llm_client.convert_to_markdown lỗi ở lần thử %d/%d",
                attempt,
                attempts,
                exc_info=True,
            )
            continue

        if text:
            return text.strip()

        logger.warning(
            "llm_client.convert_to_markdown trả về rỗng ở lần thử %d/%d",
            attempt,
            attempts,
        )

    return None
