"""Client gọi Groq API để chuyển đổi front matter/back matter sang markdown.

Dùng chung bởi ``frontmatter.py`` và ``backmatter.py`` — cả hai gọi
``convert_to_markdown`` với một khối text đã serialize sẵn (mục 1.1 spec).
Đây là text generation thuần qua chat completions (OpenAI-compatible), KHÔNG
dùng structured output: tác vụ ở đây là convert text sang markdown, không
trích field. Cấu hình (``model_name``, ``max_retries``, ``timeout_seconds``,
``groq_api_key``) đọc từ ``LLMSettings`` trong ``config.py`` (mục 4 spec).

Module này cũng sở hữu **sliding-window rate limiter** (mục 1.2, 6 spec):
Groq free tier cho ``openai/gpt-oss-120b`` giới hạn RPM 30 / TPM 8.000. Vì
xử lý tuần tự trong 1 tiến trình (mục 5 spec, không multiprocessing), state
của rate limiter là 1 singleton module-level, dùng chung xuyên suốt cả lần
chạy ``convert_directory`` (không reset theo từng file).
"""

from __future__ import annotations

import collections
import logging
import time
from functools import lru_cache

from groq import Groq

from production_legal_qa_rag.config import LLMSettings

logger = logging.getLogger(__name__)

# Cùng heuristic với `docx_reader._CHARS_PER_TOKEN` (mục 1.2 spec) -- ước
# lượng token của cả prompt trước khi gửi, để quyết định có cần chờ hay
# không. Không dùng để track budget thật (xem `_SlidingWindowRateLimiter.record`,
# đọc `usage.total_tokens` từ response sau khi gọi thành công).
_CHARS_PER_TOKEN = 2.5

# 90% giới hạn free tier (mục 1.2 spec) -- chừa margin cho sai số ước lượng
# và cho việc nhiều tiến trình/lần chạy khác có thể cùng dùng chung tài
# khoản Groq trong cùng cửa sổ 60 giây.
_RATE_LIMIT_SAFETY_FACTOR = 0.9
_RATE_LIMIT_WINDOW_SECONDS = 60.0


def _estimate_tokens(text: str) -> float:
    """Ước lượng token của một đoạn text bằng heuristic ký tự (mục 1.2 spec)."""
    return len(text) / _CHARS_PER_TOKEN


class _SlidingWindowRateLimiter:
    """Sliding-window rate limiter in-process, đơn luồng (mục 1.2, 6 spec).

    Giữ 1 deque các ``(timestamp, tokens)`` trong ``_RATE_LIMIT_WINDOW_SECONDS``
    giây gần nhất. Trước mỗi request: dọn entry đã hết hạn, chờ (``time.sleep``)
    nếu gửi ngay sẽ khiến tổng token/số request trong cửa sổ vượt ngưỡng an
    toàn. Sau khi gọi Groq thành công, ``record`` được gọi với số token THẬT
    (``usage.total_tokens``) để cập nhật cửa sổ cho các request tiếp theo.
    """

    def __init__(self, tpm_limit: int, rpm_limit: int) -> None:
        self._tpm_safe_limit = tpm_limit * _RATE_LIMIT_SAFETY_FACTOR
        self._rpm_safe_limit = rpm_limit * _RATE_LIMIT_SAFETY_FACTOR
        self._entries: collections.deque[tuple[float, int]] = collections.deque()

    def _evict_expired(self, now: float) -> None:
        while self._entries and now - self._entries[0][0] >= _RATE_LIMIT_WINDOW_SECONDS:
            self._entries.popleft()

    def wait_if_needed(self, estimated_tokens: float) -> None:
        """Chờ tới khi gửi request ước lượng ``estimated_tokens`` là an toàn.

        Không làm gì nếu cửa sổ hiện đang rỗng (không có gì để chờ hết hạn) --
        kể cả khi bản thân ``estimated_tokens`` đã vượt ngưỡng an toàn, đây là
        request đầu tiên nên không thể chờ lâu hơn được nữa.
        """
        while True:
            now = time.monotonic()
            self._evict_expired(now)
            if not self._entries:
                return

            window_tokens = sum(tokens for _, tokens in self._entries)
            window_requests = len(self._entries)
            fits_tpm = window_tokens + estimated_tokens <= self._tpm_safe_limit
            fits_rpm = window_requests + 1 <= self._rpm_safe_limit
            if fits_tpm and fits_rpm:
                return

            oldest_timestamp = self._entries[0][0]
            sleep_seconds = _RATE_LIMIT_WINDOW_SECONDS - (now - oldest_timestamp)
            if sleep_seconds > 0:
                logger.info(
                    "llm_client rate limiter: chờ %.1fs để tránh vượt TPM/RPM free tier",
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

    def record(self, timestamp: float, tokens: int) -> None:
        self._entries.append((timestamp, tokens))


@lru_cache(maxsize=1)
def _client() -> Groq:
    """Dựng Groq client, cache theo tiến trình.

    Vô hiệu hoá retry nội bộ của SDK (``max_retries=0``) vì
    ``convert_to_markdown`` tự quản lý vòng lặp retry riêng (mục 5 spec) --
    để SDK tự retry thêm sẽ nhân đôi số lần thử ngoài ý muốn.
    """
    settings = LLMSettings()
    return Groq(
        api_key=settings.groq_api_key,
        timeout=float(settings.timeout_seconds),
        max_retries=0,
    )


# Khớp giá trị default của `LLMSettings.chunk_token_limit` -- dùng khi không
# đọc được `LLMSettings` (vd. thiếu `GROQ_API_KEY`, xem `get_chunk_token_limit`).
_DEFAULT_CHUNK_TOKEN_LIMIT = 1500


def get_chunk_token_limit() -> int:
    """Đọc ``LLMSettings.chunk_token_limit``, fallback về giá trị mặc định.

    Dùng bởi ``frontmatter.py``/``backmatter.py`` để chia block thành chunk
    (mục 1.2 spec) trước khi biết có gọi được Groq hay không -- chunking là
    bước thuần cục bộ, không phụ thuộc mạng/API key, nên không nên fail cứng
    chỉ vì thiếu ``GROQ_API_KEY`` (cùng tinh thần fallback với
    ``convert_to_markdown``, vd. môi trường CI không có key thật).

    Returns:
        ``LLMSettings.chunk_token_limit`` nếu đọc được, ngược lại hằng số
        mặc định trùng giá trị default của field đó.
    """
    try:
        return LLMSettings().chunk_token_limit
    except Exception:
        logger.debug(
            "get_chunk_token_limit: không đọc được LLMSettings, dùng mặc định %d "
            "(thiếu GROQ_API_KEY?)",
            _DEFAULT_CHUNK_TOKEN_LIMIT,
            exc_info=True,
        )
        return _DEFAULT_CHUNK_TOKEN_LIMIT


@lru_cache(maxsize=1)
def _rate_limiter(tpm_limit: int, rpm_limit: int) -> _SlidingWindowRateLimiter:
    """Rate limiter singleton theo tiến trình, dùng chung xuyên suốt cả lần chạy.

    Cache theo ``(tpm_limit, rpm_limit)`` -- luôn cùng giá trị trong 1 lần
    chạy thật (đọc từ cùng ``LLMSettings``), nên ``lru_cache`` trả về đúng 1
    instance duy nhất, thoả yêu cầu "không reset theo từng file" (mục 1.2, 5
    spec).
    """
    return _SlidingWindowRateLimiter(tpm_limit=tpm_limit, rpm_limit=rpm_limit)


def convert_to_markdown(prompt: str, *, max_retries: int | None = None) -> str | None:
    """Gọi Groq để chuyển đổi một khối text sang markdown thuần.

    Không bao giờ raise ra ngoài: mọi lỗi (thiếu API key, timeout, lỗi mạng,
    response rỗng...) bị bắt, thử lại tối đa ``max_retries`` lần rồi trả
    ``None``. Caller (``frontmatter.py``, ``backmatter.py``) tự bỏ qua phần
    tương ứng và phát ``QcWarning`` — lỗi Groq không bao giờ được để thoát
    lên tới ``pipeline.convert_directory`` (mục 5, 7 spec), và không có
    fallback regex nào được thử.

    Trước mỗi lần gọi, chờ theo sliding-window rate limiter nếu cần (mục 1.2
    spec) để không vượt TPM/RPM free tier. Sau mỗi lần gọi thành công, cập
    nhật rate limiter bằng số token thật (``usage.total_tokens``).

    Args:
        prompt: Nội dung yêu cầu Groq chuyển đổi, đã kèm ngữ cảnh văn bản.
        max_retries: Số lần thử; mặc định đọc từ ``LLMSettings.max_retries``.

    Returns:
        Markdown do Groq sinh (đã strip khoảng trắng thừa hai đầu), hoặc
        ``None`` nếu hết số lần thử.
    """
    try:
        settings = LLMSettings()
    except Exception:
        logger.warning(
            "llm_client.convert_to_markdown: không đọc được LLMSettings "
            "(thiếu GROQ_API_KEY?)",
            exc_info=True,
        )
        return None

    attempts = max_retries if max_retries is not None else settings.max_retries
    limiter = _rate_limiter(settings.tpm_limit, settings.rpm_limit)

    for attempt in range(1, attempts + 1):
        limiter.wait_if_needed(_estimate_tokens(prompt))

        try:
            client = _client()
            response = client.chat.completions.create(
                model=settings.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:  # lỗi Groq không bao giờ raise ra ngoài
            logger.warning(
                "llm_client.convert_to_markdown lỗi ở lần thử %d/%d",
                attempt,
                attempts,
                exc_info=True,
            )
            continue

        total_tokens = (
            response.usage.total_tokens
            if response.usage is not None
            else int(_estimate_tokens(prompt))
        )
        limiter.record(time.monotonic(), total_tokens)

        text = response.choices[0].message.content

        if text:
            return text.strip()

        logger.warning(
            "llm_client.convert_to_markdown trả về rỗng ở lần thử %d/%d",
            attempt,
            attempts,
        )

    return None
