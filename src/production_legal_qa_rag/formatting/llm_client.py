"""Client gọi Groq API để chuyển đổi front matter/back matter sang markdown.

Dùng bởi ``pipeline.py`` — ``frontmatter.py``/``backmatter.py`` chỉ dựng
prompt (``build_prompts``, thuần, không I/O), toàn bộ việc gọi Groq nay tập
trung ở đây qua ``convert_chunks_concurrently`` (mục 1.3 spec). Đây là text
generation thuần qua chat completions (OpenAI-compatible), KHÔNG dùng
structured output: tác vụ ở đây là convert text sang markdown, không trích
field. Cấu hình (``model_name``, ``max_retries``, ``timeout_seconds``,
``groq_api_key``, ``groq_api_key_2``) đọc từ ``LLMSettings`` trong
``config.py`` (mục 4 spec).

Module này cũng sở hữu **sliding-window rate limiter** (mục 1.2, 6 spec):
Groq free tier cho ``openai/gpt-oss-120b`` giới hạn RPM 30 / TPM 8.000. Khi
chỉ có 1 key, xử lý tuần tự trong 1 tiến trình (mục 5 spec, không
multiprocessing) dùng 1 rate limiter singleton module-level, dùng chung
xuyên suốt cả lần chạy ``convert_directory`` (không reset theo từng file).

**[Mới 2026-09-17]** Khi có ``GROQ_API_KEY_2``, ``convert_chunks_concurrently``
dispatch job (trong phạm vi 1 file, mục 1.3 spec) qua 1 hàng đợi dùng chung
cho đúng 2 thread worker cố định — mỗi worker gắn chết 1 (client,
rate-limiter) độc lập, không chia sẻ state rate-limiter giữa 2 thread nên
không cần khoá cho phần đó; chỉ khoá thao tác ghi ``results[index]``.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
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
    """Sliding-window rate limiter in-process (mục 1.2, 6 spec).

    Giữ 1 deque các ``(timestamp, tokens)`` trong ``_RATE_LIMIT_WINDOW_SECONDS``
    giây gần nhất. Trước mỗi request: dọn entry đã hết hạn, chờ (``time.sleep``)
    nếu gửi ngay sẽ khiến tổng token/số request trong cửa sổ vượt ngưỡng an
    toàn. Sau khi gọi Groq thành công, ``record`` được gọi với số token THẬT
    (``usage.total_tokens``) để cập nhật cửa sổ cho các request tiếp theo.

    **[Mới 2026-09-17]** Mỗi thread worker trong ``convert_chunks_concurrently``
    (mục 1.3 spec) giữ 1 instance riêng, không chia sẻ state với nhau -- vì
    vậy không cần khoá cho các thao tác đọc/ghi ``_entries`` bên trong 1
    instance (chỉ 1 thread đụng tới nó).
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
    """Dựng Groq client cho key 1 (``GROQ_API_KEY``), cache theo tiến trình.

    Vô hiệu hoá retry nội bộ của SDK (``max_retries=0``) vì ``_convert_one``
    tự quản lý vòng lặp retry riêng (mục 5 spec) -- để SDK tự retry thêm sẽ
    nhân đôi số lần thử ngoài ý muốn.
    """
    settings = LLMSettings()
    return Groq(
        api_key=settings.groq_api_key,
        timeout=float(settings.timeout_seconds),
        max_retries=0,
    )


@lru_cache(maxsize=1)
def _client_2() -> Groq:
    """Dựng Groq client cho key 2 (``GROQ_API_KEY_2``), cache theo tiến trình.

    Chỉ được gọi khi ``LLMSettings.groq_api_key_2`` đã set (kiểm tra ở
    ``convert_chunks_concurrently`` trước khi spawn worker) -- xem mục 1.3
    spec.
    """
    settings = LLMSettings()
    return Groq(
        api_key=settings.groq_api_key_2,
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
    ``convert_chunks_concurrently``, vd. môi trường CI không có key thật).

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
    """Rate limiter singleton cho key 1, dùng chung xuyên suốt cả lần chạy.

    Cache theo ``(tpm_limit, rpm_limit)`` -- luôn cùng giá trị trong 1 lần
    chạy thật (đọc từ cùng ``LLMSettings``), nên ``lru_cache`` trả về đúng 1
    instance duy nhất, thoả yêu cầu "không reset theo từng file" (mục 1.2, 5
    spec).
    """
    return _SlidingWindowRateLimiter(tpm_limit=tpm_limit, rpm_limit=rpm_limit)


@lru_cache(maxsize=1)
def _rate_limiter_2(tpm_limit: int, rpm_limit: int) -> _SlidingWindowRateLimiter:
    """Rate limiter singleton cho key 2 -- instance RIÊNG, không share với key 1.

    Groq tính RPM/TPM theo cửa sổ trượt 60 giây riêng theo từng API key (mục
    1.3 spec) -- 2 key hoàn toàn độc lập, nên state rate-limiter cũng phải
    độc lập, không dùng chung deque với ``_rate_limiter``.
    """
    return _SlidingWindowRateLimiter(tpm_limit=tpm_limit, rpm_limit=rpm_limit)


def _convert_one(
    client: Groq,
    prompt: str,
    rate_limiter: _SlidingWindowRateLimiter,
    *,
    max_retries: int,
) -> str | None:
    """Gọi Groq để chuyển đổi một khối text sang markdown thuần.

    Không bao giờ raise ra ngoài: mọi lỗi (thiếu API key, timeout, lỗi mạng,
    response rỗng...) bị bắt, thử lại tối đa ``max_retries`` lần rồi trả
    ``None``. Caller (``convert_chunks_concurrently``) không thử fallback
    regex nào -- lỗi Groq không bao giờ được để thoát lên tới
    ``pipeline.convert_directory`` (mục 5, 7 spec).

    Trước mỗi lần gọi, chờ theo ``rate_limiter`` nếu cần (mục 1.2 spec) để
    không vượt TPM/RPM free tier. Sau mỗi lần gọi thành công, cập nhật
    ``rate_limiter`` bằng số token thật (``usage.total_tokens``).

    Args:
        client: Groq client đã cấu hình sẵn API key + timeout.
        prompt: Nội dung yêu cầu Groq chuyển đổi, đã kèm ngữ cảnh văn bản.
        rate_limiter: Rate limiter gắn với chính ``client`` này -- mỗi thread
            worker (mục 1.3 spec) truyền vào instance riêng của nó.
        max_retries: Số lần thử tối đa.

    Returns:
        Markdown do Groq sinh (đã strip khoảng trắng thừa hai đầu), hoặc
        ``None`` nếu hết số lần thử.
    """
    try:
        model_name = LLMSettings().model_name
    except Exception:
        logger.warning(
            "llm_client._convert_one: không đọc được LLMSettings (thiếu GROQ_API_KEY?)",
            exc_info=True,
        )
        return None

    for attempt in range(1, max_retries + 1):
        rate_limiter.wait_if_needed(_estimate_tokens(prompt))

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:  # lỗi Groq không bao giờ raise ra ngoài
            logger.warning(
                "llm_client._convert_one lỗi ở lần thử %d/%d",
                attempt,
                max_retries,
                exc_info=True,
            )
            continue

        total_tokens = (
            response.usage.total_tokens
            if response.usage is not None
            else int(_estimate_tokens(prompt))
        )
        rate_limiter.record(time.monotonic(), total_tokens)

        text = response.choices[0].message.content

        if text:
            return text.strip()

        logger.warning(
            "llm_client._convert_one trả về rỗng ở lần thử %d/%d",
            attempt,
            max_retries,
        )

    return None


def _worker_loop(
    job_queue: queue.Queue[tuple[int, str]],
    client: Groq,
    rate_limiter: _SlidingWindowRateLimiter,
    max_retries: int,
    results: list[str | None],
    results_lock: threading.Lock,
) -> None:
    """Vòng lặp của 1 thread worker (mục 1.3 spec).

    Lấy job kế tiếp từ ``job_queue`` (không block -- toàn bộ job đã được đẩy
    vào hàng đợi trước khi thread khởi động) tới khi hết, gọi ``_convert_one``
    bằng ``client``/``rate_limiter`` của chính thread này, ghi kết quả vào
    ``results[index]`` dưới ``results_lock``.
    """
    while True:
        try:
            index, prompt = job_queue.get_nowait()
        except queue.Empty:
            return

        try:
            result = _convert_one(client, prompt, rate_limiter, max_retries=max_retries)
        finally:
            job_queue.task_done()

        with results_lock:
            results[index] = result


def convert_chunks_concurrently(prompts: list[str]) -> list[str | None]:
    """Chuyển 1 danh sách prompt sang markdown, dùng 2 key Groq nếu có (mục 1.3).

    Gộp toàn bộ job front matter (``before``/``after``) lẫn back matter của
    **1 file đang xử lý** thành 1 lệnh gọi duy nhất — ``pipeline.py`` nối
    ``before_prompts + after_prompts + chunk_prompts`` trước khi gọi hàm này.

    - Có ``LLMSettings.groq_api_key_2``: spawn đúng 2 ``threading.Thread``,
      mỗi thread gắn chết 1 (client, rate-limiter) độc lập — worker nào rảnh
      trước lấy job tiếp theo trong hàng đợi dùng chung (dispatch động, không
      phải round-robin cố định). Join xong, trả ``results`` theo đúng thứ tự
      gốc (đánh số theo chỉ số, không theo thứ tự hoàn thành).
    - Không có ``groq_api_key_2``: không spawn thread nào — lặp tuần tự gọi
      ``_convert_one`` bằng client key 1 cho từng prompt (y hệt hành vi trước
      khi có tính năng dispatch đồng thời).
    - Không đọc được ``LLMSettings`` (vd. thiếu ``GROQ_API_KEY``): trả về
      ``None`` cho mọi prompt, không raise.

    Args:
        prompts: Danh sách prompt theo đúng thứ tự cần giữ ở kết quả trả về.

    Returns:
        ``list[str | None]`` cùng độ dài với ``prompts``, ``None`` ở vị trí
        job lỗi (hết ``max_retries``/timeout, y hệt ngữ nghĩa ``_convert_one``).
    """
    if not prompts:
        return []

    try:
        settings = LLMSettings()
    except Exception:
        logger.warning(
            "llm_client.convert_chunks_concurrently: không đọc được LLMSettings "
            "(thiếu GROQ_API_KEY?)",
            exc_info=True,
        )
        return [None] * len(prompts)

    if not settings.groq_api_key_2:
        limiter = _rate_limiter(settings.tpm_limit, settings.rpm_limit)
        client = _client()
        return [
            _convert_one(client, prompt, limiter, max_retries=settings.max_retries)
            for prompt in prompts
        ]

    job_queue: queue.Queue[tuple[int, str]] = queue.Queue()
    for index, prompt in enumerate(prompts):
        job_queue.put((index, prompt))

    results: list[str | None] = [None] * len(prompts)
    results_lock = threading.Lock()

    worker_bindings = (
        (_client(), _rate_limiter(settings.tpm_limit, settings.rpm_limit)),
        (_client_2(), _rate_limiter_2(settings.tpm_limit, settings.rpm_limit)),
    )

    threads = [
        threading.Thread(
            target=_worker_loop,
            args=(
                job_queue,
                worker_client,
                worker_rate_limiter,
                settings.max_retries,
                results,
                results_lock,
            ),
        )
        for worker_client, worker_rate_limiter in worker_bindings
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return results
