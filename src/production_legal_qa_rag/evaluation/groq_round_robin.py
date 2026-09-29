"""Round-robin 6 `ChatOpenAI` (Groq) độc lập tài khoản cho `generator_llm`.

Pattern MỚI trong repo (evaluation_spec.md mục 3.1), khác hẳn cách dùng nhiều
key Groq hiện có ở `GenerationSettings`/`JudgeSettings`/`formatting/llm_client.py`
(tách ngân sách theo bước cố định, mỗi bước luôn dùng đúng 1 key). Ở đây là
luân phiên nhiều key cho CÙNG một luồng gọi để rải tải — không tái dùng được
`_SlidingWindowRateLimiter`/`convert_chunks_concurrently` của
`formatting/llm_client.py` (thiết kế cho 2 worker thread xử lý song song một
hàng đợi job, không phải round-robin tuần tự).
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Iterator, Mapping
from time import monotonic
from typing import Any, Final

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from openai import RateLimitError
from pydantic import PrivateAttr

# Groq ghi giới hạn theo ngày trong thông điệp 429: "... on tokens per day (TPD): Limit ...".
# Giới hạn theo phút ghi "per minute (TPM/RPM)" — tạm thời, chờ là hết, khác hẳn hết ngày.
_DAILY_LIMIT_MARKERS: Final = ("per day", "(tpd)", "(rpd)")

# TPD của Groq là cửa sổ trượt nên tài khoản đã cạn thường hồi lại sau vài phút; trong
# khoảng này ưu tiên các tài khoản khác thay vì tốn `1 + max_retries` request 429 mỗi lượt.
_DAILY_COOLDOWN_SECONDS: Final = 300.0


class DailyQuotaExhaustedError(RuntimeError):
    """Mọi tài khoản đều báo hết quota THEO NGÀY; chờ/thử lại trong ngày là vô ích.

    Cố ý KHÔNG kế thừa `openai.RateLimitError` để retry của ragas không thử lại, và mọi
    lượt gọi sau đó trong cùng process được từ chối ngay (không gửi request nào nữa).
    `status_code` giữ để `last_failure` ghi được thông điệp lỗi HTTP như 429 thường.
    """

    status_code = 429


def _is_daily_limit(error: RateLimitError) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in _DAILY_LIMIT_MARKERS)


def _error_detail(error: RateLimitError) -> str:
    """Thông điệp Groq gốc (bỏ tiền tố "Error code: 429 - {...}" của SDK) để nêu trong lỗi."""
    body = error.body
    if isinstance(body, Mapping) and isinstance(body.get("message"), str):
        return str(body["message"])
    return str(error)


class GroqRoundRobinChatModel(BaseChatModel):
    """Proxy luân phiên round-robin qua N `ChatOpenAI` (Groq) độc lập tài khoản.

    Không phải rate-limiter: chỉ chọn client trước mỗi lượt gọi thật, để rải tải đều
    qua các tài khoản độc lập. Chỉ cần implement `_generate` (sync) —
    `BaseChatModel._agenerate` mặc định gọi `_generate` qua executor khi không
    override, nên round-robin vẫn đúng dù ragas gọi qua đường async.

    Không thêm rate-limiter mới: mỗi `ChatOpenAI` con giữ nguyên timeout/retry riêng từ
    `TestsetGeneratorSettings`; round-robin ở đây chỉ chọn client, không kiểm soát tốc độ.

    Thread-safe (ragas chạy nhiều worker và `_agenerate` chạy `_generate` trong
    executor): toàn bộ trạng thái dùng chung — con trỏ vòng, bộ đếm `call_counts`,
    cooldown từng tài khoản và cờ circuit breaker — chỉ được đọc/ghi dưới một
    `threading.Lock`, và lock KHÔNG giữ trong lúc gọi mạng. Mỗi lượt gọi chốt điểm bắt đầu
    đúng MỘT lần rồi tự duyệt `n` client KHÁC NHAU; các lượt gọi song song không xen kẽ
    vào con trỏ của nhau, nên điều kiện bật breaker ("cả n tài khoản đều hết quota ngày")
    không thể đúng oan (evaluation_spec.md mục 3.1).
    """

    # Phase 1 dùng đúng 6 client, mỗi client gắn 1 key cố định.
    clients: list[ChatOpenAI]

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _cycle: Iterator[int] = PrivateAttr()
    _call_counts: list[int] = PrivateAttr()
    _cooldown_until: list[float] = PrivateAttr()
    _daily_quota_message: str | None = PrivateAttr(default=None)

    def model_post_init(self, context: Any, /) -> None:
        if not self.clients:
            raise ValueError("GroqRoundRobinChatModel cần ít nhất 1 client.")
        self._cycle = itertools.cycle(range(len(self.clients)))
        self._call_counts = [0] * len(self.clients)
        self._cooldown_until = [0.0] * len(self.clients)

    @property
    def _llm_type(self) -> str:
        return "groq-round-robin"

    @property
    def call_counts(self) -> list[int]:
        """Số lượt gọi (kể cả lượt bị 429 rồi chuyển client) đã gửi tới từng client.

        Dùng để ghi `llm_calls` theo đơn vị vào `generation_progress.json` và để
        kiểm tra tải có rải đều qua các tài khoản (evaluation_spec.md mục 4.5, 9.3).
        """
        with self._lock:
            return list(self._call_counts)

    def _plan_attempts(self) -> list[int]:
        """Thứ tự thử `n` client khác nhau cho MỘT lượt gọi; raise nếu breaker đã bật.

        Điểm bắt đầu lấy đúng một lần từ vòng round-robin, rồi duyệt
        `(start + offset) % n`. Client đang cooldown (vừa báo hết quota ngày) xếp
        CUỐI (sắp xếp ổn định) nhưng vẫn nằm trong danh sách: cần bằng chứng mới của
        cả `n` client mới bật được breaker, và client đã hồi lại vẫn dùng được.
        """
        with self._lock:
            if self._daily_quota_message is not None:
                raise DailyQuotaExhaustedError(self._daily_quota_message)
            count = len(self.clients)
            start = next(self._cycle)
            now = monotonic()
            order = [(start + offset) % count for offset in range(count)]
            order.sort(key=lambda index: self._cooldown_until[index] > now)
            return order

    def _record_attempt(self, index: int) -> None:
        with self._lock:
            self._call_counts[index] += 1

    def _mark_result(self, index: int, *, daily_limited: bool) -> None:
        with self._lock:
            self._cooldown_until[index] = (
                monotonic() + _DAILY_COOLDOWN_SECONDS if daily_limited else 0.0
            )

    def _trip_breaker(self, errors: list[RateLimitError]) -> DailyQuotaExhaustedError:
        message = f"Hết quota ngày cả {len(errors)} tài khoản Groq: {_error_detail(errors[-1])}"
        with self._lock:
            self._daily_quota_message = message
        return DailyQuotaExhaustedError(message)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Gọi client kế tiếp trong vòng round-robin; bounded fallback khi 429.

        Thử tối đa `len(clients)` client KHÁC NHAU cho MỘT lượt gọi (xem
        `_plan_attempts`); hết vòng vẫn lỗi thì raise nguyên lỗi cuối, không giữ vòng
        lặp vô hạn, không tự ý bỏ qua lượt gọi. Lỗi khác 429 (400/401/403/413...) là
        lỗi tất định, không thử sang tài khoản khác.

        Nếu CẢ `n` client khác nhau đều báo hết quota THEO NGÀY trong cùng lượt gọi thì
        ghi nhớ và từ chối mọi lượt gọi sau đó ngay lập tức (`DailyQuotaExhaustedError`,
        mỗi lần raise là một instance mới): ragas không huỷ các task còn lại khi một
        task lỗi, nên không chặn ở đây thì mỗi task còn lại vẫn tự đốt
        `len(clients)` x (retry SDK) request vô ích và ăn RPD của các tài khoản.
        """
        order = self._plan_attempts()
        errors: list[RateLimitError] = []
        for index in order:
            self._record_attempt(index)
            try:
                result = self.clients[index]._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            except RateLimitError as error:
                errors.append(error)
                self._mark_result(index, daily_limited=_is_daily_limit(error))
            else:
                self._mark_result(index, daily_limited=False)
                return result

        # `model_post_init` đã chặn `clients` rỗng nên `errors` có đủ `n` phần tử khác nhau.
        if all(_is_daily_limit(error) for error in errors):
            raise self._trip_breaker(errors) from errors[-1]
        raise errors[-1]
