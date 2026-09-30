"""Round-robin nhiều `ChatOpenAI` (Groq) độc lập tài khoản cho `generator_llm`.

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
import logging
import math
import random
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from time import monotonic
from typing import Any, Final

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)

# Groq ghi giới hạn theo ngày trong thông điệp 429: "... on tokens per day (TPD): Limit ...".
# Giới hạn theo phút ghi "per minute (TPM/RPM)" — tạm thời, chờ là hết, khác hẳn hết ngày.
_DAILY_LIMIT_MARKERS: Final = ("per day", "(tpd)", "(rpd)")

# TPD của Groq là cửa sổ trượt nên tài khoản đã cạn thường hồi lại sau vài phút; trong
# khoảng này ưu tiên các tài khoản khác thay vì tốn request 429 vô ích.
_DAILY_COOLDOWN_SECONDS: Final = 300.0

# 429 theo phút (TPM/RPM): chờ là hết. Nếu không có `retry-after` thì tạm nghỉ tài khoản đó 15
# giây; kẹp [1, 60] để header lạ (0, âm, hàng giờ) không làm tài khoản mất tích hay bị đập lại
# ngay (evaluation_spec.md mục 3.2 A).
_MINUTE_COOLDOWN_DEFAULT: Final = 15.0
_MINUTE_COOLDOWN_MIN: Final = 1.0
_MINUTE_COOLDOWN_MAX: Final = 60.0
# Một lỗi mạng/5xx có thể xảy ra trước khi Groq nhận request, nên cho đúng một lượt thử nữa.
# Không giao retry này cho SDK/RAGAS vì chúng không biết trạng thái từng credential.
_MAX_TRANSIENT_RETRIES: Final = 1
_TRANSIENT_RETRY_SECONDS: Final = 1.0
_TRANSIENT_RETRY_JITTER_SECONDS: Final = 0.25


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


def _retry_after_seconds(error: RateLimitError) -> float:
    """Số giây cooldown cho 429 theo phút: `retry-after` của Groq, kẹp [1, 60], mặc định 15."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    raw = headers.get("retry-after") if headers is not None else None
    try:
        seconds = float(raw)  # type: ignore[arg-type]
    except TypeError, ValueError:
        return _MINUTE_COOLDOWN_DEFAULT
    if not math.isfinite(seconds):
        return _MINUTE_COOLDOWN_DEFAULT
    return min(max(seconds, _MINUTE_COOLDOWN_MIN), _MINUTE_COOLDOWN_MAX)


def _error_detail(error: RateLimitError) -> str:
    """Thông điệp Groq gốc (bỏ tiền tố "Error code: 429 - {...}" của SDK) để nêu trong lỗi."""
    body = error.body
    if isinstance(body, Mapping) and isinstance(body.get("message"), str):
        return str(body["message"])
    return str(error)


class TokenTotals(BaseModel):
    """Token đã tiêu (chỉ tính lượt gọi thành công) của một tài khoản Groq.

    `reasoning_tokens` là phần con của `completion_tokens` (Groq/OpenAI tính token suy
    luận vào token đầu ra), nên tổng token của tài khoản là `prompt + completion`.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Tổng token vào + ra (đã gồm token suy luận)."""
        return self.prompt_tokens + self.completion_tokens


def _as_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _extract_usage(result: ChatResult) -> TokenTotals | None:
    """Đọc `llm_output["token_usage"]` do `ChatOpenAI._create_chat_result` gắn; `None` nếu thiếu."""
    usage = (result.llm_output or {}).get("token_usage")
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("completion_tokens_details")
    reasoning = details.get("reasoning_tokens") if isinstance(details, Mapping) else 0
    return TokenTotals(
        prompt_tokens=_as_count(usage.get("prompt_tokens")),
        completion_tokens=_as_count(usage.get("completion_tokens")),
        reasoning_tokens=_as_count(reasoning),
    )


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

    Điều phối key mượt và tiết kiệm token (mục 3.2): 429 theo phút đặt cooldown ngắn theo
    `retry-after`, cả `n` tài khoản cùng cooldown phút thì chờ thay vì bắn 429 hàng loạt;
    cộng dồn token thật theo tài khoản (`token_totals`); `reasoning_effort` đặt tạm bằng
    context manager và được truyền xuống `ChatOpenAI._generate`.
    """

    # Phase 1 dùng 9 client (theo `len(clients)`), mỗi client gắn 1 key cố định.
    clients: list[ChatOpenAI]
    # Đồng hồ (giây, đơn điệu) và hàm ngủ tiêm được để test không chờ thật; `None` = dùng
    # `time.monotonic`/`time.sleep` thật (tra cứu lúc gọi, nên monkeypatch module vẫn có tác dụng).
    clock: Callable[[], float] | None = Field(default=None, exclude=True)
    sleep: Callable[[float], None] | None = Field(default=None, exclude=True)

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _cycle: Iterator[int] = PrivateAttr()
    _call_counts: list[int] = PrivateAttr()
    _cooldown_until: list[float] = PrivateAttr()
    # Tách riêng phần cooldown do hết quota NGÀY: bằng chứng ngày khác bản chất với 429 theo phút.
    _daily_cooldown_until: list[float] = PrivateAttr()
    _disabled: list[bool] = PrivateAttr()
    _daily_quota_message: str | None = PrivateAttr(default=None)
    _token_totals: list[TokenTotals] = PrivateAttr()
    _missing_usage_warned: bool = PrivateAttr(default=False)
    _reasoning_effort: str | None = PrivateAttr(default=None)

    def model_post_init(self, context: Any, /) -> None:
        if not self.clients:
            raise ValueError("GroqRoundRobinChatModel cần ít nhất 1 client.")
        count = len(self.clients)
        self._cycle = itertools.cycle(range(count))
        self._call_counts = [0] * count
        self._cooldown_until = [0.0] * count
        self._daily_cooldown_until = [0.0] * count
        self._disabled = [False] * count
        self._token_totals = [TokenTotals() for _ in range(count)]

    def _now(self) -> float:
        return self.clock() if self.clock is not None else monotonic()

    def _sleep(self, seconds: float) -> None:
        (self.sleep or time.sleep)(seconds)

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

    @property
    def token_totals(self) -> list[TokenTotals]:
        """Bản sao token đã tiêu (lượt gọi thành công) theo từng client.

        Lấy từ `ChatResult.llm_output["token_usage"]` của `ChatOpenAI`; dùng để ghi
        `tokens`/`reasoning_tokens` theo đơn vị (hiệu số trước/sau) và kiểm tra token có
        rải đều qua các tài khoản (evaluation_spec.md mục 3.2 B).
        """
        with self._lock:
            return [totals.model_copy() for totals in self._token_totals]

    @property
    def current_reasoning_effort(self) -> str | None:
        """Mức `reasoning_effort` đang áp dụng (`None` = không gửi, để Groq dùng mặc định)."""
        with self._lock:
            return self._reasoning_effort

    @contextmanager
    def reasoning_effort(self, effort: str) -> Iterator[None]:
        """Đặt tạm `reasoning_effort` cho MỌI lượt gọi (mọi luồng) trong khối `with`.

        Trạng thái nằm trên router dưới lock (không phải contextvar) vì ragas gọi
        `_generate` từ các luồng executor. Gỡ về giá trị trước đó kể cả khi có lỗi.

        Args:
            effort: Mức gửi cho Groq (`"low"`, `"medium"`, `"high"`).
        """
        with self._lock:
            previous = self._reasoning_effort
            self._reasoning_effort = effort
        try:
            yield
        finally:
            with self._lock:
                self._reasoning_effort = previous

    def _plan_attempts(self) -> list[int]:
        """Thứ tự thử `n` client khác nhau cho MỘT lượt gọi; raise nếu breaker đã bật.

        Điểm bắt đầu lấy đúng một lần từ vòng round-robin, rồi duyệt
        `(start + offset) % n`. Credential đang cooldown xếp cuối nhưng vẫn nằm trong danh
        sách để mỗi lượt có bằng chứng mới trước khi bật breaker quota ngày.
        """
        with self._lock:
            if self._daily_quota_message is not None:
                raise DailyQuotaExhaustedError(self._daily_quota_message)
            count = len(self.clients)
            start = next(self._cycle)
            now = self._now()
            order = [
                (start + offset) % count
                for offset in range(count)
                if not self._disabled[(start + offset) % count]
            ]
            if not order:
                raise RuntimeError("Không còn credential Groq hoạt động.")
            order.sort(key=lambda index: self._cooldown_until[index] > now)
            return order

    def _record_attempt(self, index: int) -> None:
        with self._lock:
            self._call_counts[index] += 1

    def _mark_result(self, index: int, *, daily_limited: bool = False) -> None:
        """Ghi kết quả: quota ngày cooldown ngắn; thành công xoá cooldown."""
        with self._lock:
            if daily_limited:
                until = self._now() + _DAILY_COOLDOWN_SECONDS
                self._daily_cooldown_until[index] = until
                self._cooldown_until[index] = max(self._cooldown_until[index], until)
            else:
                self._daily_cooldown_until[index] = 0.0
                self._cooldown_until[index] = 0.0

    def _mark_minute_limited(self, index: int, error: RateLimitError) -> None:
        """429 theo phút: cooldown `retry-after` giây, không bao giờ rút ngắn cooldown đang có."""
        until = self._now() + _retry_after_seconds(error)
        with self._lock:
            self._cooldown_until[index] = max(self._cooldown_until[index], until)

    def _disable(self, index: int) -> None:
        """Không dùng lại credential bị từ chối xác thực/quyền trong process hiện tại."""
        with self._lock:
            self._disabled[index] = True

    def _minute_wait_seconds(self) -> float:
        """Số giây cần chờ khi CẢ `n` tài khoản đều đang cooldown phút; 0 trong mọi trường hợp khác.

        Không chờ khi breaker đã bật hay khi có cooldown ngày: cần probe thật để lấy bằng
        chứng mới cho breaker.
        """
        with self._lock:
            now = self._now()
            if self._daily_quota_message is not None:
                return 0.0
            if any(until > now for until in self._daily_cooldown_until):
                return 0.0
            earliest = min(self._cooldown_until)
            if earliest <= now:
                return 0.0
            return min(earliest - now, _MINUTE_COOLDOWN_MAX)

    def _record_usage(self, index: int, result: ChatResult) -> None:
        usage = _extract_usage(result)
        warn = False
        with self._lock:
            if usage is None:
                warn = not self._missing_usage_warned
                self._missing_usage_warned = True
            else:
                totals = self._token_totals[index]
                totals.prompt_tokens += usage.prompt_tokens
                totals.completion_tokens += usage.completion_tokens
                totals.reasoning_tokens += usage.reasoning_tokens
        if warn:
            logger.warning(
                "Phản hồi Groq không có llm_output['token_usage']: coi là 0 token "
                "(số đếm token có thể thấp hơn thực tế; chỉ cảnh báo lần đầu)."
            )

    def _trip_breaker(self, errors: list[RateLimitError]) -> DailyQuotaExhaustedError:
        message = f"Hết quota ngày cả {len(errors)} tài khoản Groq: {_error_detail(errors[-1])}"
        with self._lock:
            self._daily_quota_message = message
        return DailyQuotaExhaustedError(message)

    @staticmethod
    def _is_transient(error: Exception) -> bool:
        """Các lỗi an toàn để router thử đúng một credential sẵn sàng khác."""
        if isinstance(error, (APIConnectionError, APITimeoutError)):
            return True
        return isinstance(error, APIStatusError) and (
            error.status_code == 498 or 500 <= error.status_code <= 599
        )

    @staticmethod
    def _is_auth_error(error: Exception) -> bool:
        """401/403 là lỗi credential, không gửi lại cùng credential trong lượt chạy."""
        return isinstance(error, APIStatusError) and error.status_code in (401, 403)

    def _transient_retry_delay(self) -> float:
        """Backoff ngắn có jitter để các worker không retry đồng thời."""
        return _TRANSIENT_RETRY_SECONDS + random.uniform(
            0.0, _TRANSIENT_RETRY_JITTER_SECONDS
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Gọi client kế tiếp trong vòng round-robin; bounded fallback khi 429.

        429 chỉ chuyển sang credential `ready` khác và áp cooldown theo `retry-after`.
        Timeout/kết nối/5xx/498 chỉ có MỘT retry (backoff + jitter); 400/413 không retry,
        401/403 disable credential cho phần còn lại của process. SDK/RAGAS đều được cấu
        hình `max_retries=0`, nên đây là tầng retry HTTP duy nhất.

        Khi CẢ `n` tài khoản đang cooldown vì 429 theo phút (và không có cooldown ngày), chờ
        tới lúc tài khoản sớm nhất hết cooldown (tối đa 60 giây) trước khi thử, thay vì bắn
        một loạt 429 qua cả `n` tài khoản (mục 3.2 A).

        Nếu CẢ `n` client khác nhau đều báo hết quota THEO NGÀY trong cùng lượt gọi thì
        ghi nhớ và từ chối mọi lượt gọi sau đó ngay lập tức (`DailyQuotaExhaustedError`,
        mỗi lần raise là một instance mới): ragas không huỷ các task còn lại khi một
        task lỗi, nên không chặn ở đây thì mỗi task còn lại vẫn tự đốt
        `len(clients)` x (retry SDK) request vô ích và ăn RPD của các tài khoản.
        """
        wait = self._minute_wait_seconds()
        if wait > 0:
            self._sleep(wait)
        order = self._plan_attempts()
        effort = self.current_reasoning_effort
        call_kwargs = (
            kwargs if effort is None else {"reasoning_effort": effort, **kwargs}
        )
        errors: list[RateLimitError] = []
        transient_retries = 0
        last_transient_error: Exception | None = None
        for index in order:
            self._record_attempt(index)
            try:
                result = self.clients[index]._generate(
                    messages, stop=stop, run_manager=run_manager, **call_kwargs
                )
            except RateLimitError as error:
                errors.append(error)
                if _is_daily_limit(error):
                    self._mark_result(index, daily_limited=True)
                else:
                    self._mark_minute_limited(index, error)
            except Exception as error:
                last_transient_error = error
                if self._is_auth_error(error):
                    self._disable(index)
                if (
                    not self._is_transient(error)
                    or transient_retries >= _MAX_TRANSIENT_RETRIES
                ):
                    raise
                transient_retries += 1
                self._sleep(self._transient_retry_delay())
            else:
                self._mark_result(index, daily_limited=False)
                self._record_usage(index, result)
                return result

        # `errors` chỉ chứa 429; credential disabled không phải bằng chứng quota ngày.
        if (
            errors
            and len(errors) == len(order)
            and all(_is_daily_limit(error) for error in errors)
        ):
            raise self._trip_breaker(errors) from errors[-1]
        if errors:
            raise errors[-1]
        if last_transient_error is not None:
            raise last_transient_error
        raise RuntimeError("Router Groq không gửi được request.")
