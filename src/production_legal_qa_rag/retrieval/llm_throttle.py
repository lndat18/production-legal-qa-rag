"""Throttle cửa sổ trượt 60 giây (TPM + RPM) dùng chung cho các lời gọi Groq nhẹ.

Groq tính rate limit theo (tài khoản, model). Condense, HyDE và Evidence Judge ở
3 package khác nhau cùng gọi ``gpt-oss-20b``, nên cần một bộ đếm chung theo bucket
``(model, key)`` để giãn thời gian gọi thay vì đập vào 429 (conversation_spec.md
mục 12.1). Module nằm ở ``retrieval/`` vì đây là tầng thấp nhất trong 3 package đó,
cùng lý do với ``loop_bound.py``.

Throttle chỉ giữ trong tiến trình (một worker, như admission) và chỉ giảm xác suất
chạm 429; 429 thật vẫn là chốt chặn cuối.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import math
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from production_legal_qa_rag.config import ThrottleSettings

_WINDOW_SECONDS = 60.0
_KEY_FINGERPRINT_LENGTH = 8


class ThrottleTimeout(Exception):
    """Không đủ ngân sách trong cửa sổ trượt trước hạn chờ tối đa của caller."""


@dataclass(frozen=True)
class Reservation:
    """Phần ngân sách token đã giữ chỗ cho một lời gọi, dùng để ``settle`` sau."""

    reservation_id: int
    estimated_tokens: int


@dataclass
class _WindowEntry:
    """Một lời gọi trong cửa sổ: thời điểm giữ chỗ và số token đang tính."""

    started_at: float
    tokens: int


class TokenWindowThrottle:
    """Cửa sổ trượt theo cả token (TPM) lẫn số request (RPM), hàng đợi FIFO.

    ``asyncio.Lock`` chỉ có tác dụng trong một tiến trình/một worker. Lock được
    tạo lại khi event loop đổi để một instance dùng được qua nhiều ``asyncio.run``.
    """

    def __init__(
        self,
        tpm_limit: int,
        rpm_limit: int,
        safety_factor: float = 0.9,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Khởi tạo throttle.

        Args:
            tpm_limit: Giới hạn token mỗi phút gốc của Groq.
            rpm_limit: Giới hạn request mỗi phút gốc của Groq.
            safety_factor: Hệ số nhân vào hai giới hạn để chừa biên an toàn.
            clock: Đồng hồ đơn điệu, inject khi test.
            sleep: Hàm chờ bất đồng bộ, inject khi test.
        """
        self._tpm_budget = max(1, int(tpm_limit * safety_factor))
        self._rpm_budget = max(1, int(rpm_limit * safety_factor))
        self._clock = clock
        self._sleep = sleep
        self._entries: OrderedDict[int, _WindowEntry] = OrderedDict()
        self._next_id = 0
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def acquire(
        self, estimated_tokens: int, max_wait_seconds: float
    ) -> Reservation:
        """Giữ chỗ ngân sách cho một lời gọi, chờ nếu cửa sổ sắp đầy.

        Đủ ngân sách thì trả về ngay, không cộng độ trễ. Lời gọi lớn hơn cả ngân
        sách TPM được tính bằng đúng ngân sách đó (chờ cửa sổ trống) thay vì chờ
        vô hạn.

        Args:
            estimated_tokens: Ước lượng tổng token (prompt + completion).
            max_wait_seconds: Thời gian chờ tối đa, gồm cả thời gian xếp hàng.

        Returns:
            ``Reservation`` để truyền vào ``settle`` khi có usage thật.

        Raises:
            ThrottleTimeout: Cần chờ quá ``max_wait_seconds``.
        """
        tokens = min(max(0, estimated_tokens), self._tpm_budget)
        deadline = self._clock() + max_wait_seconds
        try:
            async with asyncio.timeout(max_wait_seconds):
                async with self._get_lock():
                    return await self._reserve(tokens, deadline)
        except TimeoutError as error:
            raise ThrottleTimeout(
                f"Xếp hàng throttle quá {max_wait_seconds:.1f}s."
            ) from error

    async def _reserve(self, tokens: int, deadline: float) -> Reservation:
        while True:
            now = self._clock()
            self._evict_expired(now)
            wait_seconds = self._seconds_until_fits(tokens, now)
            if wait_seconds <= 0:
                return self._record(tokens, now)
            if now + wait_seconds > deadline:
                raise ThrottleTimeout(
                    f"Cần chờ {wait_seconds:.1f}s, vượt hạn chờ tối đa."
                )
            await self._sleep(wait_seconds)

    def settle(self, reservation: Reservation, actual_tokens: int | None) -> None:
        """Thay số ước lượng bằng token thật; ``None`` thì giữ số ước lượng.

        Args:
            reservation: Kết quả của ``acquire`` tương ứng.
            actual_tokens: ``usage.total_tokens`` của response, nếu có.
        """
        entry = self._entries.get(reservation.reservation_id)
        if entry is not None and actual_tokens is not None:
            entry.tokens = max(0, actual_tokens)

    def _evict_expired(self, now: float) -> None:
        while self._entries:
            oldest = next(iter(self._entries.values()))
            if now - oldest.started_at < _WINDOW_SECONDS:
                return
            self._entries.popitem(last=False)

    def _seconds_until_fits(self, tokens: int, now: float) -> float:
        """Số giây phải chờ để một lời gọi ``tokens`` lọt vào cửa sổ (0 = vừa)."""
        used_tokens = sum(entry.tokens for entry in self._entries.values())
        used_requests = len(self._entries)
        if (
            used_requests < self._rpm_budget
            and used_tokens + tokens <= self._tpm_budget
        ):
            return 0.0
        for entry in self._entries.values():
            used_tokens -= entry.tokens
            used_requests -= 1
            if (
                used_requests < self._rpm_budget
                and used_tokens + tokens <= self._tpm_budget
            ):
                return max(0.0, entry.started_at + _WINDOW_SECONDS - now)
        return 0.0

    def _record(self, tokens: int, now: float) -> Reservation:
        reservation_id = self._next_id
        self._next_id += 1
        self._entries[reservation_id] = _WindowEntry(started_at=now, tokens=tokens)
        return Reservation(reservation_id=reservation_id, estimated_tokens=tokens)


def estimate_tokens(
    prompt_characters: int, chars_per_token: float, expected_completion_tokens: int
) -> int:
    """Ước lượng token của một lời gọi: prompt theo số ký tự + completion kỳ vọng.

    Không dùng ``max_completion_tokens`` làm ước lượng vì condense và HyDE đặt 2048,
    cộng lại đã vượt TPM; sai số được ``settle`` bù bằng usage thật.
    """
    return math.ceil(prompt_characters / chars_per_token) + expected_completion_tokens


def read_total_tokens(response: object) -> int | None:
    """Đọc ``usage.total_tokens`` của response Groq; ``None`` nếu không có."""
    total = getattr(getattr(response, "usage", None), "total_tokens", None)
    return total if isinstance(total, int) else None


def describe_bucket(model: str, api_key: str) -> str:
    """Định danh bucket ``(model, key)`` không lộ key thật, dùng gắn metadata.

    Cùng cách băm với :func:`get_throttle` nên áp dụng được cho MỌI bước gọi
    LLM (kể cả bước không qua throttle này, ví dụ generation/guardrail —
    observability_spec.md mục 4.3) để Langfuse hiển thị đúng model/key nào
    đang được dùng, không phải chỉ 3 bước có throttle chung.

    Args:
        model: Tên model Groq.
        api_key: API key của bucket.

    Returns:
        Chuỗi ``"{model}:{fingerprint}"`` ổn định cho cùng một cặp.
    """
    fingerprint = hashlib.sha256(api_key.encode()).hexdigest()[:_KEY_FINGERPRINT_LENGTH]
    return f"{model}:{fingerprint}"


def get_throttle(model: str, api_key: str) -> TokenWindowThrottle:
    """Lấy instance throttle duy nhất của bucket ``(model, api_key)``.

    Bucket được định danh bằng ``model`` và sha256(api_key)[:8]: key thật không
    bao giờ được lưu hay log. Nhờ khoá theo key thật, hai bước cấu hình cùng key
    tự dùng chung throttle, khác key thì tách riêng.

    Args:
        model: Tên model Groq.
        api_key: API key của bucket.

    Returns:
        Throttle dùng chung của bucket, dựng từ ``ThrottleSettings`` lần đầu.
    """
    fingerprint = hashlib.sha256(api_key.encode()).hexdigest()[:_KEY_FINGERPRINT_LENGTH]
    return _get_bucket_throttle(model, fingerprint)


@functools.cache
def _get_bucket_throttle(model: str, key_fingerprint: str) -> TokenWindowThrottle:
    # `model` và `key_fingerprint` chỉ là khoá cache: mỗi bucket đúng một instance.
    settings = ThrottleSettings()
    return TokenWindowThrottle(
        tpm_limit=settings.tpm_limit,
        rpm_limit=settings.rpm_limit,
        safety_factor=settings.safety_factor,
    )
