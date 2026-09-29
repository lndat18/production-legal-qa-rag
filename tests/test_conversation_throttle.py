"""Test condense nối throttle dùng chung với HyDE (conversation_spec.md mục 12.1)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from production_legal_qa_rag.config import (
    CondenseSettings,
    HydeSettings,
    ThrottleSettings,
)
from production_legal_qa_rag.conversation.condenser import (
    CONDENSE_SYSTEM_PROMPT,
    CondenseReason,
    QueryCondenser,
    build_condense_user_message,
)
from production_legal_qa_rag.conversation.models import ChatMessage
from production_legal_qa_rag.retrieval.hyde import HydeGenerator
from production_legal_qa_rag.retrieval.llm_throttle import (
    Reservation,
    ThrottleTimeout,
    TokenWindowThrottle,
    estimate_tokens,
    get_throttle,
)

QUERY = "Còn Khoản 2?"
CONDENSED = "Khoản 2 Điều 113 Bộ luật Lao động nói gì?"
HISTORY = [
    ChatMessage(role="user", content="Khoản 1 Điều 113 BLLĐ nói gì?"),
    ChatMessage(role="assistant", content="Nói về nghỉ hằng năm."),
]


class _FakeClock:
    """Đồng hồ giả: `sleep` chỉ dịch thời gian tiến lên, ghi lại thời lượng chờ."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


class _FakeGroq:
    def __init__(
        self,
        content: str = CONDENSED,
        finish_reason: str = "stop",
        total_tokens: int | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._content = content
        self._finish_reason = finish_reason
        self._total_tokens = total_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message, finish_reason=self._finish_reason)
        usage = SimpleNamespace(
            completion_tokens=40,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=25),
            total_tokens=self._total_tokens,
        )
        return SimpleNamespace(choices=[choice], usage=usage)


class _RecordingThrottle:
    """Throttle giả ghi lại tham số `acquire` / `settle`."""

    def __init__(self) -> None:
        self.acquired: list[tuple[int, float]] = []
        self.settled: list[tuple[Reservation, int | None]] = []

    async def acquire(
        self, estimated_tokens: int, max_wait_seconds: float
    ) -> Reservation:
        self.acquired.append((estimated_tokens, max_wait_seconds))
        return Reservation(reservation_id=len(self.acquired), estimated_tokens=1)

    def settle(self, reservation: Reservation, actual_tokens: int | None) -> None:
        self.settled.append((reservation, actual_tokens))


def _throttle(clock: _FakeClock, *, rpm: int) -> TokenWindowThrottle:
    return TokenWindowThrottle(10**6, rpm, 1.0, clock=clock, sleep=clock.sleep)


def _exhausted_throttle(clock: _FakeClock) -> TokenWindowThrottle:
    """Throttle đã dùng hết RPM: lời gọi kế tiếp phải chờ 60s."""
    throttle = _throttle(clock, rpm=1)
    asyncio.run(throttle.acquire(1, 1))
    return throttle


def _condenser(fake: _FakeGroq, throttle: Any, **kwargs: Any) -> QueryCondenser:
    settings = CondenseSettings(GROQ_API_KEY_1="k")
    return QueryCondenser(settings, fake, throttle=throttle, **kwargs)


def test_condense_het_han_cho_throttle_thi_dung_cau_goc_va_khong_goi_groq() -> None:
    clock = _FakeClock()
    fake = _FakeGroq()
    condenser = _condenser(fake, _exhausted_throttle(clock))

    outcome = asyncio.run(condenser.condense_detailed(QUERY, HISTORY))

    assert outcome.text == QUERY
    assert outcome.reason is CondenseReason.GROQ_ERROR
    assert fake.calls == []
    assert clock.sleeps == []


def test_condense_public_api_tra_cau_goc_khi_throttle_het_han() -> None:
    condenser = _condenser(_FakeGroq(), _exhausted_throttle(_FakeClock()))

    assert asyncio.run(condenser.condense(QUERY, HISTORY)) == QUERY


def test_condense_khong_history_khong_cham_vao_throttle() -> None:
    throttle = _RecordingThrottle()
    fake = _FakeGroq()

    outcome = asyncio.run(_condenser(fake, throttle).condense_detailed(QUERY, []))

    assert outcome.reason is CondenseReason.NO_HISTORY
    assert throttle.acquired == []
    assert fake.calls == []


def test_condense_cho_toi_da_theo_optional_step_max_wait_seconds() -> None:
    clock = _FakeClock()
    fake = _FakeGroq()
    condenser = _condenser(
        fake,
        _exhausted_throttle(clock),
        throttle_settings=ThrottleSettings(optional_step_max_wait_seconds=100),
    )

    outcome = asyncio.run(condenser.condense_detailed(QUERY, HISTORY))

    assert outcome.reason is CondenseReason.OK
    assert outcome.text == CONDENSED
    assert clock.sleeps == [60.0]
    assert len(fake.calls) == 1


def test_condense_acquire_dung_max_wait_va_uoc_luong_theo_prompt() -> None:
    throttle = _RecordingThrottle()
    settings = ThrottleSettings()

    asyncio.run(_condenser(_FakeGroq(), throttle).condense(QUERY, HISTORY))

    expected = estimate_tokens(
        len(CONDENSE_SYSTEM_PROMPT) + len(build_condense_user_message(QUERY, HISTORY)),
        settings.chars_per_token,
        settings.condense_completion_tokens,
    )
    assert throttle.acquired == [(expected, settings.optional_step_max_wait_seconds)]
    assert settings.optional_step_max_wait_seconds == 8.0


def test_condense_settle_bang_usage_that_cua_response() -> None:
    throttle = _RecordingThrottle()

    asyncio.run(
        _condenser(_FakeGroq(total_tokens=777), throttle).condense(QUERY, HISTORY)
    )

    assert [actual for _, actual in throttle.settled] == [777]


def test_condense_settle_none_khi_response_khong_co_total_tokens() -> None:
    throttle = _RecordingThrottle()

    asyncio.run(
        _condenser(_FakeGroq(total_tokens=None), throttle).condense(QUERY, HISTORY)
    )

    assert [actual for _, actual in throttle.settled] == [None]


def test_condense_retry_finish_length_di_qua_throttle_moi_lan_goi() -> None:
    throttle = _RecordingThrottle()
    fake = _FakeGroq(content="", finish_reason="length")

    outcome = asyncio.run(_condenser(fake, throttle).condense_detailed(QUERY, HISTORY))

    assert outcome.reason is CondenseReason.FINISH_LENGTH
    assert len(fake.calls) == 2
    assert len(throttle.acquired) == 2
    assert len(throttle.settled) == 2


def test_condense_retry_het_han_throttle_thi_dung_ket_qua_lan_2_la_groq_error() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, rpm=2)
    asyncio.run(throttle.acquire(1, 1))  # còn đúng 1 request cho lần gọi đầu
    fake = _FakeGroq(content="", finish_reason="length")

    outcome = asyncio.run(_condenser(fake, throttle).condense_detailed(QUERY, HISTORY))

    assert len(fake.calls) == 1
    assert outcome.reason is CondenseReason.GROQ_ERROR
    assert outcome.text == QUERY


def test_condense_va_hyde_cung_model_cung_key_dung_chung_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THROTTLE_RPM_LIMIT", "1")
    condense_fake = _FakeGroq()
    hyde_fake = _FakeGroq(content="đoạn văn")
    condenser = QueryCondenser(CondenseSettings(GROQ_API_KEY_1="k"), condense_fake)
    hyde = HydeGenerator(HydeSettings(GROQ_API_KEY_1="k"), hyde_fake)

    outcome = asyncio.run(condenser.condense_detailed(QUERY, HISTORY))
    hyde_text = asyncio.run(hyde.generate("hỏi"))

    assert outcome.reason is CondenseReason.OK
    # Condense đã tiêu request duy nhất của bucket nên HyDE bị throttle chặn.
    assert hyde_text is None
    assert hyde_fake.calls == []


def test_condense_va_hyde_khac_key_khong_dung_chung_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THROTTLE_RPM_LIMIT", "1")
    condenser = QueryCondenser(CondenseSettings(GROQ_API_KEY_1="k1"), _FakeGroq())
    hyde_fake = _FakeGroq(content="đoạn văn")
    hyde = HydeGenerator(HydeSettings(GROQ_API_KEY_1="k2"), hyde_fake)

    outcome = asyncio.run(condenser.condense_detailed(QUERY, HISTORY))
    hyde_text = asyncio.run(hyde.generate("hỏi"))

    assert outcome.reason is CondenseReason.OK
    assert hyde_text == "đoạn văn"
    bucket_1 = get_throttle("openai/gpt-oss-20b", "k1")
    bucket_2 = get_throttle("openai/gpt-oss-20b", "k2")
    assert bucket_1 is not bucket_2
    with pytest.raises(ThrottleTimeout):
        asyncio.run(bucket_1.acquire(1, 0.05))
