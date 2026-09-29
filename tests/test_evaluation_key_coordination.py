"""Test điều phối key mượt và tiết kiệm token của `GroqRoundRobinChatModel` (evaluation_spec.md mục 3.2).

Gồm: cooldown 429 theo phút theo `retry-after` (A), đếm token thật theo tài khoản (B) và
`reasoning_effort` đặt tạm bằng context manager (C). Không cần `ragas`, không cần key Groq,
không gọi mạng: đồng hồ và `sleep` được tiêm vào router, còn request Groq của phần kiểm
`reasoning_effort` đi qua `httpx.MockTransport`.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from openai import BadRequestError, RateLimitError

from production_legal_qa_rag.evaluation import groq_round_robin
from production_legal_qa_rag.evaluation.groq_round_robin import (
    DailyQuotaExhaustedError,
    GroqRoundRobinChatModel,
    TokenTotals,
)

_PER_MINUTE = "Rate limit reached ... on tokens per minute (TPM): Limit 8000"
_PER_DAY = "Rate limit reached ... on tokens per day (TPD): Limit 200000"


def _rate_limit(message: str, retry_after: str | None = None) -> RateLimitError:
    headers = {} if retry_after is None else {"retry-after": retry_after}
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request, headers=headers)
    return RateLimitError(message, response=response, body=None)


def _usage_result(
    prompt: int = 10, completion: int = 5, reasoning: int | None = 2
) -> ChatResult:
    usage: dict[str, Any] = {"prompt_tokens": prompt}
    usage["completion_tokens"] = completion
    if reasoning is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content="ok"))],
        llm_output={"token_usage": usage},
    )


def _usage_with_none_completion() -> ChatResult:
    usage = {"prompt_tokens": 7, "completion_tokens": None}
    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content="ok"))],
        llm_output={"token_usage": usage},
    )


def _result_without_usage() -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


def _slow_usage_result() -> ChatResult:
    time.sleep(0.0003)  # nhả GIL để các luồng xen kẽ thật
    return _usage_result(3, 2, 1)


def _raising(make_error: Callable[[], Exception]) -> Callable[[], ChatResult]:
    def _behaviour() -> ChatResult:
        raise make_error()

    return _behaviour


def _minute_limited(retry_after: str | None = None) -> Callable[[], ChatResult]:
    return _raising(lambda: _rate_limit(_PER_MINUTE, retry_after))


def _bad_request() -> BadRequestError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError("sai", response=response, body=None)


def _day_limited() -> Callable[[], ChatResult]:
    return _raising(lambda: _rate_limit(_PER_DAY))


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Sleeper:
    """`sleep` giả: ghi số giây được xin ngủ và đẩy đồng hồ tiến đúng chừng đó."""

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.calls: list[float] = []
        self.on_sleep: Callable[[], None] | None = None

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self.on_sleep is not None:
            self.on_sleep()
        self.clock.now += seconds


class _Rig:
    """Router + N `ChatOpenAI` thật (không gọi mạng) + đồng hồ/`sleep` giả.

    Hành vi của client `i` do `behaviours[i]()` quyết định (trả `ChatResult` hoặc ném lỗi);
    `calls[i]`/`kwargs[i]` ghi lại các lần `_generate` của đúng client đó.
    """

    def __init__(self, count: int) -> None:
        self.clock = _Clock()
        self.sleeper = _Sleeper(self.clock)
        self.lock = threading.Lock()
        self.calls = [0] * count
        self.kwargs: list[list[dict[str, Any]]] = [[] for _ in range(count)]
        self.behaviours: list[Callable[[], ChatResult]] = [_usage_result] * count
        self.clients = [
            ChatOpenAI(api_key=f"key-{index}", model="openai/gpt-oss-120b")
            for index in range(count)
        ]
        for index, client in enumerate(self.clients):
            client._generate = self._make(index)  # type: ignore[method-assign]
        self.router = GroqRoundRobinChatModel(
            clients=self.clients, clock=self.clock, sleep=self.sleeper
        )

    def _make(self, index: int) -> Callable[..., ChatResult]:
        def _generate(_messages: object, **kwargs: Any) -> ChatResult:
            with self.lock:
                self.calls[index] += 1
                self.kwargs[index].append(kwargs)
            return self.behaviours[index]()

        return _generate


# ==========================================================================
# A. Cooldown 429 theo phút: `retry-after`, kẹp [1, 60], mặc định 15
# ==========================================================================


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [
        ("7", 7.0),
        ("2.5", 2.5),
        ("1", 1.0),
        ("60", 60.0),
        ("0", 1.0),
        ("0.2", 1.0),
        ("-5", 1.0),
        ("3600", 60.0),
        (None, 15.0),
        ("", 15.0),
        ("abc", 15.0),
        ("1m1.5s", 15.0),
    ],
)
def test_retry_after_duoc_kep_1_den_60_giay_thieu_hoac_hong_thi_mac_dinh_15(
    retry_after: str | None, expected: float
):
    error = _rate_limit(_PER_MINUTE, retry_after)

    assert groq_round_robin._retry_after_seconds(error) == expected


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [("7", 7.0), ("2.5", 2.5), (None, 15.0), ("0", 1.0), ("600", 60.0)],
)
def test_429_theo_phut_dat_cooldown_dung_so_giay_cho_tai_khoan_vua_bi_gioi_han(
    retry_after: str | None, expected: float
):
    rig = _Rig(3)
    rig.behaviours[0] = _minute_limited(retry_after)

    rig.router._generate(messages=[])  # client 0 bị 429, client 1 trả lời

    assert rig.router._cooldown_until[0] == pytest.approx(rig.clock.now + expected)
    # 429 theo phút không phải bằng chứng hết ngày nên không đụng cooldown ngày.
    assert rig.router._daily_cooldown_until[0] == 0.0
    assert rig.router._cooldown_until[1:] == [0.0, 0.0]
    assert rig.sleeper.calls == []


def test_tai_khoan_vua_bi_429_theo_phut_xep_cuoi_trong_cooldown_roi_tro_lai_dung_cho():
    rig = _Rig(3)
    rig.router._mark_minute_limited(0, _rate_limit(_PER_MINUTE, "10"))

    during = [rig.router._plan_attempts() for _ in range(3)]
    rig.clock.now += 11
    after = [rig.router._plan_attempts() for _ in range(3)]

    assert during == [[1, 2, 0], [1, 2, 0], [2, 1, 0]]
    assert after == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]


def test_cooldown_phut_khong_bao_gio_rut_ngan_cooldown_ngay_dang_co():
    rig = _Rig(3)
    router = rig.router
    router._mark_result(0, daily_limited=True)
    daily_until = rig.clock.now + groq_round_robin._DAILY_COOLDOWN_SECONDS

    router._mark_minute_limited(0, _rate_limit(_PER_MINUTE, "5"))

    assert router._cooldown_until[0] == daily_until
    assert router._daily_cooldown_until[0] == daily_until


def test_cooldown_phut_dai_hon_phan_ngay_con_lai_thi_lay_gia_tri_lon_hon():
    rig = _Rig(3)
    router = rig.router
    router._mark_result(0, daily_limited=True)
    daily_until = rig.clock.now + groq_round_robin._DAILY_COOLDOWN_SECONDS
    rig.clock.now += 297  # còn 3 giây của cooldown ngày

    router._mark_minute_limited(0, _rate_limit(_PER_MINUTE, "30"))

    assert router._cooldown_until[0] == pytest.approx(rig.clock.now + 30)
    # Bằng chứng hết ngày không bị 429 theo phút ghi đè.
    assert router._daily_cooldown_until[0] == daily_until


def test_429_theo_phut_lan_sau_ngan_hon_khong_rut_ngan_cooldown_phut_da_co():
    rig = _Rig(2)
    router = rig.router

    router._mark_minute_limited(0, _rate_limit(_PER_MINUTE, "30"))
    router._mark_minute_limited(0, _rate_limit(_PER_MINUTE, "5"))

    assert router._cooldown_until[0] == pytest.approx(rig.clock.now + 30)


# --------------------------------------------------------------------------
# Chờ khi CẢ n tài khoản cùng cooldown phút
# --------------------------------------------------------------------------


def test_ca_n_tai_khoan_cung_cooldown_phut_thi_cho_den_khi_tai_khoan_som_nhat_het_cooldown():
    rig = _Rig(3)
    for index, retry_after in enumerate(("20", "5", "10")):
        rig.behaviours[index] = _minute_limited(retry_after)
    with pytest.raises(RateLimitError):
        rig.router._generate(messages=[])  # lượt đầu chưa có cooldown nên chưa chờ
    assert rig.sleeper.calls == []
    rig.behaviours[1] = _usage_result  # tài khoản có cooldown ngắn nhất đã khoẻ lại

    result = rig.router._generate(messages=[])

    assert result.generations[0].text == "ok"
    assert rig.sleeper.calls == [pytest.approx(5.0)]
    # Sau khi chờ, tài khoản 1 hết cooldown nên được thử đầu tiên; 0 và 2 không bị đập lại.
    assert rig.calls == [1, 2, 1]


def test_thoi_gian_cho_bi_kep_toi_da_60_giay_du_retry_after_rat_lon():
    rig = _Rig(3)
    rig.behaviours = [_minute_limited("3600")] * 3
    with pytest.raises(RateLimitError):
        rig.router._generate(messages=[])
    rig.behaviours = [_usage_result] * 3

    rig.router._generate(messages=[])

    assert rig.sleeper.calls == [pytest.approx(60.0)]


def test_khong_cho_khi_cooldown_phut_da_het():
    rig = _Rig(3)
    rig.behaviours = [_minute_limited("10")] * 3
    with pytest.raises(RateLimitError):
        rig.router._generate(messages=[])
    rig.behaviours = [_usage_result] * 3
    rig.clock.now += 11

    result = rig.router._generate(messages=[])

    assert result.generations[0].text == "ok"
    assert rig.sleeper.calls == []


def test_khong_cho_khi_con_it_nhat_mot_tai_khoan_khong_cooldown():
    rig = _Rig(3)
    rig.router._mark_minute_limited(0, _rate_limit(_PER_MINUTE, "10"))
    rig.router._mark_minute_limited(1, _rate_limit(_PER_MINUTE, "10"))

    result = rig.router._generate(messages=[])

    assert result.generations[0].text == "ok"
    assert rig.sleeper.calls == []
    assert rig.calls == [0, 0, 1]  # dùng ngay tài khoản 2, không đập vào 0 và 1


def test_khong_cho_khi_co_cooldown_ngay_de_lay_bang_chung_moi_cho_breaker():
    rig = _Rig(3)
    rig.router._mark_result(0, daily_limited=True)
    rig.router._mark_minute_limited(1, _rate_limit(_PER_MINUTE, "10"))
    rig.router._mark_minute_limited(2, _rate_limit(_PER_MINUTE, "10"))

    # Cả 3 đều cooldown nhưng có cooldown ngày nên phải thử thật, không ngủ.
    result = rig.router._generate(messages=[])

    assert result.generations[0].text == "ok"
    assert rig.sleeper.calls == []
    assert rig.calls == [1, 0, 0]  # thử ngay theo vòng quay thuần, không ngủ


def test_ca_n_cooldown_ngay_van_thu_de_lay_bang_chung_khong_ngu_roi_bat_breaker():
    rig = _Rig(3)
    rig.behaviours = [_day_limited()] * 3
    for index in range(3):
        rig.router._mark_result(index, daily_limited=True)

    with pytest.raises(DailyQuotaExhaustedError):
        rig.router._generate(messages=[])
    with pytest.raises(DailyQuotaExhaustedError):
        rig.router._generate(messages=[])  # breaker đã bật: từ chối ngay

    assert rig.sleeper.calls == []
    assert rig.calls == [1, 1, 1]


def test_breaker_da_bat_thi_tu_choi_ngay_khong_ngu_du_con_cooldown_phut():
    rig = _Rig(3)
    rig.behaviours = [_day_limited()] * 3
    with pytest.raises(DailyQuotaExhaustedError):
        rig.router._generate(messages=[])
    rig.router._mark_minute_limited(0, _rate_limit(_PER_MINUTE, "30"))

    with pytest.raises(DailyQuotaExhaustedError):
        rig.router._generate(messages=[])

    assert rig.sleeper.calls == []
    assert rig.calls == [1, 1, 1]


def test_cho_xong_ma_ca_n_tai_khoan_bao_het_ngay_thi_van_bat_breaker_dung_nhu_cu():
    rig = _Rig(3)
    rig.behaviours = [_minute_limited("10")] * 3
    with pytest.raises(RateLimitError):
        rig.router._generate(messages=[])
    rig.behaviours = [_day_limited()] * 3

    with pytest.raises(DailyQuotaExhaustedError) as excinfo:
        rig.router._generate(messages=[])
    with pytest.raises(DailyQuotaExhaustedError):
        rig.router._generate(messages=[])

    assert "3 tài khoản" in str(excinfo.value)
    # Chỉ chờ đúng một lần (trước lượt thử); lượt thứ ba bị chặn ngay, không request nào.
    assert rig.sleeper.calls == [pytest.approx(10.0)]
    assert rig.calls == [2, 2, 2]


def test_429_theo_phut_lap_lai_qua_nhieu_luot_khong_bao_gio_bat_breaker():
    rig = _Rig(3)
    rig.behaviours = [_minute_limited("10")] * 3
    errors: list[BaseException] = []

    for _ in range(3):
        with pytest.raises(RateLimitError) as excinfo:
            rig.router._generate(messages=[])
        errors.append(excinfo.value)

    assert all(type(error) is RateLimitError for error in errors)
    assert rig.router._daily_quota_message is None
    assert rig.sleeper.calls == [pytest.approx(10.0), pytest.approx(10.0)]


def test_cho_khong_giu_lock_luong_khac_van_doc_duoc_trang_thai_router_khi_dang_ngu():
    rig = _Rig(2)
    rig.behaviours = [_minute_limited("10")] * 2
    with pytest.raises(RateLimitError):
        rig.router._generate(messages=[])
    rig.behaviours = [_usage_result] * 2
    readable: list[int] = []

    def _read_from_another_thread() -> None:
        reader = threading.Thread(
            target=lambda: readable.append(len(rig.router.call_counts)), daemon=True
        )
        reader.start()
        reader.join(timeout=5)

    rig.sleeper.on_sleep = _read_from_another_thread

    rig.router._generate(messages=[])

    assert rig.sleeper.calls == [pytest.approx(10.0)]
    # Đọc được `call_counts` (cần lock) ngay lúc router đang ngủ.
    assert readable == [2]


# ==========================================================================
# B. Đếm token thật theo tài khoản (mục 3.2 B)
# ==========================================================================


def test_token_totals_total_tokens_la_prompt_cong_completion_khong_cong_them_reasoning():
    totals = TokenTotals(prompt_tokens=10, completion_tokens=5, reasoning_tokens=3)

    assert totals.total_tokens == 15  # reasoning là phần con của completion


def test_token_totals_cong_don_prompt_completion_reasoning_theo_tung_client():
    rig = _Rig(3)
    for index in range(3):
        prompt = 10 * (index + 1)
        rig.behaviours[index] = functools.partial(_usage_result, prompt, 5, 2)

    for _ in range(6):  # mỗi client được chọn đầu tiên đúng 2 lần
        rig.router._generate(messages=[])

    assert rig.router.token_totals == [
        TokenTotals(prompt_tokens=20, completion_tokens=10, reasoning_tokens=4),
        TokenTotals(prompt_tokens=40, completion_tokens=10, reasoning_tokens=4),
        TokenTotals(prompt_tokens=60, completion_tokens=10, reasoning_tokens=4),
    ]
    assert rig.router.token_totals[1].total_tokens == 50


def test_luot_bi_429_khong_cong_token_chi_luot_thanh_cong_moi_tinh():
    rig = _Rig(2)
    rig.behaviours[0] = _minute_limited("5")

    rig.router._generate(messages=[])  # client 0 bị 429, client 1 thành công

    totals = rig.router.token_totals
    assert totals[0] == TokenTotals()
    assert totals[1] == TokenTotals(
        prompt_tokens=10, completion_tokens=5, reasoning_tokens=2
    )
    assert rig.router.call_counts == [1, 1]  # lượt 429 vẫn được đếm là lượt gọi


def test_usage_thieu_reasoning_details_hoac_gia_tri_none_thi_coi_la_0():
    rig = _Rig(2)
    rig.behaviours[0] = functools.partial(_usage_result, 10, 5, None)
    rig.behaviours[1] = _usage_with_none_completion

    rig.router._generate(messages=[])
    rig.router._generate(messages=[])

    assert rig.router.token_totals == [
        TokenTotals(prompt_tokens=10, completion_tokens=5, reasoning_tokens=0),
        TokenTotals(prompt_tokens=7, completion_tokens=0, reasoning_tokens=0),
    ]


def test_phan_hoi_khong_co_token_usage_thi_coi_la_0_va_chi_canh_bao_dung_mot_lan(
    caplog: pytest.LogCaptureFixture,
):
    rig = _Rig(2)
    rig.behaviours = [_result_without_usage] * 2

    with caplog.at_level(logging.WARNING, logger=groq_round_robin.logger.name):
        for _ in range(4):
            rig.router._generate(messages=[])

    warnings = [r for r in caplog.records if "token_usage" in r.getMessage()]
    assert len(warnings) == 1
    assert rig.router.token_totals == [TokenTotals(), TokenTotals()]


def test_token_totals_tra_ban_sao_sua_ban_sao_khong_lam_hong_so_dem_cua_router():
    rig = _Rig(1)
    rig.router._generate(messages=[])

    snapshot = rig.router.token_totals
    snapshot[0].prompt_tokens = 999_999
    snapshot.clear()

    expected = TokenTotals(prompt_tokens=10, completion_tokens=5, reasoning_tokens=2)
    assert rig.router.token_totals == [expected]


def test_token_totals_cong_don_thread_safe_khop_dung_so_lan_goi_thanh_cong():
    rig = _Rig(4)
    rig.behaviours = [_slow_usage_result] * 4
    threads, rounds = 8, 25
    barrier = threading.Barrier(threads)

    def _worker() -> None:
        barrier.wait()
        for _ in range(rounds):
            rig.router._generate(messages=[])

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for future in [pool.submit(_worker) for _ in range(threads)]:
            future.result()

    totals = rig.router.token_totals
    counts = rig.router.call_counts
    calls = threads * rounds
    assert sum(counts) == calls
    assert sum(item.prompt_tokens for item in totals) == 3 * calls
    assert sum(item.completion_tokens for item in totals) == 2 * calls
    assert sum(item.reasoning_tokens for item in totals) == calls
    for item, count in zip(totals, counts, strict=True):
        assert item.prompt_tokens == 3 * count  # từng client, không chỉ tổng


# ==========================================================================
# C. `reasoning_effort` chỉ truyền trong context và luôn được gỡ (mục 3.2 C)
# ==========================================================================


def test_reasoning_effort_mac_dinh_la_none_va_khong_gui_kwarg_nao_xuong_client():
    rig = _Rig(1)

    rig.router._generate(messages=[])

    assert rig.router.current_reasoning_effort is None
    assert "reasoning_effort" not in rig.kwargs[0][0]


def test_reasoning_effort_chi_duoc_truyen_trong_context_va_go_ra_sau_khi_thoat():
    rig = _Rig(1)
    router = rig.router

    router._generate(messages=[])
    with router.reasoning_effort("low"):
        inside = router.current_reasoning_effort
        router._generate(messages=[])
    router._generate(messages=[])

    sent = [kwargs.get("reasoning_effort") for kwargs in rig.kwargs[0]]
    assert inside == "low"
    assert sent == [None, "low", None]
    assert router.current_reasoning_effort is None


def test_reasoning_effort_duoc_go_ke_ca_khi_khoi_with_nem_loi():
    rig = _Rig(1)

    with pytest.raises(RuntimeError):
        with rig.router.reasoning_effort("low"):
            raise RuntimeError("lỗi giữa chừng")

    assert rig.router.current_reasoning_effort is None


def test_reasoning_effort_duoc_go_ke_ca_khi_luot_goi_ben_trong_bi_loi_tat_dinh():
    rig = _Rig(1)
    rig.behaviours = [_raising(_bad_request)]

    with pytest.raises(BadRequestError):
        with rig.router.reasoning_effort("low"):
            rig.router._generate(messages=[])

    assert rig.kwargs[0][0]["reasoning_effort"] == "low"  # đã gửi trước khi lỗi
    assert rig.router.current_reasoning_effort is None


def test_reasoning_effort_long_nhau_tra_ve_gia_tri_truoc_do_khi_thoat_khoi_trong():
    rig = _Rig(1)
    router = rig.router

    with router.reasoning_effort("medium"):
        with router.reasoning_effort("low"):
            innermost = router.current_reasoning_effort
        after_inner = router.current_reasoning_effort
    after_outer = router.current_reasoning_effort

    assert (innermost, after_inner, after_outer) == ("low", "medium", None)


def test_kwarg_cua_nguoi_goi_thang_gia_tri_cua_router():
    rig = _Rig(1)
    router = rig.router

    with router.reasoning_effort("low"):
        router._generate(messages=[], reasoning_effort="high")
    router._generate(messages=[], reasoning_effort="medium")

    sent = [kwargs["reasoning_effort"] for kwargs in rig.kwargs[0]]
    assert sent == ["high", "medium"]


def test_reasoning_effort_co_hieu_luc_voi_moi_luong_ke_ca_duong_async_ma_ragas_dung():
    rig = _Rig(2)
    router = rig.router

    with router.reasoning_effort("low"):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: router._generate(messages=[]), range(8)))
        asyncio.run(router._agenerate(messages=[]))

    sent = [kwargs.get("reasoning_effort") for per in rig.kwargs for kwargs in per]
    assert sent == ["low"] * 9
    assert router.current_reasoning_effort is None


# --------------------------------------------------------------------------
# `ChatOpenAI` thật + `httpx.MockTransport`: kwarg đi tới body request, usage đọc đúng
# --------------------------------------------------------------------------

_GROQ_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "openai/gpt-oss-120b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        },
    ],
    "usage": {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "completion_tokens_details": {"reasoning_tokens": 4},
    },
}


def _mock_groq_client(bodies: list[dict[str, Any]]) -> ChatOpenAI:
    def _handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_GROQ_COMPLETION)

    return ChatOpenAI(
        api_key="key-mock",
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-120b",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )


def test_reasoning_effort_toi_body_request_that_va_usage_cong_don_tu_chat_openai_that():
    bodies: list[dict[str, Any]] = []
    router = GroqRoundRobinChatModel(clients=[_mock_groq_client(bodies)])
    messages = [HumanMessage(content="xin chào")]

    router._generate(messages)
    with router.reasoning_effort("low"):
        router._generate(messages)
        router._generate(messages, reasoning_effort="high")

    assert len(bodies) == 3
    assert "reasoning_effort" not in bodies[0]  # ngoài context: Groq dùng mặc định
    assert bodies[1]["reasoning_effort"] == "low"
    assert bodies[2]["reasoning_effort"] == "high"  # kwarg người gọi thắng
    expected = TokenTotals(prompt_tokens=33, completion_tokens=21, reasoning_tokens=12)
    assert router.token_totals == [expected]
