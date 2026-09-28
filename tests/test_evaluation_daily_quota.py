"""Test circuit breaker hết quota theo ngày của `GroqRoundRobinChatModel` (evaluation_spec.md mục 3.1, 4.5, 8).

Bổ sung cho `test_evaluation_components.py`: phân biệt 429 theo ngày (dừng, không retry) với
429 theo phút (tạm thời, được phép thử lại), lỗi tất định (400/401/403/413) không bị nuốt
sang tài khoản khác, và breaker cũng chặn đường async mà ragas dùng. Không cần `ragas`,
không cần key Groq, không gọi mạng.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from openai import (
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
)

from production_legal_qa_rag.evaluation.groq_round_robin import (
    DailyQuotaExhaustedError,
    GroqRoundRobinChatModel,
)

_PER_MINUTE = "Rate limit reached ... on tokens per minute (TPM): Limit 8000"


def _status_error(cls: type[Any], status: int, message: str) -> Any:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls(message, response=httpx.Response(status, request=request), body=None)


def _ok() -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


class _ScriptedClients:
    """N `ChatOpenAI` thật (không gọi mạng); mỗi client ném lỗi do `errors[i]()` tạo ra."""

    def __init__(self, errors: list[Any]) -> None:
        self.errors = errors
        self.calls = [0] * len(errors)
        self.healthy = False
        self.clients = [
            ChatOpenAI(api_key=f"key-{index}", model="openai/gpt-oss-120b")
            for index in range(len(errors))
        ]
        for index, client in enumerate(self.clients):
            client._generate = self._make(index)  # type: ignore[method-assign]

    def _make(self, index: int) -> Any:
        def _generate(_messages: object, **_kwargs: object) -> ChatResult:
            self.calls[index] += 1
            if self.healthy:
                return _ok()
            raise self.errors[index]()

        return _generate

    def router(self) -> GroqRoundRobinChatModel:
        return GroqRoundRobinChatModel(clients=self.clients)


def _rate_limit(message: str) -> Any:
    return lambda: _status_error(RateLimitError, 429, message)


@pytest.mark.parametrize(
    "message",
    [
        "Rate limit reached on tokens per day (TPD): Limit 200000",
        "Rate limit reached on requests per day (RPD): Limit 1000",
        "RATE LIMIT REACHED ON TOKENS PER DAY",
    ],
)
def test_moi_dau_hieu_het_quota_ngay_deu_kich_hoat_breaker_sau_dung_1_vong(
    message: str,
):
    scripted = _ScriptedClients([_rate_limit(message)] * 4)
    router = scripted.router()

    with pytest.raises(DailyQuotaExhaustedError) as excinfo:
        router._generate(messages=[])

    assert scripted.calls == [1, 1, 1, 1]  # đúng 1 vòng, không thử lại
    assert isinstance(excinfo.value.__cause__, RateLimitError)
    assert "4 tài khoản" in str(excinfo.value)


def test_breaker_dung_yen_luot_sau_khong_gui_them_request_nao_ke_ca_khi_tai_khoan_hoi_lai():
    scripted = _ScriptedClients([_rate_limit("... tokens per day (TPD) ...")] * 3)
    router = scripted.router()
    with pytest.raises(DailyQuotaExhaustedError):
        router._generate(messages=[])
    scripted.healthy = True  # dù mạng/tài khoản "khoẻ" lại, process này không thử nữa

    for _ in range(5):
        with pytest.raises(DailyQuotaExhaustedError):
            router._generate(messages=[])

    assert scripted.calls == [1, 1, 1]
    assert router.call_counts == [1, 1, 1]


def test_breaker_chan_ca_duong_async_ma_ragas_dung_khong_gui_them_request():
    scripted = _ScriptedClients([_rate_limit("... tokens per day (TPD) ...")] * 2)
    router = scripted.router()
    with pytest.raises(DailyQuotaExhaustedError):
        asyncio.run(router._agenerate(messages=[]))

    with pytest.raises(DailyQuotaExhaustedError):
        asyncio.run(router._agenerate(messages=[]))

    assert scripted.calls == [1, 1]


def test_429_theo_phut_khong_kich_hoat_breaker_va_luot_sau_thanh_cong_khi_het_gioi_han():
    scripted = _ScriptedClients([_rate_limit(_PER_MINUTE)] * 3)
    router = scripted.router()

    with pytest.raises(RateLimitError) as first:
        router._generate(messages=[])
    scripted.healthy = True  # sang phút mới, giới hạn theo phút được reset
    result = router._generate(messages=[])

    assert not isinstance(first.value, DailyQuotaExhaustedError)
    assert result.generations[0].text == "ok"


def test_chi_can_mot_tai_khoan_bao_theo_phut_trong_vong_thi_khong_phai_het_quota_ngay():
    daily = _rate_limit("... tokens per day (TPD) ...")
    scripted = _ScriptedClients([daily, _rate_limit(_PER_MINUTE), daily])
    router = scripted.router()

    with pytest.raises(RateLimitError) as excinfo:
        router._generate(messages=[])
    with pytest.raises(RateLimitError):
        router._generate(messages=[])  # chưa bị chặn: vẫn gọi lại cả 3

    assert not isinstance(excinfo.value, DailyQuotaExhaustedError)
    assert scripted.calls == [2, 2, 2]


@pytest.mark.parametrize(
    ("cls", "status"),
    [
        (BadRequestError, 400),
        (AuthenticationError, 401),
        (PermissionDeniedError, 403),
        (APIStatusError, 413),
    ],
)
def test_loi_tat_dinh_khong_bi_nuot_sang_tai_khoan_khac_va_khong_kich_hoat_breaker(
    cls: type[Any], status: int
):
    scripted = _ScriptedClients(
        [lambda: _status_error(cls, status, "Request too large ... per day")] * 3
    )
    router = scripted.router()

    with pytest.raises(cls) as excinfo:
        router._generate(messages=[])
    scripted.healthy = True
    result = router._generate(messages=[])  # không bị "chốt" thành hết quota

    assert not isinstance(excinfo.value, DailyQuotaExhaustedError)
    assert scripted.calls == [1, 1, 0]  # dừng ngay ở lần lỗi đầu, lượt sau ở client kế
    assert result.generations[0].text == "ok"
