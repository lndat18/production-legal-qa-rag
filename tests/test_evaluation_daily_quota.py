"""Test circuit breaker hết quota theo ngày của `GroqRoundRobinChatModel` (evaluation_spec.md mục 3.1, 4.5, 8).

Bổ sung cho `test_evaluation_components.py`: phân biệt 429 theo ngày (dừng, không retry) với
429 theo phút (tạm thời, được phép thử lại), lỗi tất định (400/401/403/413) không bị nuốt
sang tài khoản khác, và breaker cũng chặn đường async mà ragas dùng. Không cần `ragas`,
không cần key Groq, không gọi mạng.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

from production_legal_qa_rag.evaluation import groq_round_robin
from production_legal_qa_rag.evaluation.groq_round_robin import (
    DailyQuotaExhaustedError,
    GroqRoundRobinChatModel,
)
from production_legal_qa_rag.evaluation.testset_generator import _describe_error

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
    waits: list[float] = []
    # Cả 3 tài khoản cùng cooldown phút thì router chờ (mục 3.2 A): ghi lại thay vì ngủ thật.
    router = GroqRoundRobinChatModel(clients=scripted.clients, sleep=waits.append)

    with pytest.raises(RateLimitError) as first:
        router._generate(messages=[])
    scripted.healthy = True  # sang phút mới, giới hạn theo phút được reset
    result = router._generate(messages=[])

    assert not isinstance(first.value, DailyQuotaExhaustedError)
    assert result.generations[0].text == "ok"
    # Thông điệp không có `retry-after` nên cooldown mặc định 15 giây; chờ phần còn lại của nó.
    assert len(waits) == 1
    assert 0 < waits[0] <= groq_round_robin._MINUTE_COOLDOWN_DEFAULT


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


# ==========================================================================
# Chốt điểm bắt đầu MỘT lần mỗi lượt gọi: đúng n client KHÁC NHAU, an toàn đa luồng
# ==========================================================================

_TPD = "Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199990"


class _ModeClients:
    """6 `ChatOpenAI` thật; client `i` báo TPD nếu `i` thuộc `daily`, ngược lại trả ok.

    Bộ đếm được bảo vệ bằng lock riêng (không dựa vào GIL) để so với `call_counts`.
    """

    def __init__(
        self, daily: set[int], *, count: int = 6, minute: set[int] | None = None
    ) -> None:
        self.daily = set(daily)
        self.minute = set(minute or ())
        self.lock = threading.Lock()
        self.calls = [0] * count
        self.clients = [
            ChatOpenAI(api_key=f"key-{index}", model="openai/gpt-oss-120b")
            for index in range(count)
        ]
        for index, client in enumerate(self.clients):
            client._generate = self._make(index)  # type: ignore[method-assign]

    def _make(self, index: int) -> Any:
        def _generate(_messages: object, **_kwargs: object) -> ChatResult:
            with self.lock:
                self.calls[index] += 1
            time.sleep(0.0003)  # nhả GIL để các luồng xen kẽ thật
            if index in self.daily:
                raise _status_error(RateLimitError, 429, _TPD)
            if index in self.minute:
                raise _status_error(RateLimitError, 429, _PER_MINUTE)
            return _ok()

        return _generate

    def router(self) -> GroqRoundRobinChatModel:
        return GroqRoundRobinChatModel(clients=self.clients)

    def total_calls(self) -> int:
        with self.lock:
            return sum(self.calls)


def _run_threads(
    router: GroqRoundRobinChatModel, *, threads: int, rounds: int
) -> list[BaseException]:
    """`threads` luồng cùng xuất phát (barrier) gọi `_generate` `rounds` lần; trả các lỗi."""
    barrier = threading.Barrier(threads)

    def _worker() -> list[BaseException]:
        barrier.wait()
        errors: list[BaseException] = []
        for _ in range(rounds):
            try:
                router._generate(messages=[])
            except BaseException as error:  # noqa: BLE001 -- test gom mọi lỗi để so sánh
                errors.append(error)
        return errors

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(_worker) for _ in range(threads)]
        return [error for future in futures for error in future.result()]


def test_lan_goi_xen_ke_khong_lam_outer_thay_lap_client_de_khong_bat_breaker_oan():
    # Tái hiện tất định lỗi cũ: một "luồng khác" (lượt gọi lồng) lấy đúng vị trí kế tiếp
    # của vòng ngay sau mỗi lần client báo TPD; cách cũ (next() mỗi lần thử) làm lượt
    # ngoài chỉ thấy 0,2,4,0,2,4 (3/6 tài khoản) rồi kết luận oan là cạn cả 6.
    scripted = _ModeClients({0, 2, 4})
    router = scripted.router()
    depth = [0]

    def _wrap(index: int, original: Any) -> Any:
        def _generate(messages: object, **kwargs: object) -> ChatResult:
            try:
                return original(messages, **kwargs)
            finally:
                if depth[0] == 0 and index in scripted.daily:
                    depth[0] += 1
                    try:
                        router._generate(messages=[])
                    finally:
                        depth[0] -= 1

        return _generate

    for index, client in enumerate(scripted.clients):
        client._generate = _wrap(index, client._generate)  # type: ignore[method-assign]

    result = router._generate(messages=[])

    assert result.generations[0].text == "ok"


@pytest.mark.parametrize(
    "daily",
    [{0, 2, 4}, {0, 1, 2}, {1, 3, 5}, {0, 1, 2, 3, 4}, {3, 4, 5}, {0, 1, 3, 4}],
)
def test_nhieu_luong_khi_mot_nua_tai_khoan_het_quota_ngay_khong_bao_gio_bat_breaker(
    daily: set[int],
):
    scripted = _ModeClients(daily)
    router = scripted.router()

    errors = _run_threads(router, threads=8, rounds=25)

    assert errors == []  # còn tài khoản khoẻ thì MỌI lượt gọi phải thành công
    assert router._daily_quota_message is None
    assert all(scripted.calls[index] > 0 for index in range(6) if index not in daily)
    assert sum(router.call_counts) == scripted.total_calls()  # không mất bộ đếm


def test_nhieu_luong_qua_duong_async_ragas_dung_cung_khong_bat_breaker_oan():
    scripted = _ModeClients({0, 2, 4})
    router = scripted.router()

    async def _many() -> list[ChatResult]:
        return await asyncio.gather(
            *[router._agenerate(messages=[]) for _ in range(60)]
        )

    results = asyncio.run(_many())

    assert len(results) == 60
    assert router._daily_quota_message is None
    assert sum(router.call_counts) == scripted.total_calls()


def test_nhieu_luong_khi_ca_6_tai_khoan_het_quota_ngay_breaker_bat_mot_lan_roi_khong_request():
    scripted = _ModeClients({0, 1, 2, 3, 4, 5})
    router = scripted.router()
    threads, rounds = 8, 5

    errors = _run_threads(router, threads=threads, rounds=rounds)

    assert len(errors) == threads * rounds
    assert all(isinstance(error, DailyQuotaExhaustedError) for error in errors)
    assert len({id(error) for error in errors}) == len(
        errors
    )  # mỗi lần một instance mới
    # Chỉ các lượt đang bay lúc bật mới tốn request (tối đa 6 mỗi luồng); sau đó là 0.
    assert 6 <= scripted.total_calls() <= threads * 6
    assert router._daily_quota_message is not None
    before = scripted.total_calls()

    for _ in range(5):
        with pytest.raises(DailyQuotaExhaustedError):
            router._generate(messages=[])

    assert scripted.total_calls() == before


def test_moi_lan_raise_la_mot_instance_moi_chi_lan_dau_giu_cause():
    scripted = _ModeClients({0, 1, 2}, count=3)
    router = scripted.router()

    with pytest.raises(DailyQuotaExhaustedError) as first:
        router._generate(messages=[])
    with pytest.raises(DailyQuotaExhaustedError) as second:
        router._generate(messages=[])

    assert first.value is not second.value
    assert isinstance(first.value.__cause__, RateLimitError)
    assert str(first.value) == str(second.value)
    assert "3 tài khoản" in str(second.value)


def test_moi_luot_goi_thu_dung_n_client_khac_nhau_kho_le_thuoc_vao_diem_bat_dau():
    scripted = _ModeClients(
        {0, 1, 2, 3, 4}, minute={5}
    )  # cả 6 đều 429, không phải cả ngày
    router = scripted.router()

    for round_index in range(1, 13):
        with pytest.raises(RateLimitError):
            router._generate(messages=[])
        assert all(
            count == round_index for count in scripted.calls
        )  # mỗi vòng đúng 6 client


# ==========================================================================
# Cooldown tài khoản vừa báo TPD (ưu tiên tài khoản khác, không cấm tuyệt đối)
# ==========================================================================


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(groq_round_robin, "monotonic", fake)
    return fake


def test_tai_khoan_vua_bao_tpd_khong_bi_goi_lai_trong_cooldown_roi_duoc_thu_lai(
    clock: _Clock,
):
    scripted = _ModeClients({0})
    router = scripted.router()

    for _ in range(12):
        router._generate(messages=[])
    assert scripted.calls[0] == 1  # chỉ lần đầu tốn request 429, sau đó xếp cuối

    clock.now += groq_round_robin._DAILY_COOLDOWN_SECONDS + 1
    for _ in range(6):
        router._generate(messages=[])

    assert scripted.calls[0] == 2  # hết cooldown: lại được thử đúng khi đến lượt


def test_cooldown_chi_la_thu_tu_uu_tien_client_dang_cooldown_van_la_phuong_an_cuoi(
    clock: _Clock,
):
    scripted = _ModeClients({0}, minute={1, 2, 3, 4, 5})
    router = scripted.router()
    with pytest.raises(RateLimitError):
        router._generate(messages=[])  # 0 cooldown, 5 tài khoản còn lại chỉ tạm 429
    scripted.daily = set()  # tài khoản 0 hồi lại (TPD là cửa sổ trượt)

    result = router._generate(messages=[])  # thử 1..5 (fail) rồi 0 (cooldown) cuối cùng

    assert result.generations[0].text == "ok"
    assert router._cooldown_until[0] == 0.0  # thành công thì xoá cooldown


def test_cooldown_khong_du_de_bat_breaker_can_bang_chung_moi_cua_ca_n_tai_khoan(
    clock: _Clock,
):
    scripted = _ModeClients({0, 1, 2, 3, 4})
    router = scripted.router()
    for _ in range(4):
        router._generate(messages=[])  # 5 tài khoản cooldown, tài khoản 5 gánh hết
    assert router._daily_quota_message is None
    scripted.daily = {0, 1, 2, 3, 4, 5}
    before = list(scripted.calls)

    with pytest.raises(DailyQuotaExhaustedError):
        router._generate(messages=[])

    assert all(now == old + 1 for now, old in zip(scripted.calls, before, strict=True))


# ==========================================================================
# Che mã tổ chức / key khi ghi `last_failure`; cắt 200 ký tự SAU khi che
# ==========================================================================

_GROQ_TPD_MESSAGE = (
    "Rate limit reached for model `openai/gpt-oss-120b` in organization "
    "`org_01k9abcdefghijklmnopqrstuv` service tier `on_demand` on tokens per day "
    "(TPD): Limit 200000, Used 199868, Requested 1900. Please try again in 1m1.5s. "
    "Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing"
)


def _groq_error(cls: type[Any], status: int, message: str) -> Any:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"message": message, "type": "tokens", "code": "rate_limit_exceeded"}
    return cls(
        f"Error code: {status} - {{'error': {body}}}",
        response=httpx.Response(status, request=request),
        body=body,
    )


def test_describe_error_che_ma_to_chuc_va_giu_phan_per_day_tpd_limit_used():
    error = _groq_error(RateLimitError, 429, _GROQ_TPD_MESSAGE)

    description = _describe_error(error)

    assert description.startswith(
        "RateLimitError (HTTP 429): Rate limit reached for model"
    )
    assert "org_01k9" not in description
    assert "org_***" in description
    assert "per day (TPD): Limit 200000, Used 199868" in description
    assert "Error code" not in description  # đã bỏ tiền tố của SDK
    assert len(description) <= len("RateLimitError (HTTP 429): ") + 200


def test_describe_error_khong_co_body_van_bo_tien_to_sdk_va_che_ma():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    error = RateLimitError(
        f"Error code: 429 - {_GROQ_TPD_MESSAGE}",
        response=httpx.Response(429, request=request),
        body=None,
    )

    description = _describe_error(error)

    assert "org_01k9" not in description
    assert "Error code" not in description
    assert "per day (TPD): Limit 200000" in description


def test_describe_error_che_moi_chuoi_giong_key_gsk():
    error = _groq_error(
        AuthenticationError, 401, "Invalid API key gsk_AbC123xyzSECRET provided"
    )

    description = _describe_error(error)

    assert "gsk_AbC123" not in description
    assert "SECRET" not in description
    assert "gsk_***" in description


def test_describe_error_cat_200_ky_tu_sau_khi_che_khong_mat_phan_can_thiet():
    long_prefix = "x" * 150
    error = _groq_error(
        RateLimitError, 429, f"{long_prefix} org_{'a' * 80} on tokens per day (TPD)"
    )

    description = _describe_error(error)

    assert "org_a" not in description
    assert "org_***" in description
    assert len(description) <= len("RateLimitError (HTTP 429): ") + 200


def test_last_failure_khi_het_quota_ngay_giu_duoc_limit_used_va_khong_lo_ma_to_chuc():
    errors = [lambda: _groq_error(RateLimitError, 429, _GROQ_TPD_MESSAGE)] * 6
    scripted = _ScriptedClients(errors)
    router = scripted.router()

    with pytest.raises(DailyQuotaExhaustedError) as excinfo:
        router._generate(messages=[])
    description = _describe_error(excinfo.value)

    assert description.startswith(
        "DailyQuotaExhaustedError (HTTP 429): Hết quota ngày cả 6"
    )
    assert "org_01k9" not in description
    assert "per day (TPD): Limit 200000, Used 199868" in description


# ==========================================================================
# Bổ sung vòng A lần 4: khe hở còn lại của lỗi đồng thời và thứ tự cooldown
# ==========================================================================


def test_nhieu_luong_5_tai_khoan_het_ngay_con_1_bao_theo_phut_khong_bao_gio_bat_breaker():
    # Lỗi cũ: mỗi luồng có thể chỉ thấy các client đã hết ngày (bỏ sót client báo theo
    # phút) rồi kết luận oan. Mỗi lượt gọi mới luôn thử đủ 6 nên luôn thấy client 5.
    scripted = _ModeClients({0, 1, 2, 3, 4}, minute={5})
    router = scripted.router()
    threads, rounds = 8, 10

    errors = _run_threads(router, threads=threads, rounds=rounds)

    assert len(errors) == threads * rounds
    assert all(type(error) is RateLimitError for error in errors)
    assert router._daily_quota_message is None
    assert scripted.calls == [threads * rounds] * 6


def test_nhieu_luong_bo_dem_tung_client_khop_dung_so_request_thuc_te():
    scripted = _ModeClients({0, 2, 4})
    router = scripted.router()

    errors = _run_threads(router, threads=8, rounds=25)

    assert errors == []
    assert router.call_counts == scripted.calls  # từng client, không chỉ tổng


def _plan_orders(router: GroqRoundRobinChatModel, calls: int) -> list[list[int]]:
    return [router._plan_attempts() for _ in range(calls)]


def test_moi_client_dang_cooldown_thu_tu_thu_giu_nguyen_vong_quay_khong_bi_xao_tron(
    clock: _Clock,
):
    router = _ModeClients(set()).router()
    for index in range(6):
        router._mark_result(index, daily_limited=True)

    orders = _plan_orders(router, 3)

    # Sắp xếp ổn định: khi mọi client cùng cooldown, thứ tự chính là vòng quay thuần.
    assert orders == [
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 0],
        [2, 3, 4, 5, 0, 1],
    ]


def test_mot_so_client_cooldown_bi_day_xuong_cuoi_va_giu_thu_tu_vong_quay_trong_moi_nhom(
    clock: _Clock,
):
    router = _ModeClients(set()).router()
    router._mark_result(1, daily_limited=True)
    router._mark_result(4, daily_limited=True)

    orders = _plan_orders(router, 2)

    assert orders == [[0, 2, 3, 5, 1, 4], [2, 3, 5, 0, 1, 4]]
    assert all(sorted(order) == list(range(6)) for order in orders)  # luôn đủ 6 client


def test_het_cooldown_thi_client_tro_lai_dung_cho_trong_vong_quay(clock: _Clock):
    router = _ModeClients(set()).router()
    router._mark_result(1, daily_limited=True)
    router._plan_attempts()
    router._plan_attempts()

    clock.now += groq_round_robin._DAILY_COOLDOWN_SECONDS + 1

    assert router._plan_attempts() == [2, 3, 4, 5, 0, 1]
    assert router._plan_attempts() == [3, 4, 5, 0, 1, 2]


def test_ca_6_client_dang_cooldown_nhung_da_hoi_lai_thi_van_duoc_thu_va_thanh_cong(
    clock: _Clock,
):
    scripted = _ModeClients(set())  # cả 6 đã "khoẻ" trở lại
    router = scripted.router()
    for index in range(6):
        router._mark_result(index, daily_limited=True)

    result = router._generate(messages=[])

    assert result.generations[0].text == "ok"
    # thử ngay client đến lượt, không bỏ cuộc
    assert scripted.calls == [1, 0, 0, 0, 0, 0]
    assert router._daily_quota_message is None
    # chỉ client vừa thành công được xoá cooldown
    assert router._cooldown_until[0] == 0.0
    assert all(router._cooldown_until[index] > clock.now for index in range(1, 6))


def test_ca_6_client_dang_cooldown_cung_can_bang_chung_moi_moi_bat_breaker(
    clock: _Clock,
):
    scripted = _ModeClients({0, 1, 2, 3, 4, 5})
    router = scripted.router()
    for index in range(6):
        router._mark_result(index, daily_limited=True)

    with pytest.raises(DailyQuotaExhaustedError):
        router._generate(messages=[])

    # đã gọi đủ 6 để lấy bằng chứng, không suy diễn từ cooldown
    assert scripted.calls == [1] * 6
