"""Test `retrieval/llm_throttle.py` và HyDE nối throttle (conversation_spec.md mục 12.1).

Đồng hồ và hàm chờ được inject (`_FakeClock`) nên không test nào ngủ thật, trừ ca
xếp hàng quá hạn (cần `asyncio.timeout` thật, ~50ms).
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from types import SimpleNamespace
from typing import Any

import pytest

from production_legal_qa_rag.config import HydeSettings, ThrottleSettings
from production_legal_qa_rag.retrieval import llm_throttle
from production_legal_qa_rag.retrieval.hyde import (
    HYDE_SYSTEM_PROMPT,
    HYDE_USER_TEMPLATE,
    HydeGenerator,
)
from production_legal_qa_rag.retrieval.llm_throttle import (
    Reservation,
    ThrottleTimeout,
    TokenWindowThrottle,
    describe_bucket,
    estimate_tokens,
    get_throttle,
    read_total_tokens,
)


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


def _throttle(
    clock: _FakeClock, *, tpm: int = 1000, rpm: int = 100, safety: float = 1.0
) -> TokenWindowThrottle:
    return TokenWindowThrottle(tpm, rpm, safety, clock=clock, sleep=clock.sleep)


# ---------------------------------------------------------------- acquire: đủ ngân sách


def test_acquire_tra_ve_ngay_khi_du_ngan_sach() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock)

    async def run() -> list[Reservation]:
        return [await throttle.acquire(500, 10), await throttle.acquire(500, 10)]

    first, second = asyncio.run(run())

    assert clock.sleeps == []
    assert first.estimated_tokens == 500
    assert second.estimated_tokens == 500
    assert first.reservation_id != second.reservation_id


def test_acquire_uoc_luong_am_duoc_tinh_bang_0() -> None:
    clock = _FakeClock()

    reservation = asyncio.run(_throttle(clock).acquire(-5, 10))

    assert reservation.estimated_tokens == 0


def test_acquire_lon_hon_ngan_sach_tpm_bi_kep_va_khong_cho_vo_han() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, tpm=1000)

    async def run() -> Reservation:
        big = await throttle.acquire(5000, 100)
        await throttle.acquire(1, 100)
        return big

    big = asyncio.run(run())

    assert big.estimated_tokens == 1000
    # Lời gọi đầu chiếm trọn ngân sách -> lời gọi sau chờ đúng 1 cửa sổ, không hơn.
    assert clock.sleeps == [60.0]


# ---------------------------------------------------------------- RPM / TPM


def test_rpm_day_thi_cho_den_khi_request_cu_nhat_het_han() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, rpm=2)

    async def run() -> None:
        await throttle.acquire(1, 100)
        await throttle.acquire(1, 100)
        assert clock.sleeps == []
        await throttle.acquire(1, 100)

    asyncio.run(run())

    assert clock.sleeps == [60.0]
    assert clock.now == 60.0


def test_rpm_cua_so_truot_chi_cho_phan_con_lai_cua_request_cu_nhat() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, rpm=2)

    async def run() -> None:
        await throttle.acquire(1, 100)
        clock.now = 30.0
        await throttle.acquire(1, 100)
        await throttle.acquire(1, 100)

    asyncio.run(run())

    assert clock.sleeps == [30.0]


def test_tpm_day_thi_cho_den_khi_du_cho_trong() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, tpm=1000)

    async def run() -> None:
        await throttle.acquire(600, 100)
        await throttle.acquire(600, 100)

    asyncio.run(run())

    assert clock.sleeps == [60.0]


def test_tpm_chi_cho_toi_moc_giai_phong_du_token_khong_doi_ca_cua_so() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, tpm=1000)

    async def run() -> None:
        await throttle.acquire(600, 100)  # t=0
        clock.now = 10.0
        await throttle.acquire(300, 100)  # t=10, tổng 900
        await throttle.acquire(500, 100)  # cần 400 -> chờ request đầu (t=0) hết hạn

    asyncio.run(run())

    assert clock.sleeps == [50.0]


def test_request_het_han_dung_tai_60_giay_khong_bi_chan_them() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, rpm=1)

    async def run() -> None:
        await throttle.acquire(1, 100)
        clock.now = 60.0
        await throttle.acquire(1, 100)

    asyncio.run(run())

    assert clock.sleeps == []


def test_safety_factor_nhan_vao_ca_tpm_va_rpm() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, tpm=1000, rpm=10, safety=0.5)

    async def within_budget() -> None:
        for _ in range(5):  # rpm budget = 5, tổng token 500 = tpm budget
            await throttle.acquire(100, 100)

    asyncio.run(within_budget())
    assert clock.sleeps == []

    asyncio.run(throttle.acquire(1, 100))
    assert clock.sleeps == [60.0]


# ---------------------------------------------------------------- ThrottleTimeout


def test_acquire_raise_khi_thoi_gian_cho_vuot_max_wait_va_khong_ngu() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, rpm=1)

    async def run() -> None:
        await throttle.acquire(1, 100)
        with pytest.raises(ThrottleTimeout):
            await throttle.acquire(1, 5)

    asyncio.run(run())

    assert clock.sleeps == []


def test_acquire_cho_dung_bang_max_wait_van_duoc_chap_nhan() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, rpm=1)

    async def run() -> None:
        await throttle.acquire(1, 100)
        await throttle.acquire(1, 60)

    asyncio.run(run())

    assert clock.sleeps == [60.0]


def test_acquire_timeout_khong_giu_cho_ngan_sach() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, tpm=1000)

    async def run() -> None:
        await throttle.acquire(800, 100)
        with pytest.raises(ThrottleTimeout):
            await throttle.acquire(800, 5)
        # Nếu lời gọi thất bại đã giữ chỗ 800 thì lời gọi này sẽ phải chờ.
        await throttle.acquire(200, 5)

    asyncio.run(run())

    assert clock.sleeps == []


def test_acquire_xep_hang_qua_han_raise_thay_vi_cho_mai() -> None:
    clock = _FakeClock()

    async def scenario() -> Reservation:
        holder_sleeping = asyncio.Event()
        release = asyncio.Event()

        async def blocking_sleep(seconds: float) -> None:
            clock.now += seconds
            holder_sleeping.set()
            await release.wait()

        throttle = TokenWindowThrottle(1000, 1, 1.0, clock=clock, sleep=blocking_sleep)
        await throttle.acquire(10, 100)
        holder = asyncio.create_task(throttle.acquire(10, 100))
        await holder_sleeping.wait()  # holder giữ lock và đang ngủ chờ cửa sổ

        with pytest.raises(ThrottleTimeout):
            await throttle.acquire(10, 0.05)

        release.set()
        return await holder

    reservation = asyncio.run(scenario())

    assert reservation.estimated_tokens == 10


def test_fifo_loi_goi_den_truoc_duoc_phuc_vu_truoc() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, rpm=1)
    served: list[str] = []

    async def caller(name: str) -> None:
        await throttle.acquire(1, 1000)
        served.append(name)

    async def run() -> None:
        await throttle.acquire(1, 1000)
        tasks = []
        for name in ("A", "B", "C"):
            tasks.append(asyncio.create_task(caller(name)))
            await asyncio.sleep(0)  # task này vào hàng đợi trước task kế tiếp
        await asyncio.gather(*tasks)

    asyncio.run(run())

    assert served == ["A", "B", "C"]
    assert clock.now == 180.0


def test_mot_instance_dung_duoc_qua_nhieu_asyncio_run() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, rpm=100)

    for _ in range(3):
        asyncio.run(throttle.acquire(1, 10))

    assert clock.sleeps == []


# ---------------------------------------------------------------- settle


def test_settle_usage_thap_hon_uoc_luong_tra_lai_ngan_sach() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, tpm=1000)

    async def run() -> None:
        reservation = await throttle.acquire(900, 100)
        throttle.settle(reservation, 100)
        await throttle.acquire(800, 100)

    asyncio.run(run())

    assert clock.sleeps == []


def test_settle_usage_cao_hon_uoc_luong_lam_cua_so_day_hon() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, tpm=1000)

    async def run() -> None:
        reservation = await throttle.acquire(100, 100)
        throttle.settle(reservation, 1000)
        await throttle.acquire(100, 100)

    asyncio.run(run())

    assert clock.sleeps == [60.0]


def test_settle_none_giu_so_uoc_luong() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, tpm=1000)

    async def run() -> None:
        reservation = await throttle.acquire(900, 100)
        throttle.settle(reservation, None)
        await throttle.acquire(800, 100)

    asyncio.run(run())

    assert clock.sleeps == [60.0]


def test_settle_reservation_het_han_khong_raise_va_khong_dung_request_moi() -> None:
    clock = _FakeClock()
    throttle = _throttle(clock, tpm=1000)

    async def run() -> None:
        old = await throttle.acquire(900, 100)
        clock.now = 61.0
        await throttle.acquire(900, 100)  # loại request cũ khỏi cửa sổ
        throttle.settle(old, 5)
        # Request mới vẫn giữ 900 token: 200 nữa phải chờ.
        await throttle.acquire(200, 100)

    asyncio.run(run())

    assert clock.sleeps == [60.0]


# ---------------------------------------------------------------- estimate / usage


def test_estimate_tokens_lam_tron_len_prompt_va_cong_completion() -> None:
    assert estimate_tokens(300, 3.0, 500) == 600
    assert estimate_tokens(301, 3.0, 0) == math.ceil(301 / 3.0) == 101


def test_estimate_tokens_khong_dung_max_completion_tokens() -> None:
    """Spec: KHÔNG lấy 2048 (max_completion_tokens) làm ước lượng completion."""
    assert estimate_tokens(0, 3.0, ThrottleSettings().condense_completion_tokens) < 2048


def test_read_total_tokens_doc_usage_total_tokens() -> None:
    response = SimpleNamespace(usage=SimpleNamespace(total_tokens=123))

    assert read_total_tokens(response) == 123


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(),
        SimpleNamespace(usage=None),
        SimpleNamespace(usage=SimpleNamespace()),
        SimpleNamespace(usage=SimpleNamespace(total_tokens=None)),
        SimpleNamespace(usage=SimpleNamespace(total_tokens="123")),
        SimpleNamespace(usage=SimpleNamespace(total_tokens=1.5)),
    ],
)
def test_read_total_tokens_tra_none_khi_thieu_hoac_sai_kieu(response: Any) -> None:
    assert read_total_tokens(response) is None


# ---------------------------------------------------------------- describe_bucket


def test_describe_bucket_khong_lo_key_that() -> None:
    """observability_spec.md mục 4.3: key_bucket gắn vào Langfuse không được chứa
    key thật, chỉ fingerprint sha256 rút gọn.
    """
    secret = "gsk_secret_value_123456"

    bucket = describe_bucket("openai/gpt-oss-20b", secret)

    assert secret not in bucket
    fingerprint = hashlib.sha256(secret.encode()).hexdigest()[:8]
    assert bucket == f"openai/gpt-oss-20b:{fingerprint}"


def test_describe_bucket_on_dinh_cho_cung_cap_model_key() -> None:
    assert describe_bucket("m", "k") == describe_bucket("m", "k")


def test_describe_bucket_khac_key_khac_bucket() -> None:
    assert describe_bucket("m", "key-a") != describe_bucket("m", "key-b")


def test_describe_bucket_khac_model_khac_bucket() -> None:
    assert describe_bucket("m1", "key-a") != describe_bucket("m2", "key-a")


def test_describe_bucket_dung_chung_cach_bam_voi_get_throttle() -> None:
    """Cùng cách băm dùng cho MỌI bước gọi LLM (kể cả bước không qua throttle
    chung, vd generation/guardrail) — fingerprint phải khớp với bucket của
    ``get_throttle`` cho cùng (model, key).
    """
    model, key = "openai/gpt-oss-20b", "gsk_xyz"

    bucket = describe_bucket(model, key)

    fingerprint = hashlib.sha256(key.encode()).hexdigest()[:8]
    assert bucket == f"{model}:{fingerprint}"
    assert get_throttle(model, key) is get_throttle(model, key)


# ---------------------------------------------------------------- get_throttle


def test_get_throttle_cung_model_cung_key_tra_cung_instance() -> None:
    assert get_throttle("m", "key-a") is get_throttle("m", "key-a")


def test_get_throttle_khac_key_khac_instance() -> None:
    assert get_throttle("m", "key-a") is not get_throttle("m", "key-b")


def test_get_throttle_khac_model_khac_instance() -> None:
    assert get_throttle("m1", "key-a") is not get_throttle("m2", "key-a")


def test_get_throttle_dung_gioi_han_tu_throttle_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THROTTLE_RPM_LIMIT", "2")
    monkeypatch.setenv("THROTTLE_SAFETY_FACTOR", "1")
    throttle = get_throttle("m", "key-a")

    async def run() -> None:
        await throttle.acquire(1, 0.05)
        await throttle.acquire(1, 0.05)
        with pytest.raises(ThrottleTimeout):
            await throttle.acquire(1, 0.05)

    asyncio.run(run())


def test_get_throttle_khoa_bucket_khong_chua_api_key_that(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "gsk_secret_value_123456"
    recorded: list[tuple[str, str]] = []
    sentinel = object()

    def spy(model: str, key_fingerprint: str) -> object:
        recorded.append((model, key_fingerprint))
        return sentinel

    monkeypatch.setattr(llm_throttle, "_get_bucket_throttle", spy)

    assert get_throttle("openai/gpt-oss-20b", secret) is sentinel

    ((model, fingerprint),) = recorded
    assert model == "openai/gpt-oss-20b"
    assert secret not in fingerprint
    assert fingerprint == hashlib.sha256(secret.encode()).hexdigest()[:8]


def test_get_throttle_khong_giu_api_key_that_trong_instance() -> None:
    secret = "gsk_secret_value_123456"

    throttle = get_throttle("m", secret)

    assert secret not in repr(vars(throttle))


# ---------------------------------------------------------------- HydeGenerator


class _FakeGroq:
    def __init__(self, content: str = "đoạn văn", total_tokens: int | None = None):
        self.calls: list[dict[str, Any]] = []
        self._content = content
        self._total_tokens = total_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self._content)
        usage = (
            None
            if self._total_tokens is None
            else SimpleNamespace(total_tokens=self._total_tokens)
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


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


def _exhausted_throttle(clock: _FakeClock) -> TokenWindowThrottle:
    """Throttle đã dùng hết RPM: lời gọi kế tiếp phải chờ 60s."""
    throttle = _throttle(clock, rpm=1)
    asyncio.run(throttle.acquire(1, 1))
    return throttle


def _hyde(fake: _FakeGroq, throttle: Any, **kwargs: Any) -> HydeGenerator:
    settings = HydeSettings(GROQ_API_KEY="k")
    return HydeGenerator(settings, fake, throttle=throttle, **kwargs)


def test_hyde_het_han_cho_throttle_thi_bo_nhanh_a_va_khong_goi_groq() -> None:
    clock = _FakeClock()
    fake = _FakeGroq()
    generator = _hyde(fake, _exhausted_throttle(clock))

    assert asyncio.run(generator.generate("hỏi")) is None
    assert fake.calls == []
    assert clock.sleeps == []


def test_hyde_cho_toi_da_theo_optional_step_max_wait_seconds() -> None:
    clock = _FakeClock()
    fake = _FakeGroq()
    generator = _hyde(
        fake,
        _exhausted_throttle(clock),
        throttle_settings=ThrottleSettings(optional_step_max_wait_seconds=100),
    )

    assert asyncio.run(generator.generate("hỏi")) == "đoạn văn"
    assert clock.sleeps == [60.0]
    assert len(fake.calls) == 1


def test_hyde_acquire_dung_max_wait_va_uoc_luong_theo_prompt() -> None:
    throttle = _RecordingThrottle()
    settings = ThrottleSettings()
    query = "Nghỉ phép mấy ngày?"

    asyncio.run(_hyde(_FakeGroq(), throttle).generate(query))

    expected = estimate_tokens(
        len(HYDE_SYSTEM_PROMPT) + len(HYDE_USER_TEMPLATE.format(query=query)),
        settings.chars_per_token,
        settings.hyde_completion_tokens,
    )
    assert throttle.acquired == [(expected, settings.optional_step_max_wait_seconds)]
    assert settings.optional_step_max_wait_seconds == 8.0


def test_hyde_settle_bang_usage_that_cua_response() -> None:
    throttle = _RecordingThrottle()

    asyncio.run(_hyde(_FakeGroq(total_tokens=321), throttle).generate("hỏi"))

    assert [actual for _, actual in throttle.settled] == [321]


def test_hyde_settle_none_khi_response_khong_co_usage() -> None:
    throttle = _RecordingThrottle()

    asyncio.run(_hyde(_FakeGroq(total_tokens=None), throttle).generate("hỏi"))

    assert [actual for _, actual in throttle.settled] == [None]


def test_hyde_khong_inject_throttle_thi_dung_bucket_chung_theo_model_va_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THROTTLE_RPM_LIMIT", "1")
    generator = HydeGenerator(HydeSettings(GROQ_API_KEY="k"), _FakeGroq())

    assert asyncio.run(generator.generate("hỏi")) == "đoạn văn"

    # HyDE đã tiêu request duy nhất của bucket (model 20b, key "k").
    shared = get_throttle("openai/gpt-oss-20b", "k")
    with pytest.raises(ThrottleTimeout):
        asyncio.run(shared.acquire(1, 0.05))
