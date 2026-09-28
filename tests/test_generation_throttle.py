"""Test Evidence Judge nối throttle; guardrail/generation không qua throttle.

Theo conversation_spec.md mục 12.1: Judge chờ tối đa bằng `timeout_seconds`, hết hạn
thì `JudgeError` (pipeline fail-closed `unable_to_verify`); guardrail và generation
đã có bucket riêng nên không đi qua `llm_throttle`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from production_legal_qa_rag.config import ThrottleSettings
from production_legal_qa_rag.generation.generator import (
    AnswerGenerator,
    GeneratedAnswer,
)
from production_legal_qa_rag.generation.guardrail import InputGuardrail
from production_legal_qa_rag.generation.judge import EvidenceJudge, JudgeError
from production_legal_qa_rag.generation.models import (
    Citation,
    GuardrailVerdict,
    JudgeVerdict,
)
from production_legal_qa_rag.generation.pipeline import GenerationPipeline
from production_legal_qa_rag.retrieval.llm_throttle import (
    Reservation,
    ThrottleTimeout,
    TokenWindowThrottle,
    _get_bucket_throttle,
    get_throttle,
)
from production_legal_qa_rag.retrieval.models import RetrievedChunk

DRAFT = "Được nghỉ 12 ngày [1]."


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


class _FakeStructuredClient:
    """Fake ChatOpenAI hỗ trợ `with_structured_output(...).ainvoke()`."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.invocations = 0

    def with_structured_output(self, schema: Any, method: str | None = None) -> Any:
        return self

    async def ainvoke(self, messages: list[dict[str, str]]) -> Any:
        self.invocations += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _RecordingThrottle:
    """Throttle giả ghi lại tham số `acquire`."""

    def __init__(self) -> None:
        self.acquired: list[tuple[int, float]] = []

    async def acquire(
        self, estimated_tokens: int, max_wait_seconds: float
    ) -> Reservation:
        self.acquired.append((estimated_tokens, max_wait_seconds))
        return Reservation(reservation_id=len(self.acquired), estimated_tokens=1)


def _judge_settings(timeout_seconds: int = 45) -> SimpleNamespace:
    return SimpleNamespace(
        api_key="judge-key",
        model_name="judge-model",
        max_retries=1,
        timeout_seconds=timeout_seconds,
    )


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        source_document="bo-luat-lao-dong",
        breadcrumb="Điều 1",
        content="Người lao động được nghỉ 12 ngày.",
    )


def _citation() -> Citation:
    return Citation(
        n=1,
        chunk_id="chunk-1",
        source_document="bo-luat-lao-dong",
        breadcrumb="Điều 1",
    )


def _exhausted_throttle(clock: _FakeClock) -> TokenWindowThrottle:
    """Throttle đã dùng hết RPM: lời gọi kế tiếp phải chờ 60s."""
    throttle = TokenWindowThrottle(10**6, 1, 1.0, clock=clock, sleep=clock.sleep)
    asyncio.run(throttle.acquire(1, 1))
    return throttle


def _judge(
    client: _FakeStructuredClient, throttle: Any, timeout_seconds: int = 45
) -> EvidenceJudge:
    return EvidenceJudge(
        _judge_settings(timeout_seconds),
        client=client,
        throttle=throttle,
    )


def _run_judge(judge: EvidenceJudge) -> JudgeVerdict:
    return asyncio.run(judge.judge("Câu hỏi", [_chunk()], DRAFT, [_citation()]))


# ---------------------------------------------------------------- Judge


def test_judge_het_han_cho_throttle_thi_raise_judge_error_va_khong_goi_groq() -> None:
    clock = _FakeClock()
    client = _FakeStructuredClient(JudgeVerdict(verdict="pass"))
    judge = _judge(client, _exhausted_throttle(clock), timeout_seconds=10)

    with pytest.raises(JudgeError) as error_info:
        _run_judge(judge)

    assert isinstance(error_info.value.__cause__, ThrottleTimeout)
    assert client.invocations == 0
    assert clock.sleeps == []


def test_judge_cho_theo_timeout_seconds_khong_theo_nguong_8_giay() -> None:
    clock = _FakeClock()
    client = _FakeStructuredClient(JudgeVerdict(verdict="pass"))
    # Cần chờ 60s: hơn ngưỡng 8s của condense/HyDE nhưng nằm trong timeout 100s.
    judge = _judge(client, _exhausted_throttle(clock), timeout_seconds=100)

    verdict = _run_judge(judge)

    assert verdict == JudgeVerdict(verdict="pass")
    assert clock.sleeps == [60.0]
    assert client.invocations == 1


def test_judge_acquire_max_wait_bang_timeout_seconds_va_uoc_luong_hop_ly() -> None:
    throttle = _RecordingThrottle()
    client = _FakeStructuredClient(JudgeVerdict(verdict="pass"))

    _run_judge(_judge(client, throttle, timeout_seconds=45))

    ((estimated_tokens, max_wait),) = throttle.acquired
    assert max_wait == 45.0
    assert estimated_tokens >= ThrottleSettings().judge_completion_tokens


def test_judge_uoc_luong_lon_hon_khi_context_dai_hon() -> None:
    short_throttle = _RecordingThrottle()
    long_throttle = _RecordingThrottle()
    client = _FakeStructuredClient(JudgeVerdict(verdict="pass"))
    long_chunk = _chunk().model_copy(update={"content": "Nghỉ phép năm. " * 200})

    long_judge = _judge(client, long_throttle)

    _run_judge(_judge(client, short_throttle))
    asyncio.run(long_judge.judge("Câu hỏi", [long_chunk], DRAFT, []))

    assert long_throttle.acquired[0][0] > short_throttle.acquired[0][0]


def test_judge_loi_provider_van_la_judge_error_sau_khi_qua_throttle() -> None:
    throttle = _RecordingThrottle()
    client = _FakeStructuredClient(RuntimeError("429"))

    with pytest.raises(JudgeError):
        _run_judge(_judge(client, throttle))

    assert len(throttle.acquired) == 1


def test_judge_khong_inject_throttle_thi_dung_bucket_chung_theo_model_va_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THROTTLE_RPM_LIMIT", "1")
    client = _FakeStructuredClient(JudgeVerdict(verdict="pass"))
    judge = EvidenceJudge(
        _judge_settings(timeout_seconds=1),
        client=client,
    )

    assert _run_judge(judge) == JudgeVerdict(verdict="pass")

    # Bucket (judge-model, judge-key) đã hết request: lần sau quá hạn 1s -> JudgeError.
    with pytest.raises(JudgeError):
        _run_judge(judge)
    assert client.invocations == 1
    with pytest.raises(ThrottleTimeout):
        asyncio.run(get_throttle("judge-model", "judge-key").acquire(1, 0.05))


# ---------------------------------------------------------------- pipeline fail-closed


class _FakeGenerator:
    def __init__(self) -> None:
        self.repair_calls = 0

    async def draft(self, query: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        return GeneratedAnswer(text=DRAFT, fragments=[DRAFT])

    async def repair(self, *args: Any) -> GeneratedAnswer:
        self.repair_calls += 1
        return GeneratedAnswer(text=DRAFT, fragments=[DRAFT])


def test_pipeline_judge_het_han_throttle_tra_refusal_unable_to_verify() -> None:
    clock = _FakeClock()
    client = _FakeStructuredClient(JudgeVerdict(verdict="pass"))
    generator = _FakeGenerator()
    pipeline = GenerationPipeline(
        guardrail=object(),
        generator=generator,
        judge=_judge(client, _exhausted_throttle(clock), timeout_seconds=10),
    )

    async def collect() -> list[Any]:
        return [event async for event in pipeline.generate("Câu hỏi", [_chunk()])]

    events = asyncio.run(collect())

    assert events[-2].type == "refusal"
    assert events[-2].reason == "unable_to_verify"
    assert events[-1].type == "done"
    assert not [event for event in events if event.type == "token"]
    assert client.invocations == 0
    assert generator.repair_calls == 0


# ---------------------------------------------------------------- guardrail / generation


class _FakeChunk:
    def __init__(self, content: str) -> None:
        self.content = content
        self.response_metadata: dict[str, Any] = {"finish_reason": "stop"}
        self.usage_metadata = None


class _FakeChatModel:
    async def astream(self, messages: list[dict[str, str]]) -> AsyncIterator[Any]:
        yield _FakeChunk(DRAFT)


def test_guardrail_khong_qua_throttle() -> None:
    settings = SimpleNamespace(
        api_key="guardrail-key", model_name="model", max_retries=2, timeout_seconds=30
    )
    client = _FakeStructuredClient(GuardrailVerdict(verdict="allow", reason="ok"))

    guardrail = InputGuardrail(settings, client=client)

    verdict = asyncio.run(guardrail.check_input("Nghỉ phép?"))

    assert verdict.verdict == "allow"
    assert _get_bucket_throttle.cache_info().currsize == 0


def test_generation_khong_qua_throttle() -> None:
    settings = SimpleNamespace(
        api_key="generation-key",
        round_robin_api_key=None,
        model_name="generation-model",
        max_retries=2,
        timeout_seconds=60,
    )
    generator = AnswerGenerator(settings, client=_FakeChatModel())

    answer = asyncio.run(generator.draft("Câu hỏi", [_chunk()]))

    assert answer.text == DRAFT
    assert _get_bucket_throttle.cache_info().currsize == 0


def test_judge_khong_inject_throttle_thi_tao_dung_mot_bucket() -> None:
    client = _FakeStructuredClient(JudgeVerdict(verdict="pass"))
    judge = EvidenceJudge(
        _judge_settings(),
        client=client,
    )

    _run_judge(judge)

    assert _get_bucket_throttle.cache_info().currsize == 1
