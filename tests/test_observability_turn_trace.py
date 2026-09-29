"""Test `observability/turn_trace.py` (observability_spec.md mục 4.5).

Ba nhóm: (1) `_turn_metadata`/`_turn_tags` đủ trường theo bảng "trường chatlog cũ ->
vị trí trên Langfuse"; (2) `update_turn_trace` fail-safe và không đưa nội dung câu hỏi/
trả lời vào log ứng dụng; (3) một test bắt thuộc tính OTel thật trên root span bằng
Langfuse client hermetic (`in_memory_langfuse` trong `conftest.py`: khoá giả, exporter
trong bộ nhớ, không gửi gì qua mạng, không đọc `.env`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from pydantic import ValidationError

from production_legal_qa_rag.conversation.models import TurnTrace
from production_legal_qa_rag.generation.models import (
    Citation,
    GuardrailVerdict,
    Usage,
    WarningEvent,
)
from production_legal_qa_rag.observability import tracing, turn_trace
from production_legal_qa_rag.observability.turn_trace import (
    RuntimeVersions,
    _turn_metadata,
    _turn_tags,
    update_turn_trace,
)

_VERSIONS = RuntimeVersions(
    prompt_version="p-v3", corpus_version="c-abc", model_name="m-120b"
)
_CITATION = Citation(
    n=1,
    chunk_id="c1",
    source_document="Luật Doanh nghiệp 2020",
    breadcrumb="Điều 4",
)
_WARNING = WarningEvent(code="truncated", message="Bị cắt", detail="chi tiết")
_USAGE = Usage(prompt_tokens=100, completion_tokens=20, reasoning_tokens=5)
_OUTCOMES = ("answered", "refused", "error", "client_disconnected")
_CACHE_STATUSES = ("answer_hit", "retrieval_hit", "miss", "bypass")
_METADATA_KEYS = {
    "request_id",
    "outcome",
    "error_code",
    "cache_status",
    "verdict",
    "standalone_query",
    "chunk_ids",
    "citations",
    "warnings",
    "usage",
    "time_to_first_token_ms",
    "latency_ms",
    "prompt_version",
    "corpus_version",
    "model_name",
}
_METADATA_PREFIX = "langfuse.observation.metadata."


def _full_trace() -> TurnTrace:
    return TurnTrace(
        raw_query="Điều kiện thành lập doanh nghiệp?",
        standalone_query="Điều kiện thành lập doanh nghiệp tư nhân là gì?",
        verdict=GuardrailVerdict(verdict="allow", reason="trong phạm vi"),
        cache_status="retrieval_hit",
        outcome="answered",
        chunk_ids=["c1", "c2"],
        answer_text="Câu trả lời đầy đủ [1].",
        citations=[_CITATION],
        warnings=[_WARNING],
        usage=_USAGE,
        time_to_first_token_ms=350,
        latency_ms=1234,
    )


class _FakeRootSpan:
    """Thay observation gốc: ghi lại các lần `.update(...)`, có thể ném lỗi."""

    def __init__(self, *, update_error: Exception | None = None) -> None:
        self.update_error = update_error
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        if self.update_error is not None:
            raise self.update_error
        self.updates.append(kwargs)


class _PropagateSpy:
    """Thay `propagate_attributes`: ghi lại kwargs mỗi lần gọi."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @contextmanager
    def __call__(self, **kwargs: Any) -> Iterator[None]:
        self.calls.append(kwargs)
        yield


def _raise_on_call(**_kwargs: Any) -> Any:
    raise ValueError("propagate lỗi khi gọi")


@contextmanager
def _raise_on_enter(**_kwargs: Any) -> Iterator[None]:
    raise ValueError("propagate lỗi khi vào context")
    yield


# ------------------------------------------------------------------ RuntimeVersions


def test_runtime_versions_hop_le_giu_nguyen_gia_tri() -> None:
    versions = RuntimeVersions(prompt_version="p", corpus_version="c", model_name="m")

    assert versions.prompt_version == "p"
    assert versions.corpus_version == "c"
    assert versions.model_name == "m"


@pytest.mark.parametrize("field", ["prompt_version", "corpus_version", "model_name"])
def test_runtime_versions_tu_choi_chuoi_rong(field: str) -> None:
    values = {"prompt_version": "p", "corpus_version": "c", "model_name": "m"}
    values[field] = ""

    with pytest.raises(ValidationError) as error:
        RuntimeVersions(**values)

    assert field in str(error.value)


def test_runtime_versions_tu_choi_thieu_truong() -> None:
    with pytest.raises(ValidationError):
        RuntimeVersions.model_validate({"prompt_version": "p", "corpus_version": "c"})


# ---------------------------------------------------------------- _turn_metadata


def test_turn_metadata_du_truong_theo_bang_muc_4_5() -> None:
    trace = _full_trace()

    metadata = _turn_metadata(trace, "req-1", _VERSIONS)

    assert set(metadata) == _METADATA_KEYS
    assert metadata["request_id"] == "req-1"
    assert metadata["outcome"] == "answered"
    assert metadata["error_code"] is None
    assert metadata["cache_status"] == "retrieval_hit"
    assert metadata["verdict"] == "allow"
    assert metadata["standalone_query"] == trace.standalone_query
    assert metadata["chunk_ids"] == ["c1", "c2"]
    assert metadata["citations"] == [_CITATION.model_dump()]
    assert metadata["warnings"] == [_WARNING.model_dump()]
    assert metadata["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "reasoning_tokens": 5,
    }
    assert metadata["time_to_first_token_ms"] == 350
    assert metadata["latency_ms"] == 1234
    assert metadata["prompt_version"] == "p-v3"
    assert metadata["corpus_version"] == "c-abc"
    assert metadata["model_name"] == "m-120b"


def test_turn_metadata_khong_lap_input_output_vao_metadata() -> None:
    """`raw_query`/`answer_text` đã là input/output của trace, không lặp ở metadata."""
    metadata = _turn_metadata(_full_trace(), "req-1", _VERSIONS)

    assert "raw_query" not in metadata
    assert "answer_text" not in metadata


def test_turn_metadata_serialize_duoc_thanh_json() -> None:
    """SDK serialize list/dict lồng nhau bằng JSON: metadata phải là kiểu JSON thuần."""
    metadata = _turn_metadata(_full_trace(), "req-1", _VERSIONS)

    assert json.loads(json.dumps(metadata)) == metadata


def test_turn_metadata_usage_none_giu_khoa_gia_tri_none() -> None:
    metadata = _turn_metadata(TurnTrace(outcome="refused"), "req-2", _VERSIONS)

    assert "usage" in metadata
    assert metadata["usage"] is None


def test_turn_metadata_verdict_none_mac_dinh_allow() -> None:
    """Lượt cache hit/lỗi sớm không có verdict: nhật ký cũ coi là `allow`."""
    trace = TurnTrace(outcome="answered", cache_status="answer_hit", verdict=None)

    assert _turn_metadata(trace, "req-2", _VERSIONS)["verdict"] == "allow"


@pytest.mark.parametrize("verdict", ["out_of_scope", "injection"])
def test_turn_metadata_verdict_khac_allow_giu_nguyen(verdict: str) -> None:
    trace = TurnTrace(
        outcome="refused",
        verdict=GuardrailVerdict(verdict=verdict, reason="lý do nội bộ"),
    )

    metadata = _turn_metadata(trace, "req-3", _VERSIONS)

    assert metadata["verdict"] == verdict
    assert "lý do nội bộ" not in json.dumps(metadata, ensure_ascii=False)


def test_turn_metadata_time_to_first_token_none_giu_khoa_gia_tri_none() -> None:
    trace = TurnTrace(outcome="error", time_to_first_token_ms=None)

    metadata = _turn_metadata(trace, "req-4", _VERSIONS)

    assert "time_to_first_token_ms" in metadata
    assert metadata["time_to_first_token_ms"] is None


def test_turn_metadata_luot_rong_van_du_truong_voi_gia_tri_mac_dinh() -> None:
    metadata = _turn_metadata(TurnTrace(), "req-5", _VERSIONS)

    assert set(metadata) == _METADATA_KEYS
    assert metadata["outcome"] == "error"
    assert metadata["cache_status"] == "miss"
    assert metadata["standalone_query"] is None
    assert metadata["error_code"] is None
    assert metadata["chunk_ids"] == []
    assert metadata["citations"] == []
    assert metadata["warnings"] == []
    assert metadata["latency_ms"] == 0


# ------------------------------------------------------------------- _turn_tags


@pytest.mark.parametrize("outcome", _OUTCOMES)
@pytest.mark.parametrize("cache_status", _CACHE_STATUSES)
def test_turn_tags_gom_outcome_va_cache(outcome: str, cache_status: str) -> None:
    trace = TurnTrace(outcome=outcome, cache_status=cache_status)

    assert _turn_tags(trace) == [f"outcome:{outcome}", f"cache:{cache_status}"]


# ------------------------------------------------------------- update_turn_trace


def test_update_turn_trace_gan_output_metadata_va_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_span = _FakeRootSpan()
    propagate_spy = _PropagateSpy()
    monkeypatch.setattr(turn_trace, "propagate_attributes", propagate_spy)
    trace = _full_trace()
    expected_metadata = _turn_metadata(trace, "req-1", _VERSIONS)
    expected_tags = ["outcome:answered", "cache:retrieval_hit"]

    update_turn_trace(root_span, trace, request_id="req-1", versions=_VERSIONS)

    assert len(root_span.updates) == 1
    assert root_span.updates[0]["output"] == "Câu trả lời đầy đủ [1]."
    assert root_span.updates[0]["metadata"] == expected_metadata
    assert propagate_spy.calls == [{"tags": expected_tags}]


def test_update_turn_trace_luot_bi_tu_choi_output_rong_khong_phai_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_span = _FakeRootSpan()
    monkeypatch.setattr(turn_trace, "propagate_attributes", _PropagateSpy())

    trace = TurnTrace(outcome="refused")

    update_turn_trace(root_span, trace, request_id="req-1", versions=_VERSIONS)

    assert root_span.updates[0]["output"] == ""


def test_update_turn_trace_nuot_loi_update_va_khong_log_noi_dung(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secret_query = "Tôi là Nguyễn Văn A, số CCCD 012345678901, có được ly hôn?"
    secret_answer = "Bạn có quyền ly hôn theo Điều 51 Luật Hôn nhân và gia đình."
    trace = TurnTrace(
        raw_query=secret_query,
        standalone_query=secret_query,
        answer_text=secret_answer,
        outcome="answered",
    )
    error = RuntimeError(f"lỗi kèm nội dung {secret_query} {secret_answer}")
    root_span = _FakeRootSpan(update_error=error)
    monkeypatch.setattr(turn_trace, "propagate_attributes", _PropagateSpy())
    caplog.set_level(logging.WARNING, logger=turn_trace.__name__)

    update_turn_trace(root_span, trace, request_id="req-99", versions=_VERSIONS)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "req-99" in message
    assert "RuntimeError" in message
    assert warnings[0].exc_info is None
    for secret in (secret_query, secret_answer):
        assert secret not in message
        assert secret not in caplog.text


def test_update_turn_trace_nuot_loi_propagate_attributes_khi_goi(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root_span = _FakeRootSpan()
    trace = _full_trace()
    monkeypatch.setattr(turn_trace, "propagate_attributes", _raise_on_call)
    caplog.set_level(logging.WARNING, logger=turn_trace.__name__)

    update_turn_trace(root_span, trace, request_id="req-7", versions=_VERSIONS)

    assert len(root_span.updates) == 1
    assert "ValueError" in caplog.text
    assert "req-7" in caplog.text


def test_update_turn_trace_nuot_loi_propagate_attributes_khi_vao_context(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root_span = _FakeRootSpan()
    trace = _full_trace()
    monkeypatch.setattr(turn_trace, "propagate_attributes", _raise_on_enter)
    caplog.set_level(logging.WARNING, logger=turn_trace.__name__)

    update_turn_trace(root_span, trace, request_id="req-8", versions=_VERSIONS)

    assert "ValueError" in caplog.text
    assert "req-8" in caplog.text


def test_update_turn_trace_khong_log_khi_thanh_cong(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root_span = _FakeRootSpan()
    trace = _full_trace()
    monkeypatch.setattr(turn_trace, "propagate_attributes", _PropagateSpy())
    caplog.set_level(logging.DEBUG, logger=turn_trace.__name__)

    update_turn_trace(root_span, trace, request_id="req-1", versions=_VERSIONS)

    assert caplog.records == []


# -------------------------------------------------------- thuộc tính OTel thật


def test_update_turn_trace_ghi_dung_thuoc_tinh_len_root_span_that(
    in_memory_langfuse: Any,
    observation_attributes: Callable[[Any], dict[str, Any]],
) -> None:
    trace = _full_trace()

    with tracing.span(
        "chat_turn",
        input=trace.raw_query,
        metadata={"request_id": "req-1"},
        user_id="user-1",
        session_id="chat-1",
    ) as root_span:
        update_turn_trace(root_span, trace, request_id="req-1", versions=_VERSIONS)
        attributes = observation_attributes(root_span)

    assert attributes["user.id"] == "user-1"
    assert attributes["session.id"] == "chat-1"
    assert list(attributes["langfuse.trace.tags"]) == [
        "outcome:answered",
        "cache:retrieval_hit",
    ]
    assert attributes["langfuse.observation.input"] == trace.raw_query
    assert attributes["langfuse.observation.output"] == trace.answer_text
    assert attributes[_METADATA_PREFIX + "request_id"] == "req-1"
    assert attributes[_METADATA_PREFIX + "outcome"] == "answered"
    assert attributes[_METADATA_PREFIX + "cache_status"] == "retrieval_hit"
    assert attributes[_METADATA_PREFIX + "verdict"] == "allow"
    assert attributes[_METADATA_PREFIX + "standalone_query"] == trace.standalone_query
    assert attributes[_METADATA_PREFIX + "prompt_version"] == "p-v3"
    assert attributes[_METADATA_PREFIX + "corpus_version"] == "c-abc"
    assert attributes[_METADATA_PREFIX + "model_name"] == "m-120b"
    assert attributes[_METADATA_PREFIX + "latency_ms"] == 1234
    assert attributes[_METADATA_PREFIX + "time_to_first_token_ms"] == 350
    chunk_ids = json.loads(attributes[_METADATA_PREFIX + "chunk_ids"])
    citations = json.loads(attributes[_METADATA_PREFIX + "citations"])
    warnings = json.loads(attributes[_METADATA_PREFIX + "warnings"])
    usage = json.loads(attributes[_METADATA_PREFIX + "usage"])
    assert chunk_ids == ["c1", "c2"]
    assert citations == [_CITATION.model_dump()]
    assert warnings == [_WARNING.model_dump()]
    assert usage == _USAGE.model_dump()


def test_update_turn_trace_luot_loi_khong_co_output_va_gan_tag_outcome_error(
    in_memory_langfuse: Any,
    observation_attributes: Callable[[Any], dict[str, Any]],
) -> None:
    trace = TurnTrace(outcome="error", error_code="llm_error", cache_status="miss")

    with tracing.span("chat_turn", user_id="user-1") as root_span:
        update_turn_trace(root_span, trace, request_id="req-2", versions=_VERSIONS)
        attributes = observation_attributes(root_span)

    assert list(attributes["langfuse.trace.tags"]) == ["outcome:error", "cache:miss"]
    assert attributes.get("langfuse.observation.output", "") == ""
    assert attributes[_METADATA_PREFIX + "error_code"] == "llm_error"
    assert _METADATA_PREFIX + "usage" not in attributes
    assert _METADATA_PREFIX + "time_to_first_token_ms" not in attributes
