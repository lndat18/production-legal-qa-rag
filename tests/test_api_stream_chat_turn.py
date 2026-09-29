"""Test `stream_chat_turn` (api/routes.py): root trace Langfuse + metrics mỗi lượt.

Đối chiếu observability_spec.md mục 4.4/4.5 và api_spec.md mục 6: mọi nhánh (trả lời,
từ chối, lỗi, cache hit, ngắt kết nối) đều đi qua cùng một `finally` nên
`update_turn_trace` chạy đúng một lần (khi root span còn active) rồi
`metrics.record_turn` chạy đúng một lần. Nhóm đầu dùng spy thay hai hàm đó; nhóm sau
dùng Langfuse client hermetic (`in_memory_langfuse`, không gửi gì qua mạng) để xác nhận
thuộc tính thật trên root span.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from production_legal_qa_rag.api import routes
from production_legal_qa_rag.conversation.models import (
    ChatMessage,
    RequestContext,
    TurnTrace,
)
from production_legal_qa_rag.generation.models import (
    DoneEvent,
    GenerationEvent,
    GuardrailVerdict,
    StatusEvent,
)
from production_legal_qa_rag.observability.turn_trace import RuntimeVersions

_VERSIONS = RuntimeVersions(prompt_version="v1", corpus_version="c1", model_name="m1")
_CONTEXT = RequestContext(user_id="user-1", chat_id="chat-1", request_id="req-1")
_QUERY = "Điều kiện thành lập doanh nghiệp?"
_MESSAGES = [ChatMessage(role="user", content=_QUERY)]
_METADATA_PREFIX = "langfuse.observation.metadata."


def _fill_answered(trace: TurnTrace) -> None:
    trace.outcome = "answered"
    trace.answer_text = "Câu trả lời."


def _fill_refused(trace: TurnTrace) -> None:
    trace.outcome = "refused"
    trace.verdict = GuardrailVerdict(verdict="out_of_scope", reason="ngoài phạm vi")


def _fill_error(trace: TurnTrace) -> None:
    trace.outcome = "error"
    trace.error_code = "llm_error"


def _fill_cache_hit(trace: TurnTrace) -> None:
    trace.outcome = "answered"
    trace.cache_status = "answer_hit"
    trace.answer_text = "Câu trả lời từ cache."


_BRANCHES = [
    pytest.param(_fill_answered, "answered", "miss", id="answered"),
    pytest.param(_fill_refused, "refused", "miss", id="refused"),
    pytest.param(_fill_error, "error", "miss", id="error"),
    pytest.param(_fill_cache_hit, "answered", "answer_hit", id="cache_hit"),
]


class _ScriptedOrchestrator:
    """Phát event cho sẵn, điền `trace` như orchestrator thật, tuỳ chọn ném lỗi."""

    def __init__(
        self,
        events: list[GenerationEvent],
        *,
        fill: Callable[[TurnTrace], None] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._events = events
        self._fill = fill
        self._error = error

    def stream(
        self, messages: list[ChatMessage], ctx: RequestContext, trace: TurnTrace
    ) -> AsyncIterator[GenerationEvent]:
        async def _gen() -> AsyncIterator[GenerationEvent]:
            for event in self._events:
                yield event
            if self._fill is not None:
                self._fill(trace)
            if self._error is not None:
                raise self._error

        return _gen()


class _BlockingOrchestrator:
    """Phát một event rồi treo, để test huỷ task giữa chừng như khi client ngắt."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    def stream(
        self, messages: list[ChatMessage], ctx: RequestContext, trace: TurnTrace
    ) -> AsyncIterator[GenerationEvent]:
        async def _gen() -> AsyncIterator[GenerationEvent]:
            yield StatusEvent(stage="guardrail")
            self.started.set()
            await asyncio.sleep(3600)

        return _gen()


class _TurnSpies:
    """Kết quả ghi lại từ `update_turn_trace` và `metrics.record_turn` (bản sao trace)."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.recorded: list[TurnTrace] = []
        self.order: list[str] = []


@pytest.fixture
def spies(monkeypatch: pytest.MonkeyPatch) -> _TurnSpies:
    spy = _TurnSpies()

    def fake_update(
        root_span: Any,
        trace: TurnTrace,
        *,
        request_id: str,
        versions: RuntimeVersions,
    ) -> None:
        spy.order.append("update")
        spy.updates.append(
            {
                "root_span": root_span,
                "trace": trace.model_copy(deep=True),
                "request_id": request_id,
                "versions": versions,
            }
        )

    def fake_record(trace: TurnTrace) -> None:
        spy.order.append("record")
        spy.recorded.append(trace.model_copy(deep=True))

    monkeypatch.setattr(routes, "update_turn_trace", fake_update)
    monkeypatch.setattr(routes.metrics, "record_turn", fake_record)
    return spy


def _turn(orchestrator: Any) -> AsyncIterator[GenerationEvent]:
    return routes.stream_chat_turn(orchestrator, _MESSAGES, _CONTEXT, _VERSIONS)


async def _drain(stream: AsyncIterator[GenerationEvent]) -> list[GenerationEvent]:
    return [event async for event in stream]


async def _cancel_midstream() -> BaseException | None:
    """Huỷ task đang tiêu thụ luồng khi orchestrator đang chờ, trả về lỗi bắt được."""
    orchestrator = _BlockingOrchestrator()
    task = asyncio.ensure_future(_drain(_turn(orchestrator)))
    await orchestrator.started.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError as error:
        return error
    return None


# ------------------------------------------------- mỗi nhánh đúng 1 lần cập nhật


@pytest.mark.parametrize(("fill", "outcome", "cache_status"), _BRANCHES)
def test_stream_chat_turn_cap_nhat_trace_va_metrics_dung_mot_lan_moi_nhanh(
    spies: _TurnSpies,
    fill: Callable[[TurnTrace], None],
    outcome: str,
    cache_status: str,
) -> None:
    events: list[GenerationEvent] = [StatusEvent(stage="guardrail"), DoneEvent()]
    orchestrator = _ScriptedOrchestrator(events, fill=fill)

    received = asyncio.run(_drain(_turn(orchestrator)))

    assert received == events
    assert len(spies.updates) == 1
    update = spies.updates[0]
    assert update["trace"].outcome == outcome
    assert update["trace"].cache_status == cache_status
    assert update["request_id"] == "req-1"
    assert update["versions"] is _VERSIONS
    assert [trace.outcome for trace in spies.recorded] == [outcome]
    assert [trace.cache_status for trace in spies.recorded] == [cache_status]


def test_stream_chat_turn_cap_nhat_trace_truoc_roi_moi_ghi_metrics(
    spies: _TurnSpies,
) -> None:
    orchestrator = _ScriptedOrchestrator([DoneEvent()], fill=_fill_answered)

    asyncio.run(_drain(_turn(orchestrator)))

    assert spies.order == ["update", "record"]


def test_stream_chat_turn_trace_nhan_du_lieu_orchestrator_dien_truoc_khi_cap_nhat(
    spies: _TurnSpies,
) -> None:
    orchestrator = _ScriptedOrchestrator([DoneEvent()], fill=_fill_refused)

    asyncio.run(_drain(_turn(orchestrator)))

    trace = spies.updates[0]["trace"]
    assert trace.verdict is not None
    assert trace.verdict.verdict == "out_of_scope"


def test_stream_chat_turn_orchestrator_nem_loi_van_cap_nhat_mot_lan_va_lan_truyen(
    spies: _TurnSpies,
) -> None:
    orchestrator = _ScriptedOrchestrator(
        [StatusEvent(stage="guardrail")], error=RuntimeError("boom")
    )

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(_drain(_turn(orchestrator)))

    assert len(spies.updates) == 1
    assert spies.updates[0]["trace"].outcome == "error"
    assert [trace.outcome for trace in spies.recorded] == ["error"]
    assert spies.order == ["update", "record"]


def test_stream_chat_turn_huy_task_giua_chung_thanh_client_disconnected(
    spies: _TurnSpies,
) -> None:
    cancelled = asyncio.run(_cancel_midstream())

    assert isinstance(cancelled, asyncio.CancelledError)
    assert len(spies.updates) == 1
    assert spies.updates[0]["trace"].outcome == "client_disconnected"
    assert [trace.outcome for trace in spies.recorded] == ["client_disconnected"]
    assert spies.order == ["update", "record"]


def test_stream_chat_turn_dong_generator_giua_chung_van_cap_nhat_mot_lan(
    spies: _TurnSpies,
) -> None:
    async def scenario() -> None:
        events: list[GenerationEvent] = [StatusEvent(stage="guardrail"), DoneEvent()]
        stream: Any = _turn(_ScriptedOrchestrator(events, fill=_fill_answered))
        await anext(stream)
        await stream.aclose()

    asyncio.run(scenario())

    assert len(spies.updates) == 1
    assert len(spies.recorded) == 1
    assert spies.order == ["update", "record"]


def test_stream_chat_turn_moi_lan_goi_la_mot_trace_rieng(spies: _TurnSpies) -> None:
    orchestrator = _ScriptedOrchestrator([DoneEvent()], fill=_fill_answered)

    async def scenario() -> None:
        await _drain(_turn(orchestrator))
        await _drain(_turn(orchestrator))

    asyncio.run(scenario())

    assert len(spies.updates) == 2
    assert spies.updates[0]["trace"] is not spies.updates[1]["trace"]
    assert len(spies.recorded) == 2


# --------------------------------------------- thuộc tính thật trên root span


def _capture_root_spans(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Bọc `update_turn_trace` thật để giữ lại observation gốc của từng lượt."""
    roots: list[Any] = []
    real_update = routes.update_turn_trace

    def capturing_update(root_span: Any, trace: TurnTrace, **kwargs: Any) -> None:
        roots.append(root_span)
        real_update(root_span, trace, **kwargs)

    monkeypatch.setattr(routes, "update_turn_trace", capturing_update)
    return roots


def test_stream_chat_turn_root_trace_that_co_input_output_user_session_va_tags(
    in_memory_langfuse: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _capture_root_spans(monkeypatch)
    orchestrator = _ScriptedOrchestrator([DoneEvent()], fill=_fill_answered)

    asyncio.run(_drain(_turn(orchestrator)))

    assert len(roots) == 1
    attributes = dict(roots[0]._otel_span.attributes)
    assert attributes["user.id"] == "user-1"
    assert attributes["session.id"] == "chat-1"
    tags = list(attributes["langfuse.trace.tags"])
    assert tags == ["outcome:answered", "cache:miss"]
    assert attributes["langfuse.observation.input"] == _QUERY
    assert attributes["langfuse.observation.output"] == "Câu trả lời."
    assert attributes[_METADATA_PREFIX + "request_id"] == "req-1"
    assert attributes[_METADATA_PREFIX + "outcome"] == "answered"
    assert attributes[_METADATA_PREFIX + "prompt_version"] == "v1"
    assert attributes[_METADATA_PREFIX + "corpus_version"] == "c1"
    assert attributes[_METADATA_PREFIX + "model_name"] == "m1"


def test_stream_chat_turn_root_trace_that_luot_cache_hit_co_tag_cache(
    in_memory_langfuse: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _capture_root_spans(monkeypatch)
    orchestrator = _ScriptedOrchestrator([DoneEvent()], fill=_fill_cache_hit)

    asyncio.run(_drain(_turn(orchestrator)))

    tags = list(roots[0]._otel_span.attributes["langfuse.trace.tags"])
    assert tags == ["outcome:answered", "cache:answer_hit"]


def test_stream_chat_turn_root_trace_that_luot_bi_ngat_co_tag_client_disconnected(
    in_memory_langfuse: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _capture_root_spans(monkeypatch)

    cancelled = asyncio.run(_cancel_midstream())

    assert isinstance(cancelled, asyncio.CancelledError)
    assert len(roots) == 1
    tags = list(roots[0]._otel_span.attributes["langfuse.trace.tags"])
    assert tags == ["outcome:client_disconnected", "cache:miss"]


def test_stream_chat_turn_root_trace_that_khong_bia_session_khi_thieu_chat_id(
    in_memory_langfuse: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _capture_root_spans(monkeypatch)
    context = RequestContext(user_id="user-2", chat_id=None, request_id="req-2")
    orchestrator = _ScriptedOrchestrator([DoneEvent()], fill=_fill_answered)

    stream = routes.stream_chat_turn(orchestrator, _MESSAGES, context, _VERSIONS)
    asyncio.run(_drain(stream))

    attributes = dict(roots[0]._otel_span.attributes)
    assert attributes["user.id"] == "user-2"
    assert "session.id" not in attributes
