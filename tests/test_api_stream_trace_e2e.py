"""E2E `POST /v1/chat/completions` stream=true: một trace Langfuse duy nhất mỗi lượt.

Đi qua toàn bộ đường thật `authenticate -> chat_completions -> stream_chat_turn ->
sse_stream -> StreamingResponse` với orchestrator giả và Langfuse in-memory
(`in_memory_langfuse`, không gửi gì qua mạng), đối chiếu observability_spec.md mục
4.4/4.5. Bug được bảo vệ: nếu mỗi event của generator chạy trong một Task riêng thì
root span `chat_turn` chỉ active ở event đầu -> trace vỡ nhiều mảnh, span con mất
parent/user/session, root mất tags và OTel log "Failed to detach context".

Ngắt kết nối ở tầng HTTP dựng bằng ASGI thủ công (`TestClient` đọc trọn body nên
không mô phỏng được client đóng socket giữa chừng): `receive()` trả `http.disconnect`
khi test cho phép, đúng cơ chế Starlette dùng để huỷ `stream_response`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import production_legal_qa_rag.api.app as app_module
from production_legal_qa_rag.api import routes
from production_legal_qa_rag.api.routes import router
from production_legal_qa_rag.config import ApiSettings
from production_legal_qa_rag.conversation.models import (
    ChatMessage,
    RequestContext,
    TurnTrace,
)
from production_legal_qa_rag.generation.models import (
    DoneEvent,
    GenerationEvent,
    StatusEvent,
    TokenEvent,
    Usage,
)
from production_legal_qa_rag.observability import tracing
from production_legal_qa_rag.observability.turn_trace import RuntimeVersions

_API_KEY = "test-chatbot-api-key"
_USER_ID = "owui-user-7"
_CHAT_ID = "owui-chat-9"
_VERSIONS = RuntimeVersions(prompt_version="v1", corpus_version="c1", model_name="m1")
_QUERY = "Điều kiện thành lập doanh nghiệp?"
_HEADERS = {
    "Authorization": f"Bearer {_API_KEY}",
    "X-OpenWebUI-User-Id": _USER_ID,
    "X-OpenWebUI-Chat-Id": _CHAT_ID,
}
_PAYLOAD = {
    "model": "legal-qa",
    "messages": [{"role": "user", "content": _QUERY}],
    "stream": True,
}
_WAIT_SECONDS = 3.0


class _FakeRedis:
    """Redis giả tối thiểu cho rate limit (INCR/EXPIRE)."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def incr(self, key: str) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key: str, ttl: int) -> None:
        return None


class _Probe:
    """Ghi lại root span (qua `update_turn_trace`) và span con của orchestrator giả."""

    def __init__(self) -> None:
        self.roots: list[Any] = []
        self.children: list[Any] = []
        self.updated = threading.Event()
        self.first_event_sent = threading.Event()
        self.generator_cancelled = threading.Event()


class _TracedOrchestrator:
    """Orchestrator giả tạo span con giữa các event như orchestrator thật.

    Args:
        probe: Nơi ghi span con.
        pause_seconds: Khoảng chờ giữa hai event (mô phỏng retrieve/generate chậm).
        block_after_first_event: Treo sau event đầu tới khi bị huỷ (mô phỏng client
            ngắt kết nối giữa lượt).
        cache_status: Giá trị điền vào `trace.cache_status` cuối lượt.
    """

    def __init__(
        self,
        probe: _Probe,
        *,
        pause_seconds: float = 0.0,
        block_after_first_event: bool = False,
        cache_status: str = "miss",
    ) -> None:
        self._probe = probe
        self._pause_seconds = pause_seconds
        self._block = block_after_first_event
        self._cache_status = cache_status
        self.started = asyncio.Event() if block_after_first_event else None

    def stream(
        self, messages: list[ChatMessage], ctx: RequestContext, trace: TurnTrace
    ) -> AsyncIterator[GenerationEvent]:
        async def _gen() -> AsyncIterator[GenerationEvent]:
            try:
                yield StatusEvent(stage="guardrail")
                with tracing.span("guardrail_child") as child:
                    self._probe.children.append(child)
                if self._block:
                    assert self.started is not None
                    self.started.set()
                    await asyncio.sleep(3600)
                yield TokenEvent(text="Xin chào")
                await asyncio.sleep(self._pause_seconds)
                with tracing.span("generate_child") as child:
                    self._probe.children.append(child)
                yield TokenEvent(text=", đây là câu trả lời.")
                trace.outcome = "answered"
                trace.cache_status = self._cache_status  # type: ignore[assignment]
                trace.answer_text = "Xin chào, đây là câu trả lời."
                yield DoneEvent(usage=Usage(prompt_tokens=10, completion_tokens=5))
            except asyncio.CancelledError:
                self._probe.generator_cancelled.set()
                raise

        return _gen()


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> _Probe:
    """Bọc `update_turn_trace` thật để giữ root span và báo hiệu khi đã cập nhật."""
    recorded = _Probe()
    real_update = routes.update_turn_trace

    def capturing_update(root_span: Any, trace: TurnTrace, **kwargs: Any) -> None:
        recorded.roots.append(root_span)
        real_update(root_span, trace, **kwargs)
        recorded.updated.set()

    monkeypatch.setattr(routes, "update_turn_trace", capturing_update)
    return recorded


def _build_app(
    orchestrator: _TracedOrchestrator, *, keepalive_seconds: float
) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app_module._register_exception_handlers(app)
    app.state.api_settings = ApiSettings(
        CHATBOT_API_KEY=_API_KEY,
        keepalive_seconds=keepalive_seconds,
        _env_file=None,
    )
    app.state.redis = _FakeRedis()
    app.state.orchestrator = orchestrator
    app.state.runtime_versions = _VERSIONS
    return app


def _assert_single_healthy_trace(
    probe: _Probe,
    attributes_of: Callable[[Any], dict[str, Any]],
    parent_id_of: Callable[[Any], str | None],
    caplog: pytest.LogCaptureFixture,
    *,
    expected_tags: list[str],
) -> None:
    assert len(probe.roots) == 1
    root = probe.roots[0]
    assert parent_id_of(root) is None
    assert len(probe.children) == 2
    for child in probe.children:
        assert child.trace_id == root.trace_id
        assert parent_id_of(child) == root.id
    attributes = attributes_of(root)
    assert attributes["user.id"] == _USER_ID
    assert attributes["session.id"] == _CHAT_ID
    assert list(attributes["langfuse.trace.tags"]) == expected_tags
    assert not [r for r in caplog.records if "detach" in r.getMessage().lower()]


@pytest.mark.parametrize("cache_status", ["miss", "answer_hit"])
def test_stream_true_qua_http_chi_tao_mot_trace_day_du(
    in_memory_langfuse: Any,
    probe: _Probe,
    caplog: pytest.LogCaptureFixture,
    observation_attributes: Callable[[Any], dict[str, Any]],
    observation_parent_id: Callable[[Any], str | None],
    cache_status: str,
) -> None:
    orchestrator = _TracedOrchestrator(probe, cache_status=cache_status)
    client = TestClient(_build_app(orchestrator, keepalive_seconds=15))

    with caplog.at_level(logging.WARNING):
        response = client.post("/v1/chat/completions", json=_PAYLOAD, headers=_HEADERS)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "Xin chào" in response.text
    assert response.text.endswith("data: [DONE]\n\n")
    _assert_single_healthy_trace(
        probe,
        observation_attributes,
        observation_parent_id,
        caplog,
        expected_tags=["outcome:answered", f"cache:{cache_status}"],
    )


def test_stream_true_co_khoang_cho_dai_hon_keepalive_van_mot_trace(
    in_memory_langfuse: Any,
    probe: _Probe,
    caplog: pytest.LogCaptureFixture,
    observation_attributes: Callable[[Any], dict[str, Any]],
    observation_parent_id: Callable[[Any], str | None],
) -> None:
    orchestrator = _TracedOrchestrator(probe, pause_seconds=0.3)
    client = TestClient(_build_app(orchestrator, keepalive_seconds=0.05))

    with caplog.at_level(logging.WARNING):
        response = client.post("/v1/chat/completions", json=_PAYLOAD, headers=_HEADERS)

    assert response.status_code == 200
    assert ": keep-alive\n\n" in response.text
    # Keep-alive không được làm mất event thật đang chờ.
    assert "đây là câu trả lời." in response.text
    assert response.text.endswith("data: [DONE]\n\n")
    _assert_single_healthy_trace(
        probe,
        observation_attributes,
        observation_parent_id,
        caplog,
        expected_tags=["outcome:answered", "cache:miss"],
    )


async def _post_then_disconnect(
    app: FastAPI, orchestrator: _TracedOrchestrator, probe: _Probe
) -> list[dict[str, Any]]:
    """Gọi app qua ASGI thô, ngắt kết nối sau khi chunk SSE đầu đã được gửi đi."""
    body = json.dumps(_PAYLOAD).encode()
    request_consumed = False
    disconnect = asyncio.Event()
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal request_consumed
        if not request_consumed:
            request_consumed = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            probe.first_event_sent.set()

    headers = [(k.lower().encode(), v.encode()) for k, v in _HEADERS.items()]
    headers += [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ]
    scope = {
        "type": "http",
        # spec 2.3: Starlette dùng `listen_for_disconnect(receive)` thay vì OSError khi send.
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    task = asyncio.ensure_future(app(scope, receive, send))
    assert orchestrator.started is not None
    await asyncio.wait_for(orchestrator.started.wait(), _WAIT_SECONDS)
    assert await asyncio.to_thread(probe.first_event_sent.wait, _WAIT_SECONDS)
    disconnect.set()
    await asyncio.wait_for(task, _WAIT_SECONDS)
    return sent


def test_client_ngat_ket_noi_giua_luot_van_mot_trace_tag_client_disconnected(
    in_memory_langfuse: Any,
    probe: _Probe,
    caplog: pytest.LogCaptureFixture,
    observation_attributes: Callable[[Any], dict[str, Any]],
) -> None:
    orchestrator = _TracedOrchestrator(probe, block_after_first_event=True)
    app = _build_app(orchestrator, keepalive_seconds=15)

    async def scenario() -> list[dict[str, Any]]:
        sent = await _post_then_disconnect(app, orchestrator, probe)
        # `update_turn_trace` chạy trong Task của generator (được huỷ cùng consumer),
        # có thể xong sau khi ASGI call kết thúc: chờ tín hiệu thay vì sleep.
        assert await asyncio.to_thread(probe.updated.wait, _WAIT_SECONDS)
        return sent

    with caplog.at_level(logging.WARNING):
        sent = asyncio.run(scenario())

    bodies = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    assert b"[DONE]" not in bodies
    assert probe.generator_cancelled.is_set()
    assert len(probe.roots) == 1
    root = probe.roots[0]
    assert [child.trace_id for child in probe.children] == [root.trace_id]
    attributes = observation_attributes(root)
    assert attributes["user.id"] == _USER_ID
    assert attributes["session.id"] == _CHAT_ID
    assert list(attributes["langfuse.trace.tags"]) == [
        "outcome:client_disconnected",
        "cache:miss",
    ]
    assert not [r for r in caplog.records if "detach" in r.getMessage().lower()]
