"""Integration tests for the HTTP surface of `api/` (api_spec.md mục 2, 4, 5, 7, 12).

Builds a FastAPI app with the real router/exception handlers but fake
Redis/Orchestrator/Engine/Repository in ``app.state`` — the same shape the
developer used manually via ``TestClient`` during implementation, now
codified as an automated regression suite.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Self

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import production_legal_qa_rag.api.app as app_module
from production_legal_qa_rag.api.routes import ChatLogTaskManager, router
from production_legal_qa_rag.chatlog.models import ChatLogMetadata, TurnRecord
from production_legal_qa_rag.config import ApiSettings
from production_legal_qa_rag.conversation.history import (
    DATA_SNAPSHOT_DISCLAIMER,
    SOURCES_FOOTER_MARKER,
)
from production_legal_qa_rag.conversation.models import (
    ChatMessage,
    RequestContext,
    TurnTrace,
)
from production_legal_qa_rag.generation.models import (
    Citation,
    CitationsEvent,
    DoneEvent,
    GenerationEvent,
    StatusEvent,
    TokenEvent,
    Usage,
)

_API_KEY = "test-chatbot-api-key"
_AUTH_HEADERS = {"Authorization": f"Bearer {_API_KEY}"}
_METADATA = ChatLogMetadata(prompt_version="v1", corpus_version="c1", model_name="m1")
_ANSWER_EVENTS: list[GenerationEvent] = [
    StatusEvent(stage="guardrail"),
    StatusEvent(stage="retrieval"),
    TokenEvent(text="Xin chào, "),
    TokenEvent(text="đây là câu trả lời."),
    CitationsEvent(
        citations=[
            Citation(
                n=1,
                chunk_id="c1",
                source_document="Luật Doanh nghiệp 2020",
                breadcrumb="Điều 4",
            )
        ]
    ),
    DoneEvent(usage=Usage(prompt_tokens=10, completion_tokens=5)),
]


class _FakeRedis:
    """Minimal Redis stand-in covering rate limit (INCR/EXPIRE) and readyz (PING)."""

    def __init__(self, *, ping_fails: bool = False) -> None:
        self.counts: dict[str, int] = {}
        self.ping_fails = ping_fails

    async def incr(self, key: str) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key: str, ttl: int) -> None:
        return None

    async def ping(self) -> None:
        if self.ping_fails:
            raise ConnectionError("redis down")


class _FakeConnection:
    def __init__(self, *, fails: bool) -> None:
        self._fails = fails

    async def execute(self, _statement: object) -> None:
        if self._fails:
            raise ConnectionError("postgres down")

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _FakeEngine:
    """Stand-in for the SQLAlchemy ``AsyncEngine`` used by ``/readyz``."""

    def __init__(self, *, fails: bool = False) -> None:
        self._fails = fails

    def connect(self) -> _FakeConnection:
        return _FakeConnection(fails=self._fails)


class _FakeOrchestrator:
    """Replays a fixed event list, recording the messages/context it received."""

    def __init__(self, events: list[GenerationEvent]) -> None:
        self._events = events
        self.received: list[tuple[list[ChatMessage], RequestContext]] = []

    def stream(
        self, messages: list[ChatMessage], ctx: RequestContext, trace: TurnTrace
    ) -> AsyncIterator[GenerationEvent]:
        self.received.append((messages, ctx))

        async def _gen() -> AsyncIterator[GenerationEvent]:
            for event in self._events:
                yield event
            # Real ChatOrchestrator fills ``trace`` as it streams (mục 7); the
            # chatlog-write test only cares that an answered turn is what gets
            # scheduled, so mimic that single field here.
            trace.outcome = "answered"

        return _gen()


class _SpyRepository:
    def __init__(self) -> None:
        self.turns: list[TurnRecord] = []

    async def record(self, turn: TurnRecord) -> None:
        self.turns.append(turn)


def _build_app(
    *,
    orchestrator: _FakeOrchestrator | None = None,
    redis: _FakeRedis | None = None,
    engine: _FakeEngine | None = None,
    repository: _SpyRepository | None = None,
    rate_limit_per_minute: int = 5,
) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app_module._register_exception_handlers(app)
    app.state.api_settings = ApiSettings(
        chatbot_api_key=_API_KEY, rate_limit_per_minute=rate_limit_per_minute
    )
    app.state.redis = redis or _FakeRedis()
    app.state.database_engine = engine or _FakeEngine()
    app.state.orchestrator = orchestrator or _FakeOrchestrator(list(_ANSWER_EVENTS))
    app.state.chatlog_tasks = ChatLogTaskManager(
        repository or _SpyRepository(), _METADATA
    )
    return app


def _chat_payload(**overrides: object) -> dict:
    payload: dict[str, object] = {
        "model": "legal-qa",
        "messages": [{"role": "user", "content": "Điều kiện thành lập doanh nghiệp?"}],
        "stream": False,
    }
    payload.update(overrides)
    return payload


# --- auth -------------------------------------------------------------


def test_chat_completions_without_bearer_returns_401_openai_body() -> None:
    app = _build_app()
    client = TestClient(app)

    response = client.post("/v1/chat/completions", json=_chat_payload())

    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "invalid_api_key"
    assert body["error"]["type"] == "invalid_request_error"
    assert "message" in body["error"]


def test_models_requires_auth_then_returns_single_model() -> None:
    app = _build_app()
    client = TestClient(app)

    unauthenticated = client.get("/v1/models")
    assert unauthenticated.status_code == 401

    authenticated = client.get("/v1/models", headers=_AUTH_HEADERS)
    assert authenticated.status_code == 200
    assert authenticated.json() == {
        "object": "list",
        "data": [
            {"id": "legal-qa", "object": "model", "owned_by": "production-legal-qa-rag"}
        ],
    }


# --- health/readiness ---------------------------------------------------


def test_healthz_is_always_200_without_auth() -> None:
    app = _build_app()
    client = TestClient(app)

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_200_when_redis_and_postgres_are_up() -> None:
    app = _build_app(redis=_FakeRedis(), engine=_FakeEngine())
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_readyz_503_when_redis_ping_fails() -> None:
    app = _build_app(redis=_FakeRedis(ping_fails=True), engine=_FakeEngine())
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_readyz_503_when_postgres_query_fails() -> None:
    app = _build_app(redis=_FakeRedis(), engine=_FakeEngine(fails=True))
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


# --- message validation --------------------------------------------------


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "câu trả lời cũ"}],
        [{"role": "system", "content": "chỉ có system"}],
    ],
)
def test_chat_completions_invalid_messages_returns_422(messages: list[dict]) -> None:
    app = _build_app()
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(messages=messages),
        headers=_AUTH_HEADERS,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_messages"


def test_chat_completions_query_too_long_returns_422() -> None:
    app = _build_app()
    client = TestClient(app)

    long_question = "a" * 1001
    response = client.post(
        "/v1/chat/completions",
        json=_chat_payload(messages=[{"role": "user", "content": long_question}]),
        headers=_AUTH_HEADERS,
    )

    assert response.status_code == 422


# --- rate limit ------------------------------------------------------------


def test_chat_completions_second_request_in_minute_gets_429_with_retry_after() -> None:
    app = _build_app(rate_limit_per_minute=1)
    client = TestClient(app)

    first = client.post(
        "/v1/chat/completions", json=_chat_payload(), headers=_AUTH_HEADERS
    )
    second = client.post(
        "/v1/chat/completions", json=_chat_payload(), headers=_AUTH_HEADERS
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limit_exceeded"
    assert "Retry-After" in second.headers
    assert int(second.headers["Retry-After"]) > 0


def test_chat_completions_rate_limit_is_per_user() -> None:
    app = _build_app(rate_limit_per_minute=1)
    client = TestClient(app)

    first_user = client.post(
        "/v1/chat/completions",
        json=_chat_payload(),
        headers={**_AUTH_HEADERS, "X-OpenWebUI-User-Id": "user-a"},
    )
    second_user = client.post(
        "/v1/chat/completions",
        json=_chat_payload(),
        headers={**_AUTH_HEADERS, "X-OpenWebUI-User-Id": "user-b"},
    )

    assert first_user.status_code == 200
    assert second_user.status_code == 200


# --- happy path: non-stream --------------------------------------------


def test_chat_completions_non_stream_returns_openai_json_with_sources() -> None:
    repository = _SpyRepository()
    app = _build_app(repository=repository)

    # Context-manager form keeps the TestClient's blocking portal (and its event
    # loop) alive after the request returns, so the fire-and-forget chatlog
    # write task (scheduled via ``asyncio.create_task`` in ``stream_chat_turn``)
    # gets a chance to run instead of being cancelled with an ephemeral portal.
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_chat_payload(stream_options={"include_usage": True}),
            headers=_AUTH_HEADERS,
        )

        assert response.status_code == 200
        assert "X-Request-Id" in response.headers
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "legal-qa"
        message = body["choices"][0]["message"]
        assert message["role"] == "assistant"
        assert "Xin chào, đây là câu trả lời." in message["content"]
        assert SOURCES_FOOTER_MARKER in message["content"]
        assert "[1] Điều 4 — Luật Doanh nghiệp 2020" in message["content"]
        assert DATA_SNAPSHOT_DISCLAIMER in message["content"]
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

        client.portal.call(asyncio.sleep, 0.1)
        assert len(repository.turns) == 1
        assert repository.turns[0].outcome == "answered"


def test_chat_completions_non_stream_without_include_usage_omits_usage() -> None:
    app = _build_app()
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions", json=_chat_payload(), headers=_AUTH_HEADERS
    )

    assert response.status_code == 200
    assert "usage" not in response.json() or response.json()["usage"] is None


# --- happy path: stream ---------------------------------------------------


def test_chat_completions_stream_returns_sse_in_expected_order() -> None:
    app = _build_app()
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions", json=_chat_payload(stream=True), headers=_AUTH_HEADERS
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text

    role_index = body.index('"role":"assistant"')
    guardrail_index = body.index("Đang kiểm tra câu hỏi…")
    retrieval_index = body.index("Đang tra cứu văn bản pháp luật…")
    token_index = body.index("Xin chào, ")
    sources_index = body.index("**Nguồn**")
    disclaimer_index = body.index("Dữ liệu pháp luật trong hệ thống")
    finish_index = body.index('"finish_reason":"stop"')
    done_index = body.index("data: [DONE]")

    assert (
        role_index
        < guardrail_index
        < retrieval_index
        < token_index
        < sources_index
        < disclaimer_index
        < finish_index
        < done_index
    )
    assert body.rstrip().endswith("data: [DONE]")
