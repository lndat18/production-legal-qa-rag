"""Tests that protect the chatlog schema, privacy, and lifecycle contract."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from sqlalchemy import DateTime, Text

import production_legal_qa_rag.api.routes as api_routes
from production_legal_qa_rag.api.routes import ChatLogTaskManager, stream_chat_turn
from production_legal_qa_rag.cache.models import CacheStatus
from production_legal_qa_rag.chatlog import (
    ChatLogMetadata,
    ChatLogRepository,
    TurnRecord,
    from_trace,
)
from production_legal_qa_rag.chatlog.tables import chat_turns, metadata
from production_legal_qa_rag.conversation.models import (
    ChatMessage,
    RequestContext,
    TurnTrace,
)
from production_legal_qa_rag.generation.models import DoneEvent

_METADATA = ChatLogMetadata(
    prompt_version="v1",
    corpus_version="corpus-2026",
    model_name="model-x",
)


def _context() -> RequestContext:
    """Build an authenticated request context."""
    return RequestContext(user_id="user", chat_id="chat", request_id="request")


def _record(**overrides: object) -> TurnRecord:
    """Build one valid row and permit targeted field replacements."""
    values: dict[str, object] = {
        "request_id": "request",
        "user_id": "user",
        "raw_query": "query",
        "outcome": "answered",
        "verdict": "allow",
        "cache_status": "miss",
        "latency_ms": 1,
        **_METADATA.model_dump(),
    }
    values.update(overrides)
    return TurnRecord(**values)  # type: ignore[arg-type]


def _trace(
    outcome: Literal[
        "answered", "refused", "error", "client_disconnected"
    ] = "answered",
    cache_status: CacheStatus = "miss",
) -> TurnTrace:
    """Build a terminal orchestrator trace."""
    return TurnTrace(
        raw_query="query",
        standalone_query="standalone",
        outcome=outcome,
        cache_status=cache_status,
        error_code="llm_error" if outcome == "error" else None,
        answer_text="answer" if outcome == "answered" else "",
        latency_ms=1,
    )


class TestTurnRecord:
    """Validate the Pydantic contract in chatlog_spec.md section 2."""

    def test_generated_id_and_default_timestamp_are_valid(self) -> None:
        """The application creates a UUID and timezone-aware UTC timestamp."""
        turn = _record()
        assert uuid.UUID(turn.id)
        assert turn.created_at.tzinfo is not None
        assert turn.created_at.utcoffset() is not None

    def test_naive_timestamp_is_rejected(self) -> None:
        """A timestamptz field must not accept an ambiguous local time."""
        with pytest.raises(ValidationError):
            _record(created_at=datetime(2026, 9, 25, 12, 0, 0))  # noqa: DTZ001

    def test_utc_timestamp_is_accepted(self) -> None:
        """An explicit UTC value remains valid."""
        created_at = datetime(2026, 9, 25, 12, 0, 0, tzinfo=UTC)
        assert _record(created_at=created_at).created_at == created_at

    @pytest.mark.parametrize(
        "field_name",
        ["prompt_version", "corpus_version", "model_name"],
    )
    def test_metadata_is_nonempty_on_row_and_runtime_object(
        self, field_name: str
    ) -> None:
        """Rows cannot be divorced from the runtime that answered them."""
        with pytest.raises(ValidationError):
            _record(**{field_name: ""})
        values = _METADATA.model_dump()
        values[field_name] = ""
        with pytest.raises(ValidationError):
            ChatLogMetadata(**values)

    def test_langfuse_trace_id_optional_defaults_to_none(self) -> None:
        """observability_spec.md mục 4.4: NULL khi Langfuse disabled (thiếu key)."""
        assert _record().langfuse_trace_id is None
        assert _record(langfuse_trace_id="trace-abc").langfuse_trace_id == "trace-abc"


class TestTraceMapping:
    """Validate conversion from the conversation trace."""

    def test_metadata_is_supplied_by_lifecycle_and_created_at_is_aware(self) -> None:
        """from_trace receives metadata instead of hidden blank defaults."""
        turn = from_trace(_trace(), _context(), metadata=_METADATA)
        assert turn.prompt_version == "v1"
        assert turn.corpus_version == "corpus-2026"
        assert turn.model_name == "model-x"
        assert turn.created_at.tzinfo is not None

    @pytest.mark.parametrize("outcome", ["refused", "error", "client_disconnected"])
    def test_nonanswer_outcomes_preserve_empty_answer(self, outcome: str) -> None:
        """Every terminal outcome is serializable as one audit row."""
        turn = from_trace(
            _trace(outcome=outcome),  # type: ignore[arg-type]
            _context(),
            metadata=_METADATA,
        )
        assert turn.outcome == outcome
        assert turn.answer_text == ""

    def test_metadata_argument_is_required(self) -> None:
        """A caller cannot accidentally create rows with empty version values."""
        with pytest.raises(TypeError):
            from_trace(_trace(), _context())  # type: ignore[call-arg]

    def test_langfuse_trace_id_is_copied_when_present(self) -> None:
        """observability_spec.md mục 4.4: `chat_turns.langfuse_trace_id` phải khớp
        đúng trace ID mà `tracing.current_trace_id()` đã gán vào `TurnTrace`."""
        trace = _trace().model_copy(update={"langfuse_trace_id": "trace-xyz"})

        turn = from_trace(trace, _context(), metadata=_METADATA)

        assert turn.langfuse_trace_id == "trace-xyz"

    def test_langfuse_trace_id_is_none_when_langfuse_disabled(self) -> None:
        turn = from_trace(_trace(), _context(), metadata=_METADATA)

        assert turn.langfuse_trace_id is None


class TestRepositoryAndSchema:
    """Validate no-throw recording and database definitions."""

    def test_record_swallows_database_failure(self) -> None:
        """Postgres failure never escapes the chat response path."""
        repo = ChatLogRepository(MagicMock())

        async def failed_insert(_: TurnRecord) -> None:
            raise RuntimeError("database unavailable")

        repo._insert = failed_insert  # type: ignore[assignment]
        asyncio.run(repo.record(_record()))

    def test_table_has_required_columns_indexes_and_timestamptz(self) -> None:
        """Core table exactly supports retention and operational analysis."""
        columns = {column.name: column for column in chat_turns.columns}
        assert {
            "id",
            "created_at",
            "request_id",
            "user_id",
            "chat_id",
            "raw_query",
            "standalone_query",
            "outcome",
            "verdict",
            "error_code",
            "cache_status",
            "chunk_ids",
            "answer_text",
            "citations",
            "warnings",
            "usage",
            "time_to_first_token_ms",
            "latency_ms",
            "prompt_version",
            "corpus_version",
            "model_name",
            "langfuse_trace_id",
        } <= set(columns)
        assert columns["id"].primary_key
        assert columns["chat_id"].nullable
        assert columns["usage"].nullable
        assert not columns["raw_query"].nullable
        assert isinstance(columns["created_at"].type, DateTime)
        assert columns["created_at"].type.timezone is True
        # observability_spec.md mục 4.4: NULL khi Langfuse disabled (thiếu key).
        assert columns["langfuse_trace_id"].nullable
        assert isinstance(columns["langfuse_trace_id"].type, Text)
        assert {index.name for index in metadata.tables["chat_turns"].indexes} == {
            "ix_chat_turns_created_at",
            "ix_chat_turns_user_id",
            "ix_chat_turns_chat_id",
        }

    def test_migration_uses_timestamptz(self) -> None:
        """Alembic must create the same timezone-aware timestamp contract."""
        path = (
            Path(__file__).resolve().parents[1]
            / "alembic/versions/0001_create_chat_turns.py"
        )
        module_spec = importlib.util.spec_from_file_location("chatlog_migration", path)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module.op = MagicMock()
        module.upgrade()
        created_at = next(
            column
            for column in module.op.create_table.call_args.args[1:]
            if column.name == "created_at"
        )
        assert isinstance(created_at.type, DateTime)
        assert created_at.type.timezone is True

    def _load_migration_0002(self) -> object:
        path = (
            Path(__file__).resolve().parents[1]
            / "alembic/versions/0002_add_langfuse_trace_id.py"
        )
        module_spec = importlib.util.spec_from_file_location(
            "langfuse_trace_id_migration", path
        )
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module

    def test_migration_0002_chains_after_0001(self) -> None:
        """Không đứt gãy lịch sử alembic: revision mới phải nối đúng head cũ."""
        module = self._load_migration_0002()

        assert module.revision == "0002_add_langfuse_trace_id"
        assert module.down_revision == "0001_create_chat_turns"

    def test_migration_0002_adds_nullable_text_column(self) -> None:
        """Cột mới phải nullable (NULL khi Langfuse disabled, mục 4.4) và kiểu Text."""
        module = self._load_migration_0002()
        module.op = MagicMock()

        module.upgrade()

        table_name, column = module.op.add_column.call_args.args
        assert table_name == "chat_turns"
        assert column.name == "langfuse_trace_id"
        assert column.nullable is True
        assert isinstance(column.type, Text)

    def test_migration_0002_downgrade_drops_column(self) -> None:
        module = self._load_migration_0002()
        module.op = MagicMock()

        module.downgrade()

        module.op.drop_column.assert_called_once_with("chat_turns", "langfuse_trace_id")


class _Repository:
    """Collect submitted rows without using Postgres."""

    def __init__(self) -> None:
        self.turns: list[TurnRecord] = []

    async def record(self, turn: TurnRecord) -> None:
        """Save the submission in memory."""
        self.turns.append(turn)


class _Orchestrator:
    """Complete a trace with a configured terminal state."""

    def __init__(self, outcome: str, cache_status: CacheStatus) -> None:
        self._outcome = outcome
        self._cache_status = cache_status

    def stream(
        self,
        messages: list[ChatMessage],
        _: RequestContext,
        trace: TurnTrace,
    ):
        """Return a one-event async generator."""

        async def events():
            trace.raw_query = messages[-1].content
            trace.standalone_query = trace.raw_query
            trace.outcome = self._outcome  # type: ignore[assignment]
            trace.cache_status = self._cache_status
            trace.answer_text = "answer" if self._outcome == "answered" else ""
            yield DoneEvent()

        return events()


class TestApiLifecycle:
    """Validate API-side scheduling, failure privacy, cancellation, and shutdown."""

    @pytest.mark.parametrize(
        ("outcome", "cache_status"),
        [
            ("answered", "miss"),
            ("refused", "miss"),
            ("error", "miss"),
            ("answered", "answer_hit"),
        ],
    )
    def test_completed_turns_schedule_exactly_one_nonblocking_record(
        self, outcome: str, cache_status: CacheStatus
    ) -> None:
        """Answered, refused, error, and answer-cache hit each log once."""
        repository = _Repository()
        manager = ChatLogTaskManager(repository, _METADATA)

        async def consume() -> None:
            events = stream_chat_turn(
                _Orchestrator(outcome, cache_status),
                [ChatMessage(role="user", content="private query")],
                _context(),
                manager,
            )
            assert [event async for event in events] == [DoneEvent()]
            assert not repository.turns
            await asyncio.sleep(0)
            assert len(repository.turns) == 1
            await manager.drain()

        asyncio.run(consume())
        assert repository.turns[0].outcome == outcome
        assert repository.turns[0].cache_status == cache_status

    def test_cancellation_records_client_disconnected_once(self) -> None:
        """A Starlette-style cancellation is not mislabelled as an answered turn."""
        repository = _Repository()
        manager = ChatLogTaskManager(repository, _METADATA)

        class WaitingOrchestrator:
            def stream(
                self,
                _: list[ChatMessage],
                __: RequestContext,
                trace: TurnTrace,
            ):
                async def events():
                    trace.raw_query = "private cancelled query"
                    yield DoneEvent()
                    await asyncio.Event().wait()

                return events()

        async def cancel() -> None:
            events = stream_chat_turn(
                WaitingOrchestrator(),
                [ChatMessage(role="user", content="private cancelled query")],
                _context(),
                manager,
            )
            await events.__anext__()
            with pytest.raises(asyncio.CancelledError):
                await events.athrow(asyncio.CancelledError)
            await asyncio.sleep(0)
            await manager.drain()

        asyncio.run(cancel())
        assert [turn.outcome for turn in repository.turns] == ["client_disconnected"]

    def test_failed_background_record_warning_has_no_turn_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Even an exception whose text has PII cannot leak to application logs."""
        raw_query = "PRIVATE_QUERY_DO_NOT_LOG"
        answer_text = "PRIVATE_ANSWER_DO_NOT_LOG"

        class FailingRepository:
            async def record(self, _: TurnRecord) -> None:
                raise RuntimeError(f"database error: {raw_query} / {answer_text}")

        async def complete() -> None:
            manager = ChatLogTaskManager(FailingRepository(), _METADATA)
            manager.schedule(
                TurnTrace(raw_query=raw_query, answer_text=answer_text),
                _context(),
            )
            await asyncio.sleep(0)
            await manager.drain()

        caplog.set_level(logging.WARNING, logger="production_legal_qa_rag.api.routes")
        asyncio.run(complete())
        assert "Task ghi chatlog thất bại" in caplog.text
        assert raw_query not in caplog.text
        assert answer_text not in caplog.text

    def test_shutdown_drain_cancels_stuck_write_within_five_seconds(self) -> None:
        """Shutdown never waits forever for a background INSERT."""
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class BlockingRepository:
            async def record(self, _: TurnRecord) -> None:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

        async def shutdown() -> float:
            manager = ChatLogTaskManager(BlockingRepository(), _METADATA)
            manager.schedule(TurnTrace(raw_query="q"), _context())
            await started.wait()
            begun = time.monotonic()
            await manager.drain()
            elapsed = time.monotonic() - begun
            await asyncio.sleep(0)
            return elapsed

        with patch("production_legal_qa_rag.api.routes._SHUTDOWN_DRAIN_SECONDS", 0.01):
            elapsed = asyncio.run(shutdown())
        assert elapsed < 0.5
        assert cancelled.is_set()

    def test_lifespan_injects_engine_repository_and_runtime_metadata_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shared API lifecycle creates runtime metadata only once.

        api_spec.md mục 7: lifespan giờ còn dựng ``ApiSettings``/``RedisSettings``,
        một Redis client và toàn bộ registry cache/admission/orchestrator, lưu vào
        ``app.state``. ``ApiSettings`` không có default cho ``CHATBOT_API_KEY`` nên
        test set env thật thay vì patch, để bảo vệ đúng contract "bắt buộc từ env".
        """
        from fastapi import FastAPI

        import production_legal_qa_rag.api.app as app_module

        monkeypatch.setenv("CHATBOT_API_KEY", "test-chatbot-api-key")

        engine = MagicMock()
        engine.dispose = AsyncMock()
        repository = _Repository()
        database = SimpleNamespace(database_url="postgresql+asyncpg://chatlog")
        cache = SimpleNamespace(corpus_version="configured-corpus")
        generation = SimpleNamespace(model_name="configured-model")
        redis_settings = SimpleNamespace(redis_url="redis://fake-redis")
        fake_redis = MagicMock()
        fake_redis.aclose = AsyncMock()
        fake_orchestrator = MagicMock()
        app = FastAPI()

        async def run_lifespan() -> None:
            with (
                patch.object(app_module, "DatabaseSettings", return_value=database),
                patch.object(app_module, "CacheSettings", return_value=cache),
                patch.object(app_module, "GenerationSettings", return_value=generation),
                patch.object(app_module, "RedisSettings", return_value=redis_settings),
                patch.object(app_module, "Redis") as redis_cls,
                patch.object(app_module, "AnswerCache") as answer_cache_cls,
                patch.object(app_module, "RetrievalCache") as retrieval_cache_cls,
                patch.object(app_module, "SingleFlight") as single_flight_cls,
                patch.object(app_module, "AdmissionController") as admission_cls,
                patch.object(
                    app_module, "ChatOrchestrator", return_value=fake_orchestrator
                ) as orchestrator_cls,
                patch.object(
                    app_module, "create_engine", return_value=engine
                ) as create_engine,
                patch.object(app_module, "ChatLogRepository", return_value=repository),
                patch.object(
                    app_module, "compute_corpus_version", return_value="runtime-corpus"
                ) as compute,
                patch.object(app_module, "PROMPT_VERSION", "runtime-prompt"),
            ):
                redis_cls.from_url.return_value = fake_redis
                async with app_module.lifespan(app):
                    manager = app.state.chatlog_tasks
                    assert app.state.chatlog_repository is repository
                    assert manager._repository is repository
                    assert manager._metadata == ChatLogMetadata(
                        prompt_version="runtime-prompt",
                        corpus_version="runtime-corpus",
                        model_name="configured-model",
                    )
                    create_engine.assert_called_once_with(database.database_url)
                    compute.assert_called_once_with(override="configured-corpus")

                    # api_spec.md mục 7: lifespan phải dựng và lưu thêm registry mới.
                    assert (
                        app.state.api_settings.chatbot_api_key.get_secret_value()
                        == "test-chatbot-api-key"
                    )
                    assert app.state.redis is fake_redis
                    assert app.state.database_engine is engine
                    assert app.state.orchestrator is fake_orchestrator
                    redis_cls.from_url.assert_called_once_with(redis_settings.redis_url)
                    answer_cache_cls.assert_called_once()
                    retrieval_cache_cls.assert_called_once()
                    single_flight_cls.assert_called_once()
                    admission_cls.assert_called_once()
                    orchestrator_cls.assert_called_once()

        asyncio.run(run_lifespan())
        engine.dispose.assert_awaited_once()
        fake_redis.aclose.assert_awaited_once()


class TestObservabilityWiring:
    """observability_spec.md mục 4.4: root span `chat_turn` gắn `langfuse_trace_id`
    vào `TurnTrace`/chatlog, `metrics.record_turn` chạy cùng chỗ ghi chatlog."""

    def test_stream_chat_turn_copies_current_trace_id_into_recorded_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`trace.langfuse_trace_id` phải lấy đúng từ `tracing.current_trace_id()`
        ngay sau khi mở root span — không tự bịa giá trị khác."""
        monkeypatch.setattr(
            api_routes.tracing, "current_trace_id", lambda: "fixed-trace-id"
        )
        repository = _Repository()
        manager = ChatLogTaskManager(repository, _METADATA)

        async def consume() -> None:
            events = stream_chat_turn(
                _Orchestrator("answered", "miss"),
                [ChatMessage(role="user", content="q")],
                _context(),
                manager,
            )
            async for _ in events:
                pass
            await asyncio.sleep(0)
            await manager.drain()

        asyncio.run(consume())

        assert repository.turns[0].langfuse_trace_id == "fixed-trace-id"

    def test_stream_chat_turn_langfuse_trace_id_none_when_disabled(self) -> None:
        """Không monkeypatch gì (Langfuse disabled mặc định, thiếu key) -> NULL."""
        repository = _Repository()
        manager = ChatLogTaskManager(repository, _METADATA)

        async def consume() -> None:
            events = stream_chat_turn(
                _Orchestrator("answered", "miss"),
                [ChatMessage(role="user", content="q")],
                _context(),
                manager,
            )
            async for _ in events:
                pass
            await asyncio.sleep(0)
            await manager.drain()

        asyncio.run(consume())

        assert repository.turns[0].langfuse_trace_id is None

    def test_stream_chat_turn_calls_record_turn_exactly_once_with_final_trace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`metrics.record_turn` phải chạy đúng 1 lần/lượt, cùng `trace` đã điền
        đầy đủ dùng để ghi chatlog — không tính toán lại từ nguồn khác."""
        recorded: list[TurnTrace] = []
        monkeypatch.setattr(api_routes.metrics, "record_turn", recorded.append)
        repository = _Repository()
        manager = ChatLogTaskManager(repository, _METADATA)

        async def consume() -> None:
            events = stream_chat_turn(
                _Orchestrator("answered", "miss"),
                [ChatMessage(role="user", content="q")],
                _context(),
                manager,
            )
            async for _ in events:
                pass

        asyncio.run(consume())

        assert len(recorded) == 1
        assert recorded[0].outcome == "answered"
        assert recorded[0].cache_status == "miss"

    def test_stream_chat_turn_calls_record_turn_even_on_cancellation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lượt bị huỷ giữa chừng vẫn phải cập nhật metric (finally, giống chatlog)."""
        recorded: list[TurnTrace] = []
        monkeypatch.setattr(api_routes.metrics, "record_turn", recorded.append)
        repository = _Repository()
        manager = ChatLogTaskManager(repository, _METADATA)

        class WaitingOrchestrator:
            def stream(
                self,
                _: list[ChatMessage],
                __: RequestContext,
                trace: TurnTrace,
            ):
                async def events():
                    yield DoneEvent()
                    await asyncio.Event().wait()

                return events()

        async def cancel() -> None:
            events = stream_chat_turn(
                WaitingOrchestrator(),
                [ChatMessage(role="user", content="q")],
                _context(),
                manager,
            )
            await events.__anext__()
            with pytest.raises(asyncio.CancelledError):
                await events.athrow(asyncio.CancelledError)

        asyncio.run(cancel())

        assert len(recorded) == 1
        assert recorded[0].outcome == "client_disconnected"


class TestLangfuseFlushOnShutdown:
    """observability_spec.md mục 4.4: `_flush_langfuse` không bao giờ chặn shutdown
    quá timeout hay để lộ exception, kể cả khi Langfuse không phản hồi."""

    def test_flush_langfuse_returns_promptly_when_client_hangs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import production_legal_qa_rag.api.app as app_module

        class _SlowClient:
            def flush(self) -> None:
                time.sleep(0.3)

        monkeypatch.setattr(app_module, "get_langfuse_client", lambda: _SlowClient())
        monkeypatch.setattr(app_module, "_LANGFUSE_FLUSH_TIMEOUT_SECONDS", 0.02)

        started = time.monotonic()
        asyncio.run(app_module._flush_langfuse())
        elapsed = time.monotonic() - started

        assert elapsed < 1.0

    def test_flush_langfuse_swallows_exception_without_raising(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import production_legal_qa_rag.api.app as app_module

        class _BrokenClient:
            def flush(self) -> None:
                raise RuntimeError("Langfuse không phản hồi")

        monkeypatch.setattr(app_module, "get_langfuse_client", lambda: _BrokenClient())
        caplog.set_level(logging.WARNING, logger="production_legal_qa_rag.api.app")

        asyncio.run(app_module._flush_langfuse())  # không được raise

        assert "flush Langfuse" in caplog.text
