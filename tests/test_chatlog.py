"""Tests cho chatlog package (chatlog_spec.md).

Bao gồm:
- Unit tests cho TurnRecord (Pydantic model, schema validation).
- Unit tests cho from_trace() (builder từ TurnTrace + RequestContext).
- Unit tests cho ChatLogRepository.record() (async, no-throw).
- Data/schema validation: cột bắt buộc, nullable, kiểu dữ liệu.
- Smoke test import public API.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from production_legal_qa_rag.cache.models import CacheStatus
from production_legal_qa_rag.chatlog import (
    ChatLogRepository,
    TurnRecord,
    create_engine as create_chatlog_engine,
    from_trace,
)
from production_legal_qa_rag.chatlog.tables import chat_turns, metadata
from production_legal_qa_rag.conversation.models import RequestContext, TurnTrace
from production_legal_qa_rag.generation.models import (
    Citation,
    GuardrailVerdict,
    Usage,
    WarningEvent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_request_context(
    user_id: str = "uid-001",
    chat_id: str | None = "chat-001",
    request_id: str = "req-001",
) -> RequestContext:
    """Tạo RequestContext mẫu."""
    return RequestContext(user_id=user_id, chat_id=chat_id, request_id=request_id)


def _make_turn_trace(
    outcome: Literal["answered", "refused", "error"] = "answered",
    cache_status: CacheStatus = "miss",
    *,
    with_verdict: bool = True,
    with_usage: bool = False,
) -> TurnTrace:
    """Tạo TurnTrace mẫu với giá trị mặc định hợp lệ."""
    verdict = GuardrailVerdict(verdict="allow", reason="ok") if with_verdict else None
    usage = (
        Usage(prompt_tokens=100, completion_tokens=50, reasoning_tokens=None)
        if with_usage
        else None
    )
    return TurnTrace(
        raw_query="Trợ cấp thôi việc là gì?",
        standalone_query="Trợ cấp thôi việc?",
        verdict=verdict,
        cache_status=cache_status,
        outcome=outcome,
        error_code=None,
        chunk_ids=["c1", "c2"],
        answer_text="Trợ cấp thôi việc là...",
        citations=[
            Citation(n=1, chunk_id="c1", source_document="BLLĐ", breadcrumb="Điều 46")
        ],
        warnings=[],
        usage=usage,
        time_to_first_token_ms=200,
        latency_ms=1000,
    )


# ---------------------------------------------------------------------------
# 1. TurnRecord — schema validation
# ---------------------------------------------------------------------------


class TestTurnRecordSchema:
    """Kiểm tra schema TurnRecord bảo vệ spec mục 2."""

    def test_id_tu_sinh_uuid(self) -> None:
        """id phải tự sinh UUID khi không truyền."""
        record = TurnRecord(
            request_id="r1",
            user_id="u1",
            raw_query="q",
            outcome="answered",
            verdict="allow",
            cache_status="miss",
            latency_ms=0,
        )
        assert uuid.UUID(record.id)  # Đúng định dạng UUID

    def test_id_co_the_truyen_vao(self) -> None:
        """id truyền tường minh phải được giữ nguyên."""
        custom_id = str(uuid.uuid4())
        record = TurnRecord(
            id=custom_id,
            request_id="r1",
            user_id="u1",
            raw_query="q",
            outcome="answered",
            verdict="allow",
            cache_status="miss",
            latency_ms=0,
        )
        assert record.id == custom_id

    def test_chat_id_nullable(self) -> None:
        """chat_id có thể là None (spec mục 2)."""
        record = TurnRecord(
            request_id="r1",
            user_id="u1",
            chat_id=None,
            raw_query="q",
            outcome="answered",
            verdict="allow",
            cache_status="miss",
            latency_ms=0,
        )
        assert record.chat_id is None

    def test_standalone_query_nullable(self) -> None:
        """standalone_query là None khi bị chặn trước condense."""
        record = TurnRecord(
            request_id="r1",
            user_id="u1",
            raw_query="q",
            standalone_query=None,
            outcome="refused",
            verdict="out_of_scope",
            cache_status="miss",
            latency_ms=0,
        )
        assert record.standalone_query is None

    def test_default_list_fields(self) -> None:
        """chunk_ids, citations, warnings mặc định là list rỗng."""
        record = TurnRecord(
            request_id="r1",
            user_id="u1",
            raw_query="q",
            outcome="answered",
            verdict="allow",
            cache_status="miss",
            latency_ms=0,
        )
        assert record.chunk_ids == []
        assert record.citations == []
        assert record.warnings == []

    def test_usage_nullable(self) -> None:
        """usage có thể là None."""
        record = TurnRecord(
            request_id="r1",
            user_id="u1",
            raw_query="q",
            outcome="answered",
            verdict="allow",
            cache_status="miss",
            latency_ms=0,
            usage=None,
        )
        assert record.usage is None

    def test_time_to_first_token_nullable(self) -> None:
        """time_to_first_token_ms có thể là None."""
        record = TurnRecord(
            request_id="r1",
            user_id="u1",
            raw_query="q",
            outcome="answered",
            verdict="allow",
            cache_status="miss",
            latency_ms=100,
            time_to_first_token_ms=None,
        )
        assert record.time_to_first_token_ms is None

    def test_truong_bat_buoc_thieu_raise(self) -> None:
        """Thiếu trường bắt buộc phải raise ValidationError."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            TurnRecord()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 2. from_trace — builder
# ---------------------------------------------------------------------------


class TestFromTrace:
    """Kiểm tra from_trace() bảo vệ spec mục 2, 4."""

    def test_answered_full(self) -> None:
        """Trường hợp answered đầy đủ: map đúng tất cả trường."""
        trace = _make_turn_trace(outcome="answered", with_usage=True)
        ctx = _make_request_context()
        record = from_trace(
            trace, ctx, prompt_version="v1", corpus_version="2026", model_name="gpt-x"
        )

        assert record.request_id == ctx.request_id
        assert record.user_id == ctx.user_id
        assert record.chat_id == ctx.chat_id
        assert record.raw_query == trace.raw_query
        assert record.standalone_query == trace.standalone_query
        assert record.outcome == "answered"
        assert record.verdict == "allow"
        assert record.cache_status == "miss"
        assert record.chunk_ids == ["c1", "c2"]
        assert record.answer_text == "Trợ cấp thôi việc là..."
        assert len(record.citations) == 1
        assert record.citations[0]["n"] == 1
        assert record.usage is not None
        assert record.usage["prompt_tokens"] == 100
        assert record.latency_ms == 1000
        assert record.prompt_version == "v1"
        assert record.corpus_version == "2026"
        assert record.model_name == "gpt-x"

    def test_refused_no_verdict(self) -> None:
        """Không có verdict → verdict_str = 'allow' (fallback)."""
        trace = _make_turn_trace(outcome="refused", with_verdict=False)
        ctx = _make_request_context()
        record = from_trace(trace, ctx)

        assert record.outcome == "refused"
        assert record.verdict == "allow"

    def test_out_of_scope_verdict(self) -> None:
        """verdict 'out_of_scope' được map đúng."""
        trace = _make_turn_trace(outcome="refused")
        trace.verdict = GuardrailVerdict(verdict="out_of_scope", reason="ngoài phạm vi")
        ctx = _make_request_context()
        record = from_trace(trace, ctx)

        assert record.verdict == "out_of_scope"

    def test_error_outcome(self) -> None:
        """outcome 'error' với error_code được map đúng."""
        trace = _make_turn_trace(outcome="error")
        trace.error_code = "timeout"
        ctx = _make_request_context()
        record = from_trace(trace, ctx)

        assert record.outcome == "error"
        assert record.error_code == "timeout"

    def test_chat_id_none(self) -> None:
        """chat_id None từ ctx được truyền đúng."""
        trace = _make_turn_trace()
        ctx = _make_request_context(chat_id=None)
        record = from_trace(trace, ctx)

        assert record.chat_id is None

    def test_cache_hit(self) -> None:
        """cache_status 'answer_hit' được map đúng."""
        trace = _make_turn_trace(cache_status="answer_hit")
        ctx = _make_request_context()
        record = from_trace(trace, ctx)

        assert record.cache_status == "answer_hit"

    def test_usage_none_khi_khong_co(self) -> None:
        """usage = None khi trace không có usage."""
        trace = _make_turn_trace(with_usage=False)
        ctx = _make_request_context()
        record = from_trace(trace, ctx)

        assert record.usage is None

    def test_citations_serialize_dung(self) -> None:
        """Citations được serialize thành list[dict]."""
        trace = _make_turn_trace()
        ctx = _make_request_context()
        record = from_trace(trace, ctx)

        assert isinstance(record.citations, list)
        assert isinstance(record.citations[0], dict)
        assert "n" in record.citations[0]
        assert "chunk_id" in record.citations[0]

    def test_default_version_fields(self) -> None:
        """Không truyền version → mặc định chuỗi rỗng."""
        trace = _make_turn_trace()
        ctx = _make_request_context()
        record = from_trace(trace, ctx)

        assert record.prompt_version == ""
        assert record.corpus_version == ""
        assert record.model_name == ""


# ---------------------------------------------------------------------------
# 3. ChatLogRepository.record() — async, no-throw
# ---------------------------------------------------------------------------


class TestChatLogRepository:
    """Kiểm tra ChatLogRepository.record() bảo vệ spec mục 4."""

    def _make_sample_record(self) -> TurnRecord:
        """Tạo TurnRecord hợp lệ để test."""
        return TurnRecord(
            request_id="req-test",
            user_id="uid-test",
            chat_id="chat-test",
            raw_query="test query",
            outcome="answered",
            verdict="allow",
            cache_status="miss",
            latency_ms=500,
        )

    def test_record_goi_insert_khi_thanh_cong(self) -> None:
        """record() phải thực hiện INSERT khi engine hoạt động bình thường."""
        mock_engine = MagicMock()
        repo = ChatLogRepository(mock_engine)

        called: list[str] = []

        async def fake_insert(turn: TurnRecord) -> None:
            called.append(turn.request_id)

        repo._insert = fake_insert  # type: ignore[assignment]

        turn = self._make_sample_record()
        asyncio.run(repo.record(turn))

        assert called == ["req-test"]

    def test_record_khong_raise_khi_exception(self) -> None:
        """record() KHÔNG được ném ngoại lệ khi _insert gặp lỗi (spec mục 4)."""
        mock_engine = MagicMock()
        repo = ChatLogRepository(mock_engine)

        async def failing_insert(turn: TurnRecord) -> None:
            raise RuntimeError("DB down")

        repo._insert = failing_insert  # type: ignore[assignment]

        turn = self._make_sample_record()
        # Không được raise — nếu raise thì test này fail đúng như spec yêu cầu
        asyncio.run(repo.record(turn))

    def test_record_khong_raise_khi_timeout(self) -> None:
        """record() KHÔNG được ném ngoại lệ khi _insert timeout (spec mục 4: 5s)."""
        mock_engine = MagicMock()
        repo = ChatLogRepository(mock_engine)

        async def slow_insert(turn: TurnRecord) -> None:
            # Giả lập timeout: sleep lâu hơn _WRITE_TIMEOUT_SECONDS
            await asyncio.sleep(100)

        repo._insert = slow_insert  # type: ignore[assignment]

        # Patch timeout ngắn để test không chờ 5s thật
        with patch(
            "production_legal_qa_rag.chatlog.repository._WRITE_TIMEOUT_SECONDS",
            0.01,
        ):
            turn = self._make_sample_record()
            asyncio.run(repo.record(turn))  # Không được raise


# ---------------------------------------------------------------------------
# 4. SQLAlchemy table schema validation
# ---------------------------------------------------------------------------


class TestChatTurnsSchema:
    """Kiểm tra schema bảng chat_turns khớp spec mục 2."""

    def test_bang_co_ten_dung(self) -> None:
        """Bảng phải có tên 'chat_turns'."""
        assert chat_turns.name == "chat_turns"

    def test_cac_cot_bat_buoc_ton_tai(self) -> None:
        """Các cột bắt buộc theo spec mục 2 phải tồn tại."""
        required_columns = {
            "id",
            "created_at",
            "request_id",
            "user_id",
            "raw_query",
            "outcome",
            "verdict",
            "cache_status",
            "chunk_ids",
            "answer_text",
            "citations",
            "warnings",
            "latency_ms",
            "prompt_version",
            "corpus_version",
            "model_name",
        }
        existing = {c.name for c in chat_turns.columns}
        assert required_columns <= existing

    def test_cot_nullable_dung_spec(self) -> None:
        """Các cột nullable đúng theo spec mục 2."""
        col_map = {c.name: c for c in chat_turns.columns}

        # Nullable
        assert col_map["chat_id"].nullable is True
        assert col_map["standalone_query"].nullable is True
        assert col_map["error_code"].nullable is True
        assert col_map["usage"].nullable is True
        assert col_map["time_to_first_token_ms"].nullable is True

        # NOT NULL
        assert col_map["id"].nullable is False
        assert col_map["user_id"].nullable is False
        assert col_map["raw_query"].nullable is False

    def test_id_la_primary_key(self) -> None:
        """id phải là primary key."""
        col_map = {c.name: c for c in chat_turns.columns}
        assert col_map["id"].primary_key is True

    def test_index_ton_tai(self) -> None:
        """Index created_at, user_id, chat_id phải được định nghĩa."""
        index_names = {idx.name for idx in metadata.tables["chat_turns"].indexes}
        assert "ix_chat_turns_created_at" in index_names
        assert "ix_chat_turns_user_id" in index_names
        assert "ix_chat_turns_chat_id" in index_names


# ---------------------------------------------------------------------------
# 5. Public API import
# ---------------------------------------------------------------------------


class TestPublicApi:
    """Smoke test import public API từ __init__.py (spec mục 7)."""

    def test_import_turn_record(self) -> None:
        """TurnRecord có thể import từ package."""
        from production_legal_qa_rag.chatlog import TurnRecord as TR

        assert TR is TurnRecord

    def test_import_from_trace(self) -> None:
        """from_trace có thể import từ package."""
        from production_legal_qa_rag.chatlog import from_trace as ft

        assert ft is from_trace

    def test_import_repository(self) -> None:
        """ChatLogRepository có thể import từ package."""
        from production_legal_qa_rag.chatlog import ChatLogRepository as CLR

        assert CLR is ChatLogRepository

    def test_import_create_engine(self) -> None:
        """create_engine có thể import từ package."""
        from production_legal_qa_rag.chatlog import create_engine as ce

        assert ce is create_chatlog_engine


# ---------------------------------------------------------------------------
# 6. DatabaseSettings
# ---------------------------------------------------------------------------


class TestDatabaseSettings:
    """Kiểm tra DatabaseSettings (spec mục 6)."""

    def test_gia_tri_mac_dinh(self) -> None:
        """database_url có giá trị mặc định hợp lệ."""
        import os

        # Đảm bảo biến môi trường không bị set
        env_bak = os.environ.pop("CHATLOG_DATABASE_URL", None)
        try:
            from production_legal_qa_rag.config import DatabaseSettings

            settings = DatabaseSettings()
            assert "postgresql+asyncpg://" in settings.database_url
        finally:
            if env_bak is not None:
                os.environ["CHATLOG_DATABASE_URL"] = env_bak

    def test_override_qua_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CHATLOG_DATABASE_URL override được database_url."""
        custom_url = "postgresql+asyncpg://user:pass@db:5432/mydb"
        monkeypatch.setenv("CHATLOG_DATABASE_URL", custom_url)

        from importlib import reload

        import production_legal_qa_rag.config as cfg_module

        reload(cfg_module)
        settings = cfg_module.DatabaseSettings()
        assert settings.database_url == custom_url
