"""Test `observability/tracing.py` (observability_spec.md mục 4).

Trọng tâm: cơ chế fail-safe khi thiếu `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`
(mục 4.1) — `span`/`generation` phải no-op, không raise, không
phụ thuộc `.env` thật của máy dev (đối chiếu quy ước `tests/test_config.py`: luôn
disable `env_file` + set biến môi trường tường minh cho quyết định).

Không test kịch bản "enabled thật" nối Langfuse self-host thật (cần network, ngoài
phạm vi unit test) — chỉ test đúng những gì `observability_spec.md` mục 4.1 cam kết:
hành vi no-op khi thiếu key, singleton, và lồng span/generation không raise.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from production_legal_qa_rag.config import LangfuseSettings
from production_legal_qa_rag.observability import tracing


def _khong_doc_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bỏ qua `.env` thật của máy dev — chỉ dùng biến môi trường test set tường minh."""
    monkeypatch.setattr(
        LangfuseSettings,
        "model_config",
        {**LangfuseSettings.model_config, "env_file": None},
    )


def _set_keys(
    monkeypatch: pytest.MonkeyPatch,
    *,
    public_key: str | None,
    secret_key: str | None,
) -> None:
    _khong_doc_dotenv(monkeypatch)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    if public_key is not None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", public_key)
    if secret_key is not None:
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", secret_key)


@pytest.fixture(autouse=True)
def _reset_singleton() -> Iterator[None]:
    """`get_langfuse_client()` cache client ở module-level — reset để mỗi test tự
    dựng lại theo đúng settings mà nó monkeypatch, không rò rỉ giữa các test/module
    khác trong cùng phiên pytest (test khác trong repo cũng gọi tới `tracing.*`).
    """
    tracing._client = None
    yield
    tracing._client = None


# --------------------------------------------------------------- get_langfuse_client


def test_get_langfuse_client_la_singleton_trong_1_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, public_key=None, secret_key=None)

    assert tracing.get_langfuse_client() is tracing.get_langfuse_client()


# --------------------------------------------------------------- span / generation


def test_span_disabled_khong_raise_va_van_chay_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, public_key=None, secret_key=None)
    executed = False

    with tracing.span("cache_lookup", input="q") as observation:
        executed = True
        observation.update(output={"hit": False})

    assert executed


def test_generation_disabled_khong_raise_va_van_chay_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, public_key=None, secret_key=None)
    executed = False

    with tracing.generation("guardrail", model="guardrail-model") as observation:
        executed = True
        observation.update(output={"verdict": "allow"})

    assert executed


def test_span_disabled_van_lan_truyen_exception_tu_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """no-op không được nuốt lỗi nghiệp vụ bên trong — chỉ mất khả năng quan sát."""
    _set_keys(monkeypatch, public_key=None, secret_key=None)

    with pytest.raises(ValueError, match="boom"), tracing.span("retrieve"):
        raise ValueError("boom")


def test_generation_disabled_van_lan_truyen_exception_tu_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, public_key=None, secret_key=None)

    with (
        pytest.raises(RuntimeError, match="groq lỗi"),
        tracing.generation("hyde", model="hyde-model"),
    ):
        raise RuntimeError("groq lỗi")


def test_generation_long_trong_span_khong_raise_khi_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cây trace mục 4.3: generation con (vd `hyde`) lồng trong span cha (vd
    `retrieve`) — disabled thì cả hai vẫn no-op, không raise khi lồng nhau.
    """
    _set_keys(monkeypatch, public_key=None, secret_key=None)

    with tracing.span("retrieve") as parent:
        with tracing.generation("hyde", model="hyde-model") as child:
            child.update(output={"used": True})
        parent.update(output={"num_chunks": 3, "chunk_ids": ["a"]})


def test_span_song_song_khong_lan_nhau_khi_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`guardrail`/`condense` chạy song song, đều là con trực tiếp của `chat_turn`
    (mục 4.3) — mở tuần tự 2 span "song song" không được raise hay phụ thuộc lẫn
    nhau khi disabled.
    """
    _set_keys(monkeypatch, public_key=None, secret_key=None)

    with tracing.span("chat_turn"):
        with tracing.generation("guardrail", model="guardrail-model") as guardrail_obs:
            guardrail_obs.update(output={"verdict": "allow"})
        with tracing.generation("condense", model="condense-model") as condense_obs:
            condense_obs.update(output={"standalone_query": "q"})


# ------------------------------------------- user_id / session_id (mục 4.2, 4.4)


class _PropagateSpy:
    """Thay `propagate_attributes`: ghi lại kwargs và thứ tự vào/ra context."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.events: list[str] = []

    @contextmanager
    def __call__(self, **kwargs: Any) -> Iterator[None]:
        self.calls.append(kwargs)
        self.events.append("enter")
        try:
            yield
        finally:
            self.events.append("exit")


def test_span_co_user_va_session_goi_propagate_attributes_quanh_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, public_key=None, secret_key=None)
    spy = _PropagateSpy()
    monkeypatch.setattr(tracing, "propagate_attributes", spy)

    with tracing.span("chat_turn", user_id="user-1", session_id="chat-1"):
        spy.events.append("body")

    assert spy.calls == [{"user_id": "user-1", "session_id": "chat-1"}]
    assert spy.events == ["enter", "body", "exit"]


@pytest.mark.parametrize(
    ("user_id", "session_id"),
    [("user-1", None), (None, "chat-1")],
)
def test_span_chi_co_mot_trong_hai_van_goi_propagate_attributes(
    monkeypatch: pytest.MonkeyPatch, user_id: str | None, session_id: str | None
) -> None:
    """Giá trị thiếu truyền `None` cho SDK (bị bỏ qua), không bịa giá trị."""
    _set_keys(monkeypatch, public_key=None, secret_key=None)
    spy = _PropagateSpy()
    monkeypatch.setattr(tracing, "propagate_attributes", spy)

    with tracing.span("chat_turn", user_id=user_id, session_id=session_id):
        pass

    assert spy.calls == [{"user_id": user_id, "session_id": session_id}]


def test_span_khong_co_user_va_session_khong_goi_propagate_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, public_key=None, secret_key=None)
    spy = _PropagateSpy()
    monkeypatch.setattr(tracing, "propagate_attributes", spy)

    with tracing.span("cache_lookup", input="q"):
        pass

    assert spy.calls == []


def test_span_co_user_va_session_van_lan_truyen_exception_tu_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, public_key=None, secret_key=None)
    spy = _PropagateSpy()
    monkeypatch.setattr(tracing, "propagate_attributes", spy)

    with (
        pytest.raises(ValueError, match="boom"),
        tracing.span("chat_turn", user_id="user-1"),
    ):
        raise ValueError("boom")

    assert spy.events == ["enter", "exit"]


def test_span_disabled_voi_user_va_session_khong_raise_va_chay_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, public_key=None, secret_key=None)
    executed = False

    with tracing.span("chat_turn", user_id="user-1", session_id="chat-1") as root:
        executed = True
        root.update(output="x")

    assert executed


def test_span_va_generation_disabled_khong_log_canh_bao_khi_su_dung(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """SDK chỉ cảnh báo một lần lúc dựng client thiếu khoá; dùng span thì im lặng."""
    _set_keys(monkeypatch, public_key=None, secret_key=None)
    tracing.get_langfuse_client()
    caplog.clear()
    caplog.set_level(logging.WARNING)

    with tracing.span("chat_turn", user_id="user-1", session_id="chat-1") as root:
        root.update(output="x")
        with tracing.generation("answer", model="m") as child:
            child.update(output="y")

    assert caplog.records == []


def test_span_that_ghi_user_id_va_session_id_len_root_va_span_con(
    in_memory_langfuse: Any,
) -> None:
    with tracing.span("chat_turn", user_id="user-1", session_id="chat-1") as root:
        root_attributes = dict(root._otel_span.attributes)
        with tracing.generation("answer", model="m") as child:
            child_attributes = dict(child._otel_span.attributes)

    assert root_attributes["user.id"] == "user-1"
    assert root_attributes["session.id"] == "chat-1"
    assert child_attributes["user.id"] == "user-1"
    assert child_attributes["session.id"] == "chat-1"


def test_span_that_khong_bia_session_id_khi_thieu(in_memory_langfuse: Any) -> None:
    with tracing.span("chat_turn", user_id="user-1") as root:
        attributes = dict(root._otel_span.attributes)

    assert attributes["user.id"] == "user-1"
    assert "session.id" not in attributes


def test_span_that_khong_co_user_va_session_thi_khong_ghi_thuoc_tinh_trace(
    in_memory_langfuse: Any,
) -> None:
    with tracing.span("cache_lookup", input="q") as observation:
        attributes = dict(observation._otel_span.attributes)

    assert "user.id" not in attributes
    assert "session.id" not in attributes
