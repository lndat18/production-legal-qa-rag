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

from collections.abc import Iterator

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
