"""Unit test LoopBoundClient và độ bền của RerankerClient/HydeGenerator (mục 9, 10)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from production_legal_qa_rag.retrieval.hyde import HydeGenerator
from production_legal_qa_rag.retrieval.loop_bound import LoopBoundClient


def test_get_ngoai_coroutine_raise_runtime_error():
    client = LoopBoundClient(object)
    with pytest.raises(RuntimeError):
        client.get()


def test_lazy_khong_tao_khi_chua_get():
    made: list[object] = []
    LoopBoundClient(lambda: made.append(1) or object())
    assert made == []


def test_cung_loop_tai_su_dung_client():
    client = LoopBoundClient(object)

    async def run() -> tuple[object, object]:
        return client.get(), client.get()

    first, second = asyncio.run(run())
    assert first is second


def test_loop_khac_tao_lai_client():
    client = LoopBoundClient(object)

    async def run() -> object:
        return client.get()

    assert asyncio.run(run()) is not asyncio.run(run())


def test_client_inject_duoc_dung_nguyen_qua_nhieu_loop():
    injected = object()
    made: list[int] = []
    client = LoopBoundClient(lambda: made.append(1) or object(), injected)

    async def run() -> object:
        return client.get()

    assert asyncio.run(run()) is injected
    assert asyncio.run(run()) is injected
    assert made == []


"""HTTP RerankerClient tests removed: reranker now executes in-process."""
"""
def _reranker(post: Any) -> RerankerClient:
    settings = SimpleNamespace(
        endpoint_url="u",
        api_key="k",
        max_retries=2,
        connect_timeout_seconds=1,
        timeout_seconds=1,
    )
    return RerankerClient(settings, client=SimpleNamespace(post=post))  # type: ignore[arg-type]


def test_reranker_exception_bat_ky_khong_retry_tra_none(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "production_legal_qa_rag.retrieval.reranker_client._BACKOFF_SECONDS", 0.0
    )
    calls = [0]

    async def post(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        raise RuntimeError("Event loop is closed")

    assert asyncio.run(_reranker(post).rerank("q", ["a"])) is None
    assert calls == [1]  # lỗi ngoài httpx: fallback ngay, không retry


def test_reranker_khong_nuot_cancelled_error():
    async def post(*args: Any, **kwargs: Any) -> Any:
        raise asyncio.CancelledError

    async def run() -> None:
        await _reranker(post).rerank("q", ["a"])

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
"""


def test_hyde_khong_nuot_cancelled_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")

    async def create(**kwargs: Any) -> Any:
        raise asyncio.CancelledError

    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    async def run() -> None:
        await HydeGenerator(client=fake).generate("q")  # type: ignore[arg-type]

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
