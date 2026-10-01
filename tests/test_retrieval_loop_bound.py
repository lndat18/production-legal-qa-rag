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


def test_hyde_khong_nuot_cancelled_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY_1", "k")

    async def create(**kwargs: Any) -> Any:
        raise asyncio.CancelledError

    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    async def run() -> None:
        await HydeGenerator(client=fake).generate("q")  # type: ignore[arg-type]

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
