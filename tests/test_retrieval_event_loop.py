"""Hồi quy PR #19: dùng lại một RetrievalPipeline qua nhiều `asyncio.run`.

HyDE (Groq) không được nuốt lỗi loop và reranker không được crash, nên
nhánh A và rerank vẫn chạy ở lần gọi thứ 2, 3 (retrieval_spec.md mục 9, 10).

Fake client mô phỏng đúng `AsyncGroq`/`httpx.AsyncClient`: gắn với event loop
lúc tạo và ném `RuntimeError('Event loop is closed')` nếu dùng ở loop khác.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from test_retrieval_pipeline import FakeDense, FakeEmbedder, FakeSparse

from production_legal_qa_rag.retrieval import hyde as hyde_module
from production_legal_qa_rag.retrieval import reranker_client as reranker_module
from production_legal_qa_rag.retrieval.hyde import HydeGenerator
from production_legal_qa_rag.retrieval.pipeline import RetrievalPipeline
from production_legal_qa_rag.retrieval.reranker_client import RerankerClient

IDS = [f"c{i}" for i in range(1, 8)]
created: list[Any] = []


class _LoopBoundFake:
    """Gắn với loop của lần dùng đầu tiên, như client thật."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.calls = 0
        created.append(self)

    def _check_loop(self) -> None:
        running = asyncio.get_running_loop()
        if self.loop is None:
            self.loop = running
        elif self.loop is not running:
            raise RuntimeError("Event loop is closed")


class FakeGroq(_LoopBoundFake):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self._check_loop()
        self.calls += 1
        message = SimpleNamespace(content="đoạn giả định")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeHttpx(_LoopBoundFake):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    async def post(self, url: str, **kwargs: Any) -> Any:
        self._check_loop()
        self.calls += 1
        n = len(kwargs["json"]["passages"])
        return SimpleNamespace(
            status_code=200, json=lambda: {"scores": [float(n - i) for i in range(n)]}
        )


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> tuple[RetrievalPipeline, FakeSparse]:
    created.clear()
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("RERANKER_ENDPOINT_URL", "https://example.test/rerank")
    monkeypatch.setenv("RERANKER_API_KEY", "k")
    monkeypatch.setattr(hyde_module, "AsyncGroq", FakeGroq)
    monkeypatch.setattr(reranker_module.httpx, "AsyncClient", FakeHttpx)
    sparse = FakeSparse({"đoạn giả định": IDS, "hỏi": IDS})
    pipe = RetrievalPipeline(
        hyde=HydeGenerator(),
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        dense_search=FakeDense(IDS),  # type: ignore[arg-type]
        sparse_index=sparse,  # type: ignore[arg-type]
        reranker=RerankerClient(),
    )
    return pipe, sparse


@pytest.mark.parametrize("run", [1, 2, 3])
def test_dung_lai_pipeline_qua_nhieu_asyncio_run(
    pipeline: tuple[RetrievalPipeline, FakeSparse], run: int
):
    pipe, sparse = pipeline
    for _ in range(run):
        result = asyncio.run(pipe.retrieve("hỏi"))
        # Nhánh A vẫn chạy (HyDE không bị nuốt lỗi loop) ...
        assert "đoạn giả định" in sparse.texts
        sparse.texts.clear()
        # ... và rerank thật thành công, không rơi vào fallback.
        assert result and all(c.rerank_score is not None for c in result)


def test_client_duoc_tao_lai_khi_loop_doi(
    pipeline: tuple[RetrievalPipeline, FakeSparse],
):
    pipe, _ = pipeline
    asyncio.run(pipe.retrieve("hỏi"))
    asyncio.run(pipe.retrieve("hỏi"))
    groq_clients = [c for c in created if isinstance(c, FakeGroq)]
    assert len(groq_clients) == 2
    assert groq_clients[0].loop is not groq_clients[1].loop


def test_reranker_khong_bao_gio_raise_khi_loi_la_exception_bat_ky():
    class Broken:
        async def post(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Event loop is closed")

    client = RerankerClient(
        SimpleNamespace(  # type: ignore[arg-type]
            endpoint_url="u",
            api_key="k",
            max_retries=0,
            connect_timeout_seconds=1,
            timeout_seconds=1,
        ),
        client=Broken(),  # type: ignore[arg-type]
    )
    assert asyncio.run(client.rerank("q", ["a"])) is None
