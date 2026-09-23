"""Test reranker_client, hyde, query_embedder, sparse_index, dense_search (fake client)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pinecone.exceptions import NotFoundException
from typer.testing import CliRunner

from production_legal_qa_rag.chunking.models import Chunk
from production_legal_qa_rag.config import (
    EmbeddingSettings,
    LLMSettings,
    RerankerSettings,
    VectorDBSettings,
)
from production_legal_qa_rag.embedding.models import PineconeMetadata
from production_legal_qa_rag.retrieval import (
    query_embedder,
    reranker_client,
    sparse_index,
)
from production_legal_qa_rag.retrieval.bm25 import BM25Encoder
from production_legal_qa_rag.retrieval.dense_search import DenseSearch
from production_legal_qa_rag.retrieval.hyde import (
    HYDE_SYSTEM_PROMPT,
    HYDE_USER_TEMPLATE,
    HydeGenerator,
)
from production_legal_qa_rag.retrieval.models import Candidate, RetrievalError
from production_legal_qa_rag.retrieval.query_embedder import QueryEmbedder
from production_legal_qa_rag.retrieval.reranker_client import RerankerClient
from production_legal_qa_rag.retrieval.sparse_index import SparseIndex, build_index


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {
        "RERANKER_ENDPOINT_URL": "http://rerank.test/rerank",
        "RERANKER_API_KEY": "k",
        "GROQ_API_KEY": "g",
        "HF_TOKEN": "h",
        "PINECONE_API_KEY": "p",
        "PINECONE_INDEX_NAME": "dense",
        "PINECONE_SPARSE_INDEX_NAME": "sparse",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(reranker_client, "_BACKOFF_SECONDS", 0.0)


# ---------------------------------------------------------------- reranker


def _rerank(handler: Any, passages: list[str] | None = None) -> tuple[Any, list[int]]:
    calls = [0]

    def counting(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        return handler(request, calls[0])

    async def run() -> Any:
        async with httpx.AsyncClient(transport=httpx.MockTransport(counting)) as http:
            client = RerankerClient(RerankerSettings(), http)
            return await client.rerank("q", passages or ["a", "b"])

    return asyncio.run(run()), calls


def test_rerank_thanh_cong_gui_api_key_va_1_request(env: None):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request, _n: int) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"scores": [0.1, 2]})

    scores, calls = _rerank(handler)
    assert scores == [0.1, 2.0]
    assert calls == [1]
    assert seen[0].headers["X-API-Key"] == "k"
    assert json.loads(seen[0].content) == {"query": "q", "passages": ["a", "b"]}


def test_rerank_passages_rong_khong_goi_mang(env: None):
    # passages=[] bị `or` thay bằng mặc định trong _rerank, nên gọi trực tiếp.
    async def run() -> Any:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(500))
        ) as http:
            return await RerankerClient(RerankerSettings(), http).rerank("q", [])

    assert asyncio.run(run()) == []


@pytest.mark.parametrize("status", [502, 503, 504])
def test_rerank_retry_khi_5xx_tam_thoi_roi_thanh_cong(env: None, status: int):
    def handler(_r: httpx.Request, n: int) -> httpx.Response:
        if n < 3:
            return httpx.Response(status)
        return httpx.Response(200, json={"scores": [1, 2]})

    scores, calls = _rerank(handler)
    assert scores == [1.0, 2.0]
    assert calls == [3]  # max_retries=2 -> tối đa 3 lần


@pytest.mark.parametrize("status", [502, 503, 504])
def test_rerank_het_retry_tra_none(env: None, status: int):
    scores, calls = _rerank(lambda r, n: httpx.Response(status))
    assert scores is None
    assert calls == [3]


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("x"), httpx.ConnectTimeout("x")],
)
def test_rerank_retry_khi_connect_error_hoac_connect_timeout(
    env: None, error: Exception
):
    def handler(request: httpx.Request, n: int) -> httpx.Response:
        if n < 2:
            raise error
        return httpx.Response(200, json={"scores": [1, 2]})

    scores, calls = _rerank(handler)
    assert scores == [1.0, 2.0]
    assert calls == [2]


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("x"),
        httpx.WriteTimeout("x"),
        httpx.ReadError("x"),
        httpx.RemoteProtocolError("x"),
    ],
)
def test_rerank_read_timeout_va_transport_error_khac_khong_retry(
    env: None, error: Exception
):
    def handler(request: httpx.Request, n: int) -> httpx.Response:
        raise error

    scores, calls = _rerank(handler)
    assert scores is None
    assert calls == [1]


def test_rerank_read_timeout_log_so_passage_va_timeout_khong_goi_y_tunnel(
    env: None, caplog: pytest.LogCaptureFixture
):
    def handler(request: httpx.Request, n: int) -> httpx.Response:
        raise httpx.ReadTimeout("x")

    passages = [f"p{i}" for i in range(36)]
    with caplog.at_level("WARNING", logger=reranker_client.logger.name):
        _rerank(handler, passages)
    assert "36 passage" in caplog.text
    assert "read timeout 90s" in caplog.text
    assert "hàng đợi" in caplog.text
    assert "ngrok" not in caplog.text and "tunnel" not in caplog.text


@pytest.mark.parametrize(
    ("n_passages", "expected_read"),
    [(2, 30.0), (12, 30.0), (13, 32.5), (36, 90.0), (44, 110.0)],
)
def test_rerank_read_timeout_ti_le_so_passage_voi_san_30s(
    env: None, n_passages: int, expected_read: float
):
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={"scores": [1.0] * n_passages})

    scores, _ = _rerank(handler, [f"p{i}" for i in range(n_passages)])
    assert scores == [1.0] * n_passages
    assert seen[0]["read"] == expected_read
    assert seen[0]["connect"] == 5.0  # connect timeout giữ nguyên


def test_rerank_loi_ket_noi_lien_tuc_tra_none(env: None):
    def handler(request: httpx.Request, n: int) -> httpx.Response:
        raise httpx.ConnectError("down")

    scores, calls = _rerank(handler)
    assert scores is None
    assert calls == [3]


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_rerank_khong_retry_khi_4xx(env: None, status: int):
    scores, calls = _rerank(lambda r, n: httpx.Response(status))
    assert scores is None
    assert calls == [1]


def test_rerank_404_khong_retry_va_log_goi_y_ngrok_runbook(
    env: None, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level("ERROR", logger=reranker_client.logger.name):
        scores, calls = _rerank(lambda r, n: httpx.Response(404))
    assert scores is None
    assert calls == [1]
    text = caplog.text
    assert "404" in text and "ngrok" in text and "mục 9.1" in text
    assert "cấu hình/payload" not in text


def test_rerank_4xx_khac_khong_goi_y_ngrok(env: None, caplog: pytest.LogCaptureFixture):
    with caplog.at_level("ERROR", logger=reranker_client.logger.name):
        _rerank(lambda r, n: httpx.Response(401))
    assert "cấu hình/payload" in caplog.text
    assert "ngrok" not in caplog.text


def test_rerank_loi_la_khong_retry_log_traceback_khong_goi_y_studio(
    env: None, caplog: pytest.LogCaptureFixture
):
    def handler(request: httpx.Request, n: int) -> httpx.Response:
        raise ValueError("bug lập trình")

    with caplog.at_level("WARNING", logger=reranker_client.logger.name):
        scores, calls = _rerank(handler)
    assert scores is None
    assert calls == [1]  # lỗi lạ không retry: fallback ngay
    assert "Traceback" in caplog.text and "bug lập trình" in caplog.text
    assert "Studio" not in caplog.text


def test_rerank_het_retry_5xx_van_goi_y_runbook(
    env: None, caplog: pytest.LogCaptureFixture
):
    with caplog.at_level("WARNING", logger=reranker_client.logger.name):
        _rerank(lambda r, n: httpx.Response(503))
    assert "mục 9.1" in caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"scores": "abc"},
        {"scores": [1.0]},
        {"scores": [1.0, 2.0, 3.0]},
        {"scores": [1.0, None]},
        {"scores": [1.0, "2"]},
        {"scores": [1.0, True]},
        {"scores": [1.0, float("nan")]},
        {"scores": [1.0, float("inf")]},
        [1.0, 2.0],
    ],
)
def test_rerank_response_khong_hop_le_tra_none_khong_retry(env: None, payload: Any):
    body = json.dumps(payload, allow_nan=True).encode()
    scores, calls = _rerank(lambda r, n: httpx.Response(200, content=body))
    assert scores is None
    assert calls == [1]


def test_rerank_response_khong_phai_json_tra_none(env: None):
    scores, calls = _rerank(lambda r, n: httpx.Response(200, content=b"<html>"))
    assert scores is None
    assert calls == [1]


# ---------------------------------------------------------------- hyde


class _FakeGroq:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.prompts: list[str] = []
        self.calls: list[dict[str, Any]] = []
        self._content = content
        self._error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        self.prompts.append(kwargs["messages"][-1]["content"])
        if self._error:
            raise self._error
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _hyde(fake: _FakeGroq, query: str = "hỏi") -> str | None:
    generator = HydeGenerator(LLMSettings(), fake)  # type: ignore[arg-type]
    return asyncio.run(generator.generate(query))


def test_hyde_tra_ve_van_ban_da_strip(env: None):
    assert _hyde(_FakeGroq("  đoạn văn \n")) == "đoạn văn"


@pytest.mark.parametrize("content", ["", "   \n", None])
def test_hyde_output_rong_coi_nhu_loi(env: None, content: str | None):
    assert _hyde(_FakeGroq(content)) is None


def test_hyde_groq_loi_tra_none(env: None):
    assert _hyde(_FakeGroq(error=RuntimeError("boom"))) is None


def test_hyde_system_prompt_chua_cac_quy_tac_cam(env: None):
    for phrase in (
        "TUYỆT ĐỐI không nêu số Điều, Khoản, Điểm, Chương, Mục",
        "tên hay số hiệu văn",
        "năm ban hành",
        "Không markdown",
    ):
        assert phrase.replace("\n", " ") in " ".join(HYDE_SYSTEM_PROMPT.split())


def test_hyde_messages_system_roi_user_chua_cau_hoi(env: None):
    fake = _FakeGroq("x")
    _hyde(fake, "Nghỉ phép mấy ngày?")
    messages = fake.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == HYDE_SYSTEM_PROMPT
    assert messages[1]["content"] == "Câu hỏi: Nghỉ phép mấy ngày?"
    assert "Nghỉ phép" not in messages[0]["content"]


def test_hyde_truyen_dung_tham_so_groq(env: None):
    fake = _FakeGroq("x")
    _hyde(fake)
    call = fake.calls[0]
    assert call["reasoning_effort"] == "low"
    assert call["temperature"] == 0.2
    assert call["max_completion_tokens"] == 2048
    assert call["model"] == LLMSettings().model_name


def test_hyde_query_chua_ngoac_nhon_khong_hong(env: None):
    fake = _FakeGroq("kết quả")
    query = "Điều {36} khoản {0} và {query} }{ quy định gì?"
    assert _hyde(fake, query) == "kết quả"
    assert fake.calls[0]["messages"][1]["content"] == HYDE_USER_TEMPLATE.format(
        query=query
    )


# ---------------------------------------------------------------- embedder


class _FakeHF:
    def __init__(self, failures: int = 0, response: Any = None):
        self.calls: list[Any] = []
        self._failures = failures
        self._response = response

    def feature_extraction(self, texts: Any) -> Any:
        self.calls.append(texts)
        if len(self.calls) <= self._failures:
            raise ConnectionError("hf")
        if self._response is not None:
            return self._response
        return [[float(i), 1.0] for i in range(len(texts))]


def _embed(fake: _FakeHF, texts: list[str]) -> list[list[float]]:
    embedder = QueryEmbedder(EmbeddingSettings(), fake)  # type: ignore[arg-type]
    return asyncio.run(embedder.embed(texts))


def test_embed_goi_pyvi_segment_va_1_request_cho_nhieu_text(
    env: None, monkeypatch: pytest.MonkeyPatch
):
    tokenized: list[str] = []

    def fake_tokenize(text: str) -> str:
        tokenized.append(text)
        return f"SEG({text})"

    monkeypatch.setattr(query_embedder.ViTokenizer, "tokenize", fake_tokenize)
    fake = _FakeHF()
    result = _embed(fake, ["a b", "c d"])

    assert tokenized == ["a b", "c d"]
    assert len(fake.calls) == 1
    assert fake.calls[0] == ["SEG(a b)", "SEG(c d)"]
    assert result == [[0.0, 1.0], [1.0, 1.0]]


def test_embed_retry_roi_thanh_cong(env: None):
    fake = _FakeHF(failures=2)
    assert len(_embed(fake, ["a"])) == 1
    assert len(fake.calls) == 3


def test_embed_het_retry_raise_retrieval_error(env: None):
    fake = _FakeHF(failures=99)
    with pytest.raises(RetrievalError):
        _embed(fake, ["a"])
    assert len(fake.calls) == 3


def test_embed_response_sai_so_luong_raise(env: None):
    with pytest.raises(RetrievalError):
        _embed(_FakeHF(response=[[1.0]]), ["a", "b"])


# ---------------------------------------------------------------- sparse index


class _FakeSparseIndex:
    def __init__(self, delete_error: Exception | None = None):
        self.delete_error = delete_error
        self.deleted = False
        self.upserts: list[list[dict[str, Any]]] = []
        self.queries: list[dict[str, Any]] = []

    def delete(self, *, delete_all: bool) -> None:
        self.deleted = delete_all
        if self.delete_error:
            raise self.delete_error

    def upsert(self, *, vectors: list[dict[str, Any]]) -> None:
        self.upserts.append(vectors)

    def query(self, **kwargs: Any) -> Any:
        self.queries.append(kwargs)
        return SimpleNamespace(matches=[SimpleNamespace(id="c1", score=0.5)])


def test_delete_all_bo_qua_namespace_not_found():
    index = _FakeSparseIndex(NotFoundException("Namespace not found"))
    sparse_index._delete_all_vectors(index)  # không raise
    assert index.deleted is True


def test_delete_all_nem_lai_not_found_khac():
    index = _FakeSparseIndex(NotFoundException("Index not found"))
    with pytest.raises(NotFoundException):
        sparse_index._delete_all_vectors(index)


def test_build_index_upsert_theo_batch_sau_khi_xoa(env: None, tmp_path: Path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    total = sparse_index.SPARSE_UPSERT_BATCH_SIZE * 2 + 5
    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            source_document="d",
            breadcrumb="Điều 1",
            content=f"người lao động {i}",
            token_count=5,
        ).model_dump()
        for i in range(total)
    ]
    (chunks_dir / "a.json").write_text(json.dumps(chunks), encoding="utf-8")
    index = _FakeSparseIndex(NotFoundException("Namespace not found"))
    client = SimpleNamespace(list_indexes=lambda: ["sparse"], Index=lambda name: index)

    count = build_index(
        chunks_dir,
        tmp_path / "bm25.json",
        VectorDBSettings(),
        client,  # type: ignore[arg-type]
    )

    assert count == total
    assert [len(b) for b in index.upserts] == [100, 100, 5]
    assert index.upserts[0][0]["id"] == "c0"
    assert set(index.upserts[0][0]["sparse_values"]) == {"indices", "values"}
    assert (tmp_path / "bm25.json").exists()


def test_build_index_ghi_params_version_3_va_token_cau_truc(env: None, tmp_path: Path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    chunk = Chunk(
        chunk_id="c1",
        source_document="d",
        breadcrumb="LUẬT X - Điều 3. Tên - Khoản 1",
        content="người lao động",
        token_count=5,
    )
    (chunks_dir / "a.json").write_text(
        json.dumps([chunk.model_dump()]), encoding="utf-8"
    )
    index = _FakeSparseIndex()
    client = SimpleNamespace(list_indexes=lambda: ["sparse"], Index=lambda name: index)
    params_path = tmp_path / "bm25.json"

    build_index(chunks_dir, params_path, VectorDBSettings(), client)  # type: ignore[arg-type]

    params = json.loads(params_path.read_text(encoding="utf-8"))
    assert params["params_version"] == 3
    assert {"điều_3", "khoản_1", "điều_3_khoản_1"} <= set(params["vocab"])
    indices = index.upserts[0][0]["sparse_values"]["indices"]
    assert params["vocab"]["điều_3_khoản_1"] in indices


def test_build_index_chunks_dir_rong_raise_value_error(env: None, tmp_path: Path):
    with pytest.raises(ValueError):
        build_index(tmp_path, tmp_path / "p.json", VectorDBSettings(), object())  # type: ignore[arg-type]


def _encoder() -> BM25Encoder:
    encoder = BM25Encoder()
    encoder.fit(["người lao động", "tiền lương"])
    return encoder


def test_sparse_query_vector_rong_khong_goi_pinecone():
    index = _FakeSparseIndex()
    result = asyncio.run(SparseIndex(_encoder(), index=index).query("zzzz qqqq"))
    assert result == []
    assert index.queries == []


def test_sparse_query_tra_hits():
    index = _FakeSparseIndex()
    result = asyncio.run(
        SparseIndex(_encoder(), index=index).query("tiền lương", top_k=7)
    )
    assert [(h.chunk_id, h.score) for h in result] == [("c1", 0.5)]
    assert index.queries[0]["top_k"] == 7
    assert index.queries[0]["sparse_vector"]["indices"]


def test_sparse_query_pinecone_loi_raise_retrieval_error():
    class Broken(_FakeSparseIndex):
        def query(self, **kwargs: Any) -> Any:
            raise RuntimeError("x")

    with pytest.raises(RetrievalError):
        asyncio.run(SparseIndex(_encoder(), index=Broken()).query("tiền lương"))


# ---------------------------------------------------------------- dense search

META = {
    "content": "nd",
    "breadcrumb": "bc",
    "source_document": "sd",
    "has_table": False,
}


class _FakeDense:
    def __init__(self, fetchable: dict[str, dict[str, Any]] | None = None):
        self.queries: list[dict[str, Any]] = []
        self.fetches: list[list[str]] = []
        self._fetchable = fetchable or {}

    def query(self, **kwargs: Any) -> Any:
        self.queries.append(kwargs)
        match = SimpleNamespace(id="a", score=0.9, metadata=META, values=[1.0, 0.0])
        return SimpleNamespace(matches=[match])

    def fetch(self, *, ids: list[str]) -> Any:
        self.fetches.append(ids)
        return SimpleNamespace(
            vectors={
                i: SimpleNamespace(metadata=META, values=self._fetchable[i]["values"])
                for i in ids
                if i in self._fetchable
            }
        )


@pytest.mark.parametrize("flag", [True, False])
def test_dense_query_include_values_theo_co(flag: bool):
    index = _FakeDense()
    hits = asyncio.run(DenseSearch(index=index).query([1.0, 0.0], include_values=flag))
    assert index.queries[0]["include_values"] is flag
    assert index.queries[0]["include_metadata"] is True
    assert hits[0].values == ([1.0, 0.0] if flag else None)
    assert isinstance(hits[0].metadata, PineconeMetadata)


def test_dense_query_loi_raise_retrieval_error():
    class Broken(_FakeDense):
        def query(self, **kwargs: Any) -> Any:
            raise RuntimeError("x")

    with pytest.raises(RetrievalError):
        asyncio.run(DenseSearch(index=Broken()).query([1.0]))


def _cand(chunk_id: str, *, meta: bool, values: bool) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        rrf_score=1.0,
        metadata=PineconeMetadata(**META) if meta else None,  # type: ignore[arg-type]
        values=[1.0] if values else None,
    )


def test_fill_missing_khong_fetch_khi_khong_thieu():
    index = _FakeDense()
    cands = [_cand("a", meta=True, values=False)]
    out = asyncio.run(DenseSearch(index=index).fill_missing(cands, need_values=False))
    assert out == cands
    assert index.fetches == []


def test_fill_missing_can_values_fetch_khi_thieu_vector():
    index = _FakeDense({"a": {"values": [3.0]}})
    cands = [_cand("a", meta=True, values=False)]
    out = asyncio.run(DenseSearch(index=index).fill_missing(cands, need_values=True))
    assert index.fetches == [["a"]]
    assert out[0].values == [3.0]


def test_fill_missing_giu_thu_tu_va_bo_chunk_khong_co_o_dense():
    index = _FakeDense({"b": {"values": [2.0]}, "d": {"values": [4.0]}})
    cands = [
        _cand("a", meta=True, values=True),
        _cand("b", meta=False, values=False),
        _cand("c", meta=False, values=False),  # không có ở dense
        _cand("d", meta=False, values=False),
    ]
    out = asyncio.run(DenseSearch(index=index).fill_missing(cands, need_values=False))
    assert index.fetches == [["b", "c", "d"]]  # đúng 1 lượt fetch
    assert [c.chunk_id for c in out] == ["a", "b", "d"]
    assert out[1].metadata is not None


def test_fetch_loi_raise_retrieval_error():
    class Broken(_FakeDense):
        def fetch(self, *, ids: list[str]) -> Any:
            raise RuntimeError("x")

    with pytest.raises(RetrievalError):
        asyncio.run(DenseSearch(index=Broken()).fetch(["a"]))


# ---------------------------------------------------------------- CLI


def test_cli_sparse_exit_code_0(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from tools import sparse_index_documents as cli

    monkeypatch.setattr(cli, "build_index", lambda _c, _p: 3)
    result = CliRunner().invoke(cli.app, ["--chunks-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "3" in result.output


def test_cli_sparse_exit_code_1_khi_loi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from tools import sparse_index_documents as cli

    def boom(_c: Path, _p: Path) -> int:
        raise ValueError("không có chunk")

    monkeypatch.setattr(cli, "build_index", boom)
    result = CliRunner().invoke(cli.app, ["--chunks-dir", str(tmp_path)])
    assert result.exit_code == 1


# ---------------------------------------------------------------- HyDE vs spec


def test_hyde_prompt_enforces_current_semantic_retrieval_contract():
    """HyDE chỉ tăng semantic recall; không được trở thành nguồn citation."""
    normalized = " ".join(HYDE_SYSTEM_PROMPT.split())

    assert "chỉ dùng để tìm kiếm điều luật tương tự" in normalized
    assert "không phải câu trả lời cho người dùng" in normalized
    assert "văn phong văn bản quy phạm pháp luật" in normalized
    assert "TUYỆT ĐỐI không nêu số Điều, Khoản, Điểm" in normalized
    assert "tên hay số hiệu văn bản, năm ban hành" in normalized
    assert "Không nêu con số, mức tiền, tỉ lệ, thời hạn cụ thể" in normalized
    assert HYDE_USER_TEMPLATE == "Câu hỏi: {query}"


def test_hyde_system_prompt_khong_co_placeholder_format():
    # System prompt dùng nguyên văn (không .format) nên không được chứa {query}.
    assert "{query}" not in HYDE_SYSTEM_PROMPT
    assert HYDE_USER_TEMPLATE.count("{query}") == 1


def test_rerank_timeout_truyen_dung_o_moi_lan_retry_connect_error(env: None):
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request, n: int) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        if n < 3:
            raise httpx.ConnectError("x")
        return httpx.Response(200, json={"scores": [1.0] * 36})

    scores, calls = _rerank(handler, [f"p{i}" for i in range(36)])
    assert calls == [3]
    assert scores == [1.0] * 36
    assert [t["read"] for t in seen] == [90.0, 90.0, 90.0]
    assert [t["connect"] for t in seen] == [5.0, 5.0, 5.0]


def test_rerank_khong_passage_khong_goi_request_va_khong_loi(env: None):
    calls = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        return httpx.Response(500)

    async def run() -> Any:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await RerankerClient(RerankerSettings(), http).rerank("q", [])

    assert asyncio.run(run()) == []
    assert calls == [0]


def test_rerank_read_timeout_log_khong_lo_noi_dung_passage_hay_secret(
    env: None, caplog: pytest.LogCaptureFixture
):
    def handler(request: httpx.Request, n: int) -> httpx.Response:
        raise httpx.ReadTimeout("x")

    passages = ["NOI-DUNG-BI-MAT-1", "NOI-DUNG-BI-MAT-2"]
    with caplog.at_level("DEBUG", logger=reranker_client.logger.name):
        _rerank(handler, passages)
    assert "NOI-DUNG-BI-MAT" not in caplog.text
    assert "http://rerank.test" not in caplog.text
    assert "X-API-Key" not in caplog.text
