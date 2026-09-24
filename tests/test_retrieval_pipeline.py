"""Test RetrievalPipeline.retrieve với các thành phần giả (mục 6.4, 10, 13B)."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from production_legal_qa_rag.embedding.models import PineconeMetadata
from production_legal_qa_rag.retrieval import pipeline as pipeline_module
from production_legal_qa_rag.retrieval.models import RetrievalError, SearchHit
from production_legal_qa_rag.retrieval.pipeline import (
    FINAL_TOP_K,
    RetrievalPipeline,
)

DIM = 4


def _meta(chunk_id: str) -> PineconeMetadata:
    return PineconeMetadata(
        content=f"nd-{chunk_id}",
        breadcrumb=f"bc-{chunk_id}",
        source_document="sd",
        has_table=False,
    )


def _vec(chunk_id: str) -> list[float]:
    n = int(chunk_id[1:])
    return [1.0, float(n % 5), float(n % 3), 0.1 * n]


class FakeHyde:
    def __init__(self, text: str | None = "giả định"):
        self.text = text

    async def generate(self, query: str) -> str | None:
        return self.text


class FakeEmbedder:
    def __init__(self, error: Exception | None = None):
        self.calls: list[list[str]] = []
        self.error = error

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.error:
            raise self.error
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class FakeDense:
    """Dense trả `dense_ids` (kèm metadata); fetch trả mọi id đã biết."""

    def __init__(self, dense_ids: list[str], known: set[str] | None = None):
        self.dense_ids = dense_ids
        self.known = known if known is not None else set(dense_ids)
        self.include_values_calls: list[bool] = []
        self.fetch_calls: list[list[str]] = []
        self.error: Exception | None = None

    async def query(
        self, embedding: Any, top_k: int = 20, *, include_values: bool = False
    ) -> list[SearchHit]:
        if self.error:
            raise self.error
        self.include_values_calls.append(include_values)
        return [
            SearchHit(
                chunk_id=i,
                score=1.0,
                metadata=_meta(i),
                values=_vec(i) if include_values else None,
            )
            for i in self.dense_ids
        ]

    async def fetch(self, chunk_ids: list[str]) -> dict[str, SearchHit]:
        self.fetch_calls.append(list(chunk_ids))
        return {
            i: SearchHit(chunk_id=i, metadata=_meta(i), values=_vec(i))
            for i in chunk_ids
            if i in self.known
        }

    async def fill_missing(self, candidates: Any, *, need_values: bool) -> Any:
        from production_legal_qa_rag.retrieval.dense_search import DenseSearch

        return await DenseSearch.fill_missing(
            self,  # type: ignore[arg-type]
            candidates,
            need_values=need_values,
        )


class FakeSparse:
    def __init__(self, by_text: dict[str, list[str]], error: Exception | None = None):
        self.by_text = by_text
        self.error = error
        self.texts: list[str] = []
        self.terms: dict[str, list[str]] = {}

    async def query(
        self, text: str, top_k: int = 20, extra_terms: Any = ()
    ) -> list[SearchHit]:
        if self.error:
            raise self.error
        self.texts.append(text)
        self.terms[text] = list(extra_terms)
        return [SearchHit(chunk_id=i, score=1.0) for i in self.by_text.get(text, [])]


class FakeReranker:
    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(self, query: str, passages: list[str]) -> list[float] | None:
        self.calls.append((query, passages))
        if self.mode == "fail":
            return None
        return [float(len(passages) - i) for i in range(len(passages))]


def _build(
    dense_ids: list[str],
    sparse: dict[str, list[str]],
    *,
    hyde: str | None = "giả định",
    reranker: str = "ok",
    known: set[str] | None = None,
) -> tuple[RetrievalPipeline, FakeDense, FakeReranker, FakeEmbedder]:
    dense = FakeDense(dense_ids, known)
    embedder = FakeEmbedder()
    rr = FakeReranker(reranker)
    pipe = RetrievalPipeline(
        hyde=FakeHyde(hyde),  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
        dense_search=dense,  # type: ignore[arg-type]
        sparse_index=FakeSparse(sparse),  # type: ignore[arg-type]
        reranker=rr,  # type: ignore[arg-type]
    )
    return pipe, dense, rr, embedder


MANY = [f"c{i}" for i in range(1, 21)]


def test_default_pipeline_khoi_tao_o_worker_va_chi_mot_lan_khi_dong_thoi(
    monkeypatch: pytest.MonkeyPatch,
):
    constructor_started = threading.Event()
    allow_constructor_to_finish = threading.Event()
    construction_threads: list[int] = []

    class FakeDefaultPipeline:
        def __init__(self) -> None:
            construction_threads.append(threading.get_ident())
            constructor_started.set()
            assert allow_constructor_to_finish.wait(timeout=1)

        async def retrieve(
            self, query: str, *, use_mmr: bool | None = None
        ) -> list[object]:
            return []

    monkeypatch.setattr(pipeline_module, "RetrievalPipeline", FakeDefaultPipeline)
    monkeypatch.setattr(pipeline_module, "_default_pipeline", None)

    async def retrieve_concurrently() -> list[list[object]]:
        requests = [
            asyncio.create_task(pipeline_module.retrieve(f"câu hỏi {index}"))
            for index in range(2)
        ]
        assert await asyncio.to_thread(constructor_started.wait, 1)
        allow_constructor_to_finish.set()
        return await asyncio.gather(*requests)

    assert asyncio.run(retrieve_concurrently()) == [[], []]
    assert len(construction_threads) == 1
    assert construction_threads[0] != threading.get_ident()


@pytest.mark.parametrize("use_mmr", [True, False])
def test_toi_da_5_chunk_va_khong_trung(use_mmr: bool):
    pipe, _, _, _ = _build(MANY, {"giả định": MANY[5:], "hỏi": MANY[:8]})
    result = asyncio.run(pipe.retrieve("hỏi", use_mmr=use_mmr))
    ids = [c.chunk_id for c in result]
    assert len(ids) == FINAL_TOP_K
    assert len(set(ids)) == len(ids)
    scores = [c.rerank_score for c in result]
    assert None not in scores
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]


def test_use_mmr_none_dung_hang_so_mac_dinh(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pipeline_module, "USE_MMR", False)
    pipe, dense, _, _ = _build(["c1"], {})
    asyncio.run(pipe.retrieve("hỏi"))
    assert dense.include_values_calls == [False, False]


def test_embed_1_request_cho_hyde_va_query_va_rerank_dung_cau_hoi_goc():
    pipe, _, rr, embedder = _build(["c1", "c2"], {})
    asyncio.run(pipe.retrieve("hỏi"))
    assert embedder.calls == [["giả định", "hỏi"]]
    assert rr.calls[0][0] == "hỏi"
    assert rr.calls[0][1] == ["bc-c1\nnd-c1", "bc-c2\nnd-c2"]


def test_mmr_bat_toi_da_2_luot_fetch_1_moi_nhanh():
    # c30, c31 chỉ có ở sparse của từng nhánh -> mỗi nhánh fetch 1 lần
    pipe, dense, _, _ = _build(
        ["c1", "c2"],
        {"giả định": ["c30"], "hỏi": ["c31"]},
        known={"c1", "c2", "c30", "c31"},
    )
    asyncio.run(pipe.retrieve("hỏi", use_mmr=True))
    assert sorted(dense.fetch_calls) == [["c30"], ["c31"]]
    assert dense.include_values_calls == [True, True]


def test_mmr_bat_khong_thieu_id_thi_khong_fetch():
    pipe, dense, _, _ = _build(["c1", "c2"], {"giả định": ["c1"], "hỏi": ["c2"]})
    asyncio.run(pipe.retrieve("hỏi", use_mmr=True))
    assert dense.fetch_calls == []


def test_mmr_tat_toi_da_1_luot_fetch_cho_ca_union():
    pipe, dense, _, _ = _build(
        ["c1", "c2"],
        {"giả định": ["c30"], "hỏi": ["c31"]},
        known={"c1", "c2", "c30", "c31"},
    )
    result = asyncio.run(pipe.retrieve("hỏi", use_mmr=False))
    assert len(dense.fetch_calls) == 1
    assert sorted(dense.fetch_calls[0]) == ["c30", "c31"]
    assert dense.include_values_calls == [False, False]
    assert {"c30", "c31"} <= {c.chunk_id for c in result}


def test_mmr_tat_khong_thieu_id_thi_khong_fetch():
    pipe, dense, _, _ = _build(["c1", "c2"], {"hỏi": ["c1"]})
    asyncio.run(pipe.retrieve("hỏi", use_mmr=False))
    assert dense.fetch_calls == []


def test_chunk_sparse_khong_co_o_dense_bi_bo_ca_hai_che_do():
    for use_mmr in (True, False):
        pipe, _, rr, _ = _build(["c1"], {"hỏi": ["c99"]}, known={"c1"})
        result = asyncio.run(pipe.retrieve("hỏi", use_mmr=use_mmr))
        assert [c.chunk_id for c in result] == ["c1"]
        assert "bc-c99\nnd-c99" not in rr.calls[0][1]


def test_groq_loi_chi_chay_nhanh_b():
    pipe, dense, _, embedder = _build(
        ["c1", "c2"], {"giả định": ["c3"], "hỏi": ["c2"]}, hyde=None
    )
    result = asyncio.run(pipe.retrieve("hỏi"))
    assert embedder.calls == [["hỏi"]]
    assert len(dense.include_values_calls) == 1  # 1 nhánh
    assert {c.chunk_id for c in result} == {"c1", "c2"}


def test_union_dedupe_theo_chunk_id_truoc_khi_rerank():
    pipe, _, rr, _ = _build(["c1", "c2"], {"giả định": ["c1"], "hỏi": ["c2"]})
    asyncio.run(pipe.retrieve("hỏi"))
    passages = rr.calls[0][1]
    assert sorted(passages) == ["bc-c1\nnd-c1", "bc-c2\nnd-c2"]


def test_reranker_loi_fallback_xen_ke_a_b_voi_score_none_mmr_bat():
    _fallback_case(use_mmr=True)


def test_reranker_loi_fallback_xen_ke_mmr_tat_chunk_chi_co_o_sparse():
    _fallback_case(use_mmr=False)


def _fallback_case(*, use_mmr: bool) -> None:
    pipe, _, _, _ = _build(MANY, {}, reranker="fail")
    # Ghi đè dense để nhánh A và B trả thứ tự khác nhau, kiểm tra xen kẽ.
    orders = {"giả định": ["c1", "c2", "c3"], "hỏi": ["c9", "c8", "c7"]}

    async def scripted(
        text: str, top_k: int = 20, extra_terms: Any = ()
    ) -> list[SearchHit]:
        return [SearchHit(chunk_id=i, score=1.0) for i in orders[text]]

    pipe._sparse_index.query = scripted  # type: ignore[method-assign]
    dense_only_empty = FakeDense([])
    pipe._dense_search = dense_only_empty  # type: ignore[assignment]
    dense_only_empty.known = {i for ids in orders.values() for i in ids}

    result = asyncio.run(pipe.retrieve("hỏi", use_mmr=use_mmr))
    ids = [c.chunk_id for c in result]
    if use_mmr:  # MMR đổi thứ tự trong nhánh, chỉ kiểm tra thành phần
        assert len(ids) == FINAL_TOP_K
        assert set(ids) <= {"c1", "c2", "c3", "c9", "c8", "c7"}
        assert set(ids) & {"c1", "c2", "c3"} and set(ids) & {"c9", "c8", "c7"}
    else:
        assert ids == ["c1", "c9", "c2", "c8", "c3"]
    assert all(c.rerank_score is None for c in result)


def test_union_rong_tra_list_rong_khong_goi_reranker():
    pipe, _, rr, _ = _build([], {})
    assert asyncio.run(pipe.retrieve("hỏi")) == []
    assert rr.calls == []


def test_hf_loi_raise_retrieval_error():
    pipe, _, _, embedder = _build(["c1"], {})
    embedder.error = RetrievalError("hf")
    with pytest.raises(RetrievalError):
        asyncio.run(pipe.retrieve("hỏi"))


def test_pinecone_dense_loi_raise_retrieval_error():
    pipe, dense, _, _ = _build(["c1"], {})
    dense.error = RetrievalError("pinecone")
    with pytest.raises(RetrievalError):
        asyncio.run(pipe.retrieve("hỏi"))


def test_pinecone_sparse_loi_raise_retrieval_error():
    pipe, _, _, _ = _build(["c1"], {})
    pipe._sparse_index = FakeSparse({}, RetrievalError("pinecone"))  # type: ignore[assignment]
    with pytest.raises(RetrievalError):
        asyncio.run(pipe.retrieve("hỏi"))


@pytest.mark.parametrize("use_mmr", [True, False])
def test_passage_rerank_bat_dau_bang_breadcrumb_roi_xuong_dong_roi_content(
    use_mmr: bool,
):
    pipe, _, rr, _ = _build(MANY, {"giả định": MANY[5:], "hỏi": MANY[:8]})
    asyncio.run(pipe.retrieve("hỏi", use_mmr=use_mmr))
    passages = rr.calls[0][1]
    assert passages
    for passage in passages:
        breadcrumb, content = passage.split("\n", 1)
        assert breadcrumb.startswith("bc-c")
        assert content == breadcrumb.replace("bc-", "nd-")


def test_build_rerank_passages_giu_thu_tu_union():
    from production_legal_qa_rag.retrieval.models import Candidate
    from production_legal_qa_rag.retrieval.pipeline import build_rerank_passages

    union = [
        Candidate(chunk_id=i, rrf_score=1.0, metadata=_meta(i)) for i in ("c3", "c1")
    ]
    assert build_rerank_passages(union) == ["bc-c3\nnd-c3", "bc-c1\nnd-c1"]


def test_retrieved_chunk_content_khong_chua_breadcrumb_va_diem_cao_dung_chunk():
    pipe, _, rr, _ = _build(["c1", "c2", "c3"], {})
    result = asyncio.run(pipe.retrieve("hỏi", use_mmr=False))
    for chunk in result:
        assert chunk.content == f"nd-{chunk.chunk_id}"
        assert "bc-" not in chunk.content
        assert chunk.breadcrumb == f"bc-{chunk.chunk_id}"
    # FakeReranker chấm giảm dần theo thứ tự passage: passage đầu điểm cao nhất.
    first_passage = rr.calls[0][1][0]
    assert result[0].rerank_score == float(len(rr.calls[0][1]))
    assert first_passage == f"bc-{result[0].chunk_id}\nnd-{result[0].chunk_id}"


@pytest.mark.parametrize(
    ("breadcrumb", "content"),
    [
        ("", "nd"),
        ("Luật > Điều 25 > Khoản 1\n(tiếp)", "nd\n\nxuống dòng"),
        ("bc {x} % \\n <b>", "{0} nd \u200b"),
    ],
)
def test_build_rerank_passages_khong_hong_voi_breadcrumb_dac_biet(
    breadcrumb: str, content: str
):
    from production_legal_qa_rag.retrieval.models import Candidate
    from production_legal_qa_rag.retrieval.pipeline import build_rerank_passages

    metadata = PineconeMetadata(
        content=content,
        breadcrumb=breadcrumb,
        source_document="sd",
        has_table=False,
    )
    union = [Candidate(chunk_id="c1", rrf_score=1.0, metadata=metadata)]
    # Ghép đúng nguyên văn, không strip/format: breadcrumb + "\n" + content.
    assert build_rerank_passages(union) == [f"{breadcrumb}\n{content}"]


def test_build_rerank_passages_rong_va_thieu_metadata():
    from production_legal_qa_rag.retrieval.models import Candidate
    from production_legal_qa_rag.retrieval.pipeline import build_rerank_passages

    assert build_rerank_passages([]) == []
    with pytest.raises(RetrievalError):
        build_rerank_passages([Candidate(chunk_id="c1", rrf_score=1.0)])
