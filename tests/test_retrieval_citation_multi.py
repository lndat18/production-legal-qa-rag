"""Test nhận diện số Điều và extras cho câu viện dẫn (retrieval_spec.md mục 8.1)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from test_retrieval_pipeline import (
    FakeDense,
    FakeEmbedder,
    FakeHyde,
    FakeReranker,
    FakeSparse,
)

from production_legal_qa_rag.retrieval import pipeline as pipeline_module
from production_legal_qa_rag.retrieval.citation import (
    CITATION_SPARSE_TOP_K,
    citation_extras,
    extract_citation_numbers,
    has_citation,
)
from production_legal_qa_rag.retrieval.models import SearchHit

# ------------------------------------------------------------- nhận diện


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Điều 36 khoản 2 quy định gì", [36]),
        ("khoản 2 điều 36 quy định gì", [36]),
        ("điều36 quy định gì", [36]),
        ("Điều thứ 5 quy định gì", [5]),
        ("Điều 36.2 quy định gì", [36]),
        ("Điều 3, 5 và 7 quy định gì", [3, 5, 7]),
        ("các Điều 3, 5", [3, 5]),
        ("Điều 3 hoặc 5", [3, 5]),
        ("Điều 3 đến Điều 5", [3, 5]),
        ("Điều 3 tới 9", [3, 9]),
        ("Điều 3 và Điều 3", [3]),
        ("Điều 3 khoản 1 và Điều 5 khoản 2", [3, 5]),
        (
            "Điều 36 người lao động được quyền gì",
            [36],
        ),  # "người" không là đơn vị loại trừ
        ("Điều 3 và 5 người lao động", [3, 5]),
    ],
)
def test_extract_citation_numbers_duong(query: str, expected: list[int]):
    assert extract_citation_numbers(query) == expected
    assert has_citation(query)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Điều 3 và 5 tháng", [3]),
        ("Điều 3, 5 ngày", [3]),
        ("Điều 3 và 12 %", [3]),
        ("Điều 3 và 5.000 đồng", [3]),
        ("Điều 3 và 2024", [3]),
        ("điều 3, điều kiện lao động", [3]),
    ],
)
def test_extract_citation_numbers_so_noi_la_dai_luong(query: str, expected: list[int]):
    assert extract_citation_numbers(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "điều 5 tháng",
        "Điều 3 ngày",
        "điều 2 lần",
        "Điều 10 %",
        "Điều 36 năm 2020",
        "điều kiện lao động là gì",
        "điều khoản hợp đồng",
        "điều hành công ty",
        "trong 3 điều kiện sau",
        "chiều 5 tôi đi làm",
        "Đ.3 quy định gì",
        "Đ3 quy định gì",
        "đ 3 quy định gì",
        "Điều II quy định gì",
        "Điều V quy định gì",
        "Điều ba quy định gì",
        "Điều thứ hai",
        "điều 2024",
        "khoản 2 quy định gì",
        "người lao động nghỉ phép",
        "",
    ],
)
def test_extract_citation_numbers_am(query: str):
    assert extract_citation_numbers(query) == []
    assert not has_citation(query)


def test_extract_citation_numbers_khong_tran_nhieu_dieu():
    # Câu nhiều Điều nằm ngoài phạm vi tối ưu (mục 1): không còn trần 3 Điều.
    assert extract_citation_numbers("Điều 1, 2, 3, 4 và Điều 5") == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "query",
    [
        "Điều " + "3, " * 5000 + "quy định gì",
        "điều" * 20000,
        "Điều 3 " + "và " * 20000,
        "Điều 3 " + "và 5 " * 20000,
        "điều thứ " * 20000,
        ("Điều 1" + "0" * 50 + " ") * 2000,
        "Điều 3" + "," * 50000,
    ],
)
def test_regex_khong_bung_no_voi_chuoi_rat_dai(query: str):
    import time

    start = time.perf_counter()
    extract_citation_numbers(query)
    assert time.perf_counter() - start < 2.0


# ------------------------------------------------------------- citation_extras


def _hits(prefix: str, count: int = 20) -> list[SearchHit]:
    return [SearchHit(chunk_id=f"{prefix}{i}", score=1.0) for i in range(1, count + 1)]


def _ids(candidates: list) -> list[str]:  # type: ignore[type-arg]
    return [c.chunk_id for c in candidates]


def test_extras_dung_k_hit_dau_sparse_tho_nhanh_b():
    extras = citation_extras(_hits("b"))
    assert _ids(extras) == [f"b{i}" for i in range(1, CITATION_SPARSE_TOP_K + 1)]


def test_extras_dedupe_va_rong():
    dup = [SearchHit(chunk_id="s1"), SearchHit(chunk_id="s1"), SearchHit(chunk_id="s2")]
    assert _ids(citation_extras(dup)) == ["s1", "s2"]
    assert citation_extras([]) == []


def test_extras_khong_con_tham_so_numbers_hay_article_hits():
    import inspect

    assert list(inspect.signature(citation_extras).parameters) == [
        "branch_b_sparse_hits"
    ]


# ------------------------------------------------------------- pipeline

DENSE = [f"c{i}" for i in range(1, 21)]
ONE = "Điều 3 khoản 1 quy định gì"
THREE = "Điều 3, 5 và 7 quy định gì"
MAX_UNION = 2 * 3 + CITATION_SPARSE_TOP_K  # BRANCH_TOP_N thu nhỏ = 3 trong test


class SparseByText(FakeSparse):
    """Ghi lại (text, top_k) mỗi lượt query."""

    def __init__(self, by_text: dict[str, list[str]]):
        super().__init__(by_text)
        self.top_ks: dict[str, int] = {}

    async def query(
        self, text: str, top_k: int = 20, extra_terms: Any = ()
    ) -> list[SearchHit]:
        self.texts.append(text)
        self.terms[text] = list(extra_terms)
        self.top_ks[text] = top_k
        return [SearchHit(chunk_id=i, score=1.0) for i in self.by_text.get(text, [])]


def _ids_of(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(1, count + 1)]


def _pipe(
    query: str,
    *,
    hyde: str | None = "giả định",
    reranker: str = "ok",
) -> tuple[pipeline_module.RetrievalPipeline, SparseByText, FakeReranker]:
    by_text = {query: _ids_of("b", 12)}
    known = set(DENSE) | set(by_text[query])
    sparse = SparseByText(by_text)
    rr = FakeReranker(reranker)
    pipe = pipeline_module.RetrievalPipeline(
        hyde=FakeHyde(hyde),  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        dense_search=FakeDense(DENSE, known),  # type: ignore[arg-type]
        sparse_index=sparse,  # type: ignore[arg-type]
        reranker=rr,  # type: ignore[arg-type]
    )
    return pipe, sparse, rr


@pytest.fixture(autouse=True)
def small_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "BRANCH_TOP_N", 3)


def _passage_ids(rr: FakeReranker) -> set[str]:
    return {p.split("\n")[0].removeprefix("bc-") for p in rr.calls[0][1]}


@pytest.mark.parametrize("use_mmr", [True, False])
@pytest.mark.parametrize("query", [ONE, THREE])
def test_moi_so_dieu_dung_2_luot_sparse_extras_la_10_hit_dau_nhanh_b(
    query: str, use_mmr: bool
):
    pipe, sparse, rr = _pipe(query)
    asyncio.run(pipe.retrieve(query, use_mmr=use_mmr))
    assert len(sparse.texts) == 2  # nhánh A + nhánh B, không có lượt phụ
    assert sparse.texts.count(query) == 1
    ids = _passage_ids(rr)
    assert set(_ids_of("b", CITATION_SPARSE_TOP_K)) <= ids
    assert "b11" not in ids
    assert len(ids) <= MAX_UNION


@pytest.mark.parametrize("use_mmr", [True, False])
def test_union_toi_da_30_voi_branch_top_n_that(
    monkeypatch: pytest.MonkeyPatch, use_mmr: bool
):
    monkeypatch.setattr(pipeline_module, "BRANCH_TOP_N", 10)  # giá trị thật
    pipe, _, rr = _pipe(THREE)
    asyncio.run(pipe.retrieve(THREE, use_mmr=use_mmr))
    assert len(rr.calls[0][1]) <= 2 * 10 + CITATION_SPARSE_TOP_K == 30


def test_groq_loi_chi_nhanh_b_van_co_extras():
    pipe, sparse, rr = _pipe(THREE, hyde=None)
    asyncio.run(pipe.retrieve(THREE, use_mmr=False))
    assert len(sparse.texts) == 1
    assert set(_ids_of("b", CITATION_SPARSE_TOP_K)) <= _passage_ids(rr)


def test_pipeline_khong_con_luot_sparse_phu_theo_dieu():
    assert not hasattr(pipeline_module.RetrievalPipeline, "_article_hits")


def test_pipeline_cau_khong_vien_dan_khong_them_luot_sparse():
    query = "người lao động nghỉ phép bao nhiêu ngày"
    pipe, sparse, rr = _pipe(query)
    asyncio.run(pipe.retrieve(query, use_mmr=False))
    assert len(sparse.texts) == 2
    assert _passage_ids(rr) <= set(DENSE) | set(_ids_of("b", 12))


@pytest.mark.parametrize("use_mmr", [True, False])
def test_cau_khong_vien_dan_so_luot_fetch_nhu_truoc(use_mmr: bool):
    query = "người lao động nghỉ phép bao nhiêu ngày"
    pipe, _, _ = _pipe(query)
    asyncio.run(pipe.retrieve(query, use_mmr=use_mmr))
    dense = pipe._dense_search
    assert len(dense.fetch_calls) <= (2 if use_mmr else 1)  # type: ignore[attr-defined]


def test_so_luot_fetch_mmr_tat_cau_vien_dan_toi_da_1():
    pipe, _, _ = _pipe(THREE)
    asyncio.run(pipe.retrieve(THREE, use_mmr=False))
    assert len(pipe._dense_search.fetch_calls) == 1  # type: ignore[attr-defined]


def test_fallback_khi_rerank_loi_van_co_extras_va_khong_crash():
    pipe, _, _ = _pipe(ONE, hyde=None, reranker="fail")
    result = asyncio.run(pipe.retrieve(ONE, use_mmr=False))
    assert result
    assert all(c.rerank_score is None for c in result)


def test_api_nhieu_dieu_da_go():
    from production_legal_qa_rag.retrieval import citation

    for name in (
        "build_article_queries",
        "citation_article_hits",
        "MAX_CITATION_ARTICLES",
        "CITATION_EXTRAS_BUDGET",
    ):
        assert not hasattr(citation, name)
    assert not hasattr(pipeline_module.RetrievalPipeline, "_article_hits")


def test_timeout_reranker_union_toi_da_30_passage_la_75s(
    monkeypatch: pytest.MonkeyPatch,
):
    import httpx

    from production_legal_qa_rag.config import RerankerSettings
    from production_legal_qa_rag.retrieval.reranker_client import RerankerClient

    monkeypatch.setenv("RERANKER_ENDPOINT_URL", "http://x")
    monkeypatch.setenv("RERANKER_API_KEY", "k")
    client = RerankerClient(RerankerSettings(), httpx.AsyncClient())
    assert client._read_timeout_seconds(2 * 10 + CITATION_SPARSE_TOP_K) == 75.0
