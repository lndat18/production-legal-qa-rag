"""Test nhận diện nhiều Điều và extras theo Điều (retrieval_spec.md mục 8.1)."""

from __future__ import annotations

import asyncio
import logging

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
    CITATION_EXTRAS_BUDGET,
    CITATION_SPARSE_TOP_K,
    MAX_CITATION_ARTICLES,
    build_article_queries,
    citation_extras,
    extract_citation_numbers,
    has_citation,
)
from production_legal_qa_rag.retrieval.models import RetrievalError, SearchHit

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
        ("điều 5 tháng", [5]),  # giới hạn đã biết: vẫn khớp
    ],
)
def test_extract_citation_numbers_duong(query: str, expected: list[int]):
    assert extract_citation_numbers(query) == expected
    assert has_citation(query)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Điều 3 và 5 người được hưởng", [3]),
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


def test_extract_citation_numbers_tran_3_dieu_va_log(
    caplog: pytest.LogCaptureFixture,
):
    assert MAX_CITATION_ARTICLES == 3
    with caplog.at_level(logging.INFO, logger="production_legal_qa_rag.retrieval"):
        assert extract_citation_numbers("Điều 1, 2, 3, 4 và Điều 5") == [1, 2, 3]
    assert "[4, 5]" in caplog.text


def test_build_article_queries_bo_so_dieu_khac_khong_xoa_khoan():
    query = "Điều 3 khoản 1 và Điều 5 khoản 2"
    assert build_article_queries(query, [3, 5]) == [
        "Điều 3 khoản 1 và khoản 2",
        "khoản 1 và Điều 5 khoản 2",
    ]


def test_build_article_queries_dieu_thu_va_dieu_vuot_tran():
    queries = build_article_queries("Điều thứ 3 và Điều thứ 5 và Điều 9", [3, 5])
    assert queries == ["Điều thứ 3 và và", "và Điều thứ 5 và"]


# ------------------------------------------------------------- citation_extras


def _hits(prefix: str, count: int = 20) -> list[SearchHit]:
    return [SearchHit(chunk_id=f"{prefix}{i}", score=1.0) for i in range(1, count + 1)]


def _ids(candidates: list) -> list[str]:  # type: ignore[type-arg]
    return [c.chunk_id for c in candidates]


def test_extras_n1_dung_k_hit_dau_sparse_tho_nhanh_b():
    extras = citation_extras([36], _hits("b"), article_hits=None)
    assert _ids(extras) == [f"b{i}" for i in range(1, CITATION_SPARSE_TOP_K + 1)]


def test_extras_n2_moi_dieu_8_xen_ke():
    extras = citation_extras([3, 5], _hits("b"), {3: _hits("x"), 5: _hits("y")})
    expected = [f"{p}{i}" for i in range(1, 9) for p in "xy"]
    assert _ids(extras) == expected
    assert len(extras) == CITATION_EXTRAS_BUDGET


def test_extras_n3_moi_dieu_5_tong_toi_da_16():
    hits = {3: _hits("x"), 5: _hits("y"), 7: _hits("z")}
    extras = citation_extras([3, 5, 7], _hits("b"), hits)
    assert _ids(extras) == [f"{p}{i}" for i in range(1, 6) for p in "xyz"]
    assert len(extras) <= CITATION_EXTRAS_BUDGET


def test_extras_dedupe_giu_vi_tri_som_nhat():
    shared = _hits("s", 3)
    extras = citation_extras([3, 5], [], {3: shared, 5: shared})
    assert _ids(extras) == ["s1", "s2", "s3"]


def test_extras_mot_dieu_loi_bo_dieu_do_quota_van_theo_n():
    extras = citation_extras([3, 5], _hits("b"), {5: _hits("y")})
    assert _ids(extras) == [f"y{i}" for i in range(1, 9)]


def test_extras_moi_luot_phu_loi_lui_ve_n1():
    for article_hits in (None, {}):
        extras = citation_extras([3, 5], _hits("b"), article_hits)
        assert _ids(extras) == [f"b{i}" for i in range(1, CITATION_SPARSE_TOP_K + 1)]


def test_extras_khong_dieu_nao_la_rong():
    assert citation_extras([], _hits("b")) == []


# ------------------------------------------------------------- pipeline

DENSE = [f"c{i}" for i in range(1, 21)]
TWO = "Điều 3 khoản 1 và Điều 5 khoản 2"
TWO_SUBQUERIES = ["Điều 3 khoản 1 và khoản 2", "khoản 1 và Điều 5 khoản 2"]
THREE = "Điều 3, 5 và 7 quy định gì"
THREE_SUBQUERIES = build_article_queries(THREE, [3, 5, 7])


class SparseByText(FakeSparse):
    """Ghi lại (text, top_k); text trong `fail_texts` ném RetrievalError."""

    def __init__(
        self, by_text: dict[str, list[str]], fail_texts: frozenset[str] = frozenset()
    ):
        super().__init__(by_text)
        self.fail_texts = fail_texts
        self.top_ks: dict[str, int] = {}

    async def query(self, text: str, top_k: int = 20) -> list[SearchHit]:
        self.texts.append(text)
        self.top_ks[text] = top_k
        if text in self.fail_texts:
            raise RetrievalError("sparse lỗi")
        return [SearchHit(chunk_id=i, score=1.0) for i in self.by_text.get(text, [])]


def _ids_of(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(1, count + 1)]


def _multi_pipe(
    query: str,
    subqueries: list[str],
    *,
    fail_texts: frozenset[str] = frozenset(),
    hyde: str | None = "giả định",
    reranker: str = "ok",
) -> tuple[pipeline_module.RetrievalPipeline, SparseByText, FakeReranker]:
    prefixes = ["x", "y", "z"]
    by_text = {sq: _ids_of(prefixes[i], 12) for i, sq in enumerate(subqueries)}
    by_text[query] = _ids_of("b", 12)
    known = set(DENSE)
    for ids in by_text.values():
        known |= set(ids)
    sparse = SparseByText(by_text, fail_texts)
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
def test_pipeline_n2_moi_dieu_8_extras_va_dung_so_luot_sparse(use_mmr: bool):
    pipe, sparse, rr = _multi_pipe(TWO, TWO_SUBQUERIES)
    asyncio.run(pipe.retrieve(TWO, use_mmr=use_mmr))
    assert len(sparse.texts) == 2 + 2
    assert set(sparse.texts) >= set(TWO_SUBQUERIES)
    assert all(sparse.top_ks[sq] == 8 for sq in TWO_SUBQUERIES)
    ids = _passage_ids(rr)
    assert set(_ids_of("x", 8)) <= ids and "x9" not in ids
    assert set(_ids_of("y", 8)) <= ids and "y9" not in ids
    assert len(ids) <= 2 * pipeline_module.BRANCH_TOP_N + CITATION_EXTRAS_BUDGET
    # n = 2 không dùng sparse thô nhánh B cho extras.
    assert "b10" not in ids


@pytest.mark.parametrize("use_mmr", [True, False])
def test_pipeline_n3_moi_dieu_5_extras_va_dung_so_luot_sparse(use_mmr: bool):
    pipe, sparse, rr = _multi_pipe(THREE, THREE_SUBQUERIES)
    asyncio.run(pipe.retrieve(THREE, use_mmr=use_mmr))
    assert len(sparse.texts) == 2 + 3
    assert all(sparse.top_ks[sq] == 5 for sq in THREE_SUBQUERIES)
    ids = _passage_ids(rr)
    for prefix in "xyz":
        assert set(_ids_of(prefix, 5)) <= ids and f"{prefix}6" not in ids
    assert len(ids) <= 2 * pipeline_module.BRANCH_TOP_N + CITATION_EXTRAS_BUDGET


def test_pipeline_so_luot_fetch_khong_doi_theo_n_mmr_tat():
    pipe, _, _ = _multi_pipe(TWO, TWO_SUBQUERIES)
    asyncio.run(pipe.retrieve(TWO, use_mmr=False))
    assert len(pipe._dense_search.fetch_calls) == 1  # type: ignore[attr-defined]


def test_pipeline_mot_luot_phu_loi_bo_extras_dieu_do(
    caplog: pytest.LogCaptureFixture,
):
    pipe, _, rr = _multi_pipe(
        TWO, TWO_SUBQUERIES, fail_texts=frozenset({TWO_SUBQUERIES[0]})
    )
    with caplog.at_level(logging.WARNING):
        asyncio.run(pipe.retrieve(TWO, use_mmr=False))
    ids = _passage_ids(rr)
    assert "x1" not in ids
    assert set(_ids_of("y", 8)) <= ids
    assert "Điều 3" in caplog.text


def test_pipeline_moi_luot_phu_loi_lui_ve_n1():
    pipe, _, rr = _multi_pipe(TWO, TWO_SUBQUERIES, fail_texts=frozenset(TWO_SUBQUERIES))
    result = asyncio.run(pipe.retrieve(TWO, use_mmr=False))
    ids = _passage_ids(rr)
    assert result
    assert set(_ids_of("b", CITATION_SPARSE_TOP_K)) <= ids
    assert "b11" not in ids
    assert not ids & set(_ids_of("x", 12) + _ids_of("y", 12))


def test_pipeline_loi_sparse_nhanh_chinh_van_raise():
    pipe, _, _ = _multi_pipe(
        TWO, TWO_SUBQUERIES, fail_texts=frozenset({TWO}), hyde=None
    )
    with pytest.raises(RetrievalError):
        asyncio.run(pipe.retrieve(TWO, use_mmr=False))


def test_pipeline_luot_phu_loi_khong_lam_hong_nhanh_chinh():
    pipe, _, rr = _multi_pipe(TWO, TWO_SUBQUERIES, fail_texts=frozenset(TWO_SUBQUERIES))
    asyncio.run(pipe.retrieve(TWO, use_mmr=True))
    assert rr.calls  # nhánh chính vẫn hoàn tất tới rerank


def test_pipeline_groq_loi_nhieu_dieu_chi_nhanh_b_va_luot_phu():
    pipe, sparse, rr = _multi_pipe(TWO, TWO_SUBQUERIES, hyde=None)
    asyncio.run(pipe.retrieve(TWO, use_mmr=False))
    assert len(sparse.texts) == 1 + 2
    assert set(_ids_of("x", 8) + _ids_of("y", 8)) <= _passage_ids(rr)


def test_pipeline_fallback_xen_ke_e_theo_dieu():
    pipe, _, _ = _multi_pipe(TWO, TWO_SUBQUERIES, hyde=None, reranker="fail")
    # Nhánh B (chỉ dense c1..c3 sau khi cắt) xen kẽ với E = x1, y1, x2, y2, ...
    pipe._sparse_index.by_text[TWO] = []  # type: ignore[attr-defined]
    result = asyncio.run(pipe.retrieve(TWO, use_mmr=False))
    assert [c.chunk_id for c in result] == ["c1", "x1", "c2", "y1", "c3"]
    assert all(c.rerank_score is None for c in result)


def test_pipeline_cau_khong_vien_dan_khong_them_luot_sparse():
    query = "người lao động nghỉ phép bao nhiêu ngày"
    pipe, sparse, rr = _multi_pipe(query, [])
    asyncio.run(pipe.retrieve(query, use_mmr=False))
    assert len(sparse.texts) == 2
    assert _passage_ids(rr) <= set(DENSE) | set(_ids_of("b", 12))


# ------------------------------------------------------------- bổ sung


def test_vuot_tran_chi_3_dieu_dau_va_phan_du_bi_bo_khoi_sub_query():
    query = "Điều 3, 5, 7 và 9 quy định gì"
    numbers = extract_citation_numbers(query)
    assert numbers == [3, 5, 7]
    for number, sub_query in zip(
        numbers, build_article_queries(query, numbers), strict=True
    ):
        assert f"{number}" in sub_query
        others = {3, 5, 7, 9} - {number}
        assert not any(f" {o} " in f" {sub_query} " for o in others)
    # Ngân sách chia theo số Điều đã cắt trần (3), không theo số Điều nhắc tới (4).
    hits = {n: [SearchHit(chunk_id=f"{n}-{i}") for i in range(20)] for n in numbers}
    extras = citation_extras(numbers, [], hits)
    assert len(extras) == 3 * (CITATION_EXTRAS_BUDGET // 3)
    assert not any(c.chunk_id.startswith("9-") for c in extras)


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
    numbers = extract_citation_numbers(query)
    build_article_queries(query, numbers)
    assert time.perf_counter() - start < 2.0
    assert len(numbers) <= MAX_CITATION_ARTICLES


def test_luot_phu_khong_nuot_cancelled_error():
    class CancellingSparse(SparseByText):
        async def query(self, text: str, top_k: int = 20) -> list[SearchHit]:
            if text in TWO_SUBQUERIES:
                raise asyncio.CancelledError
            return await super().query(text, top_k)

    pipe, _, _ = _multi_pipe(TWO, TWO_SUBQUERIES)
    pipe._sparse_index = CancellingSparse(  # type: ignore[assignment]
        pipe._sparse_index.by_text,  # type: ignore[attr-defined]
        frozenset(),
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(pipe.retrieve(TWO, use_mmr=False))


def test_n1_hoi_quy_khong_them_luot_sparse_va_extras_top_k_nhanh_b():
    query = "Điều 3 khoản 1 quy định gì"
    pipe, sparse, rr = _multi_pipe(query, [])
    asyncio.run(pipe.retrieve(query, use_mmr=False))
    assert len(sparse.texts) == 2  # A + B, không có lượt phụ
    assert set(_ids_of("b", CITATION_SPARSE_TOP_K)) <= _passage_ids(rr)


@pytest.mark.parametrize("use_mmr", [True, False])
def test_cau_khong_vien_dan_so_luot_fetch_nhu_truoc(use_mmr: bool):
    query = "người lao động nghỉ phép bao nhiêu ngày"
    pipe, _, _ = _multi_pipe(query, [])
    asyncio.run(pipe.retrieve(query, use_mmr=use_mmr))
    dense = pipe._dense_search
    assert len(dense.fetch_calls) <= (2 if use_mmr else 1)  # type: ignore[attr-defined]
