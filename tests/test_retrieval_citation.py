"""Test citation.py và extras câu hỏi viện dẫn trong pipeline (mục 8.1)."""

from __future__ import annotations

import asyncio

import pytest
from test_retrieval_pipeline import _build, _meta

from production_legal_qa_rag.retrieval import pipeline as pipeline_module
from production_legal_qa_rag.retrieval.citation import (
    CITATION_SPARSE_TOP_K,
    citation_extras,
    has_citation,
)
from production_legal_qa_rag.retrieval.models import SearchHit

CITED = "Điều 3 khoản 1 quy định gì"
UNCITED = "người lao động nghỉ phép bao nhiêu ngày"
SPARSE_B = ["c1", "c2", "c3", "c4", "c50"]  # c50 chỉ có ở sparse, hạng 5
DENSE = [f"c{i}" for i in range(1, 21)]


@pytest.mark.parametrize(
    "query",
    [
        "Điều 36 khoản 2 Bộ luật Lao động quy định gì?",
        "khoản 2 điều 36 quy định gì",
        "điều 3",
        "ĐIỀU 12 nói gì",
        "Điều\t7 nói gì",
        "Điều 3 khoản 1".encode().decode("utf-8"),
    ],
)
def test_has_citation_duong(query: str):
    assert has_citation(query)


def test_has_citation_nfc_tu_dang_to_hop():
    import unicodedata

    assert has_citation(unicodedata.normalize("NFD", "Điều 3 quy định gì"))


@pytest.mark.parametrize(
    "query",
    [
        "điều kiện lao động là gì",
        "trong 3 điều kiện sau",
        "khoản 2 quy định gì",
        "người lao động được nghỉ mấy ngày",
        "",
    ],
)
def test_has_citation_am(query: str):
    assert not has_citation(query)


def test_citation_extras_lay_top_k_theo_thu_tu():
    hits = [SearchHit(chunk_id=f"s{i}", score=10.0 - i) for i in range(8)]
    extras = citation_extras(hits)
    assert [c.chunk_id for c in extras] == [
        f"s{i}" for i in range(CITATION_SPARSE_TOP_K)
    ]
    assert all(c.metadata is None for c in extras)


@pytest.fixture(autouse=True)
def small_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cắt mạnh để c50 chắc chắn bị RRF/MMR loại khỏi nhánh.
    monkeypatch.setattr(pipeline_module, "BRANCH_TOP_N", 3)


def _pipe(query: str, **kwargs):  # type: ignore[no-untyped-def]
    known = set(DENSE) | {"c50"}
    return _build(DENSE, {query: SPARSE_B}, known=known, **kwargs)


@pytest.mark.parametrize("use_mmr", [True, False])
def test_chunk_dap_an_chi_o_sparse_top5_co_trong_union(use_mmr: bool):
    pipe, _, rr, _ = _pipe(CITED)
    asyncio.run(pipe.retrieve(CITED, use_mmr=use_mmr))
    assert "bc-c50\nnd-c50" in rr.calls[0][1]


@pytest.mark.parametrize("use_mmr", [True, False])
def test_cau_khong_vien_dan_khong_co_extras(
    use_mmr: bool, monkeypatch: pytest.MonkeyPatch
):
    def boom(*args: object) -> None:
        raise AssertionError("citation_extras không được gọi")

    monkeypatch.setattr(pipeline_module, "citation_extras", boom)
    pipe, _, rr, _ = _pipe(UNCITED)
    asyncio.run(pipe.retrieve(UNCITED, use_mmr=use_mmr))
    assert "bc-c50\nnd-c50" not in rr.calls[0][1]
    assert len(rr.calls[0][1]) <= 2 * 3


@pytest.mark.parametrize("use_mmr", [True, False])
def test_khong_them_luot_sparse_query(use_mmr: bool):
    pipe, _, _, _ = _pipe(CITED)
    asyncio.run(pipe.retrieve(CITED, use_mmr=use_mmr))
    assert len(pipe._sparse_index.texts) == 2  # type: ignore[attr-defined]


def test_so_luot_fetch_mmr_tat_toi_da_1():
    pipe, dense, _, _ = _pipe(CITED)
    asyncio.run(pipe.retrieve(CITED, use_mmr=False))
    assert len(dense.fetch_calls) == 1
    assert set(dense.fetch_calls[0]) == {"c4", "c50"}  # extras bị cắt khỏi nhánh


def test_so_luot_fetch_mmr_bat_toi_da_3():
    pipe, dense, _, _ = _pipe(CITED)
    asyncio.run(pipe.retrieve(CITED, use_mmr=True))
    assert 1 <= len(dense.fetch_calls) <= 3


def test_so_luot_fetch_cau_khong_vien_dan_mmr_bat_khong_doi():
    pipe, dense, _, _ = _pipe(UNCITED)
    asyncio.run(pipe.retrieve(UNCITED, use_mmr=True))
    # c50 chỉ ở sparse nên vẫn được fetch trong nhánh (trước MMR), không thêm lượt.
    assert len(dense.fetch_calls) == 1


def test_fallback_xen_ke_co_danh_sach_extras():
    pipe, _, _, _ = _pipe(CITED, reranker="fail")
    result = asyncio.run(pipe.retrieve(CITED, use_mmr=False))
    assert [c.chunk_id for c in result] == ["c1", "c2", "c3", "c4", "c50"]
    assert all(c.rerank_score is None for c in result)
    assert result[-1].content == _meta("c50").content


def test_groq_loi_chi_nhanh_b_van_co_extras():
    pipe, _, rr, embedder = _pipe(CITED, hyde=None)
    asyncio.run(pipe.retrieve(CITED, use_mmr=False))
    assert embedder.calls == [[CITED]]
    assert "bc-c50\nnd-c50" in rr.calls[0][1]


def test_extras_khong_co_metadata_tren_dense_bi_bo():
    pipe, _, rr, _ = _build(DENSE, {CITED: SPARSE_B}, known=set(DENSE))
    result = asyncio.run(pipe.retrieve(CITED, use_mmr=False))
    assert "c50" not in [c.chunk_id for c in result]
    assert "bc-c50\nnd-c50" not in rr.calls[0][1]
