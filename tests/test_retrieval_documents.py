"""Test token theo văn bản (`vb_*`), bảng DOCUMENTS và việc gỡ ghim (mục 6.2, 8.1)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from test_retrieval_structural import _pipe

from production_legal_qa_rag.retrieval import citation
from production_legal_qa_rag.retrieval.bm25 import BM25_PARAMS_VERSION, BM25Encoder
from production_legal_qa_rag.retrieval.citation import (
    DOCUMENTS,
    breadcrumb_structural_terms,
    build_article_queries,
    detect_document,
    extract_citation_khoans,
    extract_citation_numbers,
    parse_breadcrumb,
    structural_terms,
)

CHUNKS_DIR = Path(__file__).resolve().parent.parent / "data" / "chunks"
BLLD = "BỘ LUẬT LAO ĐỘNG"


@pytest.fixture(scope="module")
def real_chunks() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for path in sorted(CHUNKS_DIR.glob("*.json")):
        chunks.extend(json.loads(path.read_text(encoding="utf-8")))
    return chunks


# ------------------------------------------------------------- bảng DOCUMENTS


def test_bang_documents_khop_corpus_that_moi_file_dung_1_source_document():
    seen: set[str] = set()
    for path in sorted(CHUNKS_DIR.glob("*.json")):
        names = {
            c["source_document"] for c in json.loads(path.read_text(encoding="utf-8"))
        }
        assert len(names) == 1, path.name
        assert names <= DOCUMENTS.keys(), (path.name, names)
        seen |= names
    assert seen == set(DOCUMENTS)
    assert len(DOCUMENTS) == 6
    assert len({entry.key for entry in DOCUMENTS.values()}) == 6
    assert all(entry.key.isascii() for entry in DOCUMENTS.values())


# ------------------------------------------------------------- detect_document


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Điều 36 và Điều 113 Bộ luật Lao động", "blld"),
        ("Điều 3 luật lao động quy định gì", "blld"),
        ("điều 5 BLLĐ", "blld"),
        ("Điều 3 khoản 1 luật thuế TNCN", "tncn"),
        ("Điều 3 luật thuế thu nhập cá nhân", "tncn"),
        ("Điều 3 thuế thu nhập cá nhân", "tncn"),
        ("dieu 5 luat bhxh", "bhxh"),  # không dấu
        ("Điều 5 Luật BHXH", "bhxh"),
        ("Điều 5 luật bảo hiểm y tế", "bhyt"),
        ("Điều 5 nghị định lương tối thiểu", "nd_luong"),
        ("Điều 5 Nghị định quy định mức lương tối thiểu", "nd_luong"),
        ("Điều 5 nghị định điều kiện lao động", "nd_dkld"),
        # alias dài thắng alias ngắn "bộ luật lao động" nằm bên trong
        ("Điều 5 Nghị định hướng dẫn Bộ luật Lao động", "nd_dkld"),
        (
            "Điều 3 Bộ luật Lao động và Điều 5 Bộ luật Lao động",
            "blld",
        ),  # cùng 1 văn bản
    ],
)
def test_detect_document_duong(query: str, expected: str):
    assert detect_document(query) == expected


@pytest.mark.parametrize(
    "query",
    [
        "Điều 36 quy định gì",  # không nêu văn bản
        "Điều 5 nghị định quy định gì",  # "nghị định" trơn
        "Điều 5 bảo hiểm xã hội",  # tên chủ đề trơn không phải alias
        "Điều 2 mức lương tối thiểu",
        "Điều 5 lao động",
        "Điều 3 luật BHXH và Điều 5 luật BHYT",  # ≥ 2 văn bản khác nhau
        "Điều 3 luật lao động và Điều 5 luật bhxh",
        "",
    ],
)
def test_detect_document_am_hoac_mo_ho_la_none(query: str):
    assert detect_document(query) is None


# ------------------------------------------------------------- token nhất quán


def test_token_document_va_query_cung_ham_dinh_dang():
    doc_terms = breadcrumb_structural_terms(
        f"{BLLD} - Chương IV - Điều 36. Tên - Khoản 1", BLLD
    )
    assert doc_terms == [
        "điều_36",
        "khoản_1",
        "điều_36_khoản_1",
        "vb_blld",
        "vb_blld_điều_36",
        "vb_blld_điều_36_khoản_1",
    ]
    assert set(doc_terms) == set(structural_terms([36], [1], "blld"))


def test_chunk_khong_co_dieu_chi_co_vb_x_va_van_ban_la_khong_sinh_vb():
    assert breadcrumb_structural_terms(
        "LUẬT BẢO HIỂM XÃ HỘI (phần 1/2)", "LUẬT BẢO HIỂM XÃ HỘI"
    ) == ["vb_bhxh"]
    assert breadcrumb_structural_terms("X - Điều 3. T - Khoản 1", "VĂN BẢN LẠ") == [
        "điều_3",
        "khoản_1",
        "điều_3_khoản_1",
    ]
    assert breadcrumb_structural_terms("X - Điều 3. T", None) == ["điều_3"]


def test_structural_terms_khong_nen_van_ban_giu_hanh_vi_cu():
    assert structural_terms([36], [2], None) == structural_terms([36], [2])
    assert not any(t.startswith("vb_") for t in structural_terms([36], [2]))
    assert structural_terms([], [2], "blld") == []


def test_structural_terms_tran_28_token():
    terms = structural_terms([1, 2, 3], [1, 2, 3], "blld")
    assert len(terms) == 15 + 1 + 3 + 9 == 28


# ------------------------------------------------------------- corpus thật


@pytest.fixture(scope="module")
def real_index(real_chunks: list[dict[str, Any]]):  # type: ignore[no-untyped-def]
    texts = [c["breadcrumb"] + " " + c["content"] for c in real_chunks]
    terms = [
        breadcrumb_structural_terms(c["breadcrumb"], c["source_document"])
        for c in real_chunks
    ]
    encoder = BM25Encoder()
    encoder.fit(texts, extra_terms=terms)
    docs = [encoder.encode_document(t, x) for t, x in zip(texts, terms, strict=True)]
    return encoder, docs


def _rank(real_index: Any, count: int, text: str, terms: list[str]) -> list[int]:
    encoder, docs = real_index
    qv = encoder.encode_query(text, terms)
    weights = dict(zip(qv.indices, qv.values, strict=True))
    scores = [
        sum(weights.get(i, 0.0) * v for i, v in zip(d.indices, d.values, strict=True))
        for d in docs
    ]
    return sorted(range(count), key=lambda i: -scores[i])


def _query_terms(query: str) -> list[str]:
    numbers = extract_citation_numbers(query)
    return structural_terms(
        numbers, extract_citation_khoans(query), detect_document(query)
    )


# Điều 36 có chunk rất dài (~350 từ) bị chuẩn hoá độ dài BM25 phạt nên rơi xuống
# hạng 8, sau 4 chunk BHYT Điều 36; vẫn nằm trong quota extras 12/Điều (n = 2).
@pytest.mark.parametrize(
    ("number", "expected_count", "window"), [(36, 4, 12), (113, 7, 7)]
)
def test_moi_chunk_blld_dieu_n_dung_dau_khi_neu_bo_luat_lao_dong(
    real_chunks: list[dict[str, Any]],
    real_index: Any,
    number: int,
    expected_count: int,
    window: int,
):
    target = {
        i
        for i, c in enumerate(real_chunks)
        if c["source_document"] == BLLD
        and parse_breadcrumb(c["breadcrumb"]).dieu == number
    }
    assert len(target) == expected_count
    query = f"Điều {number} Bộ luật Lao động"
    order = _rank(real_index, len(real_chunks), query, _query_terms(query))
    assert target <= set(order[:window])


def test_chunk_dung_van_ban_thang_chunk_cung_dieu_khac_van_ban(
    real_chunks: list[dict[str, Any]], real_index: Any
):
    query = "Điều 36 Bộ luật Lao động"
    top = real_chunks[
        _rank(real_index, len(real_chunks), query, _query_terms(query))[0]
    ]
    assert top["source_document"] == BLLD
    other_docs = {
        c["source_document"]
        for c in real_chunks
        if parse_breadcrumb(c["breadcrumb"]).dieu == 36
    }
    assert len(other_docs) > 1  # có văn bản khác cũng có Điều 36


def test_khong_neu_van_ban_hanh_vi_nhu_cu(real_chunks: list[dict[str, Any]]):
    terms = _query_terms("Điều 3 khoản 1 quy định gì")
    assert terms == ["điều_3", "khoản_1", "điều_3_khoản_1"]


# ------------------------------------------------------------- pipeline


def test_pipeline_nhanh_b_va_sub_query_co_token_van_ban_nhanh_a_khong():
    query = "Điều 3 khoản 1 và Điều 5 khoản 2 Bộ luật Lao động"
    pipe, sparse = _pipe(["z1"], {})
    asyncio.run(pipe.retrieve(query, use_mmr=False))
    sub_3, sub_5 = build_article_queries(query, [3, 5])
    assert sparse.terms[query] == structural_terms([3, 5], [1, 2], "blld")
    assert sparse.terms[sub_3] == structural_terms([3], [1, 2], "blld")
    assert sparse.terms[sub_5] == structural_terms([5], [1, 2], "blld")
    assert "vb_blld_điều_3" in sparse.terms[sub_3]
    assert "vb_blld_điều_5" not in sparse.terms[sub_3]
    assert sparse.terms["giả định"] == []


def test_pipeline_van_ban_mo_ho_khong_sinh_token_van_ban():
    query = "Điều 3 luật BHXH và Điều 5 luật BHYT"
    pipe, sparse = _pipe(["z1"], {})
    asyncio.run(pipe.retrieve(query, use_mmr=False))
    assert all(
        not t.startswith("vb_") for terms in sparse.terms.values() for t in terms
    )


# ------------------------------------------------------------- params & gỡ ghim


def test_params_version_hien_tai_la_3():
    assert BM25_PARAMS_VERSION == 3


def test_ghim_da_go_khong_con_api_hay_hang_so():
    assert not hasattr(citation, "pin_exact_matches")
    assert not hasattr(citation, "PIN_PER_ARTICLE")
    from production_legal_qa_rag.retrieval import pipeline

    assert not hasattr(pipeline, "pin_exact_matches")


# ------------------------------------------------------------- bổ sung


def test_build_index_canh_bao_source_document_la(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    from production_legal_qa_rag.retrieval import sparse_index

    chunks = [
        {
            "chunk_id": f"c{i}",
            "source_document": name,
            "breadcrumb": f"{name} - Điều 1. T - Khoản 1",
            "content": "nội dung",
            "token_count": 3,
        }
        for i, name in enumerate([BLLD, "VĂN BẢN LẠ"])
    ]
    (tmp_path / "a.json").write_text(json.dumps(chunks), encoding="utf-8")
    index = type(
        "I",
        (),
        {
            "delete": lambda self, **k: None,
            "upsert": lambda self, **k: None,
        },
    )()
    client = type(
        "C",
        (),
        {"list_indexes": lambda self: ["sparse"], "Index": lambda self, n: index},
    )()
    from production_legal_qa_rag.config import VectorDBSettings

    for k, v in {
        "PINECONE_API_KEY": "p",
        "PINECONE_INDEX_NAME": "d",
        "PINECONE_SPARSE_INDEX_NAME": "sparse",
    }.items():
        monkeypatch.setenv(k, v)
    with caplog.at_level("WARNING", logger=sparse_index.logger.name):
        sparse_index.build_index(
            tmp_path,
            tmp_path / "p.json",
            VectorDBSettings(),
            client,  # type: ignore[arg-type]
        )
    assert "VĂN BẢN LẠ" in caplog.text
    encoder = BM25Encoder.load(tmp_path / "p.json")
    assert "vb_blld_điều_1_khoản_1" in encoder.params.vocab
    assert not any(t.startswith("vb_") and "lạ" in t for t in encoder.params.vocab)


def test_breadcrumb_terms_van_ban_la_khong_sinh_vb_va_khong_loi():
    terms = breadcrumb_structural_terms("X - Điều 1. T - Khoản 1", "KHÔNG CÓ")
    assert terms == ["điều_1", "khoản_1", "điều_1_khoản_1"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Điều 145 Nghị định 145", None),  # "Nghị định" kèm số trơn không phải alias
        ("Điều 5 luật BHXH và luật bảo hiểm xã hội", "bhxh"),  # cùng văn bản
        ("ĐIỀU 5 BỘ LUẬT LAO ĐỘNG", "blld"),
        ("Điều 5 bộ  luật   lao động", "blld"),  # khoảng trắng thừa
    ],
)
def test_detect_document_bien(query: str, expected: str | None):
    assert detect_document(query) == expected


def test_cau_chi_nen_khoan_khong_sinh_token_van_ban():
    query = "Khoản 2 Bộ luật Lao động quy định gì"
    assert extract_citation_numbers(query) == []
    pipe, sparse = _pipe(["z1"], {})
    asyncio.run(pipe.retrieve(query, use_mmr=False))
    assert all(terms == [] for terms in sparse.terms.values())


def test_detect_document_chuoi_dai_khong_bung_no():
    import time

    for query in (
        "bộ luật lao động " * 5000,
        "luật " * 40000,
        "nghị định " * 20000 + "điều kiện",
        "a" * 200000,
    ):
        start = time.perf_counter()
        detect_document(query)
        assert time.perf_counter() - start < 3.0


def test_timeout_reranker_44_passage_la_110s(monkeypatch: pytest.MonkeyPatch):
    import httpx

    from production_legal_qa_rag.config import RerankerSettings
    from production_legal_qa_rag.retrieval.reranker_client import RerankerClient

    monkeypatch.setenv("RERANKER_ENDPOINT_URL", "http://x")
    monkeypatch.setenv("RERANKER_API_KEY", "k")
    client = RerankerClient(RerankerSettings(), httpx.AsyncClient())
    assert client._read_timeout_seconds(44) == 110.0
