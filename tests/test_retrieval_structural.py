"""Test token cấu trúc BM25, params_version và ghim khớp chính xác (mục 6.2, 8.2)."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest
import test_retrieval_pipeline as pipeline_fakes
from test_retrieval_pipeline import FakeDense, FakeEmbedder, FakeHyde, FakeSparse

from production_legal_qa_rag.retrieval import pipeline as pipeline_module
from production_legal_qa_rag.retrieval.bm25 import (
    BM25_PARAMS_VERSION,
    REBUILD_COMMAND,
    BM25Encoder,
    BM25ParamsVersionError,
    tokenize,
)
from production_legal_qa_rag.retrieval.citation import (
    MAX_CITATION_KHOANS,
    PIN_PER_ARTICLE,
    BreadcrumbRef,
    breadcrumb_structural_terms,
    build_article_queries,
    extract_citation_khoans,
    extract_citation_numbers,
    parse_breadcrumb,
    pin_exact_matches,
    structural_terms,
)
from production_legal_qa_rag.retrieval.models import RetrievedChunk

CHUNKS_DIR = Path(__file__).resolve().parent.parent / "data" / "chunks"


def _load_real_chunks() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for path in sorted(CHUNKS_DIR.glob("*.json")):
        chunks.extend(json.loads(path.read_text(encoding="utf-8")))
    return chunks


@pytest.fixture(scope="module")
def real_chunks() -> list[dict[str, Any]]:
    return _load_real_chunks()


# ------------------------------------------------------------- parse breadcrumb

TITLE = "NGHỊ ĐỊNH QUY ĐỊNH CHI TIẾT VÀ HƯỚNG DẪN THI HÀNH MỘT SỐ ĐIỀU CỦA BỘ LUẬT LAO ĐỘNG VỀ ĐIỀU KIỆN LAO ĐỘNG"


@pytest.mark.parametrize(
    ("breadcrumb", "expected"),
    [
        (f"{TITLE} - Chương I - Điều 1. Phạm vi điều chỉnh - Khoản 1", (1, 1)),
        (f"{TITLE} - Chương I - Điều 1. Phạm vi điều chỉnh", (1, None)),
        (
            f"{TITLE} - Chương III - Mục 2 - Điều 8. Trợ cấp - Khoản 3 - Điểm a (phần 1/2)",
            (8, 3),
        ),
        (
            f"{TITLE} - Chương III - Mục 1 - Điều 5. Nội dung - Khoản 1 (phần 2/3)",
            (5, 1),
        ),
        (
            "BỘ LUẬT LAO ĐỘNG - Chương III - Mục 1 - Điều 22. Phụ lục hợp đồng lao động - Khoản 1",
            (22, 1),
        ),
        ("LUẬT BẢO HIỂM XÃ HỘI (phần 1/2)", (None, None)),
        (TITLE, (None, None)),  # "ĐIỀU" trong tên văn bản không phải Điều
        ("LUẬT X - Điều 48a. Chậm đóng - Khoản 1", (None, None)),  # hậu tố chữ
        ("LUẬT X - Điều 7. Tên - Khoản 3a", (7, None)),
        ("LUẬT X - Điều 12", (12, None)),
        ("", (None, None)),
    ],
)
def test_parse_breadcrumb_dinh_dang_that(
    breadcrumb: str, expected: tuple[int | None, int | None]
):
    assert parse_breadcrumb(breadcrumb) == BreadcrumbRef(*expected)


def test_parse_breadcrumb_tren_toan_bo_corpus_that(real_chunks: list[dict[str, Any]]):
    with_dieu = with_khoan = 0
    for chunk in real_chunks:
        ref = parse_breadcrumb(chunk["breadcrumb"])
        with_dieu += ref.dieu is not None
        with_khoan += ref.khoan is not None
        assert ref.khoan is None or ref.dieu is not None
    assert with_dieu > 1500 and with_khoan > 1500
    assert with_dieu < len(real_chunks)  # còn chunk không có Điều (front/back matter)


def test_breadcrumb_structural_terms_chi_sinh_token_co_trong_breadcrumb():
    assert breadcrumb_structural_terms(
        "X - Điều 3. Tên - Khoản 1 - Điểm a (phần 1/2)"
    ) == [
        "điều_3",
        "khoản_1",
        "điều_3_khoản_1",
    ]
    assert breadcrumb_structural_terms("X - Điều 3. Tên") == ["điều_3"]
    assert breadcrumb_structural_terms("LUẬT X (phần 1/2)") == []


# ------------------------------------------------------------- token phía query


def test_token_document_va_query_cung_dinh_dang():
    doc = breadcrumb_structural_terms("X - Điều 3. Tên - Khoản 1")
    query = structural_terms([3], [1])
    assert set(doc) == set(query)


def test_structural_terms_tran_toi_da_15_va_khoan_mot_minh_khong_sinh_token():
    numbers = list(extract_citation_numbers("Điều 1, 2, 3, 4"))
    khoans = extract_citation_khoans("khoản 1, khoản 2, khoản 3, khoản 4")
    assert len(numbers) == 3 and len(khoans) == MAX_CITATION_KHOANS == 3
    assert len(structural_terms(numbers, khoans)) == 3 + 3 + 9
    assert structural_terms([], [2]) == []
    assert structural_terms([36], []) == ["điều_36"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Điều 36 khoản 2 quy định gì", [2]),
        ("khoản 1 và Khoản 3, KHOẢN 1", [1, 3]),
        ("khoản2 nói gì", [2]),
        ("khoản 2024", []),
        ("không có gì", []),
    ],
)
def test_extract_citation_khoans(query: str, expected: list[int]):
    assert extract_citation_khoans(query) == expected


def test_khong_va_cham_token_cau_truc_voi_token_pyvi(real_chunks: list[dict[str, Any]]):
    pattern = re.compile(r"^(điều|khoản)_\d+(_khoản_\d+)?$")
    for chunk in real_chunks[::7]:
        for token in tokenize(chunk["breadcrumb"] + " " + chunk["content"]):
            assert not pattern.match(token), token


# ------------------------------------------------------------- encoder


def test_encode_query_va_document_nhan_extra_terms_va_bo_term_ngoai_vocab():
    corpus = ["Điều 3 khoản 1 lương", "Điều 5 khoản 2 thuế"]
    extra = [["điều_3", "khoản_1", "điều_3_khoản_1"], ["điều_5"]]
    encoder = BM25Encoder()
    encoder.fit(corpus, extra_terms=extra)
    vocab = encoder.params.vocab
    assert {"điều_3", "khoản_1", "điều_3_khoản_1", "điều_5"} <= set(vocab)
    assert vocab["điều_3"] in encoder.encode_document(corpus[0], extra[0]).indices
    assert vocab["điều_3"] not in encoder.encode_document(corpus[1], extra[1]).indices
    query = encoder.encode_query("lương", ["điều_3", "điều_999"])
    assert vocab["điều_3"] in query.indices  # có trong vocab
    assert len(query.indices) == 2  # "điều_999" ngoài vocab bị bỏ
    assert encoder.params.params_version == BM25_PARAMS_VERSION == 2


def test_fit_extra_terms_lech_do_dai_raise():
    with pytest.raises(ValueError):
        BM25Encoder().fit(["a", "b"], extra_terms=[["x"]])


@pytest.fixture(scope="module")
def real_encoder(real_chunks: list[dict[str, Any]]):  # type: ignore[no-untyped-def]
    texts = [c["breadcrumb"] + " " + c["content"] for c in real_chunks]
    terms = [breadcrumb_structural_terms(c["breadcrumb"]) for c in real_chunks]
    encoder = BM25Encoder()
    encoder.fit(texts, extra_terms=terms)
    docs = [encoder.encode_document(t, x) for t, x in zip(texts, terms, strict=True)]
    return encoder, docs


def _ranked(real_encoder: Any, chunks: list[dict[str, Any]], query: str) -> list[int]:
    encoder, docs = real_encoder
    numbers = extract_citation_numbers(query)
    khoans = extract_citation_khoans(query)
    qv = encoder.encode_query(query, structural_terms(numbers, khoans))
    weights = dict(zip(qv.indices, qv.values, strict=True))
    scores = [
        sum(weights.get(i, 0.0) * v for i, v in zip(d.indices, d.values, strict=True))
        for d in docs
    ]
    return sorted(range(len(chunks)), key=lambda i: -scores[i])


def test_chunk_dap_an_thang_chunk_cung_so_khac_dieu_tren_corpus_that(
    real_chunks: list[dict[str, Any]], real_encoder: Any
):
    query = "Điều 3 khoản 1 của luật thuế thu nhập cá nhân quy định gì?"
    top = real_chunks[_ranked(real_encoder, real_chunks, query)[0]]
    assert top["source_document"] == "LUẬT THUẾ THU NHẬP CÁ NHÂN"
    assert parse_breadcrumb(top["breadcrumb"]) == BreadcrumbRef(3, 1)


def test_dieu_113_khoan_1_bo_luat_lao_dong_trong_top3(
    real_chunks: list[dict[str, Any]], real_encoder: Any
):
    query = "Khoản 1 Điều 113 Bộ luật Lao động nói gì?"
    order = _ranked(real_encoder, real_chunks, query)[:3]
    assert any(
        parse_breadcrumb(real_chunks[i]["breadcrumb"]) == BreadcrumbRef(113, 1)
        and real_chunks[i]["source_document"] == "BỘ LUẬT LAO ĐỘNG"
        for i in order
    )


# ------------------------------------------------------------- params_version


def _write_params(path: Path, **overrides: Any) -> None:
    encoder = BM25Encoder()
    encoder.fit(["Điều 3 lương"])
    data = encoder.params.model_dump()
    data.update(overrides)
    data = {k: v for k, v in data.items() if v is not None}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_load_params_moi_thanh_cong(tmp_path: Path):
    path = tmp_path / "p.json"
    _write_params(path)
    assert BM25Encoder.load(path).params.params_version == 2


@pytest.mark.parametrize("version", [None, 1, 3])
def test_load_params_thieu_hoac_khac_version_bao_loi_ro_co_lenh_rebuild(
    tmp_path: Path, version: int | None
):
    path = tmp_path / "p.json"
    _write_params(path, params_version=version)
    with pytest.raises(BM25ParamsVersionError) as info:
        BM25Encoder.load(path)
    assert REBUILD_COMMAND == "uv run python tools/sparse_index_documents.py"
    assert REBUILD_COMMAND in str(info.value)


def test_pipeline_khong_chay_voi_params_cu(tmp_path: Path):
    path = tmp_path / "old.json"
    _write_params(path, params_version=None)
    with pytest.raises(BM25ParamsVersionError, match="sparse_index_documents"):
        pipeline_module.RetrievalPipeline(
            hyde=FakeHyde(),  # type: ignore[arg-type]
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            dense_search=FakeDense([]),  # type: ignore[arg-type]
            reranker=object(),  # type: ignore[arg-type]
            bm25_params_path=path,
        )


# ------------------------------------------------------------- token trong pipeline


@pytest.fixture(autouse=True)
def structural_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    """id "a5_2" -> Điều 5 Khoản 2; id khác -> breadcrumb không có Điều."""

    def meta(chunk_id: str) -> Any:
        from production_legal_qa_rag.embedding.models import PineconeMetadata

        match = re.fullmatch(r"a(\d+)_(\d+)", chunk_id)
        breadcrumb = (
            f"DOC - Điều {match.group(1)}. Tên - Khoản {match.group(2)}"
            if match
            else "DOC - Phụ lục"
        )
        return PineconeMetadata(
            content=f"nd-{chunk_id}",
            breadcrumb=breadcrumb,
            source_document="sd",
            has_table=False,
        )

    monkeypatch.setattr(pipeline_fakes, "_meta", meta)


class ScoreByIdReranker:
    """Chấm điểm theo id trong `scores` (id lấy từ dòng content `nd-<id>`)."""

    def __init__(self, scores: dict[str, float], *, fail: bool = False):
        self.scores = scores
        self.fail = fail

    async def rerank(self, query: str, passages: list[str]) -> list[float] | None:
        if self.fail:
            return None
        return [self.scores.get(p.split("\nnd-")[1], 0.0) for p in passages]


def _pipe(
    dense_ids: list[str],
    scores: dict[str, float],
    *,
    sparse: dict[str, list[str]] | None = None,
    hyde: str | None = "giả định",
    fail: bool = False,
) -> tuple[pipeline_module.RetrievalPipeline, FakeSparse]:
    sparse_index = FakeSparse(sparse or {})
    pipe = pipeline_module.RetrievalPipeline(
        hyde=FakeHyde(hyde),  # type: ignore[arg-type]
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        dense_search=FakeDense(dense_ids),  # type: ignore[arg-type]
        sparse_index=sparse_index,  # type: ignore[arg-type]
        reranker=ScoreByIdReranker(scores, fail=fail),  # type: ignore[arg-type]
    )
    return pipe, sparse_index


def test_pipeline_nhanh_b_co_token_cau_truc_nhanh_a_khong():
    query = "Điều 3 khoản 1 quy định gì"
    pipe, sparse = _pipe(["z1"], {})
    asyncio.run(pipe.retrieve(query, use_mmr=False))
    assert sparse.terms[query] == structural_terms([3], [1])
    assert sparse.terms["giả định"] == []


def test_pipeline_sub_query_moi_dieu_chi_co_token_cua_dieu_do():
    query = "Điều 3 khoản 1 và Điều 5 khoản 2"
    pipe, sparse = _pipe(["z1"], {})
    asyncio.run(pipe.retrieve(query, use_mmr=False))
    sub_3, sub_5 = build_article_queries(query, [3, 5])
    assert sparse.terms[sub_3] == structural_terms([3], [1, 2])
    assert sparse.terms[sub_5] == structural_terms([5], [1, 2])
    assert sparse.terms[query] == structural_terms([3, 5], [1, 2])


def test_pipeline_cau_khong_vien_dan_khong_co_token_cau_truc():
    query = "người lao động nghỉ phép"
    pipe, sparse = _pipe(["z1"], {})
    asyncio.run(pipe.retrieve(query, use_mmr=False))
    assert all(terms == [] for terms in sparse.terms.values())


# ------------------------------------------------------------- ghim


def _ids(result: list[RetrievedChunk]) -> list[str]:
    return [c.chunk_id for c in result]


def test_ghim_chunk_khop_khoan_len_dau_du_diem_rerank_thap():
    scores = {"z1": 9, "z2": 8, "z3": 7, "a5_1": 6, "a7_2": 5, "a5_2": 1}
    pipe, _ = _pipe(list(scores), scores)
    result = asyncio.run(pipe.retrieve("Điều 5 khoản 2 quy định gì", use_mmr=False))
    # a5_2 khớp cả Khoản nên chỉ nó được ghim; a5_1 (cùng Điều) không ghim.
    assert _ids(result) == ["a5_2", "z1", "z2", "z3", "a5_1"]
    assert result[0].rerank_score == 1.0  # rerank_score không đổi


def test_ghim_theo_dieu_khi_khong_co_chunk_khop_khoan():
    scores = {"z1": 9, "z2": 8, "a5_1": 3, "a5_3": 2, "a7_2": 5}
    pipe, _ = _pipe(list(scores), scores)
    result = asyncio.run(pipe.retrieve("Điều 5 khoản 2 quy định gì", use_mmr=False))
    assert _ids(result)[:2] == ["a5_1", "a5_3"]  # theo thứ tự rerank trong Điều
    assert _ids(result)[2:] == ["z1", "z2", "a7_2"]


def test_ghim_tran_2_moi_dieu_khi_1_dieu():
    scores = {"z1": 9, "z2": 8, "a5_1": 3, "a5_2": 2, "a5_3": 1}
    pipe, _ = _pipe(list(scores), scores)
    result = asyncio.run(pipe.retrieve("Điều 5 quy định gì", use_mmr=False))
    assert PIN_PER_ARTICLE == 2
    assert _ids(result) == ["a5_1", "a5_2", "z1", "z2", "a5_3"]


def test_ghim_nhieu_dieu_xen_ke_2_cong_2_va_luon_con_cho_ngu_nghia():
    scores = {"z1": 9, "a5_1": 4, "a5_2": 3, "a5_3": 2, "a7_1": 8, "a7_2": 1}
    pipe, _ = _pipe(list(scores), scores)
    result = asyncio.run(pipe.retrieve("Điều 5 và Điều 7 quy định gì", use_mmr=False))
    # Ghim 4: (5,7) hạng nhất rồi (5,7) hạng hai; xếp theo điểm rerank.
    assert _ids(result) == ["a7_1", "a5_1", "a5_2", "a7_2", "z1"]


def test_ghim_3_dieu_lay_3_hang_nhat_roi_1_chunk_thu_hai_cua_dieu_dau():
    scores = {
        "a3_1": 9, "a3_2": 8, "a5_1": 7, "a5_2": 6, "a7_1": 5, "a7_2": 4, "z1": 10,
    }  # fmt: skip
    pipe, _ = _pipe(list(scores), scores)
    result = asyncio.run(pipe.retrieve("Điều 3, 5 và 7 quy định gì", use_mmr=False))
    assert _ids(result) == ["a3_1", "a3_2", "a5_1", "a7_1", "z1"]


def test_ghim_ap_dung_ca_khi_rerank_loi_fallback_score_none():
    ids = ["z1", "z2", "z3", "a5_2"]
    pipe, _ = _pipe(ids, {}, fail=True)
    result = asyncio.run(pipe.retrieve("Điều 5 khoản 2 quy định gì", use_mmr=False))
    assert result[0].chunk_id == "a5_2"
    assert all(c.rerank_score is None for c in result)
    assert len(result) == 4


def test_cau_khong_vien_dan_khong_ghim_giu_thu_tu_rerank():
    scores = {"z1": 9, "z2": 8, "a5_2": 1}
    pipe, _ = _pipe(list(scores), scores)
    result = asyncio.run(pipe.retrieve("quy định về phụ lục", use_mmr=False))
    assert _ids(result) == ["z1", "z2", "a5_2"]
    assert [c.rerank_score for c in result] == [9.0, 8.0, 1.0]


def test_khong_co_chunk_khop_trong_union_thi_khong_ghim_va_khong_bia():
    scores = {"z1": 9, "a7_1": 8, "a8_2": 1}
    pipe, _ = _pipe(list(scores), scores)
    result = asyncio.run(pipe.retrieve("Điều 99 quy định gì", use_mmr=False))
    assert _ids(result) == ["z1", "a7_1", "a8_2"]


def _chunk(chunk_id: str, breadcrumb: str, score: float | None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_document="sd",
        breadcrumb=breadcrumb,
        content="c",
        rerank_score=score,
    )


def test_pin_exact_matches_chunk_split_phan_cung_khoan_deu_khop():
    ranked = [
        _chunk("x", "D - Phụ lục", 9),
        _chunk("p1", "D - Điều 5. T - Khoản 2 (phần 1/2)", 2),
        _chunk("p2", "D - Điều 5. T - Khoản 2 (phần 2/2)", 1),
    ]
    out = pin_exact_matches(ranked, [5], [2], final_top_k=5)
    assert [c.chunk_id for c in out] == ["p1", "p2", "x"]


def test_pin_exact_matches_khoan_tinh_rieng_tung_dieu():
    ranked = [
        _chunk("a5_1", "D - Điều 5. T - Khoản 1", 5),
        _chunk("a5_2", "D - Điều 5. T - Khoản 2", 4),
        _chunk("a7_1", "D - Điều 7. T - Khoản 1", 3),
        _chunk("z", "D - Phụ lục", 2),
    ]
    # Điều 5 có chunk khớp Khoản 2 nên chỉ ghim a5_2; Điều 7 không có -> ghim theo Điều.
    out = pin_exact_matches(ranked, [5, 7], [2], final_top_k=5)
    assert [c.chunk_id for c in out] == ["a5_2", "a7_1", "a5_1", "z"]


# ------------------------------------------------------------- bổ sung


def test_pin_khong_ghim_chunk_hau_to_chu_dieu_48a_cho_cau_hoi_dieu_48():
    ranked = [
        _chunk("z", "D - Phụ lục", 9),
        _chunk("a48a", "D - Điều 48a. Chậm đóng - Khoản 1", 8),
        _chunk("a48", "D - Điều 48. Tên - Khoản 1", 1),
    ]
    out = pin_exact_matches(ranked, [48], [1], final_top_k=5)
    assert [c.chunk_id for c in out] == ["a48", "z", "a48a"]


def test_pin_khong_hong_khi_breadcrumb_hau_to_hoac_rong():
    ranked = [
        _chunk("e", "", 3),
        _chunk("s", "D - Điều 7. T - Khoản 3a", 2),  # Khoản 3a không parse được
    ]
    out = pin_exact_matches(ranked, [7], [3], final_top_k=5)
    # Khoản không khớp thì lùi về khớp theo Điều.
    assert [c.chunk_id for c in out] == ["s", "e"]


def test_pin_nhieu_dieu_khoan_cheo_khop_khoan_bat_ky_trong_danh_sach():
    ranked = [
        _chunk("z", "D - Phụ lục", 9),
        _chunk("a3_2", "D - Điều 3. T - Khoản 2", 5),
        _chunk("a3_1", "D - Điều 3. T - Khoản 1", 4),
        _chunk("a5_2", "D - Điều 5. T - Khoản 2", 3),
        _chunk("a5_9", "D - Điều 5. T - Khoản 9", 2),
    ]
    out = pin_exact_matches(ranked, [3, 5], [1, 2], final_top_k=5)
    # Khoản chéo: mỗi Điều khớp mọi Khoản trong danh sách; tối đa 2 mỗi Điều
    # và tổng <= final_top_k - 1 = 4; a5_9 (Khoản 9) không được ghim.
    assert [c.chunk_id for c in out][:4] == ["a3_2", "a3_1", "a5_2", "z"] or (
        {c.chunk_id for c in out[:3]} == {"a3_2", "a3_1", "a5_2"}
    )
    assert out[-1].chunk_id in {"a5_9", "z"}
    assert {c.chunk_id for c in out} == {c.chunk_id for c in ranked}
    assert [c.rerank_score for c in out if c.chunk_id == "a3_2"] == [5]


def test_pin_giu_rerank_score_va_khong_doi_tap_phan_tu():
    ranked = [_chunk("z", "D - Phụ lục", 9.5), _chunk("a", "D - Điều 5. T", -3.25)]
    out = pin_exact_matches(ranked, [5], [], final_top_k=5)
    assert [(c.chunk_id, c.rerank_score) for c in out] == [("a", -3.25), ("z", 9.5)]


def test_encode_query_extra_terms_rong_giong_het_khong_tham_so():
    encoder = BM25Encoder()
    encoder.fit(
        ["Điều 3 người lao động nghỉ", "tiền lương tối thiểu"],
        extra_terms=[["điều_3"], []],
    )
    plain = encoder.encode_query("người lao động")
    assert encoder.encode_query("người lao động", ()) == plain
    assert encoder.encode_query("người lao động", []) == plain
    with_term = encoder.encode_query("người lao động", ["điều_3"])
    assert len(with_term.indices) == len(plain.indices) + 1


def test_encode_document_tinh_dung_khi_them_token_cau_truc():
    import math

    encoder = BM25Encoder()
    texts = ["Điều 3 người lao động", "tiền lương"]
    encoder.fit(texts, extra_terms=[["điều_3", "khoản_1"], []])
    params = encoder.params
    tokens = tokenize(texts[0]) + ["điều_3", "khoản_1"]
    norm = 1 - params.b + params.b * len(tokens) / params.avgdl
    doc = encoder.encode_document(texts[0], ["điều_3", "khoản_1"])
    weights = dict(zip(doc.indices, doc.values, strict=True))
    index = params.vocab["điều_3"]
    expected = 1 * (params.k1 + 1) / (1 + params.k1 * norm)
    assert math.isclose(weights[index], expected)
    # Không truyền extra_terms thì token cấu trúc vắng và độ dài ngắn hơn.
    plain = encoder.encode_document(texts[0])
    assert index not in dict(zip(plain.indices, plain.values, strict=True))


def test_load_params_file_cu_khong_co_params_version_bao_loi_ro(tmp_path: Path):
    old = tmp_path / "bm25_params.json"
    old.write_text(
        json.dumps(
            {"vocab": {"a": 0}, "idf": {"a": 1.0}, "num_documents": 2, "avgdl": 3.0}
        ),
        encoding="utf-8",
    )
    with pytest.raises(BM25ParamsVersionError) as info:
        BM25Encoder.load(old)
    message = str(info.value)
    assert "params_version=1" in message
    assert REBUILD_COMMAND in message
