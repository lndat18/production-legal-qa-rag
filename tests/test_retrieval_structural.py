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
    BreadcrumbRef,
    breadcrumb_structural_terms,
    build_article_queries,
    extract_citation_khoans,
    extract_citation_numbers,
    parse_breadcrumb,
    structural_terms,
)

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
    assert encoder.params.params_version == BM25_PARAMS_VERSION == 3


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
    assert BM25Encoder.load(path).params.params_version == 3


@pytest.mark.parametrize("version", [None, 1, 2, 4])
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
