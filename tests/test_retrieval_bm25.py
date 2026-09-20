"""Unit test cho BM25 encoder, RRF và MMR của `retrieval/` (mục 6.2, 6.3, 7)."""

from __future__ import annotations

import math

from production_legal_qa_rag.retrieval import bm25, fusion, mmr
from production_legal_qa_rag.retrieval.bm25 import BM25Encoder
from production_legal_qa_rag.retrieval.models import Candidate, SearchHit

CORPUS = [
    "Điều 3 - Khoản 1 người lao động được nghỉ hằng năm",
    "Điều 4 - Khoản 2 do bị tai nạn lao động",
    "Điều 5 - Khoản 3 mức lương tối thiểu vùng",
]


def _score(encoder: BM25Encoder, query: str, document: str) -> float:
    q = encoder.encode_query(query)
    d = encoder.encode_document(document)
    doc_weights = dict(zip(d.indices, d.values, strict=True))
    return sum(
        w * doc_weights.get(i, 0.0) for i, w in zip(q.indices, q.values, strict=True)
    )


def test_diem_dot_product_khop_cong_thuc_bm25_tinh_tay():
    encoder = BM25Encoder()
    encoder.fit(CORPUS)
    params = encoder.params
    term = "lương"
    tokens = bm25.tokenize(CORPUS[2])
    tf = tokens.count(term)
    idf = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)
    norm = 1 - params.b + params.b * len(tokens) / params.avgdl
    expected = idf * tf * (params.k1 + 1) / (tf + params.k1 * norm)
    assert math.isclose(_score(encoder, term, CORPUS[2]), expected)


def test_term_ngoai_vocabulary_bi_bo():
    encoder = BM25Encoder()
    encoder.fit(CORPUS)
    assert encoder.encode_query("zzzz qqqq").indices == []


def test_tu_do_khong_bi_loai():
    assert "do" in bm25.tokenize("do bị tai nạn")


def test_vien_dan_dieu_khoan_xep_chunk_tuong_ung_len_dau():
    encoder = BM25Encoder()
    encoder.fit(CORPUS)
    scores = [_score(encoder, "Điều 5 khoản 3", text) for text in CORPUS]
    assert scores.index(max(scores)) == 2


def test_save_load_giu_nguyen_tham_so(tmp_path):
    encoder = BM25Encoder()
    encoder.fit(CORPUS)
    path = tmp_path / "bm25" / "params.json"
    encoder.save(path)
    assert BM25Encoder.load(path).params == encoder.params


def test_rrf_cong_diem_qua_hai_danh_sach():
    dense = [SearchHit(chunk_id="a"), SearchHit(chunk_id="b")]
    sparse = [SearchHit(chunk_id="b"), SearchHit(chunk_id="c")]
    result = fusion.rrf(dense, sparse, k=60)
    assert [c.chunk_id for c in result] == ["b", "a", "c"]
    assert math.isclose(result[0].rrf_score, 1 / 62 + 1 / 61)


def test_mmr_tranh_chunk_trung_lap():
    candidates = [
        Candidate(chunk_id="a", rrf_score=1, values=[1.0, 0.2]),
        Candidate(chunk_id="a2", rrf_score=1, values=[1.0, 0.19]),
        Candidate(chunk_id="b", rrf_score=1, values=[0.2, 1.0]),
    ]
    picked = mmr.select(candidates, [1.0, 1.0], 0.5, 2)
    assert [c.chunk_id for c in picked] == ["a", "b"]
