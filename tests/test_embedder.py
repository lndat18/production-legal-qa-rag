"""Bộ test cho bước embedding (Task 12 của ``embedding_spec.md``).

Bốn test đầu chạy khô: không cần PostgreSQL, không nạp 518 MB model. Test cuối
đánh dấu ``slow`` vì nó nạp model thật.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from rag.indexing.pre_embedding.config import EMBED_DIM, NORM_TOLERANCE
from rag.indexing.pre_embedding.db import PENDING_SQL, to_pgvector
from rag.indexing.pre_embedding.models import PendingChunk
from rag.indexing.pre_embedding.qc import check


@pytest.fixture(scope="session")
def chunk() -> PendingChunk:
    return PendingChunk(
        chunk_id="18/VBHN-VPQH#d105k1p1",
        embedding_text="Thời giờ làm việc bình thường không quá 08 giờ trong 01 ngày.",
        token_count=24,
    )


def _codes(issues) -> set[str]:
    return {issue.code for issue in issues}


# ========================================================================
# TASK 6 — tuần tự hoá vector
# ========================================================================
def test_to_pgvector_round_trip_khong_mat_bit():
    """``.9g`` là mức tối thiểu để round-trip một float32 không mất bit."""
    rng = np.random.default_rng(0)
    original = rng.standard_normal(EMBED_DIM).astype(np.float32)
    original /= np.linalg.norm(original)

    literal = to_pgvector(original)
    assert literal.startswith("[") and literal.endswith("]")

    parsed = np.array(
        [float(part) for part in literal[1:-1].split(",")], dtype=np.float32
    )
    assert parsed.shape == (EMBED_DIM,)
    assert struct.pack(f"{EMBED_DIM}f", *parsed) == struct.pack(f"{EMBED_DIM}f", *original)


# ========================================================================
# TASK 8 — QC trước khi ghi
# ========================================================================
def test_qc_bat_duoc_norm_off(chunk):
    """NT-3: vector norm 0.8 phải bị chặn, không được ghi xuống bảng."""
    vector = np.full(EMBED_DIM, 0.8 / np.sqrt(EMBED_DIM), dtype=np.float32)
    issues = check(chunk, vector)
    assert "norm_off" in _codes(issues)
    assert any(issue.severity == "error" for issue in issues)
    assert abs(float(np.linalg.norm(vector)) - 1.0) > NORM_TOLERANCE


def test_qc_bat_duoc_dim_mismatch(chunk):
    vector = np.full(512, 1 / np.sqrt(512), dtype=np.float32)
    issues = check(chunk, vector)
    assert "dim_mismatch" in _codes(issues)
    assert any(issue.severity == "error" for issue in issues)


# ========================================================================
# TASK 5 — chọn chunk cần embed
# ========================================================================
def test_pending_sql_dung_is_distinct_from():
    """Với cột đang NULL, ``<>`` trả NULL và hàng bị bỏ sót im lặng."""
    assert PENDING_SQL.count("IS DISTINCT FROM") == 2
    assert "embedding_model <>" not in PENDING_SQL
    assert "embedding_version <>" not in PENDING_SQL


# ========================================================================
# TASK 3 — encode thật (nạp model)
# ========================================================================
@pytest.mark.slow
def test_encode_tra_vector_768_norm_1():
    from rag.indexing.pre_embedding.encoder import encode

    vectors = encode(["Mức lương tối thiểu vùng I"])
    assert vectors.shape == (1, EMBED_DIM)
    assert abs(float(np.linalg.norm(vectors[0])) - 1.0) <= 1e-6
