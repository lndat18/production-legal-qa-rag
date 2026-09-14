"""Data/Schema validation cho `chunking/models.py` (chunking_spec.md mục 2, 10)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from production_legal_qa_rag.chunking.models import (
    Chunk,
    ChunkingResult,
    DocumentTree,
    KhoanNode,
)

# ==========================================================================
# KhoanNode
# ==========================================================================


def test_khoan_node_bat_buoc_breadcrumb_prefix_va_content():
    with pytest.raises(ValidationError):
        KhoanNode()  # type: ignore[call-arg]


def test_khoan_node_khoan_number_mac_dinh_none():
    node = KhoanNode(breadcrumb_prefix="Văn bản mẫu - Điều 1", content="Nội dung.")
    assert node.khoan_number is None
    assert node.has_table is False
    assert node.raw_table is None


def test_khoan_node_chap_nhan_khoan_number_dang_chu_cai():
    node = KhoanNode(breadcrumb_prefix="x", content="y", khoan_number="48a")
    assert node.khoan_number == "48a"


# ==========================================================================
# DocumentTree
# ==========================================================================


def test_document_tree_khoans_mac_dinh_rong():
    tree = DocumentTree(source_document="41/2024/QH15")
    assert tree.khoans == []


def test_document_tree_bat_buoc_source_document():
    with pytest.raises(ValidationError):
        DocumentTree()  # type: ignore[call-arg]


# ==========================================================================
# Chunk -- field bắt buộc/mặc định (mục 2, bảng field)
# ==========================================================================


def _minimal_chunk(**overrides: object) -> Chunk:
    fields: dict[str, object] = {
        "chunk_id": "abc123",
        "source_document": "41/2024/QH15",
        "breadcrumb": "Luật mẫu (41/2024/QH15) - Điều 1 - Khoản 1",
        "content": "Nội dung.",
        "token_count": 5,
    }
    fields.update(overrides)
    return Chunk(**fields)  # type: ignore[arg-type]


def test_chunk_gia_tri_mac_dinh_dung_bang_field_muc_2():
    chunk = _minimal_chunk()
    assert chunk.has_table is False
    assert chunk.raw_table is None
    assert chunk.standardization_table is None
    assert chunk.is_split is False
    assert chunk.split_index is None
    assert chunk.split_total is None
    assert chunk.negation_note is None


def test_chunk_bat_buoc_cac_field_loi():
    for missing in (
        "chunk_id",
        "source_document",
        "breadcrumb",
        "content",
        "token_count",
    ):
        fields = {
            "chunk_id": "abc",
            "source_document": "doc",
            "breadcrumb": "bc",
            "content": "c",
            "token_count": 1,
        }
        del fields[missing]
        with pytest.raises(ValidationError):
            Chunk(**fields)  # type: ignore[arg-type]


def test_chunk_token_count_phai_la_so_nguyen():
    with pytest.raises(ValidationError):
        _minimal_chunk(token_count="không phải số")


def test_chunk_split_full_field(tmp_path=None):
    chunk = _minimal_chunk(
        is_split=True,
        split_index=1,
        split_total=2,
        negation_note="Câu phủ định.",
    )
    assert chunk.is_split is True
    assert chunk.split_index == 1
    assert chunk.split_total == 2
    assert chunk.negation_note == "Câu phủ định."


def test_chunk_co_bang_field_day_du():
    chunk = _minimal_chunk(
        has_table=True,
        raw_table="| A | B |\n| --- | --- |\n| 1 | 2 |",
        standardization_table="1 - B: 2",
    )
    assert chunk.has_table is True
    assert chunk.raw_table is not None
    assert chunk.standardization_table == "1 - B: 2"


def test_chunk_round_trip_json():
    chunk = _minimal_chunk(has_table=True, raw_table="| A |", standardization_table="A")
    restored = Chunk.model_validate(json.loads(chunk.model_dump_json()))
    assert restored == chunk


# ==========================================================================
# ChunkingResult
# ==========================================================================


def test_chunking_result_mac_dinh():
    result = ChunkingResult(source_path="data/markdown/a.md")
    assert result.chunks == []
    assert result.khoan_count == 0
    assert result.split_khoan_count == 0


def test_chunking_result_chua_danh_sach_chunk():
    chunk = _minimal_chunk()
    result = ChunkingResult(
        source_path="data/markdown/a.md",
        chunks=[chunk],
        khoan_count=1,
        split_khoan_count=0,
    )
    assert result.chunks == [chunk]
    assert result.khoan_count == 1
