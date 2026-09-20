"""Model dữ liệu trao đổi giữa các module của `retrieval/` (mục 2, 6, 14).

`RetrievedChunk` là output công khai của `retrieve()`; `SearchHit` và
`Candidate` là model trung gian của các bước dense/sparse/fusion/MMR.
"""

from __future__ import annotations

from pydantic import BaseModel

from production_legal_qa_rag.embedding.models import PineconeMetadata


class RetrievalError(RuntimeError):
    """Lỗi không degrade được (HF embed hoặc Pinecone hỏng) — mục 10."""


class SparseVector(BaseModel):
    """Sparse vector dạng hai list song song, khớp `SparseValues` của Pinecone."""

    indices: list[int]
    values: list[float]


class SearchHit(BaseModel):
    """Một kết quả thô từ một lần query/fetch Pinecone.

    `metadata` là `None` với sparse index (không lưu metadata); `values` là
    `None` khi query không yêu cầu `include_values`.
    """

    chunk_id: str
    score: float = 0.0
    metadata: PineconeMetadata | None = None
    values: list[float] | None = None


class Candidate(BaseModel):
    """Candidate sau RRF: một chunk kèm điểm fusion, metadata/vector nếu đã có."""

    chunk_id: str
    rrf_score: float
    metadata: PineconeMetadata | None = None
    values: list[float] | None = None


class RetrievedChunk(BaseModel):
    """Chunk trả về cho bước generation (mục 2)."""

    chunk_id: str
    source_document: str
    breadcrumb: str
    content: str
    has_table: bool = False
    raw_table: str | None = None
    rerank_score: float | None = None
