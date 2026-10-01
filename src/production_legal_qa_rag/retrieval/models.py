"""Model dữ liệu trao đổi giữa các module của `retrieval/` (mục 2, 6, 14).

`RetrievedChunk` là output công khai của `retrieve()`; `SearchHit` và
`Candidate` là model trung gian của các bước dense/sparse/fusion/MMR.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, model_validator

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


class PrecomputedQuery(BaseModel):
    """Validated HyDE and embeddings, bypassing only online query preparation."""

    hypothetical_document: str | None = None
    query_embedding: list[float]
    hypothetical_embedding: list[float] | None = None

    @model_validator(mode="after")
    def validate_vectors(self) -> PrecomputedQuery:
        """Reject empty, non-finite or mismatched precomputed vectors."""
        vectors = [self.query_embedding]
        if self.hypothetical_document is not None:
            if self.hypothetical_embedding is None:
                raise ValueError("HyDE requires a hypothetical embedding.")
            vectors.append(self.hypothetical_embedding)
        elif self.hypothetical_embedding is not None:
            raise ValueError("A hypothetical embedding requires HyDE text.")
        if not all(vectors) or any(not math.isfinite(x) for v in vectors for x in v):
            raise ValueError("Embeddings must be nonempty finite vectors.")
        if any(len(v) != len(self.query_embedding) for v in vectors):
            raise ValueError("Precomputed embedding dimensions differ.")
        return self
