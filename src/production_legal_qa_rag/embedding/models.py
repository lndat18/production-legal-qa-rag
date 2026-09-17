"""Model dữ liệu cho checkpoint embedding và record Pinecone."""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from production_legal_qa_rag.chunking.models import Chunk


class EmbeddedChunk(Chunk):
    """Một ``Chunk`` đã được model embedding chuyển thành vector."""

    embedding: list[float]


class PineconeMetadata(BaseModel):
    """Metadata tối thiểu lưu cùng một vector Pinecone."""

    content: str
    breadcrumb: str
    source_document: str
    has_table: bool
    raw_table: str | None = None

    @model_validator(mode="after")
    def _table_requires_raw_table(self) -> PineconeMetadata:
        if self.has_table and self.raw_table is None:
            raise ValueError("raw_table bắt buộc khi has_table=True")
        return self


class PineconeRecord(BaseModel):
    """Record đã validate, sẵn sàng gửi tới Pinecone."""

    id: str
    values: list[float]
    metadata: PineconeMetadata
