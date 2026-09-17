"""Tạo, xoá và upsert Pinecone index cho các checkpoint embedding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pinecone import Pinecone, ServerlessSpec
from transformers import AutoConfig

from production_legal_qa_rag.config import EmbeddingSettings, VectorDBSettings
from production_legal_qa_rag.embedding.models import (
    EmbeddedChunk,
    PineconeMetadata,
    PineconeRecord,
)

PINECONE_UPSERT_BATCH_SIZE = 100


class PineconeVectorStore:
    """Quản lý một index Pinecone và full-refresh các vector của nó."""

    def __init__(
        self,
        vector_settings: VectorDBSettings | None = None,
        embedding_settings: EmbeddingSettings | None = None,
        client: Pinecone | None = None,
    ) -> None:
        self._vector_settings = vector_settings or VectorDBSettings()
        self._embedding_settings = embedding_settings or EmbeddingSettings()
        self._client = client or Pinecone(
            api_key=self._vector_settings.pinecone_api_key
        )

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Xoá index hiện có rồi upsert toàn bộ embedded chunk theo batch.

        Args:
            embedded_chunks: Toàn bộ checkpoint cần đưa lên index.
        """
        index = self._get_or_create_index()
        index.delete(delete_all=True)

        records = [to_pinecone_record(chunk) for chunk in embedded_chunks]
        for batch in _batched(records, PINECONE_UPSERT_BATCH_SIZE):
            index.upsert(
                vectors=[
                    record.model_dump(mode="json", exclude_none=True)
                    for record in batch
                ]
            )

    def _get_or_create_index(self) -> Any:
        if self._vector_settings.index_name not in self._index_names():
            dimension = get_embedding_dimension(self._embedding_settings.model_name)
            self._client.create_index(
                name=self._vector_settings.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self._vector_settings.cloud,
                    region=self._vector_settings.region,
                ),
            )
        return self._client.Index(self._vector_settings.index_name)

    def _index_names(self) -> list[str]:
        indexes = self._client.list_indexes()
        names = getattr(indexes, "names", None)
        if callable(names):
            return list(names())
        return [str(index) for index in indexes]


def get_embedding_dimension(model_name: str) -> int:
    """Đọc ``hidden_size`` từ config model trên HuggingFace Hub.

    Args:
        model_name: Tên model embedding cần tạo index cho.

    Returns:
        Dimension vector công bố bởi model.

    Raises:
        ValueError: Khi model config không có ``hidden_size`` nguyên dương.
    """
    hidden_size = getattr(AutoConfig.from_pretrained(model_name), "hidden_size", None)
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError(f"Model {model_name!r} không có hidden_size hợp lệ.")
    return hidden_size


def to_pinecone_record(embedded_chunk: EmbeddedChunk) -> PineconeRecord:
    """Chuyển một checkpoint thành record Pinecone đã validate."""
    metadata = PineconeMetadata(
        content=embedded_chunk.content,
        breadcrumb=embedded_chunk.breadcrumb,
        source_document=embedded_chunk.source_document,
        has_table=embedded_chunk.has_table,
        raw_table=embedded_chunk.raw_table if embedded_chunk.has_table else None,
    )
    return PineconeRecord(
        id=embedded_chunk.chunk_id,
        values=embedded_chunk.embedding,
        metadata=metadata,
    )


def upsert_embedded_chunks(embedded_chunks: list[EmbeddedChunk]) -> None:
    """Full-refresh Pinecone bằng một danh sách checkpoint embedding."""
    PineconeVectorStore().upsert(embedded_chunks)


def _batched(items: Sequence[PineconeRecord], size: int) -> list[list[PineconeRecord]]:
    """Chia record theo batch gửi Pinecone."""
    return [list(items[index : index + size]) for index in range(0, len(items), size)]
