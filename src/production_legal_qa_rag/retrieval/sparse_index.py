"""Pinecone sparse index riêng cho BM25: build offline và query runtime (mục 6.2, 13A)."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException

from production_legal_qa_rag.chunking.models import Chunk
from production_legal_qa_rag.config import VectorDBSettings
from production_legal_qa_rag.retrieval.bm25 import BM25Encoder
from production_legal_qa_rag.retrieval.citation import breadcrumb_structural_terms
from production_legal_qa_rag.retrieval.models import RetrievalError, SearchHit

SPARSE_TOP_N = 20
SPARSE_UPSERT_BATCH_SIZE = 100
_INDEX_READY_TIMEOUT_SECONDS = 120.0
_INDEX_READY_POLL_INTERVAL_SECONDS = 1.0


def build_index(
    chunks_dir: Path,
    params_out_path: Path,
    settings: VectorDBSettings | None = None,
    client: Pinecone | None = None,
) -> int:
    """Fit BM25 trên toàn corpus rồi full-refresh Pinecone sparse index.

    Args:
        chunks_dir: Thư mục chứa các mảng `Chunk` JSON.
        params_out_path: Nơi ghi `bm25_params.json`.
        settings: Config Pinecone; mặc định đọc từ `.env`.
        client: Client Pinecone (để test); mặc định tạo từ `settings`.

    Returns:
        Số chunk đã upsert.

    Raises:
        ValueError: Khi không có chunk nào trong `chunks_dir`.
    """
    settings = settings or VectorDBSettings()
    client = client or Pinecone(api_key=settings.pinecone_api_key)

    chunks = _read_chunks(chunks_dir)
    if not chunks:
        raise ValueError(f"Không tìm thấy chunk nào trong {chunks_dir}")

    texts = [_bm25_text(chunk) for chunk in chunks]
    structural = [breadcrumb_structural_terms(chunk.breadcrumb) for chunk in chunks]
    encoder = BM25Encoder()
    encoder.fit(texts, extra_terms=structural)
    encoder.save(params_out_path)

    records = [
        {
            "id": chunk.chunk_id,
            "sparse_values": encoder.encode_document(text, terms).model_dump(),
        }
        for chunk, text, terms in zip(chunks, texts, structural, strict=True)
    ]

    index = _get_or_create_index(client, settings)
    _delete_all_vectors(index)
    for start in range(0, len(records), SPARSE_UPSERT_BATCH_SIZE):
        index.upsert(vectors=records[start : start + SPARSE_UPSERT_BATCH_SIZE])
    return len(records)


class SparseIndex:
    """Query runtime lên sparse index bằng BM25 encoder đã load."""

    def __init__(
        self,
        encoder: BM25Encoder,
        settings: VectorDBSettings | None = None,
        index: Any | None = None,
    ) -> None:
        if index is None:
            settings = settings or VectorDBSettings()
            index = Pinecone(api_key=settings.pinecone_api_key).Index(
                settings.sparse_index_name
            )
        self._encoder = encoder
        self._index = index

    async def query(
        self,
        text: str,
        top_k: int = SPARSE_TOP_N,
        extra_terms: Sequence[str] = (),
    ) -> list[SearchHit]:
        """Top-k chunk theo điểm BM25 của `text` (kèm token cấu trúc `extra_terms`).

        Query không có term nào trong vocabulary trả về rỗng (Pinecone từ
        chối sparse vector rỗng).

        Raises:
            RetrievalError: Khi Pinecone lỗi.
        """
        sparse_vector = self._encoder.encode_query(text, extra_terms)
        if not sparse_vector.indices:
            return []
        try:
            response = await asyncio.to_thread(
                self._index.query,
                sparse_vector=sparse_vector.model_dump(),
                top_k=top_k,
            )
        except Exception as error:
            raise RetrievalError("Pinecone sparse query lỗi.") from error
        return [
            SearchHit(chunk_id=match.id, score=match.score)
            for match in response.matches
        ]


def _read_chunks(chunks_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(chunks_dir.glob("*.json")):
        raw_chunks = json.loads(path.read_text(encoding="utf-8"))
        chunks.extend(Chunk.model_validate(raw) for raw in raw_chunks)
    return chunks


def _bm25_text(chunk: Chunk) -> str:
    """Breadcrumb đứng trước để viện dẫn Điều/Khoản match được bằng keyword."""
    return f"{chunk.breadcrumb} {chunk.content}"


def _get_or_create_index(client: Pinecone, settings: VectorDBSettings) -> Any:
    names = _index_names(client)
    if settings.sparse_index_name not in names:
        client.create_index(
            name=settings.sparse_index_name,
            metric="dotproduct",
            vector_type="sparse",
            spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
        )
        _wait_for_index_ready(client, settings.sparse_index_name)
    return client.Index(settings.sparse_index_name)


def _index_names(client: Pinecone) -> list[str]:
    indexes = client.list_indexes()
    names = getattr(indexes, "names", None)
    if callable(names):
        return list(names())
    return [str(index) for index in indexes]


def _wait_for_index_ready(client: Pinecone, name: str) -> None:
    """Chờ index vừa tạo sẵn sàng (Pinecone tạo bất đồng bộ).

    Raises:
        TimeoutError: Khi index chưa sẵn sàng trước thời hạn.
    """
    deadline = time.monotonic() + _INDEX_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            status = getattr(client.describe_index(name), "status", None)
        except NotFoundException:
            # Control plane có thể chưa thấy index vừa tạo.
            status = None
        ready = (
            status.get("ready")
            if isinstance(status, dict)
            else getattr(status, "ready", None)
        )
        if ready is True:
            return
        time.sleep(_INDEX_READY_POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Pinecone index {name!r} chưa ready sau {_INDEX_READY_TIMEOUT_SECONDS:.0f} giây."
    )


def _delete_all_vectors(index: Any) -> None:
    """Xoá vector cũ; index mới chưa có namespace mặc định được xem là đã sạch."""
    try:
        index.delete(delete_all=True)
    except NotFoundException as error:
        if "namespace not found" not in str(error).casefold():
            raise
