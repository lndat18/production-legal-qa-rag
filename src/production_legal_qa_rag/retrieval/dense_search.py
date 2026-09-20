"""Query dense index có sẵn và fetch bổ sung vector/metadata (mục 6.1, 6.4).

Dùng Pinecone SDK đồng bộ chạy trong `asyncio.to_thread` để các nhánh chạy
song song mà không phụ thuộc client async của SDK.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from pinecone import Pinecone

from production_legal_qa_rag.config import VectorDBSettings
from production_legal_qa_rag.embedding.models import PineconeMetadata
from production_legal_qa_rag.retrieval.models import (
    Candidate,
    RetrievalError,
    SearchHit,
)

logger = logging.getLogger(__name__)

DENSE_TOP_N = 20


class DenseSearch:
    """Truy vấn dense index (cosine) đã build bởi `embedding/`."""

    def __init__(
        self,
        settings: VectorDBSettings | None = None,
        index: Any | None = None,
    ) -> None:
        if index is None:
            settings = settings or VectorDBSettings()
            index = Pinecone(api_key=settings.pinecone_api_key).Index(
                settings.index_name
            )
        self._index = index

    async def query(
        self,
        embedding: Sequence[float],
        top_k: int = DENSE_TOP_N,
        *,
        include_values: bool = False,
    ) -> list[SearchHit]:
        """Top-k theo cosine, luôn kèm metadata.

        Args:
            embedding: Vector query.
            top_k: Số kết quả.
            include_values: Lấy cả vector (chỉ cần khi MMR bật).

        Raises:
            RetrievalError: Khi Pinecone lỗi.
        """
        try:
            response = await asyncio.to_thread(
                self._index.query,
                vector=list(embedding),
                top_k=top_k,
                include_metadata=True,
                include_values=include_values,
            )
        except Exception as error:
            raise RetrievalError("Pinecone dense query lỗi.") from error
        return [
            SearchHit(
                chunk_id=match.id,
                score=match.score,
                metadata=_parse_metadata(match.metadata),
                values=list(match.values) if include_values and match.values else None,
            )
            for match in response.matches
        ]

    async def fetch(self, chunk_ids: list[str]) -> dict[str, SearchHit]:
        """Fetch vector + metadata theo id; id không tồn tại vắng mặt trong kết quả.

        Raises:
            RetrievalError: Khi Pinecone lỗi.
        """
        try:
            response = await asyncio.to_thread(self._index.fetch, ids=chunk_ids)
        except Exception as error:
            raise RetrievalError("Pinecone dense fetch lỗi.") from error
        return {
            chunk_id: SearchHit(
                chunk_id=chunk_id,
                metadata=_parse_metadata(vector.metadata),
                values=list(vector.values) if vector.values else None,
            )
            for chunk_id, vector in response.vectors.items()
        }

    async def fill_missing(
        self, candidates: list[Candidate], *, need_values: bool
    ) -> list[Candidate]:
        """Bổ sung metadata (và vector nếu `need_values`) cho candidate còn thiếu.

        Chỉ fetch khi có id thiếu; candidate mà fetch không trả về (lệch build
        giữa dense và sparse index) bị bỏ kèm warning. Thứ tự được giữ nguyên.
        """
        missing = [
            c.chunk_id
            for c in candidates
            if c.metadata is None or (need_values and c.values is None)
        ]
        if not missing:
            return candidates

        fetched = await self.fetch(missing)
        filled: list[Candidate] = []
        for candidate in candidates:
            if candidate.chunk_id in missing:
                hit = fetched.get(candidate.chunk_id)
                if hit is None or hit.metadata is None:
                    logger.warning(
                        "Bỏ chunk %s: có ở sparse nhưng không có ở dense index.",
                        candidate.chunk_id,
                    )
                    continue
                candidate = candidate.model_copy(
                    update={"metadata": hit.metadata, "values": hit.values}
                )
            filled.append(candidate)
        return filled


def _parse_metadata(raw_metadata: Any) -> PineconeMetadata | None:
    if raw_metadata is None:
        return None
    return PineconeMetadata.model_validate(dict(raw_metadata))
