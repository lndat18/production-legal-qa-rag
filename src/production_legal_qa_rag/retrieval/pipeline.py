"""Điều phối `retrieve(query)`: HyDE -> hybrid 2 nhánh -> union -> rerank (mục 13B)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

from production_legal_qa_rag.embedding.models import PineconeMetadata
from production_legal_qa_rag.retrieval import fusion, mmr
from production_legal_qa_rag.retrieval.bm25 import BM25Encoder
from production_legal_qa_rag.retrieval.dense_search import DENSE_TOP_N, DenseSearch
from production_legal_qa_rag.retrieval.fusion import RRF_K
from production_legal_qa_rag.retrieval.hyde import HydeGenerator
from production_legal_qa_rag.retrieval.mmr import MMR_LAMBDA
from production_legal_qa_rag.retrieval.models import (
    Candidate,
    RetrievalError,
    RetrievedChunk,
)
from production_legal_qa_rag.retrieval.query_embedder import QueryEmbedder
from production_legal_qa_rag.retrieval.reranker_client import RerankerClient
from production_legal_qa_rag.retrieval.sparse_index import SPARSE_TOP_N, SparseIndex

logger = logging.getLogger(__name__)

FUSION_TOP_N = 20
BRANCH_TOP_N = 10
FINAL_TOP_K = 5
USE_MMR = True
DEFAULT_BM25_PARAMS_PATH = Path("data/bm25/bm25_params.json")


class RetrievalPipeline:
    """Sở hữu các client (khởi tạo 1 lần, dùng lại cho mọi query).

    Client async (Groq, httpx) gắn với event loop đầu tiên dùng chúng, nên
    một instance nên được dùng trong một `asyncio.run` duy nhất.
    """

    def __init__(
        self,
        *,
        hyde: HydeGenerator | None = None,
        embedder: QueryEmbedder | None = None,
        dense_search: DenseSearch | None = None,
        sparse_index: SparseIndex | None = None,
        reranker: RerankerClient | None = None,
        bm25_params_path: Path = DEFAULT_BM25_PARAMS_PATH,
    ) -> None:
        self._hyde = hyde or HydeGenerator()
        self._embedder = embedder or QueryEmbedder()
        self._dense_search = dense_search or DenseSearch()
        self._sparse_index = sparse_index or SparseIndex(
            BM25Encoder.load(bm25_params_path)
        )
        self._reranker = reranker or RerankerClient()

    async def retrieve(
        self, query: str, *, use_mmr: bool | None = None
    ) -> list[RetrievedChunk]:
        """Trả về tối đa `FINAL_TOP_K` chunk liên quan nhất tới `query`.

        Args:
            query: Một câu hỏi tiếng Việt độc lập.
            use_mmr: Ghi đè công tắc MMR; `None` dùng `USE_MMR`.

        Raises:
            RetrievalError: Khi HF embed hoặc Pinecone lỗi (Groq và reranker
                lỗi chỉ degrade, không raise).
        """
        use_mmr = USE_MMR if use_mmr is None else use_mmr

        hypothetical_document = await self._hyde.generate(query)
        texts = (
            [query] if hypothetical_document is None else [hypothetical_document, query]
        )
        embeddings = await self._embedder.embed(texts)
        query_embedding = embeddings[-1]

        branch_jobs = [
            self._run_branch(query, query_embedding, query_embedding, use_mmr)
        ]
        if hypothetical_document is not None:
            branch_jobs.insert(
                0,
                self._run_branch(
                    hypothetical_document, embeddings[0], query_embedding, use_mmr
                ),
            )
        branches = list(await asyncio.gather(*branch_jobs))

        union = _dedupe_by_chunk_id(branches)
        if not use_mmr:
            union = await self._dense_search.fill_missing(union, need_values=False)
            kept_ids = {candidate.chunk_id for candidate in union}
            branches = [
                [c for c in branch if c.chunk_id in kept_ids] for branch in branches
            ]

        return await self._rerank(query, union, branches)

    async def _run_branch(
        self,
        text: str,
        dense_embedding: Sequence[float],
        query_embedding: Sequence[float],
        use_mmr: bool,
    ) -> list[Candidate]:
        """Một nhánh: dense + sparse song song -> RRF -> MMR (nếu bật)."""
        dense_hits, sparse_hits = await asyncio.gather(
            self._dense_search.query(
                dense_embedding, DENSE_TOP_N, include_values=use_mmr
            ),
            self._sparse_index.query(text, SPARSE_TOP_N),
        )
        fused = fusion.rrf(dense_hits, sparse_hits, k=RRF_K)[:FUSION_TOP_N]
        if use_mmr:
            fused = await self._dense_search.fill_missing(fused, need_values=True)
            return mmr.select(fused, query_embedding, MMR_LAMBDA, BRANCH_TOP_N)
        return fused[:BRANCH_TOP_N]

    async def _rerank(
        self,
        query: str,
        union: list[Candidate],
        branches: list[list[Candidate]],
    ) -> list[RetrievedChunk]:
        if not union:
            return []
        passages = [_metadata_of(candidate).content for candidate in union]
        scores = await self._reranker.rerank(query, passages)

        if scores is None:
            logger.warning("Rerank thất bại, fallback xen kẽ các nhánh.")
            fallback = _interleave(branches)[:FINAL_TOP_K]
            return [_to_retrieved_chunk(c, None) for c in fallback]

        ranked = sorted(zip(union, scores, strict=True), key=lambda pair: -pair[1])
        return [_to_retrieved_chunk(c, score) for c, score in ranked[:FINAL_TOP_K]]


_default_pipeline: RetrievalPipeline | None = None


async def retrieve(query: str, *, use_mmr: bool | None = None) -> list[RetrievedChunk]:
    """Entry point của retrieval; tạo pipeline mặc định ở lần gọi đầu tiên.

    Args:
        query: Một câu hỏi tiếng Việt độc lập.
        use_mmr: Ghi đè công tắc MMR; `None` dùng `USE_MMR`.

    Returns:
        Tối đa `FINAL_TOP_K` chunk, giảm dần theo độ liên quan.

    Raises:
        RetrievalError: Khi HF embed hoặc Pinecone lỗi.
    """
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = RetrievalPipeline()
    return await _default_pipeline.retrieve(query, use_mmr=use_mmr)


def _dedupe_by_chunk_id(branches: list[list[Candidate]]) -> list[Candidate]:
    """Gộp các nhánh (theo thứ tự nhánh rồi hạng), mỗi chunk 1 lần."""
    seen: dict[str, Candidate] = {}
    for branch in branches:
        for candidate in branch:
            seen.setdefault(candidate.chunk_id, candidate)
    return list(seen.values())


def _interleave(branches: list[list[Candidate]]) -> list[Candidate]:
    """Xen kẽ theo hạng (A1, B1, A2, B2, ...), bỏ chunk trùng."""
    longest = max((len(branch) for branch in branches), default=0)
    ordered = [
        branch[rank]
        for rank in range(longest)
        for branch in branches
        if rank < len(branch)
    ]
    return _dedupe_by_chunk_id([ordered])


def _metadata_of(candidate: Candidate) -> PineconeMetadata:
    """Metadata của candidate; luôn có sau bước fill_missing/dense query."""
    if candidate.metadata is None:
        raise RetrievalError(f"Candidate {candidate.chunk_id} thiếu metadata.")
    return candidate.metadata


def _to_retrieved_chunk(
    candidate: Candidate, rerank_score: float | None
) -> RetrievedChunk:
    metadata = _metadata_of(candidate)
    return RetrievedChunk(
        chunk_id=candidate.chunk_id,
        source_document=metadata.source_document,
        breadcrumb=metadata.breadcrumb,
        content=metadata.content,
        has_table=metadata.has_table,
        raw_table=metadata.raw_table,
        rerank_score=rerank_score,
    )
