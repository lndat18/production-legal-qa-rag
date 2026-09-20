"""Điều phối `retrieve(query)`: HyDE -> hybrid 2 nhánh -> union -> rerank (mục 13B)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from production_legal_qa_rag.embedding.models import PineconeMetadata
from production_legal_qa_rag.retrieval import fusion, mmr
from production_legal_qa_rag.retrieval.bm25 import BM25Encoder
from production_legal_qa_rag.retrieval.citation import citation_extras, has_citation
from production_legal_qa_rag.retrieval.dense_search import DENSE_TOP_N, DenseSearch
from production_legal_qa_rag.retrieval.fusion import RRF_K
from production_legal_qa_rag.retrieval.hyde import HydeGenerator
from production_legal_qa_rag.retrieval.mmr import MMR_LAMBDA
from production_legal_qa_rag.retrieval.models import (
    Candidate,
    RetrievalError,
    RetrievedChunk,
    SearchHit,
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


class _BranchResult(NamedTuple):
    """Kết quả một nhánh: candidate đã chọn và sparse hits thô (trước RRF)."""

    candidates: list[Candidate]
    sparse_hits: list[SearchHit]


class RetrievalPipeline:
    """Sở hữu các client (khởi tạo 1 lần, dùng lại cho mọi query).

    Client async (Groq, httpx) được tạo lazy theo event loop hiện tại và tạo
    lại khi loop đổi, nên một instance dùng được qua nhiều `asyncio.run`.
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

        Câu hỏi viện dẫn Điều (`has_citation`) được thêm top sparse của nhánh B
        vào union (mục 8.1).

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
        results = list(await asyncio.gather(*branch_jobs))
        branches = [result.candidates for result in results]

        # Extras chỉ cho câu viện dẫn; dùng lại sparse hits của nhánh B (cuối
        # danh sách), không thêm lượt gọi sparse.
        extras = citation_extras(results[-1].sparse_hits) if has_citation(query) else []
        if extras:
            branches.append(extras)

        union = _dedupe_by_chunk_id(branches)
        if not use_mmr or extras:
            union = await self._dense_search.fill_missing(union, need_values=False)
            # Ánh xạ sang bản đã fill: object cũ của nhánh còn thiếu metadata.
            filled_by_id = {candidate.chunk_id: candidate for candidate in union}
            branches = [
                [filled_by_id[c.chunk_id] for c in branch if c.chunk_id in filled_by_id]
                for branch in branches
            ]

        return await self._rerank(query, union, branches)

    async def _run_branch(
        self,
        text: str,
        dense_embedding: Sequence[float],
        query_embedding: Sequence[float],
        use_mmr: bool,
    ) -> _BranchResult:
        """Một nhánh: dense + sparse song song -> RRF -> MMR (nếu bật).

        Trả thêm sparse hits thô để nhánh B cấp extras cho câu viện dẫn.
        """
        dense_hits, sparse_hits = await asyncio.gather(
            self._dense_search.query(
                dense_embedding, DENSE_TOP_N, include_values=use_mmr
            ),
            self._sparse_index.query(text, SPARSE_TOP_N),
        )
        fused = fusion.rrf(dense_hits, sparse_hits, k=RRF_K)[:FUSION_TOP_N]
        if use_mmr:
            fused = await self._dense_search.fill_missing(fused, need_values=True)
            selected = mmr.select(fused, query_embedding, MMR_LAMBDA, BRANCH_TOP_N)
            return _BranchResult(selected, sparse_hits)
        return _BranchResult(fused[:BRANCH_TOP_N], sparse_hits)

    async def _rerank(
        self,
        query: str,
        union: list[Candidate],
        branches: list[list[Candidate]],
    ) -> list[RetrievedChunk]:
        if not union:
            return []
        passages = build_rerank_passages(union)
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
    """Xen kẽ round-robin theo hạng (A1, B1, E1, A2, ...), bỏ chunk trùng.

    `branches` gồm nhánh A (nếu có), nhánh B và extras E (chỉ câu viện dẫn).
    """
    longest = max((len(branch) for branch in branches), default=0)
    ordered = [
        branch[rank]
        for rank in range(longest)
        for branch in branches
        if rank < len(branch)
    ]
    return _dedupe_by_chunk_id([ordered])


def build_rerank_passages(union: list[Candidate]) -> list[str]:
    """Dựng passage gửi reranker: `breadcrumb + "\\n" + content`, cùng thứ tự `union`.

    Breadcrumb giúp reranker thấy viện dẫn (Điều/Khoản) mà chỉ `content` không
    có; nó chỉ nằm ở input reranker, không đi vào `RetrievedChunk.content`.
    """
    passages = []
    for candidate in union:
        metadata = _metadata_of(candidate)
        passages.append(f"{metadata.breadcrumb}\n{metadata.content}")
    return passages


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
