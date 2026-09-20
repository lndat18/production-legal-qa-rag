"""Retrieval: HyDE -> hybrid search 2 nhánh -> RRF -> MMR -> union -> rerank."""

from __future__ import annotations

from production_legal_qa_rag.retrieval.models import RetrievalError, RetrievedChunk
from production_legal_qa_rag.retrieval.pipeline import RetrievalPipeline, retrieve

__all__ = ["RetrievalError", "RetrievalPipeline", "RetrievedChunk", "retrieve"]
