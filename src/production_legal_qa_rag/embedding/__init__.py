"""Sinh embedding cho chunk pháp luật và upsert chúng lên Pinecone."""

from __future__ import annotations

from production_legal_qa_rag.embedding.models import EmbeddedChunk, PineconeRecord
from production_legal_qa_rag.embedding.pipeline import embed, upsert

__all__ = ["EmbeddedChunk", "PineconeRecord", "embed", "upsert"]
