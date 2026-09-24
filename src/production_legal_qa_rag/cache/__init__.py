"""Cache câu trả lời và kết quả retrieval (cache_spec.md)."""

from production_legal_qa_rag.cache.keys import compute_corpus_version
from production_legal_qa_rag.cache.models import CachedAnswer
from production_legal_qa_rag.cache.replay import replay
from production_legal_qa_rag.cache.singleflight import SingleFlight
from production_legal_qa_rag.cache.store import AnswerCache, RetrievalCache

__all__ = [
    "AnswerCache",
    "CachedAnswer",
    "RetrievalCache",
    "SingleFlight",
    "compute_corpus_version",
    "replay",
]
