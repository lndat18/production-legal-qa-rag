"""Redis-backed cache cho answer và kết quả retrieval."""

from __future__ import annotations

import logging

from pydantic import TypeAdapter
from redis.asyncio import Redis

from production_legal_qa_rag.cache.keys import answer_key, retrieval_key
from production_legal_qa_rag.cache.models import CachedAnswer
from production_legal_qa_rag.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

ANSWER_TTL_SECONDS = 7 * 24 * 60 * 60
RETRIEVAL_TTL_SECONDS = 24 * 60 * 60
_RETRIEVED_CHUNKS_ADAPTER = TypeAdapter(list[RetrievedChunk])


class AnswerCache:
    """Lưu câu trả lời sạch theo key có version corpus, prompt và model."""

    def __init__(
        self,
        redis: Redis,
        *,
        corpus_version: str,
        prompt_version: str,
        model_name: str,
    ) -> None:
        self._redis = redis
        self._corpus_version = corpus_version
        self._prompt_version = prompt_version
        self._model_name = model_name

    async def get(self, standalone_query: str) -> CachedAnswer | None:
        """Đọc answer cache; Redis hoặc dữ liệu lỗi được coi là miss."""
        try:
            payload = await self._redis.get(self.key_for(standalone_query))
            if payload is None:
                return None
            return CachedAnswer.model_validate_json(payload)
        except Exception:
            logger.warning(
                "Không đọc được answer cache; xử lý như cache miss.", exc_info=True
            )
            return None

    async def set(self, standalone_query: str, answer: CachedAnswer) -> None:
        """Ghi answer cache trong bảy ngày; lỗi Redis không ảnh hưởng request."""
        try:
            await self._redis.set(
                self.key_for(standalone_query),
                answer.model_dump_json(),
                ex=ANSWER_TTL_SECONDS,
            )
        except Exception:
            logger.warning("Không ghi được answer cache; bỏ qua cache.", exc_info=True)

    def key_for(self, standalone_query: str) -> str:
        """Trả answer key thực tế, dùng chung với single-flight."""
        return answer_key(
            standalone_query,
            corpus_version=self._corpus_version,
            prompt_version=self._prompt_version,
            model_name=self._model_name,
        )


class RetrievalCache:
    """Lưu danh sách chunk retrieval theo corpus version trong một ngày."""

    def __init__(self, redis: Redis, *, corpus_version: str) -> None:
        self._redis = redis
        self._corpus_version = corpus_version

    async def get(self, standalone_query: str) -> list[RetrievedChunk] | None:
        """Đọc retrieval cache; lỗi Redis/dữ liệu được coi là miss."""
        try:
            payload = await self._redis.get(self.key_for(standalone_query))
            if payload is None:
                return None
            return _RETRIEVED_CHUNKS_ADAPTER.validate_json(payload)
        except Exception:
            logger.warning(
                "Không đọc được retrieval cache; xử lý như cache miss.", exc_info=True
            )
            return None

    async def set(self, standalone_query: str, chunks: list[RetrievedChunk]) -> None:
        """Ghi retrieval cache khi có ít nhất một chunk."""
        if not chunks:
            return
        try:
            await self._redis.set(
                self.key_for(standalone_query),
                _RETRIEVED_CHUNKS_ADAPTER.dump_json(chunks),
                ex=RETRIEVAL_TTL_SECONDS,
            )
        except Exception:
            logger.warning(
                "Không ghi được retrieval cache; bỏ qua cache.", exc_info=True
            )

    def key_for(self, standalone_query: str) -> str:
        """Trả retrieval key thực tế của câu hỏi."""
        return retrieval_key(standalone_query, corpus_version=self._corpus_version)
