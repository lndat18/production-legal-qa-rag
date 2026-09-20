"""Embed nhiều chuỗi query trong 1 request HF Inference API (mục 5)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from huggingface_hub import InferenceClient
from pyvi import ViTokenizer

from production_legal_qa_rag.config import EmbeddingSettings
from production_legal_qa_rag.retrieval.models import RetrievalError

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_TIMEOUT_SECONDS = 30


class QueryEmbedder:
    """Embed query bằng cùng model và tiền xử lý pyvi như lúc index."""

    def __init__(
        self,
        settings: EmbeddingSettings | None = None,
        client: InferenceClient | None = None,
    ) -> None:
        settings = settings or EmbeddingSettings()
        self._client = client or InferenceClient(
            model=settings.model_name,
            token=settings.hf_token,
            timeout=float(_TIMEOUT_SECONDS),
        )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed `texts` trong đúng 1 request (retry khi lỗi).

        Word-segment bằng `ViTokenizer` trước khi gửi; thiếu bước này vector
        query lệch không gian với vector chunk mà không báo lỗi.

        Args:
            texts: Các chuỗi cần embed.

        Returns:
            Vector theo đúng thứ tự `texts`.

        Raises:
            RetrievalError: Khi HF lỗi sau hết retry hoặc response sai dạng.
        """
        segmented = [ViTokenizer.tokenize(text) for text in texts]
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 2):
            try:
                response = await asyncio.to_thread(
                    self._client.feature_extraction, segmented
                )
                return _coerce_embeddings(response, expected_count=len(texts))
            except Exception as error:
                last_error = error
                logger.warning(
                    "HF embed query lỗi ở lần thử %d/%d",
                    attempt,
                    _MAX_RETRIES + 1,
                    exc_info=True,
                )
        raise RetrievalError("HF embed query lỗi sau khi hết retry.") from last_error


def _coerce_embeddings(response: object, *, expected_count: int) -> list[list[float]]:
    """Validate response HF thành list vector float theo đúng thứ tự input."""
    raw = response.tolist() if hasattr(response, "tolist") else response
    if not isinstance(raw, list) or len(raw) != expected_count:
        raise ValueError(f"HF response không có đúng {expected_count} vector.")
    embeddings: list[list[float]] = []
    for vector in raw:
        if not isinstance(vector, list):
            raise TypeError("HF response chứa vector không phải list.")
        embeddings.append([float(value) for value in vector])
    return embeddings
