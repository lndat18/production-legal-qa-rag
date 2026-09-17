"""Wrapper HuggingFace Inference API cho embedding tiếng Việt theo batch."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from huggingface_hub import InferenceClient
from pyvi import ViTokenizer

from production_legal_qa_rag.chunking.models import Chunk
from production_legal_qa_rag.config import EmbeddingSettings
from production_legal_qa_rag.embedding.models import EmbeddedChunk

logger = logging.getLogger(__name__)

HF_BATCH_SIZE = 25
HF_RPD_SAFE_LIMIT = 900
_MAX_RETRIES = 2
_TIMEOUT_SECONDS = 30


class HFRequestLimitExceeded(RuntimeError):
    """Được raise khi một lần chạy sắp vượt quota request HF an toàn."""


class HuggingFaceEmbedder:
    """Sinh vector cho chunk bằng một client HF và bộ đếm quota dùng chung."""

    def __init__(
        self,
        settings: EmbeddingSettings | None = None,
        client: InferenceClient | None = None,
    ) -> None:
        self._settings = settings or EmbeddingSettings()
        self._client = client or InferenceClient(
            model=self._settings.model_name,
            token=self._settings.hf_token,
            timeout=float(_TIMEOUT_SECONDS),
        )
        self._request_count = 0

    @property
    def request_count(self) -> int:
        """Trả về tổng số HTTP request đã gửi trong lần chạy hiện tại."""
        return self._request_count

    def embed_chunks(
        self, chunks: list[Chunk]
    ) -> tuple[list[EmbeddedChunk], list[str]]:
        """Embed các chunk theo batch, bỏ qua toàn bộ batch lỗi.

        Args:
            chunks: Chunk nguồn theo thứ tự cần giữ trong checkpoint.

        Returns:
            Cặp ``(embedded_chunks, skipped_chunk_ids)``. Chunk của batch lỗi
            không có trong phần tử đầu tiên.

        Raises:
            HFRequestLimitExceeded: Khi lần gọi kế tiếp vượt giới hạn RPD an toàn.
        """
        embedded_chunks: list[EmbeddedChunk] = []
        skipped_chunk_ids: list[str] = []

        for batch in _batched(chunks, HF_BATCH_SIZE):
            embeddings = self._embed_batch_with_retry(batch)
            if embeddings is None:
                chunk_ids = [chunk.chunk_id for chunk in batch]
                skipped_chunk_ids.extend(chunk_ids)
                logger.warning("Bỏ qua HF batch lỗi: chunk_id=%s", ", ".join(chunk_ids))
                continue

            embedded_chunks.extend(
                EmbeddedChunk(
                    **chunk.model_dump(),
                    embedding=embedding,
                )
                for chunk, embedding in zip(batch, embeddings, strict=True)
            )

        return embedded_chunks, skipped_chunk_ids

    def _embed_batch_with_retry(self, batch: list[Chunk]) -> list[list[float]] | None:
        segmented_contents = [ViTokenizer.tokenize(chunk.content) for chunk in batch]

        for attempt in range(1, _MAX_RETRIES + 2):
            try:
                self._consume_request_budget()
                response = self._client.feature_extraction(segmented_contents)
                return _coerce_embeddings(response, expected_count=len(batch))
            except HFRequestLimitExceeded:
                raise
            except Exception:
                logger.warning(
                    "HF batch lỗi ở lần thử %d/%d: chunk_id=%s",
                    attempt,
                    _MAX_RETRIES + 1,
                    ", ".join(chunk.chunk_id for chunk in batch),
                    exc_info=True,
                )

        return None

    def _consume_request_budget(self) -> None:
        if self._request_count >= HF_RPD_SAFE_LIMIT:
            raise HFRequestLimitExceeded(
                f"Đã đạt HF_RPD_SAFE_LIMIT={HF_RPD_SAFE_LIMIT}; dừng để tránh vượt quota HF."
            )
        self._request_count += 1


def embed_chunks(chunks: list[Chunk]) -> tuple[list[EmbeddedChunk], list[str]]:
    """Embed một danh sách chunk bằng client mới cho một lần gọi độc lập.

    Pipeline dùng trực tiếp một ``HuggingFaceEmbedder`` dùng chung cho mọi
    file để bộ đếm quota không bị reset giữa các file.
    """
    return HuggingFaceEmbedder().embed_chunks(chunks)


def _batched(items: Sequence[Chunk], size: int) -> list[list[Chunk]]:
    """Chia ``items`` thành các batch liên tiếp có kích thước tối đa ``size``."""
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def _coerce_embeddings(response: object, *, expected_count: int) -> list[list[float]]:
    """Validate response HF thành list vector float theo đúng thứ tự input."""
    raw_embeddings = response.tolist() if hasattr(response, "tolist") else response
    if not isinstance(raw_embeddings, list) or len(raw_embeddings) != expected_count:
        raise ValueError(
            f"HF response không có đúng số vector mong đợi ({expected_count} vector)."
        )

    embeddings: list[list[float]] = []
    for embedding in raw_embeddings:
        if not isinstance(embedding, list):
            raise TypeError("HF response chứa vector không phải list.")
        embeddings.append([float(value) for value in embedding])
    return embeddings
