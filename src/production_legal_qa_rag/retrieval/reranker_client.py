"""HTTP client gọi reranker tự host trên LightningAI (mục 9)."""

from __future__ import annotations

import asyncio
import logging
import math

import httpx

from production_legal_qa_rag.config import RerankerSettings
from production_legal_qa_rag.retrieval.loop_bound import LoopBoundClient

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})
_BACKOFF_SECONDS = 0.5
_RUNBOOK_HINT = "Kiểm tra Studio/tmux/ngrok theo runbook retrieval_spec.md mục 9.1."


class _NonRetryableRerankError(Exception):
    """Lỗi cấu hình/payload/response: retry vô ích, fallback ngay."""


class _EndpointNotFoundError(_NonRetryableRerankError):
    """HTTP 404 — ngrok trả 404 khi tunnel offline hoặc Studio đang sleep."""


class RerankerClient:
    """Client rerank 1 request cho toàn bộ passage, với retry và validate."""

    def __init__(
        self,
        settings: RerankerSettings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or RerankerSettings()
        self._client = LoopBoundClient(self._create_client, client)

    def _create_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(
                float(self._settings.timeout_seconds),
                connect=float(self._settings.connect_timeout_seconds),
            )
        )

    async def rerank(self, query: str, passages: list[str]) -> list[float] | None:
        """Chấm điểm từng passage theo câu hỏi gốc.

        Args:
            query: Câu hỏi gốc (không phải hypothetical document).
            passages: Passage đã dựng `breadcrumb + "\\n" + content` của từng
                chunk, cùng thứ tự với union.

        Returns:
            Score cùng thứ tự `passages`, hoặc `None` khi thất bại và caller
            phải fallback (mục 9). Không bao giờ raise.
        """
        if not passages:
            return []

        studio_hint = ""
        for attempt in range(self._settings.max_retries + 1):
            try:
                return await self._request_scores(query, passages)
            except _EndpointNotFoundError:
                logger.error(
                    "Reranker trả HTTP 404, không retry: tunnel ngrok có thể đã "
                    "offline hoặc Studio đang sleep. %s",
                    _RUNBOOK_HINT,
                )
                return None
            except _NonRetryableRerankError as error:
                logger.error("Reranker lỗi cấu hình/payload, không retry: %s", error)
                return None
            except (httpx.TransportError, _RetryableStatusError) as error:
                logger.warning(
                    "Reranker lỗi tạm thời ở lần thử %d/%d: %r",
                    attempt + 1,
                    self._settings.max_retries + 1,
                    error,
                )
                studio_hint = _RUNBOOK_HINT
                if attempt < self._settings.max_retries:
                    await asyncio.sleep(_BACKOFF_SECONDS * (2**attempt))
            except Exception:
                # Lỗi ngoài httpx thường là lỗi lập trình: kèm traceback, không
                # gợi ý kiểm tra Studio.
                logger.warning(
                    "Reranker gặp lỗi không mong đợi ở lần thử %d/%d",
                    attempt + 1,
                    self._settings.max_retries + 1,
                    exc_info=True,
                )
                studio_hint = ""
                if attempt < self._settings.max_retries:
                    await asyncio.sleep(_BACKOFF_SECONDS * (2**attempt))

        logger.warning("Reranker hết retry, dùng fallback. %s", studio_hint)
        return None

    async def _request_scores(self, query: str, passages: list[str]) -> list[float]:
        response = await self._client.get().post(
            self._settings.endpoint_url,
            headers={"X-API-Key": self._settings.api_key},
            json={"query": query, "passages": passages},
        )
        status = response.status_code
        if status in _RETRYABLE_STATUS_CODES:
            raise _RetryableStatusError(f"HTTP {status}")
        if status >= 400:
            if status == 404:
                raise _EndpointNotFoundError("HTTP 404")
            raise _NonRetryableRerankError(f"HTTP {status}")
        return _validate_scores(response, expected_count=len(passages))


class _RetryableStatusError(Exception):
    """HTTP 502/503/504 — server chưa sẵn sàng hoặc quá tải."""


def _validate_scores(response: httpx.Response, *, expected_count: int) -> list[float]:
    """Kiểm tra `scores` là list số hữu hạn, đúng độ dài passages."""
    try:
        scores = response.json()["scores"]
    except (ValueError, KeyError, TypeError) as error:
        raise _NonRetryableRerankError("Response không có `scores` hợp lệ.") from error
    if (
        not isinstance(scores, list)
        or len(scores) != expected_count
        or not all(
            isinstance(s, int | float) and not isinstance(s, bool) and math.isfinite(s)
            for s in scores
        )
    ):
        raise _NonRetryableRerankError(
            f"`scores` phải là list {expected_count} số hữu hạn."
        )
    return [float(s) for s in scores]
