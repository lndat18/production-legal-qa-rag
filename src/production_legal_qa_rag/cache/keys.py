"""Dựng cache key có version để không phát lại dữ liệu cũ."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from production_legal_qa_rag.cache.normalize import normalize_query

logger = logging.getLogger(__name__)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BM25_PARAMS_PATH = _REPOSITORY_ROOT / "data" / "bm25" / "bm25_params.json"
UNKNOWN_CORPUS_VERSION = "unknown"
_HASH_PREFIX_LENGTH = 32
_CORPUS_VERSION_LENGTH = 12


def compute_corpus_version(
    bm25_params_path: Path = DEFAULT_BM25_PARAMS_PATH,
    *,
    override: str | None = None,
) -> str:
    """Tính version corpus một lần lúc khởi động từ tham số BM25.

    Args:
        bm25_params_path: File sinh lại mỗi khi fit lại sparse index.
        override: Version do vận hành cung cấp qua ``CACHE_CORPUS_VERSION``.

    Returns:
        Override nguyên vẹn, hoặc 12 ký tự đầu SHA-256 của file, hoặc ``unknown``.
    """
    if override:
        return override
    try:
        contents = bm25_params_path.read_bytes()
    except OSError:
        logger.warning(
            "Không đọc được BM25 params để version cache: %s. Dùng version unknown; "
            "hãy đặt CACHE_CORPUS_VERSION nếu cần invalidation chính xác.",
            bm25_params_path,
            exc_info=True,
        )
        return UNKNOWN_CORPUS_VERSION
    return hashlib.sha256(contents).hexdigest()[:_CORPUS_VERSION_LENGTH]


def answer_key(
    standalone_query: str,
    *,
    corpus_version: str,
    prompt_version: str,
    model_name: str,
) -> str:
    """Trả Redis key cho answer cache đã bao gồm mọi input tạo câu trả lời."""
    digest = _query_digest(standalone_query)
    return f"rag:ans:{corpus_version}:{prompt_version}:{model_name}:{digest}"


def retrieval_key(standalone_query: str, *, corpus_version: str) -> str:
    """Trả Redis key cho retrieval cache theo corpus và exact query."""
    digest = _query_digest(standalone_query)
    return f"rag:ret:{corpus_version}:{digest}"


def lock_key(answer_cache_key: str) -> str:
    """Trả Redis key khoá single-flight tương ứng với một answer key."""
    return f"{answer_cache_key}:lock"


def _query_digest(standalone_query: str) -> str:
    normalized = normalize_query(standalone_query)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:_HASH_PREFIX_LENGTH]
