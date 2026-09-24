"""Chuẩn hoá exact-match query trước khi dựng cache key."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_PATTERN = re.compile(r"\s+")
_TRAILING_QUERY_PUNCTUATION = " ?.!"


def normalize_query(standalone_query: str) -> str:
    """Chuẩn hoá câu hỏi theo chính sách cache exact-match.

    Args:
        standalone_query: Câu hỏi độc lập sau bước condense.

    Returns:
        Câu hỏi NFC, viết thường, đã gộp khoảng trắng và bỏ dấu kết thúc vô nghĩa.
    """
    normalized = unicodedata.normalize("NFC", standalone_query).lower()
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.rstrip(_TRAILING_QUERY_PUNCTUATION)
