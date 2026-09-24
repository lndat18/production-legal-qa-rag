"""Phát lại cached answer như một luồng token nhẹ."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from typing import Final

from production_legal_qa_rag.cache.models import CachedAnswer
from production_legal_qa_rag.generation.models import (
    CitationsEvent,
    DoneEvent,
    GenerationEvent,
    TokenEvent,
)

REPLAY_DELAY_SECONDS: Final = 0.015
WORDS_PER_TOKEN_EVENT: Final = 4
_TEXT_PIECE_PATTERN = re.compile(r"\s+|\S+")


async def replay(answer: CachedAnswer) -> AsyncIterator[GenerationEvent]:
    """Phát lại answer theo mẩu khoảng bốn từ, rồi citation và done.

    Args:
        answer: Câu trả lời sạch đã deserialize từ answer cache.

    Yields:
        Token events tái tạo nguyên vẹn text, sau đó citations và done không usage.
    """
    first_token = True
    for text in _token_texts(answer.text):
        if not first_token:
            await asyncio.sleep(REPLAY_DELAY_SECONDS)
        yield TokenEvent(text=text)
        first_token = False
    yield CitationsEvent(citations=answer.citations)
    yield DoneEvent()


def _token_texts(text: str) -> list[str]:
    """Chia text mà không làm mất hay đổi bất kỳ whitespace nào."""
    tokens: list[str] = []
    current: list[str] = []
    word_count = 0
    for piece in _TEXT_PIECE_PATTERN.findall(text):
        if not piece.isspace() and word_count == WORDS_PER_TOKEN_EVENT:
            tokens.append("".join(current))
            current = []
            word_count = 0
        current.append(piece)
        if not piece.isspace():
            word_count += 1
    if current:
        tokens.append("".join(current))
    return tokens
