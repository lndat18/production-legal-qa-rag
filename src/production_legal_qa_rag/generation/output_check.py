"""Hậu kiểm trích dẫn và con số của câu trả lời đã stream."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from production_legal_qa_rag.generation.models import Citation
from production_legal_qa_rag.retrieval.models import RetrievedChunk

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_NUMBER_PATTERN = re.compile(
    r"(?<!\w)(?:\d{1,3}(?:[.,\s]\d{3})+|\d+(?:[.,]\d+)?)(?!\w)"
)
_YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")
_YEAR_PREFIX_PATTERN = re.compile(r"năm\s*$", re.IGNORECASE)
_NUMBER_SEPARATORS_PATTERN = re.compile(r"[.,\s]")


class OutputWarning(BaseModel):
    """Một cảnh báo được tạo bởi hậu kiểm thuần Python."""

    code: Literal["invalid_citation", "unverified_number"]
    message: str
    detail: str = ""


class OutputCheckResult(BaseModel):
    """Kết quả hậu kiểm gồm citation hợp lệ và mọi cảnh báo phát hiện."""

    citations: list[Citation] = Field(default_factory=list)
    warnings: list[OutputWarning] = Field(default_factory=list)


def check_output(text: str, chunks: list[RetrievedChunk]) -> OutputCheckResult:
    """Kiểm tra citation và số có căn cứ trong context.

    Args:
        text: Toàn bộ nội dung đã stream từ model.
        chunks: Các chunk thực tế được đưa vào prompt.

    Returns:
        Citation hợp lệ theo thứ tự xuất hiện và các cảnh báo hậu kiểm.
    """
    citations, invalid_numbers = _extract_citations(text, chunks)
    warnings: list[OutputWarning] = []
    if invalid_numbers:
        detail = ", ".join(str(number) for number in invalid_numbers)
        warnings.append(
            OutputWarning(
                code="invalid_citation",
                message="Câu trả lời có trích dẫn ngoài phạm vi context.",
                detail=detail,
            )
        )

    unverified_numbers = _find_unverified_numbers(text, chunks)
    if unverified_numbers:
        warnings.append(
            OutputWarning(
                code="unverified_number",
                message="Câu trả lời có con số không tìm thấy trong context.",
                detail=", ".join(unverified_numbers),
            )
        )
    return OutputCheckResult(citations=citations, warnings=warnings)


def _extract_citations(
    text: str, chunks: list[RetrievedChunk]
) -> tuple[list[Citation], list[int]]:
    """Lấy citation hợp lệ duy nhất và số citation không hợp lệ."""
    citations: list[Citation] = []
    invalid_numbers: list[int] = []
    seen_valid: set[int] = set()
    seen_invalid: set[int] = set()
    for match in _CITATION_PATTERN.finditer(text):
        number = int(match.group(1))
        if 1 <= number <= len(chunks):
            if number not in seen_valid:
                chunk = chunks[number - 1]
                citations.append(
                    Citation(
                        n=number,
                        chunk_id=chunk.chunk_id,
                        source_document=chunk.source_document,
                        breadcrumb=chunk.breadcrumb,
                    )
                )
                seen_valid.add(number)
        elif number not in seen_invalid:
            invalid_numbers.append(number)
            seen_invalid.add(number)
    return citations, invalid_numbers


def _find_unverified_numbers(text: str, chunks: list[RetrievedChunk]) -> list[str]:
    """Trả các số câu trả lời không có dạng chuẩn hoá trong context."""
    answer_without_citations = _CITATION_PATTERN.sub("", text)
    context_numbers = _context_numbers(chunks)
    unverified: list[str] = []
    seen: set[str] = set()
    for match in _NUMBER_PATTERN.finditer(answer_without_citations):
        value = match.group(0)
        normalized = _normalize_number(value)
        if _should_ignore_number(value, answer_without_citations, match.start()):
            continue
        if normalized not in context_numbers and normalized not in seen:
            unverified.append(value)
            seen.add(normalized)
    return unverified


def _context_numbers(chunks: list[RetrievedChunk]) -> set[str]:
    """Thu thập mọi cụm số từ breadcrumb, content và raw table của context."""
    numbers: set[str] = set()
    for chunk in chunks:
        values = [chunk.breadcrumb, chunk.content]
        if chunk.raw_table:
            values.append(chunk.raw_table)
        for value in values:
            numbers.update(
                _normalize_number(match.group(0))
                for match in _NUMBER_PATTERN.finditer(value)
            )
    return numbers


def _should_ignore_number(value: str, text: str, position: int) -> bool:
    """Bỏ qua list marker một chữ số và năm không đi kèm từ ``năm``."""
    if len(value) == 1:
        return True
    if _YEAR_PATTERN.fullmatch(value):
        return _YEAR_PREFIX_PATTERN.search(text[:position]) is None
    return False


def _normalize_number(value: str) -> str:
    """Bỏ dấu phân cách nghìn và khoảng trắng để so khớp số ổn định."""
    return _NUMBER_SEPARATORS_PATTERN.sub("", value)
