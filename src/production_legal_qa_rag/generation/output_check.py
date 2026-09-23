"""Deterministic hard gate cho citation, output bị cắt và số pháp lý nhạy cảm."""

from __future__ import annotations

import re

from production_legal_qa_rag.generation.models import (
    Citation,
    HardGateResult,
    OutputWarning,
    VerificationIssue,
)
from production_legal_qa_rag.retrieval.models import RetrievedChunk

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_NUMBER_PATTERN = re.compile(
    r"(?<!\w)(?:\d{1,3}(?:[.,\s]\d{3})+|\d+(?:[.,]\d+)?)(?!\w)"
)
_NUMBER_SEPARATORS_PATTERN = re.compile(r"[.,\s]")
_SENSITIVE_UNIT_PATTERN = re.compile(
    r"\s*(?:%|phần\s+trăm\b|(?:triệu|nghìn|ngàn|tỷ)\b(?:\s*(?:đồng\b|vnđ\b|vnd\b))?|đồng\b|vnđ\b|vnd\b|ngày\b|tháng\b|năm\b|giờ\b|tuổi\b)",
    re.IGNORECASE,
)


def check_output(
    text: str,
    chunks: list[RetrievedChunk],
    *,
    finish_reason: str | None = None,
) -> HardGateResult:
    """Kiểm tra điều có thể quyết định bằng code trước khi Judge chạy.

    Args:
        text: Toàn bộ draft đã được buffer.
        chunks: Context cố định thực sự được đưa vào prompt.
        finish_reason: Lý do kết thúc stream từ provider, nếu có.

    Returns:
        Citation hợp lệ, hard issue cần repair và warning mềm.
    """
    citations, invalid_numbers = _extract_citations(text, chunks)
    hard_issues: list[VerificationIssue] = []
    if finish_reason == "length":
        hard_issues.append(
            VerificationIssue(
                code="truncated",
                detail="Câu trả lời bị cắt do đạt giới hạn token của provider.",
            )
        )
    if invalid_numbers:
        hard_issues.append(
            VerificationIssue(
                code="invalid_citation",
                detail=(
                    "Citation ngoài phạm vi context: "
                    + ", ".join(f"[{number}]" for number in invalid_numbers)
                ),
            )
        )

    sensitive, ordinary = _find_unverified_numbers(text, chunks)
    if sensitive:
        hard_issues.append(
            VerificationIssue(
                code="unverified_sensitive_number",
                detail=(
                    "Số pháp lý nhạy cảm không tìm thấy trong context: "
                    + ", ".join(sensitive)
                ),
            )
        )

    warnings: list[OutputWarning] = []
    if ordinary:
        warnings.append(
            OutputWarning(
                code="unverified_number",
                message="Câu trả lời có con số chưa xác minh được bằng code.",
                detail=", ".join(ordinary),
            )
        )
    return HardGateResult(
        citations=citations, hard_issues=hard_issues, warnings=warnings
    )


def _extract_citations(
    text: str, chunks: list[RetrievedChunk]
) -> tuple[list[Citation], list[int]]:
    """Lấy citation hợp lệ duy nhất và danh sách citation ngoài context."""
    citations: list[Citation] = []
    invalid_numbers: list[int] = []
    seen_valid: set[int] = set()
    seen_invalid: set[int] = set()
    for match in _CITATION_PATTERN.finditer(text):
        number = int(match.group(1))
        if 1 <= number <= len(chunks):
            if number in seen_valid:
                continue
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


def _find_unverified_numbers(
    text: str, chunks: list[RetrievedChunk]
) -> tuple[list[str], list[str]]:
    """Tách số không có evidence thành nhóm nhạy cảm và nhóm warning mềm."""
    answer_without_citations = _CITATION_PATTERN.sub("", text)
    context_numbers = _context_numbers(chunks)
    sensitive: list[str] = []
    ordinary: list[str] = []
    seen: set[str] = set()
    for match in _NUMBER_PATTERN.finditer(answer_without_citations):
        value = match.group(0)
        normalized = _normalize_number(value)
        if normalized in context_numbers or normalized in seen:
            continue
        seen.add(normalized)
        if _is_sensitive_number(answer_without_citations, match.end()):
            sensitive.append(value)
        elif not _is_list_marker(value, answer_without_citations, match.start()):
            ordinary.append(value)
    return sensitive, ordinary


def _context_numbers(chunks: list[RetrievedChunk]) -> set[str]:
    """Thu thập số trong breadcrumb, content và bảng gốc của context."""
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


def _is_sensitive_number(text: str, end: int) -> bool:
    """Nhận diện số có đơn vị tạo nghĩa vụ hoặc quyền lợi pháp lý đáng kể."""
    return _SENSITIVE_UNIT_PATTERN.match(text[end:]) is not None


def _is_list_marker(value: str, text: str, position: int) -> bool:
    """Bỏ qua marker list một chữ số vì nó không phải claim định lượng."""
    return len(value) == 1 and text[position + len(value) :].startswith(".")


def _normalize_number(value: str) -> str:
    """Bỏ dấu phân cách để so khớp 4.960.000 với 4 960 000."""
    return _NUMBER_SEPARATORS_PATTERN.sub("", value)
