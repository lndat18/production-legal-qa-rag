"""Unit test cho `evaluation/unit_splitter.py` (evaluation_spec.md mục 4.4) — thuần Python, không cần ragas."""

from __future__ import annotations

from production_legal_qa_rag.evaluation.unit_splitter import (
    MAX_UNIT_CHARS,
    MIN_UNIT_CHARS,
    split_document,
)


def _chapter(name: str, chars: int) -> str:
    return f"## Chương {name}. TIÊU ĐỀ\n\n#### Điều 1. X\n\n" + ("a" * chars) + "\n\n"


def test_document_without_chapters_is_single_unit() -> None:
    units = split_document("a.md", "# TÊN\n\n#### Điều 1. X\n\nnội dung\n")
    assert [u.title for u in units] == ["Toàn văn"]
    assert units[0].index == 1


def test_preamble_dropped_and_small_chapters_merged() -> None:
    content = "# TÊN\n\nCăn cứ...\n\n" + _chapter("I", 1000) + _chapter("II", 1000)
    units = split_document("a.md", content)
    assert len(units) == 1
    assert units[0].title == "Chương I + Chương II"
    assert "Căn cứ" not in units[0].text


def test_large_chapter_split_by_section() -> None:
    sections = "".join(
        f"### Mục {i}. M\n\n#### Điều {i}. X\n\n" + "a" * 20_000 + "\n\n"
        for i in (1, 2)
    )
    content = "## Chương I. TIÊU ĐỀ\n\n" + sections
    units = split_document("a.md", content)
    assert [u.title.split(" — ")[-1][:6] for u in units] == ["Mục 1.", "Mục 2."]
    assert all(MIN_UNIT_CHARS <= u.char_count <= MAX_UNIT_CHARS for u in units)


def test_oversized_article_falls_back_to_paragraphs() -> None:
    body = "".join("b" * 5_000 + "\n\n" for _ in range(14))
    content = "## Chương I. TIÊU ĐỀ\n\n#### Điều 1. X\n\n" + body
    units = split_document("a.md", content)
    assert len(units) == 3
    assert all(u.char_count <= MAX_UNIT_CHARS for u in units)


def test_trailing_footnotes_stripped() -> None:
    content = _chapter("I", 8000) + "\n---\n\n[1] Luật số 1 sửa đổi...\n"
    units = split_document("a.md", content)
    assert "Luật số 1" not in units[0].text


def test_unit_index_is_sequential() -> None:
    content = _chapter("I", 8000) + _chapter("II", 8000) + _chapter("III", 8000)
    assert [u.index for u in split_document("a.md", content)] == [1, 2, 3]
