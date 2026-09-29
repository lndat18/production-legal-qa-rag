"""Chia văn bản pháp luật thành đơn vị sinh testset (evaluation_spec.md mục 4.4).

Mỗi đơn vị (Chương, hoặc Mục/phần của Chương quá lớn) là một lần dựng
`KnowledgeGraph` + sinh câu hỏi riêng, để chạy nhiều ngày và checkpoint được.
Module thuần Python (không import `ragas`), chạy được trên venv production.

Quy tắc: đơn vị chuẩn = Chương (`## `); Chương lớn hơn `MAX_UNIT_CHARS` tách theo
Mục (`### `); phần vẫn quá lớn tách tiếp theo Điều (`#### `) rồi theo đoạn văn;
phần nhỏ hơn `MIN_UNIT_CHARS` gộp với phần liền kề khi tổng không vượt trần.
"""

from __future__ import annotations

import itertools
import math
import re
from pathlib import Path
from typing import Final

from pydantic import BaseModel

MAX_UNIT_CHARS: Final = 30_000
MIN_UNIT_CHARS: Final = 6_000
TOKENS_PER_CHAR: Final = 5.5
FIXED_TOKENS_PER_UNIT: Final = 4_000

_CHAPTER_RE: Final = re.compile(r"^## ", re.MULTILINE)
_SECTION_RE: Final = re.compile(r"^### ", re.MULTILINE)
_ARTICLE_RE: Final = re.compile(r"^#### ", re.MULTILINE)
_PARAGRAPH_RE: Final = re.compile(r"\n\s*\n")
# Khối chú thích sửa đổi ở cuối file: dòng `---` rồi `[1] Luật số ...` (không phải điều luật).
_FOOTNOTES_RE: Final = re.compile(r"\n---[ \t]*\n\s*\[1\] ")


class EvalUnit(BaseModel):
    """Một đơn vị sinh testset: lát văn bản liên tục của một văn bản nguồn."""

    source_document: str
    index: int  # thứ tự 1-based trong văn bản; khoá đơn vị (ĐKLĐ có 2 "Chương XI")
    title: str
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def estimated_tokens(self) -> int:
        return round(self.char_count * TOKENS_PER_CHAR) + FIXED_TOKENS_PER_UNIT


class _Piece(BaseModel):
    title: str
    text: str
    short: str = ""  # nhãn ngắn ("Chương III", "Chương V Mục 1") dùng khi gộp tiêu đề

    def label(self) -> str:
        return self.short or self.title


def _split_at(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Cắt `text` trước mỗi dòng khớp `pattern`; phần đầu (nếu có) giữ riêng."""
    starts = [m.start() for m in pattern.finditer(text)]
    if not starts:
        return [text]
    bounds = ([0] if starts[0] != 0 else []) + starts + [len(text)]
    return [text[a:b] for a, b in itertools.pairwise(bounds)]


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0].lstrip("# ").strip()


def _short(heading: str) -> str:
    """ "Chương V. BẢO HIỂM..." -> "Chương V"; "Mục 1. CHẾ ĐỘ ỐM ĐAU" -> "Mục 1"."""
    return heading.split(". ", 1)[0]


def _pack(blocks: list[str], total: int) -> list[str]:
    """Gói các khối liên tiếp thành phần cân bằng, mỗi phần ≤ MAX_UNIT_CHARS."""
    parts = math.ceil(total / MAX_UNIT_CHARS)
    target = total / parts
    packed: list[str] = []
    current = ""
    for block in blocks:
        if current and len(current) + len(block) > MAX_UNIT_CHARS:
            packed.append(current)
            current = ""
        current += block
        if len(current) >= target and len(packed) < parts - 1:
            packed.append(current)
            current = ""
    if current:
        packed.append(current)
    return packed


def _split_oversized(piece: _Piece) -> list[_Piece]:
    """Fallback cho phần > MAX_UNIT_CHARS không có Mục: tách theo Điều, rồi đoạn văn."""
    blocks: list[str] = []
    for article in _split_at(piece.text, _ARTICLE_RE):
        if len(article) <= MAX_UNIT_CHARS:
            blocks.append(article)
            continue
        paragraphs = _PARAGRAPH_RE.split(article)
        blocks.extend(paragraph + "\n\n" for paragraph in paragraphs)
    packed = _pack(blocks, sum(len(block) for block in blocks))
    return [
        _Piece(
            title=f"{piece.title} (phần {i}/{len(packed)})",
            short=f"{piece.label()} (phần {i}/{len(packed)})",
            text=text.rstrip() + "\n",
        )
        for i, text in enumerate(packed, start=1)
    ]


def _chapter_pieces(chapter: str) -> list[_Piece]:
    """Một Chương → 1 phần, hoặc nhiều phần theo Mục/Điều/đoạn nếu quá lớn."""
    title = _first_line(chapter)
    if len(chapter) <= MAX_UNIT_CHARS:
        return [_Piece(title=title, short=_short(title), text=chapter)]

    sections = _split_at(chapter, _SECTION_RE)
    if len(sections) == 1:
        return _split_oversized(_Piece(title=title, text=chapter))

    # Phần dẫn của Chương (dòng tiêu đề `## ` trước Mục 1) dính vào Mục 1.
    if not sections[0].startswith("### "):
        sections = [sections[0] + sections[1], *sections[2:]]

    pieces: list[_Piece] = []
    for section in sections:
        heading = _SECTION_RE.search(section)
        section_heading = _first_line(section[heading.start() if heading else 0 :])
        sub = _Piece(
            title=f"{title} — {section_heading}",
            short=f"{_short(title)} {_short(section_heading)}",
            text=section,
        )
        pieces.extend(_split_oversized(sub) if len(section) > MAX_UNIT_CHARS else [sub])
    return pieces


def _join(first: _Piece, second: _Piece) -> _Piece:
    label = f"{first.label()} + {second.label()}"
    return _Piece(title=label, short=label, text=first.text + second.text)


def _merge_small(pieces: list[_Piece]) -> list[_Piece]:
    """Gộp phần < MIN_UNIT_CHARS vào phần kế (hoặc phần trước nếu ở cuối)."""
    merged: list[_Piece] = []
    for piece in pieces:
        if (
            merged
            and len(merged[-1].text) < MIN_UNIT_CHARS
            and len(merged[-1].text) + len(piece.text) <= MAX_UNIT_CHARS
        ):
            last = merged.pop()
            piece = _join(last, piece)
        merged.append(piece)
    if (
        len(merged) >= 2
        and len(merged[-1].text) < MIN_UNIT_CHARS
        and len(merged[-2].text) + len(merged[-1].text) <= MAX_UNIT_CHARS
    ):
        last = merged.pop()
        prev = merged.pop()
        merged.append(_join(prev, last))
    return merged


def _strip_footnotes(content: str) -> str:
    """Bỏ khối chú thích `[1] ... [n]` cuối văn bản (chỉ dẫn văn bản sửa đổi, không phải luật)."""
    matches = list(_FOOTNOTES_RE.finditer(content))
    return content[: matches[-1].start()] + "\n" if matches else content


def split_document(source_document: str, content: str) -> list[EvalUnit]:
    """Chia một văn bản thành các `EvalUnit` theo thứ tự xuất hiện.

    Văn bản không có tiêu đề `## ` (vd. nghị định ngắn) là một đơn vị duy nhất.
    Phần mở đầu trước Chương đầu tiên (quốc hiệu, căn cứ...) và khối chú thích
    cuối văn bản (`[1] Luật số ... sửa đổi...`) bị bỏ vì không phải nội dung
    điều luật để sinh câu hỏi.
    """
    content = _strip_footnotes(content)
    chapters = [c for c in _split_at(content, _CHAPTER_RE) if c.startswith("## ")]
    if not chapters:
        pieces = [_Piece(title="Toàn văn", text=content)]
        if len(content) > MAX_UNIT_CHARS:
            pieces = _split_oversized(pieces[0])
    else:
        pieces = [p for chapter in chapters for p in _chapter_pieces(chapter)]
    pieces = _merge_small(pieces)
    return [
        EvalUnit(source_document=source_document, index=i, title=p.title, text=p.text)
        for i, p in enumerate(pieces, start=1)
    ]


def split_directory(markdown_dir: Path) -> list[EvalUnit]:
    """Chia mọi `.md` trong `markdown_dir` (theo thứ tự tên file)."""
    paths = sorted(markdown_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"Không tìm thấy file .md nào trong {markdown_dir}")
    units: list[EvalUnit] = []
    for path in paths:
        units.extend(split_document(path.name, path.read_text(encoding="utf-8")))
    return units
