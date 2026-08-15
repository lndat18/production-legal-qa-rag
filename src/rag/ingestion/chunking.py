"""Chia Markdown văn bản pháp luật thành các bản ghi JSONL đã được kiểm tra.

Chunker luôn ưu tiên đơn vị pháp lý (Điều, Khoản, Điểm) trước khi xét giới hạn
token. Caller có thể truyền tokenizer; bộ đếm tích hợp chỉ là phương án dự
phòng không phụ thuộc thư viện cho môi trường phát triển cục bộ.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence


DEFAULT_TARGET_TOKENS = 400
DEFAULT_HARD_MAX_TOKENS = 700

# Cấu hình chạy trực tiếp. Thay đổi các giá trị này khi cần dùng thư mục hoặc
# ngưỡng cắt khác; không cần truyền tham số trên dòng lệnh.
INPUT_PATH = Path("data/markdown")
OUTPUT_DIRECTORY = Path("data/chunks")
TARGET_TOKENS = DEFAULT_TARGET_TOKENS
HARD_MAX_TOKENS = DEFAULT_HARD_MAX_TOKENS
EXCLUDED_INPUT_FILENAMES = frozenset({"stategies.md"})

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_ARTICLE_RE = re.compile(r"^Điều\s+\d+[a-z]?\b", re.IGNORECASE)
_CHAPTER_RE = re.compile(r"^Chương\s+(?:[IVXLCDM]+|\d+)\b", re.IGNORECASE)
_SECTION_RE = re.compile(r"^Mục\s+(?:[IVXLCDM]+|\d+)\b", re.IGNORECASE)
_PART_RE = re.compile(r"^Phần\s+(?:thứ\s+)?(?:[IVXLCDM]+|\d+)\b", re.IGNORECASE)
_APPENDIX_RE = re.compile(r"^Phụ\s+lục\b", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"(?m)^\s*(\d+)\.\s+")
_POINT_RE = re.compile(r"(?mi)^\s*([a-zđ])\)\s+")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class SupportsEncode(Protocol):
    """Giao diện tối thiểu của đa số tokenizer cho mô hình embedding."""

    def encode(self, text: str, **kwargs: Any) -> Sequence[Any]: ...


TokenCounter = Callable[[str], int]


def count_tokens(text: str, tokenizer: TokenCounter | SupportsEncode | None = None) -> int:
    """Đếm token bằng tokenizer embedding đã cấu hình.

    Có thể truyền callable trả về số lượng token hoặc tokenizer có ``encode``.
    Nếu không truyền, dùng phép ước lượng có hỗ trợ Unicode. Khi chạy thật nên
    truyền tokenizer của mô hình embedding để cắt và metadata dùng cùng số token
    thực tế.
    """
    if tokenizer is None:
        return len(_TOKEN_RE.findall(text))
    if callable(tokenizer):
        return int(tokenizer(text))
    return len(tokenizer.encode(text))


@dataclass(frozen=True)
class _Block:
    kind: str  # heading, văn bản, bảng
    value: str
    level: int = 0


@dataclass(frozen=True)
class _Context:
    document_title: str
    part: str | None = None
    chapter: str | None = None
    section: str | None = None
    article: str | None = None
    clause: str | None = None
    point: str | None = None
    appendix: str | None = None

    def breadcrumb(self, *, part_index: int = 1, part_count: int = 1) -> list[str]:
        lines = [self.document_title]
        lines.extend(
            value
            for value in (self.part, self.chapter, self.section, self.appendix, self.article, self.clause)
            if value
        )
        if self.point:
            point = self.point
            if part_count > 1:
                point = f"{point} — phần {part_index}/{part_count}"
            lines.append(point)
        elif part_count > 1:
            lines.append(f"Phần {part_index}/{part_count}")
        return lines

    def id_prefix(self) -> str:
        # Định danh đi theo hệ thống trích dẫn. Phần/Mục chỉ được thêm khi cần
        # để không mất ngữ cảnh ở văn bản đánh lại số Điều trong phụ lục.
        pieces = [self.document_title]
        if self.appendix:
            pieces.append(_citation_label(self.appendix, "Phụ lục"))
        if self.article:
            pieces.append(_citation_label(self.article, "Điều"))
        if self.clause:
            pieces.append(_citation_label(self.clause, "Khoản"))
        if self.point:
            pieces.append(_citation_label(self.point, "Điểm"))
        return ":".join(pieces)


def _citation_label(value: str, fallback: str) -> str:
    """Chỉ giữ nhãn trích dẫn trong ``chunk_id`` nhưng giữ tiêu đề trong content."""
    match = re.match(rf"^({re.escape(fallback)}\s+[^.\s:]+)", value, re.IGNORECASE)
    return match.group(1) if match else value.split(".", 1)[0].strip()


def parse_markdown_blocks(markdown: str) -> list[_Block]:
    """Phân tích heading, đoạn văn và bảng Markdown/HTML mà không cần thư viện."""
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[_Block] = []
    index = 0
    text_lines: list[str] = []

    def flush_text() -> None:
        nonlocal text_lines
        value = "\n".join(text_lines).strip()
        if value:
            blocks.append(_Block("text", value))
        text_lines = []

    while index < len(lines):
        line = lines[index]
        heading = _HEADING_RE.match(line)
        if heading:
            flush_text()
            blocks.append(_Block("heading", heading.group(2), len(heading.group(1))))
            index += 1
            continue

        if line.lstrip().lower().startswith("<table"):
            flush_text()
            table_lines = [line]
            index += 1
            while index < len(lines):
                table_lines.append(lines[index])
                if "</table>" in lines[index].lower():
                    index += 1
                    break
                index += 1
            blocks.append(_Block("table", "\n".join(table_lines).strip()))
            continue

        if (
            line.lstrip().startswith("|")
            and index + 1 < len(lines)
            and _TABLE_SEPARATOR_RE.match(lines[index + 1])
        ):
            flush_text()
            table_lines = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            blocks.append(_Block("table", "\n".join(table_lines).strip()))
            continue

        text_lines.append(line)
        index += 1
    flush_text()
    return blocks


def _normalise_body(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _render(context: _Context, body: str, *, part_index: int = 1, part_count: int = 1) -> str:
    breadcrumb = "\n".join(context.breadcrumb(part_index=part_index, part_count=part_count))
    body = _normalise_body(body)
    return f"{breadcrumb}\n\n{body}" if body else breadcrumb


def _semantic_pieces(text: str) -> list[str]:
    """Trả về các ranh giới an toàn: đoạn văn, câu rồi đến vế có dấu chấm phẩy."""
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ỵ0-9])", paragraph) if item.strip()]
        for sentence in sentences:
            semicolons = [item.strip() for item in re.split(r"(?<=;)\s+", sentence) if item.strip()]
            pieces.extend(semicolons)
    return pieces or [text.strip()]


def _hard_split_text(text: str, fits: Callable[[str], bool]) -> list[str]:
    """Cắt theo khoảng trắng/ký tự ở bước cuối khi không còn ranh giới ngữ nghĩa."""
    atoms = re.findall(r"\S+\s*", text)
    if not atoms:
        return []
    result: list[str] = []
    current = ""
    for atom in atoms:
        candidate = current + atom
        if current and not fits(candidate.strip()):
            result.append(current.strip())
            current = atom
        else:
            current = candidate
        if not fits(current.strip()):
            # Một từ đơn lẻ có thể vượt giới hạn token của mô hình. Cắt theo ký
            # tự để validator vẫn phản ánh đúng giới hạn trong trường hợp này.
            value = current.strip()
            current = ""
            start = 0
            while start < len(value):
                end = start + 1
                while end <= len(value) and fits(value[start:end]):
                    end += 1
                if end == start + 1:
                    raise ValueError("Breadcrumb/context alone exceeds hard_max_tokens")
                result.append(value[start : end - 1])
                start = end - 1
    if current.strip():
        result.append(current.strip())
    return result


def _split_for_context(
    text: str,
    context: _Context,
    counter: TokenCounter,
    hard_max_tokens: int,
    target_tokens: int,
    *,
    lead: str | None = None,
) -> list[str]:
    """Cắt một đơn vị pháp lý quá dài và lặp lại câu dẫn trong từng phần."""
    prefix = f"Câu dẫn: {lead}\n\n" if lead else ""

    # ``phần 12/12`` chỉ được thêm sau khi biết số mảnh. Chừa một khoảng dự phòng
    # nhỏ, có chủ đích, cho breadcrumb lặp lại để content sau cùng không vượt
    # giới hạn cứng. Điều này cần thiết khi chunk ban đầu sát giới hạn.
    effective_hard_max = max(1, hard_max_tokens - 20)

    def fits(body: str) -> bool:
        return counter(_render(context, prefix + body)) <= effective_hard_max

    if fits(text):
        return [text]
    if not fits(""):
        raise ValueError("Breadcrumb/context alone exceeds hard_max_tokens")

    result: list[str] = []
    current = ""
    for piece in _semantic_pieces(text):
        candidate = f"{current}\n\n{piece}".strip() if current else piece
        if current and (not fits(candidate) or counter(_render(context, prefix + current)) >= min(target_tokens, effective_hard_max)):
            result.append(current)
            current = piece
        else:
            current = candidate
        if not fits(current):
            # Bản thân candidate không còn ranh giới ngữ nghĩa an toàn. Thay nó
            # bằng các mảnh bảo đảm giới hạn token.
            fragments = _hard_split_text(current, fits)
            result.extend(fragments[:-1])
            current = fragments[-1] if fragments else ""
    if current:
        result.append(current)
    return result


def _split_labeled(text: str, pattern: re.Pattern[str]) -> tuple[str, list[tuple[str, str]]]:
    # Điều sửa đổi thường trích nguyên văn luật khác. Các nhãn ``1.``/``a)``
    # trong phần trích dẫn thuộc luật được dẫn chiếu, không thuộc Điều hiện tại.
    # Che các dòng trích dẫn khi tìm nhãn, nhưng giữ offset của chuỗi gốc để nội
    # dung trích dẫn vẫn nguyên vẹn trong content đầu ra.
    protected_lines: list[str] = []
    in_quote = False
    for line in text.splitlines(keepends=True):
        if in_quote or "“" in line:
            protected_lines.append("".join("\n" if char == "\n" else "\x00" for char in line))
        else:
            protected_lines.append(line)
        if "“" in line:
            in_quote = True
        if in_quote and "”" in line:
            in_quote = False
    protected = "".join(protected_lines)
    matches = list(pattern.finditer(protected))
    if not matches:
        return _normalise_body(text), []
    intro = _normalise_body(text[: matches[0].start()])
    items = [
        (
            match.group(1),
            _normalise_body(text[match.end() : matches[index + 1].start()] if index + 1 < len(matches) else text[match.end() :]),
        )
        for index, match in enumerate(matches)
    ]
    return intro, items


def _with_label(kind: str, label: str, body: str) -> str:
    """Tạo breadcrumb trích dẫn mà không sao chép toàn bộ đơn vị pháp lý.

    Khoản thường không có tiêu đề riêng. Đoạn đầu có thể dài hàng trăm token;
    nếu đưa nó vào breadcrumb thì Điểm con sẽ không thể cắt dưới giới hạn cứng.
    Điều kiện áp dụng được đưa vào ``Câu dẫn`` thay vì breadcrumb.
    """
    del body
    return f"{kind} {label}"


class MarkdownLegalChunker:
    """Chunker nhận biết cấu trúc theo ``data/chunks/strategies.md``."""

    def __init__(
        self,
        *,
        tokenizer: TokenCounter | SupportsEncode | None = None,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        hard_max_tokens: int = DEFAULT_HARD_MAX_TOKENS,
    ) -> None:
        if target_tokens <= 0 or hard_max_tokens <= 0 or target_tokens > hard_max_tokens:
            raise ValueError("Require 0 < target_tokens <= hard_max_tokens")
        self.target_tokens = target_tokens
        self.hard_max_tokens = hard_max_tokens
        self._tokenizer = tokenizer
        self._counter: TokenCounter = lambda text: count_tokens(text, tokenizer)

    def chunk_markdown(self, markdown: str, *, document_title: str | None = None) -> list[dict[str, Any]]:
        blocks = parse_markdown_blocks(markdown)
        title = document_title or self._document_title(blocks)
        chunks: list[dict[str, Any]] = []
        context = _Context(document_title=title)
        article_blocks: list[_Block] = []
        appendix_blocks: list[_Block] = []
        mode: str | None = None

        def flush() -> None:
            nonlocal article_blocks, appendix_blocks
            if mode == "article" and context.article:
                chunks.extend(self._chunk_article(context, article_blocks))
            elif mode == "appendix" and context.appendix:
                chunks.extend(self._chunk_appendix(context, appendix_blocks))
            article_blocks = []
            appendix_blocks = []

        for block in blocks:
            if block.kind == "heading":
                heading = block.value
                if _ARTICLE_RE.match(heading):
                    flush()
                    context = replace(context, article=heading, clause=None, point=None, appendix=None)
                    mode = "article"
                elif _APPENDIX_RE.match(heading):
                    flush()
                    context = replace(context, appendix=heading, article=None, clause=None, point=None)
                    mode = "appendix"
                elif _PART_RE.match(heading):
                    flush()
                    context = replace(context, part=heading, chapter=None, section=None, article=None, clause=None, point=None, appendix=None)
                    mode = None
                elif _CHAPTER_RE.match(heading):
                    flush()
                    context = replace(context, chapter=heading, section=None, article=None, clause=None, point=None, appendix=None)
                    mode = None
                elif _SECTION_RE.match(heading):
                    flush()
                    context = replace(context, section=heading, article=None, clause=None, point=None, appendix=None)
                    mode = None
                continue
            if mode == "article":
                article_blocks.append(block)
            elif mode == "appendix":
                appendix_blocks.append(block)
        flush()
        validate_or_raise(chunks, tokenizer=self._tokenizer, hard_max_tokens=self.hard_max_tokens)
        return chunks

    @staticmethod
    def _document_title(blocks: Iterable[_Block]) -> str:
        for block in blocks:
            if block.kind == "heading" and block.level == 1:
                return block.value
        raise ValueError("Markdown phải có heading cấp 1 hoặc document_title")

    def _record(
        self,
        context: _Context,
        body: str,
        *,
        content_type: str = "text",
        part_index: int = 1,
        part_count: int = 1,
        suffix: str | None = None,
    ) -> dict[str, Any]:
        content = _render(context, body, part_index=part_index, part_count=part_count)
        chunk_id = context.id_prefix()
        if suffix:
            chunk_id = f"{chunk_id}:{suffix}"
        elif part_count > 1:
            chunk_id = f"{chunk_id}:Phần {part_index}"
        metadata = {
            "document_title": context.document_title,
            "chapter": context.chapter,
            "article": context.article,
            "clause": context.clause,
            "point": context.point,
            "part_index": part_index,
            "part_count": part_count,
            "token_count": self._counter(content),
        }
        return {"chunk_id": chunk_id, "content": content, "content_type": content_type, "metadata": metadata}

    def _text_records(self, context: _Context, body: str, *, lead: str | None = None) -> list[dict[str, Any]]:
        prefix = f"Câu dẫn: {lead}\n\n" if lead else ""
        parts = _split_for_context(
            body,
            context,
            self._counter,
            self.hard_max_tokens,
            self.target_tokens,
            lead=lead,
        )
        count = len(parts)
        return [self._record(context, prefix + part, part_index=index, part_count=count) for index, part in enumerate(parts, 1)]

    def _chunk_article(self, context: _Context, blocks: list[_Block]) -> list[dict[str, Any]]:
        # Văn bản pháp luật thường đặt chú thích hoặc nội dung sửa đổi được trích
        # dẫn sau bảng nơi nhận/chữ ký. Chúng nằm ngoài Điều cuối; đặc biệt các
        # ``Điều``/``Khoản`` được trích dẫn không được thành con của Điều này.
        legal_blocks: list[_Block] = []
        for block in blocks:
            if block.kind == "table" and _is_signature_table(block.value):
                break
            legal_blocks.append(block)

        text = _normalise_body("\n\n".join(block.value for block in legal_blocks if block.kind == "text"))
        # Loader DOCX tạo bảng HTML một hàng cho phần chữ ký và nơi nhận. Bảng
        # này không có header và không phải nội dung quy phạm.
        tables: list[tuple[str, _Context, str | None]] = []
        preceding_text: list[str] = []
        for block in legal_blocks:
            if block.kind == "text":
                preceding_text.append(block.value)
            elif block.kind == "table" and _has_valid_markdown_table(block.value):
                table_context, lead = self._table_context(
                    context,
                    _normalise_body("\n\n".join(preceding_text)),
                )
                tables.append((block.value, table_context, lead))
        result: list[dict[str, Any]] = []
        if text:
            whole = _render(context, text)
            if self._counter(whole) <= self.hard_max_tokens:
                result.append(self._record(context, text))
            else:
                result.extend(self._chunk_long_article(context, text))
        for number, (table, table_context, lead) in enumerate(tables, 1):
            body = f"Câu dẫn: {lead}\n\n{table}" if lead else table
            result.append(self._record(table_context, body, content_type="table", suffix=f"Bảng {number}"))
        return result

    def _table_context(self, context: _Context, preceding_text: str) -> tuple[_Context, str | None]:
        clauses_intro, clauses = _split_labeled(preceding_text, _CLAUSE_RE)
        if not clauses:
            return context, clauses_intro or None
        clause_number, clause_body = clauses[-1]
        clause_context = replace(context, clause=_with_label("Khoản", clause_number, clause_body), point=None)
        points_intro, points = _split_labeled(clause_body, _POINT_RE)
        if points:
            label, point_body = points[-1]
            return replace(clause_context, point=f"Điểm {label}"), points_intro or point_body.split("\n", 1)[0]
        return clause_context, clause_body.split("\n", 1)[0]

    def _chunk_long_article(self, context: _Context, text: str) -> list[dict[str, Any]]:
        article_intro, clauses = _split_labeled(text, _CLAUSE_RE)
        if not clauses:
            return self._text_records(context, text)
        result: list[dict[str, Any]] = []
        # Lặp điều kiện/dẫn chiếu chung của Điều trong từng Khoản.
        for number, clause_body in clauses:
            clause_context = replace(context, clause=_with_label("Khoản", number, clause_body), point=None)
            full_clause = _render(clause_context, clause_body)
            lead = article_intro or None
            if self._counter(_render(clause_context, (f"Câu dẫn: {lead}\n\n" if lead else "") + clause_body)) <= self.hard_max_tokens:
                result.append(self._record(clause_context, (f"Câu dẫn: {lead}\n\n" if lead else "") + clause_body))
                continue
            points_intro, points = _split_labeled(clause_body, _POINT_RE)
            if not points:
                result.extend(self._text_records(clause_context, clause_body, lead=lead))
                continue
            point_lead = points_intro or lead
            for label, point_body in points:
                point_context = replace(clause_context, point=f"Điểm {label}")
                result.extend(self._text_records(point_context, point_body, lead=point_lead))
        return result

    def _chunk_appendix(self, context: _Context, blocks: list[_Block]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        text = _normalise_body("\n\n".join(block.value for block in blocks if block.kind == "text"))
        if text:
            result.extend(self._text_records(context, text))
        for number, table in enumerate((block.value for block in blocks if block.kind == "table" and _has_valid_markdown_table(block.value)), 1):
            result.append(self._record(context, table, content_type="table", suffix=f"Bảng {number}"))
        return result


def chunk_markdown(
    markdown: str,
    *,
    document_title: str | None = None,
    tokenizer: TokenCounter | SupportsEncode | None = None,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    hard_max_tokens: int = DEFAULT_HARD_MAX_TOKENS,
) -> list[dict[str, Any]]:
    """Hàm bao tiện dụng cho :class:`MarkdownLegalChunker`."""
    return MarkdownLegalChunker(
        tokenizer=tokenizer,
        target_tokens=target_tokens,
        hard_max_tokens=hard_max_tokens,
    ).chunk_markdown(markdown, document_title=document_title)


def chunk_markdown_file(
    input_path: str | Path,
    *,
    tokenizer: TokenCounter | SupportsEncode | None = None,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    hard_max_tokens: int = DEFAULT_HARD_MAX_TOKENS,
) -> list[dict[str, Any]]:
    path = Path(input_path)
    return chunk_markdown(
        path.read_text(encoding="utf-8"),
        document_title=None,
        tokenizer=tokenizer,
        target_tokens=target_tokens,
        hard_max_tokens=hard_max_tokens,
    )


def write_jsonl(
    chunks: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    tokenizer: TokenCounter | SupportsEncode | None = None,
    hard_max_tokens: int = DEFAULT_HARD_MAX_TOKENS,
) -> Path:
    """Kiểm tra và ghi mỗi đối tượng JSON UTF-8 trên một dòng."""
    items = list(chunks)
    validate_or_raise(items, tokenizer=tokenizer, hard_max_tokens=hard_max_tokens)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for chunk in items:
            output.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
    return path


def _has_valid_markdown_table(content: str) -> bool:
    lines = content.splitlines()
    for index in range(len(lines) - 1):
        if lines[index].lstrip().startswith("|") and _TABLE_SEPARATOR_RE.match(lines[index + 1]):
            header_columns = len(lines[index].strip().strip("|").split("|"))
            rows = [line for line in lines[index + 2 :] if line.lstrip().startswith("|")]
            return bool(rows) and all(len(row.strip().strip("|").split("|")) == header_columns for row in rows)
    return False


def _is_signature_table(content: str) -> bool:
    """Nhận diện bảng một hàng chứa nơi nhận/chữ ký do loader DOCX tạo ra."""
    folded = content.casefold()
    return content.lstrip().lower().startswith("<table") and any(
        marker in folded
        for marker in ("nơi nhận", "ký thay", "thủ tướng", "chủ tịch", "xác thực văn bản")
    )


def validate_chunks(
    chunks: Iterable[dict[str, Any]],
    *,
    tokenizer: TokenCounter | SupportsEncode | None = None,
    hard_max_tokens: int = DEFAULT_HARD_MAX_TOKENS,
) -> list[str]:
    """Trả về toàn bộ lỗi schema, token, thứ tự phần và bảng."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    part_groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    required_metadata = {"document_title", "chapter", "article", "clause", "point", "part_index", "part_count", "token_count"}
    for position, chunk in enumerate(chunks, 1):
        label = f"chunk {position}"
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            errors.append(f"{label}: thiếu chunk_id")
            continue
        if chunk_id in seen_ids:
            errors.append(f"{label}: chunk_id trùng lặp ({chunk_id})")
        seen_ids.add(chunk_id)
        content = chunk.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"{label}: content rỗng")
            continue
        content_type = chunk.get("content_type")
        if content_type not in {"text", "table"}:
            errors.append(f"{label}: content_type không hợp lệ")
        metadata = chunk.get("metadata")
        if not isinstance(metadata, dict) or not required_metadata.issubset(metadata):
            errors.append(f"{label}: metadata thiếu trường bắt buộc")
            continue
        actual = count_tokens(content, tokenizer)
        if metadata["token_count"] != actual:
            errors.append(f"{label}: token_count không khớp content")
        part_index, part_count = metadata["part_index"], metadata["part_count"]
        if not isinstance(part_index, int) or not isinstance(part_count, int) or not (1 <= part_index <= part_count):
            errors.append(f"{label}: part_index/part_count không hợp lệ")
        else:
            part_groups[re.sub(r":Phần \d+$", "", chunk_id)].append((part_index, part_count))
        if content_type == "text" and actual > hard_max_tokens:
            errors.append(f"{label}: text vượt hard_max_tokens ({actual}>{hard_max_tokens})")
        if content_type == "table" and not _has_valid_markdown_table(content):
            errors.append(f"{label}: bảng Markdown thiếu header hoặc số cột không nhất quán")
        title = metadata.get("document_title")
        if isinstance(title, str) and title and not content.startswith(title):
            errors.append(f"{label}: content không bắt đầu bằng breadcrumb tên văn bản")
    for base_id, parts in part_groups.items():
        counts = {count for _, count in parts}
        if len(counts) != 1:
            errors.append(f"{base_id}: part_count không nhất quán")
            continue
        expected = list(range(1, next(iter(counts)) + 1))
        if sorted(index for index, _ in parts) != expected:
            errors.append(f"{base_id}: part_index không liên tục")
    return errors


def validate_or_raise(
    chunks: Iterable[dict[str, Any]],
    *,
    tokenizer: TokenCounter | SupportsEncode | None = None,
    hard_max_tokens: int = DEFAULT_HARD_MAX_TOKENS,
) -> None:
    errors = validate_chunks(chunks, tokenizer=tokenizer, hard_max_tokens=hard_max_tokens)
    if errors:
        raise ValueError("Chunk validation failed:\n- " + "\n- ".join(errors))


def main() -> None:
    if INPUT_PATH.is_file():
        paths = [INPUT_PATH]
    else:
        paths = sorted(
            path
            for path in INPUT_PATH.glob("*.md")
            if path.name not in EXCLUDED_INPUT_FILENAMES
        )
    if not paths:
        raise SystemExit("Không tìm thấy file Markdown đầu vào.")
    for path in paths:
        chunks = chunk_markdown_file(
            path,
            target_tokens=TARGET_TOKENS,
            hard_max_tokens=HARD_MAX_TOKENS,
        )
        destination = OUTPUT_DIRECTORY / f"{path.stem}.jsonl"
        write_jsonl(chunks, destination, hard_max_tokens=HARD_MAX_TOKENS)
        print(f"{path} -> {destination} ({len(chunks)} chunks)")


if __name__ == "__main__":
    main()
