"""Bước chunking: markdown có cấu trúc → chunk sẵn sàng embed.

Cách dùng:
    python src/rag/ingestion/chunking.py --md-dir data/markdown --out-dir data/chunks

Bước này KHÔNG tính embedding và KHÔNG tạo index HNSW.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import time
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any
from typing import Any, Literal


# ========================================================================
# CONFIG
# ========================================================================
#
# Hằng số cấu hình cho bước chunking.
#
# Mọi ngưỡng token đều tính theo tokenizer thật của model embedding, không bao
# giờ theo heuristic ký tự.
CHUNKER_VERSION = "2.0.1"

EMBED_MODEL = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
MAX_TOKENS = 192
SAFETY_MARGIN = 4
BUDGET = MAX_TOKENS - SAFETY_MARGIN  # 188

# --- ngân sách động (Task 7) ---
BREADCRUMB_RESERVE_MIN = 12  # sàn dành sẵn cho breadcrumb khi cắt
BREADCRUMB_FULL_MIN = 60     # còn >= mức này -> breadcrumb đầy đủ
BREADCRUMB_DIEU_MIN = 30     # còn >= mức này -> chỉ tiêu đề Điều
BREADCRUMB_TRUNC_MIN = 12    # còn >= mức này -> tiêu đề Điều rút gọn
BREADCRUMB_HARD_CAP = 48     # trần tuyệt đối, kể cả khi dư ngân sách

STEM_MAX_TOKENS = 56         # stem vượt mức này -> xem Task 5 bậc C

# Spec Task 7 chỉ tính stem trong ngân sách cố định, nhưng chunk của Khoản còn
# mang thêm câu dẫn của Điều (Task 5b). Hai thứ cộng lại có thể nuốt gần hết
# ngân sách và làm thân không còn chỗ, vi phạm NT-4 (thân là ưu tiên cao nhất).
# Hai hằng số dưới đây chặn tình huống đó: vượt PREFIX_MAX_TOKENS thì bỏ câu
# dẫn trước, bỏ stem sau, cho tới khi thân còn ít nhất MIN_BODY_ALLOWANCE.
PREFIX_MAX_TOKENS = 72
MIN_BODY_ALLOWANCE = 48

# --- cắt chunk dài (Task 8) ---
SPLIT_TARGET_TOKENS = 150
SPLIT_OVERLAP_TOKENS = 24    # CHỈ tầng 4
MIN_TAIL_TOKENS = 32

# --- gộp Điều liệt kê ngắn (Task 4 nhánh 5) ---
# Spec ghi 150, quy từ "450 ký tự" bằng tỉ lệ 3,2 ký tự/token. Tỉ lệ thật của
# PhoBERT trên corpus này là ~5 ký tự/token: nhóm Điều <= 450 ký tự có nhiều
# nhất 90 token, còn ngưỡng 150 gộp tới 199 Điều và kéo tổng số chunk xuống
# 1.674 — ngoài khoảng nghiệm thu 1.900–1.960 của chính spec. Giữ ý định
# thiết kế (gộp ~82 Điều liệt kê ngắn), sửa con số theo tokenizer thật.
COLLAPSE_DIEU_MAX_TOKENS = 90   # toàn bộ Điều <= mức này thì gộp 1 chunk
COLLAPSE_DIEU_MIN_KHOAN = 2

# --- loại khỏi index (Task 11) ---
DUPLICATE_INDEX_THRESHOLD = 3  # nội dung lặp >= mức này -> is_indexed=False
BOILERPLATE_MIN_TOKENS = 16    # chunk ngắn hơn mức này VÀ trùng -> loại

# --- breadcrumb (Task 6) ---
# Tiền tố lặp trên mọi chunk cùng văn bản: không thêm khả năng phân biệt nào
# giữa các chunk mà chiếm chỗ cố định.
TITLE_STRIP_PREFIXES = [
    "Nghị định quy định ",
    "Nghị định về ",
    "Nghị định ",
    "Văn bản hợp nhất ",
    "Bộ luật ",
    "Luật ",
]

MD_DIR = Path("data/markdown")
CHUNK_DIR = Path("data/chunks")

# Bất đẳng thức nền: một chunk bị cắt luôn phải đủ chỗ cho breadcrumb trần,
# stem trần và ít nhất 60 token thân. Vỡ ở đây là lỗi cấu hình, phải nổ ngay
# lúc import chứ không để lộ ra lúc chạy giữa corpus.
assert BREADCRUMB_HARD_CAP + STEM_MAX_TOKENS + 60 <= BUDGET, (
    "Cấu hình ngân sách không khả thi: "
    f"{BREADCRUMB_HARD_CAP} + {STEM_MAX_TOKENS} + 60 > {BUDGET}"
)
assert BREADCRUMB_RESERVE_MIN <= BREADCRUMB_TRUNC_MIN, (
    "BREADCRUMB_RESERVE_MIN phải đủ cho mức breadcrumb thấp nhất"
)


@functools.cache
def project_root() -> Path:
    """Tìm thư mục gốc dự án bằng cách đi ngược lên tới ``pyproject.toml``."""
    for candidate in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


# ========================================================================
# MODELS
# ========================================================================
#
# Kiểu dữ liệu dùng chung cho 8 pass của bước chunking.
BlockKind = Literal["heading", "paragraph", "table", "frontmatter"]
NodeKind = Literal[
    "root", "phan", "phu_luc", "chuong", "muc", "dieu", "khoan", "phu_luc_muc"
]


@dataclass(frozen=True, slots=True)
class MdBlock:
    """Một khối markdown thô, giữ nguyên số dòng để validator kiểm độ phủ."""

    kind: BlockKind
    level: int | None
    text: str
    line_start: int
    line_end: int

    @property
    def lines(self) -> range:
        return range(self.line_start, self.line_end + 1)


@dataclass(slots=True)
class DocNode:
    """Một nút của cây tài liệu. ``body`` là nội dung thuộc trực tiếp nút này."""

    level: int
    label: str
    kind: NodeKind
    body: list[MdBlock] = field(default_factory=list)
    children: list["DocNode"] = field(default_factory=list)
    ordinal: str | None = None
    title: str | None = None
    parent: "DocNode | None" = field(default=None, repr=False, compare=False)
    heading: MdBlock | None = None

    def ancestors(self) -> list["DocNode"]:
        """Từ gốc xuống tới cha trực tiếp."""
        chain: list[DocNode] = []
        node = self.parent
        while node is not None:
            chain.append(node)
            node = node.parent
        return list(reversed(chain))

    def find_ancestor(self, kind: NodeKind) -> "DocNode | None":
        node = self.parent
        while node is not None:
            if node.kind == kind:
                return node
            node = node.parent
        return None

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(slots=True)
class ChunkUnit:
    """Đơn vị chunk do Pass 3 chọn, trước khi cắt."""

    node: DocNode
    branch: int
    body: list[MdBlock]
    # Câu dẫn của Điều cha, nhân bản vào chunk con (Task 5b). Rỗng nếu không có.
    leadin_blocks: list[MdBlock] = field(default_factory=list)
    khoan_range: str | None = None
    warning: str | None = None


@dataclass(slots=True)
class Part:
    """Một mảnh của đơn vị sau khi cắt (Pass 5)."""

    index: int
    total: int
    tier: int | None
    body_content: str
    body_embed: str
    lines: list[int]
    duplicated_lines: list[int] = field(default_factory=list)
    table: "TableInfo | None" = None
    table_markdown: str | None = None


@dataclass(slots=True)
class TableInfo:
    """Bảng đã được rule hoá thành summary + chuỗi tra cứu lexical."""

    table_id: str
    table_raw: str
    table_summary: str
    table_search_text: str
    cells: list[str]


@dataclass(slots=True)
class Chunk:
    """Bản ghi cuối cùng: đúng các cột của bảng ``chunks`` cộng vài trường nội bộ."""

    chunk_id: str
    so_hieu: str
    content: str
    embedding_text: str
    token_count: int

    breadcrumb_full: str
    breadcrumb_embed: str
    stem: str | None

    phan: str | None
    chuong: str | None
    muc: str | None
    dieu_so: str | None
    dieu_tieu_de: str | None
    khoan_so: str | None
    khoan_range: str | None
    is_phu_luc: bool

    part_index: int
    part_total: int
    split_tier: int | None
    sibling_expand: bool

    has_negation: bool
    negation_terms: list[str]
    is_indexed: bool

    has_table: bool
    table_id: str | None
    table_raw: str | None
    table_summary: str | None
    table_search_text: str | None

    amended_by: str | None
    ngay_hieu_luc: str | None
    chunker_version: str
    parser_version: str

    # --- nội bộ, không vào DB ---
    khoan_id: str = field(default="", repr=False)
    body_content: str = field(default="", repr=False)
    body_embed: str = field(default="", repr=False)
    lines: list[int] = field(default_factory=list, repr=False)
    duplicated_lines: list[int] = field(default_factory=list, repr=False)
    unit_content: str = field(default="", repr=False)
    stem_full: str = field(default="", repr=False)
    leadin_full: str = field(default="", repr=False)
    leadin_used: str = field(default="", repr=False)
    leadin_level: str | None = field(default=None, repr=False)
    stem_level: str | None = field(default=None, repr=False)
    breadcrumb_level: str = field(default="none", repr=False)
    breadcrumb_truncated: bool = field(default=False, repr=False)
    branch: int = field(default=0, repr=False)

    DB_FIELDS = (
        "chunk_id", "so_hieu", "content", "embedding_text", "token_count",
        "breadcrumb_full", "breadcrumb_embed", "stem",
        "phan", "chuong", "muc", "dieu_so", "dieu_tieu_de", "khoan_so",
        "khoan_range", "is_phu_luc",
        "part_index", "part_total", "split_tier", "sibling_expand",
        "has_negation", "negation_terms", "is_indexed",
        "has_table", "table_id", "table_raw", "table_summary",
        "table_search_text",
        "amended_by", "ngay_hieu_luc", "chunker_version", "parser_version",
    )

    def to_row(self) -> dict[str, Any]:
        """Chỉ các cột có trong bảng ``chunks``."""
        return {name: getattr(self, name) for name in self.DB_FIELDS}


@dataclass(frozen=True, slots=True)
class Issue:
    """Kết quả của Pass 7."""

    code: str
    severity: Literal["error", "warning"]
    chunk_id: str | None
    detail: str

    def __str__(self) -> str:
        where = self.chunk_id or "-"
        return f"[{self.severity}] {self.code} @ {where}: {self.detail}"


# ========================================================================
# TOKENIZER
# ========================================================================
#
# Đếm token đúng chuẩn model embedding.
#
# Sai ở tầng này thì mọi con số phía sau đều sai, nên bốn ràng buộc dưới đây là
# bắt buộc:
#
# * Đếm trên chuỗi ĐÃ segment bằng ``ViTokenizer``. Gạch dưới nối từ ghép làm
#   đổi kết quả BPE; đếm trên chuỗi thô cho số thấp hơn thực tế → tràn im lặng.
# * Dùng ``len(...["input_ids"])`` để tính cả ``[CLS]``/``[SEP]``.
# * Không heuristic ký tự ở bất kỳ đâu. ``5.310.000`` bị BPE cắt thành 6–9
#   subword, lệch rất xa tỉ lệ 3,2 ký tự/token.
# * ``from_pretrained`` chỉ chạy trong hàm có cache, không ở cấp module.
_TOKENIZER = None

RE_WS = re.compile(r"\s+")


def get_tokenizer():
    """Trả tokenizer đã cache. Lần gọi đầu mới nạp model từ đĩa."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL)
    return _TOKENIZER


def check_model_max_length() -> int:
    """Kiểm tra ``model_max_length`` khớp với ``MAX_TOKENS``.

    Khác thì dừng và báo, tuyệt đối không tự điều chỉnh ngân sách theo model:
    toàn bộ thang cắt trong spec được đo cho trần 192.
    """
    actual = int(get_tokenizer().model_max_length)
    if actual != MAX_TOKENS:
        raise RuntimeError(
            f"model_max_length của {EMBED_MODEL} là {actual}, không phải "
            f"{MAX_TOKENS}. Dừng: ngân sách trong config.py được đo cho 192."
        )
    return actual


def to_model_input(text: str) -> str:
    """Segment tiếng Việt đúng như lúc gọi model."""
    from pyvi import ViTokenizer

    return ViTokenizer.tokenize(text)


@functools.lru_cache(maxsize=200_000)
def count_tokens(text: str) -> int:
    """Số token thật khi đưa ``text`` vào model, kể cả token đặc biệt."""
    return len(get_tokenizer()(to_model_input(text))["input_ids"])


def _word_boundaries(text: str) -> list[int]:
    """Vị trí kết thúc hợp lệ của một bản cắt: cuối mỗi từ của chuỗi GỐC."""
    return [match.start() for match in RE_WS.finditer(text)] + [len(text)]


def truncate_to_tokens(text: str, n: int) -> str:
    """Cắt ``text`` về <= ``n`` token, luôn lùi về ranh giới từ.

    Nhị phân trên ranh giới khoảng trắng của chuỗi GỐC. Không cắt trên chuỗi
    đã segment: cắt giữa một từ ghép ``lao_động`` sẽ sinh ra chuỗi không còn
    đọc lại được và không tái tạo được văn bản gốc.
    """
    text = text.strip()
    if not text or n <= 0:
        return ""
    if count_tokens(text) <= n:
        return text

    bounds = _word_boundaries(text)
    low, high = 0, len(bounds) - 1  # low luôn là bản chắc chắn vừa (rỗng)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[: bounds[mid]].rstrip()
        if count_tokens(candidate) <= n:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


# ========================================================================
# BLOCK_PARSER
# ========================================================================
#
# Pass 1 — markdown → danh sách block.
#
# Không dùng thư viện markdown: quy ước heading của loader là cố định và đã
# biết, còn thư viện tổng quát sẽ diễn giải lại bảng và danh sách theo cách
# riêng của nó, làm số dòng không còn khớp với file gốc.
RE_HEADING = re.compile(r"^(#{1,5})\s+(.*)$")
RE_TABLE_LINE = re.compile(r"^\s*\|")
RE_FRONT_MATTER_FENCE = re.compile(r"^---\s*$")


def parse_blocks(md_text: str) -> list[MdBlock]:
    """Tách văn bản thành block, giữ ``line_start``/``line_end`` 1-based."""
    lines = md_text.split("\n")
    blocks: list[MdBlock] = []

    index = 0
    total = len(lines)

    # --- front matter: chỉ hợp lệ khi nằm ngay đầu file ---
    if total and RE_FRONT_MATTER_FENCE.match(lines[0]):
        for end in range(1, total):
            if RE_FRONT_MATTER_FENCE.match(lines[end]):
                blocks.append(
                    MdBlock(
                        kind="frontmatter",
                        level=None,
                        text="\n".join(lines[1:end]),
                        line_start=1,
                        line_end=end + 1,
                    )
                )
                index = end + 1
                break

    paragraph: list[str] = []
    paragraph_start = 0

    def flush_paragraph(end_line: int) -> None:
        nonlocal paragraph, paragraph_start
        if paragraph:
            blocks.append(
                MdBlock(
                    kind="paragraph",
                    level=None,
                    text="\n".join(paragraph),
                    line_start=paragraph_start,
                    line_end=end_line,
                )
            )
            paragraph = []
            paragraph_start = 0

    while index < total:
        line = lines[index]
        line_no = index + 1

        if not line.strip():
            flush_paragraph(line_no - 1)
            index += 1
            continue

        heading = RE_HEADING.match(line)
        if heading:
            flush_paragraph(line_no - 1)
            blocks.append(
                MdBlock(
                    kind="heading",
                    level=len(heading.group(1)),
                    text=heading.group(2).strip(),
                    line_start=line_no,
                    line_end=line_no,
                )
            )
            index += 1
            continue

        # Bảng phải được nhận diện TRƯỚC đoạn văn: một dòng `|` đứng ngay sau
        # câu dẫn vẫn là mở đầu bảng chứ không phải dòng tiếp của đoạn.
        if RE_TABLE_LINE.match(line):
            flush_paragraph(line_no - 1)
            end = index
            while end < total and RE_TABLE_LINE.match(lines[end]):
                end += 1
            blocks.append(
                MdBlock(
                    kind="table",
                    level=None,
                    text="\n".join(lines[index:end]),
                    line_start=line_no,
                    line_end=end,
                )
            )
            index = end
            continue

        if not paragraph:
            paragraph_start = line_no
        paragraph.append(line)
        index += 1

    flush_paragraph(total)
    return blocks


def parse_front_matter(blocks: list[MdBlock]) -> dict[str, Any]:
    """Đọc YAML đầu file. Trả dict rỗng nếu file không có front matter."""
    import yaml

    for block in blocks:
        if block.kind == "frontmatter":
            data = yaml.safe_load(block.text)
            return data if isinstance(data, dict) else {}
    return {}


def content_lines(blocks: list[MdBlock]) -> set[int]:
    """Tập dòng mang nội dung — căn cứ kiểm tra độ phủ ở Pass 7."""
    covered: set[int] = set()
    for block in blocks:
        if block.kind in ("heading", "frontmatter"):
            continue
        covered.update(block.lines)
    return covered


RE_BOLD = re.compile(r"\*\*")
RE_HEADING_MARK = re.compile(r"^#{1,5}\s+", re.MULTILINE)
RE_BULLET = re.compile(r"^\s*-\s+", re.MULTILINE)
RE_HR = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)
RE_MULTI_SPACE = re.compile(r"[ \t]+")
# Loader dùng <br> để giữ xuống dòng trong ô bảng. Thẻ này phải biến mất trước
# khi văn bản đi vào embedding hoặc tsvector, nếu không "Mức lương tối thiểu
# tháng<br>" sẽ thành tên cột trong mọi câu của table_summary.
RE_HTML_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)


def render_block(block: MdBlock) -> str:
    """Dựng lại markdown gốc của một block (dùng cho ``content``)."""
    if block.kind == "heading":
        return f"{'#' * (block.level or 1)} {block.text}"
    return block.text


def blocks_to_content(blocks: list[MdBlock]) -> str:
    """Nối block thành markdown, giữ nguyên định dạng kể cả bảng và nhãn Khoản."""
    return "\n\n".join(render_block(block) for block in blocks if block.text.strip())


def to_plain(text: str) -> str:
    """Bỏ ký tự markdown khỏi chuỗi đi vào ``embedding_text``.

    Model được train trên câu tiếng Việt tự nhiên; ``#``, ``**``, ``|`` chỉ là
    nhiễu. Dòng bảng bị bỏ hẳn — bảng đi vào embedding qua ``table_summary``
    (Task 10), không bao giờ ở dạng thô.
    """
    kept = [line for line in text.split("\n") if not RE_TABLE_LINE.match(line)]
    plain = RE_HTML_BREAK.sub(" ", "\n".join(kept))
    plain = RE_HR.sub("", plain)
    plain = RE_HEADING_MARK.sub("", plain)
    plain = RE_BULLET.sub("", plain)
    plain = RE_BOLD.sub("", plain)
    plain = plain.replace("\n", " ")
    return RE_MULTI_SPACE.sub(" ", plain).strip()


# ========================================================================
# TREE_BUILDER
# ========================================================================
#
# Pass 2 — block → cây tài liệu.
#
# Ngăn xếp theo ``level``: heading level L đẩy ngăn xếp về nút có level < L rồi
# push. Cây chấp nhận **nhảy cấp** và không bao giờ chèn nút giả — ba bất thường
# dưới đây đều có thật trong corpus và đều hợp lệ:
#
# * ``#####`` nằm trực tiếp dưới ``#`` (khối Phụ lục).
# * Điều không có Khoản (72 ca): ``body`` không rỗng, ``children`` rỗng.
# * ``body`` không rỗng trên nút có ``children`` (134 câu dẫn của Điều).
RE_PHU_LUC = re.compile(r"^PHỤ\s+LỤC\b", re.IGNORECASE)
RE_PHAN = re.compile(r"^Phần\b\s*(.*)$", re.IGNORECASE)
RE_CHUONG = re.compile(r"^Chương\s+([IVXLCDM]+|\d+)\s*\.?\s*(.*)$", re.IGNORECASE)
RE_MUC = re.compile(r"^Mục\s+(\d+[a-zđ]?)\s*\.?\s*(.*)$", re.IGNORECASE)
RE_DIEU = re.compile(r"^Điều\s+(\d+[a-zđ]?)\s*\.\s*(.*)$")
RE_KHOAN = re.compile(r"^Khoản\s+(\d+[a-zđ]?)\s*$")
RE_PHU_LUC_MUC = re.compile(r"^(\d+[a-zđ]?)\s*\.\s*(.*)$")


def classify(level: int, label: str, in_phu_luc: bool) -> tuple[NodeKind, str | None, str | None]:
    """Trả ``(kind, ordinal, title)`` cho một dòng heading."""
    if level == 1:
        if RE_PHU_LUC.match(label):
            return "phu_luc", None, label
        match = RE_PHAN.match(label)
        return "phan", (match.group(1).strip() or None) if match else None, label
    if level == 2:
        match = RE_CHUONG.match(label)
        if match:
            return "chuong", match.group(1), match.group(2).strip() or None
        return "chuong", None, label
    if level == 3:
        match = RE_MUC.match(label)
        if match:
            return "muc", match.group(1), match.group(2).strip() or None
        return "muc", None, label
    if level == 4:
        match = RE_DIEU.match(label)
        if match:
            return "dieu", match.group(1), match.group(2).strip() or None
        return "dieu", None, label

    match = RE_KHOAN.match(label)
    if match:
        return "khoan", match.group(1), None
    # Trong Phụ lục loader ghi "##### 28. Thành phố Hồ Chí Minh": nhãn đã mang
    # tên tỉnh, và tên đó là thứ duy nhất phân biệt các mục với nhau.
    match = RE_PHU_LUC_MUC.match(label)
    if match:
        return "phu_luc_muc", match.group(1), match.group(2).strip() or None
    return ("phu_luc_muc" if in_phu_luc else "khoan"), None, label


def build_tree(blocks: list[MdBlock]) -> DocNode:
    """Dựng cây từ danh sách block của Pass 1."""
    root = DocNode(level=0, label="", kind="root")
    stack: list[DocNode] = [root]
    in_phu_luc = False

    for block in blocks:
        if block.kind == "frontmatter":
            continue
        if block.kind != "heading":
            stack[-1].body.append(block)
            continue

        level = block.level or 1
        kind, ordinal, title = classify(level, block.text, in_phu_luc)
        if kind == "phu_luc":
            in_phu_luc = True
        elif level <= 4:
            in_phu_luc = False

        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()

        node = DocNode(
            level=level,
            label=block.text,
            kind=kind,
            ordinal=ordinal,
            title=title,
            parent=stack[-1],
            heading=block,
        )
        stack[-1].children.append(node)
        stack.append(node)

    return root


# ========================================================================
# UNIT_RESOLVER
# ========================================================================
#
# Pass 3 — cây tài liệu → danh sách đơn vị chunk.
#
# Năm nhánh, xét theo thứ tự, dừng ở nhánh đầu tiên khớp. Đơn vị mặc định là
# Khoản (NT-1); bốn nhánh còn lại là ngoại lệ đã được đo trên corpus.
# Nút mang nội dung ngoài khung Điều/Khoản: lời nói đầu, câu dẫn của Chương,
# dòng "(Kèm theo Nghị định số ...)" của Phụ lục. Hiếm, nhưng nếu không thành
# chunk thì các dòng đó không thuộc chunk nào và vi phạm NT-7.
LOOSE_BODY_KINDS = ("root", "phan", "phu_luc", "chuong", "muc")


def dieu_full_blocks(node: DocNode) -> list[MdBlock]:
    """Toàn bộ block của một Điều, kể cả nhãn ``##### Khoản N`` của các con."""
    blocks = list(node.body)
    for child in node.children:
        if child.heading is not None:
            blocks.append(child.heading)
        blocks.extend(child.body)
    return blocks


def _collapsible(node: DocNode) -> bool:
    """Điều liệt kê ngắn: chia theo Khoản chỉ tạo ra vector gần như rỗng nghĩa."""
    khoan_children = [child for child in node.children if child.kind == "khoan"]
    if len(khoan_children) < COLLAPSE_DIEU_MIN_KHOAN:
        return False
    if len(khoan_children) != len(node.children):
        return False
    if any(child.children for child in khoan_children):
        return False
    blocks = dieu_full_blocks(node)
    # Điều có bảng thì không gộp. ``to_plain`` bỏ hẳn dòng bảng nên phép đo
    # token ở đây thấy Điều rất ngắn, trong khi ``table_summary`` thật có thể
    # hơn 100 token — gộp xong sẽ tràn ngân sách. Gộp cũng làm mất khoan_so của
    # chunk bảng, tức mất khả năng trích dẫn tới cấp Khoản.
    if any(block.kind == "table" for block in blocks):
        return False
    plain = to_plain(blocks_to_content(blocks))
    return count_tokens(plain) <= COLLAPSE_DIEU_MAX_TOKENS


def resolve_chunk_units(root: DocNode) -> list[ChunkUnit]:
    """Duyệt cây theo thứ tự tài liệu và chọn đơn vị chunk."""
    units: list[ChunkUnit] = []
    collapsed: set[int] = set()

    for node in root.walk():
        if id(node) in collapsed:
            continue

        # --- nhánh 1: gộp Điều liệt kê ngắn ---
        if node.kind == "dieu" and node.children and _collapsible(node):
            khoan_children = [c for c in node.children if c.kind == "khoan"]
            for child in khoan_children:
                collapsed.update(id(sub) for sub in child.walk())
            ordinals = [c.ordinal for c in khoan_children if c.ordinal]
            khoan_range = f"{ordinals[0]}-{ordinals[-1]}" if ordinals else None
            units.append(
                ChunkUnit(
                    node=node,
                    branch=1,
                    body=dieu_full_blocks(node),
                    khoan_range=khoan_range,
                )
            )
            continue

        # --- nhánh 2: Khoản ---
        if node.kind == "khoan":
            if not node.body and not node.children:
                continue
            dieu = node.find_ancestor("dieu")
            units.append(
                ChunkUnit(
                    node=node,
                    branch=2,
                    body=list(node.body),
                    leadin_blocks=list(dieu.body) if dieu is not None else [],
                )
            )
            continue

        # --- nhánh 3: mục Phụ lục ---
        if node.kind == "phu_luc_muc":
            if not node.body:
                continue
            units.append(ChunkUnit(node=node, branch=3, body=list(node.body)))
            continue

        # --- nhánh 4: Điều không có Khoản ---
        if node.kind == "dieu" and not node.children:
            if not node.body:
                continue
            units.append(ChunkUnit(node=node, branch=4, body=list(node.body)))
            continue

        # --- nhánh 5: nội dung rời nằm ngoài khung Điều ---
        if node.kind in LOOSE_BODY_KINDS and node.body:
            units.append(
                ChunkUnit(
                    node=node,
                    branch=5,
                    body=list(node.body),
                    warning=f"nội dung rời trên nút {node.kind!r}: {node.label[:60]!r}",
                )
            )

    return units


def branch_distribution(units: list[ChunkUnit]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for unit in units:
        counts[unit.branch] = counts.get(unit.branch, 0) + 1
    return dict(sorted(counts.items()))


# ========================================================================
# STEM
# ========================================================================
#
# Pass 4 — trích mệnh đề chi phối (stem), câu dẫn và phát hiện phủ định.
#
# Task này quyết định chunk có nói ĐÚNG hay không sau khi bị chia. NT-5: bản
# cắt dở của một mệnh đề chi phối tệ hơn không có gì — *"Thu nhập chịu thuế gồm
# các loại sau đây, trừ thu nhập được miễn thuế quy định tại Điều 4"* mà mất vế
# ``trừ`` thì chunk khẳng định sai.
NEGATION_PATTERNS = [
    r"\btrừ\b", r"\bngoại trừ\b", r"\bloại trừ\b",
    r"\bkhông được\b", r"\bkhông phải\b", r"\bkhông bao gồm\b",
    r"\bkhông thuộc\b", r"\bkhông áp dụng\b", r"\bkhông tính\b",
    r"\bchưa được\b", r"\btrừ trường hợp\b",
]

_NEGATION_RE = [re.compile(pattern, re.IGNORECASE) for pattern in NEGATION_PATTERNS]

# Ranh giới câu: bắt buộc chữ hoa sau khoảng trắng để không cắt nhầm ở
# "khoản 2 Điều 8." nằm giữa câu.
RE_SENTENCE = re.compile(
    r"(?<=[.;])\s+(?=[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯ])"
)

RE_COLON = re.compile(r":")


def find_negations(text: str) -> list[str]:
    """Các marker phủ định xuất hiện trong ``text``, theo thứ tự khai báo."""
    found: list[str] = []
    for pattern in _NEGATION_RE:
        match = pattern.search(text)
        if match:
            found.append(match.group(0).lower())
    return found


def split_sentences(text: str) -> list[str]:
    """Tách câu. Trả về ít nhất một phần tử khi ``text`` không rỗng."""
    parts = [part.strip() for part in RE_SENTENCE.split(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def extract_stem(text: str) -> str:
    """Mệnh đề chi phối của một đơn vị.

    Định nghĩa tất định: từ đầu đơn vị tới **dấu hai chấm đầu tiên** (bao gồm
    dấu hai chấm) — trong văn bản pháp luật Việt Nam ``:`` gần như luôn đánh
    dấu ranh giới giữa mệnh đề chi phối và phần liệt kê. Không có dấu hai chấm
    thì lấy câu đầu tiên.
    """
    text = text.strip()
    if not text:
        return ""
    match = RE_COLON.search(text)
    if match:
        return text[: match.end()].strip()
    sentences = split_sentences(text)
    return sentences[0] if sentences else ""


def shorten_governing(text: str, max_tokens: int = STEM_MAX_TOKENS) -> tuple[str, str]:
    """Thang bậc A/B/C cho stem và câu dẫn.

    * A — vừa ngân sách, dùng nguyên văn.
    * B — vượt trần nhưng bản rút gọn vẫn giữ **đủ mọi** marker phủ định.
    * C — rút gọn sẽ làm mất marker phủ định → bỏ hoàn toàn (NT-5).
    """
    text = text.strip()
    if not text:
        return "", "A"
    if count_tokens(text) <= max_tokens:
        return text, "A"

    candidate = split_sentences(text)[0]
    if count_tokens(candidate) > max_tokens:
        candidate = truncate_to_tokens(candidate, max_tokens)

    if candidate and set(find_negations(text)) <= set(find_negations(candidate)):
        return candidate, "B"
    return "", "C"


# ========================================================================
# BREADCRUMB
# ========================================================================
#
# Task 6 — hai bản breadcrumb.
#
# ``breadcrumb_full`` để hiển thị và đưa vào prompt, không giới hạn độ dài.
# ``breadcrumb_embed`` sinh theo ngân sách được cấp ở Task 7 và **không chứa số
# hiệu**: ``112/VBHN-VPQH``, ``Điều 9``, ``Khoản 2`` là chuỗi số mà dense vector
# không phân biệt được, trong khi chúng đã có đủ trong metadata và kênh lexical.
# Mức breadcrumb, dùng cho bảng phân bố ở cuối lượt chạy.
LEVEL_FULL = "full"
LEVEL_DIEU_TEN = "dieu_ten"
LEVEL_DIEU = "dieu"
LEVEL_NONE = "none"


def short_title(ten_van_ban: str | None) -> str:
    """Bỏ tiền tố lặp trên mọi chunk cùng văn bản."""
    if not ten_van_ban:
        return ""
    title = ten_van_ban.strip()
    for prefix in TITLE_STRIP_PREFIXES:
        if title.lower().startswith(prefix.lower()):
            title = title[len(prefix):].strip()
            break
    return title[:1].upper() + title[1:] if title else ""


def build_breadcrumb_full(node: DocNode, doc_meta: dict[str, Any]) -> str:
    """Đường dẫn đầy đủ từ tên văn bản xuống tới nhãn của nút."""
    head = " ".join(
        part
        for part in (doc_meta.get("ten_van_ban"), doc_meta.get("so_hieu"))
        if part
    )
    parts = [head] if head else []
    for ancestor in node.ancestors():
        if ancestor.kind == "root":
            continue
        parts.append(ancestor.label)
    if node.kind != "root":
        parts.append(node.label)
    return " > ".join(part for part in parts if part)


def _dieu_title(node: DocNode) -> str:
    """Tiêu đề Điều, không kèm số. Với mục Phụ lục thì là tên tỉnh/thành."""
    if node.kind == "phu_luc_muc":
        return node.title or node.label
    if node.kind == "dieu":
        return node.title or ""
    dieu = node.find_ancestor("dieu")
    if dieu is not None:
        return dieu.title or ""
    if node.kind in ("chuong", "muc"):
        return node.title or node.label
    ancestor = node.find_ancestor("chuong") or node.find_ancestor("muc")
    return (ancestor.title or ancestor.label) if ancestor is not None else ""


def _chuong_title(node: DocNode) -> str:
    chuong = node if node.kind == "chuong" else node.find_ancestor("chuong")
    return (chuong.title or "") if chuong is not None else ""


def _muc_title(node: DocNode) -> str:
    muc = node if node.kind == "muc" else node.find_ancestor("muc")
    return (muc.title or "") if muc is not None else ""


def breadcrumb_components(node: DocNode, doc_meta: dict[str, Any]) -> list[str]:
    """Thành phần theo ưu tiên giảm dần: Điều → tên văn bản → Chương → Mục."""
    return [
        _dieu_title(node),
        short_title(doc_meta.get("ten_van_ban")),
        _chuong_title(node),
        _muc_title(node),
    ]


def _join(parts: list[str]) -> str:
    kept = [part.strip().rstrip(".") for part in parts if part and part.strip()]
    return ". ".join(kept) + "." if kept else ""


def build_breadcrumb_embed(
    node: DocNode, doc_meta: dict[str, Any], budget: int
) -> tuple[str, str, bool]:
    """Trả ``(breadcrumb_embed, mức, đã_cắt_tiêu_đề_Điều)``.

    Bốn mức theo ngân sách còn lại (Task 7c). Tiêu đề Điều là thành phần ưu
    tiên cao nhất và không bao giờ bị bỏ khi còn ngân sách — mảnh 3/5 mà không
    có tiêu đề Điều thì gần như không truy hồi được.
    """
    cap = min(budget, BREADCRUMB_HARD_CAP)
    if cap < BREADCRUMB_TRUNC_MIN:
        return "", LEVEL_NONE, False

    dieu, ten, chuong, muc = breadcrumb_components(node, doc_meta)

    if budget >= BREADCRUMB_FULL_MIN:
        wanted, level = [dieu, ten, chuong, muc], LEVEL_FULL
    elif budget >= BREADCRUMB_DIEU_MIN:
        wanted, level = [dieu, ten], LEVEL_DIEU_TEN
    else:
        wanted, level = [dieu], LEVEL_DIEU

    # Thêm dần theo ưu tiên, dừng khi thành phần tiếp theo không còn vừa.
    chosen: list[str] = []
    for part in wanted:
        if not part:
            continue
        candidate = _join(chosen + [part])
        if count_tokens(candidate) <= cap:
            chosen.append(part)

    truncated = False
    if not chosen and dieu:
        # Tiêu đề Điều dài hơn cả ngân sách (Điều 129 Luật BHXH ~78 token):
        # cắt về đúng ngân sách chứ không bỏ.
        cut = truncate_to_tokens(dieu, max(cap - 1, 1))
        if cut:
            chosen, truncated = [cut], True

    text = _join(chosen)
    if not text:
        return "", LEVEL_NONE, truncated
    if len(chosen) >= 3:
        level = LEVEL_FULL
    elif len(chosen) == 2:
        level = LEVEL_DIEU_TEN
    else:
        level = LEVEL_DIEU
    return text, level, truncated


# ========================================================================
# BUDGET
# ========================================================================
#
# Task 7 — cấp phát ngân sách token động.
#
# Vì sao động: breadcrumb mang **3 từ khóa mới** cho Khoản ngắn (<200 ký tự, 880
# ca) nhưng **0 từ khóa mới** cho Khoản dài (>500 ký tự, 331 ca). Ngân sách chỉ
# khan hiếm khi thân dài, nên cấp phát phải phụ thuộc độ dài thân.
#
# Phá vòng phụ thuộc (ngân sách thân ↔ breadcrumb) bằng cách dành sẵn một sàn
# cố định ``BREADCRUMB_RESERVE_MIN`` lúc cắt, rồi cấp phát thật cho từng mảnh
# sau khi đã biết độ dài của nó.
@dataclass(slots=True)
class Allocation:
    """Kết quả bước 1–2: phần cố định và ngân sách còn lại cho thân."""

    body_allowance: int
    stem_embed: str
    stem_level: str
    leadin_embed: str
    leadin_level: str
    will_split: bool


def plan_budget(body_plain: str, stem_text: str, leadin_text: str) -> Allocation:
    """Thứ tự ưu tiên NT-4: thân → stem → breadcrumb."""
    stem_embed, stem_level = shorten_governing(stem_text)
    leadin_embed, leadin_level = shorten_governing(leadin_text)

    def tokens(text: str) -> int:
        return count_tokens(text) if text else 0

    body_tokens = count_tokens(body_plain) if body_plain else 0

    allowance_single = BUDGET - tokens(leadin_embed) - BREADCRUMB_RESERVE_MIN
    if body_tokens <= allowance_single:
        # Không bị chia: stem đã nằm sẵn trong thân, nhân bản sẽ lặp hai lần.
        return Allocation(
            body_allowance=allowance_single,
            stem_embed="",
            stem_level=stem_level,
            leadin_embed=leadin_embed,
            leadin_level=leadin_level,
            will_split=False,
        )

    # Từ đây là nhánh có chia: stem được nhân bản vào MỌI mảnh, nên phần cố
    # định phình gấp đôi. Quá trần thì bỏ câu dẫn trước (nó chỉ là ngữ cảnh của
    # Điều), giữ stem vì stem thuộc chính đơn vị này. Mức "D" phân biệt với bậc
    # C của NT-5: ở đây bỏ vì hết ngân sách, không phải vì mất marker phủ định.
    if tokens(stem_embed) + tokens(leadin_embed) > PREFIX_MAX_TOKENS:
        leadin_embed, leadin_level = "", "D"

    allowance = BUDGET - tokens(stem_embed) - tokens(leadin_embed) - BREADCRUMB_RESERVE_MIN
    if allowance < MIN_BODY_ALLOWANCE and leadin_embed:
        leadin_embed, leadin_level = "", "D"
        allowance = BUDGET - tokens(stem_embed) - BREADCRUMB_RESERVE_MIN
    if allowance < MIN_BODY_ALLOWANCE and stem_embed:
        stem_embed, stem_level = "", "D"
        allowance = BUDGET - BREADCRUMB_RESERVE_MIN

    return Allocation(
        body_allowance=max(allowance, MIN_BODY_ALLOWANCE),
        stem_embed=stem_embed,
        stem_level=stem_level,
        leadin_embed=leadin_embed,
        leadin_level=leadin_level,
        will_split=True,
    )


def breadcrumb_budget(body_embed: str, stem_embed: str, leadin_embed: str) -> int:
    """Bước 4: phần ngân sách còn lại của MỘT mảnh cụ thể.

    Trả về ``remaining`` chưa cắt trần: bốn mức của 7c được quyết theo
    ``remaining``, còn ``BREADCRUMB_HARD_CAP`` do ``build_breadcrumb_embed``
    áp lên độ dài chuỗi. Cắt trần ngay ở đây thì mức "đầy đủ" (>= 60) không
    bao giờ đạt được.
    """
    used = count_tokens(body_embed) if body_embed else 0
    used += count_tokens(stem_embed) if stem_embed else 0
    used += count_tokens(leadin_embed) if leadin_embed else 0
    return max(BUDGET - used, 0)


# ========================================================================
# SPLITTER
# ========================================================================
#
# Pass 5 — cắt đơn vị dài thành mảnh (thang 5 tầng).
#
# Tiêu chí nghiệm thu cho mọi tầng: một mảnh phải **đọc độc lập mà không dẫn đến
# hiểu sai pháp lý**. Mảnh cần ba thứ — phạm vi chủ thể, điều kiện áp dụng,
# ngoại lệ. Mất bất kỳ cái nào thì mảnh đó nói sai chứ không phải nói thiếu.
# Nhận diện điểm, dùng chung định nghĩa với loader_spec.md.
RE_DIEM = re.compile(r"^(?:([a-zđư])\)|-)\s+(.*)$")

# Vế mở điều kiện chỉ tính khi đứng ĐẦU câu và viết hoa. Bắt cả dạng thường
# giữa câu sẽ dính "cho đến khi Chính phủ có quy định mới" hay "nếu quy đổi
# theo tháng ... không được thấp hơn" — những câu đã đủ nghĩa, không phải vế
# mở bị bỏ dở.
RE_CONDITION_OPEN = re.compile(r"(?:^|(?<=[.;])\s+)(Trường hợp|Khi|Nếu)\b")
RE_THEN = re.compile(r"\bthì\b", re.IGNORECASE)
RE_GOM = re.compile(r"\bgồm\b")

TIER_DIEM = 1
TIER_PARAGRAPH = 2
TIER_SEMICOLON = 3
TIER_SENTENCE = 4
TIER_COMMA = 5


@dataclass(slots=True)
class Atom:
    """Đơn vị nhỏ nhất không bị cắt tiếp ở tầng hiện hành."""

    raw: str
    plain: str
    lines: list[int]
    is_table: bool = False
    tier: int | None = None
    sep: str = "\n\n"
    sentences: list[str] = field(default_factory=list)


RE_CLAUSE_END = re.compile(r"[.;]")


def _clause_stop(text: str, start: int, limit: int) -> int:
    """Cuối mệnh đề chứa vế mở: dấu ``.``/``;`` gần nhất, hoặc vế mở kế tiếp.

    ``Trường hợp X thì Y`` luôn nằm gọn trong một câu. Tìm ``thì`` xa hơn phạm
    vi đó sẽ coi là "đã đóng" một vế mở thực ra bị bỏ ngỏ.
    """
    match = RE_CLAUSE_END.search(text, start, limit)
    return match.start() if match else limit


def condition_unclosed(text: str) -> bool:
    """Vế mở điều kiện CUỐI CÙNG còn treo — cắt ngay đây là khẳng định sai."""
    matches = list(RE_CONDITION_OPEN.finditer(text))
    if not matches:
        return False
    last = matches[-1]
    stop = _clause_stop(text, last.end(), len(text))
    return RE_THEN.search(text, last.end(), stop) is None


def unclosed_openers(text: str, key_length: int = 40) -> list[str]:
    """Mọi vế mở không có ``thì`` trước vế mở kế tiếp, kèm khoá nhận dạng.

    Khoá là đoạn văn ngay sau vế mở; nó cho phép đối chiếu cùng một vế mở giữa
    mảnh và bản gốc, kể cả khi mảnh bị cắt ngang.
    """
    matches = list(RE_CONDITION_OPEN.finditer(text))
    keys: list[str] = []
    for position, match in enumerate(matches):
        limit = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        stop = _clause_stop(text, match.end(), limit)
        if RE_THEN.search(text, match.end(), stop) is None:
            keys.append(text[match.end(): match.end() + key_length].strip())
    return keys


def _atomize(blocks: list[MdBlock]) -> list[Atom]:
    """Block → atom, có tách theo dòng khi một block chứa nhiều điểm.

    Loader gộp các dòng liền nhau thành một đoạn, nên ``- Vùng I ...`` và
    ``- Vùng II ...`` của Phụ lục nằm chung một block. Không tách theo dòng thì
    tầng 1 không bao giờ nhận ra điểm và cả khối rơi thẳng xuống tầng 3.
    """
    atoms: list[Atom] = []
    for block in blocks:
        if block.kind == "paragraph":
            lines = block.text.split("\n")
            if sum(1 for line in lines if RE_DIEM.match(line)) >= 2:
                atoms.extend(_atomize_lines(block, lines))
                continue
        raw = render_block(block)
        atoms.append(
            Atom(
                raw=raw,
                # Nhãn "##### Khoản N" ở lại trong content nhưng không đi vào
                # embedding: số hiệu không giúp dense vector phân biệt gì.
                plain="" if block.kind == "heading" else to_plain(raw),
                lines=list(block.lines),
                is_table=block.kind == "table",
            )
        )
    return atoms


def _atomize_lines(block: MdBlock, lines: list[str]) -> list[Atom]:
    """Mỗi điểm một atom; dòng không phải điểm nối vào điểm ngay trước nó."""
    groups: list[tuple[list[str], int]] = []
    for offset, line in enumerate(lines):
        if RE_DIEM.match(line) or not groups:
            groups.append(([line], block.line_start + offset))
        else:
            groups[-1][0].append(line)

    atoms: list[Atom] = []
    for text_lines, start in groups:
        raw = "\n".join(text_lines)
        atoms.append(
            Atom(
                raw=raw,
                plain=to_plain(raw),
                lines=list(range(start, start + len(text_lines))),
                sep="\n",
            )
        )
    if atoms:
        atoms[-1].sep = "\n\n"
    return atoms


def _is_diem(atom: Atom) -> bool:
    first = atom.raw.split("\n", 1)[0]
    return bool(RE_DIEM.match(first))


def _split_semicolon(text: str) -> list[str]:
    pieces = [piece.strip() for piece in re.split(r"(?<=;)\s+", text) if piece.strip()]
    return pieces


def _split_comma(text: str) -> list[str]:
    return [piece.strip() for piece in re.split(r"(?<=,)\s+", text) if piece.strip()]


def _text_pieces(plain: str) -> tuple[list[str], int]:
    """Chọn tầng 3/4/5 cho một khối văn bản liền và trả về các mảnh thô."""
    if plain.count(";") >= 2:
        pieces = _split_semicolon(plain)
        if len(pieces) >= 2:
            return pieces, TIER_SEMICOLON
    sentences = split_sentences(plain)
    if len(sentences) >= 2:
        return sentences, TIER_SENTENCE
    pieces = _split_comma(plain)
    if len(pieces) >= 2:
        return pieces, TIER_COMMA
    return [plain], TIER_COMMA


def _hard_split(text: str, allowance: int) -> list[str]:
    """Lưới an toàn cuối cùng: cắt theo ranh giới từ khi không còn dấu nào.

    Chỉ chạm tới khi một mệnh đề duy nhất dài hơn cả ngân sách và không có
    dấu chấm, chấm phẩy hay dấu phẩy nào. Giữ NT-3: thà mảnh xấu còn hơn
    tràn trần token trong im lặng.
    """
    pieces: list[str] = []
    remaining = text.strip()
    while remaining and count_tokens(to_plain(remaining)) > allowance:
        head = truncate_to_tokens(remaining, allowance)
        if not head:
            break
        pieces.append(head)
        remaining = remaining[len(head):].strip()
    if remaining:
        pieces.append(remaining)
    return pieces or [text]


def _explode_text(plain: str, allowance: int, floor: int = 0) -> list[tuple[str, int]]:
    """Cắt một khối văn bản liền xuống dưới ngân sách, trả ``(mảnh, tầng)``."""
    pieces, tier = _text_pieces(plain) if floor < TIER_COMMA else ([plain], TIER_COMMA)
    if tier <= floor or len(pieces) < 2:
        # Tầng hiện hành không cắt được gì thêm: hạ xuống tầng kế tiếp.
        if floor >= TIER_COMMA:
            return [(piece, TIER_COMMA) for piece in _hard_split(plain, allowance)]
        return _explode_text(plain, allowance, floor=max(floor + 1, tier))

    out: list[tuple[str, int]] = []
    for piece in pieces:
        if count_tokens(to_plain(piece)) <= allowance:
            out.append((piece, tier))
        else:
            out.extend(_explode_text(piece, allowance, floor=tier))
    return out


def _explode(atom: Atom, allowance: int) -> list[Atom]:
    """Một atom vượt ngân sách rơi xuống tầng 3/4/5.

    Cắt trên chuỗi RAW chứ không phải chuỗi đã bỏ markdown: ``content`` của
    các mảnh ghép lại phải bằng đúng nội dung gốc (NT-2), mà chuỗi plain đã
    mất dấu ``- `` đầu điểm.
    """
    pieces = _explode_text(atom.raw, allowance)
    if len(pieces) < 2:
        return [atom]
    exploded = [
        Atom(raw=piece, plain=to_plain(piece), lines=list(atom.lines), tier=tier, sep=" ")
        for piece, tier in pieces
    ]
    exploded[-1].sep = atom.sep
    return exploded


def _group(atoms: list[Atom], allowance: int, target: int) -> list[list[Atom]]:
    """Gom atom tới ``target``, không bao giờ vượt ``allowance``."""
    groups: list[list[Atom]] = []
    current: list[Atom] = []

    def text_of(items: list[Atom]) -> str:
        return " ".join(item.plain for item in items if item.plain)

    for atom in atoms:
        # NT-6: bảng là nguyên tử, luôn đứng riêng một mảnh.
        if atom.is_table:
            if current:
                groups.append(current)
                current = []
            groups.append([atom])
            continue

        if not current:
            current = [atom]
            continue

        candidate = current + [atom]
        size = count_tokens(text_of(candidate))
        # Chỉ hoãn cắt khi cặp ràng buộc THẬT SỰ bắc qua ranh giới: vế mở còn
        # treo ở mảnh này và vế "thì" nằm trong atom kế tiếp. Nhiều câu luật
        # viết "Trường hợp X, Y phải Z" không có "thì" — hoãn cắt vì chúng chỉ
        # làm mảnh phình vô ích.
        pair_spans = condition_unclosed(text_of(current)) and bool(
            RE_THEN.search(atom.plain)
        )
        if size > allowance:
            groups.append(current)
            current = [atom]
        elif size > target and not pair_spans:
            groups.append(current)
            current = [atom]
        else:
            current = candidate

    if current:
        groups.append(current)

    # Nhóm chỉ có heading (không chữ nào) phải nhập vào nhóm sau, nếu không sẽ
    # sinh chunk có embedding_text rỗng.
    merged: list[list[Atom]] = []
    for group in groups:
        if merged and not text_of(merged[-1]).strip():
            merged[-1] = merged[-1] + group
        else:
            merged.append(group)
    groups = merged

    # Đuôi ngắn gộp vào mảnh trước, kể cả khi vượt target.
    if len(groups) >= 2 and not groups[-1][0].is_table and not groups[-2][0].is_table:
        tail = count_tokens(text_of(groups[-1]))
        merged = count_tokens(text_of(groups[-2] + groups[-1]))
        if tail < MIN_TAIL_TOKENS and merged <= allowance:
            groups[-2] = groups[-2] + groups[-1]
            groups.pop()
    return groups


def _tier_of(group: list[Atom], unit_tier: int) -> int:
    tiers = {atom.tier for atom in group if atom.tier is not None}
    return max(tiers) if tiers else unit_tier


def _build_prefix(atom_raw: str, context: str) -> str:
    """Tiền tố lặp cho tầng 5.

    Không lặp tiền tố thì mảnh 2 của mục 28 Phụ lục chỉ còn là danh sách tên
    phường trần trụi — không vùng, không tỉnh, vô nghĩa với truy vấn
    *"phường Bàn Cờ thuộc vùng mấy"*.
    """
    head_line = to_plain(atom_raw.split("\n", 1)[0])
    match = RE_GOM.search(head_line)
    if match:
        before = head_line[: match.start()].strip().rstrip(",").strip()
        return f"{before}, thuộc {context}, gồm" if context else f"{before}, gồm"
    before = head_line.split(",", 1)[0].strip()
    return f"{before}, thuộc {context}," if context else before


def split_unit(
    blocks: list[MdBlock],
    allowance: int,
    prefix_context: str = "",
) -> list[Part]:
    """Cắt thân đơn vị thành các mảnh <= ``allowance`` token."""
    atoms = _atomize(blocks)
    if not atoms:
        return []

    whole_plain = " ".join(atom.plain for atom in atoms if atom.plain)
    if not any(atom.is_table for atom in atoms) and count_tokens(whole_plain) <= allowance:
        return [
            Part(
                index=1,
                total=1,
                tier=None,
                body_content=_join_raw(atoms),
                body_embed=whole_plain,
                lines=sorted({line for atom in atoms for line in atom.lines}),
            )
        ]

    # --- chọn tầng ở mức khối ---
    diem_count = sum(1 for atom in atoms if _is_diem(atom))
    if diem_count >= 2:
        unit_tier = TIER_DIEM
    elif len([atom for atom in atoms if not atom.is_table]) >= 2:
        unit_tier = TIER_PARAGRAPH
    else:
        unit_tier = TIER_SENTENCE

    prefix_by_atom: dict[int, str] = {}
    expanded: list[Atom] = []
    for atom in atoms:
        if atom.is_table or count_tokens(atom.plain) <= allowance:
            expanded.append(atom)
            continue
        pieces = _explode(atom, allowance)
        if len(pieces) > 1 and any(piece.tier == TIER_COMMA for piece in pieces):
            prefix = _build_prefix(atom.raw, prefix_context)
            for piece in pieces:
                prefix_by_atom[id(piece)] = prefix
        expanded.extend(pieces)

    groups = _group(expanded, allowance, min(SPLIT_TARGET_TOKENS, allowance))

    parts: list[Part] = []
    total = len(groups)
    previous_sentence = ""
    for index, group in enumerate(groups, start=1):
        tier = None if total == 1 else _tier_of(group, unit_tier)
        body_content = _join_raw(group)
        body_embed = " ".join(atom.plain for atom in group if atom.plain).strip()

        prefix = prefix_by_atom.get(id(group[0]), "")
        if prefix:
            body_embed = _apply_prefix(body_embed, prefix)

        # Chồng lấn CHỈ ở tầng 4 và CHỈ vào embedding_text.
        if tier == TIER_SENTENCE and previous_sentence and index > 1:
            overlap = truncate_to_tokens(previous_sentence, SPLIT_OVERLAP_TOKENS)
            if overlap and count_tokens(f"{overlap} {body_embed}") <= allowance:
                body_embed = f"{overlap} {body_embed}"

        sentences = split_sentences(
            " ".join(atom.plain for atom in group if atom.plain)
        )
        previous_sentence = sentences[-1] if sentences else ""

        parts.append(
            Part(
                index=index,
                total=total,
                tier=tier,
                body_content=body_content,
                body_embed=body_embed,
                lines=sorted({line for atom in group for line in atom.lines}),
            )
        )
    return parts


def _apply_prefix(body_embed: str, prefix: str) -> str:
    """Mảnh 1 đã có sẵn phần đầu dòng: thay bằng tiền tố đã chèn ngữ cảnh."""
    match = RE_GOM.search(body_embed)
    if match and match.start() < 120:
        return f"{prefix} {body_embed[match.end():].strip()}".strip()
    return f"{prefix} {body_embed}".strip()


def _join_raw(atoms: list[Atom]) -> str:
    out: list[str] = []
    for position, atom in enumerate(atoms):
        out.append(atom.raw)
        if position < len(atoms) - 1:
            out.append(atom.sep)
    return "".join(out).strip()


# ========================================================================
# TABLE
# ========================================================================
#
# Task 10 — bảng: ``table_raw`` + ``table_summary`` sinh bằng rule.
#
# Không dùng LLM. Ba trường tách bạch theo vai trò:
#
# * ``table_raw`` — markdown đầy đủ, chỉ nằm trong ``content``. Không bao giờ
#   vào ``embedding_text`` hay ``search_vector``: ``|`` và ``---`` làm nhiễu cả
#   hai kênh.
# * ``table_summary`` — câu tiếng Việt tự nhiên, vào embedding và tsvector.
# * ``table_search_text`` — biến thể chuẩn hoá cho BM25, chỉ vào tsvector.
RE_UNIT = re.compile(r"^(.*?)\s*\(\s*Đơn\s*vị\s*:\s*([^)]*)\)\s*$", re.IGNORECASE)
RE_SEPARATOR = re.compile(r"^[\s|:-]+$")
RE_THOUSAND = re.compile(r"\b\d{1,3}(?:\.\d{3})+\b")
RE_ROMAN = re.compile(r"\b(?:VIII|VII|VI|IV|IX|XII|XI|X|V|III|II|I)\b")

ROMAN_VALUES = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
    "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
}


def parse_table(markdown: str) -> tuple[list[str], list[list[str]]]:
    """Tách bảng markdown thành ``(header, rows)``, bỏ dòng phân cách."""
    rows: list[list[str]] = []
    for line in markdown.split("\n"):
        if not line.strip().startswith("|"):
            continue
        if RE_SEPARATOR.match(line.strip()):
            continue
        cells = [
            RE_HTML_BREAK.sub(" ", cell).strip()
            for cell in line.strip().strip("|").split("|")
        ]
        rows.append(cells)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def clean_header(header: str) -> tuple[str, str]:
    """``"Mức lương tối thiểu tháng(Đơn vị: đồng/tháng)"`` → tên cột + đơn vị.

    Đơn vị gắn vào giá trị ô chứ không lặp trong mọi câu của summary.
    """
    header = RE_HTML_BREAK.sub(" ", header)
    match = RE_UNIT.match(header)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return header.strip(), ""


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def build_summary(caption: str, header: list[str], rows: list[list[str]]) -> str:
    """Template 10b: mỗi hàng một câu, mỗi ô một mệnh đề ``... là ...``."""
    names_units = [clean_header(cell) for cell in header]
    sentences: list[str] = []
    for row in rows:
        if not row:
            continue
        label = row[0].strip()
        clauses = []
        for position in range(1, len(row)):
            if position >= len(names_units):
                break
            name, unit = names_units[position]
            value = row[position].strip()
            if not value:
                continue
            clauses.append(
                f"{_lower_first(name)} là {value} {unit}".strip()
            )
        if clauses:
            sentences.append(f"{label}: {'; '.join(clauses)}.")
        elif label:
            sentences.append(f"{label}.")
    return " ".join(filter(None, [caption, *sentences])).strip()


def build_summary_compact(
    caption: str, header: list[str], rows: list[list[str]]
) -> str:
    """Bản nén: tên cột nêu một lần, mỗi hàng chỉ còn giá trị.

    Dùng khi template đầy đủ vượt ngân sách. Vẫn giữ đủ mọi giá trị ô nên
    ràng buộc phủ 10e không đổi.
    """
    names_units = [clean_header(cell) for cell in header]
    columns = ", ".join(
        f"{name} ({unit})" if unit else name for name, unit in names_units[1:] if name
    )
    sentences: list[str] = []
    for row in rows:
        if not row:
            continue
        values = "; ".join(value.strip() for value in row[1:] if value.strip())
        label = row[0].strip()
        sentences.append(f"{label}: {values}." if values else f"{label}.")
    lead = f"Bảng theo {columns}." if columns else ""
    return " ".join(filter(None, [caption, lead, *sentences])).strip()


def _numeric_variants(value: str) -> list[str]:
    """Ba bẫy chuẩn hoá xử lý cùng lúc: phân cách nghìn, số La Mã, dấu."""
    variants: list[str] = []
    for match in RE_THOUSAND.finditer(value):
        variants.append(match.group(0).replace(".", ""))
    for match in RE_ROMAN.finditer(value):
        variants.append(str(ROMAN_VALUES[match.group(0)]))
    return variants


def build_search_text(header: list[str], rows: list[list[str]]) -> str:
    """Chuỗi tra cứu lexical: mọi ô, cộng biến thể số.

    ``unaccent`` do PostgreSQL lo (Task 12d), ở đây chỉ lo hai bẫy còn lại.
    """
    tokens: list[str] = []
    for cell in header:
        name, unit = clean_header(cell)
        tokens.extend(filter(None, [name, unit]))
    for row in rows:
        for cell in row:
            value = cell.strip()
            if not value:
                continue
            tokens.append(value)
            tokens.extend(_numeric_variants(value))
    return " ".join(tokens)


def all_cells(header: list[str], rows: list[list[str]]) -> list[str]:
    cells = [cell.strip() for cell in header if cell.strip()]
    for row in rows:
        cells.extend(cell.strip() for cell in row if cell.strip())
    return cells


def build_table_info(
    table_id: str, markdown: str, caption: str, budget: int
) -> TableInfo:
    """Dựng đủ ba trường của bảng, chọn template vừa ngân sách.

    Thang xuống dần: template đầy đủ → bản nén → caption rút gọn. Cắt theo
    nhóm hàng (10c) chỉ dùng đến khi bản nén vẫn tràn, và do ``assembler`` gọi.
    """
    header, rows = parse_table(markdown)
    caption_short, _ = shorten_governing(caption)
    summary = build_summary(caption_short, header, rows)
    if count_tokens(summary) > budget:
        compact = build_summary_compact(caption_short, header, rows)
        if count_tokens(compact) < count_tokens(summary):
            summary = compact
    if count_tokens(summary) > budget:
        stripped = build_summary_compact("", header, rows)
        if count_tokens(stripped) < count_tokens(summary):
            summary = stripped
    return TableInfo(
        table_id=table_id,
        table_raw=markdown,
        table_summary=summary,
        table_search_text=build_search_text(header, rows),
        cells=all_cells(header, rows),
    )


def split_summary_rows(
    caption: str, header: list[str], rows: list[list[str]], budget: int
) -> list[str]:
    """Cắt summary theo NHÓM HÀNG khi một mảnh vẫn không đủ chỗ.

    Ranh giới cắt luôn là ranh giới hàng, và mỗi mảnh mang lại đầy đủ caption
    cùng tên cột — mất header thì các hàng còn lại vô nghĩa.
    """
    chunks: list[str] = []
    current: list[list[str]] = []
    for row in rows:
        candidate = current + [row]
        if current and count_tokens(build_summary(caption, header, candidate)) > budget:
            chunks.append(build_summary(caption, header, current))
            current = [row]
        else:
            current = candidate
    if current:
        chunks.append(build_summary(caption, header, current))
    return chunks


# ========================================================================
# ASSEMBLER
# ========================================================================
#
# Pass 6 — lắp ráp ``content`` và ``embedding_text`` cho từng mảnh.
#
# Hai trường sinh độc lập:
#
# * ``content`` — trung thực với văn bản gốc, giữ nguyên markdown kể cả bảng và
#   nhãn ``##### Khoản N``, **không bao giờ bị cắt vì token** (NT-2).
# * ``embedding_text`` — văn bản THÔ (chưa segment), không ký tự markdown, và
#   không bao giờ vượt ``BUDGET`` trong im lặng (NT-3).
class TokenOverflow(RuntimeError):
    """Vượt trần token là exception, không phải warning (NT-3)."""


def _node_path_id(node: DocNode) -> str:
    """Định danh ổn định cho nhánh 5 (nội dung rời ngoài khung Điều)."""
    if node.kind == "root":
        return "pre"
    if node.kind == "phu_luc":
        return "pl"
    parts: list[str] = []
    for ancestor in [*node.ancestors(), node]:
        if ancestor.kind == "phan":
            parts.append(f"ph{ancestor.ordinal or ''}")
        elif ancestor.kind == "chuong":
            parts.append(f"c{ancestor.ordinal or ''}")
        elif ancestor.kind == "muc":
            parts.append(f"m{ancestor.ordinal or ''}")
        elif ancestor.kind == "phu_luc":
            parts.append("pl")
    return "".join(parts) or "x"


def khoan_id_for(unit: ChunkUnit, so_hieu: str) -> str:
    """Định danh đơn vị (chưa kèm số mảnh) — mọi mảnh cùng đơn vị dùng chung."""
    node = unit.node
    if unit.branch == 1:
        return f"{so_hieu}#d{node.ordinal}"
    if unit.branch == 2:
        dieu = node.find_ancestor("dieu")
        dieu_so = dieu.ordinal if dieu is not None else "x"
        return f"{so_hieu}#d{dieu_so}k{node.ordinal}"
    if unit.branch == 3:
        return f"{so_hieu}#pl#m{node.ordinal}"
    if unit.branch == 4:
        return f"{so_hieu}#d{node.ordinal}"
    return f"{so_hieu}#{_node_path_id(node)}"


def _table_blocks(blocks: list[MdBlock]) -> list[MdBlock]:
    return [block for block in blocks if block.kind == "table"]


def _caption_for(blocks: list[MdBlock], table: MdBlock) -> str:
    """Câu dẫn ngay trước bảng."""
    caption = ""
    for block in blocks:
        if block is table:
            break
        if block.kind == "paragraph":
            caption = to_plain(block.text)
    return caption


def _finalize(
    breadcrumb_embed: str,
    stem_embed: str,
    leadin_embed: str,
    body_embed: str,
    node: DocNode,
    doc_meta: dict[str, Any],
    chunk_id: str,
) -> tuple[str, str, str, str]:
    """Kiểm tra token là bước CUỐI CÙNG, không phải bước đầu.

    Tổng token của chuỗi ghép không bằng tổng token từng phần (BPE và
    ViTokenizer làm việc trên toàn chuỗi), nên phải đo lại trên chuỗi thật rồi
    thu hồi theo đúng thứ tự ưu tiên ngược: breadcrumb → câu dẫn → stem.
    """
    for _ in range(8):
        text = " ".join(
            piece for piece in (breadcrumb_embed, stem_embed, leadin_embed, body_embed)
            if piece
        ).strip()
        total = count_tokens(text)
        if total <= BUDGET:
            return text, breadcrumb_embed, stem_embed, leadin_embed

        overflow = total - BUDGET
        if breadcrumb_embed:
            target = count_tokens(breadcrumb_embed) - overflow - 1
            breadcrumb_embed = truncate_to_tokens(breadcrumb_embed, max(target, 0))
            continue
        if leadin_embed:
            leadin_embed = ""
            continue
        if stem_embed:
            stem_embed = ""
            continue
        break

    text = body_embed.strip()
    if count_tokens(text) > BUDGET:
        raise TokenOverflow(
            f"{chunk_id}: thân mảnh {count_tokens(text)} token > {BUDGET}. "
            "Splitter đã trả về mảnh vượt ngân sách."
        )
    return text, "", "", ""


def _stem_already_in_body(stem_embed: str, body_embed: str) -> bool:
    """Thân mảnh đã chứa stem chưa — so sau khi chuẩn hoá khoảng trắng.

    Dùng ``in`` chứ không ``startswith``: stem có thể là bản rút gọn bậc B nên
    không khớp đầu chuỗi tuyệt đối, nhưng vẫn nằm trọn trong thân.
    """
    stem = RE_WS.sub(" ", stem_embed).strip()
    return bool(stem) and stem in RE_WS.sub(" ", body_embed)


def _mark_duplicates(parts: list[Part]) -> None:
    """Dòng xuất hiện ở nhiều mảnh là nhân bản có chủ đích, phải đánh dấu."""
    seen: set[int] = set()
    for part in parts:
        repeated = [line for line in part.lines if line in seen]
        part.duplicated_lines = sorted(set(part.duplicated_lines) | set(repeated))
        seen.update(part.lines)


def build_parts(
    unit: ChunkUnit,
    allowance: int,
    table_ids: dict[int, str],
) -> list[Part]:
    """Cắt thân đơn vị; bảng đi đường riêng vì summary thay chỗ cho markdown."""
    tables = _table_blocks(unit.body)
    node = unit.node
    prefix_context = node.title or node.label if node.kind == "phu_luc_muc" else (
        (node.find_ancestor("dieu").title or "")
        if node.find_ancestor("dieu") is not None
        else ""
    )

    if not tables:
        return split_unit(unit.body, allowance, prefix_context)

    parts: list[Part] = []
    for table in tables:
        caption = _caption_for(unit.body, table)
        table_id = table_ids[id(table)]
        info = build_table_info(table_id, table.text, caption, allowance)
        pre_blocks = [
            block
            for block in unit.body
            if block.line_end < table.line_start and to_plain(block.text)
        ]
        header, rows = parse_table(table.text)

        body_embed = info.table_summary
        if count_tokens(body_embed) <= allowance:
            content_blocks = pre_blocks + [table]
            part = Part(
                index=len(parts) + 1,
                total=0,
                tier=None,
                body_content=blocks_to_content(content_blocks),
                body_embed=body_embed,
                lines=sorted(
                    {line for block in content_blocks for line in block.lines}
                ),
                table_markdown=table.text,
            )
            part.table = info
            parts.append(part)
        else:
            # 10c — cắt theo NHÓM HÀNG, mỗi mảnh mang lại đủ caption và tên cột.
            caption_short = info.table_summary.split(".")[0] + "."
            for position, summary in enumerate(
                split_summary_rows(caption_short, header, rows, allowance)
            ):
                piece = Part(
                    index=len(parts) + 1,
                    total=0,
                    tier=None,
                    body_content=(
                        blocks_to_content(pre_blocks + [table]) if position == 0
                        else blocks_to_content([table])
                    ),
                    body_embed=summary,
                    lines=sorted(
                        {line for block in pre_blocks + [table] for line in block.lines}
                    ),
                    table_markdown=table.text,
                )
                piece.table = info
                parts.append(piece)

    # Đoạn văn sau bảng (nếu có) thành mảnh riêng — bảng là nguyên tử (NT-6).
    last_table = tables[-1]
    post_blocks = [
        block
        for block in unit.body
        if block.line_start > last_table.line_end and to_plain(block.text)
    ]
    if post_blocks:
        for piece in split_unit(post_blocks, allowance, prefix_context):
            piece.index = len(parts) + 1
            parts.append(piece)

    total = len(parts)
    for position, piece in enumerate(parts, start=1):
        piece.index = position
        piece.total = total
    return parts


def unique_khoan_id(
    unit: ChunkUnit, so_hieu: str, registry: dict[str, int] | None
) -> str:
    """Định danh đơn vị, có hậu tố khi số Khoản lặp lại trong cùng một Điều.

    Điều 219 Bộ luật Lao động trích nguyên văn các Điều của luật khác, nên
    loader sinh ra nhiều ``##### Khoản 1`` dưới cùng một Điều. Không khử trùng
    thì hai đơn vị khác nhau dùng chung ``chunk_id`` và bản sau đè bản trước
    lúc upsert.
    """
    base = khoan_id_for(unit, so_hieu)
    if registry is None:
        return base
    seen = registry.get(base, 0) + 1
    registry[base] = seen
    return base if seen == 1 else f"{base}_{seen}"


def build_chunks(
    unit: ChunkUnit,
    doc_meta: dict[str, Any],
    table_ids: dict[int, str],
    id_registry: dict[str, int] | None = None,
) -> list[Chunk]:
    """Một đơn vị chunk → một hoặc nhiều bản ghi ``Chunk``."""
    node = unit.node
    so_hieu = str(doc_meta.get("so_hieu") or "").strip() or "UNKNOWN"

    body_markdown = blocks_to_content(unit.body)
    body_plain = to_plain(body_markdown)
    leadin_content = blocks_to_content(unit.leadin_blocks)
    leadin_plain = to_plain(leadin_content)

    stem_text = extract_stem(body_plain)
    allocation = plan_budget(body_plain, stem_text, leadin_plain)

    parts = build_parts(unit, allocation.body_allowance, table_ids)
    if not parts:
        return []
    total = len(parts)
    for part in parts:
        part.total = total
    _mark_duplicates(parts)

    breadcrumb_full = build_breadcrumb_full(node, doc_meta)
    khoan_id = unique_khoan_id(unit, so_hieu, id_registry)
    dieu = node if node.kind == "dieu" else node.find_ancestor("dieu")
    chuong = node.find_ancestor("chuong") or (node if node.kind == "chuong" else None)
    muc = node.find_ancestor("muc") or (node if node.kind == "muc" else None)
    phan = node.find_ancestor("phan")
    is_phu_luc = node.kind == "phu_luc_muc" or node.find_ancestor("phu_luc") is not None

    leadin_lines = sorted({line for block in unit.leadin_blocks for line in block.lines})

    chunks: list[Chunk] = []
    for part in parts:
        stem_embed = allocation.stem_embed if total > 1 else ""
        # Mảnh 1 luôn bắt đầu từ đầu đơn vị nên thân đã chứa sẵn stem. Nhân bản
        # thêm lần nữa làm vector lệch về mệnh đề chi phối và nhẹ đi phần liệt
        # kê — chính thứ phân biệt mảnh 1 với các mảnh sau. Cùng lập luận với
        # ca ``part_total == 1``, chỉ khác là áp cho mảnh đầu của đơn vị bị chia.
        if stem_embed and _stem_already_in_body(stem_embed, part.body_embed):
            stem_embed = ""
        leadin_embed = allocation.leadin_embed
        remaining = breadcrumb_budget(part.body_embed, stem_embed, leadin_embed)
        breadcrumb_embed, breadcrumb_level, truncated = build_breadcrumb_embed(
            node, doc_meta, remaining
        )

        chunk_id = khoan_id if unit.branch == 1 else f"{khoan_id}p{part.index}"
        embedding_text, breadcrumb_embed, stem_embed, leadin_embed = _finalize(
            breadcrumb_embed, stem_embed, leadin_embed, part.body_embed,
            node, doc_meta, chunk_id,
        )
        if not breadcrumb_embed:
            breadcrumb_level = LEVEL_NONE

        content = "\n\n".join(
            piece for piece in (breadcrumb_full, leadin_content, part.body_content)
            if piece.strip()
        )

        negation_terms = sorted(
            set(find_negations(part.body_content)) | set(find_negations(stem_embed))
        )
        info = part.table

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                so_hieu=so_hieu,
                content=content,
                embedding_text=embedding_text,
                token_count=count_tokens(embedding_text),
                breadcrumb_full=breadcrumb_full,
                breadcrumb_embed=breadcrumb_embed,
                stem=allocation.stem_embed or None,
                phan=phan.label if phan is not None else None,
                chuong=chuong.label if chuong is not None else None,
                muc=muc.label if muc is not None else None,
                dieu_so=dieu.ordinal if dieu is not None else None,
                dieu_tieu_de=dieu.title if dieu is not None else None,
                khoan_so=node.ordinal if node.kind in ("khoan", "phu_luc_muc") else None,
                khoan_range=unit.khoan_range,
                is_phu_luc=is_phu_luc,
                part_index=part.index,
                part_total=total,
                split_tier=part.tier,
                sibling_expand=total > 1,
                has_negation=bool(negation_terms),
                negation_terms=negation_terms,
                is_indexed=True,
                has_table=info is not None,
                table_id=info.table_id if info else None,
                table_raw=info.table_raw if info else None,
                table_summary=info.table_summary if info else None,
                table_search_text=info.table_search_text if info else None,
                amended_by=None,
                ngay_hieu_luc=doc_meta.get("ngay_hieu_luc"),
                chunker_version=CHUNKER_VERSION,
                parser_version=str(doc_meta.get("parser_version") or ""),
                khoan_id=khoan_id,
                unit_content=body_markdown,
                stem_full=stem_text,
                leadin_full=leadin_plain,
                leadin_used=leadin_embed,
                body_content=part.body_content,
                body_embed=part.body_embed,
                lines=part.lines,
                duplicated_lines=sorted(set(part.duplicated_lines) | set(leadin_lines)),
                leadin_level=allocation.leadin_level if leadin_plain else None,
                stem_level=allocation.stem_level if stem_text else None,
                breadcrumb_level=breadcrumb_level,
                breadcrumb_truncated=truncated,
                branch=unit.branch,
            )
        )
    return chunks


# ========================================================================
# VALIDATOR
# ========================================================================
#
# Pass 7 — validator.
#
# Mức ``error`` dừng pipeline: mọi mã lỗi ở đây đều là lỗi **im lặng** nếu không
# kiểm — chunk vẫn ghi được vào DB, vẫn embed được, chỉ là nói sai hoặc mất nội
# dung. Mức ``warning`` ghi log và chạy tiếp.
RE_WHITESPACE = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    """Chuẩn hoá khoảng trắng trước khi so nội dung."""
    return RE_WHITESPACE.sub(" ", text).strip()


def validate(chunks: list[Chunk], source_md: str) -> list[Issue]:
    """Trả toàn bộ ``Issue`` tìm được; không ném exception."""
    issues: list[Issue] = []

    def error(code: str, chunk_id: str | None, detail: str) -> None:
        issues.append(Issue(code=code, severity="error", chunk_id=chunk_id, detail=detail))

    def warn(code: str, chunk_id: str | None, detail: str) -> None:
        issues.append(Issue(code=code, severity="warning", chunk_id=chunk_id, detail=detail))

    seen_ids: set[str] = set()
    covered: set[int] = set()
    duplicated: set[int] = set()
    by_unit: dict[str, list[Chunk]] = defaultdict(list)
    tables: dict[str, set[str]] = defaultdict(set)

    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            error("duplicate_chunk_id", chunk.chunk_id, "chunk_id trùng")
        seen_ids.add(chunk.chunk_id)

        if chunk.token_count > BUDGET:
            error(
                "token_overflow", chunk.chunk_id,
                f"{chunk.token_count} token > {BUDGET}",
            )

        stripped = chunk.embedding_text.strip()
        if not stripped or stripped == chunk.breadcrumb_embed.strip():
            error("empty_embedding_text", chunk.chunk_id, "chỉ có breadcrumb hoặc rỗng")

        for marker in ("|", "#"):
            if marker in chunk.embedding_text:
                error(
                    "markdown_in_embedding", chunk.chunk_id,
                    f"embedding_text còn ký tự markdown {marker!r}",
                )

        if not chunk.chunk_id.startswith(chunk.so_hieu):
            error("chunk_id_prefix", chunk.chunk_id, "chunk_id không mở đầu bằng số hiệu")

        # NT-5: bản rút gọn không được đánh rơi marker phủ định.
        if chunk.stem and not set(find_negations(chunk.stem_full)) <= set(
            find_negations(chunk.stem)
        ):
            error("negation_stripped", chunk.chunk_id, "stem rút gọn mất marker phủ định")
        leadin_embed = _leadin_in(chunk)
        if leadin_embed and not set(find_negations(chunk.leadin_full)) <= set(
            find_negations(leadin_embed)
        ):
            error("negation_stripped", chunk.chunk_id, "câu dẫn rút gọn mất marker phủ định")

        # Stem chỉ được nhân bản khi thân KHÔNG có sẵn nó. Lặp hai lần trong
        # cùng embedding_text làm vector lệch và ăn mất ngân sách token.
        if chunk.stem and normalize_ws(chunk.embedding_text).count(
            normalize_ws(chunk.stem)
        ) > 1:
            error("stem_duplicated", chunk.chunk_id, "stem lặp hai lần trong embedding_text")

        if _condition_broken(chunk):
            error(
                "condition_split", chunk.chunk_id,
                "cắt giữa vế mở điều kiện và vế 'thì' của nó",
            )

        if chunk.has_table and chunk.table_summary is not None:
            tables[str(chunk.table_id)].add(chunk.table_raw or "")
            haystack = normalize_ws(
                f"{chunk.table_summary} {chunk.table_search_text or ''}"
            ).lower()
            for cell in _cells_of(chunk):
                if normalize_ws(cell).lower() and normalize_ws(cell).lower() not in haystack:
                    error(
                        "table_cell_unreachable", chunk.chunk_id,
                        f"ô {cell!r} không có trong summary lẫn search_text",
                    )

        by_unit[chunk.khoan_id].append(chunk)
        covered.update(chunk.lines)
        duplicated.update(chunk.duplicated_lines)

        # --- cảnh báo ---
        if count_tokens(chunk.body_embed) < 20:
            warn("very_short_chunk", chunk.chunk_id, "thân dưới 20 token")
        if chunk.leadin_level == "C":
            warn("leadin_dropped", chunk.chunk_id, "câu dẫn rơi bậc C")
        if chunk.stem_level == "C":
            warn("stem_dropped", chunk.chunk_id, "stem rơi bậc C")
        if chunk.breadcrumb_level == "none":
            warn("breadcrumb_omitted", chunk.chunk_id, "không còn chỗ cho breadcrumb")
        if chunk.breadcrumb_truncated:
            warn("breadcrumb_truncated", chunk.chunk_id, "tiêu đề Điều bị cắt")
        if chunk.part_total > 6:
            warn("deep_split", chunk.chunk_id, f"chia thành {chunk.part_total} mảnh")
        if chunk.split_tier == 5:
            warn("tier5_split", chunk.chunk_id, "không có ranh giới ngữ nghĩa sạch")
        if not chunk.is_indexed:
            warn("chunk_deindexed", chunk.chunk_id, "loại khỏi vector search")

    for table_id, raws in tables.items():
        if len(raws) > 1:
            error("table_split", table_id, "cùng table_id nhưng table_raw khác nhau")

    for khoan_id, group in by_unit.items():
        ordered = sorted(group, key=lambda chunk: chunk.part_index)
        joined = normalize_ws(" ".join(chunk.body_content for chunk in ordered))
        original = normalize_ws(ordered[0].unit_content)
        if joined != original:
            error(
                "content_mismatch", khoan_id,
                "ghép content các mảnh khác nội dung gốc",
            )

    blocks = parse_blocks(source_md)
    # Dòng nhân bản có chủ đích (câu dẫn của Điều, mảnh dùng chung atom) vẫn là
    # dòng đã được phủ — validator chỉ trừ chúng ra khi soát phần "thừa".
    missing = sorted(content_lines(blocks) - covered - duplicated)
    if missing:
        error(
            "coverage_gap", None,
            f"{len(missing)} dòng nội dung không thuộc chunk nào: {missing[:10]}",
        )

    return issues


def _condition_broken(chunk: Chunk) -> bool:
    """Vế mở điều kiện bị tách khỏi vế ``thì`` của CHÍNH NÓ vì cắt mảnh.

    Đối chiếu từng vế mở với bản gốc: rất nhiều câu luật viết *"Trường hợp X,
    Y phải Z"* không dùng ``thì`` bao giờ — đó là văn phong, không phải vết
    cắt. Chỉ báo lỗi khi ở bản gốc vế mở đó CÓ ``thì`` mà mảnh này đã mất.
    """
    part_keys = unclosed_openers(normalize_ws(chunk.body_embed))
    if not part_keys:
        return False
    origin_keys = unclosed_openers(normalize_ws(chunk.unit_content))
    for key in part_keys:
        if not key:
            continue
        if any(
            other.startswith(key) or key.startswith(other) for other in origin_keys
        ):
            continue
        return True
    return False


def _leadin_in(chunk: Chunk) -> str:
    """Phần câu dẫn thực sự đi vào ``embedding_text`` của chunk."""
    return chunk.leadin_used


def _cells_of(chunk: Chunk) -> list[str]:
    """Giá trị ô phải tra cứu được. Header so theo tên cột đã tách đơn vị."""

    header, rows = parse_table(chunk.table_raw or "")
    cells: list[str] = []
    for cell in header:
        name, unit = clean_header(cell)
        cells.extend(part for part in (name, unit) if part)
    for row in rows:
        cells.extend(cell.strip() for cell in row if cell.strip())
    return cells


def summarize(issues: list[Issue]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        counts[issue.code] = counts.get(issue.code, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def errors(issues: list[Issue]) -> list[Issue]:
    return [issue for issue in issues if issue.severity == "error"]


# ========================================================================
# LLAMA_ADAPTER
# ========================================================================
#
# Task 15 — cầu nối sang LlamaIndex mà không phá ngân sách token.
#
# Bẫy: LlamaIndex mặc định **nối metadata vào text trước khi embed**. Với
# ``get_content(MetadataMode.EMBED)``, mọi key không nằm trong
# ``excluded_embed_metadata_keys`` bị prepend vào chuỗi gửi đi, nên chuỗi thực tế
# dài hơn ``embedding_text`` đã tính và bị truncate ở 192 — âm thầm, không
# exception.
#
# Quyết định đã chốt: tính embedding **ngoài** LlamaIndex, từ cột
# ``embedding_text``, rồi gán ``node.embedding``.
def to_text_node(row: dict[str, Any]):
    from llama_index.core.schema import TextNode

    metadata = {
        "so_hieu": row["so_hieu"],
        "dieu_so": row["dieu_so"],
        "khoan_so": row["khoan_so"],
        "breadcrumb": row["breadcrumb_full"],
        "ngay_hieu_luc": str(row["ngay_hieu_luc"]),
        "has_negation": row["has_negation"],
        "sibling_expand": row["sibling_expand"],
        "table_id": row["table_id"],
    }
    return TextNode(
        id_=row["chunk_id"],
        text=row["content"],
        metadata=metadata,
        excluded_embed_metadata_keys=list(metadata.keys()),
        excluded_llm_metadata_keys=["table_id", "sibling_expand"],
    )


# ========================================================================
# WRITER
# ========================================================================
#
# Pass 8 (phần tệp) — ghi JSONL và sổ trạng thái để chạy lại biết SKIP.
STATE_FILE = ".chunker_state.json"


def markdown_hash(md_text: str) -> str:
    """Cùng công thức với ``loader.write_atomic`` để so được với DB."""
    return sha256(md_text.encode("utf-8")).hexdigest()


def write_jsonl(chunks: list[Chunk], path: Path) -> Path:
    """Một chunk một dòng, ``ensure_ascii=False`` để đọc được bằng mắt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_row(), ensure_ascii=False) + "\n")
    temporary.replace(path)
    return path


def write_issues(issues: list[Any], path: Path) -> Path:
    """Bảng warning lưu lại để rà thủ công."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for issue in issues:
            handle.write(
                json.dumps(
                    {
                        "code": issue.code,
                        "severity": issue.severity,
                        "chunk_id": issue.chunk_id,
                        "detail": issue.detail,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def load_state(out_dir: Path) -> dict[str, dict[str, str]]:
    """Sổ trạng thái cho chế độ không có PostgreSQL."""
    path = out_dir / STATE_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def save_state(out_dir: Path, state: dict[str, dict[str, str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / STATE_FILE).write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def state_is_current(state: dict[str, dict[str, str]], name: str, md_hash: str) -> bool:
    """SKIP khi markdown không đổi **và** ``chunker_version`` không đổi.

    Thiếu vế thứ hai thì sửa rule cắt xong chunk cũ tồn tại vĩnh viễn.
    """
    record = state.get(name)
    if not record:
        return False
    return (
        record.get("markdown_hash") == md_hash
        and record.get("chunker_version") == CHUNKER_VERSION
    )


# ========================================================================
# DB
# ========================================================================
#
# PostgreSQL + pgvector. Chỉ main process gọi module này.
#
# Bước chunking **không** tính embedding và **không** tạo index HNSW: build index
# trên bảng rỗng rồi insert dần chậm hơn nhiều lần so với build sau khi nạp xong.
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          TEXT PRIMARY KEY,
    document_id       BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    so_hieu           TEXT NOT NULL,

    content           TEXT NOT NULL,
    embedding_text    TEXT NOT NULL,
    token_count       INT  NOT NULL,
    embedding         VECTOR(768),
    search_vector     tsvector,

    breadcrumb_full   TEXT NOT NULL,
    breadcrumb_embed  TEXT NOT NULL DEFAULT '',
    stem              TEXT,

    phan              TEXT,
    chuong            TEXT,
    muc               TEXT,
    dieu_so           TEXT,
    dieu_tieu_de      TEXT,
    khoan_so          TEXT,
    khoan_range       TEXT,
    is_phu_luc        BOOLEAN NOT NULL DEFAULT false,

    part_index        INT NOT NULL DEFAULT 1,
    part_total        INT NOT NULL DEFAULT 1,
    split_tier        INT,
    sibling_expand    BOOLEAN NOT NULL DEFAULT false,

    has_negation      BOOLEAN NOT NULL DEFAULT false,
    negation_terms    TEXT[],
    is_indexed        BOOLEAN NOT NULL DEFAULT true,

    has_table         BOOLEAN NOT NULL DEFAULT false,
    table_id          TEXT,
    table_raw         TEXT,
    table_summary     TEXT,
    table_search_text TEXT,

    amended_by        TEXT,
    ngay_hieu_luc     DATE,
    chunker_version   TEXT NOT NULL,
    parser_version    TEXT NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT chunks_token_chk CHECK (token_count <= 192)
);

CREATE INDEX IF NOT EXISTS chunks_doc_idx    ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_sohieu_idx ON chunks (so_hieu, dieu_so, khoan_so);
CREATE INDEX IF NOT EXISTS chunks_khoan_idx  ON chunks (so_hieu, dieu_so, khoan_so, part_index);
CREATE INDEX IF NOT EXISTS chunks_table_idx  ON chunks (table_id) WHERE table_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS chunks_gin_idx    ON chunks USING gin (search_vector);
"""

# CHỈ chạy ở bước indexing, sau khi toàn bộ embedding đã được điền.
# Vector phải được CHUẨN HOÁ VỀ NORM 1 lúc ghi để dùng inner product; trộn lẫn
# vector chưa chuẩn hoá với vector_ip_ops cho kết quả xếp hạng sai mà không có
# lỗi nào báo.
HNSW_SQL = """
CREATE INDEX IF NOT EXISTS chunks_hnsw_idx ON chunks
  USING hnsw (embedding vector_ip_ops)
  WITH (m = 16, ef_construction = 64);
"""

# 12d — chunk bảng loại trừ table_raw khỏi tsvector, nếu không thì ký tự `|`
# và `---` lọt vào chỉ mục lexical.
SEARCH_VECTOR_SQL = """
UPDATE chunks SET search_vector =
  to_tsvector('simple', unaccent(
    CASE WHEN has_table
         THEN coalesce(table_summary,'') || ' ' || coalesce(table_search_text,'')
         ELSE content
    END
  ))
WHERE document_id = %s
"""

# 12e — hợp nhất hai kênh bằng RRF ngay trong SQL, không kéo về Python.
RRF_SEARCH_SQL = """
WITH dense AS (
    SELECT chunk_id,
           row_number() OVER (ORDER BY embedding <#> %(query_vector)s) AS rank
    FROM chunks
    WHERE is_indexed AND embedding IS NOT NULL
    ORDER BY embedding <#> %(query_vector)s
    LIMIT %(candidates)s
),
lexical AS (
    SELECT chunk_id,
           row_number() OVER (
               ORDER BY ts_rank_cd(search_vector, plainto_tsquery('simple',
                        unaccent(%(query_text)s))) DESC
           ) AS rank
    FROM chunks
    WHERE search_vector @@ plainto_tsquery('simple', unaccent(%(query_text)s))
    LIMIT %(candidates)s
)
SELECT c.chunk_id,
       c.content,
       c.breadcrumb_full,
       coalesce(1.0 / (60 + dense.rank), 0)
     + coalesce(1.0 / (60 + lexical.rank), 0) AS rrf_score
FROM dense
FULL OUTER JOIN lexical USING (chunk_id)
JOIN chunks c USING (chunk_id)
ORDER BY rrf_score DESC
LIMIT %(top_k)s
"""

_COLUMNS = (
    "chunk_id", "document_id", "so_hieu", "content", "embedding_text",
    "token_count", "breadcrumb_full", "breadcrumb_embed", "stem",
    "phan", "chuong", "muc", "dieu_so", "dieu_tieu_de", "khoan_so",
    "khoan_range", "is_phu_luc", "part_index", "part_total", "split_tier",
    "sibling_expand", "has_negation", "negation_terms", "is_indexed",
    "has_table", "table_id", "table_raw", "table_summary", "table_search_text",
    "amended_by", "ngay_hieu_luc", "chunker_version", "parser_version",
)


def load_dotenv(path: Path | None = None) -> None:
    """Nạp ``.env`` nếu biến chưa được set. Chỉ gọi từ CLI."""
    env_path = path if path is not None else project_root() / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "Chưa có DATABASE_URL. Chạy `docker compose up -d` rồi thử lại, "
            "hoặc dùng --no-db để chỉ ghi JSONL."
        )
    return url


def connect(dsn: str):
    import psycopg
    from psycopg.rows import dict_row

    return psycopg.connect(dsn, row_factory=dict_row)


def init_schema(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute(SCHEMA)
    conn.commit()


def find_document(conn, markdown_path: Path) -> dict[str, Any] | None:
    """Tìm bản ghi ``documents`` của một file markdown.

    So khớp theo đường dẫn đầy đủ trước, rồi tới tên file — loader có thể đã
    ghi đường dẫn tương đối hoặc tuyệt đối tuỳ lệnh chạy.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM documents WHERE markdown_path = %s",
            (str(markdown_path),),
        )
        row = cursor.fetchone()
        if row:
            return row
        cursor.execute(
            "SELECT * FROM documents WHERE markdown_path LIKE %s ORDER BY id LIMIT 1",
            (f"%{markdown_path.name}",),
        )
        return cursor.fetchone()


def chunks_are_current(conn, document_id: int) -> bool:
    """Đã có chunk và toàn bộ sinh bởi đúng phiên bản chunker hiện hành."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT count(*) AS total,
                   count(*) FILTER (WHERE chunker_version = %s) AS current
            FROM chunks WHERE document_id = %s
            """,
            (CHUNKER_VERSION, document_id),
        )
        row = cursor.fetchone() or {}
    return bool(row.get("total")) and row["total"] == row["current"]


def replace_document_chunks(conn, document_id: int, chunks: list[Chunk]) -> int:
    """Xoá sạch chunk cũ rồi upsert chunk mới trong một transaction.

    Xoá trước là bắt buộc: khi ``part_total`` giảm giữa hai lần chạy, các mảnh
    thừa của lần trước sẽ mồ côi và vẫn nằm trong index.
    """
    placeholders = ", ".join(["%s"] * len(_COLUMNS))
    updates = ", ".join(
        f"{name} = EXCLUDED.{name}" for name in _COLUMNS if name != "chunk_id"
    )
    statement = (
        f"INSERT INTO chunks ({', '.join(_COLUMNS)}) VALUES ({placeholders}) "
        f"ON CONFLICT (chunk_id) DO UPDATE SET {updates}"
    )
    rows = []
    for chunk in chunks:
        record = chunk.to_row()
        record["document_id"] = document_id
        rows.append([record[name] for name in _COLUMNS])

    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        cursor.executemany(statement, rows)
        cursor.execute(SEARCH_VECTOR_SQL, (document_id,))
    conn.commit()
    return len(rows)


def count_chunks(conn) -> int:
    with conn.cursor() as cursor:
        cursor.execute("SELECT count(*) AS total FROM chunks")
        row = cursor.fetchone() or {"total": 0}
    return int(row["total"])


# ========================================================================
# PIPELINE
# ========================================================================
#
# Ghép 8 pass thành một hàm thuần cho mỗi tài liệu, cộng lớp điều phối.
#
# NT-8: ``chunk_markdown`` không I/O, không mạng, không LLM. Cùng đầu vào cho
# cùng đầu ra byte-for-byte. Toàn bộ I/O nằm ở nửa dưới của module.
def _table_ids(blocks: list[MdBlock], so_hieu: str) -> dict[int, str]:
    """``table_id`` sinh tất định theo thứ tự bảng trong tài liệu, đếm từ 1.

    Sinh ở pha sau sẽ buộc phải reindex toàn bộ corpus.
    """
    ids: dict[int, str] = {}
    order = 0
    for block in blocks:
        if block.kind == "table":
            order += 1
            ids[id(block)] = f"{so_hieu}#tbl{order}"
    return ids


def chunk_markdown(md_text: str, doc_meta: dict[str, Any] | None = None) -> list[Chunk]:
    """Markdown có cấu trúc → danh sách chunk. Hàm thuần."""
    blocks = parse_blocks(md_text)
    meta = dict(parse_front_matter(blocks))
    if doc_meta:
        meta.update(doc_meta)

    tree = build_tree(blocks)
    units = resolve_chunk_units(tree)
    table_ids = _table_ids(blocks, str(meta.get("so_hieu") or "UNKNOWN"))

    chunks: list[Chunk] = []
    id_registry: dict[str, int] = {}
    for unit in units:
        chunks.extend(build_chunks(unit, meta, table_ids, id_registry))
    return chunks


def body_signature(chunk: Chunk) -> str:
    """Nội dung thân đã chuẩn hoá khoảng trắng, bỏ breadcrumb và câu dẫn."""
    return normalize_ws(chunk.body_content).lower()


def mark_deindexed(chunks: list[Chunk]) -> int:
    """Task 11 — loại chunk lặp lại và ngắn khỏi vector search.

    Hai điều kiện phải đúng **cùng lúc**. Chỉ dựa vào điều kiện trùng lặp sẽ
    loại nhầm những Khoản trùng nhau nhưng có nội dung thật, ví dụ *"Hồ sơ đề
    nghị hưởng bảo hiểm xã hội một lần bao gồm: a) Sổ bảo hiểm xã hội; ..."*
    xuất hiện ở hai Điều khác nhau — đó là quy định thật, phải giữ trong index.
    """
    counts = Counter(body_signature(chunk) for chunk in chunks)
    deindexed = 0
    for chunk in chunks:
        signature = body_signature(chunk)
        if not signature:
            continue
        short = count_tokens(chunk.body_content) < BOILERPLATE_MIN_TOKENS
        repeated = counts[signature] >= DUPLICATE_INDEX_THRESHOLD
        if short and repeated:
            chunk.is_indexed = False
            deindexed += 1
    return deindexed


def unit_stats(md_text: str) -> dict[int, int]:
    """Phân bố theo nhánh của Pass 3 — dùng cho báo cáo nghiệm thu."""
    blocks = parse_blocks(md_text)
    return branch_distribution(resolve_chunk_units(build_tree(blocks)))


# ==========================================================================
# ĐIỀU PHỐI (có I/O — tách hẳn khỏi phần thuần ở trên)
# ==========================================================================


@dataclass(slots=True)
class DocResult:
    """Kết quả chunking của một tài liệu, trước khi quyết định ghi hay SKIP."""

    path: Path
    markdown_hash: str
    chunks: list[Chunk]
    issues: list[Issue] = field(default_factory=list)
    skipped: bool = False
    reason: str = ""


def build_all(paths: list[Path]) -> list[DocResult]:
    """Chunk toàn bộ tài liệu rồi mới đánh dấu ``is_indexed``.

    Task 11 đếm nội dung trùng trên **toàn corpus**, nên không thể xử lý xong
    từng file một: phải có đủ chunk của mọi file trong tay trước đã. Kể cả tài
    liệu sắp bị SKIP cũng phải chunk, nếu không thì số đếm trùng bị thiếu và
    quyết định ``is_indexed`` của các file khác sai theo.
    """
    results: list[DocResult] = []
    for path in paths:
        md_text = path.read_text(encoding="utf-8")
        results.append(
            DocResult(
                path=path,
                markdown_hash=markdown_hash(md_text),
                chunks=chunk_markdown(md_text),
            )
        )

    mark_deindexed([chunk for result in results for chunk in result.chunks])

    for result in results:
        result.issues = validate(result.chunks, result.path.read_text(encoding="utf-8"))
    return results


def report(results: list[DocResult]) -> str:
    """Năm bảng nghiệm thu mà spec đòi in ở cuối lượt chạy."""
    chunks = [chunk for result in results for chunk in result.chunks]
    if not chunks:
        return "Không có chunk nào."

    lines: list[str] = []

    def table(title: str, counts: dict[Any, int]) -> None:
        total = sum(counts.values()) or 1
        lines.append(f"\n{title}")
        for key, value in counts.items():
            lines.append(f"  {str(key):<24} {value:>6}  {value / total:6.1%}")

    units: Counter[int] = Counter()
    for result in results:
        units.update(unit_stats(result.path.read_text(encoding="utf-8")))
    table("Task 4 — đơn vị chunk theo nhánh", dict(sorted(units.items())))

    stem_levels = Counter(chunk.stem_level or "-" for chunk in chunks)
    leadin_levels = Counter(chunk.leadin_level or "-" for chunk in chunks)
    table("Task 5 — bậc rút gọn của stem", dict(sorted(stem_levels.items())))
    table("Task 5 — bậc rút gọn của câu dẫn", dict(sorted(leadin_levels.items())))

    order = {"full": 0, "dieu_ten": 1, "dieu": 2, "none": 3}
    breadcrumbs = Counter(chunk.breadcrumb_level for chunk in chunks)
    table(
        "Task 7 — mức breadcrumb",
        dict(sorted(breadcrumbs.items(), key=lambda item: order.get(item[0], 9))),
    )

    tiers = Counter(
        chunk.split_tier if chunk.split_tier is not None else "không chia"
        for chunk in chunks
    )
    table(
        "Task 8 — mảnh theo tầng cắt",
        dict(sorted(tiers.items(), key=lambda item: str(item[0]))),
    )

    counts = sorted(chunk.token_count for chunk in chunks)
    buckets: Counter[str] = Counter()
    for value in counts:
        low = min(value // 20 * 20, 180)
        buckets[f"{low:>3}–{low + 19:>3}"] += 1
    table("Task 9 — histogram token của embedding_text", dict(sorted(buckets.items())))
    middle = counts[len(counts) // 2]
    lines.append(
        f"\n  trung vị {middle} · p90 {counts[int(0.9 * len(counts))]} "
        f"· max {counts[-1]} · trần {BUDGET}"
    )
    lines.append(f"  tổng {len(chunks)} chunk, "
                 f"{sum(1 for chunk in chunks if not chunk.is_indexed)} chunk ngoài index")
    return "\n".join(lines)


def run(
    md_dir: Path,
    out_dir: Path,
    use_db: bool = True,
    dsn: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    issues_path: Path | None = None,
) -> int:
    """Chạy trọn bước chunking. Trả mã thoát cho CLI."""
    paths = sorted(path for path in md_dir.glob("*.md") if path.is_file())
    if not paths:
        print(f"Không tìm thấy .md nào trong {md_dir}")
        return 1

    print(f"Chunker {CHUNKER_VERSION} · {len(paths)} tài liệu · trần {BUDGET} token")
    check_model_max_length()

    started = time.monotonic()
    results = build_all(paths)
    elapsed = time.monotonic() - started

    all_issues = [issue for result in results for issue in result.issues]
    blocking = errors(all_issues)
    print(report(results))
    print(f"\nXong {elapsed:.1f}s · issue: {summarize(all_issues)}")

    if issues_path is not None:
        write_issues(all_issues, issues_path)
        print(f"Ghi issue: {issues_path}")

    if blocking:
        print(f"\nDỪNG: {len(blocking)} lỗi mức error, không ghi gì cả.")
        for issue in blocking[:20]:
            print(f"  {issue}")
        return 2

    connection = None
    if use_db:
        connection = connect(dsn or get_database_url())
        init_schema(connection)

    state = load_state(out_dir)
    written = skipped = 0
    try:
        for result in results:
            target = out_dir / f"{result.path.stem}.jsonl"
            document = find_document(connection, result.path) if connection else None
            skip, reason = _decide(
                result, target, state, connection, document, force
            )
            if skip:
                result.skipped, result.reason = True, reason
                skipped += 1
                print(f"SKIP    {result.path.name}  ({reason})")
                continue

            if dry_run:
                print(f"PROCESS {result.path.name}  ({reason})")
                continue

            write_jsonl(result.chunks, target)
            if connection is not None and document is not None:
                replace_document_chunks(connection, document["id"], result.chunks)
            state[result.path.name] = {
                "markdown_hash": result.markdown_hash,
                "chunker_version": CHUNKER_VERSION,
                "chunks": str(len(result.chunks)),
            }
            written += 1
            note = "JSONL + DB" if document is not None else "JSONL"
            print(f"OK      {result.path.name}  {len(result.chunks):>5} chunk  [{note}]")

        if not dry_run:
            save_state(out_dir, state)
        if connection is not None:
            print(f"\nTổng chunk trong DB: {count_chunks(connection)}")
    finally:
        if connection is not None:
            connection.close()

    print(f"\nGhi {written} tài liệu, SKIP {skipped}.")
    print("Bước này KHÔNG tính embedding và KHÔNG tạo index HNSW.")
    return 0


def _decide(
    result: DocResult,
    target: Path,
    state: dict[str, dict[str, str]],
    connection: Any,
    document: dict[str, Any] | None,
    force: bool,
) -> tuple[bool, str]:
    """SKIP khi markdown và ``chunker_version`` đều không đổi."""
    if force:
        return False, "force"
    if not target.is_file():
        return False, "chưa có JSONL"
    if connection is not None:
        if document is None:
            return False, "chưa có bản ghi documents"
        if document.get("markdown_hash") != result.markdown_hash:
            return False, "markdown đổi"
        if not chunks_are_current(connection, document["id"]):
            return False, "chunker_version đổi"
        return True, "DB đã khớp"
    if state_is_current(state, result.path.name, result.markdown_hash):
        return True, "sổ trạng thái đã khớp"
    return False, "markdown hoặc chunker_version đổi"


# ========================================================================
# CLI
# ========================================================================
#
# Giao diện dòng lệnh cho bước chunking.
#
#     python -m rag.ingestion.pre_chunker.cli --md-dir data/markdown --out-dir data/chunks
#
# Bước này **không** tính embedding và **không** tạo index HNSW: build index trên
# bảng rỗng rồi insert dần chậm hơn nhiều lần so với build sau khi nạp xong.
def _resolve(value: str, default: Path) -> Path:
    """Đường dẫn tương đối luôn tính từ gốc dự án, không từ thư mục hiện tại."""
    path = Path(value) if value else default
    return path if path.is_absolute() else project_root() / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pre_chunker",
        description=(
            "Markdown có cấu trúc → chunk sẵn sàng embed (JSONL + PostgreSQL). "
            "KHÔNG tính embedding, KHÔNG tạo index HNSW."
        ),
    )
    parser.add_argument("--md-dir", default=str(MD_DIR), help="thư mục .md đầu vào")
    parser.add_argument("--out-dir", default=str(CHUNK_DIR), help="thư mục .jsonl đầu ra")
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="chỉ ghi JSONL, dùng sổ trạng thái thay cho PostgreSQL",
    )
    parser.add_argument("--dsn", default=None, help="DSN PostgreSQL, mặc định DATABASE_URL")
    parser.add_argument(
        "--force", action="store_true", help="bỏ qua SKIP, chunk lại toàn bộ"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="chỉ in quyết định PROCESS/SKIP"
    )
    parser.add_argument(
        "--issues",
        default=None,
        help="file JSONL ghi warning, mặc định <out-dir>/_issues.jsonl",
    )
    parser.add_argument("--version", action="version", version=CHUNKER_VERSION)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    load_dotenv()

    md_dir = _resolve(arguments.md_dir, MD_DIR)
    out_dir = _resolve(arguments.out_dir, CHUNK_DIR)
    issues_path = (
        _resolve(arguments.issues, out_dir / "_issues.jsonl")
        if arguments.issues
        else out_dir / "_issues.jsonl"
    )

    return run(
        md_dir=md_dir,
        out_dir=out_dir,
        use_db=not arguments.no_db,
        dsn=arguments.dsn,
        force=arguments.force,
        dry_run=arguments.dry_run,
        issues_path=issues_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
