"""Chuyển văn bản pháp luật .docx sang Markdown có cấu trúc.

Cách dùng:
    python src/rag/ingestion/loader.py (--raw-dir) data/raw -(-out-dir) data/markdown
"""

from __future__ import annotations

import argparse
import collections
import functools
import math
import os
import psycopg
import re
import traceback
import unicodedata

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from hashlib import sha256
from html import escape
from pathlib import Path
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from tqdm import tqdm
from typing import Any
from typing import Literal


# ==========================================================================
# CẤU HÌNH
# ==========================================================================

# Tăng thủ công mỗi khi rule nhận diện heading thay đổi. Bump trong cùng commit
# với thay đổi golden file: hai việc đó là một sự kiện.
PARSER_VERSION = "1.0.0"

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUT_DIR = Path("data/markdown")

MAX_WORKERS = 2

# Ngân sách cho MỘT lượt worker, không phải cho cả batch. pipeline nhân số này
# với số lượt để ra timeout tổng cho as_completed.
WORKER_TIMEOUT_SEC = 120

# Dòng heading dài hơn ngưỡng này bị cảnh báo suspicious_heading_length.
# Heading dài nhất có thật trong corpus là Mục 3 của Bộ luật lao động (~168).
HEADING_MAX_LEN = 200

# Trong Phụ lục ghi "##### 28. Thành phố Hồ Chí Minh" thay cho "##### Khoản 28",
# để tên tỉnh/thành nằm trong breadcrumb của mọi chunk con.
PHU_LUC_HEADING_WITH_TITLE = True

# Chốt chặn cho tùy chọn trên: mục Phụ lục dài hơn ngưỡng này quay về dạng
# "##### Khoản N" để heading không phình.
PHU_LUC_HEADING_MAX_CHARS = 120

# Chú thích dài hơn ngưỡng này chỉ inline đoạn đầu, phần còn lại dời xuống cuối
# file dưới dòng in đậm **[n]** (không phải heading).
FOOTNOTE_INLINE_MAX_CHARS = 1200

# Giữ nguyên văn vùng dẫn nhập trong body. Với văn bản hợp nhất, danh sách luật
# sửa đổi kèm ngày hiệu lực là thông tin pháp lý thực và là mục tiêu truy hồi.
KEEP_PREAMBLE = True


@functools.cache
def project_root() -> Path:
    """Tìm thư mục gốc dự án bằng cách đi ngược lên tới ``pyproject.toml``.

    Không dùng ``parents[N]`` vì độ sâu của file thay đổi khi gộp ở Task 12.
    """
    for candidate in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def load_dotenv(path: Path | None = None) -> None:
    """Nạp file ``.env`` vào ``os.environ`` nếu biến chưa được set sẵn.

    Parser tối giản dạng ``KEY=VALUE`` để không phải thêm dependency. Chỉ được
    gọi từ CLI/main process, không bao giờ ở cấp module.
    """
    env_path = path if path is not None else project_root() / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        # Biến môi trường thật luôn thắng file .env.
        os.environ.setdefault(key, value)


def get_database_url() -> str:
    """Đọc ``DATABASE_URL``. Chỉ gọi từ main process."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "Chưa có DATABASE_URL. Chạy `cp .env.example .env` rồi "
            "`docker compose up -d`, hoặc export DATABASE_URL thủ công."
        )
    return url


# ==========================================================================
# REGEX NHẬN DIỆN CẤU TRÚC
# ==========================================================================

ROMAN = r"[IVXLCDM]+"

# "Phần thứ nhất" dùng số thứ tự tiếng Việt, không phải chữ số La Mã. Corpus
# hiện tại không có Phần nào, nhưng Bộ luật Dân sự thì có.
ORDINAL_WORD = r"nhất|hai|ba|tư|bốn|năm|sáu|bảy|tám|chín|mười"

RE_PHU_LUC = re.compile(r"^PHỤ\s+LỤC\b", re.IGNORECASE)
RE_PHAN = re.compile(
    rf"^Phần\s+(?:thứ\s+)?({ROMAN}|\d+|{ORDINAL_WORD})\b", re.IGNORECASE
)
RE_CHUONG = re.compile(rf"^Chương\s+({ROMAN}|\d+)\b", re.IGNORECASE)
RE_MUC = re.compile(r"^Mục\s+(\d+[a-zđ]?)\b", re.IGNORECASE)

# Hỗ trợ số hiệu chữ: Điều 41a, Điều 7a, Điều 48b — phổ biến ở văn bản hợp nhất
# khi bổ sung điều mới vào giữa. Bắt buộc có dấu chấm sau số hiệu, nhờ vậy các
# dòng "Điều 2 của Luật số 46/2014/QH13 ... quy định như sau:" trong vùng chú
# thích không bị nhận nhầm thành heading.
RE_DIEU = re.compile(r"^Điều\s+(\d+[a-zđ]?)\s*\.\s*(.*)$")

# Khoản = bắt đầu bằng số. Không kèm điều kiện ngữ cảnh.
# Nhánh (?:\[\d+\])? giữ lại làm lớp phòng vệ thứ hai; lớp phòng vệ thật là
# bước gỡ marker ở strip_all, vì corpus có dạng "3.3[3]" mà nhánh này
# không cứu được.
RE_KHOAN = re.compile(r"^(\d+[a-zđ]?)\s*\.\s*(?:\[\d+\])?\s+(.*)$")

# Điểm KHÔNG bao giờ tạo heading. Chỉ dùng cho QC và nhận biết ngữ cảnh.
# Hai hình thức: chữ cái + ")" trong thân văn bản, dấu "-" trong Phụ lục.
RE_DIEM = re.compile(r"^(?:([a-zđư])\)|-)\s+(.*)$")

# Bảng chữ cái tiếng Việt dùng cho điểm (không có f, j, w, z).
VIETNAMESE_POINT_LETTERS = "a b c d đ e g h i k l m n o p q r s t u ư v x y".split()

# --- Chú thích sửa đổi ---------------------------------------------------

# Marker trong thân văn bản. Chữ số lặp có thể nằm hai bên: "3.3[3]", "[4]4".
# Chỉ nuốt pre/post khi nó BẰNG num, nếu không "10.[15]" sẽ mất số khoản 10.
RE_MARKER = re.compile(r"(?:(?P<pre>\d{1,3}))?\[(?P<num>\d{1,3})\](?:(?P<post>\d{1,3}))?")

# Dòng định nghĩa chú thích có 4 dạng trong corpus:
#   "[1] ..."   "3[3] ..."   "[4]4 ..."   và dạng trần "2 Điểm này được sửa..."
RE_FN_DEF_BRACKET = re.compile(r"^(?:(\d{1,3}))?\[(\d{1,3})\](?:(\d{1,3}))?\s*(.*)$")

# Dạng trần: KHÔNG có dấu chấm sau số. Đó là điều phân biệt nó với RE_KHOAN,
# nên dòng "1. Luật này có hiệu lực..." được trích dẫn bên trong chú thích
# không thể bị nhận nhầm thành một định nghĩa chú thích mới.
RE_FN_DEF_BARE = re.compile(r"^(\d{1,3})\s+(\S.*)$")

RE_FOOTNOTE_MARKER = re.compile(r"\[(\d+)\]")

# Sau khi gỡ marker dính liền, "a)Thành lập" và "1.Cá nhân" mất khoảng trắng.
RE_MISSING_SPACE = re.compile(r"^([a-zđư]\)|\d{1,3}\.)(?=\S)")

# --- Bảng ----------------------------------------------------------------

# Nhận diện bảng theo NỘI DUNG, không theo vị trí: bảng chữ ký của Nghị định
# 293 nằm ở giữa văn bản, toàn bộ Phụ lục nằm sau nó. Hai trong sáu bảng chữ ký
# lại có ô đầu rỗng nên riêng "Nơi nhận:" là không đủ.
# Không neo "^": các pattern này dò trên TOÀN BỘ bảng đã render, mà chuỗi đó bắt
# đầu bằng "<table>" hoặc "| ". Neo đầu chuỗi sẽ không bao giờ khớp.
RE_SIGNATURE_CELL = re.compile(
    r"Nơi\s+nhận:"
    r"|XÁC\s+THỰC\s+VĂN\s+BẢN\s+HỢP\s+NHẤT"
    r"|CHỦ\s+NHIỆM"
    r"|TM\.\s"
    r"|KT\.\s"
    r"|THỦ\s+TƯỚNG",
    re.IGNORECASE,
)
RE_QUOC_HIEU_CELL = re.compile(
    r"CỘNG\s+HÒA\s+XÃ\s+HỘI\s+CHỦ\s+NGHĨA\s+VIỆT\s+NAM", re.IGNORECASE
)
RE_ATTACHMENT_TABLE = re.compile(
    r"FILE\s+ĐƯỢC\s+ĐÍNH\s+KÈM\s+THEO\s+VĂN\s+BẢN", re.IGNORECASE
)

# --- Front matter --------------------------------------------------------

RE_SO_HIEU = re.compile(r"Số:\s*([^\s|<]+)")

# "Hà Nội, ngày 20 tháng 5 năm 2026" và "Hà Nội ngày 10 tháng 11 năm 2025"
# (Nghị định 293 thiếu dấu phẩy).
RE_NGAY = re.compile(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE)

RE_HIEU_LUC = re.compile(
    r"có\s+hiệu\s+lực(?:\s+thi\s+hành)?(?:\s+kể)?\s+từ\s+ngày\s+"
    r"(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
    re.IGNORECASE,
)

DOC_TYPE_ALT = r"Bộ luật|Luật|Nghị định|Nghị quyết|Thông tư|Quyết định"

RE_BAN_HANH = re.compile(rf"\bban\s+hành\s+((?:{DOC_TYPE_ALT})\s+.+?)\s*\.?\s*$", re.IGNORECASE)
RE_TEN_SO = re.compile(rf"^((?:{DOC_TYPE_ALT})\s+.+?)\s+số\s+\d+/", re.IGNORECASE)
RE_DOC_TYPE = re.compile(
    r"^(BỘ\s+LUẬT|LUẬT|NGHỊ\s+ĐỊNH|NGHỊ\s+QUYẾT|QUYẾT\s+ĐỊNH|THÔNG\s+TƯ)$"
)

# Tiêu đề Phụ lục của Nghị định 293 dính cả phần "(Kèm theo ...)" dài ~90 ký tự.
RE_KEM_THEO = re.compile(r"\s*(\(\s*Kèm\s+theo\b.*)$", re.IGNORECASE | re.DOTALL)


def is_structural(text: str) -> bool:
    """Dòng có phải nhãn cấu trúc (Phụ lục/Phần/Chương/Mục/Điều) hay không."""
    return any(
        pattern.match(text)
        for pattern in (RE_PHU_LUC, RE_PHAN, RE_CHUONG, RE_MUC, RE_DIEU)
    )


def sort_key(number: str) -> tuple[int, str]:
    """Khóa so sánh số hiệu dạng "48a".

    ``int()`` trần sẽ báo giả trên chuỗi hợp lệ 48 -> 48a -> 48b -> 49 của Luật
    Bảo hiểm y tế, và 1 -> 1a -> 1b -> 1c của Bộ luật lao động.
    """
    digits = "".join(character for character in number if character.isdigit())
    suffix = number[len(digits):]
    return (int(digits) if digits else 0, suffix)


# ==========================================================================
# KIỂU DỮ LIỆU
# ==========================================================================

@dataclass(frozen=True, slots=True)
class Block:
    """Một phần tử nội dung lấy từ DOCX, giữ đúng thứ tự xuất hiện."""

    kind: Literal["paragraph", "table"]
    text: str  # với table: markdown (hoặc HTML) đã render sẵn
    style: str | None = None  # tên style gốc, chỉ là tín hiệu phụ
    is_bold: bool = False


@dataclass(frozen=True, slots=True)
class QcWarning:
    """Cảnh báo QC. Không bao giờ làm fail file.

    Tên có tiền tố ``Qc`` là cố ý: ``Warning`` trần sẽ che builtin exception,
    vô hại trong package nhưng nguy hiểm thật trong file gộp ở Task 12.
    """

    code: str
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class Footnote:
    """Một chú thích sửa đổi ở cuối văn bản."""

    number: int
    paragraphs: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


@dataclass(frozen=True, slots=True)
class FrontMatter:
    """Metadata cấp văn bản, render thành YAML ở đầu file .md."""

    so_hieu: str | None
    loai_van_ban: str | None
    ten_van_ban: str | None
    co_quan_ban_hanh: str | None
    ngay_ban_hanh: str | None
    ngay_hieu_luc: str | None
    is_van_ban_hop_nhat: bool
    is_phu_luc: bool
    source_path: str | None
    parser_version: str

    def to_yaml(self) -> str:
        """Render YAML bằng tay để golden file ổn định theo byte.

        Không dùng ``yaml.safe_dump``: nó sắp lại khóa và escape tiếng Việt,
        làm golden nhiễu mỗi lần nâng thư viện.
        """
        lines = ["---"]
        for key, value in (
            ("so_hieu", self.so_hieu),
            ("loai_van_ban", self.loai_van_ban),
            ("ten_van_ban", self.ten_van_ban),
            ("co_quan_ban_hanh", self.co_quan_ban_hanh),
            ("ngay_ban_hanh", self.ngay_ban_hanh),
            ("ngay_hieu_luc", self.ngay_hieu_luc),
            ("is_van_ban_hop_nhat", self.is_van_ban_hop_nhat),
            ("is_phu_luc", self.is_phu_luc),
            ("source_path", self.source_path),
            ("parser_version", self.parser_version),
        ):
            lines.append(f"{key}: {_yaml_scalar(value)}")
        lines.append("---")
        return "\n".join(lines)


def _yaml_scalar(value: str | bool | None) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


@dataclass(frozen=True, slots=True)
class PendingDocument:
    """Một DOCX đã được main process chọn để gửi sang worker."""

    source_path: Path
    source_hash: str
    output_path: Path


# ==========================================================================
# CHÚ THÍCH SỬA ĐỔI
# ==========================================================================

def find_region_start(blocks: list[Block]) -> tuple[int | None, list[QcWarning]]:
    """Tìm chỉ số block bắt đầu vùng chú thích ở cuối văn bản.

    Mỗi văn bản có chú thích chỉ có ĐÚNG MỘT paragraph mở đầu bằng ``[1]``
    (chấp nhận chữ số lặp phía trước), và nó luôn nằm ngay sau bảng chữ ký.
    """
    warnings: list[QcWarning] = []
    hits: list[int] = []

    for index, block in enumerate(blocks):
        if block.kind != "paragraph":
            continue
        match = RE_FN_DEF_BRACKET.match(block.text)
        if match is None or match.group(2) != "1":
            continue
        # Loại trường hợp "7[1]" — chữ số lặp phải bằng số chú thích.
        if match.group(1) is not None and match.group(1) != "1":
            continue
        hits.append(index)

    if not hits:
        return None, warnings

    if len(hits) > 1:
        warnings.append(
            QcWarning(
                "ambiguous_footnote_region",
                f"tìm thấy {len(hits)} dòng mở đầu [1], lấy dòng cuối",
            )
        )

    return hits[-1], warnings


def parse_region(blocks: list[Block]) -> tuple[dict[int, Footnote], list[QcWarning]]:
    """Phân tích vùng chú thích thành map ``số hiệu -> nội dung``.

    Dùng bộ đếm kỳ vọng vì corpus có bốn dạng dòng định nghĩa, trong đó một
    dạng KHÔNG có ngoặc ("2 Điểm này được sửa đổi..." ở Bộ luật lao động).
    Parser ``^\\[(\\d+)\\]`` thuần sẽ mất hẳn chú thích đó.

    Dạng trần chỉ được chấp nhận khi số bằng đúng số kỳ vọng kế tiếp. Nhờ vậy
    nội dung trích dẫn bên trong chú thích không bị nhận nhầm: các dòng đó có
    dấu chấm ("1. Luật này có hiệu lực...") nên RE_FN_DEF_BARE không khớp, và
    kể cả khớp thì bộ đếm cũng đã vượt qua số đó.
    """
    warnings: list[QcWarning] = []
    collected: dict[int, list[str]] = {}
    expected = 1
    current: list[str] | None = None

    for block in blocks:
        if block.kind != "paragraph":
            if current is not None:
                current.append(block.text)
            continue

        text = block.text

        match = RE_FN_DEF_BRACKET.match(text)
        if match is not None and match.group(2):
            number = int(match.group(2))
            if number != expected:
                warnings.append(
                    QcWarning(
                        "footnote_number_gap",
                        f"kỳ vọng [{expected}], gặp [{number}]",
                    )
                )
            rest = match.group(4).strip()
            current = collected.setdefault(number, [])
            if rest:
                current.append(rest)
            expected = number + 1
            continue

        bare = RE_FN_DEF_BARE.match(text)
        if bare is not None and int(bare.group(1)) == expected:
            number = expected
            current = collected.setdefault(number, [])
            current.append(bare.group(2).strip())
            expected = number + 1
            continue

        if current is not None:
            current.append(text)
        else:
            # Rác đứng trước chú thích [1]; giữ lại để không mất nội dung im lặng.
            collected.setdefault(0, []).append(text)

    footnotes = {
        number: Footnote(number=number, paragraphs=tuple(paragraphs))
        for number, paragraphs in collected.items()
        if number > 0
    }
    return footnotes, warnings


def strip_markers(text: str) -> tuple[str, list[int]]:
    """Gỡ mọi marker ``[n]`` khỏi một dòng, trả về dòng sạch và số hiệu đã gặp.

    Marker trong corpus có chữ số lặp ở một trong hai bên: "3.3[3]", "1.4[4]",
    "[4]4", "a)2[2]". Chỉ nuốt chữ số lặp khi nó BẰNG số chú thích — nếu không,
    "10.[15]" sẽ mất số khoản 10.
    """
    found: list[int] = []

    def _replace(match) -> str:
        number = match.group("num")
        pre = match.group("pre")
        post = match.group("post")
        found.append(int(number))
        keep_pre = pre if pre and pre != number else ""
        keep_post = post if post and post != number else ""
        return keep_pre + keep_post

    stripped = RE_MARKER.sub(_replace, text)
    # "a)Thành lập" -> "a) Thành lập"; "1.Cá nhân" -> "1. Cá nhân"
    stripped = RE_MISSING_SPACE.sub(r"\1 ", stripped)
    # Marker giữa câu để lại khoảng trắng thừa trước dấu câu.
    stripped = stripped.replace(" ,", ",").replace(" ;", ";").replace(" .", ".")
    return " ".join(stripped.split()), found


def strip_all(
    blocks: list[Block],
) -> tuple[list[Block], dict[int, list[int]], list[QcWarning]]:
    """Gỡ marker khỏi toàn bộ body, ghi nhớ vị trí theo chỉ số block.

    ``refs`` khóa theo chỉ số block là đủ mịn: đích chèn blockquote theo spec là
    "cuối khối tương ứng", không phải offset ký tự. Nhờ vậy marker nằm trên tiêu
    đề Điều, trên dòng dẫn nhập hay trên Khoản đều đi cùng một nhánh code.
    """
    warnings: list[QcWarning] = []
    cleaned: list[Block] = []
    refs: dict[int, list[int]] = {}

    for index, block in enumerate(blocks):
        if block.kind == "table":
            # Không bao giờ gỡ marker trong bảng. Corpus hiện không có ca nào —
            # đây là dây bẫy, không phải feature.
            if RE_FOOTNOTE_MARKER.search(block.text):
                warnings.append(
                    QcWarning("footnote_marker_in_table", f"block {index}")
                )
            cleaned.append(block)
            continue

        text, numbers = strip_markers(block.text)
        if numbers:
            refs[index] = numbers
        cleaned.append(
            Block(kind="paragraph", text=text, style=block.style, is_bold=block.is_bold)
        )

    return cleaned, refs, warnings


def render_blockquote(
    footnote: Footnote, *, inline_max: int
) -> tuple[str, str | None]:
    """Render chú thích thành blockquote.

    Trả về ``(inline, deferred)``. Chú thích ngắn thì chèn nguyên văn tại chỗ.
    Chú thích dài (BHYT [114], BLLĐ [6], TNCN [3] trích nguyên cả Điều của luật
    khác) chỉ inline đoạn đầu, phần còn lại dời xuống cuối file để không làm
    phồng khối Khoản cha thêm nhiều KB.
    """
    paragraphs = [p for p in footnote.paragraphs if p.strip()]
    if not paragraphs:
        return "", None

    total = sum(len(p) for p in paragraphs)

    if total <= inline_max:
        body = "\n>\n".join(f"> {p}" for p in paragraphs)
        return f"> **Sửa đổi:** {body.removeprefix('> ')}", None

    head = paragraphs[0]
    inline = (
        f"> **Sửa đổi:** {head}\n>\n"
        f"> _(Trích dẫn đầy đủ: xem chú thích [{footnote.number}] ở cuối văn bản.)_"
    )
    deferred = "\n\n".join([f"**[{footnote.number}]**", *paragraphs])
    return inline, deferred


# ==========================================================================
# FRONT MATTER
# ==========================================================================

# Số block đầu văn bản còn được coi là vùng dẫn nhập khi dò loại văn bản.
_DOC_TYPE_SEARCH_LIMIT = 20

# Số đoạn sau heading Điều hiệu lực còn được dò ngày hiệu lực.
_HIEU_LUC_LOOKAHEAD = 3


def _parse_pipe_table(markdown: str) -> list[list[str]]:
    """Tách bảng Markdown dạng pipe về lại ma trận ô."""
    rows: list[list[str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= {"-", " "} and cell for cell in cells):
            continue  # dòng phân cách header
        rows.append(cells)
    return rows


def _cell(rows: list[list[str]], row: int, column: int) -> str:
    if row < len(rows) and column < len(rows[row]):
        return rows[row][column]
    return ""


def _iso_date(match) -> str:
    day, month, year = match.group(1), match.group(2), match.group(3)
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def extract_quoc_hieu(table_block: Block | None) -> dict[str, str | None]:
    """Trích số hiệu, cơ quan ban hành và ngày ban hành từ bảng quốc hiệu."""
    result: dict[str, str | None] = {
        "so_hieu": None,
        "co_quan_ban_hanh": None,
        "ngay_ban_hanh": None,
    }
    if table_block is None:
        return result

    rows = _parse_pipe_table(table_block.text)

    match = RE_SO_HIEU.search(table_block.text)
    if match is not None:
        result["so_hieu"] = match.group(1).rstrip(".,;")

    # Ô (0,0) là tên cơ quan, phía dưới có dòng gạch ngang trang trí.
    # Giữ nguyên dạng viết hoa của nguồn: title-case tiếng Việt là phỏng đoán
    # ("VĂN PHÒNG QUỐC HỘI" -> .title() cho "Văn Phòng Quốc Hội", sai).
    issuer = _cell(rows, 0, 0).split("<br>")[0].strip()
    if issuer and set(issuer) - {"-", " "}:
        result["co_quan_ban_hanh"] = issuer

    # Ngày ban hành nằm ở ô cạnh số hiệu. Dò trên các dòng có "Số:" hoặc dòng
    # cuối để tránh bắt nhầm ngày trong tên cơ quan.
    for row in rows:
        joined = " | ".join(row)
        if "Số:" not in joined:
            continue
        date_match = RE_NGAY.search(joined)
        if date_match is not None:
            result["ngay_ban_hanh"] = _iso_date(date_match)
            break
    else:
        date_match = RE_NGAY.search(table_block.text)
        if date_match is not None:
            result["ngay_ban_hanh"] = _iso_date(date_match)

    return result


def _find_ten_van_ban(blocks: list[Block], limit: int) -> str | None:
    for block in blocks[:limit]:
        if block.kind != "paragraph":
            continue
        match = RE_BAN_HANH.search(block.text)
        if match is not None:
            return match.group(1).strip().rstrip(".")
    for block in blocks[:limit]:
        if block.kind != "paragraph":
            continue
        match = RE_TEN_SO.match(block.text)
        if match is not None:
            return match.group(1).strip()
    return None


def _find_loai_van_ban(blocks: list[Block], so_hieu: str | None) -> str | None:
    if so_hieu and "VBHN" in so_hieu.upper():
        return "Văn bản hợp nhất"
    for block in blocks[:_DOC_TYPE_SEARCH_LIMIT]:
        if block.kind == "paragraph" and RE_DOC_TYPE.match(block.text):
            return block.text.capitalize()
    return None


def _find_ngay_hieu_luc(blocks: list[Block]) -> str | None:
    """Ưu tiên Điều về hiệu lực thi hành, sau đó mới tới dòng dẫn nhập.

    Với văn bản hợp nhất, giá trị trả về là ngày hiệu lực của luật GỐC, không
    phải của luật sửa đổi mới nhất. Đó là đúng — Luật Bảo hiểm y tế hợp nhất trả
    về 2009-07-01 của luật 2008. Đừng "sửa" thành ngày mới nhất.
    """
    for index, block in enumerate(blocks):
        if block.kind != "paragraph":
            continue
        dieu = RE_DIEU.match(block.text)
        if dieu is None or "hiệu lực" not in dieu.group(2).casefold():
            continue
        for following in blocks[index + 1 : index + 1 + _HIEU_LUC_LOOKAHEAD]:
            if following.kind != "paragraph":
                continue
            match = RE_HIEU_LUC.search(following.text)
            if match is not None:
                return _iso_date(match)

    for block in blocks:
        if block.kind != "paragraph" or "số " not in block.text.casefold():
            continue
        match = RE_HIEU_LUC.search(block.text)
        if match is not None:
            return _iso_date(match)

    return None


def build_frontmatter(
    body_blocks: list[Block],
    quoc_hieu: dict[str, str | None],
    *,
    source_path: str | None,
    is_phu_luc: bool,
) -> tuple[FrontMatter, list[QcWarning]]:
    """Dựng front matter từ bảng quốc hiệu và thân văn bản đã gỡ marker."""
    warnings: list[QcWarning] = []

    so_hieu = quoc_hieu.get("so_hieu")
    co_quan_ban_hanh = quoc_hieu.get("co_quan_ban_hanh")
    ngay_ban_hanh = quoc_hieu.get("ngay_ban_hanh")

    ten_van_ban = _find_ten_van_ban(body_blocks, _DOC_TYPE_SEARCH_LIMIT)
    loai_van_ban = _find_loai_van_ban(body_blocks, so_hieu)
    ngay_hieu_luc = _find_ngay_hieu_luc(body_blocks)

    is_van_ban_hop_nhat = bool(so_hieu and "VBHN" in so_hieu.upper())
    if not is_van_ban_hop_nhat:
        is_van_ban_hop_nhat = any(
            block.kind == "paragraph" and "hợp nhất" in block.text.casefold()
            for block in body_blocks[:_DOC_TYPE_SEARCH_LIMIT]
        )

    # so_hieu và ngay_hieu_luc là cặp bắt buộc theo Task 9.
    for field, value in (("so_hieu", so_hieu), ("ngay_hieu_luc", ngay_hieu_luc)):
        if not value:
            warnings.append(QcWarning("missing_frontmatter_field", field))

    for field, value in (
        ("loai_van_ban", loai_van_ban),
        ("ten_van_ban", ten_van_ban),
        ("co_quan_ban_hanh", co_quan_ban_hanh),
        ("ngay_ban_hanh", ngay_ban_hanh),
    ):
        if not value:
            warnings.append(QcWarning("missing_optional_frontmatter_field", field))

    front_matter = FrontMatter(
        so_hieu=so_hieu,
        loai_van_ban=loai_van_ban,
        ten_van_ban=ten_van_ban,
        co_quan_ban_hanh=co_quan_ban_hanh,
        ngay_ban_hanh=ngay_ban_hanh,
        ngay_hieu_luc=ngay_hieu_luc,
        is_van_ban_hop_nhat=is_van_ban_hop_nhat,
        is_phu_luc=is_phu_luc,
        source_path=source_path,
        parser_version=PARSER_VERSION,
    )
    return front_matter, warnings


# ==========================================================================
# PASS 2 — NHẬN DIỆN CẤU TRÚC VÀ GÁN HEADING
# ==========================================================================

def _is_uppercase_title(text: str) -> bool:
    """Đoạn có phải dòng tiêu đề viết hoa hay không."""
    letters = "".join(character for character in text if character.isalpha())
    return len(letters) >= 2 and letters.isupper()


def _triage_tables(
    blocks: list[Block],
) -> tuple[Block | None, list[Block], list[QcWarning]]:
    """Tách bảng quốc hiệu và loại bảng nhiễu.

    Nhận diện theo NỘI DUNG, không theo vị trí: bảng chữ ký của Nghị định 293
    nằm ở block 39/170, toàn bộ Phụ lục nằm sau nó. Hai trong sáu bảng chữ ký
    lại có ô đầu rỗng nên riêng "Nơi nhận:" không đủ.
    """
    warnings: list[QcWarning] = []
    quoc_hieu: Block | None = None
    kept: list[Block] = []

    for index, block in enumerate(blocks):
        if block.kind != "table":
            kept.append(block)
            continue

        if (
            quoc_hieu is None
            and index < 3
            and RE_QUOC_HIEU_CELL.search(block.text)
        ):
            quoc_hieu = block
            continue

        if RE_SIGNATURE_CELL.search(block.text):
            warnings.append(QcWarning("dropped_noi_nhan_table", f"block {index}"))
            continue

        if RE_ATTACHMENT_TABLE.search(block.text):
            warnings.append(QcWarning("dropped_attachment_table", f"block {index}"))
            continue

        kept.append(block)

    if quoc_hieu is None:
        warnings.append(QcWarning("missing_quoc_hieu_table", ""))

    return quoc_hieu, kept, warnings


def _find_preamble_end(blocks: list[Block]) -> int:
    """Chỉ số block của heading cấu trúc đầu tiên.

    Vùng trước đó là dẫn nhập và được xuất nguyên văn. Văn bản hợp nhất liệt kê
    luật sửa đổi bằng dòng đánh số ngay trong vùng này ("1. Luật Nhà giáo số
    73/2025/QH15..."); nếu chạy RE_KHOAN ở đây thì sinh ra "##### Khoản 1-4"
    giả trước cả Chương I.
    """
    for index, block in enumerate(blocks):
        if block.kind == "paragraph" and is_structural(block.text):
            return index
    return len(blocks)


def _join_title(
    blocks: list[Block],
    index: int,
    refs: dict[int, list[int]],
    separator: str,
) -> tuple[str, str | None, int, list[int]]:
    """Gộp nhãn cấu trúc với dòng tiêu đề viết hoa ngay sau nó.

    Trong corpus, "Chương I" và "NHỮNG QUY ĐỊNH CHUNG" là hai paragraph riêng.
    Ví dụ của spec ("## Chương I. Những quy định chung") chỉ tạo được khi gộp.

    Trả về ``(tiêu_đề, phần_dư, số_block_đã_tiêu_thụ, refs_gộp)``. Phần dư là
    đoạn "(Kèm theo ...)" tách khỏi tiêu đề Phụ lục của Nghị định 293 — nếu
    không tách thì heading dài ~190 ký tự.
    """
    title = blocks[index].text
    collected = list(refs.get(index, []))
    consumed = 1
    extra: str | None = None

    following = index + 1
    if following < len(blocks) and blocks[following].kind == "paragraph":
        candidate = blocks[following].text
        # Tách "(Kèm theo ...)" TRƯỚC khi kiểm tra viết hoa: tiêu đề Phụ lục của
        # Nghị định 293 là một paragraph duy nhất kết thúc bằng phần trong ngoặc
        # có chữ thường, nếu kiểm tra sau thì dòng này không bao giờ được gộp.
        match = RE_KEM_THEO.search(candidate)
        if match is not None:
            extra = match.group(1).strip()
            candidate = candidate[: match.start()].strip()

        if candidate and _is_uppercase_title(candidate) and not is_structural(candidate):
            title = f"{title}{separator}{candidate}"
            # Marker của Chương IV Luật Thuế TNCN nằm ở dòng thứ hai
            # ("ĐIỀU KHOẢN THI HÀNH[3]"), phải gộp theo.
            collected.extend(refs.get(following, []))
            consumed = 2
        else:
            extra = None

    return title, extra, consumed, collected


def _emit(
    blocks: list[Block],
    refs: dict[int, list[int]],
    footnote_map: dict[int, Footnote],
    preamble_end: int,
) -> tuple[list[str], list[str], bool, list[QcWarning]]:
    """Sinh các khối markdown theo bảng ưu tiên của Task 5."""
    warnings: list[QcWarning] = []
    parts: list[str] = []
    deferred: list[str] = []
    in_phu_luc = False
    used: set[int] = set()

    def flush(numbers: list[int]) -> None:
        for number in numbers:
            footnote = footnote_map.get(number)
            if footnote is None:
                warnings.append(QcWarning("orphan_footnote", f"[{number}]"))
                continue
            used.add(number)
            inline, tail = render_blockquote(
                footnote, inline_max=FOOTNOTE_INLINE_MAX_CHARS
            )
            if inline:
                parts.append(inline)
            if tail is not None:
                deferred.append(tail)
                warnings.append(
                    QcWarning("long_footnote_deferred", f"[{number}]")
                )

    index = 0
    while index < len(blocks):
        block = blocks[index]
        here = refs.get(index, [])

        if block.kind == "table":
            parts.append(block.text)
            flush(here)
            index += 1
            continue

        if index < preamble_end:
            parts.append(block.text)
            flush(here)
            index += 1
            continue

        text = block.text

        if RE_PHU_LUC.match(text):
            title, extra, consumed, merged = _join_title(blocks, index, refs, " — ")
            parts.append(f"# {title}")
            flush(merged)
            if extra:
                parts.append(extra)
            in_phu_luc = True
            index += consumed
            continue

        if RE_PHAN.match(text):
            title, extra, consumed, merged = _join_title(blocks, index, refs, ". ")
            parts.append(f"# {title}")
            flush(merged)
            if extra:
                parts.append(extra)
            in_phu_luc = False
            index += consumed
            continue

        if RE_CHUONG.match(text):
            title, extra, consumed, merged = _join_title(blocks, index, refs, ". ")
            parts.append(f"## {title}")
            flush(merged)
            if extra:
                parts.append(extra)
            in_phu_luc = False
            index += consumed
            continue

        if RE_MUC.match(text):
            parts.append(f"### {text}")
            flush(here)
            index += 1
            continue

        if RE_DIEU.match(text):
            # NT-4: giữ nguyên tiêu đề gốc của Điều.
            parts.append(f"#### {text}")
            flush(here)
            index += 1
            continue

        khoan = RE_KHOAN.match(text)
        if khoan is not None:
            number, content = khoan.group(1), khoan.group(2)
            if (
                in_phu_luc
                and PHU_LUC_HEADING_WITH_TITLE
                and len(content) <= PHU_LUC_HEADING_MAX_CHARS
            ):
                # Tên tỉnh/thành là thông tin định danh quan trọng nhất của mục,
                # giữ nó trên dòng heading để breadcrumb của chunk con có tên.
                parts.append(f"##### {number}. {content}")
            else:
                # NT-3: heading của Khoản chỉ chứa nhãn, nội dung xuống dòng.
                parts.append(f"##### Khoản {number}")
                parts.append(content)
            flush(here)
            index += 1
            continue

        # Còn lại, gồm điểm "a)" và "- ...": xuất nguyên văn.
        parts.append(text)
        flush(here)
        index += 1

    for number in sorted(set(footnote_map) - used):
        warnings.append(QcWarning("unused_footnote", f"[{number}]"))

    return parts, deferred, in_phu_luc, warnings


def normalize(
    blocks: list[Block], *, source_path: str | None = None
) -> tuple[str, list[QcWarning]]:
    """Chuyển danh sách Block thành markdown có heading. Hàm thuần."""
    warnings: list[QcWarning] = []

    # S0 — triage bảng, phải chạy trước mọi thứ khác.
    quoc_hieu_block, body, table_warnings = _triage_tables(blocks)
    warnings.extend(table_warnings)

    # S1 — cắt vùng chú thích TRƯỚC khi nhận diện heading.
    region_start, region_warnings = find_region_start(body)
    warnings.extend(region_warnings)
    if region_start is None:
        footnote_map: dict[int, Footnote] = {}
    else:
        footnote_map, parse_warnings = parse_region(body[region_start:])
        warnings.extend(parse_warnings)
        body = body[:region_start]

    # S2 — gỡ marker TRƯỚC khi nhận diện heading.
    body, refs, strip_warnings = strip_all(body)
    warnings.extend(strip_warnings)

    # S3 — biên vùng dẫn nhập.
    preamble_end = _find_preamble_end(body)
    if not KEEP_PREAMBLE:
        # Bỏ hẳn dẫn nhập khỏi body; refs phải dịch chỉ số theo.
        body = body[preamble_end:]
        refs = {
            index - preamble_end: numbers
            for index, numbers in refs.items()
            if index >= preamble_end
        }
        preamble_end = 0

    # S5 — emit (chạy trước front matter vì is_phu_luc do vòng emit xác định).
    parts, deferred, is_phu_luc, emit_warnings = _emit(
        body, refs, footnote_map, preamble_end
    )
    warnings.extend(emit_warnings)

    # S4 — front matter, dựng trên body đã gỡ marker.
    front_matter, fm_warnings = build_frontmatter(
        body,
        extract_quoc_hieu(quoc_hieu_block),
        source_path=source_path,
        is_phu_luc=is_phu_luc,
    )
    warnings.extend(fm_warnings)

    # S6 — ghép.
    sections = [front_matter.to_yaml(), *parts]
    if deferred:
        sections.extend(deferred)
    markdown = "\n\n".join(section for section in sections if section) + "\n"

    # S7 — QC.
    warnings.extend(validate(markdown))

    return markdown, warnings


# ==========================================================================
# QC
# ==========================================================================

def _strip_front_matter(markdown: str) -> list[str]:
    """Bỏ khối YAML đầu file, trả về các dòng còn lại."""
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return lines
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return lines[index + 1 :]
    return lines


def _monotonic_breaks(numbers: list[str]) -> list[str]:
    """Các vị trí số hiệu không tăng dần, so sánh theo (số, hậu tố)."""
    breaks: list[str] = []
    previous: tuple[int, str] | None = None
    for number in numbers:
        current = sort_key(number)
        if previous is not None and current <= previous:
            breaks.append(number)
        previous = current
    return breaks


def validate(markdown: str) -> list[QcWarning]:
    """Chạy toàn bộ rule QC trên markdown."""
    warnings: list[QcWarning] = []
    lines = _strip_front_matter(markdown)

    in_phu_luc = False
    # seen_dieu là sticky trong thân văn bản: Chương/Mục không reset nó. Đó là
    # NT-1 — level trên bị bỏ trống là hợp lệ.
    seen_dieu = False
    reported_level_skip = False
    in_code_or_table = False

    heading_lines: list[tuple[int, str, str]] = []  # (số dòng, dấu #, nội dung)
    dieu_numbers: list[str] = []
    khoan_by_parent: dict[str, list[str]] = {}
    parent = "<none>"
    dieu_has_content: dict[str, bool] = {}
    current_dieu: str | None = None

    for line_number, raw in enumerate(lines, start=1):
        line = raw.rstrip()

        if line.startswith("<table>"):
            in_code_or_table = True
        if line.startswith("</table>"):
            in_code_or_table = False
            continue
        if in_code_or_table or line.startswith("|"):
            if current_dieu is not None:
                dieu_has_content[current_dieu] = True
            continue

        if not line.startswith("#"):
            if line.strip() and current_dieu is not None:
                dieu_has_content[current_dieu] = True
            continue

        hashes = line[: len(line) - len(line.lstrip("#"))]
        content = line[len(hashes) :].strip()
        heading_lines.append((line_number, hashes, content))

        if len(hashes) > 5:
            warnings.append(
                QcWarning("heading_too_deep", f"dòng {line_number}: {hashes}")
            )

        if len(hashes) > len(content) and not content:
            continue

        if len(hashes) == 1:
            in_phu_luc = RE_PHU_LUC.match(content) is not None
            parent = "phu_luc" if in_phu_luc else f"phan:{content}"
            current_dieu = None
        elif len(hashes) == 4:
            seen_dieu = True
            current_dieu = content
            dieu_has_content.setdefault(content, False)
            match = RE_DIEU.match(content)
            if match is not None:
                dieu_numbers.append(match.group(1))
                parent = f"dieu:{match.group(1)}"
            else:
                parent = f"dieu:{content}"
        elif len(hashes) == 5:
            # Trong Phụ lục, "#####" nằm trực tiếp dưới "#" là hợp lệ. Không có
            # ngoại lệ này thì mọi văn bản có Phụ lục đều báo cảnh báo giả.
            if not in_phu_luc and not seen_dieu and not reported_level_skip:
                warnings.append(
                    QcWarning("heading_level_skip", f"dòng {line_number}: {content}")
                )
                reported_level_skip = True
            number = content.removeprefix("Khoản ").split(".")[0].strip()
            khoan_by_parent.setdefault(parent, []).append(number)
            if current_dieu is not None:
                dieu_has_content[current_dieu] = True

        if len(line) > HEADING_MAX_LEN:
            warnings.append(
                QcWarning(
                    "suspicious_heading_length",
                    f"dòng {line_number}: {len(line)} ký tự",
                )
            )

    if not heading_lines:
        warnings.append(QcWarning("no_heading", ""))

    for number in _monotonic_breaks(dieu_numbers):
        warnings.append(QcWarning("dieu_not_monotonic", f"Điều {number}"))

    for parent_key, numbers in khoan_by_parent.items():
        for number in _monotonic_breaks(numbers):
            warnings.append(
                QcWarning("khoan_not_monotonic", f"{parent_key} -> khoản {number}")
            )

    for dieu, has_content in dieu_has_content.items():
        if not has_content:
            warnings.append(QcWarning("empty_dieu", dieu))

    for match in RE_FOOTNOTE_MARKER.finditer(markdown):
        # Marker còn sót ngoài blockquote/vùng dời là lỗi thật; trong hai vùng
        # đó thì "[n]" là nhãn cố ý.
        line_start = markdown.rfind("\n", 0, match.start()) + 1
        prefix = markdown[line_start : match.start()].lstrip()
        if prefix.startswith(">") or prefix.startswith("**"):
            continue
        warnings.append(QcWarning("orphan_footnote", match.group(0)))

    return warnings


# ==========================================================================
# PASS 1 — ĐỌC DOCX
# ==========================================================================

NON_BREAKING_SPACE = " "


def normalize_text(raw: str) -> str:
    """Chuẩn hóa NFC, thay non-breaking space, gộp khoảng trắng, strip."""
    text = unicodedata.normalize("NFC", raw)
    text = text.replace(NON_BREAKING_SPACE, " ")
    return " ".join(text.split())


def _cell_lines(cell) -> list[str]:
    """Lấy các dòng có nội dung trong một ô Word."""
    return [
        normalize_text(line)
        for paragraph in cell.paragraphs
        for line in paragraph.text.splitlines()
        if line.strip()
    ]


def _cell_to_markdown(cell) -> str:
    """Chuyển nội dung một ô Word thành nội dung hợp lệ trong bảng Markdown."""
    return "<br>".join(_cell_lines(cell)).replace("|", r"\|")


def _single_row_table_to_html(table: Table) -> str:
    """Xuất bảng một hàng không có header bằng HTML hợp lệ trong Markdown.

    Sau khi bảng chữ ký bị loại ở bước triage, các bảng một hàng còn lại là
    bảng công thức của Nghị định 145 ("Tiền lương làm thêm giờ = ... x ...").
    Bảng pipe sẽ đẩy số hạng đầu tiên lên làm header, nên dùng HTML.
    """
    cells = [
        f"      <td>{'<br>'.join(escape(line) for line in _cell_lines(cell))}</td>"
        for cell in table.rows[0].cells
    ]
    return "\n".join(
        [
            "<table>",
            "  <tbody>",
            "    <tr>",
            *cells,
            "    </tr>",
            "  </tbody>",
            "</table>",
        ]
    )


def table_to_markdown(table: Table) -> str:
    """Chuyển bảng DOCX sang cú pháp bảng phù hợp trong Markdown."""
    rows = [[_cell_to_markdown(cell) for cell in row.cells] for row in table.rows]

    if not rows:
        return ""

    if len(rows) == 1:
        return _single_row_table_to_html(table)

    column_count = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (column_count - len(row)) for row in rows]

    header = normalized_rows[0]
    body_rows = normalized_rows[1:]
    separator = ["---"] * column_count

    return "\n".join(
        f"| {' | '.join(row)} |" for row in (header, separator, *body_rows)
    )


def _paragraph_is_bold(paragraph: Paragraph) -> bool:
    """Đoạn có in đậm toàn bộ hay không.

    Chỉ là tín hiệu phụ. Rule nhận diện heading không bao giờ được đọc giá trị
    này: 100% paragraph trong corpus có style "Normal" và bold không phủ đủ.
    """
    runs = [run for run in paragraph.runs if run.text.strip()]
    return bool(runs) and all(run.bold for run in runs)


def read_docx(path: str | Path) -> list[Block]:
    """Đọc DOCX, giữ đúng thứ tự xen kẽ giữa paragraph và table.

    Duyệt ``document.element.body`` chứ không duyệt ``doc.paragraphs`` rồi
    ``doc.tables`` riêng — cách sau làm mất thứ tự (lỗi số 1 ở Phụ lục C).
    """
    document = Document(str(path))
    blocks: list[Block] = []

    for element in document.element.body.iterchildren():
        if isinstance(element, CT_P):
            paragraph = Paragraph(element, document)
            text = normalize_text(paragraph.text)
            if not text:
                continue
            style = paragraph.style.name if paragraph.style is not None else None
            blocks.append(
                Block(
                    kind="paragraph",
                    text=text,
                    style=style,
                    is_bold=_paragraph_is_bold(paragraph),
                )
            )
        elif isinstance(element, CT_Tbl):
            markdown_table = table_to_markdown(Table(element, document))
            if markdown_table:
                blocks.append(Block(kind="table", text=markdown_table))

    return blocks


# ==========================================================================
# GHI FILE
# ==========================================================================

def write_atomic(path: str | Path, content: str) -> str:
    """Ghi nội dung và trả về sha256 của nó.

    Ghi ra file ``.tmp`` cùng thư mục rồi ``os.replace`` để không bao giờ để lại
    markdown hỏng một nửa khi worker bị kill. ``os.replace`` là nguyên tử khi
    nguồn và đích nằm trên cùng filesystem, nên file tạm phải cùng thư mục.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return sha256(content.encode("utf-8")).hexdigest()


# ==========================================================================
# POSTGRESQL (chỉ main process)
# ==========================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id              BIGSERIAL PRIMARY KEY,
    source_path     TEXT UNIQUE NOT NULL,
    source_hash     TEXT NOT NULL,
    markdown_path   TEXT,
    markdown_hash   TEXT,
    parser_version  TEXT NOT NULL,
    status          TEXT NOT NULL,
    warnings        JSONB,
    error_message   TEXT,
    processed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT documents_status_chk
        CHECK (status IN ('processing', 'completed', 'failed'))
);

CREATE INDEX IF NOT EXISTS documents_status_idx ON documents (status);
"""


def connect(dsn: str) -> psycopg.Connection:
    """Mở kết nối. Chỉ main process gọi hàm này."""
    return psycopg.connect(dsn, row_factory=dict_row)


def init_schema(conn: psycopg.Connection) -> None:
    """Tạo bảng nếu chưa có. Idempotent, chạy được nhiều lần."""
    with conn.cursor() as cursor:
        cursor.execute(SCHEMA)
    conn.commit()


def get_document(conn: psycopg.Connection, source_path: str) -> dict[str, Any] | None:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM documents WHERE source_path = %s", (source_path,)
        )
        return cursor.fetchone()


def mark_processing(
    conn: psycopg.Connection,
    source_path: str,
    source_hash: str,
    parser_version: str,
) -> None:
    """Gộp INSERT/UPDATE thành một câu lệnh để không tạo bản ghi trùng."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO documents (
                source_path, source_hash, parser_version, status,
                markdown_path, markdown_hash, warnings, error_message, processed_at
            )
            VALUES (%s, %s, %s, 'processing', NULL, NULL, NULL, NULL, NULL)
            ON CONFLICT (source_path) DO UPDATE SET
                source_hash    = EXCLUDED.source_hash,
                parser_version = EXCLUDED.parser_version,
                status         = 'processing',
                markdown_hash  = NULL,
                warnings       = NULL,
                error_message  = NULL,
                processed_at   = NULL,
                updated_at     = now()
            """,
            (source_path, source_hash, parser_version),
        )


def mark_completed(
    conn: psycopg.Connection,
    source_path: str,
    source_hash: str,
    markdown_path: str,
    markdown_hash: str,
    warnings: list[dict[str, str]],
) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents
            SET source_hash   = %s,
                markdown_path = %s,
                markdown_hash = %s,
                status        = 'completed',
                warnings      = %s,
                error_message = NULL,
                processed_at  = now(),
                updated_at    = now()
            WHERE source_path = %s
            """,
            (source_hash, markdown_path, markdown_hash, Jsonb(warnings), source_path),
        )


def mark_failed(
    conn: psycopg.Connection, source_path: str, error_message: str
) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents
            SET status        = 'failed',
                processed_at  = NULL,
                error_message = %s,
                updated_at    = now()
            WHERE source_path = %s
            """,
            (error_message, source_path),
        )


def reset_stale_processing(conn: psycopg.Connection) -> int:
    """Đưa các bản ghi ``processing`` còn sót từ lần chạy trước về ``failed``."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            UPDATE documents
            SET status        = 'failed',
                error_message = 'stale processing state',
                updated_at    = now()
            WHERE status = 'processing'
            """
        )
        count = cursor.rowcount
    conn.commit()
    return count


def status_stats(conn: psycopg.Connection) -> list[tuple[str, int]]:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT status, count(*) AS n FROM documents GROUP BY status ORDER BY status"
        )
        return [(row["status"], row["n"]) for row in cursor.fetchall()]


def warning_stats(conn: psycopg.Connection) -> list[tuple[str, int]]:
    """Đếm cảnh báo theo mã trên toàn bộ bảng."""
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT item ->> 'code' AS code, count(*) AS n
            FROM documents, jsonb_array_elements(warnings) AS item
            WHERE warnings IS NOT NULL
            GROUP BY code
            ORDER BY n DESC, code
            """
        )
        return [(row["code"], row["n"]) for row in cursor.fetchall()]


# ==========================================================================
# ENTRYPOINT CỦA WORKER
# ==========================================================================

def process_one(source_path: str, output_path: str) -> dict:
    """Chuyển một DOCX thành Markdown. KHÔNG chạm PostgreSQL."""
    blocks = read_docx(source_path)
    if not blocks:
        raise ValueError(f"Không trích xuất được nội dung: {source_path}")

    markdown, warnings = normalize(blocks, source_path=source_path)
    markdown_hash = write_atomic(output_path, markdown)

    return {
        "source_path": source_path,
        "markdown_path": output_path,
        "markdown_hash": markdown_hash,
        "warnings": [warning.as_dict() for warning in warnings],
        "pid": os.getpid(),
    }


# ==========================================================================
# ĐIỀU PHỐI
# ==========================================================================

def sha256_file(path: Path, chunk_size: int = 64 * 1024) -> str:
    """Tính SHA-256 theo từng khối để không nạp toàn bộ DOCX vào bộ nhớ."""
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scan_docx_files(raw_dir: Path) -> list[Path]:
    """Quét đệ quy, trả về danh sách DOCX theo thứ tự ổn định.

    Bỏ qua file tạm của Word (tên bắt đầu bằng ``~$``) — mở một file đó bằng
    python-docx sẽ lỗi và làm bẩn bảng trạng thái.
    """
    if not raw_dir.exists():
        return []
    return sorted(
        (
            path
            for path in raw_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() == ".docx"
            and not path.name.startswith("~$")
        ),
        key=lambda path: str(path).casefold(),
    )


def output_path_for(source: Path, raw_dir: Path, out_dir: Path) -> Path:
    """Giữ nguyên cấu trúc thư mục con, đổi đuôi thành ``.md``."""
    relative = source.relative_to(raw_dir)
    return (out_dir / relative).with_suffix(".md")


def decide(
    record: dict[str, Any] | None,
    current_hash: str,
    output_path: Path,
    *,
    force: bool,
) -> tuple[bool, str]:
    """Quyết định xử lý hay bỏ qua một file, kèm lý do để in ở --dry-run."""
    if force:
        return True, "force"
    if record is None:
        return True, "file mới"
    if record["status"] == "failed":
        return True, "retry sau lỗi"
    if record["source_hash"] != current_hash:
        return True, "nội dung đổi"
    if record["parser_version"] != PARSER_VERSION:
        return True, "parser_version đổi"
    # markdown_path có thể là NULL với bản ghi processing dở dang.
    if not record["markdown_path"] or not Path(record["markdown_path"]).exists():
        return True, "output bị xóa"
    return False, "không đổi"


def _print_summary(
    successful: list[str],
    failed: list[tuple[str, str]],
    skipped: list[str],
    warning_counter: collections.Counter,
) -> None:
    print("\nKết quả")
    print("─" * 36)
    print(f"✓ Thành công : {len(successful)}")
    print(f"✗ Thất bại   : {len(failed)}")
    print(f"↷ Bỏ qua     : {len(skipped)}")

    if warning_counter:
        print("\nCảnh báo theo mã")
        print("─" * 36)
        width = max(len(code) for code in warning_counter)
        for code, count in warning_counter.most_common():
            print(f"  {code:<{width}}  {count}")

    for source_path, message in failed:
        print(f"\n✗ {Path(source_path).name}\n  {message}")


def run(
    *,
    raw_dir: Path,
    out_dir: Path,
    dsn: str,
    force: bool = False,
    dry_run: bool = False,
    single_file: Path | None = None,
) -> int:
    """Chạy toàn bộ pipeline. Trả về exit code."""
    raw_dir = raw_dir.resolve()
    out_dir = out_dir.resolve()

    if single_file is not None:
        sources = [single_file.resolve()]
        raw_dir = sources[0].parent
    else:
        sources = scan_docx_files(raw_dir)

    if not sources:
        print(f"Không tìm thấy .docx nào trong {raw_dir}")
        return 0

    successful: list[str] = []
    failed: list[tuple[str, str]] = []
    skipped: list[str] = []
    pending: list[PendingDocument] = []
    warning_counter: collections.Counter = collections.Counter()

    with connect(dsn) as conn:
        init_schema(conn)
        recovered = reset_stale_processing(conn)
        if recovered:
            print(f"Khôi phục {recovered} tài liệu bị gián đoạn để retry.")

        for source_path in sources:
            try:
                current_hash = sha256_file(source_path)
            except OSError as error:
                failed.append((str(source_path), repr(error)))
                continue

            output_path = output_path_for(source_path, raw_dir, out_dir)
            record = get_document(conn, str(source_path))
            should, reason = decide(record, current_hash, output_path, force=force)

            if dry_run:
                label = "PROCESS" if should else "SKIP   "
                print(f"{label}  {source_path.name}  ({reason})")
                if should:
                    pending.append(
                        PendingDocument(source_path, current_hash, output_path)
                    )
                else:
                    skipped.append(str(source_path))
                continue

            if not should:
                skipped.append(str(source_path))
                continue

            mark_processing(
                conn, str(source_path), current_hash, PARSER_VERSION
            )
            pending.append(PendingDocument(source_path, current_hash, output_path))

        if dry_run:
            print(f"\nPROCESS: {len(pending)}   SKIP: {len(skipped)}")
            return 0

        # Trạng thái processing phải bền vững trước khi tạo worker.
        conn.commit()

        if pending and single_file is not None:
            # Chạy tuần tự trong main process để dễ debug và đặt breakpoint.
            for document in pending:
                _handle_sequential(
                    conn, document, successful, failed, warning_counter
                )
        elif pending:
            _handle_parallel(conn, pending, successful, failed, warning_counter)

    _print_summary(successful, failed, skipped, warning_counter)
    return 1 if failed else 0


def _record_success(
    conn,
    document: PendingDocument,
    result: dict,
    successful: list[str],
    warning_counter: collections.Counter,
) -> None:
    mark_completed(
        conn,
        str(document.source_path),
        document.source_hash,
        result["markdown_path"],
        result["markdown_hash"],
        result["warnings"],
    )
    successful.append(str(document.source_path))
    warning_counter.update(warning["code"] for warning in result["warnings"])


def _handle_sequential(
    conn,
    document: PendingDocument,
    successful: list[str],
    failed: list[tuple[str, str]],
    warning_counter: collections.Counter,
) -> None:
    try:
        result = process_one(str(document.source_path), str(document.output_path))
        _record_success(conn, document, result, successful, warning_counter)
        print(f"✓ {document.source_path.name}")
    except Exception as error:
        traceback.print_exc()
        mark_failed(conn, str(document.source_path), repr(error))
        failed.append((str(document.source_path), repr(error)))
    finally:
        conn.commit()


def _handle_parallel(
    conn,
    pending: list[PendingDocument],
    successful: list[str],
    failed: list[tuple[str, str]],
    warning_counter: collections.Counter,
) -> None:
    worker_count = min(MAX_WORKERS, len(pending))
    # as_completed nhận ngân sách cho CẢ vòng lặp, không phải mỗi task. Nhân
    # theo số lượt để một batch lớn không bị timeout oan.
    rounds = math.ceil(len(pending) / worker_count)
    timeout = WORKER_TIMEOUT_SEC * rounds
    width = max(len(document.source_path.name) for document in pending)

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_document = {
            executor.submit(
                process_one, str(document.source_path), str(document.output_path)
            ): document
            for document in pending
        }

        progress = tqdm(
            total=len(future_to_document),
            desc="Converting DOCX",
            unit="file",
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
        )
        try:
            for future in as_completed(future_to_document, timeout=timeout):
                document = future_to_document[future]
                try:
                    result = future.result()
                    _record_success(
                        conn, document, result, successful, warning_counter
                    )
                    message = f"✓ {document.source_path.name:<{width}} PID {result['pid']}"
                except Exception as error:
                    traceback.print_exc()
                    mark_failed(conn, str(document.source_path), repr(error))
                    failed.append((str(document.source_path), repr(error)))
                    message = (
                        f"✗ {document.source_path.name:<{width}} "
                        f"Lỗi: {type(error).__name__}"
                    )
                finally:
                    # Commit từng kết quả để không mất trạng thái của các file đã
                    # xong nếu main process bị ngắt.
                    conn.commit()
                    progress.update(1)
                tqdm.write(message)
        except BrokenProcessPool as error:
            # Worker chết đột ngột (OOM, segfault). Đánh dấu phần chưa xong.
            tqdm.write(f"✗ ProcessPool hỏng: {error!r}")
            for future, document in future_to_document.items():
                if future.done() or str(document.source_path) in successful:
                    continue
                mark_failed(conn, str(document.source_path), repr(error))
                failed.append((str(document.source_path), repr(error)))
            conn.commit()
        except TimeoutError as error:
            tqdm.write(f"✗ Quá thời gian {timeout}s: {error!r}")
            for future, document in future_to_document.items():
                if future.done():
                    continue
                future.cancel()
                mark_failed(conn, str(document.source_path), f"timeout {timeout}s")
                failed.append((str(document.source_path), f"timeout {timeout}s"))
            conn.commit()
        finally:
            progress.close()


# ==========================================================================
# CLI
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loader.py",
        description="Chuyển văn bản pháp luật .docx sang Markdown có cấu trúc.",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=None, help="Thư mục .docx đầu vào."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="Thư mục .md đầu ra."
    )
    parser.add_argument(
        "--force", action="store_true", help="Xử lý lại toàn bộ, bỏ qua logic SKIP."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Chỉ xử lý một file, chạy tuần tự trong main process để dễ debug.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Chỉ in PROCESS/SKIP, không ghi gì."
    )
    parser.add_argument(
        "--stats", action="store_true", help="In thống kê từ PostgreSQL rồi thoát."
    )
    return parser


def _print_stats(dsn: str) -> int:
    with connect(dsn) as conn:
        init_schema(conn)
        statuses = status_stats(conn)
        warnings = warning_stats(conn)

    print("Trạng thái")
    print("─" * 36)
    if statuses:
        for status, count in statuses:
            print(f"  {status:<12} {count}")
    else:
        print("  (chưa có bản ghi nào)")

    print("\nCảnh báo theo mã")
    print("─" * 36)
    if warnings:
        width = max(len(code) for code, _ in warnings)
        for code, count in warnings:
            print(f"  {code:<{width}}  {count}")
    else:
        print("  (không có)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    load_dotenv()
    dsn = get_database_url()

    if args.stats:
        return _print_stats(dsn)

    root = project_root()
    raw_dir = args.raw_dir if args.raw_dir is not None else root / DEFAULT_RAW_DIR
    out_dir = args.out_dir if args.out_dir is not None else root / DEFAULT_OUT_DIR

    return run(
        raw_dir=raw_dir,
        out_dir=out_dir,
        dsn=dsn,
        force=args.force,
        dry_run=args.dry_run,
        single_file=args.file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
