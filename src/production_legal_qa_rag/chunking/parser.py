"""Đọc markdown output của `formatting/`, dựng `DocumentTree` (mục 2, 4.6, 10).

Duyệt các block Markdown (tách theo dòng trống — đúng cách `formatting/`
ghép các phần lại) theo thứ tự, quy heading về cấp Phần/Phụ Lục -> Chương ->
Mục -> Điều -> Khoản theo mapping của `formatting_spec.md` mục 3, rồi gom nội
dung thân thành từng `KhoanNode`. Trích `source_document` + nội dung 2 "Khoản
ngầm định cấp văn bản" (frontmatter/backmatter, mục 4.6) trước/sau vùng cấu
trúc đó.

Không còn YAML front matter nào để đọc — `formatting/` không sinh YAML/field
cấu trúc nào (formatting_spec.md mục 1.1); toàn bộ file `.md` là markdown
thuần, `source_document` được suy ra deterministic từ heading `#` đầu tiên
trong vùng frontmatter (mục 2), không phải field trích xuất theo schema.

QUYẾT ĐỊNH THIẾT KẾ (spec không định nghĩa hoặc mô tả không khớp thực tế —
xem báo cáo bàn giao):

- **Marker backmatter thật khác mô tả `**[n]**`/`footnotes.py::render_blockquote`
  ở chunking_spec.md mục 4.6**: module `footnotes.py` đó đã bị xoá khỏi
  `formatting/` từ bản redesign Groq (2026-09-16, xem formatting_spec.md mục
  1.1) — tài liệu mục 4.6 là mô tả cũ chưa cập nhật theo redesign này. Marker
  thật, xác nhận trên toàn bộ 6 file `data/markdown/*.md`, là dòng `---` do
  `formatting/pipeline.py._compose_markdown` chèn để ngăn cách back matter
  (formatting_spec.md mục 1.1 "Ghép output cuối cùng", mục 2) — dùng
  `patterns.RE_BACKMATTER_SEPARATOR`, xem docstring ở đó.
- Điều không có Khoản con (nội dung nằm thẳng dưới heading Điều), hoặc đoạn
  mở đầu đứng trước Khoản đầu tiên của 1 Điều: được gom thành 1 `KhoanNode`
  với ``khoan_number=None`` ("Khoản ngầm định cấp Điều") thay vì bị bỏ qua,
  để không mất nội dung thật trong corpus (vd. Điều 46/47, Điều 48a
  `Luật bảo hiểm y tế.md`). Breadcrumb của Khoản ngầm định dừng ở cấp Điều
  (không có đoạn "- Khoản").
- Blockquote chú thích sửa đổi (dòng bắt đầu bằng ">") bị loại khỏi nội dung
  Khoản hoàn toàn — đây là metadata lịch sử sửa đổi, không phải nội dung
  pháp lý cần embed.
- Nội dung nằm trực tiếp dưới heading Phần/Phụ Lục/Chương/Mục MÀ KHÔNG có
  Khoản/Điều nào theo sau (vd. dòng "(Kèm theo Nghị định số...)" ngay dưới
  tiêu đề Phụ Lục) bị bỏ qua — ngoài phạm vi "1 chunk = 1 Khoản" của spec.
- Heading Khoản trong Phụ Lục dạng gộp (``##### 1. Thành phố Hà Nội``, xem
  `formatting/emitter.py::_emit_khoan`) VẪN được coi là 1 Khoản hợp lệ dù
  KHÔNG có Điều bao ngoài (Phụ Lục danh mục địa bàn nằm thẳng dưới `#`, không
  qua Chương/Điều) — nếu không, toàn bộ danh mục địa bàn (dữ liệu thật, có ý
  nghĩa tra cứu) sẽ bị mất trắng.
- Bảng công thức 1 hàng dạng HTML thô (``<table>...</table>``, do
  `formatting/tables.py::_single_row_table_to_html` sinh ra cho các bảng
  công thức như "Tiền lương làm thêm giờ = ... x ...") được coi là bảng
  giống bảng pipe — `has_table=True`, giữ nguyên không cắt (mục 5.1), dù
  mục 5 của spec chỉ mô tả literal bảng pipe.
- Khoản lồng trong Khoản (vd. Điều 219 `Văn bản hợp nhất bộ luật lao động.md`
  trích dẫn nguyên văn nhiều Điều/Khoản của luật khác, mà `formatting/` vẫn
  render bằng chính heading cấp 5 ``##### Khoản N`` cho nội dung trích dẫn):
  phát hiện qua độ sâu dấu ngoặc kép trích dẫn kiểu Việt "“" / "”" (quy ước
  soạn thảo văn bản pháp luật khi sửa đổi, bổ sung — trích nguyên văn điều
  luật được đặt trong 1 cặp ngoặc kép). Heading cấp 5 xuất hiện khi độ sâu
  ngoặc kép > 0 (đang ở trong 1 đoạn trích dẫn) không được coi là 1 Khoản
  mới cấp Điều — nội dung này chỉ là văn bản trích dẫn, không phải cấu trúc
  thật của văn bản đang parse. Heading đó được gộp làm văn bản thường vào
  Khoản đang mở (không flush, không đổi `khoan_number`/breadcrumb), để toàn
  bộ đoạn trích dẫn (bao gồm các "Khoản" lồng bên trong nó) nằm chung 1
  `KhoanNode` với Khoản thật đang chứa nó.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

from production_legal_qa_rag.chunking.models import DocumentTree, KhoanNode
from production_legal_qa_rag.chunking.patterns import (
    RE_BACKMATTER_SEPARATOR,
    RE_CHUONG,
    RE_DIEU,
    RE_HEADING,
    RE_HTML_TABLE_LINE,
    RE_KHOAN_LABEL,
    RE_KHOAN_MERGED,
    RE_MUC,
    RE_PHAN,
    RE_TABLE_LINE,
    is_structural_heading,
)

_RE_BLOCK_SPLIT = re.compile(r"\n{2,}")
_RE_LEADING_ASTERISKS = re.compile(r"^\*+")
_RE_TRAILING_ASTERISKS = re.compile(r"\*+$")


def _split_blocks(text: str) -> list[str]:
    """Tách toàn bộ file markdown thành các block theo dòng trống, đã `strip()`."""
    blocks = []
    for block in _RE_BLOCK_SPLIT.split(text):
        stripped = block.strip()
        if stripped:
            blocks.append(stripped)
    return blocks


def _find_structural_start(blocks: list[str]) -> int:
    """Chỉ số block đầu tiên là heading cấu trúc thật (mục 4.6).

    Trả về ``len(blocks)`` nếu không tìm thấy heading cấu trúc nào (toàn bộ
    file coi như frontmatter — không xảy ra trên corpus thật, nhưng không
    crash).
    """
    for index, block in enumerate(blocks):
        match = RE_HEADING.match(block)
        if match and is_structural_heading(len(match.group(1)), match.group(2).strip()):
            return index
    return len(blocks)


def _strip_markdown_emphasis(text: str) -> str:
    """Bỏ `**`/`*` bao quanh 1 dòng (vd. `"**LUẬT**"` -> `"LUẬT"`, mục 2)."""
    stripped = text.strip()
    stripped = _RE_LEADING_ASTERISKS.sub("", stripped)
    stripped = _RE_TRAILING_ASTERISKS.sub("", stripped)
    return stripped.strip()


def _extract_frontmatter(blocks: list[str], path: Path) -> tuple[str, str | None]:
    """Trích `source_document` + nội dung frontmatter từ các block trước
    heading cấu trúc đầu tiên (mục 2, 4.6).

    `source_document` = đoạn văn liền kề ngay phía trên heading `#` (H1) đầu
    tiên trong vùng frontmatter, nối với chính heading đó (bỏ `**`/`#`, cách
    nhau 1 khoảng trắng). Nếu không có đoạn nào đứng ngay trước heading ->
    chỉ lấy heading. Vì vùng frontmatter (theo định nghĩa `_find_structural_start`)
    chỉ chứa các heading KHÔNG cấu trúc, 1 heading H1 tìm thấy ở đây chắc
    chắn là heading tên văn bản (mục 4.6), không phải Phần/Phụ Lục thật.
    """
    if not blocks:
        return path.stem, None

    title_index: int | None = None
    heading_text: str | None = None
    for index, block in enumerate(blocks):
        match = RE_HEADING.match(block)
        if match and len(match.group(1)) == 1:
            title_index = index
            heading_text = match.group(2).strip()
            break

    if title_index is None or heading_text is None:
        # Không tìm thấy heading tên văn bản (không xảy ra trên corpus thật,
        # xem formatting_spec.md mục 1.1 "frontmatter_title_not_found") ->
        # fallback tên file, giữ nguyên toàn bộ frontmatter làm nội dung.
        return path.stem, "\n\n".join(blocks)

    if title_index > 0:
        preceding = _strip_markdown_emphasis(blocks[title_index - 1])
        source_document = f"{preceding} {heading_text}" if preceding else heading_text
    else:
        source_document = heading_text

    content_parts = [
        heading_text if index == title_index else block
        for index, block in enumerate(blocks)
    ]
    return source_document, "\n\n".join(content_parts)


def _phan_segment(text: str) -> str:
    match = RE_PHAN.match(text)
    return f"Phần {match.group(1)}" if match else text


def _chuong_segment(text: str) -> str:
    match = RE_CHUONG.match(text)
    return f"Chương {match.group(1)}" if match else text


def _muc_segment(text: str) -> str:
    match = RE_MUC.match(text)
    return f"Mục {match.group(1)}" if match else text


def _dieu_segment(text: str) -> str:
    match = RE_DIEU.match(text)
    if not match:
        return text
    number, title = match.group(1), match.group(2).strip()
    return f"Điều {number}. {title}" if title else f"Điều {number}"


def _build_prefix(
    source_document: str,
    phan: str | None,
    chuong: str | None,
    muc: str | None,
    dieu: str | None,
) -> str:
    parts = [source_document, phan, chuong, muc, dieu]
    return " - ".join(part for part in parts if part)


def parse_markdown(path: str | Path) -> DocumentTree:
    """Đọc 1 file markdown, dựng `DocumentTree` chứa toàn bộ `KhoanNode` +
    nội dung 2 "Khoản ngầm định cấp văn bản" (frontmatter/backmatter).

    Args:
        path: Đường dẫn tới file `.md` nguồn (output của `formatting/`).

    Returns:
        Cây breadcrumb -> Khoản, sẵn sàng cho `splitter.split_khoan`/
        `splitter.split_implicit_khoan`.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    blocks = _split_blocks(text)

    start_index = _find_structural_start(blocks)
    source_document, frontmatter_content = _extract_frontmatter(
        blocks[:start_index], path
    )

    khoans: list[KhoanNode] = []
    backmatter_content: str | None = None

    phan: str | None = None
    chuong: str | None = None
    muc: str | None = None
    dieu: str | None = None
    dieu_seen = False
    khoan_number: str | None = None
    prefix = source_document
    paragraphs: list[str] = []
    tables: list[str] = []
    # Độ sâu dấu ngoặc kép trích dẫn "“"/"”" hiện tại -- dùng để phát hiện
    # Khoản lồng trong đoạn trích dẫn (xem docstring đầu file). Reset về 0
    # mỗi khi gặp heading cấp 1-4 (quy ước soạn thảo không để 1 đoạn trích
    # dẫn vắt qua ranh giới Phần/Chương/Mục/Điều).
    quote_depth = 0

    def flush() -> None:
        # Vùng nội dung chỉ được ghi thành KhoanNode khi có nhãn Khoản rõ ràng
        # (kể cả Khoản trong Phụ Lục, không có Điều bao ngoài — vd. danh mục
        # địa bàn "##### 1. Thành phố Hà Nội"), hoặc là nội dung nằm trực tiếp
        # dưới 1 Điều (Khoản ngầm định). Nội dung khác (thân Phần/Phụ Lục/
        # Chương/Mục trước Khoản/Điều đầu tiên) bị bỏ qua.
        if khoan_number is None and not dieu_seen:
            return
        content = "\n\n".join(paragraphs).strip()
        has_table = bool(tables)
        if not content and not has_table:
            return
        khoans.append(
            KhoanNode(
                breadcrumb_prefix=prefix,
                khoan_number=khoan_number,
                content=content,
                has_table=has_table,
                raw_table="\n\n".join(tables) if has_table else None,
            )
        )

    found_backmatter = False
    index = start_index
    while index < len(blocks):
        block = blocks[index]
        heading_match = RE_HEADING.match(block)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()

            if level == 5 and quote_depth > 0:
                # Khoản lồng trong đoạn trích dẫn (xem docstring đầu file):
                # không phải cấu trúc thật của văn bản đang parse -- gộp làm
                # văn bản thường vào Khoản thật đang mở, không flush, không
                # đổi khoan_number/breadcrumb.
                paragraphs.append(heading_text)
                index += 1
                continue

            flush()
            paragraphs = []
            tables = []
            if level != 5:
                quote_depth = 0

            if level == 1:
                phan = _phan_segment(heading_text)
                chuong = muc = dieu = None
                dieu_seen = False
                khoan_number = None
            elif level == 2:
                chuong = _chuong_segment(heading_text)
                muc = dieu = None
                dieu_seen = False
                khoan_number = None
            elif level == 3:
                muc = _muc_segment(heading_text)
                dieu = None
                dieu_seen = False
                khoan_number = None
            elif level == 4:
                dieu = _dieu_segment(heading_text)
                dieu_seen = True
                khoan_number = None
            elif level == 5:
                label_match = RE_KHOAN_LABEL.match(heading_text)
                merged_match = (
                    None if label_match else RE_KHOAN_MERGED.match(heading_text)
                )
                if label_match:
                    khoan_number = label_match.group(1)
                elif merged_match:
                    khoan_number = merged_match.group(1)
                    paragraphs = [merged_match.group(2).strip()]
                else:
                    # Heading Khoản không khớp dạng nào đã biết: bỏ nhãn, giữ
                    # nội dung theo sau như Khoản ngầm định thay vì bỏ luôn
                    # nội dung đó. Nếu heading lỗi này xuất hiện 2 lần trong
                    # cùng 1 Điều (dữ liệu hỏng/OCR), 2 Khoản ngầm định sẽ
                    # cùng breadcrumb_prefix -> cùng chunk_id; invariant check
                    # chung ở `pipeline.py::_ensure_unique_chunk_ids` sẽ raise
                    # và chặn riêng file đó, thay vì âm thầm ghi đè.
                    khoan_number = None

            prefix = _build_prefix(source_document, phan, chuong, muc, dieu)
            index += 1
            continue

        # Marker backmatter thật (mục 4.6, xem docstring đầu file): 1 dòng
        # "---" đứng riêng do formatting/pipeline.py chèn, luôn xuất hiện sau
        # heading cuối cùng của file (nếu có backmatter). Chốt nốt Khoản đang
        # mở rồi dừng vòng lặp cấu trúc — toàn bộ phần còn lại là backmatter.
        if RE_BACKMATTER_SEPARATOR.match(block):
            flush()
            remaining = blocks[index + 1 :]
            backmatter_content = "\n\n".join(remaining) if remaining else None
            found_backmatter = True
            break

        quote_depth = max(quote_depth + block.count("“") - block.count("”"), 0)
        if block.startswith(">"):
            index += 1
            continue  # blockquote chú thích sửa đổi — không phải nội dung Khoản
        if RE_TABLE_LINE.match(block) or RE_HTML_TABLE_LINE.match(block):
            tables.append(block)
            index += 1
            continue
        paragraphs.append(block)
        index += 1

    if not found_backmatter:
        flush()

    if quote_depth != 0:
        # Cơ chế quote_depth (xem docstring đầu file) chỉ an toàn trên corpus
        # hiện tại vì đã xác nhận thủ công ngoặc kép cân bằng ở cả 6 file
        # `data/markdown/*.md`. Cảnh báo runtime ở đây để phát hiện SỚM văn
        # bản mới có ngoặc kép không cân (thay vì âm thầm nuốt mất 1 Khoản
        # thật).
        warnings.warn(
            f"{path}: ngoặc kép trích dẫn không cân (quote_depth={quote_depth} "
            "cuối file) -- có thể đã bỏ sót nội dung Khoản thật, xem "
            "parser.py docstring mục 'Khoản lồng trong Khoản'.",
            stacklevel=2,
        )

    return DocumentTree(
        source_document=source_document,
        frontmatter_content=frontmatter_content,
        backmatter_content=backmatter_content,
        khoans=khoans,
    )
