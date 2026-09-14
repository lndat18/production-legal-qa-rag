"""Đọc markdown đã có front matter, dựng `DocumentTree` (mục 3, 10).

Duyệt các block Markdown (tách theo dòng trống — đúng cách `formatting/`
ghép các phần lại bằng ``"\\n\\n".join(...)``) theo thứ tự, quy heading về
cấp Phần/Phụ Lục -> Chương -> Mục -> Điều -> Khoản theo mapping của
`formatting_spec.md` mục 3, rồi gom nội dung thân thành từng `KhoanNode`.

QUYẾT ĐỊNH THIẾT KẾ (spec không định nghĩa, xem báo cáo bàn giao):

- Điều không có Khoản con (nội dung nằm thẳng dưới heading Điều), hoặc đoạn
  mở đầu đứng trước Khoản đầu tiên của 1 Điều: được gom thành 1 `KhoanNode`
  với ``khoan_number=None`` ("Khoản ngầm định") thay vì bị bỏ qua, để không
  mất nội dung thật trong corpus (vd. Điều 46/47, Điều 48a
  `Luật bảo hiểm y tế.md`). Breadcrumb của Khoản ngầm định dừng ở cấp Điều
  (không có đoạn "- Khoản").
- Blockquote chú thích sửa đổi (dòng bắt đầu bằng ">") bị loại khỏi nội dung
  Khoản hoàn toàn — đây là metadata lịch sử sửa đổi
  (xem `formatting/footnotes.py::render_blockquote`), không phải nội dung
  pháp lý cần embed.
- Nội dung nằm trực tiếp dưới heading Phần/Phụ Lục/Chương/Mục MÀ KHÔNG có
  Khoản/Điều nào theo sau (vd. dòng "(Kèm theo Nghị định số...)" ngay dưới
  tiêu đề Phụ Lục) bị bỏ qua — ngoài phạm vi "1 chunk = 1 Khoản" của spec.
- Heading Khoản trong Phụ Lục dạng gộp (``##### 1. Thành phố Hà Nội``, xem
  `formatting/emitter.py::_emit_khoan`) VẪN được coi là 1 Khoản hợp lệ dù
  KHÔNG có Điều bao ngoài (Phụ Lục danh mục địa bàn nằm thẳng dưới `#`, không
  qua Chương/Điều) — nếu không, toàn bộ danh mục địa bàn (dữ liệu thật, có ý
  nghĩa tra cứu) sẽ bị mất trắng. `formatting/emitter.py` tự nhận nó là đơn
  vị chunking ("tên tỉnh/thành... cần có trong breadcrumb của chunk con").
  Phần tiêu đề ngắn được coi là đoạn văn bản đầu tiên của nội dung Khoản
  (không thêm slot riêng vào breadcrumb, breadcrumb vẫn theo đúng format
  literal "- Khoản {m}" của mục 3, chỉ thiếu đoạn "- Điều ...").
- Bảng công thức 1 hàng dạng HTML thô (``<table>...</table>``, do
  `formatting/tables.py::_single_row_table_to_html` sinh ra cho các bảng
  công thức như "Tiền lương làm thêm giờ = ... x ...") được coi là bảng
  giống bảng pipe — `has_table=True`, giữ nguyên không cắt (mục 5.1), dù
  mục 5 của spec chỉ mô tả literal bảng pipe. Xem `tables.py` để biết cách
  chuẩn hoá riêng cho dạng này.
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
  `KhoanNode` với Khoản thật đang chứa nó — vừa giữ đúng "vị trí pháp lý"
  (breadcrumb trỏ về Khoản thật), vừa tránh `chunk_id` trùng lặp khi nhiều
  đoạn trích dẫn tự đánh số lại từ "Khoản 1" (mục 1, 2). Không dùng số thứ
  tự Khoản để phát hiện (vd. "đã thấy nhãn này trong Điều hiện tại") vì số
  trong đoạn trích dẫn có thể trùng ngẫu nhiên với số Khoản thật kế tiếp
  (false negative), trong khi ngoặc kép là tín hiệu cấu trúc trực tiếp của
  chính quy ước soạn thảo tạo ra tình huống này.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import yaml

from production_legal_qa_rag.chunking.models import DocumentTree, KhoanNode
from production_legal_qa_rag.chunking.patterns import (
    RE_CHUONG,
    RE_DIEU,
    RE_HEADING,
    RE_HTML_TABLE_LINE,
    RE_KHOAN_LABEL,
    RE_KHOAN_MERGED,
    RE_MUC,
    RE_PHAN,
    RE_TABLE_LINE,
)

_RE_BLOCK_SPLIT = re.compile(r"\n{2,}")


def _split_front_matter(text: str) -> tuple[dict[str, object], str]:
    """Tách khối YAML front matter (đã sinh bởi `formatting/frontmatter.py`)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            yaml_block = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            data = yaml.safe_load(yaml_block) or {}
            return (data if isinstance(data, dict) else {}), body
    return {}, text


def _split_blocks(body: str) -> list[str]:
    """Tách thân markdown thành các block theo dòng trống, đã `strip()`.

    `_split_front_matter` nối phần thân bằng ``"\\n".join(...)``, có thể để
    lại 1 ký tự "\\n" thừa ở đầu thân (khi không có đoạn mở đầu trước heading
    đầu tiên). "\\n" đơn lẻ đó không đủ để `_RE_BLOCK_SPLIT` tách ra, nên
    dính vào block đầu tiên và làm hỏng `RE_HEADING.match` (dùng ``^``, không
    ``MULTILINE``). `strip()` từng block loại bỏ khoảng trắng thừa này (và
    mọi khoảng trắng đầu/cuối tương tự) trước khi so khớp heading.
    """
    blocks = []
    for block in _RE_BLOCK_SPLIT.split(body):
        stripped = block.strip()
        if stripped:
            blocks.append(stripped)
    return blocks


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
    doc_prefix: str,
    phan: str | None,
    chuong: str | None,
    muc: str | None,
    dieu: str | None,
) -> str:
    parts = [doc_prefix, phan, chuong, muc, dieu]
    return " - ".join(part for part in parts if part)


def parse_markdown(path: str | Path) -> DocumentTree:
    """Đọc 1 file markdown, dựng `DocumentTree` chứa toàn bộ `KhoanNode`.

    Args:
        path: Đường dẫn tới file `.md` nguồn (output của `formatting/`).

    Returns:
        Cây breadcrumb -> Khoản, sẵn sàng cho `splitter.split_khoan`.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    front_matter, body = _split_front_matter(raw)

    so_hieu = front_matter.get("so_hieu")
    ten_van_ban = front_matter.get("ten_van_ban")
    source_document = str(so_hieu) if so_hieu else path.stem
    if ten_van_ban and so_hieu:
        doc_prefix = f"{ten_van_ban} ({so_hieu})"
    else:
        doc_prefix = str(ten_van_ban) if ten_van_ban else source_document

    khoans: list[KhoanNode] = []

    phan: str | None = None
    chuong: str | None = None
    muc: str | None = None
    dieu: str | None = None
    dieu_seen = False
    khoan_number: str | None = None
    prefix = doc_prefix
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

    for block in _split_blocks(body):
        heading_match = RE_HEADING.match(block)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()

            if level == 5 and quote_depth > 0:
                # Khoản lồng trong đoạn trích dẫn (xem docstring đầu file):
                # không phải cấu trúc thật của văn bản đang parse -- gộp làm
                # văn bản thường vào Khoản thật đang mở, không flush, không
                # đổi khoan_number/breadcrumb.
                paragraphs.append(text)
                continue

            flush()
            paragraphs = []
            tables = []
            if level != 5:
                quote_depth = 0

            if level == 1:
                phan = _phan_segment(text)
                chuong = muc = dieu = None
                dieu_seen = False
                khoan_number = None
            elif level == 2:
                chuong = _chuong_segment(text)
                muc = dieu = None
                dieu_seen = False
                khoan_number = None
            elif level == 3:
                muc = _muc_segment(text)
                dieu = None
                dieu_seen = False
                khoan_number = None
            elif level == 4:
                dieu = _dieu_segment(text)
                dieu_seen = True
                khoan_number = None
            elif level == 5:
                label_match = RE_KHOAN_LABEL.match(text)
                merged_match = None if label_match else RE_KHOAN_MERGED.match(text)
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
                    # và chặn riêng file đó (không âm thầm ghi đè), thay vì
                    # chặn ở đây ngay từ ca đầu (vốn vô hại 1 mình).
                    khoan_number = None

            prefix = _build_prefix(doc_prefix, phan, chuong, muc, dieu)
            continue

        stripped = block.strip()
        quote_depth = max(quote_depth + stripped.count("“") - stripped.count("”"), 0)
        if stripped.startswith(">"):
            continue  # blockquote chú thích sửa đổi — không phải nội dung Khoản
        if RE_TABLE_LINE.match(stripped) or RE_HTML_TABLE_LINE.match(stripped):
            tables.append(block)
            continue
        paragraphs.append(block)

    flush()

    if quote_depth != 0:
        # Cơ chế quote_depth (xem docstring đầu file) chỉ an toàn trên corpus
        # hiện tại vì đã xác nhận thủ công ngoặc kép cân bằng ở cả 6 file
        # `data/markdown/*.md`. Cảnh báo runtime ở đây để phát hiện SỚM văn
        # bản mới có ngoặc kép không cân (thay vì âm thầm nuốt mất 1 Khoản
        # thật, xem test_HAN_CHE_... trong test_chunking_parser.py).
        warnings.warn(
            f"{path}: ngoặc kép trích dẫn không cân (quote_depth={quote_depth} "
            "cuối file) -- có thể đã bỏ sót nội dung Khoản thật, xem "
            "parser.py docstring mục 'Khoản lồng trong Khoản'.",
            stacklevel=2,
        )

    return DocumentTree(source_document=source_document, khoans=khoans)
