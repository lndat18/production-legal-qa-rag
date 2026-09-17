"""Front matter: nội dung trước heading cấu trúc đầu tiên, chuyển bằng Groq.

Xác định biên là 100% deterministic (``find_boundary``, dùng
``patterns.is_structural``, không đổi so với bản cũ). Chuyển đổi NỘI DUNG là
Groq (``llm_client.convert_to_markdown``) — không còn trích field
(``so_hieu``/``loai_van_ban``/``ten_van_ban``/...), không sinh YAML (mục 1.1
spec). Front matter luôn nhỏ trên corpus thực tế (mục 1.2 spec) nên thường
chỉ tạo 1 chunk, nhưng vẫn đi qua ``docx_reader.chunk_blocks_for_llm`` để
nhất quán với back matter và an toàn nếu front matter lớn hơn dự kiến. Bất
kỳ chunk nào lỗi/timeout → bỏ qua toàn bộ front matter (không render), phát
``QcWarning`` — không có fallback regex nào.

**[CẬP NHẬT 2026-09-17]** Đúng 1 dòng — tên đầy đủ văn bản — được xác định
deterministic bằng ``find_title`` và tự chèn thành heading ``# ...``, KHÔNG
qua Groq (Groq không nhất quán khi được giao tự quyết định chèn heading).
``convert_frontmatter`` cắt ``blocks`` quanh dòng đó, convert phần trước/sau
độc lập qua Groq.

**[CẬP NHẬT 2026-09-17, sửa lại lần 2 — heuristic đơn giản hoá theo cấu trúc
thật]** Bản đầu của ``find_title`` dựa vào bold + toàn chữ hoa của chính
dòng tên văn bản — kiểm tra trên toàn bộ 6 file ``data/raw/*.docx`` thật cho
thấy dòng này không phải lúc nào cũng bold (2/6 file dạng Nghị định). Nay
đổi sang định vị bằng vị trí tương đối so với dòng loại văn bản
(``patterns.RE_DOC_TYPE_ONLY``): block tên văn bản luôn là block paragraph
ngay sau dòng loại văn bản, không cần điều kiện bold/viết hoa riêng. Xem
formatting_spec.md mục 1.1.
"""

from __future__ import annotations

from production_legal_qa_rag.formatting import llm_client
from production_legal_qa_rag.formatting.docx_reader import (
    Block,
    chunk_blocks_for_llm,
    serialize_blocks_for_llm,
)
from production_legal_qa_rag.formatting.models import QcWarning
from production_legal_qa_rag.formatting.patterns import RE_DOC_TYPE_ONLY, is_structural

_FRONTMATTER_PROMPT_TEMPLATE = """\
Bạn là trợ lý chuyển đổi văn bản pháp luật Việt Nam từ định dạng gốc sang \
Markdown thuần.

Đoạn văn bản dưới đây là phần MỞ ĐẦU của một văn bản pháp luật (quốc hiệu, \
số hiệu, ngày ban hành, tên văn bản, các dòng "Căn cứ...", câu ban hành mở \
đầu), đứng TRƯỚC Chương/Mục/Điều/Khoản đầu tiên. Hãy chuyển nguyên văn đoạn \
này sang Markdown, yêu cầu:
- Giữ đúng thứ tự, giữ TOÀN BỘ nội dung, không tóm tắt, không thêm/bớt.
- Giữ in đậm/in nghiêng nếu bản gốc có (đã được đánh dấu sẵn bằng ** và *).
- KHÔNG dùng heading Markdown (không có dòng bắt đầu bằng #), kể cả cho tên \
loại văn bản (vd. "NGHỊ ĐỊNH") hay tên đầy đủ văn bản — chỉ dùng đoạn văn \
thường và in đậm/nghiêng, giống cách văn bản gốc trình bày.
- Bảng quốc hiệu (nếu có, dạng 2 cột CHÍNH PHỦ / CỘNG HÒA XÃ HỘI CHỦ NGHĨA \
VIỆT NAM...) không cần giữ đúng bố cục song song — Markdown thuần không hỗ \
trợ cột, trình bày tuần tự hợp lý là đủ, miễn giữ đúng nội dung.

Văn bản:
{text}
"""


def find_boundary(blocks: list[Block]) -> int:
    """Chỉ số block đầu tiên khớp regex heading cấu trúc (mục 1.1 spec).

    Mọi block trước đó thuộc front matter. Văn bản hợp nhất liệt kê luật sửa
    đổi bằng dòng đánh số ngay trong vùng dẫn nhập ("1. Luật Nhà giáo số
    73/2025/QH15..."); ``is_structural`` không khớp Khoản nên các dòng đó
    không bị nhận nhầm thành biên.

    Args:
        blocks: Toàn bộ Block đọc được từ DOCX, theo đúng thứ tự xuất hiện.

    Returns:
        Chỉ số block đầu tiên thuộc vùng nội dung ở giữa. Bằng ``len(blocks)``
        nếu không có heading cấu trúc nào (toàn bộ văn bản là front matter).
    """
    for index, block in enumerate(blocks):
        if block.kind == "paragraph" and is_structural(block.text):
            return index
    return len(blocks)


def find_title(blocks: list[Block]) -> int | None:
    """Chỉ số block chứa tên đầy đủ văn bản, xác định deterministic.

    Định vị bằng vị trí tương đối so với dòng loại văn bản (khớp
    ``patterns.RE_DOC_TYPE_ONLY``, vd. "LUẬT", "NGHỊ ĐỊNH") — đã kiểm tra
    trên toàn bộ 6 file ``data/raw/*.docx`` thật: dòng loại văn bản và dòng
    tên/nội dung chính luôn là 2 block paragraph liên tiếp, tách biệt
    (không bao giờ gộp sẵn 1 dòng), và dòng tên không phải lúc nào cũng in
    đậm (2/6 file dạng Nghị định hoàn toàn không bold) — vì vậy heuristic
    này KHÔNG dựa vào bold/viết hoa của chính dòng tên văn bản:

    1. Tìm block đầu tiên (``kind == "paragraph"``) khớp ``RE_DOC_TYPE_ONLY``
       -- ``type_index``.
    2. Nếu ``blocks[type_index + 1]`` tồn tại, ``kind == "paragraph"`` và có
       text (khác rỗng) -- đó là block tên văn bản, trả về ``type_index + 1``.
    3. Ngược lại (không tìm thấy ``type_index``, hoặc không có block hợp lệ
       ngay sau nó) -- trả về ``None``.

    Args:
        blocks: Block front matter, đã cắt bởi ``find_boundary``.

    Returns:
        Chỉ số block tên văn bản, hoặc ``None`` nếu không tìm được dòng loại
        văn bản hoặc không có block hợp lệ ngay sau nó.
    """
    type_index: int | None = None
    for index, block in enumerate(blocks):
        if block.kind == "paragraph" and RE_DOC_TYPE_ONLY.match(block.text):
            type_index = index
            break

    if type_index is None:
        return None

    title_index = type_index + 1
    if title_index >= len(blocks):
        return None

    title_block = blocks[title_index]
    if title_block.kind != "paragraph" or not title_block.text:
        return None

    return title_index


def _convert_blocks(blocks: list[Block]) -> tuple[str, list[QcWarning]]:
    """Chia ``blocks`` thành chunk, gọi Groq tuần tự và nối kết quả.

    Hàm nội bộ dùng bởi ``convert_frontmatter`` cho cả trường hợp không tìm
    thấy tên văn bản (convert nguyên khối) lẫn trường hợp tìm thấy (convert
    ``before``/``after`` độc lập, mục 1.1 spec).

    Args:
        blocks: Danh sách Block cần convert (có thể rỗng).

    Returns:
        Cặp ``(markdown, warnings)``. ``markdown`` rỗng khi ``blocks`` rỗng
        (không gọi Groq) hoặc khi một chunk lỗi/timeout (phát
        ``llm_frontmatter_conversion_failed``, không fallback, không ghép
        phần dở dang).
    """
    if not blocks:
        return "", []

    chunks = chunk_blocks_for_llm(blocks, llm_client.get_chunk_token_limit())
    markdown_parts: list[str] = []
    for chunk in chunks:
        prompt = _FRONTMATTER_PROMPT_TEMPLATE.format(
            text=serialize_blocks_for_llm(chunk)
        )
        markdown = llm_client.convert_to_markdown(prompt)
        if markdown is None:
            return "", [QcWarning(code="llm_frontmatter_conversion_failed", detail="")]
        markdown_parts.append(markdown)

    return "\n\n".join(markdown_parts), []


def convert_frontmatter(blocks: list[Block]) -> tuple[str, list[QcWarning]]:
    """Chuyển vùng front matter (block trước heading đầu tiên) sang markdown.

    Tìm dòng tên văn bản bằng ``find_title`` (mục 1.1 spec). Nếu tìm thấy:
    cắt ``blocks`` thành ``before``/block tên văn bản/``after``, convert
    ``before`` và ``after`` độc lập qua Groq (``_convert_blocks``), rồi chèn
    ``# <nguyên văn dòng tên văn bản>`` xen giữa — dòng heading luôn được
    chèn, kể cả khi ``before``/``after`` lỗi và bị bỏ qua. Nếu không tìm
    thấy: convert nguyên khối ``blocks`` như cũ (không có heading), phát
    thêm ``QcWarning`` (``frontmatter_title_not_found``).

    Args:
        blocks: Block front matter, đã cắt bởi ``find_boundary``.

    Returns:
        Cặp ``(markdown, warnings)``. ``markdown`` rỗng khi không có front
        matter (``blocks`` rỗng — không gọi Groq).
    """
    if not blocks:
        return "", []

    title_index = find_title(blocks)
    if title_index is None:
        markdown, warnings = _convert_blocks(blocks)
        return markdown, [
            *warnings,
            QcWarning(code="frontmatter_title_not_found", detail=""),
        ]

    before_blocks = blocks[:title_index]
    title_block = blocks[title_index]
    after_blocks = blocks[title_index + 1 :]

    before_markdown, before_warnings = _convert_blocks(before_blocks)
    after_markdown, after_warnings = _convert_blocks(after_blocks)

    parts: list[str] = []
    if before_markdown:
        parts.append(before_markdown)
    parts.append(f"# {title_block.text}")
    if after_markdown:
        parts.append(after_markdown)

    return "\n\n".join(parts), [*before_warnings, *after_warnings]
