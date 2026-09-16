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
"""

from __future__ import annotations

from production_legal_qa_rag.formatting import llm_client
from production_legal_qa_rag.formatting.docx_reader import (
    Block,
    chunk_blocks_for_llm,
    serialize_blocks_for_llm,
)
from production_legal_qa_rag.formatting.models import QcWarning
from production_legal_qa_rag.formatting.patterns import is_structural

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


def convert_frontmatter(blocks: list[Block]) -> tuple[str, list[QcWarning]]:
    """Chuyển vùng front matter (block trước heading đầu tiên) sang markdown.

    Chia ``blocks`` thành chunk (``docx_reader.chunk_blocks_for_llm``, mục
    1.2 spec), gọi Groq tuần tự cho từng chunk rồi nối kết quả theo đúng thứ
    tự, cách nhau 1 dòng trống. Bất kỳ chunk nào lỗi → toàn bộ front matter
    bị bỏ qua (không ghép phần dở dang).

    Args:
        blocks: Block front matter, đã cắt bởi ``find_boundary``.

    Returns:
        Cặp ``(markdown, warnings)``. ``markdown`` rỗng khi không có front
        matter (``blocks`` rỗng — không gọi Groq) hoặc khi một chunk lỗi/
        timeout (phát ``llm_frontmatter_conversion_failed``, không fallback).
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
