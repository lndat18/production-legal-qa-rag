"""Back matter: nội dung sau bảng chữ ký cuối cùng (nếu có), chuyển bằng Gemini.

Đổi tên từ ``footnotes.py``, thiết kế lại hoàn toàn (mục 1.1 spec): không
còn khớp marker ``[n]`` hay khôi phục inline vào Khoản/Điều — các hàm cũ
(``find_region_start``, ``parse_region``, ``strip_markers``, ``strip_all``,
``render_blockquote``) đã xoá. Biên back matter vẫn xác định 100%
deterministic bằng ``tables.is_signature_table`` (không đổi); nội dung được
gửi nguyên khối cho Gemini, không trích field, không khớp lại marker với
chú thích tương ứng.
"""

from __future__ import annotations

from production_legal_qa_rag.formatting import llm_client, tables
from production_legal_qa_rag.formatting.docx_reader import (
    Block,
    serialize_blocks_for_llm,
)
from production_legal_qa_rag.formatting.models import QcWarning

_BACKMATTER_PROMPT_TEMPLATE = """\
Bạn là trợ lý chuyển đổi văn bản pháp luật Việt Nam từ định dạng gốc sang \
Markdown thuần.

Đoạn văn bản dưới đây nằm ở CUỐI văn bản, sau khối chữ ký — có thể là chú \
thích sửa đổi (thường đánh số dạng "[1]", "[2]"...), phần "Nơi nhận:", hoặc \
khối xác thực văn bản hợp nhất (vd. "XÁC THỰC VĂN BẢN HỢP NHẤT" kèm danh \
sách căn cứ sửa đổi từng điều). Hãy chuyển nguyên văn đoạn này sang Markdown, \
yêu cầu:
- Giữ đúng thứ tự, giữ TOÀN BỘ nội dung, không tóm tắt, không thêm/bớt.
- Giữ in đậm/in nghiêng nếu bản gốc có (đã được đánh dấu sẵn bằng ** và *).
- KHÔNG dùng heading Markdown (không có dòng bắt đầu bằng #) — chỉ dùng đoạn \
văn thường và in đậm/nghiêng.

Văn bản:
{text}
"""


def find_boundary(blocks: list[Block]) -> int | None:
    """Chỉ số bảng chữ ký CUỐI CÙNG trong ``blocks``, hoặc ``None`` nếu không có.

    Đây là block đánh dấu ranh giới: mọi block sau nó (nếu còn) thuộc back
    matter (mục 1.1 spec). Lấy chỉ số cuối cùng vì một văn bản có thể có
    nhiều bảng dạng chữ ký (vd. Phụ lục có bảng ký riêng).

    Args:
        blocks: Block thuộc vùng SAU biên front matter, theo đúng thứ tự gốc.
    """
    indices = [
        index for index, block in enumerate(blocks) if tables.is_signature_table(block)
    ]
    return indices[-1] if indices else None


def split_backmatter(
    blocks: list[Block],
) -> tuple[list[Block], list[Block], list[QcWarning]]:
    """Tách ``blocks`` (sau biên front matter) thành vùng giữa và back matter.

    Bảng chữ ký bị loại khỏi vùng giữa — chỉ dùng để tìm biên, không phải nội
    dung QA hữu ích (``dropped_noi_nhan_table``, kế thừa hành vi bản cũ).

    Args:
        blocks: Block thuộc vùng SAU biên front matter.

    Returns:
        Bộ ba ``(block còn lại của vùng giữa, block back matter, cảnh báo QC)``.
        Back matter rỗng khi không có bảng chữ ký nào hoặc không còn block
        nào sau bảng chữ ký cuối cùng — cả hai đều là trạng thái hợp lệ.
    """
    boundary = find_boundary(blocks)
    if boundary is None:
        return blocks, [], []

    warnings = [QcWarning(code="dropped_noi_nhan_table", detail=f"block {boundary}")]
    return blocks[:boundary], blocks[boundary + 1 :], warnings


def convert_backmatter(blocks: list[Block]) -> tuple[str, list[QcWarning]]:
    """Chuyển vùng back matter sang markdown.

    Args:
        blocks: Block back matter, đã tách bởi ``split_backmatter``.

    Returns:
        Cặp ``(markdown, warnings)``. ``markdown`` rỗng khi không có back
        matter (``blocks`` rỗng — không gọi Gemini, không phát warning, mục
        1.1 spec) hoặc khi Gemini lỗi/timeout (phát
        ``llm_backmatter_conversion_failed``, không fallback).
    """
    if not blocks:
        return "", []

    prompt = _BACKMATTER_PROMPT_TEMPLATE.format(text=serialize_blocks_for_llm(blocks))
    markdown = llm_client.convert_to_markdown(prompt)
    if markdown is None:
        return "", [QcWarning(code="llm_backmatter_conversion_failed", detail="")]
    return markdown, []
