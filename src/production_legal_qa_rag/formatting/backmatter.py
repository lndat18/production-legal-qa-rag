"""Back matter: nội dung sau bảng chữ ký cuối cùng (nếu có), chuyển bằng Groq.

Đổi tên từ ``footnotes.py``, thiết kế lại hoàn toàn (mục 1.1 spec): không
còn khớp marker ``[n]`` hay khôi phục inline vào Khoản/Điều — các hàm cũ
(``find_region_start``, ``parse_region``, ``strip_markers``, ``strip_all``,
``render_blockquote``) đã xoá. Biên back matter vẫn xác định 100%
deterministic bằng ``tables.is_signature_table`` (không đổi); nội dung được
chia chunk (mục 1.2 spec — back matter một số văn bản vượt xa TPM free tier
nếu gửi nguyên khối) rồi gửi từng chunk cho Groq, không trích field, không
khớp lại marker với chú thích tương ứng.

**[CẬP NHẬT 2026-09-17, sửa lại — dispatch đồng thời 2 key, mục 1.3]** Tách
"dựng job" (``build_prompts``, thuần, không I/O) khỏi "gọi Groq" (nay do
``pipeline.py`` điều phối 1 lần qua ``llm_client.convert_chunks_concurrently``,
gộp chung với job front matter) khỏi "ghép kết quả" (``assemble``, thuần,
không I/O). Module này không còn tự gọi Groq.
"""

from __future__ import annotations

from production_legal_qa_rag.formatting import llm_client, tables
from production_legal_qa_rag.formatting.docx_reader import (
    Block,
    chunk_blocks_for_llm,
    serialize_blocks_for_llm,
)
from production_legal_qa_rag.formatting.models import QcWarning
from production_legal_qa_rag.formatting.patterns import escape_setext_underline

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
- Nếu gặp dòng chỉ toàn ký tự gạch ngang/gạch bằng (`-`/`=`, vd. đường kẻ \
trang trí), PHẢI cách dòng text phía trên bằng 1 dòng trống, không đặt liền \
kề — tránh vô tình tạo thành setext heading.

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


def build_prompts(blocks: list[Block]) -> list[str]:
    """Dựng danh sách prompt Groq cho back matter (thuần, không gọi Groq).

    Chia ``blocks`` thành chunk (``docx_reader.chunk_blocks_for_llm``, mục
    1.2 spec — bắt buộc vì back matter một số văn bản, vd. Luật bảo hiểm y
    tế, vượt xa TPM free tier nếu gửi nguyên khối) rồi dựng 1 prompt cho mỗi
    chunk.

    Args:
        blocks: Block back matter, đã tách bởi ``split_backmatter``.

    Returns:
        Danh sách prompt theo đúng thứ tự chunk, rỗng khi không có back
        matter (``blocks`` rỗng — không có gì để gọi Groq, mục 1.1 spec).
    """
    if not blocks:
        return []

    chunks = chunk_blocks_for_llm(blocks, llm_client.get_chunk_token_limit())
    return [
        _BACKMATTER_PROMPT_TEMPLATE.format(text=serialize_blocks_for_llm(chunk))
        for chunk in chunks
    ]


def assemble(chunk_results: list[str | None]) -> tuple[str, list[QcWarning]]:
    """Nối kết quả Groq của back matter thành markdown cuối cùng (thuần).

    Nối theo đúng thứ tự chunk gốc, cách nhau 1 dòng trống, rồi áp
    ``patterns.escape_setext_underline`` (mục 1.1 spec, bug setext heading).

    Args:
        chunk_results: Kết quả Groq (``str | None``) theo đúng thứ tự chunk
            gốc, đã được ``pipeline.py`` cắt ra từ kết quả
            ``llm_client.convert_chunks_concurrently``.

    Returns:
        Cặp ``(markdown, warnings)``. ``markdown`` rỗng khi không có back
        matter (``chunk_results`` rỗng — không gọi Groq, không phát warning,
        mục 1.1 spec) hoặc khi có ít nhất 1 job lỗi (``None``) — phát
        ``QcWarning`` (``llm_backmatter_conversion_failed``), không ghép
        phần dở dang.
    """
    if not chunk_results:
        return "", []
    if any(result is None for result in chunk_results):
        return "", [QcWarning(code="llm_backmatter_conversion_failed", detail="")]

    markdown_parts: list[str] = [
        result for result in chunk_results if result is not None
    ]
    markdown = "\n\n".join(markdown_parts)
    return escape_setext_underline(markdown), []
