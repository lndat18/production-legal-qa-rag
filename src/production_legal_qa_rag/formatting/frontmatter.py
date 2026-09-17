"""Front matter: nội dung trước heading cấu trúc đầu tiên, chuyển bằng Groq.

Xác định biên là 100% deterministic (``find_boundary``, dùng
``patterns.is_structural``, không đổi so với bản cũ). Chuyển đổi NỘI DUNG là
Groq — không còn trích field (``so_hieu``/``loai_van_ban``/``ten_van_ban``/
...), không sinh YAML (mục 1.1 spec). Front matter luôn nhỏ trên corpus thực
tế (mục 1.2 spec) nên thường chỉ tạo 1 chunk, nhưng vẫn đi qua
``docx_reader.chunk_blocks_for_llm`` để nhất quán với back matter và an toàn
nếu front matter lớn hơn dự kiến. Bất kỳ chunk nào lỗi/timeout → bỏ qua toàn
bộ front matter (không render), phát ``QcWarning`` — không có fallback regex
nào.

**[CẬP NHẬT 2026-09-17]** Đúng 1 dòng — tên đầy đủ văn bản — được xác định
deterministic bằng ``find_title`` và tự chèn thành heading ``# ...``, KHÔNG
qua Groq (Groq không nhất quán khi được giao tự quyết định chèn heading).

**[CẬP NHẬT 2026-09-17, sửa lại lần 2 — heuristic đơn giản hoá theo cấu trúc
thật]** Bản đầu của ``find_title`` dựa vào bold + toàn chữ hoa của chính
dòng tên văn bản — kiểm tra trên toàn bộ 6 file ``data/raw/*.docx`` thật cho
thấy dòng này không phải lúc nào cũng bold (2/6 file dạng Nghị định). Nay
đổi sang định vị bằng vị trí tương đối so với dòng loại văn bản
(``patterns.RE_DOC_TYPE_ONLY``): block tên văn bản luôn là block paragraph
ngay sau dòng loại văn bản, không cần điều kiện bold/viết hoa riêng. Xem
formatting_spec.md mục 1.1.

**[CẬP NHẬT 2026-09-17, sửa lại — dispatch đồng thời 2 key, mục 1.3]** Tách
"dựng job" (``build_prompts``, thuần, không I/O) khỏi "gọi Groq" (nay do
``pipeline.py`` điều phối 1 lần qua ``llm_client.convert_chunks_concurrently``,
gộp chung với job back matter để cả 2 worker luôn bận) khỏi "ghép kết quả"
(``assemble``, thuần, không I/O). Module này không còn tự gọi Groq.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from production_legal_qa_rag.formatting import llm_client
from production_legal_qa_rag.formatting.docx_reader import (
    Block,
    chunk_blocks_for_llm,
    serialize_blocks_for_llm,
)
from production_legal_qa_rag.formatting.models import QcWarning
from production_legal_qa_rag.formatting.patterns import (
    RE_DOC_TYPE_ONLY,
    escape_setext_underline,
    is_structural,
)

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
- Nếu gặp dòng chỉ toàn ký tự gạch ngang/gạch bằng (`-`/`=`, vd. đường kẻ \
trang trí ngay dưới tên cơ quan), PHẢI cách dòng text phía trên bằng 1 dòng \
trống, không đặt liền kề — tránh vô tình tạo thành setext heading.

Văn bản:
{text}
"""


@dataclass(frozen=True, slots=True)
class FrontMatterJobs:
    """Job cần gửi Groq cho front matter (thuần, chưa gọi API — mục 1.3 spec).

    Kết quả của ``build_prompts``: ``pipeline.py`` gộp ``before_prompts`` +
    ``after_prompts`` (cùng job back matter) thành 1 danh sách duy nhất, gọi
    ``llm_client.convert_chunks_concurrently`` 1 lần, rồi cắt kết quả trả về
    theo đúng ranh giới (độ dài) 2 danh sách này để truyền vào ``assemble``.
    """

    title_text: str | None
    before_prompts: list[str] = field(default_factory=list)
    after_prompts: list[str] = field(default_factory=list)


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


def _build_chunk_prompts(blocks: list[Block]) -> list[str]:
    """Chia ``blocks`` thành chunk (mục 1.2 spec) và dựng prompt Groq cho mỗi chunk.

    Hàm thuần, không I/O — dùng bởi ``build_prompts`` cho cả trường hợp không
    tìm thấy tên văn bản (toàn bộ ``blocks`` thành 1 danh sách prompt duy
    nhất) lẫn trường hợp tìm thấy (``before``/``after`` độc lập, mục 1.1
    spec).

    Args:
        blocks: Danh sách Block cần dựng prompt (có thể rỗng).

    Returns:
        Danh sách prompt, rỗng khi ``blocks`` rỗng.
    """
    if not blocks:
        return []

    chunks = chunk_blocks_for_llm(blocks, llm_client.get_chunk_token_limit())
    return [
        _FRONTMATTER_PROMPT_TEMPLATE.format(text=serialize_blocks_for_llm(chunk))
        for chunk in chunks
    ]


def build_prompts(blocks: list[Block]) -> FrontMatterJobs:
    """Dựng danh sách prompt Groq cho front matter (thuần, không gọi Groq).

    Tìm dòng tên văn bản bằng ``find_title`` (mục 1.1 spec). Nếu tìm thấy:
    cắt ``blocks`` thành ``before`` (gồm cả dòng loại văn bản)/block tên văn
    bản/``after``, mỗi phần chunk + dựng prompt độc lập. Nếu không tìm thấy:
    toàn bộ ``blocks`` thành 1 danh sách prompt duy nhất (``before_prompts``),
    ``after_prompts`` rỗng, ``title_text=None``.

    Args:
        blocks: Block front matter, đã cắt bởi ``find_boundary``.

    Returns:
        ``FrontMatterJobs`` — rỗng hoàn toàn (không prompt nào, ``title_text``
        ``None``) khi ``blocks`` rỗng (không có front matter).
    """
    if not blocks:
        return FrontMatterJobs(title_text=None, before_prompts=[], after_prompts=[])

    title_index = find_title(blocks)
    if title_index is None:
        return FrontMatterJobs(
            title_text=None,
            before_prompts=_build_chunk_prompts(blocks),
            after_prompts=[],
        )

    before_blocks = blocks[:title_index]
    title_block = blocks[title_index]
    after_blocks = blocks[title_index + 1 :]

    return FrontMatterJobs(
        title_text=title_block.text,
        before_prompts=_build_chunk_prompts(before_blocks),
        after_prompts=_build_chunk_prompts(after_blocks),
    )


def _assemble_results(
    results: list[str | None],
) -> tuple[str, list[QcWarning]]:
    """Nối kết quả Groq của 1 phần (``before`` hoặc ``after``) theo thứ tự chunk.

    Args:
        results: Kết quả Groq (``str | None``) theo đúng thứ tự chunk gốc,
            đã được ``pipeline.py`` cắt ra từ kết quả
            ``llm_client.convert_chunks_concurrently``.

    Returns:
        Cặp ``(markdown, warnings)``. ``markdown`` rỗng khi ``results`` rỗng
        (không có chunk nào — phần này không tồn tại) hoặc khi có ít nhất 1
        job lỗi (``None``) — phát ``QcWarning``
        (``llm_frontmatter_conversion_failed``), không ghép phần dở dang.
    """
    if not results:
        return "", []
    if any(result is None for result in results):
        return "", [QcWarning(code="llm_frontmatter_conversion_failed", detail="")]

    markdown_parts: list[str] = [result for result in results if result is not None]
    return "\n\n".join(markdown_parts), []


def assemble(
    title_text: str | None,
    before_results: list[str | None],
    after_results: list[str | None],
) -> tuple[str, list[QcWarning]]:
    """Ghép kết quả Groq của front matter thành markdown cuối cùng (thuần).

    Chèn ``# <nguyên văn title_text>`` xen giữa ``before``/``after`` nếu tìm
    thấy tên văn bản — dòng heading luôn được chèn, kể cả khi ``before``/
    ``after`` lỗi và bị bỏ qua. Không có ``title_text`` (front matter không
    rỗng nhưng không tìm được dòng loại văn bản) → phát ``QcWarning``
    (``frontmatter_title_not_found``). Áp ``patterns.escape_setext_underline``
    lên kết quả cuối cùng (mục 1.1 spec, bug setext heading).

    Args:
        title_text: ``FrontMatterJobs.title_text`` từ ``build_prompts``.
        before_results: Kết quả Groq của ``before_prompts``, cùng thứ tự.
        after_results: Kết quả Groq của ``after_prompts``, cùng thứ tự.

    Returns:
        Cặp ``(markdown, warnings)``. ``markdown`` rỗng khi không có front
        matter nào (``title_text is None`` và cả 2 danh sách kết quả đều
        rỗng — tương ứng ``blocks`` rỗng ở ``build_prompts``, không phải lỗi,
        không phát warning).
    """
    if title_text is None and not before_results and not after_results:
        return "", []

    before_markdown, before_warnings = _assemble_results(before_results)
    after_markdown, after_warnings = _assemble_results(after_results)

    parts: list[str] = []
    if before_markdown:
        parts.append(before_markdown)
    if title_text is not None:
        parts.append(f"# {title_text}")
    if after_markdown:
        parts.append(after_markdown)

    markdown = "\n\n".join(parts)
    if markdown:
        markdown = escape_setext_underline(markdown)

    warnings = [*before_warnings, *after_warnings]
    if title_text is None:
        warnings.append(QcWarning(code="frontmatter_title_not_found", detail=""))

    return markdown, warnings
