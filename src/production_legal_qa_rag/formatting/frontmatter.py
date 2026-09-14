"""Trích xuất metadata cấp văn bản (số hiệu, loại văn bản, ngày...).

Đường chính đọc **giá trị** field front matter là LLM (``llm_client``, schema
``FrontMatterExtraction`` — formatting_spec.md mục 1, 6): ``build_frontmatter``
gọi ``extract_frontmatter_llm`` trước, dùng giá trị LLM trả về. Các hàm regex
ở dưới (``extract_quoc_hieu`` đọc từ bảng quốc hiệu tách ra bởi
``tables.triage_tables``; ``_find_ten_van_ban``/``_find_loai_van_ban``/
``_find_ngay_hieu_luc`` dò trên thân văn bản đã gỡ marker chú thích) giữ
nguyên logic, đổi vai trò thành **baseline**: dùng để fallback khi LLM trả
``None`` cho field đó, và để so sánh phát ``QcWarning`` khi hai nguồn lệch
nhau.
"""

from __future__ import annotations

import re

from production_legal_qa_rag.formatting import llm_client
from production_legal_qa_rag.formatting.docx_reader import Block
from production_legal_qa_rag.formatting.models import (
    FrontMatter,
    FrontMatterExtraction,
    QcWarning,
)
from production_legal_qa_rag.formatting.patterns import (
    RE_BAN_HANH,
    RE_DIEU,
    RE_DOC_TYPE,
    RE_HIEU_LUC,
    RE_NGAY,
    RE_SO_HIEU,
    RE_TEN_SO,
)
from production_legal_qa_rag.formatting.tables import parse_pipe_table

# Số block đầu văn bản còn được coi là vùng dẫn nhập khi dò loại văn bản.
_DOC_TYPE_SEARCH_LIMIT = 20

# Số đoạn sau heading Điều hiệu lực còn được dò ngày hiệu lực.
_HIEU_LUC_LOOKAHEAD = 3

# Field bắt buộc: thiếu ở LLM luôn phát `llm_frontmatter_extraction_failed`
# và fallback baseline, kể cả khi extraction tổng thể không None.
_REQUIRED_LLM_FIELDS = ("so_hieu", "ngay_hieu_luc")
_OPTIONAL_LLM_FIELDS = (
    "loai_van_ban",
    "ten_van_ban",
    "co_quan_ban_hanh",
    "ngay_ban_hanh",
)

# Số ký tự tối đa đưa vào prompt LLM — chốt chặn cuối cùng, phòng trường hợp
# `_compose_llm_prompt_text` (đã tự giới hạn theo đoạn dẫn nhập + ngữ cảnh
# quanh "hiệu lực") vẫn tạo ra văn bản dài bất thường. Giới hạn thấp vì hạn
# mức Groq free tier chỉ 8000 token/phút — gửi cả thân văn bản thật (hàng
# nghìn token) từng bị từ chối 413 "Request too large" khi thử thật.
_LLM_BODY_CHAR_LIMIT = 6_000

# Số đoạn dẫn nhập đầu văn bản đưa vào prompt — đủ chứa quốc hiệu/tiêu đề,
# không cần quét sâu vì các field này (trừ ngày hiệu lực) luôn nằm ở đây.
_LLM_INTRO_PARAGRAPH_LIMIT = 30

# Số đoạn lấy thêm quanh mỗi lần xuất hiện từ khóa "hiệu lực" — ngày hiệu lực
# thường nằm ở Điều cuối văn bản, ngoài vùng dẫn nhập.
_LLM_HIEU_LUC_WINDOW = 2

_FRONTMATTER_PROMPT_TEMPLATE = """\
Bạn là trợ lý trích xuất metadata từ văn bản pháp luật Việt Nam.

Đọc văn bản dưới đây và trích các giá trị: số hiệu văn bản, loại văn bản, \
tên đầy đủ của văn bản, cơ quan ban hành, ngày ban hành và ngày văn bản có \
hiệu lực thi hành. Nếu không tìm thấy giá trị nào trong văn bản, để trống \
(null), không suy đoán hay bịa ra giá trị.

Văn bản:
{text}
"""


def _cell(rows: list[list[str]], row: int, column: int) -> str:
    if row < len(rows) and column < len(rows[row]):
        return rows[row][column]
    return ""


def _iso_date(match: re.Match[str]) -> str:
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

    rows = parse_pipe_table(table_block.text)

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
    phải của luật sửa đổi mới nhất. Đó là đúng — Luật Bảo hiểm y tế hợp nhất
    trả về 2009-07-01 của luật 2008. Đừng "sửa" thành ngày mới nhất.
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


def _hieu_luc_context_indices(blocks: list[Block]) -> set[int]:
    """Chỉ số các đoạn quanh mỗi lần xuất hiện từ khóa "hiệu lực".

    Không gửi toàn bộ thân văn bản cho LLM (từng bị Groq trả 413 "Request
    too large" khi thử thật — hạn mức free tier chỉ 8000 token/phút): chỉ
    trích phần ngữ cảnh nhỏ quanh từ khóa, đủ cho LLM đọc ra ngày hiệu lực
    mà không kéo theo toàn văn.
    """
    matches = {
        index
        for index, block in enumerate(blocks)
        if block.kind == "paragraph" and "hiệu lực" in block.text.casefold()
    }
    return {
        i
        for index in matches
        for i in range(
            max(0, index - _LLM_HIEU_LUC_WINDOW),
            min(len(blocks), index + _LLM_HIEU_LUC_WINDOW + 1),
        )
    }


def _compose_llm_prompt_text(
    quoc_hieu_block: Block | None, body_blocks: list[Block]
) -> str:
    """Ghép nội dung gốc thành phần văn bản đưa vào prompt LLM.

    Chỉ trích phần liên quan tới front matter (không gửi toàn văn, xem
    ``_hieu_luc_context_indices``): bảng quốc hiệu, đoạn dẫn nhập đầu văn
    bản (chứa số hiệu/tên/loại văn bản) và ngữ cảnh quanh từ khóa "hiệu lực"
    (ngày hiệu lực thường ở Điều cuối văn bản).
    """
    selected_indices = set(range(min(_LLM_INTRO_PARAGRAPH_LIMIT, len(body_blocks))))
    selected_indices |= _hieu_luc_context_indices(body_blocks)

    lines: list[str] = []
    if quoc_hieu_block is not None:
        lines.append(quoc_hieu_block.text)
    lines.extend(
        body_blocks[index].text
        for index in sorted(selected_indices)
        if body_blocks[index].kind == "paragraph"
    )
    return "\n".join(lines)[:_LLM_BODY_CHAR_LIMIT]


def extract_frontmatter_llm(
    quoc_hieu_block: Block | None, body_blocks: list[Block]
) -> FrontMatterExtraction | None:
    """Gọi LLM đọc giá trị field front matter, làm đường chính (mục 1, 6).

    Trả ``None`` khi ``llm_client.extract_structured`` thất bại (lỗi/timeout/
    hết số lần thử) — caller (``build_frontmatter``) tự fallback baseline.
    """
    prompt = _FRONTMATTER_PROMPT_TEMPLATE.format(
        text=_compose_llm_prompt_text(quoc_hieu_block, body_blocks)
    )
    result = llm_client.extract_structured(prompt, FrontMatterExtraction)
    if not isinstance(result, FrontMatterExtraction):
        return None
    return result


def _resolve_frontmatter_fields(
    extraction: FrontMatterExtraction | None,
    baseline: dict[str, str | None],
) -> tuple[dict[str, str | None], list[QcWarning]]:
    """Chọn giá trị cuối cùng cho mỗi field front matter và phát QcWarning.

    LLM là đường chính: dùng giá trị LLM khi có. Field bắt buộc (số hiệu,
    ngày hiệu lực) thiếu ở LLM luôn fallback baseline và phát
    ``llm_frontmatter_extraction_failed``; field không bắt buộc thiếu ở LLM
    fallback baseline lặng lẽ (không đổi kết quả trạng thái field từ trước
    tới nay: field vẫn thiếu thì ``missing_optional_frontmatter_field`` ở
    ``build_frontmatter`` vẫn báo như cũ). Field có cả hai giá trị nhưng
    lệch nhau vẫn giữ giá trị LLM, chỉ phát ``llm_frontmatter_mismatch`` —
    không tự động chọn baseline khi có lệch (mục 7 spec).
    """
    warnings: list[QcWarning] = []
    resolved: dict[str, str | None] = {}

    for field in (*_REQUIRED_LLM_FIELDS, *_OPTIONAL_LLM_FIELDS):
        llm_value = getattr(extraction, field, None) if extraction is not None else None
        baseline_value = baseline.get(field)

        if not llm_value:
            if field in _REQUIRED_LLM_FIELDS:
                warnings.append(
                    QcWarning(code="llm_frontmatter_extraction_failed", detail=field)
                )
            resolved[field] = baseline_value
            continue

        if baseline_value and llm_value != baseline_value:
            warnings.append(QcWarning(code="llm_frontmatter_mismatch", detail=field))
        resolved[field] = llm_value

    return resolved, warnings


def build_frontmatter(
    body_blocks: list[Block],
    quoc_hieu_block: Block | None,
    *,
    source_path: str | None,
    is_phu_luc: bool,
) -> tuple[FrontMatter, list[QcWarning]]:
    """Dựng front matter từ bảng quốc hiệu và thân văn bản đã gỡ marker.

    LLM (``extract_frontmatter_llm``) là đường chính đọc giá trị field;
    regex baseline (``extract_quoc_hieu``, ``_find_ten_van_ban``,
    ``_find_loai_van_ban``, ``_find_ngay_hieu_luc``) luôn được tính để
    fallback/so sánh (``_resolve_frontmatter_fields``, mục 6, 7 spec).
    """
    warnings: list[QcWarning] = []

    quoc_hieu = extract_quoc_hieu(quoc_hieu_block)
    baseline: dict[str, str | None] = {
        "so_hieu": quoc_hieu.get("so_hieu"),
        "co_quan_ban_hanh": quoc_hieu.get("co_quan_ban_hanh"),
        "ngay_ban_hanh": quoc_hieu.get("ngay_ban_hanh"),
    }
    baseline["ten_van_ban"] = _find_ten_van_ban(body_blocks, _DOC_TYPE_SEARCH_LIMIT)
    baseline["loai_van_ban"] = _find_loai_van_ban(body_blocks, baseline["so_hieu"])
    baseline["ngay_hieu_luc"] = _find_ngay_hieu_luc(body_blocks)

    extraction = extract_frontmatter_llm(quoc_hieu_block, body_blocks)
    fields, llm_warnings = _resolve_frontmatter_fields(extraction, baseline)
    warnings.extend(llm_warnings)

    so_hieu = fields["so_hieu"]
    co_quan_ban_hanh = fields["co_quan_ban_hanh"]
    ngay_ban_hanh = fields["ngay_ban_hanh"]
    ten_van_ban = fields["ten_van_ban"]
    loai_van_ban = fields["loai_van_ban"]
    ngay_hieu_luc = fields["ngay_hieu_luc"]

    is_van_ban_hop_nhat = bool(so_hieu and "VBHN" in so_hieu.upper())
    if not is_van_ban_hop_nhat:
        is_van_ban_hop_nhat = any(
            block.kind == "paragraph" and "hợp nhất" in block.text.casefold()
            for block in body_blocks[:_DOC_TYPE_SEARCH_LIMIT]
        )

    # so_hieu và ngay_hieu_luc là cặp bắt buộc.
    for field, value in (("so_hieu", so_hieu), ("ngay_hieu_luc", ngay_hieu_luc)):
        if not value:
            warnings.append(QcWarning(code="missing_frontmatter_field", detail=field))

    for field, value in (
        ("loai_van_ban", loai_van_ban),
        ("ten_van_ban", ten_van_ban),
        ("co_quan_ban_hanh", co_quan_ban_hanh),
        ("ngay_ban_hanh", ngay_ban_hanh),
    ):
        if not value:
            warnings.append(
                QcWarning(code="missing_optional_frontmatter_field", detail=field)
            )

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
    )
    return front_matter, warnings
