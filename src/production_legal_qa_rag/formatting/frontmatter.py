"""Trích xuất metadata cấp văn bản (số hiệu, loại văn bản, ngày...).

Số hiệu, cơ quan ban hành và ngày ban hành lấy từ bảng quốc hiệu (tách ra
bởi ``tables.triage_tables``); tên văn bản, loại văn bản và ngày hiệu lực dò
trên thân văn bản đã gỡ marker chú thích.
"""

from __future__ import annotations

import re

from production_legal_qa_rag.formatting.docx_reader import Block
from production_legal_qa_rag.formatting.models import FrontMatter, QcWarning
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
