"""Bộ test cho bước formatting (DOCX -> Markdown), theo `formatting_spec.md`.

Cấu trúc theo từng module trong `src/production_legal_qa_rag/formatting/`:
patterns, footnotes, tables, frontmatter, emitter, validator, models
(schema), docx_reader (đọc DOCX thật) và pipeline (integration trên
`data/raw/*.docx` thật + CLI).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from typer.testing import CliRunner

from production_legal_qa_rag.formatting import (
    emitter,
    footnotes,
    frontmatter,
    pipeline,
    tables,
    validator,
)
from production_legal_qa_rag.formatting.docx_reader import Block, read_docx
from production_legal_qa_rag.formatting.footnotes import Footnote
from production_legal_qa_rag.formatting.models import (
    FormattingResult,
    FrontMatter,
    QcWarning,
)
from production_legal_qa_rag.formatting.patterns import (
    RE_DIEM,
    RE_DIEU,
    RE_KHOAN,
    is_structural,
    normalize_text,
    sort_key,
)
from production_legal_qa_rag.formatting.pipeline import (
    convert_directory,
    convert_docx_to_markdown,
    write_atomic,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REFERENCE_MD_DIR = PROJECT_ROOT / "data" / "markdown"

RAW_FILES = sorted(RAW_DIR.glob("*.docx")) if RAW_DIR.exists() else []

# Toàn bộ mã QcWarning hợp lệ mà package hiện phát ra (mục 3, 7 của spec: QC
# warnings không bao giờ làm fail file, nhưng không được có mã MỚI ngoài danh
# sách này). Cập nhật danh sách này trong cùng lúc thêm rule QC mới.
KNOWN_WARNING_CODES = {
    "orphan_footnote",
    "long_footnote_deferred",
    "unused_footnote",
    "dropped_noi_nhan_table",
    "dropped_attachment_table",
    "missing_quoc_hieu_table",
    "missing_frontmatter_field",
    "missing_optional_frontmatter_field",
    "ambiguous_footnote_region",
    "footnote_number_gap",
    "footnote_marker_in_table",
    "heading_too_deep",
    "heading_level_skip",
    "suspicious_heading_length",
    "no_heading",
    "dieu_not_monotonic",
    "khoan_not_monotonic",
    "empty_dieu",
}


def P(text: str, *, style: str | None = None, bold: bool = False) -> Block:
    return Block(kind="paragraph", text=text, style=style, is_bold=bold)


def T(text: str) -> Block:
    return Block(kind="table", text=text)


# ==========================================================================
# models.py -- schema
# ==========================================================================


def test_qc_warning_default_detail_rong():
    warning = QcWarning(code="no_heading")
    assert warning.detail == ""


def test_front_matter_to_yaml_gia_tri_none_thanh_null():
    front_matter = FrontMatter(source_path="a.docx")
    lines = dict(
        line.split(": ", 1) for line in front_matter.to_yaml().splitlines()[1:-1]
    )
    assert lines["so_hieu"] == "null"
    assert lines["is_van_ban_hop_nhat"] == "false"
    assert lines["is_phu_luc"] == "false"


def _parse_front_matter_yaml(rendered: str) -> dict:
    """`to_yaml()` sinh hai dòng `---` (mở/đóng) kiểu front matter, không phải
    một document YAML đơn: `yaml.safe_load` trần sẽ coi dòng `---` đóng là mở
    đầu document thứ hai (rỗng) và báo lỗi ComposerError. Lấy document đầu
    tiên bằng `safe_load_all` để phản ánh đúng cách frontmatter thật được
    parse (bỏ qua document rỗng theo sau)."""
    return next(yaml.safe_load_all(rendered))


def test_front_matter_to_yaml_escape_dau_ngoac_kep():
    front_matter = FrontMatter(ten_van_ban='Luật "sửa đổi" số 1')
    rendered = front_matter.to_yaml()
    assert '\\"sửa đổi\\"' in rendered
    # Phải là YAML hợp lệ, parse lại đúng giá trị gốc.
    parsed = _parse_front_matter_yaml(rendered)
    assert parsed["ten_van_ban"] == 'Luật "sửa đổi" số 1'


def test_front_matter_to_yaml_la_yaml_hop_le_va_khop_field():
    front_matter = FrontMatter(
        so_hieu="293/2025/NĐ-CP",
        loai_van_ban="Nghị định",
        is_phu_luc=True,
    )
    parsed = _parse_front_matter_yaml(front_matter.to_yaml())
    assert parsed["so_hieu"] == "293/2025/NĐ-CP"
    assert parsed["is_phu_luc"] is True
    assert parsed["parser_version"] == front_matter.parser_version


def test_formatting_result_warnings_mac_dinh_rong():
    result = FormattingResult(markdown="x", front_matter=FrontMatter())
    assert result.warnings == []


# ==========================================================================
# patterns.py
# ==========================================================================


def test_normalize_text_gop_khoang_trang_va_nbsp():
    assert normalize_text("Điều 1.  Phạm  vi") == "Điều 1. Phạm vi"


def test_normalize_text_strip_dau_cuoi():
    assert normalize_text("  Chương I  ") == "Chương I"


@pytest.mark.parametrize(
    ("smaller", "larger"),
    [
        ("48", "48a"),
        ("48a", "48b"),
        ("48b", "49"),
        ("1", "1a"),
        ("9", "10"),
    ],
)
def test_sort_key_thu_tu_so_hieu_co_hau_to_chu(smaller, larger):
    assert sort_key(smaller) < sort_key(larger)


def test_re_dieu_ho_tro_so_hieu_chu():
    match = RE_DIEU.match("Điều 41a. Điều khoản chuyển tiếp")
    assert match is not None
    assert match.group(1) == "41a"
    assert match.group(2) == "Điều khoản chuyển tiếp"


def test_re_dieu_khong_khop_khi_thieu_dau_cham():
    # "Điều 2 của Luật số 46/2014/QH13 quy định..." (trích dẫn trong chú
    # thích) không được nhận nhầm thành heading Điều.
    assert RE_DIEU.match("Điều 2 của Luật số 46/2014/QH13 quy định như sau:") is None


def test_re_khoan_khop_va_giu_noi_dung():
    match = RE_KHOAN.match("1. Người lao động làm việc theo hợp đồng.")
    assert match is not None
    assert match.group(1) == "1"
    assert match.group(2) == "Người lao động làm việc theo hợp đồng."


def test_re_diem_hai_dang_chu_va_gach_dau_dong():
    letter = RE_DIEM.match("a) Doanh nghiệp theo quy định.")
    dash = RE_DIEM.match("- Một mục không đánh số.")
    assert letter is not None and letter.group(1) == "a"
    assert dash is not None and dash.group(1) is None


def test_is_structural_nhan_dien_dieu_la_cau_truc():
    assert is_structural("Điều 5. Tên điều")
    assert is_structural("Chương IV")
    assert not is_structural("Người lao động làm việc theo hợp đồng.")


# ==========================================================================
# footnotes.py
# ==========================================================================


@pytest.mark.parametrize(
    ("raw", "expected_text", "expected_numbers"),
    [
        ("3.3[3] nội dung", "3. nội dung", [3]),
        ("10.[15] nội dung", "10. nội dung", [15]),
        ("1.4[4] nội dung", "1. nội dung", [4]),
    ],
)
def test_strip_markers_chi_nuot_chu_so_lap_bang_num(
    raw, expected_text, expected_numbers
):
    stripped, found = footnotes.strip_markers(raw)
    assert stripped == expected_text
    assert found == expected_numbers


def test_strip_markers_khong_marker_giu_nguyen():
    stripped, found = footnotes.strip_markers("Không có chú thích nào ở đây.")
    assert stripped == "Không có chú thích nào ở đây."
    assert found == []


def test_strip_markers_va_them_lai_khoang_trang_bi_mat():
    stripped, found = footnotes.strip_markers("a)[1]Thành lập doanh nghiệp")
    assert stripped == "a) Thành lập doanh nghiệp"
    assert found == [1]


def test_strip_all_khong_go_marker_trong_bang():
    blocks = [T("| a[1] | b |\n| --- | --- |\n| c | d |")]
    cleaned, refs, warnings = footnotes.strip_all(blocks)
    assert cleaned[0].text == blocks[0].text
    assert refs == {}
    assert any(w.code == "footnote_marker_in_table" for w in warnings)


def test_find_region_start_mot_ket_qua():
    blocks = [P("Nội dung thân văn bản."), P("[1] Nội dung sửa đổi đầu tiên.")]
    index, warnings = footnotes.find_region_start(blocks)
    assert index == 1
    assert warnings == []


def test_find_region_start_khong_co():
    blocks = [P("Không có chú thích nào.")]
    index, warnings = footnotes.find_region_start(blocks)
    assert index is None
    assert warnings == []


def test_find_region_start_nhieu_ket_qua_canh_bao_va_lay_cuoi():
    blocks = [P("[1] Lần đầu (giả)."), P("Ở giữa."), P("[1] Lần cuối (thật).")]
    index, warnings = footnotes.find_region_start(blocks)
    assert index == 2
    assert any(w.code == "ambiguous_footnote_region" for w in warnings)


def test_parse_region_dang_ngoac_va_dang_tran():
    blocks = [
        P("[1] Nội dung chú thích một."),
        P("2 Nội dung chú thích hai."),
    ]
    footnote_map, warnings = footnotes.parse_region(blocks)
    assert footnote_map[1].text == "Nội dung chú thích một."
    assert footnote_map[2].text == "Nội dung chú thích hai."
    assert warnings == []


def test_parse_region_bao_lo_hong_so_hieu():
    blocks = [P("[1] Đầu tiên."), P("[3] Nhảy cóc.")]
    footnote_map, warnings = footnotes.parse_region(blocks)
    assert set(footnote_map) == {1, 3}
    assert any(w.code == "footnote_number_gap" for w in warnings)


def test_parse_region_trich_dan_khong_bi_nham_thanh_dinh_nghia_moi():
    # "1. Luật này có hiệu lực..." có dấu chấm nên không khớp RE_FN_DEF_BARE.
    blocks = [
        P("[1] Trích dẫn: 1. Luật này có hiệu lực kể từ ngày công bố."),
    ]
    footnote_map, _ = footnotes.parse_region(blocks)
    assert footnote_map[1].paragraphs == (
        "Trích dẫn: 1. Luật này có hiệu lực kể từ ngày công bố.",
    )


def test_render_blockquote_ngan_inline_toan_bo():
    footnote = Footnote(number=1, paragraphs=("Nội dung ngắn.",))
    inline, deferred = footnotes.render_blockquote(footnote, inline_max=1200)
    assert inline == "> **Sửa đổi:** Nội dung ngắn."
    assert deferred is None


def test_render_blockquote_dai_thi_doi_xuong_cuoi():
    footnote = Footnote(number=2, paragraphs=("x" * 2000,))
    inline, deferred = footnotes.render_blockquote(footnote, inline_max=1200)
    assert "xem chú thích [2] ở cuối văn bản" in inline
    assert deferred is not None
    assert deferred.startswith("**[2]**")


def test_render_blockquote_rong_tra_ve_rong():
    footnote = Footnote(number=3, paragraphs=("   ",))
    inline, deferred = footnotes.render_blockquote(footnote, inline_max=1200)
    assert inline == ""
    assert deferred is None


# ==========================================================================
# tables.py
# ==========================================================================


def _make_docx_table(rows: list[list[str]]):
    document = Document()
    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            table.cell(row_index, col_index).text = value
    return table


def test_table_to_markdown_bang_du_lieu_ra_pipe_table():
    table = _make_docx_table([["Vùng", "Mức lương"], ["I", "5.310.000"]])
    rendered = tables.table_to_markdown(table)
    lines = rendered.splitlines()
    assert lines[0] == "| Vùng | Mức lương |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| I | 5.310.000 |"


def test_table_to_markdown_mot_hang_ra_html():
    table = _make_docx_table([["A = B x C"]])
    rendered = tables.table_to_markdown(table)
    assert rendered.startswith("<table>")
    assert "A = B x C" in rendered


def test_parse_pipe_table_bo_dong_phan_cach():
    markdown = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    assert tables.parse_pipe_table(markdown) == [["a", "b"], ["1", "2"]]


def test_triage_tables_tach_quoc_hieu_va_loai_bang_nhieu():
    blocks = [
        T("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nSố: 293/2025/NĐ-CP"),
        P("Điều 1. Phạm vi điều chỉnh"),
        T("| Nơi nhận: | |\n| --- | --- |\n| - Như trên | |"),
        T("FILE ĐƯỢC ĐÍNH KÈM THEO VĂN BẢN"),
        T("| Vùng | Mức |\n| --- | --- |\n| I | 5.000.000 |"),
    ]
    quoc_hieu, kept, warnings = tables.triage_tables(blocks)
    assert quoc_hieu is blocks[0]
    assert kept == [blocks[1], blocks[4]]
    codes = {w.code for w in warnings}
    assert codes == {"dropped_noi_nhan_table", "dropped_attachment_table"}


def test_triage_tables_canh_bao_khi_khong_co_quoc_hieu():
    blocks = [P("Điều 1. Phạm vi điều chỉnh")]
    quoc_hieu, kept, warnings = tables.triage_tables(blocks)
    assert quoc_hieu is None
    assert kept == blocks
    assert any(w.code == "missing_quoc_hieu_table" for w in warnings)


# ==========================================================================
# frontmatter.py
# ==========================================================================


def test_extract_quoc_hieu_tu_bang():
    table_block = T(
        "| CHÍNH PHỦ | CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM |\n"
        "| --- | --- |\n"
        "| Số: 293/2025/NĐ-CP | Hà Nội, ngày 10 tháng 11 năm 2025 |"
    )
    result = frontmatter.extract_quoc_hieu(table_block)
    assert result["so_hieu"] == "293/2025/NĐ-CP"
    assert result["co_quan_ban_hanh"] == "CHÍNH PHỦ"
    assert result["ngay_ban_hanh"] == "2025-11-10"


def test_extract_quoc_hieu_khong_co_bang():
    result = frontmatter.extract_quoc_hieu(None)
    assert result == {
        "so_hieu": None,
        "co_quan_ban_hanh": None,
        "ngay_ban_hanh": None,
    }


def test_build_frontmatter_day_du_truong():
    body = [
        P("NGHỊ ĐỊNH"),
        P("Quy định về ban hành Nghị định về mức lương tối thiểu."),
        P("Điều 10. Hiệu lực thi hành"),
        P("Nghị định này có hiệu lực thi hành kể từ ngày 01 tháng 01 năm 2026."),
    ]
    quoc_hieu = {
        "so_hieu": "293/2025/NĐ-CP",
        "co_quan_ban_hanh": "CHÍNH PHỦ",
        "ngay_ban_hanh": "2025-11-10",
    }
    front_matter, warnings = frontmatter.build_frontmatter(
        body, quoc_hieu, source_path="data/raw/x.docx", is_phu_luc=False
    )
    assert front_matter.so_hieu == "293/2025/NĐ-CP"
    assert front_matter.loai_van_ban == "Nghị định"
    assert front_matter.ngay_hieu_luc == "2026-01-01"
    assert front_matter.is_van_ban_hop_nhat is False
    assert warnings == []


def test_build_frontmatter_thieu_truong_bat_buoc_canh_bao():
    front_matter, warnings = frontmatter.build_frontmatter(
        [P("Một dòng bất kỳ.")],
        {"so_hieu": None, "co_quan_ban_hanh": None, "ngay_ban_hanh": None},
        source_path=None,
        is_phu_luc=False,
    )
    codes = {w.code for w in warnings}
    assert "missing_frontmatter_field" in codes
    assert front_matter.so_hieu is None


def test_build_frontmatter_van_ban_hop_nhat_qua_so_hieu():
    front_matter, _ = frontmatter.build_frontmatter(
        [P("một dòng")],
        {"so_hieu": "01/VBHN-BLĐTBXH", "co_quan_ban_hanh": None, "ngay_ban_hanh": None},
        source_path=None,
        is_phu_luc=False,
    )
    assert front_matter.is_van_ban_hop_nhat is True
    assert front_matter.loai_van_ban == "Văn bản hợp nhất"


# ==========================================================================
# emitter.py
# ==========================================================================


def test_emit_anh_xa_heading_co_ban():
    blocks = [
        P("Chương I"),
        P("NHỮNG QUY ĐỊNH CHUNG"),
        P("Điều 1. Phạm vi điều chỉnh"),
        P("1. Nội dung khoản một."),
        P("a) Điểm a của khoản một."),
    ]
    preamble_end = emitter.find_preamble_end(blocks)
    parts, deferred, in_phu_luc, warnings = emitter.emit(blocks, {}, {}, preamble_end)
    assert parts[0] == "## Chương I. NHỮNG QUY ĐỊNH CHUNG"
    assert parts[1] == "#### Điều 1. Phạm vi điều chỉnh"
    assert parts[2] == "##### Khoản 1"
    assert parts[3] == "Nội dung khoản một."
    assert parts[4] == "a) Điểm a của khoản một."
    assert deferred == []
    assert in_phu_luc is False
    assert warnings == []


def test_emit_phu_luc_khoan_ngan_gop_vao_heading():
    blocks = [
        P("PHỤ LỤC"),
        P("DANH MỤC TỈNH THÀNH"),
        P("28. Thành phố Hồ Chí Minh"),
    ]
    preamble_end = emitter.find_preamble_end(blocks)
    parts, _, in_phu_luc, _ = emitter.emit(blocks, {}, {}, preamble_end)
    assert parts[0] == "# PHỤ LỤC — DANH MỤC TỈNH THÀNH"
    assert parts[1] == "##### 28. Thành phố Hồ Chí Minh"
    assert in_phu_luc is True


def test_emit_phu_luc_khoan_dai_khong_gop_vao_heading():
    long_content = "Nội dung dài " * 20  # > 120 ký tự
    blocks = [P("PHỤ LỤC"), P("BẢNG DANH MỤC"), P(f"1. {long_content}")]
    preamble_end = emitter.find_preamble_end(blocks)
    parts, _, _, _ = emitter.emit(blocks, {}, {}, preamble_end)
    assert parts[1] == "##### Khoản 1"
    assert parts[2] == long_content


def test_emit_tach_phan_kem_theo_khoi_tieu_de_phu_luc():
    blocks = [
        P("PHỤ LỤC"),
        P("BẢNG LƯƠNG (Kèm theo Nghị định số 293/2025/NĐ-CP)"),
        P("1. Nội dung."),
    ]
    preamble_end = emitter.find_preamble_end(blocks)
    parts, _, _, _ = emitter.emit(blocks, {}, {}, preamble_end)
    assert parts[0] == "# PHỤ LỤC — BẢNG LƯƠNG"
    assert parts[1] == "(Kèm theo Nghị định số 293/2025/NĐ-CP)"


def test_emit_chen_chu_thich_inline_va_canh_bao_orphan():
    blocks = [P("Điều 1. Phạm vi"), P("1. Nội dung khoản một.")]
    refs = {1: [7]}
    footnote_map = {5: Footnote(number=5, paragraphs=("Nội dung chú thích 5.",))}
    preamble_end = emitter.find_preamble_end(blocks)
    parts, deferred, _, warnings = emitter.emit(
        blocks, refs, footnote_map, preamble_end
    )
    assert any(part.startswith("> **Sửa đổi:**") is False for part in parts)
    assert any(w.code == "orphan_footnote" and w.detail == "[7]" for w in warnings)
    assert any(w.code == "unused_footnote" and w.detail == "[5]" for w in warnings)
    assert deferred == []


def test_emit_footnote_dai_dua_xuong_cuoi():
    blocks = [P("Điều 1. Phạm vi")]
    refs = {0: [1]}
    footnote_map = {1: Footnote(number=1, paragraphs=("x" * 2000,))}
    preamble_end = emitter.find_preamble_end(blocks)
    _parts, deferred, _, warnings = emitter.emit(
        blocks, refs, footnote_map, preamble_end
    )
    assert deferred and deferred[0].startswith("**[1]**")
    assert any(w.code == "long_footnote_deferred" for w in warnings)


def test_find_preamble_end_bo_qua_danh_so_o_dan_nhap():
    blocks = [
        P("Căn cứ Luật Nhà giáo số 73/2025/QH15..."),
        P("1. Luật Nhà giáo số 73/2025/QH15."),
        P("Chương I"),
        P("NHỮNG QUY ĐỊNH CHUNG"),
    ]
    assert emitter.find_preamble_end(blocks) == 2


# ==========================================================================
# validator.py
# ==========================================================================


def test_validate_khong_co_heading_canh_bao():
    warnings = validator.validate("---\nso_hieu: null\n---\n\nMột đoạn văn thường.")
    assert any(w.code == "no_heading" for w in warnings)


def test_validate_heading_qua_sau_canh_bao():
    markdown = "#### Điều 1. A\n\n###### quá sâu"
    warnings = validator.validate(markdown)
    assert any(w.code == "heading_too_deep" for w in warnings)


def test_validate_bo_qua_level_skip_hop_le_trong_phu_luc():
    markdown = "# PHỤ LỤC\n\n##### 1. Hà Nội\n\nNội dung."
    warnings = validator.validate(markdown)
    assert not any(w.code == "heading_level_skip" for w in warnings)


def test_validate_bao_level_skip_ngoai_phu_luc():
    markdown = "### Mục 1\n\n##### Khoản 1\n\nNội dung."
    warnings = validator.validate(markdown)
    assert any(w.code == "heading_level_skip" for w in warnings)


def test_validate_dieu_khong_tang_dan():
    markdown = "#### Điều 2. B\n\nNội dung.\n\n#### Điều 1. A\n\nNội dung."
    warnings = validator.validate(markdown)
    assert any(w.code == "dieu_not_monotonic" for w in warnings)


def test_validate_khoan_khong_tang_dan():
    markdown = (
        "#### Điều 1. A\n\n##### Khoản 2\n\nNội dung.\n\n##### Khoản 1\n\nNội dung."
    )
    warnings = validator.validate(markdown)
    assert any(w.code == "khoan_not_monotonic" for w in warnings)


def test_validate_dieu_rong_canh_bao():
    markdown = "#### Điều 1. A\n\n#### Điều 2. B\n\nNội dung điều 2."
    warnings = validator.validate(markdown)
    assert any(w.code == "empty_dieu" and w.detail == "Điều 1. A" for w in warnings)


def test_validate_marker_mo_coi_ngoai_blockquote():
    markdown = "#### Điều 1. A\n\nNội dung còn sót marker [9] chưa được chèn lại."
    warnings = validator.validate(markdown)
    assert any(w.code == "orphan_footnote" for w in warnings)


def test_validate_marker_trong_blockquote_khong_bi_bao():
    markdown = "#### Điều 1. A\n\n> **Sửa đổi:** đã dời [9] xuống cuối."
    warnings = validator.validate(markdown)
    assert not any(w.code == "orphan_footnote" for w in warnings)


def test_validate_heading_dai_bat_thuong():
    markdown = "#### Điều 1. " + "A" * 200
    warnings = validator.validate(markdown)
    assert any(w.code == "suspicious_heading_length" for w in warnings)


# ==========================================================================
# docx_reader.py -- đọc DOCX thật, giữ thứ tự paragraph/table xen kẽ
# ==========================================================================


def test_read_docx_giu_dung_thu_tu_xen_ke(tmp_path: Path):
    document = Document()
    document.add_paragraph("Điều 1. Phạm vi điều chỉnh")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Cột A"
    table.cell(0, 1).text = "Cột B"
    table.cell(1, 0).text = "1"
    table.cell(1, 1).text = "2"
    document.add_paragraph("1. Nội dung khoản một.")
    document.add_paragraph("   ")  # đoạn trắng, phải bị bỏ qua

    docx_path = tmp_path / "sample.docx"
    document.save(docx_path)

    blocks = read_docx(docx_path)
    assert [block.kind for block in blocks] == ["paragraph", "table", "paragraph"]
    assert blocks[0].text == "Điều 1. Phạm vi điều chỉnh"
    assert blocks[1].text.startswith("| Cột A | Cột B |")
    assert blocks[2].text == "1. Nội dung khoản một."


def test_read_docx_in_dam_toan_doan_la_bold(tmp_path: Path):
    document = Document()
    paragraph = document.add_paragraph()
    run = paragraph.add_run("NGHỊ ĐỊNH")
    run.bold = True

    docx_path = tmp_path / "bold.docx"
    document.save(docx_path)

    blocks = read_docx(docx_path)
    assert blocks[0].is_bold is True


# ==========================================================================
# pipeline.py -- integration trên corpus thật + CLI + atomic write
# ==========================================================================


@pytest.fixture(scope="session")
def real_results() -> dict[str, FormattingResult]:
    return {path.stem: convert_docx_to_markdown(path) for path in RAW_FILES}


@pytest.mark.skipif(not RAW_FILES, reason="data/raw/*.docx không tồn tại")
def test_data_raw_co_du_6_file():
    assert len(RAW_FILES) == 6


@pytest.mark.skipif(not RAW_FILES, reason="data/raw/*.docx không tồn tại")
@pytest.mark.parametrize("path", RAW_FILES, ids=lambda p: p.stem)
def test_convert_khop_voi_tham_chieu_data_markdown(path: Path):
    """Khớp byte-for-byte với `data/markdown/*.md` hiện có (mục 7 spec).

    `data/markdown/*.md` không phải golden-file chính thức nhưng là tham
    chiếu đã được đối chiếu thủ công (theo commit message); một khác biệt ở
    đây là tín hiệu mạnh cho REVISE, không phải lý do tự động coi là đúng.
    """
    reference_path = REFERENCE_MD_DIR / f"{path.stem}.md"
    if not reference_path.exists():
        pytest.skip(f"Không có tham chiếu cho {path.stem}")
    # Tham chiếu được sinh bằng CLI gọi với `--raw-dir data/raw` (đường dẫn
    # tương đối, từ thư mục gốc repo) nên `source_path` trong front matter là
    # tương đối. Dùng cùng dạng đường dẫn ở đây để so khớp byte-for-byte thật
    # sự (không lệch mỗi khác biệt chỉ vì absolute/relative).
    relative_path = Path("data") / "raw" / path.name
    result = convert_docx_to_markdown(relative_path)
    assert result.markdown == reference_path.read_text(encoding="utf-8")


@pytest.mark.skipif(not RAW_FILES, reason="data/raw/*.docx không tồn tại")
@pytest.mark.parametrize("path", RAW_FILES, ids=lambda p: p.stem)
def test_convert_la_ham_thuan_deterministic(path: Path):
    """NT: cùng input phải ra byte-for-byte cùng output, chạy lại nhiều lần."""
    first = convert_docx_to_markdown(path).markdown
    second = convert_docx_to_markdown(path).markdown
    assert first == second


@pytest.mark.skipif(not RAW_FILES, reason="data/raw/*.docx không tồn tại")
@pytest.mark.parametrize("path", RAW_FILES, ids=lambda p: p.stem)
def test_convert_front_matter_hop_le(path: Path):
    result = convert_docx_to_markdown(path)
    assert result.markdown.startswith("---\n")
    parsed = yaml.safe_load(result.markdown.split("\n---\n", 1)[0] + "\n")
    assert parsed["parser_version"] == result.front_matter.parser_version
    assert parsed["so_hieu"] == result.front_matter.so_hieu


@pytest.mark.skipif(not RAW_FILES, reason="data/raw/*.docx không tồn tại")
def test_convert_khong_co_ma_canh_bao_moi(real_results):
    """Mục 7 spec: không phát sinh mã QC warning mới ngoài danh sách đã biết."""
    unknown: set[str] = set()
    for result in real_results.values():
        unknown |= {w.code for w in result.warnings} - KNOWN_WARNING_CODES
    assert not unknown, (
        f"Mã QC warning mới, chưa có trong KNOWN_WARNING_CODES: {unknown}"
    )


@pytest.mark.skipif(not RAW_FILES, reason="data/raw/*.docx không tồn tại")
def test_convert_file_khong_ton_tai_bao_loi_ro_rang():
    with pytest.raises(PackageNotFoundError):
        convert_docx_to_markdown(RAW_DIR / "khong_ton_tai.docx")


def test_write_atomic_ghi_dung_noi_dung(tmp_path: Path):
    output_path = tmp_path / "out" / "file.md"
    write_atomic(output_path, "nội dung")
    assert output_path.read_text(encoding="utf-8") == "nội dung"
    # Không để lại file tạm.
    assert list(tmp_path.rglob("*.tmp")) == []


def test_write_atomic_don_dep_file_tam_khi_loi(tmp_path: Path, monkeypatch):
    output_path = tmp_path / "file.md"

    def _boom(_fileno):
        raise OSError("giả lập lỗi ghi đĩa")

    monkeypatch.setattr(pipeline.os, "fsync", _boom)
    with pytest.raises(OSError):
        write_atomic(output_path, "nội dung")
    assert not output_path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_convert_directory_loi_mot_file_khong_chan_ca_batch(tmp_path: Path, capsys):
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()

    good_source = RAW_FILES[0] if RAW_FILES else None
    if good_source is None:
        pytest.skip("data/raw/*.docx không tồn tại")
    shutil.copy(good_source, raw_dir / "hop_le.docx")
    (raw_dir / "hong.docx").write_bytes(b"khong phai file docx that")
    (raw_dir / "~$khoa.docx").write_bytes(b"file tam cua Word, phai bi bo qua")

    exit_code = convert_directory(raw_dir, out_dir)

    assert exit_code == 1
    assert (out_dir / "hop_le.md").exists()
    assert not (out_dir / "hong.md").exists()
    assert not (out_dir / "~$khoa.md").exists()

    captured = capsys.readouterr()
    assert "hop_le.docx" in captured.out
    assert "Thành công : 1" in captured.out
    assert "Thất bại   : 1" in captured.out


def test_convert_directory_khong_co_docx_tra_ve_0(tmp_path: Path):
    raw_dir = tmp_path / "raw_rong"
    raw_dir.mkdir()
    assert convert_directory(raw_dir, tmp_path / "out") == 0


def test_convert_directory_khong_tra_ve_file_tam(tmp_path: Path):
    if not RAW_FILES:
        pytest.skip("data/raw/*.docx không tồn tại")
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    shutil.copy(RAW_FILES[0], raw_dir / RAW_FILES[0].name)

    convert_directory(raw_dir, out_dir)

    assert list(out_dir.rglob("*.tmp")) == []


# ==========================================================================
# tools/format_documents.py -- CLI Typer
# ==========================================================================


def test_cli_chuyen_doi_toan_bo_thu_muc(tmp_path: Path):
    if not RAW_FILES:
        pytest.skip("data/raw/*.docx không tồn tại")
    from tools.format_documents import app

    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "out"
    raw_dir.mkdir()
    shutil.copy(RAW_FILES[0], raw_dir / RAW_FILES[0].name)

    runner = CliRunner()
    result = runner.invoke(app, ["--raw-dir", str(raw_dir), "--out-dir", str(out_dir)])

    assert result.exit_code == 0
    assert (out_dir / f"{RAW_FILES[0].stem}.md").exists()


def test_cli_thuc_thi_duoc_qua_module_chinh():
    """`python tools/format_documents.py --help` không lỗi (kiểm tra entry point).

    Ép NO_COLOR + COLUMNS rộng: Rich (dùng bởi Typer) tô màu ANSI và tự xuống
    dòng theo bề rộng terminal, nên trên CI (không phải TTY, COLUMNS khác máy
    dev) "raw-dir" có thể bị mã màu hoặc dấu xuống dòng chen vào giữa.
    """
    env = {**os.environ, "NO_COLOR": "1", "COLUMNS": "200"}
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "tools" / "format_documents.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )
    assert completed.returncode == 0
    assert "raw-dir" in completed.stdout
