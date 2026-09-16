"""Bộ test cho bước formatting (DOCX -> Markdown), theo `formatting_spec.md`.

Thiết kế mới (mục 1.1): front matter/back matter không còn trích field/YAML,
chỉ xác định biên deterministic rồi gọi Gemini (`llm_client.convert_to_markdown`)
convert nguyên khối sang markdown thuần. Test suite này KHÔNG BAO GIỜ gọi
Gemini API thật (fixture `_default_llm_stub` mock `llm_client.convert_to_markdown`
cho toàn bộ session, mặc định trả `None`) -- free tier chỉ 20 request/ngày.

Cấu trúc theo từng module: models, patterns, docx_reader, tables, emitter,
validator, frontmatter, backmatter, llm_client, pipeline (integration trên
`data/raw/*.docx` thật + CLI).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import get_args
from unittest.mock import Mock

import pytest
from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from pydantic import ValidationError
from typer.testing import CliRunner

from production_legal_qa_rag.config import LLMSettings
from production_legal_qa_rag.formatting import (
    backmatter,
    emitter,
    frontmatter,
    llm_client,
    pipeline,
    tables,
    validator,
)
from production_legal_qa_rag.formatting.docx_reader import (
    Block,
    read_docx,
    serialize_blocks_for_llm,
)
from production_legal_qa_rag.formatting.models import (
    FormattingResult,
    QcWarning,
    QcWarningCode,
)
from production_legal_qa_rag.formatting.patterns import (
    RE_DIEM,
    RE_DIEU,
    RE_FOOTNOTE_MARKER,
    RE_KHOAN,
    is_structural,
    normalize_text,
    sort_key,
    strip_markers,
)
from production_legal_qa_rag.formatting.pipeline import (
    convert_directory,
    convert_docx_to_markdown,
    write_atomic,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REFERENCE_MD_DIR = PROJECT_ROOT / "data" / "markdown"

# Rich (dùng bởi Typer để render --help) vẫn chèn mã CSI cho style (bold, dim)
# dù đã set NO_COLOR -- NO_COLOR chỉ tắt màu, không tắt style. Gỡ mã ANSI trước
# khi so khớp chuỗi để không phụ thuộc vào việc Rich style output ra sao.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

RAW_FILES = sorted(RAW_DIR.glob("*.docx")) if RAW_DIR.exists() else []

# Toàn bộ mã QcWarning hợp lệ đọc trực tiếp từ `QcWarningCode` (models.py) --
# không hard-code lại danh sách ở đây, tránh lệch nhau khi thêm rule QC mới.
KNOWN_WARNING_CODES = set(get_args(QcWarningCode))


def P(text: str, *, style: str | None = None, bold: bool = False) -> Block:
    return Block(kind="paragraph", text=text, style=style, is_bold=bold)


def T(text: str) -> Block:
    return Block(kind="table", text=text)


# ==========================================================================
# Không bao giờ gọi Gemini API thật trong test suite (CI không có
# GEMINI_API_KEY; máy dev có thể có key thật -- không nên phụ thuộc vào việc
# thiếu key mới an toàn, mock hẳn ở mức `llm_client`).
# ==========================================================================

# Giữ tham chiếu tới hàm THẬT trước khi fixture dưới đây ghi đè
# `llm_client.convert_to_markdown` -- các test của "llm_client.py" cần gọi
# đúng implementation thật (chỉ mock `_client`, không mock `convert_to_markdown`
# chính nó) để kiểm tra logic try/except/retry thật của nó.
_REAL_CONVERT_TO_MARKDOWN = llm_client.convert_to_markdown


@pytest.fixture(autouse=True)
def _default_llm_stub(monkeypatch: pytest.MonkeyPatch):
    """Mặc định `llm_client.convert_to_markdown` trả `None` (Gemini "lỗi")
    cho MỌI test, trừ khi test tự monkeypatch lại giá trị khác bên trong.
    """
    monkeypatch.setattr(llm_client, "convert_to_markdown", lambda *a, **k: None)


def _forbid_llm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*_args, **_kwargs):
        raise AssertionError("llm_client.convert_to_markdown không được gọi ở đây")

    monkeypatch.setattr(llm_client, "convert_to_markdown", _fail)


def _sequential_llm(
    monkeypatch: pytest.MonkeyPatch, values: list[str | None]
) -> list[str]:
    """Trả lần lượt từng giá trị trong `values` cho mỗi lần gọi, đúng thứ tự
    (front rồi back, theo `pipeline.convert_docx_to_markdown`). Trả về danh
    sách các prompt đã nhận được, để test kiểm tra số lần gọi.
    """
    iterator = iter(values)
    calls: list[str] = []

    def fake(prompt: str, *, max_retries: int | None = None) -> str | None:
        calls.append(prompt)
        return next(iterator)

    monkeypatch.setattr(llm_client, "convert_to_markdown", fake)
    return calls


# ==========================================================================
# models.py -- schema
# ==========================================================================


def test_qc_warning_default_detail_rong():
    warning = QcWarning(code="no_heading")
    assert warning.detail == ""


def test_qc_warning_code_khong_hop_le_bi_tu_choi():
    with pytest.raises(ValidationError):
        QcWarning(code="ma_khong_ton_tai")  # type: ignore[arg-type]


def test_formatting_result_warnings_mac_dinh_rong():
    result = FormattingResult(markdown="x")
    assert result.warnings == []


def test_formatting_result_thieu_markdown_bao_loi():
    with pytest.raises(ValidationError):
        FormattingResult()  # type: ignore[call-arg]


def test_qc_warning_code_gom_du_hai_ma_gemini_moi():
    assert "llm_frontmatter_conversion_failed" in KNOWN_WARNING_CODES
    assert "llm_backmatter_conversion_failed" in KNOWN_WARNING_CODES


# ==========================================================================
# patterns.py
# ==========================================================================


def test_normalize_text_gop_khoang_trang_va_nbsp():
    assert normalize_text("Điều 1.  Phạm  vi") == "Điều 1. Phạm vi"


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
    # "Điều 2 của Luật số 46/2014/QH13 quy định..." (trích dẫn) không được
    # nhận nhầm thành heading Điều.
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


def test_is_structural_nhan_dien_dieu_va_chuong_la_cau_truc():
    assert is_structural("Điều 5. Tên điều")
    assert is_structural("Chương IV")
    assert not is_structural("Người lao động làm việc theo hợp đồng.")


def test_is_structural_khong_nhan_dien_khoan():
    # Khoản KHÔNG được coi là biên front matter -- các dòng "1. Luật Nhà giáo
    # số 73/2025/QH15..." trong dẫn nhập văn bản hợp nhất không được nhận
    # nhầm thành biên (mục 1.1 spec).
    assert not is_structural("1. Luật Nhà giáo số 73/2025/QH15.")


# ==========================================================================
# patterns.py -- strip_markers()/RE_FOOTNOTE_MARKER, marker chú thích "[n]"
# dính liền trong vùng nội dung ở giữa (formatting_spec.md mục 1.1, "Lưu ý
# quan trọng"). Regression cho lỗi footnote marker phá RE_KHOAN/RE_DIEU.
# ==========================================================================


def test_strip_markers_go_marker_dinh_ngay_sau_so_khoan():
    # "1.[2] Bảo hiểm..." -- không strip trước thì RE_KHOAN (yêu cầu khoảng
    # trắng ngay sau dấu chấm) sẽ không khớp và mất heading Khoản.
    assert (
        strip_markers("1.[2] Bảo hiểm y tế là hình thức bắt buộc.")
        == "1. Bảo hiểm y tế là hình thức bắt buộc."
    )


def test_strip_markers_go_marker_o_cuoi_dong():
    assert strip_markers("Điều 7a. Bảo hiểm Xã hội[16]") == "Điều 7a. Bảo hiểm Xã hội"


def test_strip_markers_khong_nuot_so_hieu_khi_co_dau_cham_o_giua():
    # "10.[15]" -- chữ số trước "[" không liền kề (có dấu chấm chen giữa) nên
    # KHÔNG bị coi là chữ số lặp cần nuốt, số hiệu "10." phải giữ nguyên.
    assert strip_markers("10.[15] Điều khoản") == "10. Điều khoản"


def test_strip_markers_bo_chu_so_lap_ngay_truoc_marker():
    # "3.3[3]" -- chữ số lặp dính liền ngay trước "[" và BẰNG số marker, bị
    # nuốt cùng marker, chỉ giữ lại "3." gốc.
    assert strip_markers("3.3[3] Nội dung.") == "3. Nội dung."


def test_strip_markers_va_lap_lai_khoang_trang_thieu_sau_marker():
    # Sau khi gỡ marker dính liền, "a)[3]Thành lập" mất khoảng trắng --
    # RE_MISSING_SPACE phải vá lại.
    assert strip_markers("a)[3]Thành lập") == "a) Thành lập"


def test_strip_markers_don_dep_khoang_trang_thua_truoc_dau_cau():
    assert strip_markers("Nội dung[5] , tiếp theo.") == "Nội dung, tiếp theo."


def test_strip_markers_khong_co_marker_giu_nguyen():
    assert strip_markers("Không có marker nào ở đây.") == "Không có marker nào ở đây."


def test_re_footnote_marker_khop_va_lay_duoc_so():
    match = RE_FOOTNOTE_MARKER.search("Bảo hiểm Xã hội[16]")
    assert match is not None
    assert match.group(1) == "16"


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
    assert blocks[0].is_italic is False


def test_read_docx_in_nghieng_toan_doan_la_italic(tmp_path: Path):
    document = Document()
    paragraph = document.add_paragraph()
    run = paragraph.add_run("Độc lập - Tự do - Hạnh phúc")
    run.italic = True

    docx_path = tmp_path / "italic.docx"
    document.save(docx_path)

    blocks = read_docx(docx_path)
    assert blocks[0].is_italic is True
    assert blocks[0].is_bold is False


def test_serialize_blocks_for_llm_bold_duoc_bao_bang_sao_kep():
    blocks = [P("CHÍNH PHỦ", bold=True)]
    assert serialize_blocks_for_llm(blocks) == "**CHÍNH PHỦ**"


def test_serialize_blocks_for_llm_italic_duoc_bao_bang_mot_sao():
    blocks = [Block(kind="paragraph", text="Độc lập - Tự do", is_italic=True)]
    assert serialize_blocks_for_llm(blocks) == "*Độc lập - Tự do*"


def test_serialize_blocks_for_llm_dam_va_nghieng_duoc_bao_bang_ba_sao():
    blocks = [Block(kind="paragraph", text="CHÍNH PHỦ", is_bold=True, is_italic=True)]
    assert serialize_blocks_for_llm(blocks) == "***CHÍNH PHỦ***"


def test_serialize_blocks_for_llm_khong_dinh_dang_giu_nguyen():
    blocks = [P("Số: 293/2025/NĐ-CP")]
    assert serialize_blocks_for_llm(blocks) == "Số: 293/2025/NĐ-CP"


def test_serialize_blocks_for_llm_nhieu_block_cach_nhau_dong_trong():
    blocks = [P("A"), P("B")]
    assert serialize_blocks_for_llm(blocks) == "A\n\nB"


def test_serialize_blocks_for_llm_giu_nguyen_bang():
    table_text = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    blocks = [T(table_text)]
    assert serialize_blocks_for_llm(blocks) == table_text


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


def test_is_signature_table_nhan_dien_theo_noi_dung():
    assert tables.is_signature_table(T("| TM. THỦ TƯỚNG | |\n| --- | --- |\n| A | B |"))
    assert tables.is_signature_table(
        T("| Nơi nhận: | |\n| --- | --- |\n| - Như trên | |")
    )


def test_is_signature_table_khong_nhan_dien_bang_thuong():
    assert not tables.is_signature_table(
        T("| Vùng | Mức |\n| --- | --- |\n| I | 5.000 |")
    )


def test_is_signature_table_bo_qua_paragraph():
    # Chỉ block dạng bảng mới được xét, dù nội dung đoạn văn khớp regex.
    assert not tables.is_signature_table(P("TM. THỦ TƯỚNG"))


def test_filter_middle_tables_loai_bang_dinh_kem():
    blocks = [
        P("Điều 1. Phạm vi điều chỉnh"),
        T("FILE ĐƯỢC ĐÍNH KÈM THEO VĂN BẢN"),
        T("| Vùng | Mức |\n| --- | --- |\n| I | 5.000 |"),
    ]
    kept, warnings = tables.filter_middle_tables(blocks)
    assert kept == [blocks[0], blocks[2]]
    assert [w.code for w in warnings] == ["dropped_attachment_table"]


def test_filter_middle_tables_khong_co_bang_dinh_kem_giu_nguyen():
    blocks = [P("Điều 1. A"), T("| Vùng | Mức |\n| --- | --- |\n| I | 5.000 |")]
    kept, warnings = tables.filter_middle_tables(blocks)
    assert kept == blocks
    assert warnings == []


# ==========================================================================
# emitter.py -- không đổi so với bản cũ, chỉ đổi signature (không còn
# refs/footnote_map/preamble_end -- front/back matter đã được cắt từ trước).
# ==========================================================================


def test_emit_anh_xa_heading_co_ban():
    blocks = [
        P("Chương I"),
        P("NHỮNG QUY ĐỊNH CHUNG"),
        P("Điều 1. Phạm vi điều chỉnh"),
        P("1. Nội dung khoản một."),
        P("a) Điểm a của khoản một."),
    ]
    parts = emitter.emit(blocks)
    assert parts[0] == "## Chương I. NHỮNG QUY ĐỊNH CHUNG"
    assert parts[1] == "#### Điều 1. Phạm vi điều chỉnh"
    assert parts[2] == "##### Khoản 1"
    assert parts[3] == "Nội dung khoản một."
    assert parts[4] == "a) Điểm a của khoản một."


def test_emit_phan_ghep_tieu_de_va_khong_o_trong_phu_luc():
    blocks = [P("Phần thứ nhất"), P("QUY ĐỊNH CHUNG"), P("1. Nội dung.")]
    parts = emitter.emit(blocks)
    assert parts[0] == "# Phần thứ nhất. QUY ĐỊNH CHUNG"
    assert parts[1] == "##### Khoản 1"
    assert parts[2] == "Nội dung."


def test_emit_bang_duoc_xuat_nguyen_van():
    table_text = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    blocks = [P("Điều 1. A"), T(table_text)]
    parts = emitter.emit(blocks)
    assert parts[1] == table_text


def test_emit_phu_luc_khoan_ngan_gop_vao_heading():
    blocks = [
        P("PHỤ LỤC"),
        P("DANH MỤC TỈNH THÀNH"),
        P("28. Thành phố Hồ Chí Minh"),
    ]
    parts = emitter.emit(blocks)
    assert parts[0] == "# PHỤ LỤC — DANH MỤC TỈNH THÀNH"
    assert parts[1] == "##### 28. Thành phố Hồ Chí Minh"


def test_emit_phu_luc_khoan_dai_khong_gop_vao_heading():
    long_content = "Nội dung dài " * 20  # > 120 ký tự
    blocks = [P("PHỤ LỤC"), P("BẢNG DANH MỤC"), P(f"1. {long_content}")]
    parts = emitter.emit(blocks)
    assert parts[1] == "##### Khoản 1"
    assert parts[2] == long_content


def test_emit_tach_phan_kem_theo_khoi_tieu_de_phu_luc():
    blocks = [
        P("PHỤ LỤC"),
        P("BẢNG LƯƠNG (Kèm theo Nghị định số 293/2025/NĐ-CP)"),
        P("1. Nội dung."),
    ]
    parts = emitter.emit(blocks)
    assert parts[0] == "# PHỤ LỤC — BẢNG LƯƠNG"
    assert parts[1] == "(Kèm theo Nghị định số 293/2025/NĐ-CP)"


def test_emit_ra_khoi_phu_luc_khi_gap_chuong_khoan_khong_gop_nua():
    blocks = [
        P("PHỤ LỤC"),
        P("DANH MỤC"),
        P("28. Hà Nội"),
        P("Chương I"),
        P("PHẦN MỚI"),
        P("1. Nội dung."),
    ]
    parts = emitter.emit(blocks)
    assert parts[-2] == "##### Khoản 1"
    assert parts[-1] == "Nội dung."


# --- Regression: marker "[n]" dính liền phá RE_KHOAN/RE_DIEU nếu không --
# gỡ trước khi khớp heading (formatting_spec.md mục 1.1, "Lưu ý quan
# trọng"; xem thêm test strip_markers ở patterns.py) --------------------


def test_emit_go_marker_truoc_khi_khop_khoan():
    blocks = [
        P("Điều 1. Phạm vi điều chỉnh"),
        P("1.[2] Bảo hiểm y tế là hình thức bắt buộc."),
    ]
    parts = emitter.emit(blocks)
    assert parts[1] == "##### Khoản 1"
    assert parts[2] == "Bảo hiểm y tế là hình thức bắt buộc."


def test_emit_go_marker_truoc_khi_khop_dieu():
    blocks = [P("Điều 7a.[3] Bảo hiểm xã hội")]
    parts = emitter.emit(blocks)
    assert parts[0] == "#### Điều 7a. Bảo hiểm xã hội"


def test_emit_khong_go_marker_trong_bang():
    # Bảng KHÔNG bị strip -- marker còn sót trong bảng là dấu hiệu bất
    # thường, validator.py cảnh báo riêng (orphan_footnote), emitter không
    # tự ý sửa nội dung bảng.
    table_text = "| a | [2] |\n| --- | --- |\n| 1 | 2 |"
    blocks = [P("Điều 1. A"), T(table_text)]
    parts = emitter.emit(blocks)
    assert parts[1] == table_text


# ==========================================================================
# validator.py -- chỉ chạy trên markdown vùng nội dung ở giữa, không còn
# khối YAML/blockquote chú thích để bóc tách.
# ==========================================================================


def test_validate_khong_co_heading_canh_bao():
    warnings = validator.validate("Một đoạn văn thường không có heading nào.")
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


def test_validate_dieu_co_bang_khong_bi_bao_rong():
    markdown = "#### Điều 1. A\n\n<table>\n<tr><td>x</td></tr>\n</table>"
    warnings = validator.validate(markdown)
    assert not any(w.code == "empty_dieu" for w in warnings)


def test_validate_heading_dai_bat_thuong():
    markdown = "#### Điều 1. " + "A" * 200
    warnings = validator.validate(markdown)
    assert any(w.code == "suspicious_heading_length" for w in warnings)


def test_validate_marker_con_sot_bi_canh_bao_orphan_footnote():
    # emitter._strip_block_markers không strip trong bảng (chỉ paragraph) --
    # nếu marker còn sót ở bất kỳ đâu trong output, đó là bất thường cần rà
    # lại thủ công (formatting_spec.md mục 1.1).
    markdown = "#### Điều 1. A\n\nNội dung còn sót [5] marker."
    warnings = validator.validate(markdown)
    matching = [w for w in warnings if w.code == "orphan_footnote"]
    assert len(matching) == 1
    assert matching[0].detail == "[5]"


def test_validate_nhieu_marker_con_sot_moi_marker_mot_canh_bao():
    markdown = "Nội dung [1] và [2] còn sót."
    warnings = validator.validate(markdown)
    matching = [w for w in warnings if w.code == "orphan_footnote"]
    assert [w.detail for w in matching] == ["[1]", "[2]"]


def test_validate_khong_con_marker_khong_bi_canh_bao_orphan_footnote():
    markdown = "#### Điều 1. A\n\nNội dung sạch không có marker."
    warnings = validator.validate(markdown)
    assert not any(w.code == "orphan_footnote" for w in warnings)


# ==========================================================================
# frontmatter.py
# ==========================================================================


def test_frontmatter_find_boundary_tra_ve_chi_so_dau_tien_la_heading():
    blocks = [P("CHÍNH PHỦ"), P("Điều 1. Phạm vi.")]
    assert frontmatter.find_boundary(blocks) == 1


def test_frontmatter_find_boundary_khong_co_heading_tra_ve_do_dai_list():
    blocks = [P("CHÍNH PHỦ"), P("Số: 1/2025/NĐ-CP")]
    assert frontmatter.find_boundary(blocks) == len(blocks)


def test_frontmatter_find_boundary_bo_qua_dong_danh_so_o_dan_nhap():
    blocks = [
        P("Căn cứ Luật Nhà giáo số 73/2025/QH15..."),
        P("1. Luật Nhà giáo số 73/2025/QH15."),
        P("Chương I"),
        P("NHỮNG QUY ĐỊNH CHUNG"),
    ]
    assert frontmatter.find_boundary(blocks) == 2


def test_frontmatter_find_boundary_chi_xet_paragraph():
    blocks = [T("Điều 1. Trong bảng không phải heading thật.")]
    assert frontmatter.find_boundary(blocks) == len(blocks)


def test_convert_frontmatter_rong_khong_goi_llm(monkeypatch: pytest.MonkeyPatch):
    _forbid_llm_calls(monkeypatch)
    markdown, warnings = frontmatter.convert_frontmatter([])
    assert markdown == ""
    assert warnings == []


def test_convert_frontmatter_thanh_cong(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        llm_client, "convert_to_markdown", lambda *a, **k: "**CHÍNH PHỦ**"
    )
    markdown, warnings = frontmatter.convert_frontmatter([P("CHÍNH PHỦ", bold=True)])
    assert markdown == "**CHÍNH PHỦ**"
    assert warnings == []


def test_convert_frontmatter_gemini_loi_phat_canh_bao():
    # Fixture `_default_llm_stub` đã trả None mặc định.
    markdown, warnings = frontmatter.convert_frontmatter([P("CHÍNH PHỦ")])
    assert markdown == ""
    assert [w.code for w in warnings] == ["llm_frontmatter_conversion_failed"]


def test_convert_frontmatter_prompt_chua_noi_dung_block(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, str] = {}

    def fake(prompt: str, *, max_retries: int | None = None) -> str:
        captured["prompt"] = prompt
        return "kết quả"

    monkeypatch.setattr(llm_client, "convert_to_markdown", fake)
    frontmatter.convert_frontmatter([P("CHÍNH PHỦ", bold=True)])
    assert "**CHÍNH PHỦ**" in captured["prompt"]


# ==========================================================================
# backmatter.py
# ==========================================================================


def test_backmatter_find_boundary_lay_bang_ky_cuoi_cung():
    blocks = [T("TM. THỦ TƯỚNG"), P("Ở giữa."), T("Nơi nhận:")]
    assert backmatter.find_boundary(blocks) == 2


def test_backmatter_find_boundary_khong_co_bang_ky_tra_ve_none():
    blocks = [P("Nội dung.")]
    assert backmatter.find_boundary(blocks) is None


def test_backmatter_split_backmatter_co_noi_dung_sau_bang_ky():
    blocks = [P("Ở giữa."), T("TM. THỦ TƯỚNG"), P("[1] Ghi chú.")]
    middle, back, warnings = backmatter.split_backmatter(blocks)
    assert middle == [blocks[0]]
    assert back == [blocks[2]]
    assert [w.code for w in warnings] == ["dropped_noi_nhan_table"]


def test_backmatter_split_backmatter_khong_con_block_sau_bang_ky():
    blocks = [P("Ở giữa."), T("TM. THỦ TƯỚNG")]
    middle, back, warnings = backmatter.split_backmatter(blocks)
    assert middle == [blocks[0]]
    assert back == []
    assert [w.code for w in warnings] == ["dropped_noi_nhan_table"]


def test_backmatter_split_backmatter_khong_co_bang_ky():
    blocks = [P("Ở giữa.")]
    middle, back, warnings = backmatter.split_backmatter(blocks)
    assert middle == blocks
    assert back == []
    assert warnings == []


def test_convert_backmatter_rong_khong_goi_llm(monkeypatch: pytest.MonkeyPatch):
    _forbid_llm_calls(monkeypatch)
    markdown, warnings = backmatter.convert_backmatter([])
    assert markdown == ""
    assert warnings == []


def test_convert_backmatter_thanh_cong(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        llm_client, "convert_to_markdown", lambda *a, **k: "[1] Ghi chú sửa đổi."
    )
    markdown, warnings = backmatter.convert_backmatter([P("[1] Ghi chú sửa đổi.")])
    assert markdown == "[1] Ghi chú sửa đổi."
    assert warnings == []


def test_convert_backmatter_gemini_loi_phat_canh_bao():
    markdown, warnings = backmatter.convert_backmatter([P("[1] Ghi chú.")])
    assert markdown == ""
    assert [w.code for w in warnings] == ["llm_backmatter_conversion_failed"]


def test_convert_backmatter_prompt_giu_nguyen_marker_khong_bi_strip(
    monkeypatch: pytest.MonkeyPatch,
):
    # Khác với vùng nội dung ở giữa (strip_markers), marker "[n]" ở back
    # matter là số thứ tự chú thích thật (vd. "[1] Luật Công nghiệp...") --
    # PHẢI giữ nguyên khi gửi cho Gemini (formatting_spec.md mục 1.1, "Lưu ý
    # quan trọng").
    captured: dict[str, str] = {}

    def fake(prompt: str, *, max_retries: int | None = None) -> str:
        captured["prompt"] = prompt
        return "kết quả"

    monkeypatch.setattr(llm_client, "convert_to_markdown", fake)
    backmatter.convert_backmatter([P("[1] Luật Công nghiệp có hiệu lực từ...")])
    assert "[1] Luật Công nghiệp có hiệu lực từ..." in captured["prompt"]


# ==========================================================================
# llm_client.py -- Gemini (google-genai), mock hoàn toàn (không gọi API thật)
# ==========================================================================


def test_client_gemini_dung_tham_so_tu_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-khong-goi-thuc-te")
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, *, api_key: str, http_options) -> None:
            captured["api_key"] = api_key
            captured["timeout"] = http_options.timeout

    monkeypatch.setattr(llm_client.genai, "Client", FakeClient)
    llm_client._client.cache_clear()
    try:
        llm_client._client()
        assert captured["api_key"] == "fake-key-khong-goi-thuc-te"
        assert captured["timeout"] == 30 * 1000
    finally:
        llm_client._client.cache_clear()


def test_convert_to_markdown_thanh_cong_goi_dung_tham_so(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-khong-goi-thuc-te")
    fake_client = Mock()
    fake_client.models.generate_content.return_value = Mock(text="  Kết quả markdown  ")
    monkeypatch.setattr(llm_client, "_client", lambda: fake_client)

    result = _REAL_CONVERT_TO_MARKDOWN("prompt nội dung")

    assert result == "Kết quả markdown"
    kwargs = fake_client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "gemini-3.6-flash"
    assert kwargs["contents"] == "prompt nội dung"


def test_convert_to_markdown_loi_roi_thu_lai_thanh_cong(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-khong-goi-thuc-te")
    fake_client = Mock()
    fake_client.models.generate_content.side_effect = [
        RuntimeError("lỗi mạng giả lập"),
        Mock(text="Kết quả sau khi thử lại"),
    ]
    monkeypatch.setattr(llm_client, "_client", lambda: fake_client)

    result = _REAL_CONVERT_TO_MARKDOWN("prompt")

    assert result == "Kết quả sau khi thử lại"
    assert fake_client.models.generate_content.call_count == 2


def test_convert_to_markdown_het_so_lan_thu_tra_ve_none(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-khong-goi-thuc-te")
    fake_client = Mock()
    fake_client.models.generate_content.side_effect = RuntimeError("lỗi giả lập")
    monkeypatch.setattr(llm_client, "_client", lambda: fake_client)

    result = _REAL_CONVERT_TO_MARKDOWN("prompt")

    assert result is None
    assert fake_client.models.generate_content.call_count == 2  # max_retries mặc định


def test_convert_to_markdown_ket_qua_rong_bi_coi_la_loi_va_thu_lai(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-khong-goi-thuc-te")
    fake_client = Mock()
    fake_client.models.generate_content.side_effect = [
        Mock(text=""),
        Mock(text="Nội dung thật"),
    ]
    monkeypatch.setattr(llm_client, "_client", lambda: fake_client)

    result = _REAL_CONVERT_TO_MARKDOWN("prompt")

    assert result == "Nội dung thật"
    assert fake_client.models.generate_content.call_count == 2


def test_convert_to_markdown_max_retries_ghi_de_settings(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-khong-goi-thuc-te")
    fake_client = Mock()
    fake_client.models.generate_content.side_effect = RuntimeError("lỗi giả lập")
    monkeypatch.setattr(llm_client, "_client", lambda: fake_client)

    result = _REAL_CONVERT_TO_MARKDOWN("prompt", max_retries=1)

    assert result is None
    assert fake_client.models.generate_content.call_count == 1


def test_convert_to_markdown_thieu_gemini_api_key_tra_ve_none_khong_raise(
    monkeypatch: pytest.MonkeyPatch,
):
    """Không có `GEMINI_API_KEY` (đúng thực trạng CI) -- `LLMSettings()`
    raise `ValidationError`, `convert_to_markdown` phải bắt và trả `None`,
    không để lộ exception (mục 5, 7 spec: lỗi Gemini không bao giờ chặn
    pipeline)."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        LLMSettings, "model_config", {**LLMSettings.model_config, "env_file": None}
    )

    result = _REAL_CONVERT_TO_MARKDOWN("prompt")
    assert result is None


# ==========================================================================
# pipeline.py -- _compose_markdown (ghép 3 phần)
# ==========================================================================


def test_compose_markdown_day_du_ba_phan():
    markdown = pipeline._compose_markdown("FRONT", "MIDDLE", "BACK")
    assert markdown == "FRONT\n\nMIDDLE\n\n---\n\nBACK\n"


def test_compose_markdown_khong_co_front_matter():
    markdown = pipeline._compose_markdown("", "MIDDLE", "")
    assert markdown == "MIDDLE\n"


def test_compose_markdown_khong_co_back_matter():
    markdown = pipeline._compose_markdown("FRONT", "MIDDLE", "")
    assert markdown == "FRONT\n\nMIDDLE\n"


# ==========================================================================
# pipeline.py -- convert_docx_to_markdown (integration DOCX thật, LLM mock)
# ==========================================================================


def _build_docx(path: Path, items: list[tuple[str, object]]) -> None:
    """DSL nhỏ để dựng DOCX test: ("p", text) | ("p_bold", text) | ("table", rows)."""
    document = Document()
    for kind, payload in items:
        if kind == "p":
            document.add_paragraph(str(payload))
        elif kind == "p_bold":
            paragraph = document.add_paragraph()
            run = paragraph.add_run(str(payload))
            run.bold = True
        elif kind == "table":
            rows = payload
            assert isinstance(rows, list)
            table = document.add_table(rows=len(rows), cols=len(rows[0]))
            for row_index, row in enumerate(rows):
                for col_index, value in enumerate(row):
                    table.cell(row_index, col_index).text = value
        else:
            raise ValueError(f"kind không hỗ trợ: {kind}")
    document.save(path)


def test_convert_docx_to_markdown_ghep_du_front_middle_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _sequential_llm(monkeypatch, ["**CHÍNH PHỦ**", "[1] Ghi chú sửa đổi."])
    docx_path = tmp_path / "full.docx"
    _build_docx(
        docx_path,
        [
            ("p_bold", "CHÍNH PHỦ"),
            ("p", "Điều 1. Phạm vi điều chỉnh"),
            ("p", "1. Nội dung khoản một."),
            ("table", [["TM. THỦ TƯỚNG"], ["Nguyễn Văn A"]]),
            ("p", "[1] Ghi chú sửa đổi."),
        ],
    )

    result = convert_docx_to_markdown(docx_path)

    assert result.markdown == (
        "**CHÍNH PHỦ**\n\n"
        "#### Điều 1. Phạm vi điều chỉnh\n\n"
        "##### Khoản 1\n\n"
        "Nội dung khoản một.\n\n"
        "---\n\n"
        "[1] Ghi chú sửa đổi.\n"
    )
    assert "TM. THỦ TƯỚNG" not in result.markdown
    codes = {w.code for w in result.warnings}
    assert "dropped_noi_nhan_table" in codes
    assert "llm_frontmatter_conversion_failed" not in codes
    assert "llm_backmatter_conversion_failed" not in codes


def test_convert_docx_to_markdown_khong_co_back_matter_khong_goi_llm_hai_lan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _sequential_llm(monkeypatch, ["FRONT_MD"])
    docx_path = tmp_path / "no_back.docx"
    _build_docx(
        docx_path,
        [
            ("p", "CHÍNH PHỦ"),
            ("p", "Điều 1. Phạm vi điều chỉnh"),
            ("p", "1. Nội dung khoản một."),
        ],
    )

    result = convert_docx_to_markdown(docx_path)

    assert result.markdown == (
        "FRONT_MD\n\n#### Điều 1. Phạm vi điều chỉnh\n\n##### Khoản 1\n\nNội dung khoản một.\n"
    )
    assert len(calls) == 1  # chỉ front matter gọi Gemini
    codes = {w.code for w in result.warnings}
    assert "llm_backmatter_conversion_failed" not in codes
    assert "dropped_noi_nhan_table" not in codes


def test_convert_docx_to_markdown_gemini_loi_bo_qua_front_matter_khong_fail(
    tmp_path: Path,
):
    # Fixture `_default_llm_stub` đã trả None mặc định (Gemini "lỗi").
    docx_path = tmp_path / "front_fail.docx"
    _build_docx(
        docx_path,
        [
            ("p", "CHÍNH PHỦ"),
            ("p", "Điều 1. Phạm vi điều chỉnh"),
            ("p", "1. Nội dung."),
        ],
    )

    result = convert_docx_to_markdown(docx_path)

    assert (
        result.markdown
        == "#### Điều 1. Phạm vi điều chỉnh\n\n##### Khoản 1\n\nNội dung.\n"
    )
    assert any(w.code == "llm_frontmatter_conversion_failed" for w in result.warnings)


def test_convert_docx_to_markdown_gemini_loi_bo_qua_back_matter_khong_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _sequential_llm(monkeypatch, ["FRONT_MD", None])
    docx_path = tmp_path / "back_fail.docx"
    _build_docx(
        docx_path,
        [
            ("p", "CHÍNH PHỦ"),
            ("p", "Điều 1. Phạm vi điều chỉnh"),
            ("p", "1. Nội dung."),
            ("table", [["TM. THỦ TƯỚNG"], ["Nguyễn Văn A"]]),
            ("p", "[1] Ghi chú."),
        ],
    )

    result = convert_docx_to_markdown(docx_path)

    assert result.markdown == (
        "FRONT_MD\n\n#### Điều 1. Phạm vi điều chỉnh\n\n##### Khoản 1\n\nNội dung.\n"
    )
    assert "---" not in result.markdown
    codes = {w.code for w in result.warnings}
    assert "llm_backmatter_conversion_failed" in codes
    assert "dropped_noi_nhan_table" in codes


def test_convert_docx_to_markdown_khong_co_front_matter_khong_goi_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _forbid_llm_calls(monkeypatch)
    docx_path = tmp_path / "no_front.docx"
    _build_docx(
        docx_path,
        [
            ("p", "Điều 1. Phạm vi điều chỉnh"),
            ("p", "1. Nội dung."),
        ],
    )

    result = convert_docx_to_markdown(docx_path)

    assert (
        result.markdown
        == "#### Điều 1. Phạm vi điều chỉnh\n\n##### Khoản 1\n\nNội dung.\n"
    )
    assert result.warnings == []


def test_convert_docx_to_markdown_file_rong_bao_loi_ro_rang(tmp_path: Path):
    docx_path = tmp_path / "rong.docx"
    Document().save(docx_path)

    with pytest.raises(ValueError, match="Không trích xuất được nội dung"):
        convert_docx_to_markdown(docx_path)


def test_convert_docx_to_markdown_deterministic_khi_llm_on_dinh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(llm_client, "convert_to_markdown", lambda *a, **k: "FRONT_MD")
    docx_path = tmp_path / "det.docx"
    _build_docx(
        docx_path,
        [("p", "CHÍNH PHỦ"), ("p", "Điều 1. A"), ("p", "1. Nội dung.")],
    )

    first = convert_docx_to_markdown(docx_path).markdown
    second = convert_docx_to_markdown(docx_path).markdown
    assert first == second


# ==========================================================================
# pipeline.py -- integration trên corpus thật (data/raw/*.docx), LLM mock
# ==========================================================================


@pytest.mark.skipif(not RAW_FILES, reason="data/raw/*.docx không tồn tại")
def test_data_raw_co_du_6_file():
    assert len(RAW_FILES) == 6


def _heading_lines(markdown: str) -> list[str]:
    return [line for line in markdown.splitlines() if line.startswith("#")]


@pytest.mark.skipif(not RAW_FILES, reason="data/raw/*.docx không tồn tại")
@pytest.mark.parametrize("path", RAW_FILES, ids=lambda p: p.stem)
def test_convert_heading_khop_voi_tham_chieu_data_markdown(path: Path):
    """Mục 7 spec: heading mapping cho phần nội dung ở giữa phải khớp với
    `data/markdown/*.md` (tham chiếu, không phải golden-file chính thức).
    Front/back matter do Gemini sinh không so khớp -- ở đây LLM bị mock trả
    `None`, nên chỉ heading của vùng nội dung ở giữa được so sánh (front/back
    matter không chứa heading nào, mục 1.1 spec, nên không ảnh hưởng danh
    sách heading)."""
    reference_path = REFERENCE_MD_DIR / f"{path.stem}.md"
    if not reference_path.exists():
        pytest.skip(f"Không có tham chiếu cho {path.stem}")

    result = convert_docx_to_markdown(path)
    actual_headings = _heading_lines(result.markdown)
    expected_headings = _heading_lines(reference_path.read_text(encoding="utf-8"))
    assert actual_headings == expected_headings


@pytest.mark.skipif(not RAW_FILES, reason="data/raw/*.docx không tồn tại")
@pytest.mark.parametrize("path", RAW_FILES, ids=lambda p: p.stem)
def test_convert_la_ham_thuan_deterministic(path: Path):
    """NT: cùng input phải ra byte-for-byte cùng output, chạy lại nhiều lần
    (LLM mock trả None ổn định, nên toàn bộ output -- không chỉ vùng giữa --
    deterministic ở đây)."""
    first = convert_docx_to_markdown(path).markdown
    second = convert_docx_to_markdown(path).markdown
    assert first == second


@pytest.mark.skipif(not RAW_FILES, reason="data/raw/*.docx không tồn tại")
def test_convert_khong_co_ma_canh_bao_moi():
    """Mục 7 spec: không phát sinh mã QC warning mới ngoài `QcWarningCode`."""
    unknown: set[str] = set()
    for path in RAW_FILES:
        result = convert_docx_to_markdown(path)
        unknown |= {w.code for w in result.warnings} - KNOWN_WARNING_CODES
    assert not unknown, f"Mã QC warning mới, chưa có trong QcWarningCode: {unknown}"


@pytest.mark.skipif(not RAW_FILES, reason="data/raw/*.docx không tồn tại")
def test_convert_file_khong_ton_tai_bao_loi_ro_rang():
    with pytest.raises(PackageNotFoundError):
        convert_docx_to_markdown(RAW_DIR / "khong_ton_tai.docx")


# ==========================================================================
# pipeline.py -- write_atomic / convert_directory
# ==========================================================================


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

    Ép NO_COLOR + COLUMNS rộng: Rich (dùng bởi Typer để render --help) tô màu
    ANSI và tự xuống dòng theo bề rộng terminal, nên trên CI (không phải TTY,
    COLUMNS khác máy dev) "raw-dir" có thể bị mã màu hoặc dấu xuống dòng chen
    vào giữa.
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
    plain_stdout = _ANSI_ESCAPE_RE.sub("", completed.stdout)
    assert "raw-dir" in plain_stdout
