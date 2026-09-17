"""Unit test cho `chunking/parser.py` (chunking_spec.md mục 2, 3, 4.6, 10).

Dựng `DocumentTree` từ markdown tổng hợp (không phụ thuộc `data/markdown/`
thật, trừ 1 nhóm test đối chiếu trực tiếp với file thật để khoá hành vi lại).

Mọi fixture (trừ nhóm đối chiếu file thật) mô phỏng đúng quy ước THẬT của
`formatting/` (không còn YAML front matter -- `formatting/` không sinh field
cấu trúc nào, xem `formatting_spec.md` mục 1.1): 1 dòng loại văn bản (vd.
"LUẬT"), rồi tới đúng 1 heading `#` (H1) là tên văn bản, đứng trước heading
cấu trúc Phần/Chương/Mục/Điều/Khoản đầu tiên -- xem `parser.py` docstring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from production_legal_qa_rag.chunking import splitter
from production_legal_qa_rag.chunking.parser import parse_markdown
from production_legal_qa_rag.chunking.pipeline import convert_markdown_to_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdown"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _doc(body: str, *, doc_type: str = "LUẬT", title: str = "VĂN BẢN MẪU") -> str:
    """Dựng markdown giống output THẬT của `formatting/`: 1 dòng loại văn bản
    ngay trước heading `#` tên văn bản (frontmatter, mục 2), rồi tới `body`
    (cấu trúc Phần/Chương/Mục/Điều/Khoản)."""
    return f"**{doc_type}**\n\n# {title}\n\n{body}"


# ==========================================================================
# Frontmatter / source_document (mục 2, 4.6)
# ==========================================================================


def test_source_document_noi_doan_truoc_h1_voi_heading(tmp_path: Path):
    text = _doc(
        "#### Điều 1. Phạm vi điều chỉnh\n\n##### Khoản 1\n\nNội dung khoản 1.\n",
        doc_type="LUẬT",
        title="BẢO HIỂM XÃ HỘI",
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.source_document == "LUẬT BẢO HIỂM XÃ HỘI"


def test_source_document_bo_dau_nhan_manh_markdown_cua_doan_truoc(tmp_path: Path):
    text = (
        "**NGHỊ ĐỊNH**\n\n# QUY ĐỊNH VỀ ABC\n\n#### Điều 1. X\n\n##### Khoản 1\n\nY.\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.source_document == "NGHỊ ĐỊNH QUY ĐỊNH VỀ ABC"


def test_source_document_chi_lay_heading_neu_khong_co_doan_truoc(tmp_path: Path):
    text = "# VĂN BẢN MẪU\n\n#### Điều 1. X\n\n##### Khoản 1\n\nY.\n"
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.source_document == "VĂN BẢN MẪU"


def test_source_document_fallback_ten_file_khi_khong_co_heading_h1(tmp_path: Path):
    text = "Không có heading H1 nào trong toàn bộ file.\n\n#### Điều 1. X\n\n##### Khoản 1\n\nY.\n"
    path = _write(tmp_path, "khong_co_h1.md", text)
    tree = parse_markdown(path)
    assert tree.source_document == "khong_co_h1"


def test_frontmatter_content_giu_nguyen_van_doan_mo_dau(tmp_path: Path):
    text = (
        "**LUẬT**\n\n# BẢO HIỂM XÃ HỘI\n\n"
        "Căn cứ Hiến pháp...\n\nQuốc hội ban hành Luật này.\n\n"
        "#### Điều 1. X\n\n##### Khoản 1\n\nY.\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.frontmatter_content is not None
    assert "LUẬT" in tree.frontmatter_content
    assert "BẢO HIỂM XÃ HỘI" in tree.frontmatter_content
    assert "Căn cứ Hiến pháp..." in tree.frontmatter_content
    assert "Quốc hội ban hành Luật này." in tree.frontmatter_content
    # Chỉ phần trước heading cấu trúc đầu tiên, không lẫn nội dung Điều 1.
    assert "Điều 1" not in tree.frontmatter_content


def test_khong_co_frontmatter_khi_heading_cau_truc_ngay_dau_file(tmp_path: Path):
    text = "#### Điều 1. X\n\n##### Khoản 1\n\nY.\n"
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.frontmatter_content is None


# ==========================================================================
# Cây Phần/Chương/Mục/Điều/Khoản (mục 3)
# ==========================================================================


def test_breadcrumb_prefix_day_du_phan_chuong_muc_dieu(tmp_path: Path):
    text = _doc(
        "# Phần I\n\n## Chương II\n\n### Mục 3\n\n#### Điều 5. Tên điều\n\n"
        "##### Khoản 1\n\nNội dung.\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.khoans[0].breadcrumb_prefix == (
        "LUẬT VĂN BẢN MẪU - Phần I - Chương II - Mục 3 - Điều 5. Tên điều"
    )


def test_bo_qua_cap_khong_ton_tai_trong_van_ban(tmp_path: Path):
    # Văn bản không có Phần/Mục -> breadcrumb bỏ qua các cấp đó (mục 3).
    text = _doc("## Chương I\n\n#### Điều 1. Tên điều\n\n##### Khoản 1\n\nNội dung.\n")
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.khoans[0].breadcrumb_prefix == (
        "LUẬT VĂN BẢN MẪU - Chương I - Điều 1. Tên điều"
    )


def test_dieu_khong_co_ten_khong_them_dau_cham_thua(tmp_path: Path):
    text = _doc("#### Điều 1.\n\n##### Khoản 1\n\nNội dung.\n")
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.khoans[0].breadcrumb_prefix == "LUẬT VĂN BẢN MẪU - Điều 1"


def test_reset_chuong_muc_khi_sang_phan_moi(tmp_path: Path):
    text = _doc(
        "# Phần I\n\n## Chương I\n\n#### Điều 1. Điều đầu\n\n##### Khoản 1\n\n"
        "Nội dung 1.\n\n# Phần II\n\n#### Điều 2. Điều sau\n\n##### Khoản 1\n\n"
        "Nội dung 2.\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    prefixes = [khoan.breadcrumb_prefix for khoan in tree.khoans]
    assert prefixes == [
        "LUẬT VĂN BẢN MẪU - Phần I - Chương I - Điều 1. Điều đầu",
        "LUẬT VĂN BẢN MẪU - Phần II - Điều 2. Điều sau",
    ]


# ==========================================================================
# Khoản ngầm định (Điều không chia Khoản / đoạn mở đầu trước Khoản đầu tiên)
# ==========================================================================


def test_dieu_khong_co_khoan_con_thanh_khoan_ngam_dinh(tmp_path: Path):
    text = _doc(
        "#### Điều 1. Điều không chia khoản\n\n"
        "Nội dung nằm thẳng dưới Điều, không có heading Khoản.\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert len(tree.khoans) == 1
    khoan = tree.khoans[0]
    assert khoan.khoan_number is None
    assert khoan.content == "Nội dung nằm thẳng dưới Điều, không có heading Khoản."
    # Breadcrumb dừng ở cấp Điều, không có "- Khoản".
    assert khoan.breadcrumb_prefix == "LUẬT VĂN BẢN MẪU - Điều 1. Điều không chia khoản"


def test_doan_mo_dau_truoc_khoan_dau_tien_thanh_khoan_ngam_dinh(tmp_path: Path):
    text = _doc(
        "#### Điều 1. Điều có đoạn mở đầu\n\n"
        "Đoạn mở đầu trước Khoản đầu tiên.\n\n"
        "##### Khoản 1\n\nNội dung khoản 1.\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert len(tree.khoans) == 2
    assert tree.khoans[0].khoan_number is None
    assert tree.khoans[0].content == "Đoạn mở đầu trước Khoản đầu tiên."
    assert tree.khoans[1].khoan_number == "1"
    assert tree.khoans[1].content == "Nội dung khoản 1."


def test_noi_dung_truoc_dieu_dau_tien_bi_bo_qua(tmp_path: Path):
    # Nội dung nằm trực tiếp dưới Phần/Chương/Mục (chưa vào Điều/Khoản nào)
    # bị bỏ qua -- ngoài phạm vi "1 chunk = 1 Khoản" (mục 1).
    text = _doc(
        "# Phụ Lục\n\n(Kèm theo Nghị định số ABC)\n\n"
        "##### 1. Thành phố Hà Nội\n\nNội dung địa bàn.\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert len(tree.khoans) == 1
    assert "Kèm theo" not in tree.khoans[0].content


# ==========================================================================
# Khoản trong Phụ Lục (heading gộp "N. tên")
# ==========================================================================


def test_khoan_phu_luc_dang_gop_khong_can_dieu_bao_ngoai(tmp_path: Path):
    text = _doc(
        "# PHỤ LỤC\n\n##### 1. Thành phố Hà Nội\n\n"
        "Vùng I gồm các phường trung tâm.\n\n"
        "##### 2. Tỉnh Cao Bằng\n\nVùng III gồm các xã còn lại.\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert len(tree.khoans) == 2
    assert tree.khoans[0].khoan_number == "1"
    assert tree.khoans[0].content.startswith("Thành phố Hà Nội")
    assert tree.khoans[1].khoan_number == "2"
    assert tree.khoans[1].content.startswith("Tỉnh Cao Bằng")


# ==========================================================================
# Blockquote (chú thích sửa đổi) bị loại bỏ
# ==========================================================================


def test_blockquote_sua_doi_bi_loai_khoi_noi_dung(tmp_path: Path):
    text = _doc(
        "#### Điều 1. Tên điều\n\n##### Khoản 1\n\n"
        "Nội dung chính của khoản.\n\n"
        "> **Sửa đổi:** Khoản này được sửa đổi theo Luật số X.\n>\n"
        "> _(Trích dẫn đầy đủ ở cuối văn bản.)_\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert len(tree.khoans) == 1
    content = tree.khoans[0].content
    assert content == "Nội dung chính của khoản."
    assert "Sửa đổi" not in content
    assert "Trích dẫn" not in content


# ==========================================================================
# Bảng markdown / bảng HTML single-row
# ==========================================================================


def test_khoan_co_bang_markdown_has_table_va_raw_table(tmp_path: Path):
    text = _doc(
        "#### Điều 3. Mức lương tối thiểu\n\n##### Khoản 1\n\n"
        "Quy định mức lương tối thiểu theo vùng như sau:\n\n"
        "| Vùng | Mức lương tối thiểu tháng<br>(Đơn vị: đồng/tháng) |\n"
        "| --- | --- |\n"
        "| Vùng I | 5.310.000 |\n"
        "| Vùng II | 4.730.000 |\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    khoan = tree.khoans[0]
    assert khoan.has_table is True
    assert khoan.raw_table is not None
    assert khoan.raw_table.startswith("| Vùng |")
    assert "Vùng II | 4.730.000 |" in khoan.raw_table
    # Nội dung tường thuật (câu dẫn) không lẫn dòng bảng.
    assert "|" not in khoan.content


def test_khoan_co_bang_html_single_row_has_table(tmp_path: Path):
    text = _doc(
        "#### Điều 55. Tiền lương làm thêm giờ\n\n##### Khoản 1\n\n"
        "Công thức tính như sau:\n\n"
        "<table>\n  <tbody>\n    <tr>\n      <td>Tiền lương làm thêm giờ</td>\n"
        "      <td>=</td>\n      <td>Đơn giá</td>\n    </tr>\n  </tbody>\n</table>\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    khoan = tree.khoans[0]
    assert khoan.has_table is True
    assert khoan.raw_table is not None
    assert khoan.raw_table.strip().lower().startswith("<table")


def test_khoan_khong_co_bang_has_table_false(tmp_path: Path):
    text = _doc("#### Điều 1. Tên điều\n\n##### Khoản 1\n\nKhông có bảng nào ở đây.\n")
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.khoans[0].has_table is False
    assert tree.khoans[0].raw_table is None


# ==========================================================================
# Backmatter (mục 4.6) -- marker THẬT là dòng "---" (khác mô tả "**[n]**"
# lỗi thời trong chunking_spec.md, xem patterns.py/parser.py docstring).
# ==========================================================================


def test_backmatter_tach_rieng_khoi_khoan_cuoi_cung(tmp_path: Path):
    text = _doc(
        "#### Điều 1. Tên điều\n\n##### Khoản 1\n\nNội dung khoản cuối cùng.\n\n"
        "---\n\n"
        "[1] Điều 41 của Luật khác quy định như sau:\n\n"
        "“Điều 41. Hiệu lực thi hành”.\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert len(tree.khoans) == 1
    assert tree.khoans[0].content == "Nội dung khoản cuối cùng."
    assert "Nội dung khoản cuối cùng" not in (tree.backmatter_content or "")
    assert tree.backmatter_content is not None
    assert "[1] Điều 41" in tree.backmatter_content
    assert "Điều 41. Hiệu lực thi hành" in tree.backmatter_content


def test_khong_co_backmatter_khi_khong_co_dong_gach_ngang_3_ky_tu(tmp_path: Path):
    text = _doc("#### Điều 1. Tên điều\n\n##### Khoản 1\n\nNội dung.\n")
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.backmatter_content is None


def test_gach_ngang_trang_tri_dai_hon_3_ky_tu_khong_bi_hieu_nham_la_backmatter(
    tmp_path: Path,
):
    # "--------" (8 ký tự) trong frontmatter không khớp RE_BACKMATTER_SEPARATOR
    # (chỉ khớp đúng "---" 3 ký tự) -- không bị cắt nhầm.
    text = (
        "VĂN PHÒNG QUỐC HỘI\n\n--------\n\n**LUẬT**\n\n# VĂN BẢN MẪU\n\n"
        "#### Điều 1. Tên điều\n\n##### Khoản 1\n\nNội dung.\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.backmatter_content is None
    assert len(tree.khoans) == 1
    assert tree.khoans[0].content == "Nội dung."


# ==========================================================================
# Đối chiếu với file thật (khoá hành vi lại, chunking_spec.md mục 11)
# ==========================================================================


def test_parse_file_that_luong_toi_thieu_dieu_3_khoan_1_co_bang():
    path = MARKDOWN_DIR / "Quy định mức lương tối thiểu.md"
    if not path.exists():
        pytest.skip("data/markdown/Quy định mức lương tối thiểu.md không tồn tại")
    tree = parse_markdown(path)
    matches = [
        khoan
        for khoan in tree.khoans
        if khoan.breadcrumb_prefix.endswith("Điều 3. Mức lương tối thiểu")
        and khoan.khoan_number == "1"
    ]
    assert len(matches) == 1
    khoan = matches[0]
    assert khoan.has_table is True
    assert khoan.raw_table is not None
    assert khoan.raw_table.count("\n") >= 5  # header + dòng phân cách + 4 dòng dữ liệu
    for region in ("Vùng I", "Vùng II", "Vùng III", "Vùng IV"):
        assert region in khoan.raw_table


def test_parse_file_that_luat_bao_hiem_xa_hoi_co_backmatter():
    path = MARKDOWN_DIR / "Luật bảo hiểm xã hội.md"
    if not path.exists():
        pytest.skip("data/markdown/Luật bảo hiểm xã hội.md không tồn tại")
    tree = parse_markdown(path)
    assert tree.source_document == "LUẬT BẢO HIỂM XÃ HỘI"
    assert tree.backmatter_content is not None
    assert "Điều 41. Hiệu lực thi hành" in tree.backmatter_content
    # Khoản 15 Điều 141 (Khoản ngầm định gần nhất trước backmatter) không còn
    # lẫn nội dung backmatter (mục 4.6, mục 11).
    khoan_15 = [
        khoan
        for khoan in tree.khoans
        if khoan.breadcrumb_prefix.endswith("Điều 141. Quy định chuyển tiếp")
        and khoan.khoan_number == "15"
    ]
    assert len(khoan_15) == 1
    assert khoan_15[0].content == "Chính phủ quy định chi tiết Điều này."


# ==========================================================================
# Regression: Khoản lồng trong đoạn trích dẫn nguyên văn điều luật khác (vd.
# Điều 219 `Văn bản hợp nhất bộ luật lao động.md` trích Điều 54/55 Luật BHXH
# bằng chính heading cấp 5 "##### Khoản N", khiến các Khoản trích dẫn tự đánh
# số lại từ 1 -> trùng breadcrumb/chunk_id với Khoản thật). `quote_depth`
# (đếm độ sâu dấu ngoặc kép "“"/"”") gộp heading cấp 5 xuất hiện trong đoạn
# trích dẫn làm văn bản thường thay vì tạo `KhoanNode` mới.
# ==========================================================================


def _dieu_219_style_doc() -> str:
    return _doc(
        "#### Điều 219. Sửa đổi nhiều luật\n\n"
        "##### Khoản 1\n\n"
        "Sửa đổi Điều 54 như sau:\n\n"
        "a) Sửa đổi Điều 54 như sau:\n\n"
        "“Điều 54. Điều kiện\n\n"
        "##### Khoản 1\n\n"
        "Nội dung khoản 1 trích dẫn.\n\n"
        "##### Khoản 2\n\n"
        "Nội dung khoản 2 trích dẫn.”;\n\n"
        "b) Một điểm khác.\n\n"
        "##### Khoản 2\n\n"
        "Nội dung khoản 2 thật.\n"
    )


def test_khoan_long_trong_trich_dan_khong_tao_khoan_moi(tmp_path: Path):
    tree = parse_markdown(_write(tmp_path, "a.md", _dieu_219_style_doc()))
    # Chỉ 2 KhoanNode THẬT được tạo ra (Khoản 1, Khoản 2 của chính Điều 219)
    # -- các "Khoản 1"/"Khoản 2" bên trong đoạn trích dẫn KHÔNG tạo KhoanNode
    # riêng (tránh chunk_id trùng lặp khi sinh chunk ở splitter.py).
    assert [khoan.khoan_number for khoan in tree.khoans] == ["1", "2"]


def test_khoan_long_trong_trich_dan_duoc_gop_lam_van_ban_thuong(tmp_path: Path):
    tree = parse_markdown(_write(tmp_path, "a.md", _dieu_219_style_doc()))

    khoan_1 = tree.khoans[0]
    # Heading "##### Khoản 1"/"##### Khoản 2" bên trong trích dẫn bị gộp làm
    # văn bản thường (chỉ còn "Khoản 1"/"Khoản 2" trong content, không có "#").
    assert "##### Khoản" not in khoan_1.content
    assert "Khoản 1" in khoan_1.content
    assert "Khoản 2" in khoan_1.content
    assert "Nội dung khoản 1 trích dẫn." in khoan_1.content
    assert "b) Một điểm khác." in khoan_1.content

    khoan_2 = tree.khoans[1]
    assert khoan_2.content == "Nội dung khoản 2 thật."


def test_khoan_long_trong_trich_dan_chunk_id_khong_trung_lap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(splitter, "count_tokens", lambda text: len(text.split()))
    path = _write(tmp_path, "a.md", _dieu_219_style_doc())
    result = convert_markdown_to_chunks(path)
    ids = [chunk.chunk_id for chunk in result.chunks]
    assert len(ids) == len(set(ids))


def test_quote_depth_reset_o_ranh_gioi_dieu_moi(tmp_path: Path):
    # Giới hạn "bán kính nổ" của quote_depth: nếu 1 đoạn trích dẫn trong Điều
    # trước đó LỠ không đóng ngoặc kép (lỗi soạn thảo/trích xuất), Điều MỚI
    # (heading cấp 1-4) vẫn reset quote_depth về 0 -- không kéo lỗi sang toàn
    # bộ phần còn lại của văn bản.
    text = _doc(
        "#### Điều 1. Điều có trích dẫn lỗi\n\n"
        "##### Khoản 1\n\n"
        "Trích dẫn: “Điều 54. Điều kiện (thiếu dấu đóng ngoặc kép).\n\n"
        "#### Điều 2. Điều kế tiếp\n\n"
        "##### Khoản 1\n\n"
        "Nội dung khoản 1 của Điều 2, được nhận diện đúng nhờ quote_depth reset.\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert [(khoan.breadcrumb_prefix, khoan.khoan_number) for khoan in tree.khoans] == [
        ("LUẬT VĂN BẢN MẪU - Điều 1. Điều có trích dẫn lỗi", "1"),
        ("LUẬT VĂN BẢN MẪU - Điều 2. Điều kế tiếp", "1"),
    ]
    assert tree.khoans[1].content == (
        "Nội dung khoản 1 của Điều 2, được nhận diện đúng nhờ quote_depth reset."
    )


def test_quote_depth_khong_can_canh_bao_runtime(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
):
    text = _doc(
        "#### Điều 1. Điều có trích dẫn lỗi\n\n"
        "##### Khoản 1\n\n"
        "Trích dẫn: “Điều 54. Điều kiện (thiếu dấu đóng ngoặc kép).\n"
    )
    parse_markdown(_write(tmp_path, "a.md", text))
    messages = [str(warning.message) for warning in recwarn.list]
    assert any("quote_depth" in message for message in messages)


def test_quote_depth_can_bang_khong_canh_bao(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
):
    text = _doc(
        "#### Điều 1. Điều có trích dẫn cân\n\n"
        "##### Khoản 1\n\n"
        "Trích dẫn: “Điều 54. Điều kiện.”\n"
    )
    parse_markdown(_write(tmp_path, "a.md", text))
    assert len(recwarn.list) == 0


# ==========================================================================
# Regression: chunk_id trùng lặp khi 2 heading cấp 5 không khớp dạng nào đã
# biết ("Khoản ngầm định") xuất hiện trong cùng 1 Điều (dữ liệu lỗi/OCR).
# Cả 2 KhoanNode ngầm định dừng ở cùng breadcrumb_prefix (cấp Điều) ->
# `_make_chunk_id` sinh cùng chunk_id -> `pipeline.py::_ensure_unique_chunk_ids`
# raise `ValueError` rõ ràng thay vì âm thầm ghi đè.
# ==========================================================================


def _dieu_2_heading_khong_khop_dang_nao() -> str:
    return _doc(
        "#### Điều 1. Điều có heading lỗi\n\n"
        "##### ???\n\n"
        "Nội dung khoản ngầm định thứ nhất.\n\n"
        "##### ???\n\n"
        "Nội dung khoản ngầm định thứ hai.\n"
    )


def test_2_heading_loi_cung_dieu_tao_2_khoan_ngam_dinh_cung_breadcrumb(
    tmp_path: Path,
):
    tree = parse_markdown(
        _write(tmp_path, "a.md", _dieu_2_heading_khong_khop_dang_nao())
    )
    assert len(tree.khoans) == 2
    assert tree.khoans[0].khoan_number is None
    assert tree.khoans[1].khoan_number is None
    assert tree.khoans[0].breadcrumb_prefix == tree.khoans[1].breadcrumb_prefix
    assert tree.khoans[0].content != tree.khoans[1].content


def test_2_heading_loi_cung_dieu_chunk_id_trung_lap_bi_chan_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(splitter, "count_tokens", lambda text: len(text.split()))
    path = _write(tmp_path, "a.md", _dieu_2_heading_khong_khop_dang_nao())
    with pytest.raises(ValueError, match="chunk_id trùng lặp"):
        convert_markdown_to_chunks(path)
