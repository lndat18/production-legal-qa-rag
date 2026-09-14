"""Unit test cho `chunking/parser.py` (chunking_spec.md mục 3, 10).

Dựng `DocumentTree` từ markdown tổng hợp (không phụ thuộc `data/markdown/`
thật, trừ 1 nhóm test đối chiếu trực tiếp với file thật để khoá hành vi lại).

Mọi fixture (trừ 2 nhóm regression test ở cuối file) đều có 1 đoạn tiêu đề
văn bản ("Tiêu đề văn bản mẫu.") ngay sau front matter, trước heading cấu
trúc đầu tiên — đúng quy ước thật của `formatting/emitter.py` (xem toàn bộ
`data/markdown/*.md`: luôn có đoạn "LUẬT"/"NGHỊ ĐỊNH" + tên văn bản trước
heading đầu tiên).

2 nhóm regression test ở cuối file khoá lại hành vi của 2 bug thật đã được
`parser.py` sửa (commit f4c3d2e, sau feedback REVISE vòng 1 của tester):
heading ngay sau front matter, và Khoản lồng trong đoạn trích dẫn nguyên văn
điều luật khác (`quote_depth`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from production_legal_qa_rag.chunking import splitter
from production_legal_qa_rag.chunking.parser import parse_markdown
from production_legal_qa_rag.chunking.pipeline import convert_markdown_to_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdown"

_TIEU_DE = "Tiêu đề văn bản mẫu."


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _doc(front_matter: str, body: str) -> str:
    return f"---\n{front_matter}\n---\n\n{_TIEU_DE}\n\n{body}"


# ==========================================================================
# Front matter / source_document / breadcrumb prefix cấp văn bản
# ==========================================================================


def test_source_document_uu_tien_so_hieu(tmp_path: Path):
    text = _doc(
        'so_hieu: "41/2024/QH15"\nten_van_ban: "Luật Bảo hiểm xã hội"',
        "#### Điều 1. Phạm vi điều chỉnh\n\n##### Khoản 1\n\nNội dung khoản 1.\n",
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.source_document == "41/2024/QH15"


def test_source_document_fallback_ten_file_khi_thieu_so_hieu(tmp_path: Path):
    text = _doc(
        'ten_van_ban: "Văn bản không có số hiệu"',
        "#### Điều 1. Phạm vi điều chỉnh\n\n##### Khoản 1\n\nNội dung.\n",
    )
    path = _write(tmp_path, "khong_so_hieu.md", text)
    tree = parse_markdown(path)
    assert tree.source_document == "khong_so_hieu"


def test_breadcrumb_prefix_gom_ten_van_ban_va_so_hieu(tmp_path: Path):
    text = _doc(
        'so_hieu: "41/2024/QH15"\nten_van_ban: "Luật Bảo hiểm xã hội"',
        "#### Điều 1. Phạm vi điều chỉnh\n\n##### Khoản 1\n\nNội dung khoản 1.\n",
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.khoans[0].breadcrumb_prefix == (
        "Luật Bảo hiểm xã hội (41/2024/QH15) - Điều 1. Phạm vi điều chỉnh"
    )


# ==========================================================================
# Cây Phần/Chương/Mục/Điều/Khoản
# ==========================================================================


def test_breadcrumb_prefix_day_du_phan_chuong_muc_dieu(tmp_path: Path):
    text = _doc(
        'so_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"',
        "# Phần I\n\n## Chương II\n\n### Mục 3\n\n#### Điều 5. Tên điều\n\n"
        "##### Khoản 1\n\nNội dung.\n",
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.khoans[0].breadcrumb_prefix == (
        "Văn bản mẫu (01/2020/QH) - Phần I - Chương II - Mục 3 - Điều 5. Tên điều"
    )


def test_bo_qua_cap_khong_ton_tai_trong_van_ban(tmp_path: Path):
    # Văn bản không có Phần/Mục -> breadcrumb bỏ qua các cấp đó (mục 3).
    text = _doc(
        'so_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"',
        "## Chương I\n\n#### Điều 1. Tên điều\n\n##### Khoản 1\n\nNội dung.\n",
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.khoans[0].breadcrumb_prefix == (
        "Văn bản mẫu (01/2020/QH) - Chương I - Điều 1. Tên điều"
    )


def test_dieu_khong_co_ten_khong_them_dau_cham_thua(tmp_path: Path):
    text = _doc(
        'so_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"',
        "#### Điều 1.\n\n##### Khoản 1\n\nNội dung.\n",
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.khoans[0].breadcrumb_prefix == "Văn bản mẫu (01/2020/QH) - Điều 1"


def test_reset_chuong_muc_khi_sang_phan_moi(tmp_path: Path):
    text = _doc(
        'so_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"',
        "# Phần I\n\n## Chương I\n\n#### Điều 1. Điều đầu\n\n##### Khoản 1\n\n"
        "Nội dung 1.\n\n# Phần II\n\n#### Điều 2. Điều sau\n\n##### Khoản 1\n\n"
        "Nội dung 2.\n",
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    prefixes = [khoan.breadcrumb_prefix for khoan in tree.khoans]
    assert prefixes == [
        "Văn bản mẫu (01/2020/QH) - Phần I - Chương I - Điều 1. Điều đầu",
        "Văn bản mẫu (01/2020/QH) - Phần II - Điều 2. Điều sau",
    ]


# ==========================================================================
# Khoản ngầm định (Điều không chia Khoản / đoạn mở đầu trước Khoản đầu tiên)
# ==========================================================================


def test_dieu_khong_co_khoan_con_thanh_khoan_ngam_dinh(tmp_path: Path):
    text = _doc(
        'so_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"',
        "#### Điều 1. Điều không chia khoản\n\n"
        "Nội dung nằm thẳng dưới Điều, không có heading Khoản.\n",
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert len(tree.khoans) == 1
    khoan = tree.khoans[0]
    assert khoan.khoan_number is None
    assert khoan.content == "Nội dung nằm thẳng dưới Điều, không có heading Khoản."
    # Breadcrumb dừng ở cấp Điều, không có "- Khoản" (mục parser.py docstring).
    assert (
        khoan.breadcrumb_prefix
        == "Văn bản mẫu (01/2020/QH) - Điều 1. Điều không chia khoản"
    )


def test_doan_mo_dau_truoc_khoan_dau_tien_thanh_khoan_ngam_dinh(tmp_path: Path):
    text = _doc(
        'so_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"',
        "#### Điều 1. Điều có đoạn mở đầu\n\n"
        "Đoạn mở đầu trước Khoản đầu tiên.\n\n"
        "##### Khoản 1\n\nNội dung khoản 1.\n",
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
        'so_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"',
        "# Phụ Lục\n\n(Kèm theo Nghị định số ABC)\n\n"
        "##### 1. Thành phố Hà Nội\n\nNội dung địa bàn.\n",
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert len(tree.khoans) == 1
    assert "Kèm theo" not in tree.khoans[0].content


# ==========================================================================
# Khoản trong Phụ Lục (heading gộp "N. tên")
# ==========================================================================


def test_khoan_phu_luc_dang_gop_khong_can_dieu_bao_ngoai(tmp_path: Path):
    text = _doc(
        'so_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"',
        "# PHỤ LỤC\n\n##### 1. Thành phố Hà Nội\n\n"
        "Vùng I gồm các phường trung tâm.\n\n"
        "##### 2. Tỉnh Cao Bằng\n\nVùng III gồm các xã còn lại.\n",
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
        'so_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"',
        "#### Điều 1. Tên điều\n\n##### Khoản 1\n\n"
        "Nội dung chính của khoản.\n\n"
        "> **Sửa đổi:** Khoản này được sửa đổi theo Luật số X.\n>\n"
        "> _(Trích dẫn đầy đủ ở cuối văn bản.)_\n",
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
        'so_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"',
        "#### Điều 3. Mức lương tối thiểu\n\n##### Khoản 1\n\n"
        "Quy định mức lương tối thiểu theo vùng như sau:\n\n"
        "| Vùng | Mức lương tối thiểu tháng<br>(Đơn vị: đồng/tháng) |\n"
        "| --- | --- |\n"
        "| Vùng I | 5.310.000 |\n"
        "| Vùng II | 4.730.000 |\n",
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
        'so_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"',
        "#### Điều 55. Tiền lương làm thêm giờ\n\n##### Khoản 1\n\n"
        "Công thức tính như sau:\n\n"
        "<table>\n  <tbody>\n    <tr>\n      <td>Tiền lương làm thêm giờ</td>\n"
        "      <td>=</td>\n      <td>Đơn giá</td>\n    </tr>\n  </tbody>\n</table>\n",
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    khoan = tree.khoans[0]
    assert khoan.has_table is True
    assert khoan.raw_table is not None
    assert khoan.raw_table.strip().lower().startswith("<table")


def test_khoan_khong_co_bang_has_table_false(tmp_path: Path):
    text = _doc(
        'so_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"',
        "#### Điều 1. Tên điều\n\n##### Khoản 1\n\nKhông có bảng nào ở đây.\n",
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert tree.khoans[0].has_table is False
    assert tree.khoans[0].raw_table is None


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


# ==========================================================================
# Regression: heading đứng NGAY sau front matter (không có đoạn văn bản mở
# đầu nào ở giữa) -- sửa ở commit f4c3d2e (`_split_blocks` giờ `strip()` từng
# block trước khi match `RE_HEADING`). Trước fix, "\n" thừa do
# `_split_front_matter` nối `lines[index + 1:]` bằng "\n" dính vào block đầu
# tiên khiến `RE_HEADING.match` (dùng `^`/`$`, không `re.MULTILINE`) thất bại
# âm thầm, rớt hết heading/Khoản theo sau. Giữ lại các test này (đã sửa từ
# "BUG_..." khoá hành vi sai) làm regression test cho tương lai.
# ==========================================================================


def test_heading_ngay_sau_front_matter_van_duoc_nhan_dien(tmp_path: Path):
    text = (
        '---\nso_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"\n---\n\n'
        "#### Điều 1. Điều không chia khoản\n\n"
        "Nội dung nằm thẳng dưới Điều, không có heading Khoản.\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert len(tree.khoans) == 1
    assert (
        tree.khoans[0].content
        == "Nội dung nằm thẳng dưới Điều, không có heading Khoản."
    )


def test_heading_dau_tien_sau_front_matter_khong_bi_mat_chuong(tmp_path: Path):
    text = (
        '---\nso_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"\n---\n\n'
        "## Chương I\n\n#### Điều 1. Tên điều\n\n##### Khoản 1\n\nNội dung.\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert len(tree.khoans) == 1
    assert (
        tree.khoans[0].breadcrumb_prefix
        == "Văn bản mẫu (01/2020/QH) - Chương I - Điều 1. Tên điều"
    )


# ==========================================================================
# Regression: Khoản lồng trong đoạn trích dẫn nguyên văn điều luật khác (vd.
# Điều 219 `Văn bản hợp nhất bộ luật lao động.md` trích Điều 54/55 Luật BHXH
# bằng chính heading cấp 5 "##### Khoản N", khiến các Khoản trích dẫn tự đánh
# số lại từ 1 -> trùng breadcrumb/chunk_id với Khoản thật). Sửa ở commit
# f4c3d2e bằng `quote_depth` (đếm độ sâu dấu ngoặc kép "“"/"”"): heading cấp 5
# xuất hiện khi đang trong 1 đoạn trích dẫn (quote_depth > 0) được gộp làm
# văn bản thường vào Khoản thật đang mở thay vì tạo `KhoanNode` mới.
# ==========================================================================


def _dieu_219_style_doc() -> str:
    return _doc(
        'so_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"',
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
        "Nội dung khoản 2 thật.\n",
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
        'so_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"',
        "#### Điều 1. Điều có trích dẫn lỗi\n\n"
        "##### Khoản 1\n\n"
        "Trích dẫn: “Điều 54. Điều kiện (thiếu dấu đóng ngoặc kép).\n\n"
        "#### Điều 2. Điều kế tiếp\n\n"
        "##### Khoản 1\n\n"
        "Nội dung khoản 1 của Điều 2, được nhận diện đúng nhờ quote_depth reset.\n",
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert [(khoan.breadcrumb_prefix, khoan.khoan_number) for khoan in tree.khoans] == [
        ("Văn bản mẫu (01/2020/QH) - Điều 1. Điều có trích dẫn lỗi", "1"),
        ("Văn bản mẫu (01/2020/QH) - Điều 2. Điều kế tiếp", "1"),
    ]
    assert tree.khoans[1].content == (
        "Nội dung khoản 1 của Điều 2, được nhận diện đúng nhờ quote_depth reset."
    )


def test_HAN_CHE_biet_truoc_ngoac_kep_khong_can_trong_cung_1_dieu_nuot_khoan_that(
    tmp_path: Path,
):
    """Hạn chế đã biết của cách sửa `quote_depth` (không phải bug mới, ghi lại
    để không ai ngạc nhiên nếu gặp lại): nếu 1 đoạn trích dẫn KHÔNG đóng ngoặc
    kép đúng cách NGAY TRONG CÙNG 1 Điều (không có heading cấp 1-4 nào đứng
    giữa để reset `quote_depth`), Khoản thật đứng sau trong cùng Điều đó bị
    nuốt nhầm vào Khoản trước làm văn bản trích dẫn. Rủi ro thấp trên corpus
    hiện tại (đã xác nhận số lượng "“"/"”" cân bằng ở cả 6 file
    `data/markdown/*.md`), nhưng là hạn chế thật, đáng theo dõi nếu văn bản
    mới dùng dấu ngoặc kép không chuẩn ("straight quotes" thay vì "smart
    quotes") hoặc bị lỗi trích xuất làm mất 1 dấu đóng ngoặc.
    """
    text = _doc(
        'so_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"',
        "#### Điều 1. Điều có trích dẫn lỗi\n\n"
        "##### Khoản 1\n\n"
        "Trích dẫn: “Điều 54. Điều kiện (thiếu dấu đóng ngoặc kép ở đây).\n\n"
        "##### Khoản 2\n\n"
        "Nội dung khoản 2 thật sự nhưng bị nuốt vào Khoản 1 do ngoặc kép không cân.\n",
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert len(tree.khoans) == 1
    assert "Nội dung khoản 2 thật sự" in tree.khoans[0].content
