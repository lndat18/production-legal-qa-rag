"""Unit test cho `chunking/parser.py` (chunking_spec.md mục 3, 10).

Dựng `DocumentTree` từ markdown tổng hợp (không phụ thuộc `data/markdown/`
thật, trừ 1 nhóm test đối chiếu trực tiếp với file thật để khoá hành vi lại).

Mọi fixture (trừ nhóm test tái hiện lỗi ở cuối file) đều có 1 đoạn tiêu đề
văn bản ("Tiêu đề văn bản mẫu.") ngay sau front matter, trước heading cấu
trúc đầu tiên — đúng quy ước thật của `formatting/emitter.py` (xem toàn bộ
`data/markdown/*.md`: luôn có đoạn "LUẬT"/"NGHỊ ĐỊNH" + tên văn bản trước
heading đầu tiên). Xem nhóm test cuối file để biết lý do quy ước này quan
trọng: heading đứng NGAY sau front matter (không có đoạn văn bản nào ở
giữa) không được nhận diện đúng — lỗi thật trong `parser.py`, xem feedback.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from production_legal_qa_rag.chunking.parser import parse_markdown

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
# BUG THẬT: heading đứng NGAY sau front matter (không có đoạn văn bản mở đầu
# nào ở giữa) không được nhận diện -- xem feedback gửi developer.
#
# `_split_front_matter` nối `lines[index + 1:]` (bắt đầu bằng dòng trống ngay
# sau front matter) bằng "\n", nên `body` luôn có 1 ký tự "\n" thừa ở đầu.
# `_split_blocks` chỉ tách trên `\n{2,}` (>= 2 dòng trống), nên "\n" đơn lẻ
# này KHÔNG bị tách ra -- nó dính vào block đầu tiên. Nếu block đầu tiên đó
# là 1 heading, `RE_HEADING.match(block)` (dùng `^`/`$`, không có
# `re.MULTILINE`) thất bại vì block không bắt đầu bằng "#" ở vị trí 0 nữa,
# mà bắt đầu bằng "\n#...". Toàn bộ heading/Khoản bên dưới bị rơi rụng âm
# thầm, không có exception nào được raise.
#
# Corpus hiện tại (`data/markdown/*.md`) luôn có đoạn tiêu đề ("LUẬT",
# "NGHỊ ĐỊNH", tên văn bản...) trước heading cấu trúc đầu tiên nên bug này
# đang bị che khuất -- nhưng là hành vi sai thật, vi phạm mục 1 "dựng lại
# cây cấu trúc ... từ heading markdown" (input hợp lệ theo mục 2 chỉ yêu cầu
# "front matter YAML + heading", không đảm bảo có đoạn văn bản mở đầu).
# ==========================================================================


def test_BUG_heading_ngay_sau_front_matter_khong_duoc_nhan_dien(tmp_path: Path):
    text = (
        '---\nso_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"\n---\n\n'
        "#### Điều 1. Điều không chia khoản\n\n"
        "Nội dung nằm thẳng dưới Điều, không có heading Khoản.\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    # Hành vi ĐÚNG mong đợi (giống hệt test có đoạn tiêu đề ở trên): 1 Khoản
    # ngầm định được dựng ra. Assertion dưới đây khoá lại hành vi SAI hiện
    # tại (rớt hết nội dung) để CI báo đỏ cho tới khi developer sửa.
    assert len(tree.khoans) == 1, (
        "parser.py: heading đứng ngay sau front matter (không có đoạn văn bản "
        "mở đầu) không được nhận diện -- xem docstring nhóm test này."
    )


def test_BUG_heading_dau_tien_sau_front_matter_khong_bi_mat_chuong(tmp_path: Path):
    text = (
        '---\nso_hieu: "01/2020/QH"\nten_van_ban: "Văn bản mẫu"\n---\n\n'
        "## Chương I\n\n#### Điều 1. Tên điều\n\n##### Khoản 1\n\nNội dung.\n"
    )
    tree = parse_markdown(_write(tmp_path, "a.md", text))
    assert len(tree.khoans) == 1, (
        "parser.py: '## Chương I' đứng ngay sau front matter bị nuốt vào nội "
        "dung thay vì được nhận diện là heading cấp Chương."
    )
    assert (
        tree.khoans[0].breadcrumb_prefix
        == "Văn bản mẫu (01/2020/QH) - Chương I - Điều 1. Tên điều"
    )
