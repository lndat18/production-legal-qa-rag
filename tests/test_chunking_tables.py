"""Unit test cho `chunking/tables.py` (chunking_spec.md mục 5.2, 5.3)."""

from __future__ import annotations

from production_legal_qa_rag.chunking.tables import parse_pipe_table, standardize_table

# ==========================================================================
# parse_pipe_table
# ==========================================================================


def test_parse_pipe_table_bo_dong_phan_cach():
    markdown = "| Vùng | Mức lương |\n| --- | --- |\n| Vùng I | 5.310.000 |\n"
    rows = parse_pipe_table(markdown)
    assert rows == [["Vùng", "Mức lương"], ["Vùng I", "5.310.000"]]


def test_parse_pipe_table_ho_tro_can_le_trong_dong_phan_cach():
    markdown = "| A | B |\n| :--- | ---: |\n| 1 | 2 |\n"
    rows = parse_pipe_table(markdown)
    assert rows == [["A", "B"], ["1", "2"]]


def test_parse_pipe_table_giu_dung_thu_tu_hang():
    markdown = (
        "| Vùng | Mức |\n| --- | --- |\n"
        "| Vùng I | 1 |\n| Vùng II | 2 |\n| Vùng III | 3 |\n| Vùng IV | 4 |\n"
    )
    rows = parse_pipe_table(markdown)
    assert [row[0] for row in rows[1:]] == ["Vùng I", "Vùng II", "Vùng III", "Vùng IV"]


def test_parse_pipe_table_bo_qua_dong_khong_bat_dau_bang_pipe():
    markdown = "Câu dẫn trước bảng.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    rows = parse_pipe_table(markdown)
    assert rows == [["A", "B"], ["1", "2"]]


def test_parse_pipe_table_rong_neu_khong_co_dong_pipe():
    assert parse_pipe_table("Không có bảng nào ở đây.") == []


# ==========================================================================
# standardize_table (mục 5.3) -- đúng mã giả + ví dụ trong spec
# ==========================================================================


def test_standardize_table_vi_du_luong_toi_thieu_dung_spec():
    raw_table = (
        "| Vùng | Mức lương tối thiểu tháng<br>(Đơn vị: đồng/tháng) "
        "| Mức lương tối thiểu giờ<br>(Đơn vị: đồng/giờ) |\n"
        "| --- | --- | --- |\n"
        "| Vùng I | 5.310.000 | 25.500 |\n"
        "| Vùng II | 4.730.000 | 22.700 |\n"
        "| Vùng III | 4.140.000 | 20.000 |\n"
        "| Vùng IV | 3.700.000 | 17.800 |\n"
    )
    result = standardize_table(raw_table)
    lines = result.splitlines()
    assert len(lines) == 4
    assert lines[0] == (
        "Vùng I - Mức lương tối thiểu tháng(Đơn vị: đồng/tháng): 5.310.000 "
        "- Mức lương tối thiểu giờ(Đơn vị: đồng/giờ): 25.500"
    )
    assert lines[1] == (
        "Vùng II - Mức lương tối thiểu tháng(Đơn vị: đồng/tháng): 4.730.000 "
        "- Mức lương tối thiểu giờ(Đơn vị: đồng/giờ): 22.700"
    )
    assert lines[3] == (
        "Vùng IV - Mức lương tối thiểu tháng(Đơn vị: đồng/tháng): 3.700.000 "
        "- Mức lương tối thiểu giờ(Đơn vị: đồng/giờ): 17.800"
    )


def test_standardize_table_xoa_br_khong_chen_khoang_trang():
    raw_table = "| Tên | Cột A<br>(ghi chú) |\n| --- | --- |\n| Dòng 1 | Giá trị 1 |\n"
    result = standardize_table(raw_table)
    assert result == "Dòng 1 - Cột A(ghi chú): Giá trị 1"
    assert "<br>" not in result


def test_standardize_table_nhieu_cot_noi_bang_gach_ngang():
    raw_table = (
        "| Nhãn | Cột 1 | Cột 2 | Cột 3 |\n| --- | --- | --- | --- |\n"
        "| Dòng | A | B | C |\n"
    )
    result = standardize_table(raw_table)
    assert result == "Dòng - Cột 1: A - Cột 2: B - Cột 3: C"


def test_standardize_table_rong_neu_khong_co_hang(tmp_path=None):
    assert standardize_table("Không phải bảng.") == ""


def test_standardize_table_khong_co_du_lieu_chi_co_header():
    raw_table = "| A | B |\n| --- | --- |\n"
    assert standardize_table(raw_table) == ""


# ==========================================================================
# standardize_table -- bảng HTML single-row (mở rộng ngoài spec, xem
# `tables.py` docstring / báo cáo bàn giao developer)
# ==========================================================================


def test_standardize_table_html_single_row_noi_cac_o_td():
    raw_table = (
        "<table>\n  <tbody>\n    <tr>\n"
        "      <td>Tiền lương làm thêm giờ</td>\n"
        "      <td>=</td>\n"
        "      <td>Tiền lương giờ thực trả</td>\n"
        "      <td>x</td>\n"
        "      <td>Số giờ làm thêm</td>\n"
        "    </tr>\n  </tbody>\n</table>\n"
    )
    result = standardize_table(raw_table)
    assert result == (
        "Tiền lương làm thêm giờ = Tiền lương giờ thực trả x Số giờ làm thêm"
    )


def test_standardize_table_html_single_row_giai_ma_html_entity():
    raw_table = "<table><tr><td>A &amp; B</td><td>&gt; 0</td></tr></table>"
    result = standardize_table(raw_table)
    assert result == "A & B > 0"


def test_standardize_table_html_single_row_xoa_br_trong_o():
    raw_table = "<table><tr><td>Dòng 1<br>Dòng 2</td></tr></table>"
    result = standardize_table(raw_table)
    assert result == "Dòng 1 Dòng 2"
