"""Unit test cho `chunking/patterns.py` (chunking_spec.md mục 3, 4, 5, 10).

Regex nhận diện heading Markdown (Phần/Chương/Mục/Điều/Khoản), nhãn Điểm,
dòng bảng markdown/HTML, marker backmatter và các hàm tách câu/mệnh đề dùng ở
tầng fallback (mục 4.4).

Không còn `NEGATION_KEYWORDS`/`RE_NEGATION` -- quyết định thiết kế mục 12 của
spec đã bỏ field `negation_note` riêng (câu dẫn luôn được lặp lại nguyên văn,
không phân biệt có từ khoá phủ định hay không, mục 4.5), nên `patterns.py`
hiện tại không còn các định nghĩa đó.
"""

from __future__ import annotations

from production_legal_qa_rag.chunking.patterns import (
    RE_BACKMATTER_SEPARATOR,
    RE_CHUONG,
    RE_DIEM,
    RE_DIEU,
    RE_HEADING,
    RE_HTML_TABLE_LINE,
    RE_KHOAN_LABEL,
    RE_KHOAN_MERGED,
    RE_MUC,
    RE_PHAN,
    RE_PHU_LUC,
    RE_TABLE_LINE,
    is_structural_heading,
    split_finer,
    split_sentences,
)

# ==========================================================================
# RE_HEADING
# ==========================================================================


def test_re_heading_nhan_dung_cap_do_va_noi_dung():
    match = RE_HEADING.match("#### Điều 3. Mức lương tối thiểu")
    assert match is not None
    assert match.group(1) == "####"
    assert match.group(2) == "Điều 3. Mức lương tối thiểu"


def test_re_heading_khong_khop_dong_khong_phai_heading():
    assert RE_HEADING.match("Đây không phải heading") is None


def test_re_heading_toi_da_5_dau_thang():
    match = RE_HEADING.match("##### Khoản 1")
    assert match is not None
    assert match.group(1) == "#####"


def test_re_heading_6_dau_thang_khong_khop_gioi_han_5_cap():
    assert RE_HEADING.match("###### Không phải cấp hợp lệ") is None


# ==========================================================================
# RE_PHAN / RE_CHUONG / RE_MUC / RE_DIEU / RE_PHU_LUC
# ==========================================================================


def test_re_chuong_bat_so_la_ma():
    match = RE_CHUONG.match("Chương II. Quy định chung")
    assert match is not None
    assert match.group(1) == "II"


def test_re_muc_bat_so_kem_chu_cai():
    match = RE_MUC.match("Mục 2a. Điều kiện")
    assert match is not None
    assert match.group(1) == "2a"


def test_re_phan_bat_so_thu_tu_tieng_viet():
    match = RE_PHAN.match("Phần thứ nhất")
    assert match is not None
    assert match.group(1) == "nhất"


def test_re_phan_khong_khop_phu_luc():
    # Tiêu đề Phụ Lục dùng "PHỤ LỤC" -- có regex riêng (RE_PHU_LUC), không
    # khớp RE_PHAN.
    assert RE_PHAN.match("PHỤ LỤC — DANH MỤC ĐỊA BÀN") is None


def test_re_phu_luc_khop_khong_phan_biet_hoa_thuong():
    assert RE_PHU_LUC.match("Phụ lục I — Danh mục") is not None
    assert RE_PHU_LUC.match("PHỤ LỤC") is not None


def test_re_dieu_bat_so_va_ten_dieu():
    match = RE_DIEU.match("Điều 3. Mức lương tối thiểu")
    assert match is not None
    assert match.group(1) == "3"
    assert match.group(2) == "Mức lương tối thiểu"


def test_re_dieu_khong_khop_neu_thieu_dau_cham():
    assert RE_DIEU.match("Điều 3 Mức lương tối thiểu") is None


def test_re_dieu_so_kem_chu_cai():
    match = RE_DIEU.match("Điều 48a. Quy định chuyển tiếp")
    assert match is not None
    assert match.group(1) == "48a"


# ==========================================================================
# RE_KHOAN_LABEL / RE_KHOAN_MERGED
# ==========================================================================


def test_re_khoan_label_khop_dang_chuan():
    match = RE_KHOAN_LABEL.match("Khoản 4")
    assert match is not None
    assert match.group(1) == "4"


def test_re_khoan_label_khong_khop_neu_co_them_noi_dung():
    assert RE_KHOAN_LABEL.match("Khoản 4. Nội dung") is None


def test_re_khoan_merged_khop_dang_phu_luc():
    match = RE_KHOAN_MERGED.match("1. Thành phố Hà Nội")
    assert match is not None
    assert match.group(1) == "1"
    assert match.group(2) == "Thành phố Hà Nội"


# ==========================================================================
# RE_DIEM
# ==========================================================================


def test_re_diem_khop_nhan_chu_cai_thuong():
    match = RE_DIEM.match("a) Nội dung điểm a")
    assert match is not None
    assert match.group(1) == "a"
    assert match.group(2) == "Nội dung điểm a"


def test_re_diem_khop_chu_dac_biet_tieng_viet():
    for label in ("đ", "ư"):
        match = RE_DIEM.match(f"{label}) Nội dung")
        assert match is not None
        assert match.group(1) == label


def test_re_diem_khong_khop_danh_sach_gach_dau_dong():
    # Danh sách "-" trong Phụ Lục KHÔNG được coi là ranh giới Điểm (quyết định
    # thiết kế, xem parser.py docstring).
    assert RE_DIEM.match("- Vùng I gồm các phường...") is None


def test_re_diem_khong_khop_neu_thieu_dau_ngoac_dong():
    assert RE_DIEM.match("a Nội dung") is None


# ==========================================================================
# RE_TABLE_LINE / RE_HTML_TABLE_LINE
# ==========================================================================


def test_re_table_line_khop_dong_bat_dau_bang_pipe():
    assert RE_TABLE_LINE.match("| Vùng | Mức lương |") is not None


def test_re_table_line_khop_du_co_khoang_trang_dau_dong():
    assert RE_TABLE_LINE.match("   | Vùng I | 5.310.000 |") is not None


def test_re_table_line_khong_khop_van_ban_thuong():
    assert RE_TABLE_LINE.match("Đây là một câu văn bản.") is None


def test_re_html_table_line_khop_the_table_mo():
    assert RE_HTML_TABLE_LINE.match("<table>") is not None
    assert RE_HTML_TABLE_LINE.match("  <table>") is not None


def test_re_html_table_line_khong_khop_van_ban_thuong():
    assert RE_HTML_TABLE_LINE.match("Tiền lương làm thêm giờ") is None


# ==========================================================================
# RE_BACKMATTER_SEPARATOR (mục 4.6) -- marker THẬT: dòng "---" 3 ký tự đúng
# khít, khác mô tả "**[n]**" trong chunking_spec.md (đã lỗi thời, xem
# patterns.py docstring / báo cáo bàn giao developer).
# ==========================================================================


def test_re_backmatter_separator_khop_dung_3_gach_ngang():
    assert RE_BACKMATTER_SEPARATOR.match("---") is not None


def test_re_backmatter_separator_khong_khop_gach_ngang_trang_tri_dai_hon():
    # Dòng gạch ngang trang trí trong frontmatter (vd. "--------",
    # "---------------") không phải marker backmatter thật.
    assert RE_BACKMATTER_SEPARATOR.match("--------") is None
    assert RE_BACKMATTER_SEPARATOR.match("---------------") is None


def test_re_backmatter_separator_khong_khop_it_hon_3_gach_ngang():
    assert RE_BACKMATTER_SEPARATOR.match("--") is None


def test_re_backmatter_separator_khong_khop_van_ban_thuong():
    assert RE_BACKMATTER_SEPARATOR.match("Đây là một câu văn bản.") is None


# ==========================================================================
# is_structural_heading (mục 4.6) -- phân biệt heading tên văn bản (H1
# frontmatter) với heading cấu trúc Phần/Phụ Lục/Chương/Mục/Điều/Khoản thật.
# ==========================================================================


def test_is_structural_heading_level_1_phu_luc_la_cau_truc():
    assert is_structural_heading(1, "PHỤ LỤC — DANH MỤC ĐỊA BÀN") is True


def test_is_structural_heading_level_1_phan_la_cau_truc():
    assert is_structural_heading(1, "Phần thứ nhất") is True


def test_is_structural_heading_level_1_ten_van_ban_khong_phai_cau_truc():
    # Heading "#" tên văn bản trong frontmatter (vd. "BẢO HIỂM XÃ HỘI") không
    # khớp Phần/Phụ Lục -> không phải heading cấu trúc.
    assert is_structural_heading(1, "BẢO HIỂM XÃ HỘI") is False


def test_is_structural_heading_level_2_chuong_la_cau_truc():
    assert is_structural_heading(2, "Chương I") is True


def test_is_structural_heading_level_2_khong_khop_khong_phai_cau_truc():
    assert is_structural_heading(2, "Tiêu đề bất kỳ") is False


def test_is_structural_heading_level_4_dieu_la_cau_truc():
    assert is_structural_heading(4, "Điều 1. Phạm vi điều chỉnh") is True


def test_is_structural_heading_level_5_khoan_label_la_cau_truc():
    assert is_structural_heading(5, "Khoản 1") is True


def test_is_structural_heading_level_5_khoan_merged_la_cau_truc():
    assert is_structural_heading(5, "1. Thành phố Hà Nội") is True


def test_is_structural_heading_level_ngoai_pham_vi_khong_phai_cau_truc():
    assert is_structural_heading(6, "Bất kỳ") is False


# ==========================================================================
# split_sentences / split_finer (mục 4.4)
# ==========================================================================


def test_split_sentences_tach_theo_dau_cham_va_khoang_trang():
    result = split_sentences("Câu một. Câu hai. Câu ba.")
    assert result == ["Câu một.", "Câu hai.", "Câu ba."]


def test_split_sentences_tach_theo_dau_cham_phay():
    result = split_sentences("Vế một; vế hai.")
    assert result == ["Vế một;", "vế hai."]


def test_split_sentences_khong_tach_neu_khong_co_khoang_trang_sau_dau_cham():
    # "1.234" (số có dấu chấm phân cách nghìn) không bị tách nhầm vì không có
    # khoảng trắng ngay sau dấu chấm.
    result = split_sentences("Số tiền 1.234.000 đồng.")
    assert result == ["Số tiền 1.234.000 đồng."]


def test_split_sentences_bo_qua_phan_tu_rong():
    result = split_sentences("Câu một.   Câu hai.")
    assert all(part.strip() for part in result)


def test_split_finer_tach_theo_dau_phay():
    result = split_finer("phần một, phần hai, phần ba")
    assert result == ["phần một", "phần hai", "phần ba"]


def test_split_finer_khong_tach_neu_khong_co_dau_phay():
    assert split_finer("một câu không có dấu phẩy") == ["một câu không có dấu phẩy"]
