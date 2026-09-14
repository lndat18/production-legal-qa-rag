"""Unit test cho `chunking/patterns.py` (chunking_spec.md mục 3, 4, 5, 10).

Regex nhận diện heading Markdown (Phần/Chương/Mục/Điều/Khoản), nhãn Điểm,
dòng bảng markdown/HTML, từ khoá phủ định và các hàm tách câu/mệnh đề dùng ở
tầng fallback (mục 4.4).
"""

from __future__ import annotations

from production_legal_qa_rag.chunking.patterns import (
    NEGATION_KEYWORDS,
    RE_CHUONG,
    RE_DIEM,
    RE_DIEU,
    RE_HEADING,
    RE_HTML_TABLE_LINE,
    RE_KHOAN_LABEL,
    RE_KHOAN_MERGED,
    RE_MUC,
    RE_NEGATION,
    RE_PHAN,
    RE_TABLE_LINE,
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


# ==========================================================================
# RE_PHAN / RE_CHUONG / RE_MUC / RE_DIEU
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
    # Tiêu đề Phụ Lục dùng chữ "PHỤ LỤC", không phải "Phần" -> không khớp
    # (parser.py xử lý bằng cách giữ nguyên text làm đoạn Phần, mục "quyết
    # định thiết kế" trong parser.py docstring).
    assert RE_PHAN.match("PHỤ LỤC — DANH MỤC ĐỊA BÀN") is None


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
    # thiết kế của developer, phù hợp mục 3/10: spec chỉ định nghĩa nhãn
    # a)/b)/c..., không nhắc "-").
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
# NEGATION_KEYWORDS / RE_NEGATION (mục 4.5)
# ==========================================================================


def test_negation_keywords_dung_danh_sach_literal_cua_spec():
    assert set(NEGATION_KEYWORDS) == {
        "trừ",
        "ngoại trừ",
        "loại trừ",
        "không áp dụng",
        "không thuộc",
        "không bao gồm",
    }


def test_re_negation_khop_cau_co_khong_ap_dung():
    match = RE_NEGATION.search(
        "Luật này không áp dụng đối với bảo hiểm y tế mang tính kinh doanh."
    )
    assert match is not None
    assert match.group(0).lower() == "không áp dụng"


def test_re_negation_khop_khong_phan_biet_hoa_thuong():
    assert RE_NEGATION.search("KHÔNG ÁP DỤNG đối với trường hợp X.") is not None


def test_re_negation_khong_khop_van_ban_khong_co_tu_khoa():
    assert RE_NEGATION.search("Người sử dụng lao động phải trả lương đầy đủ.") is None


def test_re_negation_tru_don_le_gay_false_positive_da_biet_truoc():
    """`"trừ"` trần trụi (đúng theo danh sách literal mục 4.5) khớp cả vào
    "khấu trừ"/"trừ đi" — false positive đã được developer báo cáo trước, giữ
    nguyên theo đúng yêu cầu spec (không phải bug cần sửa, xem báo cáo bàn
    giao của developer, điểm 9)."""
    assert RE_NEGATION.search("Mức khấu trừ thuế thu nhập cá nhân.") is not None
    assert RE_NEGATION.search("Số tiền trừ đi các khoản chi phí.") is not None


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
