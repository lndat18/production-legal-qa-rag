"""Unit + integration test cho `chunking/splitter.py` (chunking_spec.md mục 4, 5).

`count_tokens` được monkeypatch bằng bộ đếm giả (đếm từ theo khoảng trắng)
để test thuật toán cắt độc lập với việc nạp model PhoBERT thật (nhanh, không
cần mạng) -- việc đếm token THẬT (word-segment PhoBERT) được xác nhận riêng
ở `tests/test_chunking_tokenizer.py` (đánh dấu `slow`).
"""

from __future__ import annotations

import hashlib

import pytest

from production_legal_qa_rag.chunking import splitter
from production_legal_qa_rag.chunking.models import KhoanNode
from production_legal_qa_rag.chunking.splitter import (
    _compose_split_breadcrumb,
    _explode_oversized,
    _khoan_base_breadcrumb,
    _make_chunk_id,
    _pack_units,
    _split_into_points,
    _Unit,
    split_khoan,
)


def _fake_count_tokens(text: str) -> int:
    """Đếm 'token' = số từ theo khoảng trắng -- đủ để test logic cận dưới/
    fallback mà không cần nạp tokenizer PhoBERT thật."""
    return len(text.split())


@pytest.fixture(autouse=True)
def _mock_tokenizer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(splitter, "count_tokens", _fake_count_tokens)


# ==========================================================================
# _split_into_points (mục 4.2)
# ==========================================================================


def test_split_into_points_tach_dung_dau_muc_va_diem():
    content = (
        "Đoạn mở đầu.\n\n"
        "a) Nội dung điểm a.\n\n"
        "b) Nội dung điểm b.\n\n"
        "c) Nội dung điểm c."
    )
    preamble, points = _split_into_points(content)
    assert preamble == "Đoạn mở đầu."
    assert points == [
        ("a", "Nội dung điểm a."),
        ("b", "Nội dung điểm b."),
        ("c", "Nội dung điểm c."),
    ]


def test_split_into_points_khong_co_diem_toan_bo_la_preamble():
    content = "Chỉ có văn xuôi, không có Điểm gắn nhãn nào."
    preamble, points = _split_into_points(content)
    assert preamble == content
    assert points == []


def test_split_into_points_gop_nhieu_doan_vao_cung_1_diem():
    content = "a) Câu đầu của điểm a.\n\nCâu tiếp theo vẫn thuộc điểm a."
    preamble, points = _split_into_points(content)
    assert preamble == ""
    assert len(points) == 1
    label, text = points[0]
    assert label == "a"
    assert text == "Câu đầu của điểm a.\n\nCâu tiếp theo vẫn thuộc điểm a."


def test_split_into_points_danh_sach_gach_dau_dong_khong_phai_diem():
    content = "Mở đầu.\n\n- Mục gạch đầu dòng, không phải Điểm."
    preamble, points = _split_into_points(content)
    assert points == []
    assert "gạch đầu dòng" in preamble


# ==========================================================================
# _pack_units -- thuật toán "cận dưới" (mục 4.3)
# ==========================================================================


def test_pack_units_khong_vuot_ngan_sach_va_ghep_toi_da():
    units = [
        _Unit(kind="unit0", label=None, text="p0 p0"),  # 2 tokens
        _Unit(kind="point", label="a", text="a1 a2 a3"),  # 3 tokens
        _Unit(kind="point", label="b", text="b1 b2 b3"),  # 3 tokens
        _Unit(kind="point", label="c", text="c1 c2 c3 c4"),  # 4 tokens
        _Unit(kind="point", label="d", text="d1"),  # 1 token
    ]
    groups = _pack_units(units, max_tokens=5)

    # Không nhóm nào vượt ngân sách.
    assert all(_fake_count_tokens(group.text) <= 5 for group in groups)
    assert [group.point_labels for group in groups] == [["a"], ["b"], ["c", "d"]]
    assert groups[0].text == "p0 p0\n\na1 a2 a3"
    assert groups[1].text == "b1 b2 b3"
    assert groups[2].text == "c1 c2 c3 c4\n\nd1"


def test_pack_units_don_vi_don_le_vua_khop_ngan_sach_khong_bi_tach():
    units = [_Unit(kind="point", label="a", text="a1 a2 a3 a4 a5")]
    groups = _pack_units(units, max_tokens=5)
    assert len(groups) == 1
    assert groups[0].text == "a1 a2 a3 a4 a5"


def test_pack_units_khong_overlap_giua_cac_nhom_tang_diem():
    units = [
        _Unit(kind="point", label="a", text="a1 a2 a3"),
        _Unit(kind="point", label="b", text="b1 b2 b3"),
    ]
    groups = _pack_units(units, max_tokens=3)
    all_words = " ".join(group.text for group in groups).split()
    # Mỗi từ chỉ xuất hiện đúng 1 lần -- không overlap ở tầng Điểm (mục 4.3).
    assert sorted(all_words) == sorted(["a1", "a2", "a3", "b1", "b2", "b3"])


# ==========================================================================
# _explode_oversized -- fallback theo câu có overlap (mục 4.4)
# ==========================================================================


def test_explode_oversized_fallback_cau_co_overlap_1_cau_cuoi():
    unit = _Unit(kind="point", label="a", text="S1. S2. S3.")
    groups = _explode_oversized(unit, max_tokens=2)

    assert [group.text for group in groups] == ["S1. S2.", "S2. S3."]
    # Câu "S2." lặp lại ở cuối chunk trước và đầu chunk sau (overlap 1 câu).
    assert groups[0].text.endswith("S2.")
    assert groups[1].text.startswith("S2.")


def test_explode_oversized_khong_vuot_ngan_sach_sau_khi_tach():
    unit = _Unit(kind="point", label="a", text="S1. S2. S3.")
    groups = _explode_oversized(unit, max_tokens=2)
    assert all(_fake_count_tokens(group.text) <= 2 for group in groups)


def test_explode_oversized_tang_2_tach_theo_dau_phay_khi_khong_co_dau_cau():
    # Không có "." hay ";" -> tầng câu (tier 0) không tách được (len==1) ->
    # rơi xuống tầng mệnh đề theo dấu phẩy (tier 1, `split_finer`).
    unit = _Unit(kind="sentence", label=None, text="c1, c2, c3")
    groups = _explode_oversized(unit, max_tokens=1)
    texts = [group.text for group in groups]
    assert texts == ["c1", "c2", "c3"]


def test_explode_oversized_het_tang_van_giu_nguyen_vuot_ngan_sach():
    # Không dấu câu, không dấu phẩy -> hết tầng tách, chấp nhận giữ nguyên
    # vượt ngân sách (không có "tầng dưới" nào được spec định nghĩa).
    unit = _Unit(kind="sentence", label=None, text="mot_tu_rat_dai_khong_the_tach")
    groups = _explode_oversized(unit, max_tokens=0)
    assert len(groups) == 1
    assert groups[0].text == "mot_tu_rat_dai_khong_the_tach"


# ==========================================================================
# Breadcrumb & chunk_id
# ==========================================================================


def test_khoan_base_breadcrumb_co_khoan_number():
    khoan = KhoanNode(
        breadcrumb_prefix="Văn bản - Điều 1", khoan_number="4", content="x"
    )
    assert _khoan_base_breadcrumb(khoan) == "Văn bản - Điều 1 - Khoản 4"


def test_khoan_base_breadcrumb_khong_co_khoan_number():
    khoan = KhoanNode(
        breadcrumb_prefix="Văn bản - Điều 1", khoan_number=None, content="x"
    )
    assert _khoan_base_breadcrumb(khoan) == "Văn bản - Điều 1"


def test_compose_split_breadcrumb_khong_doi_neu_chi_1_chunk():
    result = _compose_split_breadcrumb("base", ["a"], 1, 1, "câu phủ định")
    assert result == "base"


def test_compose_split_breadcrumb_1_diem():
    result = _compose_split_breadcrumb("base", ["a"], 1, 2, None)
    assert result == "base - Điểm a (phần 1/2)"


def test_compose_split_breadcrumb_nhieu_diem_gop():
    result = _compose_split_breadcrumb("base", ["a", "b"], 1, 2, None)
    assert result == "base - Điểm a, b (phần 1/2)"


def test_compose_split_breadcrumb_khong_co_diem_tach_theo_cau():
    result = _compose_split_breadcrumb("base", [], 2, 3, None)
    assert result == "base (phần 2/3)"


def test_compose_split_breadcrumb_kem_cau_phu_dinh():
    result = _compose_split_breadcrumb("base", ["a"], 1, 2, "Câu phủ định.")
    assert result == "base - Điểm a (phần 1/2) - Câu phủ định."


def test_make_chunk_id_deterministic():
    id1 = _make_chunk_id("41/2024/QH15", "breadcrumb A")
    id2 = _make_chunk_id("41/2024/QH15", "breadcrumb A")
    assert id1 == id2
    assert id1 == hashlib.sha256(b"41/2024/QH15::breadcrumb A").hexdigest()


def test_make_chunk_id_khac_nhau_khi_breadcrumb_khac():
    id1 = _make_chunk_id("41/2024/QH15", "breadcrumb A")
    id2 = _make_chunk_id("41/2024/QH15", "breadcrumb B")
    assert id1 != id2


def test_make_chunk_id_khac_nhau_khi_source_document_khac():
    id1 = _make_chunk_id("doc-A", "breadcrumb")
    id2 = _make_chunk_id("doc-B", "breadcrumb")
    assert id1 != id2


# ==========================================================================
# split_khoan -- điểm vào công khai (mục 4, 5)
# ==========================================================================


def test_split_khoan_giu_nguyen_khi_khong_vuot_ngan_sach():
    khoan = KhoanNode(
        breadcrumb_prefix="Văn bản - Điều 1",
        khoan_number="1",
        content="Một câu ngắn.",
    )
    chunks = split_khoan(khoan, source_document="doc", max_tokens=100)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.is_split is False
    assert chunk.split_index is None
    assert chunk.split_total is None
    assert chunk.breadcrumb == "Văn bản - Điều 1 - Khoản 1"
    assert chunk.content == "Một câu ngắn."


def test_split_khoan_negation_note_gan_ngay_ca_khi_khong_bi_cat():
    # Developer note #2: negation_note tính cho MỌI Khoản, kể cả không bị
    # cắt -- ví dụ thật "Luật bảo hiểm y tế.md" dòng 58 (Khoản 3 Điều 1).
    khoan = KhoanNode(
        breadcrumb_prefix="Văn bản - Điều 1",
        khoan_number="3",
        content="Luật này không áp dụng đối với bảo hiểm y tế mang tính kinh doanh.",
    )
    chunks = split_khoan(khoan, source_document="doc", max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].is_split is False
    assert chunks[0].negation_note == (
        "Luật này không áp dụng đối với bảo hiểm y tế mang tính kinh doanh."
    )
    # Breadcrumb KHÔNG đổi khi Khoản không thực sự bị cắt (mục 3: "Khi Khoản
    # bị cắt nhỏ...").
    assert chunks[0].breadcrumb == "Văn bản - Điều 1 - Khoản 3"


def test_split_khoan_cat_theo_diem_va_lan_truyen_negation_note_toi_moi_chunk():
    khoan = KhoanNode(
        breadcrumb_prefix="Văn bản - Điều 9",
        khoan_number="4",
        content=(
            "Áp dụng trừ các trường hợp đặc biệt sau đây\n\n"
            "a) a1 a2 a3 a4 a5 a6 a7\n\n"
            "b) b1 b2 b3 b4 b5 b6 b7\n\n"
            "c) c1 c2 c3"
        ),
    )
    chunks = split_khoan(khoan, source_document="doc", max_tokens=10)

    assert len(chunks) == 3
    negation_sentence = "Áp dụng trừ các trường hợp đặc biệt sau đây"
    # Mục 4.5: negation_note lặp lại ở MỌI chunk con, kể cả chunk không chứa
    # đơn vị #0 (câu phủ định).
    for chunk in chunks:
        assert chunk.negation_note == negation_sentence
        assert chunk.is_split is True

    assert chunks[0].split_index == 1
    assert chunks[0].split_total == 3
    assert (
        chunks[0].breadcrumb
        == f"Văn bản - Điều 9 - Khoản 4 (phần 1/3) - {negation_sentence}"
    )
    assert "a1" not in chunks[0].content  # chunk 1 chỉ có đơn vị #0

    assert chunks[1].breadcrumb == (
        f"Văn bản - Điều 9 - Khoản 4 - Điểm a (phần 2/3) - {negation_sentence}"
    )
    assert chunks[2].breadcrumb == (
        f"Văn bản - Điều 9 - Khoản 4 - Điểm b, c (phần 3/3) - {negation_sentence}"
    )
    for chunk in chunks:
        assert chunk.token_count <= 10


def test_split_khoan_cat_theo_cau_khong_co_nhan_diem():
    khoan = KhoanNode(
        breadcrumb_prefix="Văn bản - Điều 2",
        khoan_number="1",
        content=(
            "Câu một hai ba bốn năm sáu bảy. "
            "Câu hai ba bốn năm sáu bảy tám. "
            "Câu cuối ngắn."
        ),
    )
    chunks = split_khoan(khoan, source_document="doc", max_tokens=8)

    assert len(chunks) == 3
    for index, chunk in enumerate(chunks, start=1):
        assert chunk.is_split is True
        assert chunk.split_index == index
        assert chunk.split_total == 3
        # Không có nhãn Điểm rõ ràng -> breadcrumb chỉ có "(phần i/n)", không
        # có đoạn "- Điểm ..." (mục 3).
        assert "Điểm" not in chunk.breadcrumb
        assert chunk.breadcrumb == f"Văn bản - Điều 2 - Khoản 1 (phần {index}/3)"
        assert chunk.token_count <= 8


def test_split_khoan_khong_bao_gio_vuot_qua_max_tokens_sau_khi_cat(monkeypatch):
    khoan = KhoanNode(
        breadcrumb_prefix="Văn bản - Điều 9",
        khoan_number="4",
        content=(
            "Đoạn mở đầu dài dòng nhiều từ để chắc chắn vượt ngân sách token "
            "ngay từ đầu vòng lặp cắt Khoản.\n\n"
            "a) Nội dung điểm a cũng khá dài để kiểm tra thuật toán cận dưới hoạt động đúng cách.\n\n"
            "b) Điểm b ngắn.\n\n"
            "c) Điểm c cũng ngắn."
        ),
    )
    chunks = split_khoan(khoan, source_document="doc", max_tokens=6)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 6 or chunk.token_count == _fake_count_tokens(
            chunk.content
        )


# ==========================================================================
# split_khoan -- ngoại lệ bảng (mục 5.1-5.4)
# ==========================================================================


def test_split_khoan_co_bang_giu_nguyen_1_chunk_bat_ke_ngan_sach():
    raw_table = (
        "| Vùng | Mức lương tối thiểu tháng<br>(Đơn vị: đồng/tháng) |\n"
        "| --- | --- |\n"
        "| Vùng I | 5.310.000 |\n"
        "| Vùng II | 4.730.000 |\n"
    )
    khoan = KhoanNode(
        breadcrumb_prefix="Văn bản - Điều 3",
        khoan_number="1",
        content="Câu dẫn trước bảng.",
        has_table=True,
        raw_table=raw_table,
    )
    # max_tokens=1 (cực nhỏ) để chắc chắn nội dung thật sự vượt ngân sách.
    chunks = split_khoan(khoan, source_document="doc", max_tokens=1)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.has_table is True
    assert chunk.is_split is False
    assert chunk.split_index is None
    assert chunk.raw_table == raw_table
    assert chunk.standardization_table is not None
    assert "Vùng I" in chunk.standardization_table
    # content = văn bản tường thuật + standardization_table, KHÔNG chứa cú
    # pháp pipe-table thô (mục 5.4).
    assert "|" not in chunk.content
    assert chunk.content.startswith("Câu dẫn trước bảng.")
    assert chunk.token_count > 1  # cố tình vượt ngân sách, chấp nhận (mục 5.1)


def test_split_khoan_co_bang_khong_co_van_ban_tuong_thuat():
    khoan = KhoanNode(
        breadcrumb_prefix="Văn bản - Điều 3",
        khoan_number="1",
        content="",
        has_table=True,
        raw_table="| A | B |\n| --- | --- |\n| 1 | 2 |\n",
    )
    chunks = split_khoan(khoan, source_document="doc", max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].content == chunks[0].standardization_table
