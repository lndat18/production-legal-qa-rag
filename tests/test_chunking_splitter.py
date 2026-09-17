"""Unit + integration test cho `chunking/splitter.py` (chunking_spec.md mục 4, 5, 4.6).

`count_tokens` được monkeypatch bằng bộ đếm giả (đếm từ theo khoảng trắng)
để test thuật toán cắt độc lập với việc nạp model PhoBERT thật (nhanh, không
cần mạng) -- việc đếm token THẬT (word-segment PhoBERT) được xác nhận riêng
ở `tests/test_chunking_tokenizer.py` (đánh dấu `slow`).

Không còn `negation_note` -- câu dẫn được lặp lại trực tiếp vào ĐẦU `content`
của mọi chunk con (mục 4.5), không phải field riêng như thiết kế cũ.
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
    split_implicit_khoan,
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
        _Unit(kind="point", label="p0", text="p0 p0"),  # 2 tokens
        _Unit(kind="point", label="a", text="a1 a2 a3"),  # 3 tokens
        _Unit(kind="point", label="b", text="b1 b2 b3"),  # 3 tokens
        _Unit(kind="point", label="c", text="c1 c2 c3 c4"),  # 4 tokens
        _Unit(kind="point", label="d", text="d1"),  # 1 token
    ]
    groups = _pack_units(units, 5)

    # Không nhóm nào vượt ngân sách.
    assert all(_fake_count_tokens(group.text) <= 5 for group in groups)
    assert [group.point_labels for group in groups] == [
        ["p0", "a"],
        ["b"],
        ["c", "d"],
    ]
    assert groups[0].text == "p0 p0\n\na1 a2 a3"
    assert groups[1].text == "b1 b2 b3"
    assert groups[2].text == "c1 c2 c3 c4\n\nd1"


def test_pack_units_don_vi_don_le_vua_khop_ngan_sach_khong_bi_tach():
    units = [_Unit(kind="point", label="a", text="a1 a2 a3 a4 a5")]
    groups = _pack_units(units, 5)
    assert len(groups) == 1
    assert groups[0].text == "a1 a2 a3 a4 a5"


def test_pack_units_khong_overlap_giua_cac_nhom_tang_diem():
    units = [
        _Unit(kind="point", label="a", text="a1 a2 a3"),
        _Unit(kind="point", label="b", text="b1 b2 b3"),
    ]
    groups = _pack_units(units, 3)
    all_words = " ".join(group.text for group in groups).split()
    # Mỗi từ chỉ xuất hiện đúng 1 lần -- không overlap ở tầng Điểm (mục 4.3).
    assert sorted(all_words) == sorted(["a1", "a2", "a3", "b1", "b2", "b3"])


# ==========================================================================
# _explode_oversized -- fallback theo câu có overlap (mục 4.4)
# ==========================================================================


def test_explode_oversized_fallback_cau_co_overlap_1_cau_cuoi():
    unit = _Unit(kind="point", label="a", text="S1. S2. S3.")
    groups = _explode_oversized(unit, 2)

    assert [group.text for group in groups] == ["S1. S2.", "S2. S3."]
    # Câu "S2." lặp lại ở cuối chunk trước và đầu chunk sau (overlap 1 câu).
    assert groups[0].text.endswith("S2.")
    assert groups[1].text.startswith("S2.")


def test_explode_oversized_khong_vuot_ngan_sach_sau_khi_tach():
    unit = _Unit(kind="point", label="a", text="S1. S2. S3.")
    groups = _explode_oversized(unit, 2)
    assert all(_fake_count_tokens(group.text) <= 2 for group in groups)


def test_explode_oversized_tang_2_tach_theo_dau_phay_khi_khong_co_dau_cau():
    # Không có "." hay ";" -> tầng câu (tier 0) không tách được (len==1) ->
    # rơi xuống tầng mệnh đề theo dấu phẩy (tier 1, `split_finer`).
    unit = _Unit(kind="sentence", label=None, text="c1, c2, c3")
    groups = _explode_oversized(unit, 1)
    texts = [group.text for group in groups]
    assert texts == ["c1", "c2", "c3"]


def test_explode_oversized_het_tang_van_giu_nguyen_vuot_ngan_sach():
    # Không dấu câu, không dấu phẩy -> hết tầng tách, chấp nhận giữ nguyên
    # vượt ngân sách (không có "tầng dưới" nào được spec định nghĩa).
    unit = _Unit(kind="sentence", label=None, text="mot_tu_rat_dai_khong_the_tach")
    groups = _explode_oversized(unit, 0)
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
    result = _compose_split_breadcrumb("base", ["a"], 1, 1)
    assert result == "base"


def test_compose_split_breadcrumb_1_diem():
    result = _compose_split_breadcrumb("base", ["a"], 1, 2)
    assert result == "base - Điểm a (phần 1/2)"


def test_compose_split_breadcrumb_nhieu_diem_gop():
    result = _compose_split_breadcrumb("base", ["a", "b"], 1, 2)
    assert result == "base - Điểm a, b (phần 1/2)"


def test_compose_split_breadcrumb_khong_co_diem_tach_theo_cau():
    result = _compose_split_breadcrumb("base", [], 2, 3)
    assert result == "base (phần 2/3)"


def test_make_chunk_id_deterministic():
    id1 = _make_chunk_id("LUẬT BẢO HIỂM XÃ HỘI", "breadcrumb A")
    id2 = _make_chunk_id("LUẬT BẢO HIỂM XÃ HỘI", "breadcrumb A")
    assert id1 == id2
    assert (
        id1 == hashlib.sha256("LUẬT BẢO HIỂM XÃ HỘI::breadcrumb A".encode()).hexdigest()
    )


def test_make_chunk_id_khac_nhau_khi_breadcrumb_khac():
    id1 = _make_chunk_id("doc", "breadcrumb A")
    id2 = _make_chunk_id("doc", "breadcrumb B")
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


def test_split_khoan_cat_theo_diem_va_lap_lai_cau_dan_toi_moi_chunk():
    # Ví dụ tinh thần Điều 85 Khoản 1 `Luật bảo hiểm xã hội.md` (mục 4.5,
    # mục 11): câu dẫn được lặp lại NGUYÊN VĂN vào đầu content của MỌI chunk
    # con, kể cả chunk không chứa đơn vị #0 gốc. Số liệu chọn sao cho câu dẫn
    # (2 token) + budget_hiệu_dụng (5 token, sau khi trừ câu dẫn khỏi
    # max_tokens=7) đủ chỗ cho từng Điểm (3 token) riêng lẻ, không rơi vào
    # trường hợp hiếm "câu dẫn+Điểm đầu vượt ngân sách" của mục 4.4.
    khoan = KhoanNode(
        breadcrumb_prefix="Văn bản - Điều 9",
        khoan_number="4",
        content=("Câu dẫn\n\na) a1 a2 a3\n\nb) b1 b2 b3\n\nc) c1 c2 c3"),
    )
    chunks = split_khoan(khoan, source_document="doc", max_tokens=7)

    assert len(chunks) == 3
    preamble = "Câu dẫn"
    for chunk in chunks:
        assert chunk.content.startswith(preamble)
        assert chunk.is_split is True
        assert chunk.token_count <= 7

    assert chunks[0].split_index == 1
    assert chunks[0].split_total == 3
    assert chunks[0].breadcrumb == "Văn bản - Điều 9 - Khoản 4 - Điểm a (phần 1/3)"
    assert "a1" in chunks[0].content

    assert chunks[1].breadcrumb == "Văn bản - Điều 9 - Khoản 4 - Điểm b (phần 2/3)"
    assert chunks[2].breadcrumb == "Văn bản - Điều 9 - Khoản 4 - Điểm c (phần 3/3)"


def test_split_khoan_cau_dan_lap_lai_ke_ca_chi_1_chunk_con(monkeypatch):
    # mục 4.3: "kể cả chunk con chỉ có đúng 1 chunk duy nhất (Khoản bị cắt
    # nhưng chỉ sinh ra 1 chunk, hiếm gặp)" -- ở đây ép budget để 1 Điểm duy
    # nhất tự nó không tách được thành > 1 group.
    khoan = KhoanNode(
        breadcrumb_prefix="Văn bản - Điều 9",
        khoan_number="4",
        content="Câu dẫn dài\n\na) noi dung diem a",
    )
    # full content vượt max_tokens=3 nhưng chỉ có 1 Điểm -> 1 group duy nhất.
    chunks = split_khoan(khoan, source_document="doc", max_tokens=3)
    assert len(chunks) == 1
    assert chunks[0].content.startswith("Câu dẫn dài")
    assert chunks[0].is_split is False


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


def test_split_khoan_khong_bao_gio_vuot_qua_max_tokens_sau_khi_cat():
    # Số liệu chọn sao cho câu dẫn nhỏ (2 token) so với max_tokens=6 (budget
    # hiệu dụng=4), mỗi Điểm cũng đủ nhỏ để tự nó nằm gọn trong budget --
    # tình huống bình thường của mục 4.3 (không rơi vào trường hợp hiếm mục
    # 4.4 "câu dẫn tự nó đã gần/vượt max_tokens").
    khoan = KhoanNode(
        breadcrumb_prefix="Văn bản - Điều 9",
        khoan_number="4",
        content="Mở đầu\n\na) a1 a2 a3\n\nb) b1 b2 b3 b4\n\nc) c1 c2",
    )
    chunks = split_khoan(khoan, source_document="doc", max_tokens=6)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 6
        assert chunk.content.startswith("Mở đầu")


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


# ==========================================================================
# split_implicit_khoan -- frontmatter/backmatter (mục 4.6)
# ==========================================================================


def test_split_implicit_khoan_giu_nguyen_khi_khong_vuot_ngan_sach():
    chunks = split_implicit_khoan(
        "Nội dung mở đầu ngắn.",
        breadcrumb_prefix="LUẬT VĂN BẢN MẪU",
        source_document="LUẬT VĂN BẢN MẪU",
        max_tokens=100,
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.is_split is False
    assert chunk.breadcrumb == "LUẬT VĂN BẢN MẪU"
    assert chunk.content == "Nội dung mở đầu ngắn."


def test_split_implicit_khoan_cat_khi_vuot_ngan_sach_them_phan_i_n():
    content = " ".join(f"từ{i}" for i in range(1, 30))
    chunks = split_implicit_khoan(
        content,
        breadcrumb_prefix="LUẬT VĂN BẢN MẪU",
        source_document="LUẬT VĂN BẢN MẪU",
        max_tokens=5,
    )
    assert len(chunks) > 1
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        assert chunk.is_split is True
        assert chunk.split_index == index
        assert chunk.split_total == total
        assert chunk.breadcrumb == f"LUẬT VĂN BẢN MẪU (phần {index}/{total})"
        # Không nhãn Điểm nào (mục 4.6: không có cấu trúc Điểm).
        assert "Điểm" not in chunk.breadcrumb


def test_split_implicit_khoan_breadcrumb_prefix_khac_nhau_frontmatter_backmatter():
    # mục 4.6: 2 vùng dùng breadcrumb khác nhau để tránh trùng chunk_id.
    front = split_implicit_khoan(
        "Nội dung.",
        breadcrumb_prefix="LUẬT VĂN BẢN MẪU",
        source_document="LUẬT VĂN BẢN MẪU",
        max_tokens=100,
    )
    back = split_implicit_khoan(
        "Nội dung.",
        breadcrumb_prefix="LUẬT VĂN BẢN MẪU - Chú thích sửa đổi (cuối văn bản)",
        source_document="LUẬT VĂN BẢN MẪU",
        max_tokens=100,
    )
    assert front[0].chunk_id != back[0].chunk_id
    assert back[0].breadcrumb == "LUẬT VĂN BẢN MẪU - Chú thích sửa đổi (cuối văn bản)"


def test_split_implicit_khoan_khong_overlap_giua_cac_phan():
    content = " ".join(f"từ{i}" for i in range(1, 30))
    chunks = split_implicit_khoan(
        content,
        breadcrumb_prefix="doc",
        source_document="doc",
        max_tokens=5,
    )
    all_words = " ".join(chunk.content for chunk in chunks).split()
    # chunk_overlap=0 (mục 4.6) -- không lặp lại nội dung giữa các phần.
    assert sorted(all_words) == sorted(content.split())
