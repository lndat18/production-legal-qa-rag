"""Tiêu chí hoàn thành `chunking_spec.md` mục 11 -- chạy pipeline THẬT trên
toàn bộ `data/markdown/*.md`, dùng tokenizer PhoBERT thật (không mock).

Đánh dấu `slow` (nạp model thật, chạy trên toàn bộ corpus) -- CI job `checks`
bỏ qua (`pytest -m "not slow"`); tester chạy riêng làm bằng chứng xác nhận
thủ công yêu cầu ở mục 11 trước khi merge (đọc báo cáo tester để biết kết quả
chạy, không phụ thuộc CI job `checks` chạy lại các test này mỗi lần).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from production_legal_qa_rag.chunking.pipeline import convert_markdown_to_chunks
from production_legal_qa_rag.config import EmbeddingSettings

pytestmark = pytest.mark.slow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_DIR = PROJECT_ROOT / "data" / "markdown"
MARKDOWN_FILES = sorted(MARKDOWN_DIR.glob("*.md")) if MARKDOWN_DIR.exists() else []

_MAX_TOKENS = EmbeddingSettings().max_tokens


def _require_corpus() -> None:
    if not MARKDOWN_FILES:
        pytest.skip("data/markdown/*.md không tồn tại")


# ==========================================================================
# "mọi chunk có token_count <= MAX_TOKENS trừ chunk has_table=True"
# ==========================================================================


def test_toan_bo_corpus_khong_vuot_ngan_sach_token_tru_bang():
    _require_corpus()
    violations: list[str] = []
    for path in MARKDOWN_FILES:
        result = convert_markdown_to_chunks(path)
        assert len(result.chunks) > 0, f"{path.name}: không sinh ra chunk nào"
        for chunk in result.chunks:
            if not chunk.has_table and chunk.token_count > _MAX_TOKENS:
                violations.append(
                    f"{path.name}: {chunk.breadcrumb!r} ({chunk.token_count} > {_MAX_TOKENS})"
                )
    assert not violations, "Chunk vượt MAX_TOKENS mà không có bảng:\n" + "\n".join(
        violations
    )


def test_toan_bo_corpus_breadcrumb_khong_rong():
    _require_corpus()
    for path in MARKDOWN_FILES:
        result = convert_markdown_to_chunks(path)
        for chunk in result.chunks:
            assert chunk.breadcrumb.strip() != "", f"{path.name}: breadcrumb rỗng"


# ==========================================================================
# Ví dụ thật negation_note (mục 4.5, mục 11)
# ==========================================================================


def test_negation_note_luat_bao_hiem_y_te_dong_58():
    path = MARKDOWN_DIR / "Luật bảo hiểm y tế.md"
    if not path.exists():
        pytest.skip(f"{path} không tồn tại")
    result = convert_markdown_to_chunks(path)
    matches = [
        chunk
        for chunk in result.chunks
        if chunk.breadcrumb.endswith(
            "Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng - Khoản 3"
        )
    ]
    assert len(matches) == 1
    chunk = matches[0]
    assert chunk.is_split is False
    assert (
        chunk.negation_note
        == "Luật này không áp dụng đối với bảo hiểm y tế mang tính kinh doanh."
    )


def test_negation_note_luat_bao_hiem_xa_hoi_dieu_2_khoan_7():
    path = MARKDOWN_DIR / "Luật bảo hiểm xã hội.md"
    if not path.exists():
        pytest.skip(f"{path} không tồn tại")
    result = convert_markdown_to_chunks(path)
    matches = [
        chunk
        for chunk in result.chunks
        if "Điều 2." in chunk.breadcrumb
        and chunk.breadcrumb.rstrip().split(" - ")[-1].startswith("Khoản 7")
    ]
    assert len(matches) >= 1
    negation_sentence = (
        "Trường hợp không thuộc đối tượng tham gia bảo hiểm xã hội bắt buộc bao gồm:"
    )
    # Mục 4.5: MỌI chunk con sinh ra từ Khoản 7 đều có negation_note khớp câu
    # gốc (kể cả khi Khoản không bị cắt -- developer note #2).
    for chunk in matches:
        assert chunk.negation_note == negation_sentence


# ==========================================================================
# Điều 3 - Khoản 1 (bảng 4 vùng lương) `Quy định mức lương tối thiểu.md`
# ==========================================================================


def test_bang_luong_toi_thieu_dieu_3_khoan_1():
    path = MARKDOWN_DIR / "Quy định mức lương tối thiểu.md"
    if not path.exists():
        pytest.skip(f"{path} không tồn tại")
    result = convert_markdown_to_chunks(path)
    matches = [
        chunk
        for chunk in result.chunks
        if chunk.breadcrumb.endswith("Điều 3. Mức lương tối thiểu - Khoản 1")
    ]
    assert len(matches) == 1
    chunk = matches[0]

    assert chunk.is_split is False
    assert chunk.has_table is True
    assert chunk.raw_table is not None
    header_line = (
        "| Vùng | Mức lương tối thiểu tháng<br>(Đơn vị: đồng/tháng) | "
        "Mức lương tối thiểu giờ<br>(Đơn vị: đồng/giờ) |"
    )
    for expected_line in (
        header_line,
        "| Vùng I | 5.310.000 | 25.500 |",
        "| Vùng II | 4.730.000 | 22.700 |",
        "| Vùng III | 4.140.000 | 20.000 |",
        "| Vùng IV | 3.700.000 | 17.800 |",
    ):
        assert expected_line in chunk.raw_table

    assert chunk.standardization_table is not None
    lines = chunk.standardization_table.splitlines()
    assert len(lines) == 4
    assert lines[0] == (
        "Vùng I - Mức lương tối thiểu tháng(Đơn vị: đồng/tháng): 5.310.000 "
        "- Mức lương tối thiểu giờ(Đơn vị: đồng/giờ): 25.500"
    )


# ==========================================================================
# Deterministic (mục 9, 11)
# ==========================================================================


def test_deterministic_chay_lai_ra_chunk_id_giong_het():
    _require_corpus()
    for path in MARKDOWN_FILES:
        result_1 = convert_markdown_to_chunks(path)
        result_2 = convert_markdown_to_chunks(path)
        ids_1 = [chunk.chunk_id for chunk in result_1.chunks]
        ids_2 = [chunk.chunk_id for chunk in result_2.chunks]
        assert ids_1 == ids_2, f"{path.name}: chunk_id không deterministic"
        contents_1 = [chunk.content for chunk in result_1.chunks]
        contents_2 = [chunk.content for chunk in result_2.chunks]
        assert contents_1 == contents_2, f"{path.name}: content không deterministic"


def test_deterministic_chunk_id_khong_trung_lap_trong_cung_1_file():
    _require_corpus()
    for path in MARKDOWN_FILES:
        result = convert_markdown_to_chunks(path)
        ids = [chunk.chunk_id for chunk in result.chunks]
        assert len(ids) == len(set(ids)), f"{path.name}: có chunk_id trùng lặp"


# ==========================================================================
# CLI/pipeline batch -- 1 file lỗi không chặn cả batch (mục 9, 11), chạy
# trên bản sao thật của toàn bộ corpus + 1 file hỏng chèn thêm.
# ==========================================================================


def test_convert_directory_tren_toan_bo_corpus_that_co_summary(tmp_path: Path, capsys):
    _require_corpus()
    import shutil

    from production_legal_qa_rag.chunking.pipeline import convert_directory

    markdown_dir = tmp_path / "markdown"
    out_dir = tmp_path / "out"
    markdown_dir.mkdir()
    for path in MARKDOWN_FILES:
        shutil.copy(path, markdown_dir / path.name)
    (markdown_dir / "hong.md").write_bytes(b"\xff\xfe\x00 khong phai utf-8 hop le")

    exit_code = convert_directory(markdown_dir, out_dir)

    assert exit_code == 1  # có 1 file lỗi
    for path in MARKDOWN_FILES:
        assert (out_dir / f"{path.stem}.json").exists()
    assert not (out_dir / "hong.json").exists()

    captured = capsys.readouterr()
    assert f"Thành công        : {len(MARKDOWN_FILES)}" in captured.out
    assert "Thất bại          : 1" in captured.out
