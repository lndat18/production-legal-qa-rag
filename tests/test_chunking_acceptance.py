"""Tiêu chí hoàn thành `chunking_spec.md` mục 11 -- chạy pipeline THẬT trên
toàn bộ `data/markdown/*.md`, dùng tokenizer PhoBERT thật (không mock).

Đánh dấu `slow` (nạp model thật, chạy trên toàn bộ corpus) -- CI job `checks`
bỏ qua (`pytest -m "not slow"`); tester chạy riêng làm bằng chứng xác nhận
thủ công yêu cầu ở mục 11 trước khi merge.

Các giá trị literal đối chiếu trong file này đã được xác nhận thủ công bằng
cách chạy `tools/chunk_documents.py` thật trên `data/markdown/` (kết quả ghi
trong `data/chunks/*.json`, đã commit) trước khi viết test.
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


def _require_file(name: str) -> Path:
    path = MARKDOWN_DIR / name
    if not path.exists():
        pytest.skip(f"{path} không tồn tại")
    return path


# ==========================================================================
# "mọi chunk có token_count <= MAX_TOKENS trừ chunk has_table=True" (mục 11)
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
# Câu dẫn lặp lại vào content của mọi chunk con khi Khoản có Điểm bị cắt
# (mục 4.5, mục 11) -- ví dụ thật: Điều 2 Khoản 1 `Luật bảo hiểm xã hội.md`.
# ==========================================================================


def test_cau_dan_lap_lai_dieu_2_khoan_1_luat_bao_hiem_xa_hoi():
    path = _require_file("Luật bảo hiểm xã hội.md")
    result = convert_markdown_to_chunks(path)
    matches = [
        chunk
        for chunk in result.chunks
        if chunk.breadcrumb.startswith(
            "LUẬT BẢO HIỂM XÃ HỘI - Chương I - Điều 2. Đối tượng tham gia bảo hiểm "
            "xã hội bắt buộc và bảo hiểm xã hội tự nguyện - Khoản 1"
        )
    ]
    assert len(matches) > 1, "Điều 2 Khoản 1 phải bị cắt thành nhiều chunk con"
    preamble = (
        "Người lao động là công dân Việt Nam thuộc đối tượng tham gia bảo hiểm "
        "xã hội bắt buộc bao gồm:"
    )
    for chunk in matches:
        assert chunk.is_split is True
        assert chunk.content.startswith(preamble), (
            f"{chunk.breadcrumb!r} không bắt đầu bằng câu dẫn gốc"
        )
        assert "Điểm" in chunk.breadcrumb
        assert "(phần" in chunk.breadcrumb


# ==========================================================================
# Điều 3 - Khoản 1 (bảng 4 vùng lương) `Quy định mức lương tối thiểu.md`
# (mục 5, mục 11)
# ==========================================================================


def test_bang_luong_toi_thieu_dieu_3_khoan_1():
    path = _require_file("Quy định mức lương tối thiểu.md")
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
    # content = văn bản tường thuật + standardization_table (mục 5.4) --
    # không chứa cú pháp pipe-table thô.
    assert "|" not in chunk.content
    assert chunk.standardization_table in chunk.content


# ==========================================================================
# Frontmatter/backmatter `Luật bảo hiểm xã hội.md` (mục 4.6, mục 11)
# ==========================================================================


def test_frontmatter_luat_bao_hiem_xa_hoi_sinh_chunk_rieng():
    path = _require_file("Luật bảo hiểm xã hội.md")
    result = convert_markdown_to_chunks(path)
    frontmatter_chunks = [
        chunk
        for chunk in result.chunks
        if chunk.breadcrumb == "LUẬT BẢO HIỂM XÃ HỘI"
        or chunk.breadcrumb.startswith("LUẬT BẢO HIỂM XÃ HỘI (phần")
    ]
    assert len(frontmatter_chunks) >= 1
    joined_content = "\n".join(chunk.content for chunk in frontmatter_chunks)
    assert "LUẬT" in joined_content
    assert "BẢO HIỂM XÃ HỘI" in joined_content
    assert "Quốc hội ban hành" in joined_content or "được sửa đổi" in joined_content


def test_backmatter_luat_bao_hiem_xa_hoi_tach_rieng_khong_lan_khoan_15():
    path = _require_file("Luật bảo hiểm xã hội.md")
    result = convert_markdown_to_chunks(path)

    backmatter_chunks = [
        chunk
        for chunk in result.chunks
        if chunk.breadcrumb.startswith(
            "LUẬT BẢO HIỂM XÃ HỘI - Chú thích sửa đổi (cuối văn bản)"
        )
    ]
    assert len(backmatter_chunks) >= 1
    joined_backmatter = "\n".join(chunk.content for chunk in backmatter_chunks)
    assert "Điều 41. Hiệu lực thi hành" in joined_backmatter

    khoan_15 = [
        chunk
        for chunk in result.chunks
        if chunk.breadcrumb
        == "LUẬT BẢO HIỂM XÃ HỘI - Chương XI - Điều 141. Quy định chuyển tiếp - Khoản 15"
    ]
    assert len(khoan_15) == 1
    assert khoan_15[0].content == "Chính phủ quy định chi tiết Điều này."
    assert "Điều 41" not in khoan_15[0].content
    assert "Luật Nhà giáo" not in khoan_15[0].content


# ==========================================================================
# source_document đúng trên toàn bộ 6 file (mục 2, mục 11)
# ==========================================================================


def test_source_document_dung_tren_toan_bo_6_file():
    _require_corpus()
    expected = {
        "Luật bảo hiểm xã hội.md": "LUẬT BẢO HIỂM XÃ HỘI",
        "Luật bảo hiểm y tế.md": "LUẬT BẢO HIỂM Y TẾ",
        "Luật thuế thu nhập cá nhân.md": "LUẬT THUẾ THU NHẬP CÁ NHÂN",
        "Quy định mức lương tối thiểu.md": (
            "NGHỊ ĐỊNH QUY ĐỊNH MỨC LƯƠNG TỐI THIỂU ĐỐI VỚI NGƯỜI LAO ĐỘNG LÀM "
            "VIỆC THEO HỢP ĐỒNG LAO ĐỘNG"
        ),
        "Văn bản hợp nhất bộ luật lao động.md": "BỘ LUẬT LAO ĐỘNG",
        "Điều kiện lao động và quan hệ lao động.md": (
            "NGHỊ ĐỊNH QUY ĐỊNH CHI TIẾT VÀ HƯỚNG DẪN THI HÀNH MỘT SỐ ĐIỀU CỦA BỘ "
            "LUẬT LAO ĐỘNG VỀ ĐIỀU KIỆN LAO ĐỘNG VÀ QUAN HỆ LAO ĐỘNG"
        ),
    }
    for name, expected_source_document in expected.items():
        path = _require_file(name)
        result = convert_markdown_to_chunks(path)
        source_documents = {chunk.source_document for chunk in result.chunks}
        assert source_documents == {expected_source_document}, (
            f"{name}: source_document sai -- {source_documents}"
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


def test_convert_directory_khop_dung_output_da_commit_san():
    """`data/chunks/*.json` đã commit sẵn (kết quả chạy CLI thật) phải khớp
    CHÍNH XÁC kết quả chạy lại `convert_directory` trên `data/markdown/` --
    khoá lại tính deterministic + tránh output đã commit bị lệch code."""
    _require_corpus()
    import json

    from production_legal_qa_rag.chunking.pipeline import convert_directory

    chunks_dir = PROJECT_ROOT / "data" / "chunks"
    if not chunks_dir.exists():
        pytest.skip("data/chunks/ không tồn tại")

    out_dir = Path(__file__).resolve().parents[0] / "_tmp_chunks_check"
    import shutil

    if out_dir.exists():
        shutil.rmtree(out_dir)
    try:
        convert_directory(MARKDOWN_DIR, out_dir)
        for path in MARKDOWN_FILES:
            expected_path = chunks_dir / f"{path.stem}.json"
            if not expected_path.exists():
                continue
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            actual = json.loads(
                (out_dir / f"{path.stem}.json").read_text(encoding="utf-8")
            )
            assert actual == expected, (
                f"{path.name}: output lệch data/chunks/ đã commit"
            )
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
