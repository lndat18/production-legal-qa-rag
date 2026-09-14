"""Unit test cho `chunking/pipeline.py` + `tools/chunk_documents.py` (mục 9, 11).

`count_tokens` được monkeypatch (không cần nạp model thật) để giữ test
nhanh -- việc chạy pipeline với tokenizer THẬT trên toàn bộ
`data/markdown/*.md` được xác nhận riêng ở `tests/test_chunking_acceptance.py`
(đánh dấu `slow`, đúng tiêu chí hoàn thành mục 11).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from production_legal_qa_rag.chunking import pipeline, splitter
from production_legal_qa_rag.chunking.models import Chunk
from production_legal_qa_rag.chunking.pipeline import (
    convert_directory,
    convert_markdown_to_chunks,
    write_atomic,
)


def _fake_count_tokens(text: str) -> int:
    return len(text.split())


@pytest.fixture(autouse=True)
def _mock_tokenizer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(splitter, "count_tokens", _fake_count_tokens)
    # `EmbeddingSettings().max_tokens` được đọc mỗi lần gọi
    # `convert_markdown_to_chunks` -- ép về giá trị lớn, ổn định, không phụ
    # thuộc `.env`/biến môi trường thật của máy chạy test.
    monkeypatch.setenv("MAX_TOKENS", "1000")


def _write_markdown(path: Path, khoan_content: str = "Nội dung khoản 1.") -> None:
    path.write_text(
        "---\n"
        'so_hieu: "01/2020/QH"\n'
        'ten_van_ban: "Văn bản mẫu"\n'
        "---\n\n"
        "Tiêu đề văn bản mẫu.\n\n"
        "#### Điều 1. Tên điều\n\n"
        "##### Khoản 1\n\n"
        f"{khoan_content}\n",
        encoding="utf-8",
    )


# ==========================================================================
# convert_markdown_to_chunks
# ==========================================================================


def test_convert_markdown_to_chunks_tra_ve_dung_so_khoan_va_chunk(tmp_path: Path):
    path = tmp_path / "a.md"
    _write_markdown(path)
    result = convert_markdown_to_chunks(path)
    assert result.khoan_count == 1
    assert result.split_khoan_count == 0
    assert len(result.chunks) == 1
    assert result.chunks[0].content == "Nội dung khoản 1."
    assert result.source_path == str(path)


def test_convert_markdown_to_chunks_dem_dung_so_khoan_bi_cat(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("MAX_TOKENS", "3")
    path = tmp_path / "a.md"
    _write_markdown(
        path,
        khoan_content="Một hai ba bốn năm. Sáu bảy tám chín mười. Mười một mười hai.",
    )
    result = convert_markdown_to_chunks(path)
    assert result.khoan_count == 1
    assert result.split_khoan_count == 1
    assert len(result.chunks) > 1


# ==========================================================================
# write_atomic (giống triết lý `formatting/pipeline.py::write_atomic`)
# ==========================================================================


def _sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="abc",
        source_document="doc",
        breadcrumb="bc",
        content="nội dung",
        token_count=2,
    )


def test_write_atomic_ghi_dung_noi_dung_jsonl(tmp_path: Path):
    output_path = tmp_path / "out" / "file.jsonl"
    write_atomic(output_path, [_sample_chunk(), _sample_chunk()])

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert list(tmp_path.rglob("*.tmp")) == []


def test_write_atomic_ghi_danh_sach_rong(tmp_path: Path):
    output_path = tmp_path / "empty.jsonl"
    write_atomic(output_path, [])
    assert output_path.read_text(encoding="utf-8") == ""


def test_write_atomic_don_dep_file_tam_khi_loi(tmp_path: Path, monkeypatch):
    output_path = tmp_path / "file.jsonl"

    def _boom(_fileno):
        raise OSError("giả lập lỗi ghi đĩa")

    monkeypatch.setattr(pipeline.os, "fsync", _boom)
    with pytest.raises(OSError):
        write_atomic(output_path, [_sample_chunk()])
    assert not output_path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


# ==========================================================================
# convert_directory -- lỗi 1 file không chặn cả batch (mục 9, 11)
# ==========================================================================


def test_convert_directory_loi_mot_file_khong_chan_ca_batch(tmp_path: Path, capsys):
    markdown_dir = tmp_path / "markdown"
    out_dir = tmp_path / "out"
    markdown_dir.mkdir()

    _write_markdown(markdown_dir / "hop_le.md")
    # File "lỗi": byte không phải UTF-8 hợp lệ -- `Path.read_text(encoding=
    # "utf-8")` bên trong `parse_markdown` sẽ raise `UnicodeDecodeError`.
    (markdown_dir / "hong.md").write_bytes(b"\xff\xfe\x00 khong phai utf-8 hop le")

    exit_code = convert_directory(markdown_dir, out_dir)

    assert exit_code == 1
    assert (out_dir / "hop_le.jsonl").exists()
    assert not (out_dir / "hong.jsonl").exists()

    captured = capsys.readouterr()
    assert "Thành công" in captured.out
    assert "Thất bại" in captured.out
    assert "1" in captured.out


def test_convert_directory_khong_co_md_tra_ve_0(tmp_path: Path):
    markdown_dir = tmp_path / "markdown_rong"
    markdown_dir.mkdir()
    assert convert_directory(markdown_dir, tmp_path / "out") == 0


def test_convert_directory_thu_muc_khong_ton_tai_tra_ve_0(tmp_path: Path):
    assert convert_directory(tmp_path / "khong_ton_tai", tmp_path / "out") == 0


def test_convert_directory_khong_de_lai_file_tam(tmp_path: Path):
    markdown_dir = tmp_path / "markdown"
    out_dir = tmp_path / "out"
    markdown_dir.mkdir()
    _write_markdown(markdown_dir / "a.md")

    convert_directory(markdown_dir, out_dir)

    assert list(out_dir.rglob("*.tmp")) == []


def test_convert_directory_giu_cau_truc_thu_muc_con(tmp_path: Path):
    markdown_dir = tmp_path / "markdown"
    out_dir = tmp_path / "out"
    (markdown_dir / "sub").mkdir(parents=True)
    _write_markdown(markdown_dir / "sub" / "a.md")

    convert_directory(markdown_dir, out_dir)

    assert (out_dir / "sub" / "a.jsonl").exists()


def test_convert_directory_deterministic_qua_2_lan_chay(tmp_path: Path):
    markdown_dir = tmp_path / "markdown"
    out_dir_1 = tmp_path / "out1"
    out_dir_2 = tmp_path / "out2"
    markdown_dir.mkdir()
    _write_markdown(markdown_dir / "a.md")

    convert_directory(markdown_dir, out_dir_1)
    convert_directory(markdown_dir, out_dir_2)

    content_1 = (out_dir_1 / "a.jsonl").read_text(encoding="utf-8")
    content_2 = (out_dir_2 / "a.jsonl").read_text(encoding="utf-8")
    assert content_1 == content_2


# ==========================================================================
# tools/chunk_documents.py -- CLI Typer
# ==========================================================================


def test_cli_chuyen_doi_toan_bo_thu_muc(tmp_path: Path):
    from tools.chunk_documents import app

    markdown_dir = tmp_path / "markdown"
    out_dir = tmp_path / "out"
    markdown_dir.mkdir()
    _write_markdown(markdown_dir / "a.md")

    runner = CliRunner()
    result = runner.invoke(
        app, ["--markdown-dir", str(markdown_dir), "--out-dir", str(out_dir)]
    )

    assert result.exit_code == 0
    assert (out_dir / "a.jsonl").exists()


def test_cli_tra_ve_exit_code_1_khi_co_file_loi(tmp_path: Path):
    from tools.chunk_documents import app

    markdown_dir = tmp_path / "markdown"
    out_dir = tmp_path / "out"
    markdown_dir.mkdir()
    (markdown_dir / "hong.md").write_bytes(b"\xff\xfe\x00 khong phai utf-8 hop le")

    runner = CliRunner()
    result = runner.invoke(
        app, ["--markdown-dir", str(markdown_dir), "--out-dir", str(out_dir)]
    )

    assert result.exit_code == 1
