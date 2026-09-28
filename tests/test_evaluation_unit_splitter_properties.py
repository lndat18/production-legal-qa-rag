"""Test thuộc tính (property) cho `evaluation/unit_splitter.py` trên văn bản tổng hợp.

Bổ sung cho `test_evaluation_unit_splitter.py` (ví dụ tay) và
`test_evaluation_corpus_units.py` (corpus thật): sinh nhiều văn bản ngẫu nhiên (seed cố
định) với số Chương/Mục/Điều/đoạn khác nhau và kiểm các bất biến của evaluation_spec.md
mục 4.4 — không đơn vị nào vượt `MAX_UNIT_CHARS`, không mất nội dung điều luật, bỏ đúng
phần mở đầu và khối chú thích, số thứ tự liên tục. Thuần Python, không cần `ragas`.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from production_legal_qa_rag.evaluation.unit_splitter import (
    MAX_UNIT_CHARS,
    EvalUnit,
    split_directory,
    split_document,
)

# Ký tự thân bài là "q": không xuất hiện trong tiêu đề, phần mở đầu hay chú thích nên
# đếm "q" trong các đơn vị = đếm nội dung điều luật còn giữ lại.
_BODY_CHAR = "q"
_PREAMBLE = "# TÊN VĂN BẢN\n\nCăn cứ ban hành\n\n"
_FOOTNOTES = "\n---\n\n[1] Luật số 1 sửa đổi, bổ sung một số điều.\n\n[2] Luật số 2.\n"


def _random_document(rng: random.Random, *, footnotes: bool) -> tuple[str, int]:
    """Trả về (nội dung .md, tổng số ký tự thân bài) của một văn bản ngẫu nhiên."""
    parts = [_PREAMBLE]
    body_chars = 0
    article = 0
    for chapter in range(1, rng.randint(1, 7) + 1):
        parts.append(f"## Chương {chapter}. TIÊU ĐỀ\n\n")
        section_count = rng.randint(0, 4)  # 0 = Chương không chia Mục
        for section in range(1, max(section_count, 1) + 1):
            if section_count:
                parts.append(f"### Mục {section}. M\n\n")
            for _ in range(rng.randint(1, 8)):
                article += 1
                parts.append(f"#### Điều {article}. X\n\n")
                for _ in range(rng.randint(1, 6)):
                    size = rng.randint(500, 4000)
                    parts.append(_BODY_CHAR * size + "\n\n")
                    body_chars += size
    if footnotes:
        parts.append(_FOOTNOTES)
    return "".join(parts), body_chars


def _body_chars(units: list[EvalUnit]) -> int:
    return sum(unit.text.count(_BODY_CHAR) for unit in units)


@pytest.mark.parametrize("seed", range(40))
def test_van_ban_ngau_nhien_khong_don_vi_nao_vuot_tran_va_khong_mat_noi_dung(
    seed: int,
):
    rng = random.Random(seed)
    content, body_chars = _random_document(rng, footnotes=rng.random() < 0.5)

    units = split_document("a.md", content)

    assert units
    assert all(unit.char_count <= MAX_UNIT_CHARS for unit in units)
    assert all(unit.text.strip() for unit in units)
    assert _body_chars(units) == body_chars


@pytest.mark.parametrize("seed", range(40))
def test_van_ban_ngau_nhien_bo_mo_dau_chu_thich_va_danh_so_lien_tuc(seed: int):
    rng = random.Random(1000 + seed)
    content, _ = _random_document(rng, footnotes=True)

    units = split_document("a.md", content)

    assert [unit.index for unit in units] == list(range(1, len(units) + 1))
    joined = "".join(unit.text for unit in units)
    assert "Căn cứ ban hành" not in joined
    assert "[1] Luật số 1" not in joined
    assert "[2] Luật số 2" not in joined
    assert units[0].text.startswith("## ")


@pytest.mark.parametrize("seed", range(10))
def test_chia_hai_lan_cho_ket_qua_giong_het(seed: int):
    content, _ = _random_document(random.Random(seed), footnotes=True)

    assert split_document("a.md", content) == split_document("a.md", content)


def test_hai_chuong_cung_so_la_ma_van_co_khoa_don_vi_khac_nhau():
    chapter = "## Chương XI. GIẢI QUYẾT TRANH CHẤP\n\n#### Điều 1. X\n\n"
    content = chapter + "q" * 8000 + "\n\n" + chapter + "q" * 8000 + "\n\n"

    units = split_document("a.md", content)

    assert [unit.index for unit in units] == [1, 2]
    assert units[0].title == units[1].title == "Chương XI. GIẢI QUYẾT TRANH CHẤP"


def test_chuong_lon_khong_muc_va_khong_dieu_tach_theo_doan_van_va_khong_mat_chu():
    body = "".join("q" * 3000 + "\n\n" for _ in range(40))
    content = "## Chương I. TIÊU ĐỀ\n\n" + body

    units = split_document("a.md", content)

    assert len(units) >= 4
    assert all(unit.char_count <= MAX_UNIT_CHARS for unit in units)
    assert _body_chars(units) == 3000 * 40
    assert units[0].title.startswith("Chương I. TIÊU ĐỀ (phần 1/")


def test_chu_thich_giua_van_ban_khong_co_dong_1_khong_bi_cat():
    content = (
        "## Chương I. A\n\n#### Điều 1. X\n\n"
        + "q" * 7000
        + "\n\n---\n\nĐiều khoản chuyển tiếp\n\n"
        + "q" * 7000
        + "\n"
    )

    units = split_document("a.md", content)

    assert _body_chars(units) == 14000
    assert "Điều khoản chuyển tiếp" in "".join(unit.text for unit in units)


def test_split_directory_theo_thu_tu_ten_file_va_bao_loi_khi_thieu_file(
    tmp_path: Path,
):
    with pytest.raises(FileNotFoundError):
        split_directory(tmp_path)
    chapter = "## Chương I. A\n\n#### Điều 1. X\n\n" + "q" * 7000 + "\n"
    (tmp_path / "b.md").write_text(chapter, encoding="utf-8")
    (tmp_path / "a.md").write_text(chapter, encoding="utf-8")
    (tmp_path / "ghi-chu.txt").write_text("không phải markdown", encoding="utf-8")

    units = split_directory(tmp_path)

    assert [(u.source_document, u.index) for u in units] == [("a.md", 1), ("b.md", 1)]
