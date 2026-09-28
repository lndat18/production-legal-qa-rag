"""Kiểm tra `unit_splitter`/`allocate_questions` trên corpus THẬT và artifact đã commit.

Đối chiếu số đo ở evaluation_spec.md mục 4.4 (50 đơn vị, 720.573 ký tự, không đơn vị nào
vượt 30.000) trên `data/markdown/` thật, và bảo đảm `data/eval/units/` + `units_plan.md`
(commit vào repo để người dùng duyệt) không lệch khỏi kết quả chia của code hiện tại.
Thuần Python, không cần `ragas`, không cần key Groq, không gọi mạng.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import unicodedata
from itertools import pairwise
from pathlib import Path

import pytest
from typer.testing import CliRunner

from production_legal_qa_rag.evaluation import testset_generator as tg
from production_legal_qa_rag.evaluation.unit_splitter import (
    MAX_UNIT_CHARS,
    MIN_UNIT_CHARS,
    EvalUnit,
    split_directory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_DIR = REPO_ROOT / "data" / "markdown"
EVAL_DIR = REPO_ROOT / "data" / "eval"

# Số đo tại thời điểm chốt spec (2026-09-28); đổi corpus hoặc quy tắc chia thì phải
# duyệt lại `units_plan.md` (mục 4.4) và cập nhật các hằng số này có chủ đích.
EXPECTED_UNIT_COUNT = 50
EXPECTED_TOTAL_CHARS = 720_573
EXPECTED_UNITS_PER_DOCUMENT = {
    "Luật bảo hiểm xã hội.md": 12,
    "Luật bảo hiểm y tế.md": 6,
    "Luật thuế thu nhập cá nhân.md": 3,
    "Quy định mức lương tối thiểu.md": 1,
    "Văn bản hợp nhất bộ luật lao động.md": 16,
    "Điều kiện lao động và quan hệ lao động.md": 12,
}
_FOOTNOTE_BLOCK_START = re.compile(r"\A\s*---[ \t]*\n\s*\[1\] ")
_FOOTNOTE_BLOCK_ANYWHERE = re.compile(r"\n---[ \t]*\n\s*\[1\] ")
_PLAN_ROW = re.compile(r"\| \d+ \|")


@pytest.fixture(scope="module")
def real_units() -> list[EvalUnit]:
    return split_directory(MARKDOWN_DIR)


def _unit_file(unit: EvalUnit) -> Path:
    stem = Path(unit.source_document).stem
    return EVAL_DIR / "units" / f"{stem}__{unit.index:02d}.md"


def _plan_rows(plan: str) -> list[list[str]]:
    return [line.split("|") for line in plan.splitlines() if _PLAN_ROW.match(line)]


# ==========================================================================
# unit_splitter trên corpus thật (mục 4.4)
# ==========================================================================


def test_corpus_that_chia_dung_50_don_vi_theo_tung_van_ban(real_units: list[EvalUnit]):
    counts: dict[str, int] = {}
    for unit in real_units:
        counts[unit.source_document] = counts.get(unit.source_document, 0) + 1

    assert len(real_units) == EXPECTED_UNIT_COUNT
    assert counts == EXPECTED_UNITS_PER_DOCUMENT


def test_corpus_that_khong_don_vi_nao_vuot_tran_hoac_duoi_san(
    real_units: list[EvalUnit],
):
    sizes = [unit.char_count for unit in real_units]

    assert max(sizes) <= MAX_UNIT_CHARS
    assert min(sizes) >= MIN_UNIT_CHARS
    assert all(unit.text.strip() for unit in real_units)


def test_corpus_that_tong_ky_tu_khop_so_do_cua_spec(real_units: list[EvalUnit]):
    assert sum(unit.char_count for unit in real_units) == EXPECTED_TOTAL_CHARS


def test_corpus_that_khoa_don_vi_duy_nhat_va_so_thu_tu_lien_tuc(
    real_units: list[EvalUnit],
):
    keys = [tg.unit_key(unit) for unit in real_units]

    # ĐKLĐ có hai "Chương XI" nhưng khoá (theo số thứ tự) vẫn khác nhau.
    assert len(set(keys)) == len(keys)
    for document, count in EXPECTED_UNITS_PER_DOCUMENT.items():
        indexes = [u.index for u in real_units if u.source_document == document]
        assert indexes == list(range(1, count + 1))


def test_corpus_that_tieu_de_don_vi_khong_trung_trong_cung_van_ban(
    real_units: list[EvalUnit],
):
    # Khôi phục sau khi chết giữa hai lần ghi (mục 4.5) khớp dòng raw với đơn vị theo
    # (source_document, source_section = title): tiêu đề trùng sẽ làm nhầm đơn vị.
    pairs = [(u.source_document, u.title) for u in real_units]

    assert len(set(pairs)) == len(pairs)


def test_corpus_that_ten_van_ban_o_dang_nfc(real_units: list[EvalUnit]):
    names = {unit.source_document for unit in real_units}

    assert all(unicodedata.is_normalized("NFC", name) for name in names)


def test_corpus_that_chi_bo_mo_dau_va_khoi_chu_thich_khong_mat_noi_dung(
    real_units: list[EvalUnit],
):
    for path in sorted(MARKDOWN_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        units = [u for u in real_units if u.source_document == path.name]

        position = 0
        first_start: int | None = None
        for unit in units:
            start = content.find(unit.text.rstrip(), position)
            assert start >= 0, f"{path.name}#{unit.index} không liên tục với nguồn"
            if first_start is None:
                first_start = start
            else:  # giữa hai đơn vị liền kề chỉ được có khoảng trắng
                assert not content[position:start].strip(), f"{path.name}#{unit.index}"
            position = start + len(unit.text.rstrip())

        assert first_start is not None
        # Phần mở đầu bị bỏ không có tiêu đề Chương; phần đuôi bị bỏ chỉ là khoảng
        # trắng hoặc khối chú thích `---` + `[1] ...`.
        assert not re.search(r"^## ", content[:first_start], re.MULTILINE)
        tail = content[position:]
        assert not tail.strip() or _FOOTNOTE_BLOCK_START.match(tail), path.name


def test_corpus_that_khong_don_vi_nao_con_khoi_chu_thich(real_units: list[EvalUnit]):
    leftovers = [u for u in real_units if _FOOTNOTE_BLOCK_ANYWHERE.search(u.text)]

    assert leftovers == []


# ==========================================================================
# allocate_questions / order_units / select_units trên đơn vị thật (mục 4, 4.5)
# ==========================================================================


def test_corpus_that_allocate_questions_dung_192_24_24(real_units: list[EvalUnit]):
    quotas = tg.allocate_questions(real_units)

    assert set(quotas) == {tg.unit_key(u) for u in real_units}
    assert sum(q.single_hop for q in quotas.values()) == 192
    assert sum(q.abstract for q in quotas.values()) == 24
    assert sum(q.specific for q in quotas.values()) == 24
    assert tg.GENERATE_SIZE == 240
    # Đơn vị nào cũng có việc để chạy; có đơn vị chỉ chạy single-hop (mục 4).
    assert all(q.single_hop >= 1 for q in quotas.values())
    assert any(q.abstract + q.specific == 0 for q in quotas.values())


def test_corpus_that_don_vi_lon_khong_nhan_it_hon_don_vi_nho_qua_1_cau(
    real_units: list[EvalUnit],
):
    quotas = tg.allocate_questions(real_units)
    ordered = tg.order_units(real_units)

    single_hop = [quotas[tg.unit_key(u)].single_hop for u in ordered]

    # Tỷ lệ theo ký tự + phần dư lớn nhất: xếp nhỏ -> lớn thì số câu chỉ có thể lệch
    # 1 do làm tròn, không bao giờ giảm nhiều hơn thế.
    assert all(later >= earlier - 1 for earlier, later in pairwise(single_hop))


def test_corpus_that_thu_tu_chay_nho_truoc_lon_sau_khop_units_plan(
    real_units: list[EvalUnit],
):
    ordered = tg.order_units(real_units)
    plan = (EVAL_DIR / "units_plan.md").read_text(encoding="utf-8")
    plan_chars = [int(row[5].strip().replace(",", "")) for row in _plan_rows(plan)]

    assert [u.char_count for u in ordered] == plan_chars
    assert tg.unit_key(ordered[0]) == "Văn bản hợp nhất bộ luật lao động.md#13"
    assert tg.unit_key(ordered[-1]) == "Quy định mức lương tối thiểu.md#1"


def test_corpus_that_select_units_chap_nhan_ten_nfd_va_khong_duoi_md(
    real_units: list[EvalUnit],
):
    typed_as_nfd = unicodedata.normalize("NFD", "Luật bảo hiểm y tế")

    whole_document = tg.select_units(real_units, [typed_as_nfd])
    single_unit = tg.select_units(real_units, ["Luật bảo hiểm y tế#6"])

    assert len(whole_document) == 6
    assert {u.source_document for u in whole_document} == {"Luật bảo hiểm y tế.md"}
    assert [tg.unit_key(u) for u in single_unit] == ["Luật bảo hiểm y tế.md#6"]


# ==========================================================================
# Artifact đã commit không lệch khỏi code (data/eval/units, units_plan.md)
# ==========================================================================


def test_artifact_units_commit_khop_tung_don_vi_ma_code_chia_ra(
    real_units: list[EvalUnit],
):
    committed = sorted((EVAL_DIR / "units").glob("*.md"))

    assert len(committed) == EXPECTED_UNIT_COUNT
    assert {p.name for p in committed} == {_unit_file(u).name for u in real_units}
    mismatched = [
        tg.unit_key(u)
        for u in real_units
        if _unit_file(u).read_text(encoding="utf-8") != u.text
    ]
    assert mismatched == []


def test_artifact_units_plan_khop_tong_so_va_tung_dong_khop_don_vi(
    real_units: list[EvalUnit],
):
    plan = (EVAL_DIR / "units_plan.md").read_text(encoding="utf-8")
    by_key = {tg.unit_key(u): u for u in real_units}
    rows = _plan_rows(plan)

    assert f"{EXPECTED_UNIT_COUNT} đơn vị, {EXPECTED_TOTAL_CHARS:,} ký tự" in plan
    assert len(rows) == EXPECTED_UNIT_COUNT
    for row in rows:
        unit = by_key[f"{row[2].strip()}#{row[3].strip()}"]
        assert row[4].strip() == unit.title
        assert int(row[5].strip().replace(",", "")) == unit.char_count


# ==========================================================================
# --dry-run trên corpus thật: không ragas, không key, không ghi file (mục 4.5, 7)
# ==========================================================================


def test_dry_run_corpus_that_in_50_dong_va_tom_tat_khong_ghi_file(tmp_path: Path):
    output_dir = tmp_path / "eval"

    plan = tg.plan_generation(MARKDOWN_DIR, output_dir)

    assert not output_dir.exists()
    lines = plan.splitlines()
    assert len(_plan_rows(plan)) == 50
    assert "Đã xong 0/50 đơn vị, còn 50 đơn vị" in plan
    assert "~4.16M token" in plan
    assert plan.count("CHẠY TIẾP THEO") == 1
    # Đơn vị nhỏ nhất chạy trước và là đơn vị được đánh dấu chạy tiếp theo.
    assert "Văn bản hợp nhất bộ luật lao động.md#13" in lines[2]
    assert "CHẠY TIẾP THEO" in lines[2]


def test_cli_dry_run_corpus_that_khong_can_key_groq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import generate_testset

    for suffix in ("", "_2", "_3", "_4", "_5", "_6"):
        monkeypatch.delenv(f"GROQ_API_KEY{suffix}", raising=False)
    args = [
        "generate",
        "--dry-run",
        "--markdown-dir",
        str(MARKDOWN_DIR),
        "--output-dir",
        str(tmp_path / "eval"),
    ]

    result = CliRunner().invoke(generate_testset.app, args)

    assert result.exit_code == 0, result.output
    assert "Đã xong 0/50 đơn vị" in result.output
    assert not (tmp_path / "eval").exists()


def test_dry_run_khong_keo_ragas_hay_ragas_runner_vao_process(tmp_path: Path):
    # Process con sạch: các test khác trong cùng session có thể đã import ragas.
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "import tools.generate_testset\n"
        "from production_legal_qa_rag.evaluation import testset_generator as tg\n"
        "tg.plan_generation(tg.DEFAULT_MARKDOWN_DIR, Path(sys.argv[1]))\n"
        "loaded = [m for m in sys.modules if m == 'ragas' or m.startswith('ragas.')]\n"
        "assert not loaded, loaded\n"
        "assert 'production_legal_qa_rag.evaluation.ragas_runner' not in sys.modules\n"
    )
    python_path = os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)])

    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "eval")],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": python_path},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


# ==========================================================================
# tools/split_eval_units.py (trước đây chưa có test)
# ==========================================================================


def _small_corpus(markdown_dir: Path) -> None:
    markdown_dir.mkdir(parents=True)
    chapter = "## Chương {n}. TIÊU ĐỀ\n\n#### Điều 1. X\n\n" + "a" * 7000 + "\n\n"
    both = chapter.format(n="I") + chapter.format(n="II")
    (markdown_dir / "A.md").write_text(both, encoding="utf-8")
    (markdown_dir / "B.md").write_text(chapter.format(n="I"), encoding="utf-8")


def test_split_eval_units_cli_ghi_dung_file_don_vi_va_ke_hoach(tmp_path: Path):
    from tools import split_eval_units

    markdown_dir, output_dir = tmp_path / "md", tmp_path / "eval"
    _small_corpus(markdown_dir)
    args = ["--markdown-dir", str(markdown_dir), "--output-dir", str(output_dir)]

    result = CliRunner().invoke(split_eval_units.app, args)

    assert result.exit_code == 0, result.output
    written = sorted(p.name for p in (output_dir / "units").glob("*.md"))
    assert written == ["A__01.md", "A__02.md", "B__01.md"]
    for unit in split_directory(markdown_dir):
        stem = Path(unit.source_document).stem
        path = output_dir / "units" / f"{stem}__{unit.index:02d}.md"
        assert path.read_text(encoding="utf-8") == unit.text
    plan = (output_dir / "units_plan.md").read_text(encoding="utf-8")
    assert "3 đơn vị" in plan
    assert "CẢNH BÁO" not in result.output


def test_split_eval_units_cli_xoa_file_cu_va_chay_lai_cho_ket_qua_giong_nhau(
    tmp_path: Path,
):
    from tools import split_eval_units

    markdown_dir, output_dir = tmp_path / "md", tmp_path / "eval"
    _small_corpus(markdown_dir)
    args = ["--markdown-dir", str(markdown_dir), "--output-dir", str(output_dir)]
    CliRunner().invoke(split_eval_units.app, args)
    stale = output_dir / "units" / "Da_xoa__09.md"
    stale.write_text("đơn vị của lần chia trước", encoding="utf-8")
    first_plan = (output_dir / "units_plan.md").read_text(encoding="utf-8")

    result = CliRunner().invoke(split_eval_units.app, args)

    assert result.exit_code == 0, result.output
    assert not stale.exists()
    assert (output_dir / "units_plan.md").read_text(encoding="utf-8") == first_plan
