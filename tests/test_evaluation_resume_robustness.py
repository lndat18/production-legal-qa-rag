"""Test bổ sung cho resume/checkpoint, phân bổ số câu và `finalize` (evaluation_spec.md mục 4.2, 4.5, 8).

Bổ sung cho `test_evaluation_testset_generator.py`: mô phỏng chết THẬT giữa hai lần ghi,
Ctrl+C, lỗi HTTP có nội dung nhạy cảm, file trạng thái/raw hỏng, nguồn đổi sau khi đã
sinh, `--only` nhiều đơn vị, và thuộc tính (property) của `allocate_questions`/
`finalize_testset`. Không cần `ragas`, không cần key Groq, không gọi mạng.
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from production_legal_qa_rag.evaluation import testset_generator as tg
from production_legal_qa_rag.evaluation.models import (
    GenerationProgress,
    GoldenTestCase,
)
from production_legal_qa_rag.evaluation.unit_splitter import EvalUnit

SINGLE = "single_hop_specific_query_synthesizer"
ABSTRACT = "multi_hop_abstract_query_synthesizer"
SPECIFIC = "multi_hop_specific_query_synthesizer"
_NAME_OF_KIND = {"single_hop": SINGLE, "abstract": ABSTRACT, "specific": SPECIFIC}


def _chapter(name: str, chars: int) -> str:
    return (
        f"## Chương {name}. TIÊU ĐỀ {name}\n\n#### Điều 1. X\n\n" + "a" * chars + "\n\n"
    )


def _write_corpus(markdown_dir: Path) -> None:
    """5 đơn vị: A#1 9.000, A#2 7.000, B#1 8.000, B#2 7.000, C#1 9.000 ký tự."""
    markdown_dir.mkdir(parents=True, exist_ok=True)
    both_a = _chapter("I", 9000) + _chapter("II", 7000)
    both_b = _chapter("I", 8000) + _chapter("II", 7000)
    (markdown_dir / "A.md").write_text(both_a, encoding="utf-8")
    (markdown_dir / "B.md").write_text(both_b, encoding="utf-8")
    (markdown_dir / "C.md").write_text(_chapter("I", 9000), encoding="utf-8")


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    markdown_dir = tmp_path / "markdown"
    _write_corpus(markdown_dir)
    return markdown_dir, tmp_path / "eval"


class HttpLikeError(Exception):
    """Giống lỗi HTTP của openai/Groq: có `status_code` và thông điệp nhiều dòng."""

    status_code = 429


class Runner:
    """Sinh đúng số câu theo quota; có thể ép một lỗi bất kỳ ở một đơn vị."""

    def __init__(
        self,
        fail_on: str | None = None,
        error: BaseException | None = None,
        prefix: str = "Câu hỏi",
        llm_calls: int = 3,
    ) -> None:
        self.fail_on = fail_on
        self.error = error or RuntimeError("lỗi")
        self.prefix = prefix
        self.llm_calls = llm_calls
        self.calls: list[str] = []
        self._serial = 0

    def run_unit(
        self,
        unit: EvalUnit,
        quota: tg.QuestionQuota,
        knowledge_graph_path: Path,
        *,
        reuse_knowledge_graph: bool,
    ) -> tg.UnitResult:
        key = tg.unit_key(unit)
        self.calls.append(key)
        if key == self.fail_on:
            raise self.error
        cases = []
        for kind in tg.QUESTION_TYPES:
            for _ in range(getattr(quota, kind)):
                self._serial += 1
                cases.append(
                    GoldenTestCase(
                        user_input=f"{self.prefix} {self._serial}?",
                        reference="Đáp án",
                        reference_contexts=["Ngữ cảnh"],
                        synthesizer_name=_NAME_OF_KIND[kind],
                    )
                )
        return tg.UnitResult(cases=cases, llm_calls=self.llm_calls)


def _unit(document: str, index: int, chars: int) -> EvalUnit:
    return EvalUnit(
        source_document=document, index=index, title=f"Chương {index}", text="a" * chars
    )


def _case(document: str, kind: str, serial: int) -> GoldenTestCase:
    return GoldenTestCase(
        user_input=f"Câu {document} {kind} {serial}?",
        reference="Đáp án",
        reference_contexts=["Ngữ cảnh"],
        synthesizer_name=_NAME_OF_KIND[kind],
        source_document=document,
        source_section="Chương I",
    )


def _read_rows(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / tg.RAW_TESTSET_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


# ==========================================================================
# Chết THẬT giữa hai lần ghi, Ctrl+C, lỗi HTTP nhạy cảm (mục 4.5, 8)
# ==========================================================================


def test_chet_khi_ghi_progress_lan_sau_khong_sinh_lai_va_khong_trung_dong(
    dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
):
    markdown_dir, output_dir = dirs
    real_save = tg.save_progress

    def _chet_luc_ghi_progress(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("mất điện giữa hai bước")

    monkeypatch.setattr(tg, "save_progress", _chet_luc_ghi_progress)
    with pytest.raises(OSError):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=Runner())
    monkeypatch.setattr(tg, "save_progress", real_save)
    rows_after_crash = _read_rows(output_dir)
    assert not (output_dir / tg.PROGRESS_FILENAME).exists()
    resumed = Runner(prefix="Lần hai")

    tg.generate_testset(markdown_dir, output_dir, unit_runner=resumed)

    assert resumed.calls == ["B.md#2", "B.md#1", "A.md#1", "C.md#1"]
    rows = _read_rows(output_dir)
    assert rows[: len(rows_after_crash)] == rows_after_crash
    user_inputs = [row["user_input"] for row in rows]
    assert len(set(user_inputs)) == len(user_inputs) == 240
    recovered = tg.load_progress(output_dir / tg.PROGRESS_FILENAME).units["A.md#2"]
    section_rows = [r for r in rows if r["source_section"] == recovered.title]
    in_a = [r for r in section_rows if r["source_document"] == "A.md"]
    assert sum(recovered.questions.values()) == len(in_a) == len(rows_after_crash)


def test_ctrl_c_giua_don_vi_ghi_last_failure_khong_ghi_don_vi_do_va_nem_lai(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    runner = Runner(fail_on="B.md#2", error=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)

    assert runner.calls == ["A.md#2", "B.md#2"]
    progress = tg.load_progress(output_dir / tg.PROGRESS_FILENAME)
    assert set(progress.units) == {"A.md#2"}
    assert progress.last_failure is not None
    assert progress.last_failure.unit == "B.md#2"
    assert progress.last_failure.error == "KeyboardInterrupt"
    assert {r["source_document"] for r in _read_rows(output_dir)} == {"A.md"}


def test_loi_http_chi_ghi_dong_dau_da_cat_ngan_va_khong_lo_noi_dung(
    dirs: tuple[Path, Path], caplog: pytest.LogCaptureFixture
):
    markdown_dir, output_dir = dirs
    message = "Rate limit reached " + "x" * 500 + "\nNỘI DUNG BÍ MẬT của câu hỏi"
    runner = Runner(fail_on="A.md#2", error=HttpLikeError(message))

    with (
        caplog.at_level(logging.INFO, logger=tg.logger.name),
        pytest.raises(tg.UnitGenerationError) as excinfo,
    ):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)

    failure = tg.load_progress(output_dir / tg.PROGRESS_FILENAME).last_failure
    assert failure is not None
    assert failure.error.startswith("HttpLikeError (HTTP 429): Rate limit reached")
    assert len(failure.error) <= len("HttpLikeError (HTTP 429): ") + 200
    for text in (failure.error, str(excinfo.value), caplog.text):
        assert "BÍ MẬT" not in text


def test_loi_khong_co_status_http_chi_ghi_ten_loai_khong_ghi_thong_diep(
    dirs: tuple[Path, Path], caplog: pytest.LogCaptureFixture
):
    markdown_dir, output_dir = dirs
    runner = Runner(fail_on="A.md#2", error=ValueError("câu hỏi bí mật: Điều 5"))

    with (
        caplog.at_level(logging.INFO, logger=tg.logger.name),
        pytest.raises(tg.UnitGenerationError),
    ):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)

    failure = tg.load_progress(output_dir / tg.PROGRESS_FILENAME).last_failure
    assert failure is not None
    assert failure.error == "ValueError"
    assert "bí mật" not in caplog.text


def test_hai_lan_loi_lien_tiep_last_failure_tro_sang_don_vi_moi(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    with pytest.raises(tg.UnitGenerationError):
        tg.generate_testset(
            markdown_dir, output_dir, unit_runner=Runner(fail_on="B.md#1")
        )

    with pytest.raises(tg.UnitGenerationError):
        tg.generate_testset(
            markdown_dir, output_dir, unit_runner=Runner(fail_on="A.md#1")
        )

    progress = tg.load_progress(output_dir / tg.PROGRESS_FILENAME)
    assert set(progress.units) == {"A.md#2", "B.md#2", "B.md#1"}
    assert progress.last_failure is not None
    assert progress.last_failure.unit == "A.md#1"


# ==========================================================================
# Ghi nguyên tử, file .tmp còn sót, raw hỏng, progress hỏng (mục 4.5, 8)
# ==========================================================================


def test_file_tmp_con_sot_tu_lan_chet_truoc_khong_lam_hong_lan_ghi_sau(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    output_dir.mkdir()
    for name in (tg.PROGRESS_FILENAME, tg.RAW_TESTSET_FILENAME):
        (output_dir / f"{name}.tmp").write_text("{dở dang", encoding="utf-8")

    tg.generate_testset(markdown_dir, output_dir, unit_runner=Runner())

    assert len(_read_rows(output_dir)) == 240
    assert len(tg.load_progress(output_dir / tg.PROGRESS_FILENAME).units) == 5
    assert not list(output_dir.glob("*.tmp"))


def test_append_raw_cases_loi_khi_doi_ten_thi_raw_cu_con_nguyen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    raw_path = tmp_path / tg.RAW_TESTSET_FILENAME
    tg.append_raw_cases(raw_path, [_case("A.md", "single_hop", 1)])
    original = raw_path.read_bytes()

    def _doi_ten_loi(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("đầy đĩa")

    monkeypatch.setattr(os, "replace", _doi_ten_loi)
    with pytest.raises(OSError):
        tg.append_raw_cases(raw_path, [_case("A.md", "single_hop", 2)])

    assert raw_path.read_bytes() == original


@pytest.mark.parametrize("content", ["{không phải json", '{"a": 1}', "[1, 2]", ""])
def test_generate_raw_hong_thi_raise_khong_ghi_de_va_khong_goi_runner(
    dirs: tuple[Path, Path], content: str
):
    markdown_dir, output_dir = dirs
    output_dir.mkdir()
    raw_path = output_dir / tg.RAW_TESTSET_FILENAME
    raw_path.write_text(content, encoding="utf-8")
    runner = Runner()

    with pytest.raises(tg.EvalInputError):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)

    assert runner.calls == []
    assert raw_path.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("content", ["", "[]", "null", "[1]", '{"units": []}'])
def test_progress_sai_hinh_dang_thi_raise_ke_ca_file_rong(
    dirs: tuple[Path, Path], content: str
):
    markdown_dir, output_dir = dirs
    output_dir.mkdir()
    (output_dir / tg.PROGRESS_FILENAME).write_text(content, encoding="utf-8")
    runner = Runner()

    with pytest.raises(tg.EvalInputError, match="hỏng"):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)

    assert runner.calls == []


def test_progress_khong_phai_utf8_la_loi_dau_vao_khong_coi_nhu_chua_lam_gi(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    output_dir.mkdir()
    (output_dir / tg.PROGRESS_FILENAME).write_bytes(b"\xff\xfe{\x00")
    runner = Runner()

    with pytest.raises(tg.EvalInputError, match="hỏng"):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)

    assert runner.calls == []


def test_raw_khong_phai_utf8_la_loi_dau_vao_ca_o_generate_lan_finalize(
    dirs: tuple[Path, Path],
):
    from tools import generate_testset

    markdown_dir, output_dir = dirs
    output_dir.mkdir()
    raw_path = output_dir / tg.RAW_TESTSET_FILENAME
    raw_path.write_bytes("[]".encode("utf-16"))  # vd. lưu nhầm UTF-16 khi sửa tay
    runner = Runner()

    with pytest.raises(tg.EvalInputError, match="không phải"):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)
    with pytest.raises(tg.EvalInputError):
        tg.finalize_golden_testset(output_dir)
    generate_result = CliRunner().invoke(
        generate_testset.app,
        [
            "generate",
            "--markdown-dir",
            str(markdown_dir),
            "--output-dir",
            str(output_dir),
        ],
    )
    finalize_result = CliRunner().invoke(
        generate_testset.app, ["finalize", "--output-dir", str(output_dir)]
    )

    assert runner.calls == []
    assert generate_result.exit_code == 2
    assert finalize_result.exit_code == 2
    assert not (output_dir / tg.GOLDEN_TESTSET_FILENAME).exists()


def test_progress_khong_phai_utf8_qua_cli_thoat_ma_2(dirs: tuple[Path, Path]):
    from tools import generate_testset

    markdown_dir, output_dir = dirs
    output_dir.mkdir()
    (output_dir / tg.PROGRESS_FILENAME).write_bytes(b"\xff\xfe{\x00")

    result = CliRunner().invoke(
        generate_testset.app,
        [
            "generate",
            "--markdown-dir",
            str(markdown_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 2


def test_dry_run_va_cli_dry_run_cung_dung_khi_progress_hong(dirs: tuple[Path, Path]):
    from tools import generate_testset

    markdown_dir, output_dir = dirs
    output_dir.mkdir()
    (output_dir / tg.PROGRESS_FILENAME).write_text("{hỏng", encoding="utf-8")
    args = [
        "generate",
        "--dry-run",
        "--markdown-dir",
        str(markdown_dir),
        "--output-dir",
        str(output_dir),
    ]

    with pytest.raises(tg.EvalInputError):
        tg.plan_generation(markdown_dir, output_dir)
    result = CliRunner().invoke(generate_testset.app, args)

    assert result.exit_code == 2
    assert "hỏng" in result.output


# ==========================================================================
# Nguồn đổi sau khi đã sinh: chars lệch (mục 4.5, 8)
# ==========================================================================


def test_nguon_them_chu_sau_khi_da_sinh_thi_dung_ke_ca_khi_only_chon_don_vi_khac(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    tg.generate_testset(markdown_dir, output_dir, unit_runner=Runner())
    with (markdown_dir / "A.md").open("a", encoding="utf-8") as handle:
        handle.write("thêm một đoạn vào Chương II\n")
    runner = Runner()

    with pytest.raises(tg.EvalInputError, match=r"A\.md#2"):
        tg.generate_testset(markdown_dir, output_dir, only=["C.md"], unit_runner=runner)
    with pytest.raises(tg.EvalInputError, match=r"A\.md#2"):
        tg.plan_generation(markdown_dir, output_dir)

    assert runner.calls == []


# ==========================================================================
# --only, --testset-size, --append (mục 4.2, 4.5)
# ==========================================================================


def test_only_nhieu_don_vi_van_chay_theo_thu_tu_nho_den_lon_khong_theo_thu_tu_go(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    runner = Runner()

    tg.generate_testset(
        markdown_dir, output_dir, only=["C.md", "A.md#2"], unit_runner=runner
    )

    assert runner.calls == ["A.md#2", "C.md#1"]


def test_testset_size_chia_cho_dung_cac_don_vi_da_chon_tong_dung_n(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs

    tg.generate_testset(
        markdown_dir,
        output_dir,
        only=["A.md#2", "B.md#2"],
        testset_size=20,
        unit_runner=Runner(),
    )

    progress = tg.load_progress(output_dir / tg.PROGRESS_FILENAME)
    assert sum(sum(u.questions.values()) for u in progress.units.values()) == 20
    assert len(_read_rows(output_dir)) == 20


def test_append_khi_don_vi_moi_chi_co_trong_raw_cong_don_len_so_da_khoi_phuc(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    output_dir.mkdir()
    recovered_row = _case("A.md", "single_hop", 1).model_copy(
        update={"source_section": "Chương II. TIÊU ĐỀ II"}
    )
    raw_path = output_dir / tg.RAW_TESTSET_FILENAME
    raw_path.write_text(
        json.dumps([recovered_row.model_dump()], ensure_ascii=False), encoding="utf-8"
    )
    runner = Runner(prefix="Sinh bù")

    tg.generate_testset(
        markdown_dir,
        output_dir,
        only=["A.md#2"],
        append=True,
        testset_size=10,
        unit_runner=runner,
    )

    assert runner.calls == ["A.md#2"]
    progress = tg.load_progress(output_dir / tg.PROGRESS_FILENAME)
    questions = progress.units["A.md#2"].questions
    assert questions == {"single_hop": 1 + 8, "abstract": 1, "specific": 1}


def test_append_toan_cau_trung_thi_khong_them_dong_va_progress_khong_tang_cau(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    tg.generate_testset(markdown_dir, output_dir, unit_runner=Runner())
    rows_before = _read_rows(output_dir)
    before = tg.load_progress(output_dir / tg.PROGRESS_FILENAME).units["A.md#2"]

    # Cùng prefix và bộ đếm bắt đầu lại từ 1 -> mọi `user_input` đều trùng câu đã có.
    report = tg.generate_testset(
        markdown_dir,
        output_dir,
        only=["A.md#2"],
        append=True,
        testset_size=10,
        unit_runner=Runner(),
    )

    assert report.new_questions == 0
    assert _read_rows(output_dir) == rows_before
    after = tg.load_progress(output_dir / tg.PROGRESS_FILENAME).units["A.md#2"]
    assert after.questions == before.questions
    assert after.llm_calls == before.llm_calls + 3


# ==========================================================================
# allocate_questions / split_question_mix: thuộc tính với nhiều kích thước (mục 4)
# ==========================================================================


def test_split_question_mix_tong_dung_va_multi_hop_doi_xung_voi_moi_tong_den_500():
    for total in range(501):
        mix = tg.split_question_mix(total)

        assert mix.total == total
        assert mix.abstract == mix.specific
        assert mix.single_hop >= 0
        assert abs(mix.single_hop - 0.8 * total) <= 1


@pytest.mark.parametrize("seed", range(20))
def test_allocate_questions_tong_tung_loai_dung_voi_kich_thuoc_ngau_nhien(seed: int):
    rng = random.Random(seed)
    units = [
        _unit("A.md", index, rng.randint(6_000, 30_000))
        for index in range(1, rng.randint(1, 60) + 1)
    ]
    total = rng.choice([0, 1, 7, 10, 99, 180, 240, 241, 500])

    quotas = tg.allocate_questions(units, total)

    mix = tg.split_question_mix(total)
    assert set(quotas) == {tg.unit_key(u) for u in units}
    assert sum(q.single_hop for q in quotas.values()) == mix.single_hop
    assert sum(q.abstract for q in quotas.values()) == mix.abstract
    assert sum(q.specific for q in quotas.values()) == mix.specific
    assert all(min(q.single_hop, q.abstract, q.specific) >= 0 for q in quotas.values())


@pytest.mark.parametrize("seed", range(20))
def test_allocate_questions_lech_toi_da_1_cau_so_voi_ty_le_ky_tu(seed: int):
    rng = random.Random(seed)
    units = [_unit("A.md", i, rng.randint(6_000, 30_000)) for i in range(1, 51)]
    total_chars = sum(u.char_count for u in units)

    quotas = tg.allocate_questions(units, tg.GENERATE_SIZE)

    for unit in units:
        exact = 192 * unit.char_count / total_chars
        assert abs(quotas[tg.unit_key(unit)].single_hop - exact) < 1


# ==========================================================================
# finalize: đúng công thức khoá phân tầng của spec mục 4.2, thuộc tính
# ==========================================================================


def _named(names: list[str]) -> list[GoldenTestCase]:
    """Tên dạng "X1" = văn bản X, câu thứ 1; cùng loại câu nên nhóm theo văn bản."""
    return [
        GoldenTestCase(
            user_input=name,
            reference="Đáp án",
            reference_contexts=["Ngữ cảnh"],
            synthesizer_name=SINGLE,
            source_document=name[0],
        )
        for name in names
    ]


def _chosen_names(raw_names: list[str], target: int) -> list[str]:
    return [c.user_input for c in tg.finalize_testset(_named(raw_names), target)]


def test_finalize_nhom_lon_khong_nuot_nhom_nho_theo_khoa_thu_hang_tren_kich_thuoc():
    # X có 4 dòng (khoá 0,125/0,375/0,625/0,875), Y có 1 dòng (khoá 0,5).
    chosen = _chosen_names(["X1", "X2", "X3", "X4", "Y1"], target=3)

    assert chosen == ["X1", "X2", "Y1"]


def test_finalize_thu_tu_file_xen_ke_khong_doi_ket_qua_theo_nhom_va_giu_thu_tu_file():
    chosen = _chosen_names(["X1", "Y1", "X2", "X3", "X4"], target=3)

    assert chosen == ["X1", "Y1", "X2"]


def test_finalize_hoa_khoa_thi_lay_dong_dung_truoc_trong_file():
    # Hai nhóm cỡ 2: khoá 0,25/0,25/0,75/0,75; hoà khoá thì lấy dòng đứng trước.
    chosen = _chosen_names(["X1", "Y1", "X2", "Y2"], target=3)

    assert chosen == ["X1", "Y1", "X2"]


@pytest.mark.parametrize("seed", range(15))
def test_finalize_thuoc_tinh_dung_target_tat_dinh_giu_thu_tu_va_lay_tien_to_moi_nhom(
    seed: int,
):
    rng = random.Random(seed)
    kinds = list(_NAME_OF_KIND)
    raw = [
        _case(rng.choice("ABCDE") + ".md", rng.choice(kinds), serial)
        for serial in range(rng.randint(180, 400))
    ]

    chosen = tg.finalize_testset(raw)

    assert len(chosen) == tg.TARGET_SIZE
    assert chosen == tg.finalize_testset(list(raw))
    positions = [raw.index(c) for c in chosen]
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)
    groups: dict[tuple[str | None, str | None], list[int]] = {}
    for position, case in enumerate(raw):
        groups.setdefault((case.source_document, case.synthesizer_name), []).append(
            position
        )
    chosen_positions = set(positions)
    for group_positions in groups.values():
        kept = [p for p in group_positions if p in chosen_positions]
        assert kept == group_positions[: len(kept)]


def test_finalize_bo_han_mot_van_ban_thi_khong_co_dong_nao_cua_van_ban_do(
    tmp_path: Path,
):
    raw = [_case(doc, "single_hop", i) for doc in ("A.md", "C.md") for i in range(110)]
    (tmp_path / tg.RAW_TESTSET_FILENAME).write_text(
        json.dumps([c.model_dump() for c in raw], ensure_ascii=False), encoding="utf-8"
    )

    cases = tg.finalize_golden_testset(tmp_path)

    assert len(cases) == 180
    assert {c.source_document for c in cases} == {"A.md", "C.md"}
    assert abs(sum(c.source_document == "A.md" for c in cases) - 90) <= 1


def test_finalize_khong_sua_raw_va_chay_lai_cho_file_giong_het(tmp_path: Path):
    raw = [_case("A.md", "single_hop", i) for i in range(200)]
    raw_path = tmp_path / tg.RAW_TESTSET_FILENAME
    raw_path.write_text(
        json.dumps([c.model_dump() for c in raw], ensure_ascii=False), encoding="utf-8"
    )
    raw_bytes = raw_path.read_bytes()

    tg.finalize_golden_testset(tmp_path)
    first = (tmp_path / tg.GOLDEN_TESTSET_FILENAME).read_bytes()
    tg.finalize_golden_testset(tmp_path)

    assert raw_path.read_bytes() == raw_bytes
    assert (tmp_path / tg.GOLDEN_TESTSET_FILENAME).read_bytes() == first


def test_finalize_du_dung_target_khi_nguoi_dung_xoa_vua_du(tmp_path: Path):
    raw = [_case("A.md", "single_hop", i) for i in range(180)]
    (tmp_path / tg.RAW_TESTSET_FILENAME).write_text(
        json.dumps([c.model_dump() for c in raw], ensure_ascii=False), encoding="utf-8"
    )

    cases = tg.finalize_golden_testset(tmp_path)

    assert cases == raw


# ==========================================================================
# Data/schema: GenerationProgress, GoldenTestCase (mục 4.5, 5)
# ==========================================================================

_UNIT_KEY = "Luật bảo hiểm y tế.md#6"
_SPEC_PROGRESS_JSON = """
{
  "units": {
    "Luật bảo hiểm y tế.md#6": {
      "title": "Chương IX + Chương X", "chars": 6361, "estimated_tokens": 38000,
      "questions": {"single_hop": 4, "abstract": 1, "specific": 0},
      "llm_calls": 61, "seconds": 412.5, "completed_at": "2026-09-29T09:41:07+07:00"
    }
  },
  "last_failure": {
    "unit": "Văn bản hợp nhất bộ luật lao động.md#5", "error": "RateLimitError: ...",
    "at": "2026-09-29T10:12:55+07:00"
  }
}
"""


def test_progress_parse_dung_vi_du_trong_spec_giu_mui_gio():
    progress = GenerationProgress.model_validate_json(_SPEC_PROGRESS_JSON)

    unit = progress.units[_UNIT_KEY]
    assert unit.title == "Chương IX + Chương X"
    assert (unit.chars, unit.estimated_tokens, unit.llm_calls) == (6361, 38000, 61)
    assert unit.seconds == 412.5
    assert unit.questions == {"single_hop": 4, "abstract": 1, "specific": 0}
    assert unit.completed_at.utcoffset() == timedelta(hours=7)
    assert progress.last_failure is not None
    assert progress.last_failure.unit == "Văn bản hợp nhất bộ luật lao động.md#5"


def test_progress_ghi_roi_doc_lai_qua_file_giong_het(tmp_path: Path):
    path = tmp_path / tg.PROGRESS_FILENAME
    progress = GenerationProgress.model_validate_json(_SPEC_PROGRESS_JSON)

    tg.save_progress(path, progress)

    assert tg.load_progress(path) == progress
    text = path.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert _UNIT_KEY in text


def test_progress_rong_khong_chia_se_dict_giua_cac_instance():
    first, second = GenerationProgress(), GenerationProgress()
    sample = GenerationProgress.model_validate_json(_SPEC_PROGRESS_JSON)

    assert first.units == {} and first.last_failure is None
    first.units["x"] = sample.units[_UNIT_KEY]
    assert second.units == {}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda unit: unit.pop("chars"),
        lambda unit: unit.pop("completed_at"),
        lambda unit: unit.update(questions={"single_hop": "nhiều"}),
        lambda unit: unit.update(completed_at="không phải ngày"),
        lambda unit: unit.update(seconds="lâu"),
    ],
)
def test_progress_don_vi_sai_hoac_thieu_truong_thi_bi_tu_choi(mutate: Any):
    data = json.loads(_SPEC_PROGRESS_JSON)
    mutate(data["units"][_UNIT_KEY])

    with pytest.raises(ValidationError):
        GenerationProgress.model_validate(data)


def test_progress_last_failure_thieu_truong_hoac_units_sai_kieu_bi_tu_choi():
    data = json.loads(_SPEC_PROGRESS_JSON)
    del data["last_failure"]["at"]

    with pytest.raises(ValidationError):
        GenerationProgress.model_validate(data)
    with pytest.raises(ValidationError):
        GenerationProgress.model_validate({"units": ["A.md#1"]})


def test_progress_last_failure_null_hop_le():
    progress = GenerationProgress.model_validate({"last_failure": None})

    assert progress.last_failure is None


def test_golden_test_case_reference_contexts_phai_la_list_va_round_trip_giu_nguon():
    with pytest.raises(ValidationError):
        GoldenTestCase.model_validate(
            {"user_input": "a", "reference": "b", "reference_contexts": "c"}
        )
    case = _case("A.md", "abstract", 3)

    assert GoldenTestCase.model_validate(case.model_dump()) == case
    assert GoldenTestCase.model_validate_json(case.model_dump_json()) == case
