"""Test đơn vị `partial`, đếm token theo đơn vị và dry-run dùng hệ số đo được (evaluation_spec.md mục 3.2 B, 3.3).

Bổ sung cho `test_evaluation_testset_generator.py`: `ScriptedRunner` giả lập đúng hợp đồng
`UnitRunner` (kết quả kèm `interruption`, `skipped_samples`, `tokens`, hoặc raise
`UnitGenerationError` khi không sinh được câu nào). Phần điều phối thật (raw nối trước progress,
`last_failure`, quota còn lại, mã thoát CLI) chạy nguyên bản. Không cần `ragas`, không cần key
Groq, không gọi mạng.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from production_legal_qa_rag.evaluation import testset_generator as tg
from production_legal_qa_rag.evaluation.groq_round_robin import (
    DailyQuotaExhaustedError,
)
from production_legal_qa_rag.evaluation.models import (
    GenerationProgress,
    GoldenTestCase,
    UnitProgress,
)
from production_legal_qa_rag.evaluation.unit_splitter import (
    FIXED_TOKENS_PER_UNIT,
    EvalUnit,
    split_directory,
)

_NAME_OF_KIND = {
    "single_hop": "single_hop_specific_query_synthesizer",
    "abstract": "multi_hop_abstract_query_synthesizer",
    "specific": "multi_hop_specific_query_synthesizer",
}
_QUOTA_MESSAGE = "Hết quota ngày cả 9 tài khoản Groq"


def _chapter(name: str, chars: int) -> str:
    return (
        f"## Chương {name}. TIÊU ĐỀ {name}\n\n#### Điều 1. X\n\n" + "a" * chars + "\n\n"
    )


def _write_corpus(markdown_dir: Path) -> None:
    """5 đơn vị theo thứ tự chạy: A#2, B#2 (7.000), B#1 (8.000), A#1, C#1 (9.000 ký tự)."""
    markdown_dir.mkdir(parents=True, exist_ok=True)
    (markdown_dir / "A.md").write_text(
        _chapter("I", 9000) + _chapter("II", 7000), encoding="utf-8"
    )
    (markdown_dir / "B.md").write_text(
        _chapter("I", 8000) + _chapter("II", 7000), encoding="utf-8"
    )
    (markdown_dir / "C.md").write_text(_chapter("I", 9000), encoding="utf-8")


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    markdown_dir = tmp_path / "markdown"
    _write_corpus(markdown_dir)
    return markdown_dir, tmp_path / "eval"


class ScriptedRunner:
    """Runner giả: sinh đúng quota, có thể ngắt giữa đơn vị hoặc ném lỗi theo kịch bản.

    `stop_after[key]`: chỉ trả N câu đầu rồi báo `interruption` (hết quota ngày) — dùng đúng
    MỘT lần, các lần chạy sau của đơn vị đó sinh đủ. `fail_with[key]`: ném lỗi này (một lần).
    """

    def __init__(
        self,
        *,
        stop_after: dict[str, int] | None = None,
        fail_with: dict[str, BaseException] | None = None,
        skipped: int = 0,
        tokens: int = 1000,
        reasoning_tokens: int = 200,
        prefix: str = "Câu hỏi",
    ) -> None:
        self.stop_after = dict(stop_after or {})
        self.fail_with = dict(fail_with or {})
        self.skipped = skipped
        self.tokens = tokens
        self.reasoning_tokens = reasoning_tokens
        self.prefix = prefix
        self.calls: list[tuple[str, tg.QuestionQuota]] = []
        self.reuse_flags: list[bool] = []
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
        self.calls.append((key, quota))
        self.reuse_flags.append(reuse_knowledge_graph)
        if key in self.fail_with:
            raise self.fail_with.pop(key)
        cases = self._cases_for(quota)
        interruption: BaseException | None = None
        if key in self.stop_after:
            cases = cases[: self.stop_after.pop(key)]
            interruption = DailyQuotaExhaustedError(_QUOTA_MESSAGE)
        return tg.UnitResult(
            cases=cases,
            llm_calls=5,
            tokens=self.tokens,
            reasoning_tokens=self.reasoning_tokens,
            skipped_samples=self.skipped,
            interruption=interruption,
        )

    def _cases_for(self, quota: tg.QuestionQuota) -> list[GoldenTestCase]:
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
        return cases


def _units_by_key(markdown_dir: Path) -> dict[str, EvalUnit]:
    return {tg.unit_key(unit): unit for unit in split_directory(markdown_dir)}


def _quota(markdown_dir: Path, key: str) -> tg.QuestionQuota:
    return tg.allocate_questions(split_directory(markdown_dir))[key]


def _read_rows(output_dir: Path) -> list[dict[str, Any]]:
    return tg.read_raw_rows(output_dir / tg.RAW_TESTSET_FILENAME)


def _progress(output_dir: Path) -> GenerationProgress:
    return tg.load_progress(output_dir / tg.PROGRESS_FILENAME)


def _progress_of(unit: EvalUnit, **overrides: Any) -> UnitProgress:
    fields: dict[str, Any] = {
        "title": unit.title,
        "chars": unit.char_count,
        "estimated_tokens": unit.estimated_tokens,
        "questions": {"single_hop": 0, "abstract": 0, "specific": 0},
        "llm_calls": 3,
        "seconds": 1.5,
        "completed_at": datetime(2026, 9, 29, 10, 0).astimezone(),
    }
    fields.update(overrides)
    return UnitProgress(**fields)


def _save_units(output_dir: Path, units: dict[str, UnitProgress]) -> None:
    path = output_dir / tg.PROGRESS_FILENAME
    tg.save_progress(path, GenerationProgress(units=units))


def _stop_first_unit(markdown_dir: Path, output_dir: Path, after: int = 5) -> None:
    """Chạy đến khi A.md#2 bị ngắt sau `after` câu (đơn vị `partial`, chương trình dừng)."""
    runner = ScriptedRunner(stop_after={"A.md#2": after}, prefix="Lần một")
    with pytest.raises(tg.UnitGenerationError):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)


# ==========================================================================
# Ngắt giữa đơn vị: giữ phần đã sinh, `partial`, chạy tiếp phần còn thiếu (mục 3.3)
# ==========================================================================


def test_ngat_giua_don_vi_giu_cau_da_xong_ghi_partial_last_failure_va_dung(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    quota = _quota(markdown_dir, "A.md#2")
    assert quota.single_hop > 5 and quota.abstract > 0
    runner = ScriptedRunner(stop_after={"A.md#2": 5})

    with pytest.raises(tg.UnitGenerationError) as excinfo:
        tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)

    assert excinfo.value.unit_key == "A.md#2"
    assert [key for key, _ in runner.calls] == ["A.md#2"]  # không thử đơn vị kế
    rows = _read_rows(output_dir)
    sources = {(row["source_document"], row["source_section"]) for row in rows}
    assert len(rows) == 5
    assert sources == {("A.md", "Chương II. TIÊU ĐỀ II")}
    progress = _progress(output_dir)
    partial = progress.units["A.md#2"]
    assert set(progress.units) == {"A.md#2"}
    assert partial.status == "partial"
    assert partial.questions == {"single_hop": 5, "abstract": 0, "specific": 0}
    assert partial.llm_calls == 5
    assert partial.tokens == 1000
    assert partial.reasoning_tokens == 200
    assert isinstance(partial.completed_at, datetime)
    assert progress.last_failure is not None
    assert progress.last_failure.unit == "A.md#2"
    error = progress.last_failure.error
    assert error.startswith("DailyQuotaExhaustedError (HTTP 429)")
    assert not list(output_dir.glob("*.tmp"))


def test_chay_tiep_don_vi_partial_chi_sinh_phan_con_thieu_roi_chuyen_done(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    quota = _quota(markdown_dir, "A.md#2")
    _stop_first_unit(markdown_dir, output_dir)
    before = _progress(output_dir).units["A.md#2"]
    resumed = ScriptedRunner(prefix="Lần hai")

    report = tg.generate_testset(markdown_dir, output_dir, unit_runner=resumed)

    key, remaining = resumed.calls[0]
    expected_remaining = tg.QuestionQuota(
        single_hop=quota.single_hop - 5,
        abstract=quota.abstract,
        specific=quota.specific,
    )
    assert key == "A.md#2"
    assert remaining == expected_remaining
    assert resumed.reuse_flags[0] is True  # dùng lại KG còn sót của đơn vị dở
    order = [unit_key for unit_key, _ in resumed.calls]
    assert order == ["A.md#2", "B.md#2", "B.md#1", "A.md#1", "C.md#1"]
    assert report.new_questions == 240 - 5
    progress = _progress(output_dir)
    done = progress.units["A.md#2"]
    assert done.status == "done"
    assert done.questions == quota.model_dump()
    assert done.llm_calls == 10  # cộng dồn hai lần chạy
    assert done.tokens == 2000
    assert done.reasoning_tokens == 400
    assert done.seconds >= before.seconds
    assert done.completed_at >= before.completed_at
    assert progress.last_failure is None
    rows = _read_rows(output_dir)
    user_inputs = [row["user_input"] for row in rows]
    in_unit = [row for row in rows if row["source_section"] == done.title]
    assert len(set(user_inputs)) == len(user_inputs) == 240
    assert len([r for r in in_unit if r["source_document"] == "A.md"]) == quota.total


def test_quota_con_lai_ke_ve_0_khi_da_co_nhieu_hon_quota(dirs: tuple[Path, Path]):
    markdown_dir, output_dir = dirs
    quota = _quota(markdown_dir, "A.md#2")
    assert quota.abstract >= 1
    unit = _units_by_key(markdown_dir)["A.md#2"]
    questions = {"single_hop": quota.single_hop + 50, "abstract": 1, "specific": 0}
    partial = _progress_of(unit, status="partial", questions=questions)
    _save_units(output_dir, {"A.md#2": partial})
    runner = ScriptedRunner()

    tg.generate_testset(markdown_dir, output_dir, only=["A.md#2"], unit_runner=runner)

    expected = tg.QuestionQuota(
        single_hop=0, abstract=quota.abstract - 1, specific=quota.specific
    )
    assert runner.calls == [("A.md#2", expected)]


def test_ngat_nhieu_lan_lien_tiep_cong_don_va_van_partial_den_khi_du(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    quota = _quota(markdown_dir, "A.md#2")
    _stop_first_unit(markdown_dir, output_dir, after=5)
    second = ScriptedRunner(stop_after={"A.md#2": 3}, prefix="Lần hai")
    with pytest.raises(tg.UnitGenerationError):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=second)
    mid = _progress(output_dir)

    third = ScriptedRunner(prefix="Lần ba")
    tg.generate_testset(markdown_dir, output_dir, unit_runner=third)

    assert second.calls[0][1].single_hop == quota.single_hop - 5
    assert mid.units["A.md#2"].status == "partial"
    assert mid.units["A.md#2"].questions["single_hop"] == 8
    assert mid.units["A.md#2"].llm_calls == 10
    assert mid.last_failure is not None
    assert mid.last_failure.unit == "A.md#2"
    assert third.calls[0][1].single_hop == quota.single_hop - 8
    done = _progress(output_dir).units["A.md#2"]
    assert done.status == "done"
    assert done.questions == quota.model_dump()
    assert done.llm_calls == 15
    assert done.tokens == 3000


def test_dry_run_va_tom_tat_coi_partial_la_do_va_don_vi_chay_tiep_theo(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    quota = _quota(markdown_dir, "A.md#2")
    _stop_first_unit(markdown_dir, output_dir)

    plan = tg.plan_generation(markdown_dir, output_dir)
    summary = tg.summarize_progress(markdown_dir, output_dir)

    row = plan.splitlines()[2]
    assert "A.md#2" in row
    assert f"dở (đã có 5/{quota.total} câu)" in row
    assert "CHẠY TIẾP THEO" in row
    # Đơn vị partial vẫn nằm trong danh sách chờ nên chưa được tính là đã xong.
    assert "Đã xong 0/5 đơn vị, còn 5 đơn vị" in plan
    assert "Đã xong 0/5 đơn vị, còn 5 đơn vị" in summary


def test_dry_run_don_vi_done_van_hien_thi_xong_khong_bi_lan_voi_partial(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    tg.generate_testset(markdown_dir, output_dir, unit_runner=ScriptedRunner())

    plan = tg.plan_generation(markdown_dir, output_dir)

    assert "dở" not in plan
    assert plan.count("xong ") >= 5


# ==========================================================================
# `--append` trên đơn vị đã `done` bị ngắt giữa chừng GIỮ `done` (mục 3.3)
# ==========================================================================


def _run_append_interrupted(markdown_dir: Path, output_dir: Path) -> ScriptedRunner:
    runner = ScriptedRunner(stop_after={"A.md#2": 4}, prefix="Sinh bù")
    with pytest.raises(tg.UnitGenerationError):
        tg.generate_testset(
            markdown_dir,
            output_dir,
            only=["A.md#2"],
            append=True,
            testset_size=10,
            unit_runner=runner,
        )
    return runner


def test_append_tren_don_vi_done_bi_ngat_giu_done_van_ghi_raw_va_last_failure(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    tg.generate_testset(markdown_dir, output_dir, unit_runner=ScriptedRunner())
    rows_before = _read_rows(output_dir)
    before = _progress(output_dir).units["A.md#2"]

    _run_append_interrupted(markdown_dir, output_dir)

    rows = _read_rows(output_dir)
    progress = _progress(output_dir)
    after = progress.units["A.md#2"]
    assert rows[: len(rows_before)] == rows_before
    assert len(rows) == len(rows_before) + 4
    assert after.status == "done"
    assert sum(after.questions.values()) == sum(before.questions.values()) + 4
    assert after.llm_calls == before.llm_calls + 5
    assert progress.last_failure is not None
    assert progress.last_failure.unit == "A.md#2"
    # Đơn vị vẫn `done` nên lần chạy thường sau đó không bị kẹt và không chạy lại nó.
    later = ScriptedRunner()
    report = tg.generate_testset(markdown_dir, output_dir, unit_runner=later)
    assert later.calls == []
    assert "A.md#2" in report.skipped_units


def test_cli_don_vi_bi_ngat_giua_chung_thoat_ma_1_va_in_tom_tat(
    dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
):
    from tools import generate_testset

    markdown_dir, output_dir = dirs
    runner = ScriptedRunner(stop_after={"A.md#2": 5})
    monkeypatch.setattr(tg, "build_unit_runner", lambda *_a, **_k: runner)

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

    assert result.exit_code == 1, result.output
    assert "A.md#2" in result.output
    assert "Đã xong 0/5 đơn vị" in result.output
    assert _progress(output_dir).units["A.md#2"].status == "partial"


def test_cli_append_tren_don_vi_done_bi_ngat_thoat_ma_1_va_giu_done(
    dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
):
    from tools import generate_testset

    markdown_dir, output_dir = dirs
    tg.generate_testset(markdown_dir, output_dir, unit_runner=ScriptedRunner())
    runner = ScriptedRunner(stop_after={"A.md#2": 4}, prefix="Sinh bù")
    monkeypatch.setattr(tg, "build_unit_runner", lambda *_a, **_k: runner)

    result = CliRunner().invoke(
        generate_testset.app,
        [
            "generate",
            "--only",
            "A.md#2",
            "--append",
            "--testset-size",
            "10",
            "--markdown-dir",
            str(markdown_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 1, result.output
    progress = _progress(output_dir)
    assert progress.units["A.md#2"].status == "done"
    assert progress.last_failure is not None
    assert progress.last_failure.unit == "A.md#2"


# ==========================================================================
# Sample lỗi bị bỏ (`skipped_samples`), mọi sample lỗi thì không ghi gì
# ==========================================================================


def test_sample_loi_bi_bo_don_vi_van_done_va_skipped_samples_cong_don(
    dirs: tuple[Path, Path], caplog: pytest.LogCaptureFixture
):
    markdown_dir, output_dir = dirs

    runner = ScriptedRunner(skipped=2)
    with caplog.at_level(logging.INFO, logger=tg.logger.name):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)

    progress = _progress(output_dir)
    assert all(unit.status == "done" for unit in progress.units.values())
    assert all(unit.skipped_samples == 2 for unit in progress.units.values())
    assert "bỏ 2 sample lỗi" in caplog.text
    appended = ScriptedRunner(skipped=3, prefix="Sinh bù")
    tg.generate_testset(
        markdown_dir,
        output_dir,
        only=["A.md#2"],
        append=True,
        testset_size=10,
        unit_runner=appended,
    )
    assert _progress(output_dir).units["A.md#2"].skipped_samples == 5


def _no_case_error(cause: BaseException | None) -> tg.UnitGenerationError:
    try:
        raise tg.UnitGenerationError("A.md#2", "không sinh được câu nào") from cause
    except tg.UnitGenerationError as error:
        return error


@pytest.mark.parametrize(
    ("cause", "expected_prefix"),
    [
        (
            DailyQuotaExhaustedError(_QUOTA_MESSAGE),
            "DailyQuotaExhaustedError (HTTP 429)",
        ),
        (ValueError("nội dung bí mật"), "ValueError"),
        (None, "UnitGenerationError"),
    ],
    ids=["loi-quota", "loi-parse", "khong-co-loi-goc"],
)
def test_khong_sinh_duoc_cau_nao_last_failure_theo_loi_goc_va_khong_ghi_raw_progress(
    dirs: tuple[Path, Path],
    caplog: pytest.LogCaptureFixture,
    cause: BaseException | None,
    expected_prefix: str,
):
    markdown_dir, output_dir = dirs
    runner = ScriptedRunner(fail_with={"A.md#2": _no_case_error(cause)})

    with (
        caplog.at_level(logging.INFO, logger=tg.logger.name),
        pytest.raises(tg.UnitGenerationError) as excinfo,
    ):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)

    progress = _progress(output_dir)
    assert progress.units == {}
    assert not (output_dir / tg.RAW_TESTSET_FILENAME).exists()
    assert progress.last_failure is not None
    assert progress.last_failure.unit == "A.md#2"
    assert progress.last_failure.error.startswith(expected_prefix)
    assert expected_prefix in str(excinfo.value)
    assert [key for key, _ in runner.calls] == ["A.md#2"]
    for text in (progress.last_failure.error, str(excinfo.value), caplog.text):
        assert "bí mật" not in text


# ==========================================================================
# Raw nối TRƯỚC progress ngay cả khi đơn vị là `partial` (mục 3.3, 4.5)
# ==========================================================================


def test_partial_ghi_raw_truoc_progress_chet_giua_hai_buoc_thi_lan_sau_khong_sinh_lai(
    dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
):
    markdown_dir, output_dir = dirs
    rows_when_saving: list[int] = []
    real_save = tg.save_progress

    def _die_on_save(*_args: Any, **_kwargs: Any) -> None:
        rows_when_saving.append(len(_read_rows(output_dir)))
        raise OSError("mất điện giữa hai bước")

    monkeypatch.setattr(tg, "save_progress", _die_on_save)
    crashed = ScriptedRunner(stop_after={"A.md#2": 5})
    with pytest.raises(OSError):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=crashed)
    monkeypatch.setattr(tg, "save_progress", real_save)

    # Lúc progress được ghi (và chết), 5 dòng của đơn vị đã nằm trong raw.
    assert rows_when_saving == [5]
    assert not (output_dir / tg.PROGRESS_FILENAME).exists()
    resumed = ScriptedRunner(prefix="Lần hai")

    tg.generate_testset(markdown_dir, output_dir, unit_runner=resumed)

    assert "A.md#2" not in [key for key, _ in resumed.calls]
    user_inputs = [row["user_input"] for row in _read_rows(output_dir)]
    assert len(set(user_inputs)) == len(user_inputs)


def test_unit_result_khong_dump_interruption_ra_du_lieu_luu_tru():
    result = tg.UnitResult(cases=[], llm_calls=1, interruption=RuntimeError("x"))

    assert "interruption" not in result.model_dump()


# ==========================================================================
# Token theo đơn vị: ghi, cộng dồn, bản ghi cũ không có số đo (mục 3.2 B)
# ==========================================================================


def test_token_ghi_theo_don_vi_va_cong_don_khi_append(dirs: tuple[Path, Path]):
    markdown_dir, output_dir = dirs
    tg.generate_testset(markdown_dir, output_dir, unit_runner=ScriptedRunner())
    first = _progress(output_dir)
    appended = ScriptedRunner(tokens=500, reasoning_tokens=50, prefix="Sinh bù")

    tg.generate_testset(
        markdown_dir,
        output_dir,
        only=["A.md#2"],
        append=True,
        testset_size=10,
        unit_runner=appended,
    )

    assert all(unit.tokens == 1000 for unit in first.units.values())
    assert all(unit.reasoning_tokens == 200 for unit in first.units.values())
    after = _progress(output_dir).units["A.md#2"]
    assert after.tokens == 1500
    assert after.reasoning_tokens == 250


def _legacy_unit_json(unit: EvalUnit) -> dict[str, Any]:
    """Bản ghi đơn vị kiểu cũ: chưa có `status`, `skipped_samples`, `tokens`, `reasoning_tokens`."""
    return {
        "title": unit.title,
        "chars": unit.char_count,
        "estimated_tokens": unit.estimated_tokens,
        "questions": {"single_hop": 4, "abstract": 1, "specific": 1},
        "llm_calls": 30,
        "seconds": 120.5,
        "completed_at": "2026-09-27T10:00:00+07:00",
    }


def test_ban_ghi_cu_khong_co_tokens_thi_tong_sau_cong_don_la_none_khong_ra_so_sai(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    unit = _units_by_key(markdown_dir)["A.md#2"]
    output_dir.mkdir()
    payload = {"units": {"A.md#2": _legacy_unit_json(unit)}, "last_failure": None}
    progress_path = output_dir / tg.PROGRESS_FILENAME
    progress_path.write_text(json.dumps(payload), encoding="utf-8")
    runner = ScriptedRunner(tokens=1000, reasoning_tokens=200, prefix="Sinh bù")

    tg.generate_testset(
        markdown_dir,
        output_dir,
        only=["A.md#2"],
        append=True,
        testset_size=10,
        unit_runner=runner,
    )

    after = _progress(output_dir).units["A.md#2"]
    assert after.tokens is None  # cộng phần mới vào tổng thiếu sẽ ra số sai
    assert after.reasoning_tokens is None
    assert after.questions == {"single_hop": 12, "abstract": 2, "specific": 2}
    assert after.llm_calls == 35
    assert after.status == "done"


def test_progress_file_cu_khong_co_cac_truong_moi_doc_thanh_done_va_khong_co_so_do(
    dirs: tuple[Path, Path], tmp_path: Path
):
    markdown_dir, _ = dirs
    units = _units_by_key(markdown_dir)
    payload = {
        "units": {key: _legacy_unit_json(unit) for key, unit in units.items()},
        "last_failure": None,
    }
    path = tmp_path / "generation_progress.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    progress = tg.load_progress(path)

    assert set(progress.units) == set(units)
    for unit_progress in progress.units.values():
        assert unit_progress.status == "done"
        assert unit_progress.skipped_samples == 0
        assert unit_progress.tokens is None
        assert unit_progress.reasoning_tokens is None


def test_progress_partial_ghi_roi_doc_lai_giu_nguyen_status_skipped_va_token(
    dirs: tuple[Path, Path], tmp_path: Path
):
    markdown_dir, _ = dirs
    unit = _units_by_key(markdown_dir)["A.md#2"]
    partial = _progress_of(
        unit, status="partial", skipped_samples=2, tokens=12345, reasoning_tokens=678
    )
    path = tmp_path / "generation_progress.json"

    tg.save_progress(path, GenerationProgress(units={"A.md#2": partial}))
    loaded = tg.load_progress(path).units["A.md#2"]

    assert loaded == partial
    assert (loaded.status, loaded.skipped_samples) == ("partial", 2)
    assert (loaded.tokens, loaded.reasoning_tokens) == (12345, 678)


def test_status_khong_hop_le_bi_tu_choi():
    unit = EvalUnit(source_document="A.md", index=1, title="Chương I", text="a" * 10)

    with pytest.raises(ValidationError):
        _progress_of(unit, status="xong")


# ==========================================================================
# `--dry-run` dùng hệ số token/ký tự ĐO ĐƯỢC khi đủ 3 đơn vị `done` có `tokens` > 0
# ==========================================================================

_MEASURED_FACTOR = 20.0


def _measured_tokens(unit: EvalUnit) -> int:
    """Token đúng bằng `hệ số x ký tự + phần cố định` (nên đo lại đúng hệ số)."""
    return round(unit.char_count * _MEASURED_FACTOR) + FIXED_TOKENS_PER_UNIT


def _measured(units: dict[str, EvalUnit], keys: list[str]) -> dict[str, UnitProgress]:
    """Các đơn vị `done` mang số đo token nhất quán với hệ số `_MEASURED_FACTOR`."""
    return {
        key: _progress_of(units[key], tokens=_measured_tokens(units[key]))
        for key in keys
    }


def _estimate(units: list[EvalUnit], factor: float | None) -> int:
    if factor is None:
        return sum(unit.estimated_tokens for unit in units)
    fixed = FIXED_TOKENS_PER_UNIT
    return sum(round(unit.char_count * factor) + fixed for unit in units)


def _summary_of(plan: str) -> str:
    return plan.splitlines()[-1]


def test_dry_run_dung_he_so_do_duoc_khi_du_3_don_vi_done_co_tokens(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    units = _units_by_key(markdown_dir)
    _save_units(output_dir, _measured(units, ["A.md#2", "B.md#2", "B.md#1"]))
    remaining = [units["A.md#1"], units["C.md#1"]]
    measured_tokens = _estimate(remaining, _MEASURED_FACTOR)
    default_tokens = _estimate(remaining, None)

    summary = _summary_of(tg.plan_generation(markdown_dir, output_dir))
    overall = tg.summarize_progress(markdown_dir, output_dir)

    assert "Đã xong 3/5 đơn vị, còn 2 đơn vị" in summary
    assert f"~{measured_tokens / 1e6:.2f}M token" in summary
    assert f"~{default_tokens / 1e6:.2f}M token" not in summary
    # Phần cố định 4.000/đơn vị đã bị trừ trước khi chia: đo lại đúng 20,00 chứ không phải cao hơn.
    assert " (hệ số đo được 20.00 token/ký tự)" in summary
    assert " (hệ số đo được 20.00 token/ký tự)" in overall


def test_dry_run_giu_he_so_5_5_khi_chua_du_3_don_vi_co_so_do(dirs: tuple[Path, Path]):
    markdown_dir, output_dir = dirs
    units = _units_by_key(markdown_dir)
    _save_units(output_dir, _measured(units, ["A.md#2", "B.md#2"]))
    remaining = [units["B.md#1"], units["A.md#1"], units["C.md#1"]]

    summary = _summary_of(tg.plan_generation(markdown_dir, output_dir))

    assert "hệ số đo được" not in summary
    assert f"~{_estimate(remaining, None) / 1e6:.2f}M token" in summary


def test_dry_run_bo_qua_ban_ghi_cu_tokens_none_tokens_0_va_don_vi_partial(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    units = _units_by_key(markdown_dir)
    progress = {
        "A.md#2": _progress_of(units["A.md#2"], tokens=None),
        "B.md#2": _progress_of(units["B.md#2"], tokens=0),
        "B.md#1": _progress_of(units["B.md#1"], tokens=500_000),
        "A.md#1": _progress_of(units["A.md#1"], status="partial", tokens=900_000),
    }
    _save_units(output_dir, progress)

    summary = _summary_of(tg.plan_generation(markdown_dir, output_dir))

    # Chỉ B.md#1 là `done` có tokens > 0: dưới ngưỡng 3 nên chưa tin hệ số đo được.
    assert "hệ số đo được" not in summary


def test_dry_run_don_vi_partial_co_tokens_khong_lam_lech_he_so_do_duoc(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    units = _units_by_key(markdown_dir)
    progress = _measured(units, ["A.md#2", "B.md#2", "B.md#1"])
    progress["A.md#1"] = _progress_of(
        units["A.md#1"], status="partial", tokens=10_000_000
    )
    _save_units(output_dir, progress)
    remaining = [units["A.md#1"], units["C.md#1"]]

    summary = _summary_of(tg.plan_generation(markdown_dir, output_dir))

    assert " (hệ số đo được 20.00 token/ký tự)" in summary
    assert f"~{_estimate(remaining, _MEASURED_FACTOR) / 1e6:.2f}M token" in summary
