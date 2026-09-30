"""Unit test logic thuần của `evaluation/testset_generator.py` (evaluation_spec.md mục 4, 4.2, 4.5, 8).

Không cần `ragas` lẫn key Groq: mọi lần sinh câu hỏi đi qua `FakeUnitRunner`, corpus là
file `.md` nhỏ dựng trong `tmp_path`. Phần chạm `ragas` thật nằm ở `test_evaluation.py`.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from production_legal_qa_rag.evaluation import testset_generator as tg
from production_legal_qa_rag.evaluation.groq_round_robin import (
    DailyQuotaExhaustedError,
)
from production_legal_qa_rag.evaluation.models import (
    GenerationProgress,
    GoldenTestCase,
    UnitFailure,
)
from production_legal_qa_rag.evaluation.unit_splitter import EvalUnit, split_directory

SINGLE = "single_hop_specific_query_synthesizer"
ABSTRACT = "multi_hop_abstract_query_synthesizer"
SPECIFIC = "multi_hop_specific_query_synthesizer"
_NAME_OF_KIND = {"single_hop": SINGLE, "abstract": ABSTRACT, "specific": SPECIFIC}


def _chapter(name: str, chars: int) -> str:
    return (
        f"## Chương {name}. TIÊU ĐỀ {name}\n\n#### Điều 1. X\n\n" + "a" * chars + "\n\n"
    )


def _write_corpus(markdown_dir: Path) -> None:
    """3 văn bản, 5 đơn vị: kích thước 7.000/8.000/9.000 (hai đơn vị bằng nhau ở 2 văn bản)."""
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


class FakeUnitRunner:
    """Sinh đúng số câu theo quota; có thể ép lỗi ở một đơn vị."""

    def __init__(
        self, fail_on: str | None = None, llm_calls: int = 7, prefix: str = "Câu hỏi"
    ) -> None:
        self.prefix = prefix
        self.fail_on = fail_on
        self.llm_calls = llm_calls
        self.calls: list[str] = []
        self.kg_paths: list[Path] = []
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
        self.calls.append(key)
        self.kg_paths.append(knowledge_graph_path)
        self.reuse_flags.append(reuse_knowledge_graph)
        if key == self.fail_on:
            raise RuntimeError("Groq hết quota: nội dung bí mật không được log")
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
        # KG chỉ được lưu khi sinh xong, giống runner thật (không ghi KG dở, mục 8).
        knowledge_graph_path.parent.mkdir(parents=True, exist_ok=True)
        knowledge_graph_path.write_text("{}", encoding="utf-8")
        return tg.UnitResult(cases=cases, llm_calls=self.llm_calls)


class DailyQuotaRunner(FakeUnitRunner):
    """Fake riêng cho ngoại lệ duy nhất phải dừng toàn bộ job."""

    def run_unit(
        self,
        unit: EvalUnit,
        quota: tg.QuestionQuota,
        knowledge_graph_path: Path,
        *,
        reuse_knowledge_graph: bool,
    ) -> tg.UnitResult:
        if tg.unit_key(unit) == self.fail_on:
            self.calls.append(tg.unit_key(unit))
            raise DailyQuotaExhaustedError("hết quota ngày")
        return super().run_unit(
            unit,
            quota,
            knowledge_graph_path,
            reuse_knowledge_graph=reuse_knowledge_graph,
        )


def _unit(document: str, index: int, chars: int) -> EvalUnit:
    return EvalUnit(
        source_document=document, index=index, title=f"Chương {index}", text="a" * chars
    )


def _case(
    document: str, kind: str, serial: int, section: str = "Chương I"
) -> GoldenTestCase:
    return GoldenTestCase(
        user_input=f"Câu {document} {kind} {serial}?",
        reference="Đáp án",
        reference_contexts=["Ngữ cảnh"],
        synthesizer_name=_NAME_OF_KIND[kind],
        source_document=document,
        source_section=section,
    )


# ==========================================================================
# allocate_questions / thứ tự chạy / --only (mục 4, 4.5)
# ==========================================================================


def test_split_question_mix_240_la_192_24_24():
    mix = tg.split_question_mix(240)

    assert (mix.single_hop, mix.abstract, mix.specific) == (192, 24, 24)


def test_allocate_questions_tong_tung_loai_dung_192_24_24():
    units = [
        _unit("A.md", i, chars)
        for i, chars in enumerate([6001, 7333, 12_345, 29_999, 8_000, 6_500], 1)
    ]

    quotas = tg.allocate_questions(units)

    assert len(quotas) == len(units)
    assert sum(q.single_hop for q in quotas.values()) == 192
    assert sum(q.abstract for q in quotas.values()) == 24
    assert sum(q.specific for q in quotas.values()) == 24
    assert sum(q.total for q in quotas.values()) == 240


def test_allocate_questions_ty_le_theo_ky_tu_va_phan_du_lon_nhat():
    # 3 đơn vị 1:1:2 chia 10 câu (mix 8/1/1): single 8 -> 2/2/4, multi-hop 1 câu -> đơn vị lớn nhất.
    units = [_unit("A.md", 1, 1000), _unit("A.md", 2, 1000), _unit("A.md", 3, 2000)]

    quotas = tg.allocate_questions(units, total=10)

    assert [quotas[tg.unit_key(u)].single_hop for u in units] == [2, 2, 4]
    assert [quotas[tg.unit_key(u)].abstract for u in units] == [0, 0, 1]
    assert [quotas[tg.unit_key(u)].specific for u in units] == [0, 0, 1]


def test_allocate_questions_hoa_phan_du_thi_lay_don_vi_dung_truoc():
    units = [_unit("A.md", 1, 1000), _unit("A.md", 2, 1000)]

    quotas = tg.allocate_questions(
        units, total=10
    )  # multi-hop 1 câu, hai đơn vị bằng nhau

    assert quotas["A.md#1"].abstract == 1
    assert quotas["A.md#2"].abstract == 0


def test_allocate_questions_khong_co_don_vi_thi_rong():
    assert tg.allocate_questions([]) == {}


def test_order_units_nho_truoc_hoa_thi_theo_ten_van_ban_roi_so_thu_tu():
    units = [
        _unit("B.md", 1, 8000),
        _unit("A.md", 2, 7000),
        _unit("A.md", 1, 9000),
        _unit("A.md", 3, 7000),
        _unit("B.md", 2, 7000),
    ]

    ordered = tg.order_units(units)

    assert [tg.unit_key(u) for u in ordered] == [
        "A.md#2",
        "A.md#3",
        "B.md#2",
        "B.md#1",
        "A.md#1",
    ]


def test_select_units_rong_la_tat_ca_va_chap_nhan_ten_khong_duoi_md():
    units = [_unit("A.md", 1, 10), _unit("A.md", 2, 10), _unit("B.md", 1, 10)]

    assert tg.select_units(units, []) == units
    assert [tg.unit_key(u) for u in tg.select_units(units, ["A"])] == [
        "A.md#1",
        "A.md#2",
    ]
    assert [tg.unit_key(u) for u in tg.select_units(units, ["B.md#1", "A.md#2"])] == [
        "A.md#2",
        "B.md#1",
    ]


def test_select_units_ten_khong_ton_tai_liet_ke_ten_hop_le():
    units = [_unit("A.md", 1, 10)]

    with pytest.raises(tg.EvalInputError, match="Hợp lệ: A.md"):
        tg.select_units(units, ["Z"])


def test_select_units_so_thu_tu_khong_ton_tai():
    units = [_unit("A.md", 1, 10), _unit("A.md", 2, 10)]

    with pytest.raises(tg.EvalInputError, match="chỉ có đơn vị số 1-2"):
        tg.select_units(units, ["A.md#9"])


# ==========================================================================
# generate_testset: chạy, checkpoint, resume (mục 4.5)
# ==========================================================================


def test_generate_chay_theo_thu_tu_nho_den_lon_va_ghi_raw_kg_progress(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    runner = FakeUnitRunner()

    report = tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)

    # 7.000: A#2, B#2 (hoà -> tên văn bản); 8.000: B#1; 9.000: A#1, C#1.
    assert runner.calls == ["A.md#2", "B.md#2", "B.md#1", "A.md#1", "C.md#1"]
    assert report.generated_units == runner.calls
    assert report.new_questions == 240
    assert runner.kg_paths[0] == output_dir / "knowledge_graph" / "A__02.json"
    assert all(path.exists() for path in runner.kg_paths)

    rows = json.loads((output_dir / tg.RAW_TESTSET_FILENAME).read_text("utf-8"))
    assert len(rows) == 240
    assert {r["source_document"] for r in rows} == {"A.md", "B.md", "C.md"}
    assert {r["source_section"] for r in rows if r["source_document"] == "A.md"} == {
        "Chương I. TIÊU ĐỀ I",
        "Chương II. TIÊU ĐỀ II",
    }
    assert Counter(r["synthesizer_name"] for r in rows) == {
        SINGLE: 192,
        ABSTRACT: 24,
        SPECIFIC: 24,
    }

    progress = tg.load_progress(output_dir / tg.PROGRESS_FILENAME)
    assert set(progress.units) == set(runner.calls)
    expected_units = {tg.unit_key(u): u for u in split_directory(markdown_dir)}
    quotas = tg.allocate_questions(list(expected_units.values()))
    done = progress.units["A.md#2"]
    assert done.chars == expected_units["A.md#2"].char_count
    assert done.title == "Chương II. TIÊU ĐỀ II"
    assert done.llm_calls == 7
    assert done.questions == quotas["A.md#2"].model_dump()
    assert progress.last_failure is None
    assert not list(output_dir.glob("*.tmp"))


def test_generate_lan_hai_bo_qua_don_vi_da_xong_ke_ca_khi_xoa_het_dong_trong_raw(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    tg.generate_testset(markdown_dir, output_dir, unit_runner=FakeUnitRunner())
    # Người dùng xoá sạch dòng của một đơn vị khi đọc lướt: vẫn KHÔNG được sinh lại.
    raw_path = output_dir / tg.RAW_TESTSET_FILENAME
    rows = json.loads(raw_path.read_text("utf-8"))
    kept = [r for r in rows if r["source_document"] != "C.md"]
    raw_path.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    second = FakeUnitRunner()

    report = tg.generate_testset(markdown_dir, output_dir, unit_runner=second)

    assert second.calls == []
    assert report.generated_units == []
    assert len(report.skipped_units) == 5
    assert json.loads(raw_path.read_text("utf-8")) == kept  # không đụng vào raw


def test_generate_khong_dung_runner_khi_khong_con_gi_de_chay(
    dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
):
    markdown_dir, output_dir = dirs
    tg.generate_testset(markdown_dir, output_dir, unit_runner=FakeUnitRunner())

    def _khong_duoc_goi(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("không được dựng runner (cần key Groq) khi đã xong hết")

    monkeypatch.setattr(tg, "build_unit_runner", _khong_duoc_goi)

    report = tg.generate_testset(markdown_dir, output_dir)

    assert report.generated_units == []


def test_generate_loi_khong_quota_bo_qua_unit_va_chay_tiep(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    runner = FakeUnitRunner(fail_on="B.md#1")

    report = tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)

    assert runner.calls == ["A.md#2", "B.md#2", "B.md#1", "A.md#1", "C.md#1"]
    assert report.skipped_units == ["B.md#1"]
    progress = tg.load_progress(output_dir / tg.PROGRESS_FILENAME)
    assert progress.units["B.md#1"].status == "skipped"
    assert progress.units["B.md#1"].skipped_stage == "knowledge_graph"
    assert progress.units["B.md#1"].error_type == "RuntimeError"
    assert progress.last_failure is None
    rows = json.loads((output_dir / tg.RAW_TESTSET_FILENAME).read_text("utf-8"))
    assert {r["source_document"] for r in rows} == {"A.md", "B.md", "C.md"}
    assert not any(
        row["source_document"] == "B.md"
        and row["source_section"] == "Chương I. TIÊU ĐỀ I"
        for row in rows
    )


def test_generate_het_quota_ngay_dung_ngay_va_ghi_last_failure(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    runner = DailyQuotaRunner(fail_on="B.md#1")

    with pytest.raises(tg.UnitGenerationError) as excinfo:
        tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)

    assert excinfo.value.unit_key == "B.md#1"
    assert runner.calls == ["A.md#2", "B.md#2", "B.md#1"]
    progress = tg.load_progress(output_dir / tg.PROGRESS_FILENAME)
    assert progress.last_failure is not None
    assert progress.last_failure.unit == "B.md#1"


def test_generate_chi_chay_lai_unit_skipped_khi_retry_skipped_va_only(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    tg.generate_testset(
        markdown_dir, output_dir, unit_runner=FakeUnitRunner(fail_on="B.md#1")
    )
    resumed = FakeUnitRunner()

    tg.generate_testset(
        markdown_dir,
        output_dir,
        only=["B.md#1"],
        retry_skipped=True,
        unit_runner=resumed,
    )

    assert resumed.calls == ["B.md#1"]
    progress = tg.load_progress(output_dir / tg.PROGRESS_FILENAME)
    assert len(progress.units) == 5
    assert progress.last_failure is None
    assert progress.units["B.md#1"].status == "done"


def test_generate_loi_khong_quota_o_unit_dau_tien_ghi_skipped_khong_last_failure(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs

    tg.generate_testset(
        markdown_dir, output_dir, unit_runner=FakeUnitRunner(fail_on="A.md#2")
    )

    progress = tg.load_progress(output_dir / tg.PROGRESS_FILENAME)
    assert progress.units["A.md#2"].status == "skipped"
    assert progress.last_failure is None
    assert (output_dir / tg.RAW_TESTSET_FILENAME).exists()


def test_generate_chet_giua_hai_buoc_don_vi_co_dong_trong_raw_coi_la_da_xong(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    output_dir.mkdir()
    # Raw đã có 2 dòng của A.md#2 nhưng chưa có progress (chết giữa hai lần ghi).
    rows = [
        _case("A.md", "single_hop", 1, "Chương II. TIÊU ĐỀ II").model_dump(),
        _case("A.md", "abstract", 2, "Chương II. TIÊU ĐỀ II").model_dump(),
    ]
    (output_dir / tg.RAW_TESTSET_FILENAME).write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )
    runner = FakeUnitRunner()

    report = tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)

    assert "A.md#2" not in runner.calls
    assert "A.md#2" in report.skipped_units
    progress = tg.load_progress(output_dir / tg.PROGRESS_FILENAME)
    assert progress.units["A.md#2"].questions == {
        "single_hop": 1,
        "abstract": 1,
        "specific": 0,
    }


def test_generate_chars_lech_so_voi_nguon_thi_dung_voi_loi_ro(dirs: tuple[Path, Path]):
    markdown_dir, output_dir = dirs
    tg.generate_testset(markdown_dir, output_dir, unit_runner=FakeUnitRunner())
    progress_path = output_dir / tg.PROGRESS_FILENAME
    progress = tg.load_progress(progress_path)
    progress.units["B.md#1"].chars += 1
    tg.save_progress(progress_path, progress)
    runner = FakeUnitRunner()

    with pytest.raises(tg.EvalInputError, match=r"B\.md#1"):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)

    assert runner.calls == []


def test_generate_don_vi_trong_progress_khong_con_trong_nguon_thi_dung(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    tg.generate_testset(markdown_dir, output_dir, unit_runner=FakeUnitRunner())
    (markdown_dir / "C.md").unlink()

    with pytest.raises(tg.EvalInputError, match=r"C\.md#1.*không còn tồn tại"):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=FakeUnitRunner())


@pytest.mark.parametrize("content", ["{không phải json", '{"units": {"x": 1}}'])
def test_generate_progress_hong_thi_raise_khong_coi_nhu_chua_lam_gi(
    dirs: tuple[Path, Path], content: str
):
    markdown_dir, output_dir = dirs
    output_dir.mkdir()
    (output_dir / tg.PROGRESS_FILENAME).write_text(content, encoding="utf-8")
    runner = FakeUnitRunner()

    with pytest.raises(tg.EvalInputError, match="hỏng"):
        tg.generate_testset(markdown_dir, output_dir, unit_runner=runner)

    assert runner.calls == []


def test_generate_only_chi_chay_don_vi_da_chon_va_quota_tinh_tren_ca_corpus(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    runner = FakeUnitRunner()
    full_quota = tg.allocate_questions(split_directory(markdown_dir))

    tg.generate_testset(markdown_dir, output_dir, only=["B.md#2"], unit_runner=runner)

    assert runner.calls == ["B.md#2"]
    progress = tg.load_progress(output_dir / tg.PROGRESS_FILENAME)
    assert (
        sum(progress.units["B.md#2"].questions.values()) == full_quota["B.md#2"].total
    )


def test_generate_only_ten_sai_bao_loi_truoc_khi_dung_runner(dirs: tuple[Path, Path]):
    markdown_dir, output_dir = dirs
    runner = FakeUnitRunner()

    with pytest.raises(tg.EvalInputError, match="Hợp lệ"):
        tg.generate_testset(markdown_dir, output_dir, only=["Z.md"], unit_runner=runner)

    assert runner.calls == []


def test_generate_append_chay_lai_don_vi_da_xong_noi_them_va_cong_don(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    tg.generate_testset(markdown_dir, output_dir, unit_runner=FakeUnitRunner())
    raw_path = output_dir / tg.RAW_TESTSET_FILENAME
    before = json.loads(raw_path.read_text("utf-8"))
    before_progress = tg.load_progress(output_dir / tg.PROGRESS_FILENAME)
    runner = FakeUnitRunner(prefix="Câu hỏi sinh bù")

    tg.generate_testset(
        markdown_dir,
        output_dir,
        only=["A.md#2"],
        append=True,
        reuse_knowledge_graph=True,
        testset_size=10,
        unit_runner=runner,
    )

    assert runner.calls == ["A.md#2"]
    assert runner.reuse_flags == [True]
    after = json.loads(raw_path.read_text("utf-8"))
    assert after[: len(before)] == before  # không ghi đè/sắp xếp lại dòng cũ
    assert len(after) == len(before) + 10
    new_progress = tg.load_progress(output_dir / tg.PROGRESS_FILENAME)
    old, new = before_progress.units["A.md#2"], new_progress.units["A.md#2"]
    assert sum(new.questions.values()) == sum(old.questions.values()) + 10
    assert new.llm_calls == old.llm_calls + 7


def test_generate_skipped_khong_tu_chay_lai_con_done_chi_reuse_khi_co_co(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    first = FakeUnitRunner(fail_on="B.md#2")
    tg.generate_testset(markdown_dir, output_dir, unit_runner=first)
    assert first.reuse_flags == [True] * len(first.calls)

    second = FakeUnitRunner()
    tg.generate_testset(markdown_dir, output_dir, unit_runner=second)
    assert second.calls == []  # skipped không tự chạy lại

    appended = FakeUnitRunner()
    tg.generate_testset(
        markdown_dir, output_dir, only=["A.md#2"], append=True, unit_runner=appended
    )
    with_flag = FakeUnitRunner()
    tg.generate_testset(
        markdown_dir,
        output_dir,
        only=["A.md#2"],
        append=True,
        reuse_knowledge_graph=True,
        unit_runner=with_flag,
    )
    assert appended.reuse_flags == [False]  # đơn vị đã xong, không cờ -> dựng lại
    assert with_flag.reuse_flags == [True]


def test_append_khong_kem_only_bao_loi_dau_vao_truoc_khi_dung_runner(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    runner = FakeUnitRunner()

    with pytest.raises(tg.EvalInputError, match="--only"):
        tg.generate_testset(markdown_dir, output_dir, append=True, unit_runner=runner)
    with pytest.raises(tg.EvalInputError, match="--only"):
        tg.plan_generation(markdown_dir, output_dir, append=True)
    with pytest.raises(tg.EvalInputError, match="--only"):
        tg.generate_testset(
            markdown_dir, output_dir, retry_skipped=True, unit_runner=runner
        )

    assert runner.calls == []
    assert not output_dir.exists()


def test_cli_append_khong_kem_only_thoat_ma_2(dirs: tuple[Path, Path]):
    from tools import generate_testset

    markdown_dir, output_dir = dirs

    result = CliRunner().invoke(
        generate_testset.app,
        [
            "generate",
            "--append",
            "--markdown-dir",
            str(markdown_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 2
    assert "--only" in result.output


def test_cli_dry_run_append_khong_kem_only_cung_thoat_ma_2_con_co_only_thi_in_ke_hoach(
    dirs: tuple[Path, Path],
):
    from tools import generate_testset

    markdown_dir, output_dir = dirs
    base = ["--markdown-dir", str(markdown_dir), "--output-dir", str(output_dir)]

    without_only = CliRunner().invoke(
        generate_testset.app, ["generate", "--dry-run", "--append", *base]
    )
    with_only = CliRunner().invoke(
        generate_testset.app,
        ["generate", "--dry-run", "--append", "--only", "A.md#2", *base],
    )

    assert without_only.exit_code == 2
    assert "--only" in without_only.output
    assert "Lỗi đầu vào" in without_only.output
    assert with_only.exit_code == 0, with_only.output
    assert "A.md#2" in with_only.output
    assert not output_dir.exists()  # dry-run không tạo gì


def test_cli_retry_skipped_khong_kem_only_thoat_ma_2(dirs: tuple[Path, Path]):
    from tools import generate_testset

    markdown_dir, output_dir = dirs
    result = CliRunner().invoke(
        generate_testset.app,
        [
            "generate",
            "--retry-skipped",
            "--markdown-dir",
            str(markdown_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 2
    assert "--only" in result.output


def test_append_khong_kem_only_bi_chan_du_progress_da_co_khong_chay_lai_50_don_vi(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    tg.generate_testset(markdown_dir, output_dir, unit_runner=FakeUnitRunner())
    raw_before = (output_dir / tg.RAW_TESTSET_FILENAME).read_bytes()
    runner = FakeUnitRunner()

    with pytest.raises(tg.EvalInputError, match="--only"):
        tg.generate_testset(
            markdown_dir,
            output_dir,
            append=True,
            reuse_knowledge_graph=True,
            unit_runner=runner,
        )

    assert runner.calls == []
    assert (output_dir / tg.RAW_TESTSET_FILENAME).read_bytes() == raw_before


def test_thieu_thu_muc_markdown_la_loi_dau_vao_thoat_ma_2(tmp_path: Path):
    from tools import generate_testset

    missing = tmp_path / "khong-co"
    with pytest.raises(tg.EvalInputError, match="Không đọc được"):
        tg.plan_generation(missing, tmp_path / "eval")

    result = CliRunner().invoke(
        generate_testset.app,
        ["generate", "--markdown-dir", str(missing), "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 2
    assert "Lỗi đầu vào" in result.output


def test_append_raw_cases_bo_cau_trung_user_input_va_giu_nguyen_dong_cu(tmp_path: Path):
    raw_path = tmp_path / "raw.json"
    old_row = {
        **_case("A.md", "single_hop", 1).model_dump(),
        "ghi_chu": "người dùng sửa tay",
    }
    raw_path.write_text(json.dumps([old_row], ensure_ascii=False), encoding="utf-8")

    added = tg.append_raw_cases(
        raw_path, [_case("A.md", "single_hop", 1), _case("A.md", "single_hop", 2)]
    )

    assert [c.user_input for c in added] == ["Câu A.md single_hop 2?"]
    rows = json.loads(raw_path.read_text("utf-8"))
    assert rows[0] == old_row
    assert len(rows) == 2


def test_save_progress_nguyen_tu_loi_khi_doi_ten_thi_file_cu_con_nguyen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / tg.PROGRESS_FILENAME
    tg.save_progress(path, GenerationProgress())
    original = path.read_text("utf-8")
    progress = GenerationProgress()
    progress.last_failure = UnitFailure(
        unit="A.md#1", error="X", at=datetime.now().astimezone()
    )

    def _replace_loi(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("mất điện")

    monkeypatch.setattr(os, "replace", _replace_loi)

    with pytest.raises(OSError):
        tg.save_progress(path, progress)

    assert path.read_text("utf-8") == original


def test_khoa_va_duong_dan_kg_dung_dinh_dang_spec():
    unit = _unit("Luật bảo hiểm y tế.md", 6, 10)

    assert tg.unit_key(unit) == "Luật bảo hiểm y tế.md#6"
    assert (
        tg.knowledge_graph_path(Path("data/eval"), unit)
        == Path("data/eval") / "knowledge_graph" / "Luật bảo hiểm y tế__06.json"
    )


def test_generate_van_ban_rong_bao_loi_truoc_khi_dung_runner(tmp_path: Path):
    markdown_dir = tmp_path / "md"
    markdown_dir.mkdir()
    (markdown_dir / "rong.md").write_text("  \n", encoding="utf-8")
    runner = FakeUnitRunner()

    with pytest.raises(tg.EvalInputError, match="trống"):
        tg.generate_testset(markdown_dir, tmp_path / "eval", unit_runner=runner)

    assert runner.calls == []


# ==========================================================================
# --dry-run (không LLM, không key) (mục 4.5)
# ==========================================================================


def test_plan_generation_in_thu_tu_trang_thai_va_tom_tat_khong_can_key(
    dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
):
    markdown_dir, output_dir = dirs
    tg.generate_testset(
        markdown_dir, output_dir, unit_runner=FakeUnitRunner(fail_on="B.md#1")
    )

    def _khong_duoc_goi(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("dry-run không được dựng runner/đọc key Groq")

    monkeypatch.setattr(tg, "build_unit_runner", _khong_duoc_goi)
    for env_name in ("GROQ_API_KEY_1", "GROQ_API_KEY_2"):
        monkeypatch.delenv(env_name, raising=False)

    plan = tg.plan_generation(markdown_dir, output_dir)

    lines = plan.splitlines()
    keys_in_order = [line.split("|")[2].strip() for line in lines[2:7]]
    assert keys_in_order == ["A.md#2", "B.md#2", "B.md#1", "A.md#1", "C.md#1"]
    assert "xong" in lines[2]
    assert "bỏ qua knowledge_graph" in lines[4]
    assert "CHẠY TIẾP THEO" not in plan
    assert "Đã xong 4/5 đơn vị, còn 0 đơn vị; bỏ qua 1 unit" in plan
    assert "ngày" in plan


def test_plan_generation_khong_ghi_file(dirs: tuple[Path, Path]):
    markdown_dir, output_dir = dirs

    plan = tg.plan_generation(markdown_dir, output_dir)

    assert not output_dir.exists()
    assert "Đã xong 0/5 đơn vị" in plan
    assert plan.count("chưa") == 5


def test_plan_generation_append_coi_don_vi_da_xong_la_se_chay_lai(
    dirs: tuple[Path, Path],
):
    markdown_dir, output_dir = dirs
    tg.generate_testset(markdown_dir, output_dir, unit_runner=FakeUnitRunner())

    plan = tg.plan_generation(markdown_dir, output_dir, only=["A.md#2"], append=True)

    assert "CHẠY TIẾP THEO" in plan
    assert "còn 1 đơn vị" in plan


# ==========================================================================
# finalize (mục 4.2)
# ==========================================================================


def _stratified_raw() -> list[GoldenTestCase]:
    """3 văn bản x (single/abstract/specific) với kích thước nhóm khác nhau, tổng 240."""
    shape = {
        "A.md": {"single_hop": 90, "abstract": 12, "specific": 12},
        "B.md": {"single_hop": 60, "abstract": 6, "specific": 6},
        "C.md": {"single_hop": 42, "abstract": 6, "specific": 6},
    }
    cases = []
    serial = 0
    for document, kinds in shape.items():
        for kind, count in kinds.items():
            for _ in range(count):
                serial += 1
                cases.append(_case(document, kind, serial))
    return cases


def test_finalize_testset_cat_dung_180_giu_ty_le_theo_van_ban_va_loai_cau():
    raw = _stratified_raw()
    assert len(raw) == 240

    chosen = tg.finalize_testset(raw)

    assert len(chosen) == tg.TARGET_SIZE == 180
    by_group = Counter((c.source_document, c.synthesizer_name) for c in chosen)
    raw_group = Counter((c.source_document, c.synthesizer_name) for c in raw)
    for group, count in raw_group.items():
        # tỷ lệ 180/240 = 0,75, lệch tối đa 1 câu mỗi nhóm do làm tròn
        assert abs(by_group[group] - count * 0.75) <= 1
    assert by_group[("A.md", ABSTRACT)] >= 8  # không dồn hết vào vài văn bản đầu file


def test_finalize_testset_tat_dinh_va_giu_thu_tu_file():
    raw = _stratified_raw()

    first = tg.finalize_testset(raw)
    second = tg.finalize_testset(list(raw))

    assert first == second
    positions = [raw.index(c) for c in first]
    assert positions == sorted(positions)


def test_finalize_testset_du_dung_target_thi_giu_nguyen():
    raw = _stratified_raw()[:180]

    assert tg.finalize_testset(raw) == raw


def test_finalize_testset_thieu_thi_bao_loi_kem_goi_y_sinh_bu():
    raw = _stratified_raw()[:170]

    with pytest.raises(tg.EvalInputError) as excinfo:
        tg.finalize_testset(raw)

    message = str(excinfo.value)
    assert "thiếu 10 câu" in message
    assert "--reuse-knowledge-graph --append --testset-size 13" in message


def test_finalize_golden_testset_ghi_file_dung_180(tmp_path: Path):
    raw_path = tmp_path / tg.RAW_TESTSET_FILENAME
    raw_path.write_text(
        json.dumps([c.model_dump() for c in _stratified_raw()], ensure_ascii=False),
        encoding="utf-8",
    )

    cases = tg.finalize_golden_testset(tmp_path)

    written = json.loads((tmp_path / tg.GOLDEN_TESTSET_FILENAME).read_text("utf-8"))
    assert len(cases) == len(written) == 180
    assert set(written[0]) >= {
        "user_input",
        "reference",
        "reference_contexts",
        "synthesizer_name",
        "source_document",
        "source_section",
    }


def test_finalize_golden_testset_thieu_thi_khong_ghi_file(tmp_path: Path):
    (tmp_path / tg.RAW_TESTSET_FILENAME).write_text(
        json.dumps([c.model_dump() for c in _stratified_raw()[:50]]), encoding="utf-8"
    )

    with pytest.raises(tg.EvalInputError, match="thiếu 130"):
        tg.finalize_golden_testset(tmp_path)

    assert not (tmp_path / tg.GOLDEN_TESTSET_FILENAME).exists()


def test_finalize_golden_testset_raw_khong_ton_tai(tmp_path: Path):
    with pytest.raises(tg.EvalInputError, match="generate"):
        tg.finalize_golden_testset(tmp_path)


@pytest.mark.parametrize(
    ("bad_row", "expected"),
    [
        (
            {"user_input": "  ", "reference": "b", "reference_contexts": ["c"]},
            "user_input",
        ),
        (
            {"user_input": "a", "reference": "b", "reference_contexts": []},
            "reference_contexts",
        ),
        ({"user_input": "a", "reference_contexts": ["c"]}, "không hợp lệ"),
    ],
)
def test_finalize_golden_testset_dong_rong_hoac_thieu_thi_chi_ra_vi_tri(
    tmp_path: Path, bad_row: dict[str, Any], expected: str
):
    rows = [_case("A.md", "single_hop", 1).model_dump(), bad_row]
    (tmp_path / tg.RAW_TESTSET_FILENAME).write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(tg.EvalInputError, match=rf"dòng 2.*{expected}"):
        tg.finalize_golden_testset(tmp_path, target=1)


# ==========================================================================
# CLI mỏng tools/generate_testset.py (mục 7)
# ==========================================================================


def test_cli_dry_run_khong_can_key_va_in_ke_hoach(dirs: tuple[Path, Path]):
    from tools import generate_testset

    markdown_dir, output_dir = dirs

    result = CliRunner().invoke(
        generate_testset.app,
        [
            "generate",
            "--dry-run",
            "--markdown-dir",
            str(markdown_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "CHẠY TIẾP THEO" in result.output
    assert "Đã xong 0/5 đơn vị" in result.output
    assert not output_dir.exists()


def test_cli_only_lap_lai_duoc_va_truyen_dung_tham_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from tools import generate_testset

    captured: dict[str, Any] = {}

    def _fake_generate(markdown_dir: Path, output_dir: Path, **kwargs: Any) -> Any:
        captured.update(kwargs, markdown_dir=markdown_dir, output_dir=output_dir)
        return tg.GenerationReport(
            generated_units=["A.md#1"], skipped_units=[], new_questions=3
        )

    monkeypatch.setattr(generate_testset, "generate_testset", _fake_generate)
    monkeypatch.setattr(generate_testset, "summarize_progress", lambda *_a: "TÓM TẮT")

    result = CliRunner().invoke(
        generate_testset.app,
        [
            "generate",
            "--only",
            "A.md#1",
            "--only",
            "B",
            "--reuse-knowledge-graph",
            "--append",
            "--retry-skipped",
            "--testset-size",
            "12",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert list(captured["only"]) == ["A.md#1", "B"]
    assert captured["reuse_knowledge_graph"] is True
    assert captured["append"] is True
    assert captured["retry_skipped"] is True
    assert captured["testset_size"] == 12
    assert captured["output_dir"] == tmp_path
    assert "+3 câu" in result.output
    assert "TÓM TẮT" in result.output


def test_cli_don_vi_loi_thoat_ma_khac_0_va_in_tom_tat(
    monkeypatch: pytest.MonkeyPatch,
):
    from tools import generate_testset

    def _fake_generate(*_args: Any, **_kwargs: Any) -> Any:
        raise tg.UnitGenerationError("A.md#1", "RateLimitError")

    monkeypatch.setattr(generate_testset, "generate_testset", _fake_generate)
    monkeypatch.setattr(generate_testset, "summarize_progress", lambda *_a: "TÓM TẮT")

    result = CliRunner().invoke(generate_testset.app, ["generate"])

    assert result.exit_code == 1
    assert "A.md#1" in result.output
    assert "TÓM TẮT" in result.output


def test_cli_loi_dau_vao_thoat_ma_2(dirs: tuple[Path, Path]):
    from tools import generate_testset

    markdown_dir, output_dir = dirs

    result = CliRunner().invoke(
        generate_testset.app,
        [
            "generate",
            "--only",
            "khong-co",
            "--markdown-dir",
            str(markdown_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 2
    assert "Hợp lệ" in result.output


def test_cli_finalize_thieu_cau_thoat_ma_2_thanh_cong_thi_in_so_luong(tmp_path: Path):
    from tools import generate_testset

    raw_path = tmp_path / tg.RAW_TESTSET_FILENAME
    raw_path.write_text(
        json.dumps([c.model_dump() for c in _stratified_raw()[:10]]), encoding="utf-8"
    )
    short = CliRunner().invoke(
        generate_testset.app, ["finalize", "--output-dir", str(tmp_path)]
    )
    raw_path.write_text(
        json.dumps([c.model_dump() for c in _stratified_raw()]), encoding="utf-8"
    )
    ok = CliRunner().invoke(
        generate_testset.app, ["finalize", "--output-dir", str(tmp_path)]
    )

    assert short.exit_code == 2
    assert ok.exit_code == 0, ok.output
    assert "180/180" in ok.output
