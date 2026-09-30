"""Test điều phối dịch trên raw, model dữ liệu, `--dry-run`, CLI và `finalize` (evaluation_spec.md mục 12.5-12.6).

Bổ sung cho `test_evaluation_translation.py` (hàm thuần + client HTTP): ở đây kiểm
`translate_testcase`, `translate_raw` (resume, ghi nguyên tử sau mỗi mẫu, dừng sau 3 lỗi
liên tiếp, `--retry-flagged`, mã thoát), `plan_translation`, tương thích ngược của
`GoldenTestCase`, lệnh `translate`/`finalize` qua Typer `CliRunner`. Mạng được thay bằng
`FakeTranslator` hoặc module `requests` giả trong `sys.modules`; không cần `ragas`, không
đọc `.env` thật (cwd luôn là thư mục tạm).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from production_legal_qa_rag.evaluation import testset_generator as tg
from production_legal_qa_rag.evaluation import translation as tr
from production_legal_qa_rag.evaluation.models import GoldenTestCase

URL = "https://script.example.invalid/macros/s/SECRET-DEPLOY-ID/exec"
KEY = "s3cr3t-key-value"
REPO_ROOT = Path(__file__).resolve().parents[1]

VI_CONTEXT = (
    "Điều 5. Quyền và nghĩa vụ của người sử dụng lao động\n"
    "1. Người sử dụng lao động có quyền tuyển dụng, bố trí, quản lý lao động."
)
EN_CONTEXT = (
    "Article 5. Rights and obligations of the employer\n"
    "1. The employer has the right to recruit, assign and manage employees."
)
VI_CONTEXT_FROM_EN = (
    "Điều 5. Quyền và nghĩa vụ của người sử dụng lao động\n"
    "1. Người sử dụng lao động có quyền tuyển dụng, bố trí và quản lý người lao động."
)


def _en_q(i: int) -> str:
    return f"What must employer {i} do under Article {i}?"


def _vi_q(i: int) -> str:
    return f"Người sử dụng lao động {i} phải làm gì theo Điều {i}?"


def _en_a(i: int) -> str:
    return f"Employer {i} must pay wages on time under Clause 2 of Article {i}."


def _vi_a(i: int) -> str:
    return f"Người sử dụng lao động {i} phải trả lương đúng hạn theo Khoản 2 Điều {i}."


def _pairs(count: int) -> dict[str, str | Exception]:
    pairs: dict[str, str | Exception] = {}
    for i in range(count):
        pairs[_en_q(i)] = _vi_q(i)
        pairs[_en_a(i)] = _vi_a(i)
    return pairs


class FakeTranslator:
    """`TextTranslator` giả: tra từ điển; giá trị `Exception` thì ném; thiếu khoá là lỗi test."""

    def __init__(self, responses: Mapping[str, str | Exception]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def translate(self, text: str) -> str:
        self.calls.append(text)
        response = self._responses[text]
        if isinstance(response, Exception):
            raise response
        return response


def _row(
    user_input: str,
    reference: str,
    contexts: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "user_input": user_input,
        "reference": reference,
        "reference_contexts": contexts if contexts is not None else [VI_CONTEXT],
        **extra,
    }


def _en_row(i: int, **extra: Any) -> dict[str, Any]:
    return _row(_en_q(i), _en_a(i), **extra)


def _vi_row(i: int, **extra: Any) -> dict[str, Any]:
    return _row(_vi_q(i), _vi_a(i), **extra)


def _case(**fields: Any) -> GoldenTestCase:
    return GoldenTestCase.model_validate(_row(_vi_q(0), _vi_a(0)) | fields)


def _write_raw(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / tg.RAW_TESTSET_FILENAME
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _read_raw(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / tg.RAW_TESTSET_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def out(tmp_path: Path) -> Path:
    return tmp_path / "eval"


# ---------------------------------------------------------------------------
# translate_testcase (12.4, 12.5)
# ---------------------------------------------------------------------------


def test_translate_testcase_khong_co_truong_tieng_anh_tra_ve_chinh_no():
    case = _case()

    assert tr.translate_testcase(case, FakeTranslator({}), {}) is case


def test_translate_testcase_chi_user_input_tieng_anh():
    case = _case(user_input=_en_q(1))
    translator = FakeTranslator(_pairs(2))

    result = tr.translate_testcase(case, translator, {})

    assert result.user_input == _vi_q(1)
    assert result.reference == _vi_a(0)
    assert result.reference_contexts == [VI_CONTEXT]
    assert result.original_en == {"user_input": _en_q(1)}
    assert result.translation_review is None
    assert translator.calls == [_en_q(1)]
    assert case.user_input == _en_q(1)  # bản vào không bị sửa tại chỗ


def test_translate_testcase_user_input_va_reference_deu_tieng_anh():
    case = _case(user_input=_en_q(1), reference=_en_a(1))

    result = tr.translate_testcase(case, FakeTranslator(_pairs(2)), {})

    assert (result.user_input, result.reference) == (_vi_q(1), _vi_a(1))
    assert result.original_en == {"user_input": _en_q(1), "reference": _en_a(1)}
    assert result.translation_review is None


def test_translate_testcase_reference_contexts_dich_va_gan_co_contexts_translated():
    case = _case(reference_contexts=[VI_CONTEXT, EN_CONTEXT])
    translator = FakeTranslator({EN_CONTEXT: VI_CONTEXT_FROM_EN})

    result = tr.translate_testcase(case, translator, {})

    assert result.reference_contexts == [VI_CONTEXT, VI_CONTEXT_FROM_EN]
    assert result.original_en == {"reference_contexts": [VI_CONTEXT, EN_CONTEXT]}
    assert result.translation_review == "contexts_translated"
    assert translator.calls == [EN_CONTEXT]


def test_translate_testcase_co_nang_thang_contexts_translated():
    mismatch = _vi_q(1).replace("Điều 1", "Điều 99")
    translator = FakeTranslator({_en_q(1): mismatch, EN_CONTEXT: VI_CONTEXT_FROM_EN})
    case = _case(user_input=_en_q(1), reference_contexts=[EN_CONTEXT])

    result = tr.translate_testcase(case, translator, {})

    assert result.translation_review == "citation_mismatch"
    assert result.user_input == _en_q(1)  # giữ bản gốc
    assert result.reference_contexts == [VI_CONTEXT_FROM_EN]  # vẫn đã ghi bản dịch
    assert result.original_en == {"reference_contexts": [EN_CONTEXT]}


def test_translate_testcase_translate_error_de_co_truoc_do():
    mismatch = _vi_q(1).replace("Điều 1", "Điều 99")
    translator = FakeTranslator({_en_q(1): mismatch, _en_a(1): tr.TranslateError("x")})
    case = _case(user_input=_en_q(1), reference=_en_a(1))

    result = tr.translate_testcase(case, translator, {})

    assert result.translation_review == "translate_error"  # đè citation_mismatch
    assert (result.user_input, result.reference) == (_en_q(1), _en_a(1))


def test_translate_testcase_retry_flagged_contexts_da_dich_truoc_do_gan_contexts_translated():
    case = _case(
        user_input=_en_q(1),
        reference_contexts=[VI_CONTEXT_FROM_EN],
        original_en={"reference_contexts": [EN_CONTEXT]},
        translation_review="citation_mismatch",
    )

    result = tr.translate_testcase(case, FakeTranslator({_en_q(1): _vi_q(1)}), {})

    assert result.user_input == _vi_q(1)
    assert result.original_en == {
        "reference_contexts": [EN_CONTEXT],
        "user_input": _en_q(1),
    }
    assert result.translation_review == "contexts_translated"


def test_translate_testcase_chi_giu_ly_do_co_dau_tien():
    mismatch = _vi_q(1).replace("Điều 1", "Điều 99")
    translator = FakeTranslator({_en_q(1): mismatch, _en_a(1): _en_a(1)})
    case = _case(user_input=_en_q(1), reference=_en_a(1))

    result = tr.translate_testcase(case, translator, {})

    assert result.translation_review == "citation_mismatch"  # không phải still_english
    assert (result.user_input, result.reference) == (_en_q(1), _en_a(1))


def test_translate_testcase_truong_bi_tu_choi_giu_goc_va_original_en_la_none():
    mismatch = _vi_q(1).replace("Điều 1", "Điều 99")
    case = _case(user_input=_en_q(1))

    result = tr.translate_testcase(case, FakeTranslator({_en_q(1): mismatch}), {})

    assert result.user_input == _en_q(1)
    assert result.original_en is None  # không phải {}
    assert result.translation_review == "citation_mismatch"


def test_translate_testcase_dong_qua_dai_la_too_long_khong_goi_mang():
    translator = FakeTranslator({})
    case = _case(user_input="word " * 1000)

    result = tr.translate_testcase(case, translator, {})

    assert result.translation_review == "too_long"
    assert result.user_input == "word " * 1000
    assert translator.calls == []


def test_translate_testcase_loi_mang_giu_truong_da_dich_va_dung_phan_con_lai():
    case = _case(
        user_input=_en_q(1),
        reference=_en_a(1),
        reference_contexts=[EN_CONTEXT],
    )
    translator = FakeTranslator(
        {_en_q(1): _vi_q(1), _en_a(1): tr.TranslateError("x")},
    )

    result = tr.translate_testcase(case, translator, {})

    assert result.translation_review == "translate_error"
    assert result.user_input == _vi_q(1)
    assert result.original_en == {"user_input": _en_q(1)}
    assert result.reference == _en_a(1)
    assert result.reference_contexts == [EN_CONTEXT]
    assert translator.calls == [_en_q(1), _en_a(1)]  # không thử contexts sau lỗi mạng


def test_translate_testcase_loi_mang_o_truong_dau_khong_doi_gi_ngoai_co():
    case = _case(user_input=_en_q(1), reference=_en_a(1))
    translator = FakeTranslator({_en_q(1): tr.TranslateError("x")})

    result = tr.translate_testcase(case, translator, {})

    assert result.translation_review == "translate_error"
    assert (result.user_input, result.reference) == (_en_q(1), _en_a(1))
    assert result.original_en is None
    assert translator.calls == [_en_q(1)]


def test_translate_testcase_loi_cau_hinh_duoc_truyen_ra_khong_gan_co():
    case = _case(user_input=_en_q(1))
    translator = FakeTranslator({_en_q(1): tr.TranslatorConfigError("forbidden")})

    with pytest.raises(tr.TranslatorConfigError):
        tr.translate_testcase(case, translator, {})


def test_translate_testcase_giu_original_en_cu_va_gop_truong_moi():
    case = _case(
        user_input=_vi_q(1),
        reference=_en_a(1),
        original_en={"user_input": _en_q(1)},
        translation_review="citation_mismatch",
    )

    result = tr.translate_testcase(case, FakeTranslator(_pairs(2)), {})

    assert result.original_en == {"user_input": _en_q(1), "reference": _en_a(1)}
    assert result.reference == _vi_a(1)
    assert result.translation_review is None  # dịch lại thành công thì xoá cờ


def test_translate_testcase_ap_bang_thuat_ngu_len_ban_dich_khong_len_ban_goc():
    vi_answer = "Lương hưu xã hội của người sử dụng lao động 1 theo Khoản 2 Điều 1."
    translator = FakeTranslator({_en_a(1): vi_answer})
    glossary = {"lương hưu xã hội": "trợ cấp hưu trí xã hội"}
    case = _case(reference=_en_a(1))

    result = tr.translate_testcase(case, translator, glossary)

    assert result.reference.startswith("Trợ cấp hưu trí xã hội của")
    assert result.original_en == {"reference": _en_a(1)}


# ---------------------------------------------------------------------------
# translate_raw: ghi raw, resume, dừng, mã thoát (12.5, 12.6)
# ---------------------------------------------------------------------------


def test_translate_raw_chi_doi_truong_da_dich_giu_thu_tu_va_truong_thua(out: Path):
    rows = [
        _vi_row(0, synthesizer_name="s", note="sửa tay"),
        _en_row(1, synthesizer_name="s", source_document="a.md", note="sửa tay"),
        _vi_row(2),
        _en_row(3),
    ]
    _write_raw(out, rows)
    translator = FakeTranslator(_pairs(4))

    report = tr.translate_raw(out, translator)

    result = _read_raw(out)
    assert [r["user_input"] for r in result] == [_vi_q(i) for i in range(4)]
    assert result[0] == rows[0]  # dòng không cần dịch: không đổi, không thêm khoá
    assert result[2] == rows[2]
    assert result[1]["reference"] == _vi_a(1)
    assert result[1]["reference_contexts"] == [VI_CONTEXT]
    assert result[1]["original_en"] == {
        "user_input": _en_q(1),
        "reference": _en_a(1),
    }
    assert result[1]["translation_review"] is None
    assert result[1]["note"] == "sửa tay"
    assert result[1]["source_document"] == "a.md"
    assert result[1]["synthesizer_name"] == "s"
    assert translator.calls == [_en_q(1), _en_a(1), _en_q(3), _en_a(3)]
    assert report.translated_samples == 2
    assert report.translated_fields == 4
    assert report.not_needed == 2
    assert report.skipped_flagged == 0
    assert report.flagged_total == 0
    assert report.exit_code == 0


def test_translate_raw_chay_lai_la_idempotent_khong_goi_mang_khong_ghi_lai(out: Path):
    path = _write_raw(out, [_en_row(0), _vi_row(1), _en_row(2)])
    tr.translate_raw(out, FakeTranslator(_pairs(3)))
    after_first = path.read_bytes()
    second = FakeTranslator({})

    report = tr.translate_raw(out, second)

    assert second.calls == []
    assert path.read_bytes() == after_first
    assert report.translated_samples == 0
    assert report.not_needed == 3
    assert report.exit_code == 0


def test_translate_raw_ghi_sau_moi_mau_nen_chet_giua_chung_van_lam_tiep_duoc(
    out: Path,
):
    _write_raw(out, [_en_row(0), _vi_row(1), _en_row(2), _en_row(3)])
    responses = _pairs(4)
    responses[_en_q(3)] = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        tr.translate_raw(out, FakeTranslator(responses))

    partial = _read_raw(out)
    assert [r["user_input"] for r in partial] == [
        _vi_q(0),
        _vi_q(1),
        _vi_q(2),
        _en_q(3),
    ]
    resumed = FakeTranslator(_pairs(4))

    report = tr.translate_raw(out, resumed)

    assert resumed.calls == [_en_q(3), _en_a(3)]
    assert report.translated_samples == 1
    assert [r["user_input"] for r in _read_raw(out)] == [_vi_q(i) for i in range(4)]


def test_translate_raw_loi_cau_hinh_dung_ngay_nhung_mau_truoc_do_da_ghi(out: Path):
    _write_raw(out, [_en_row(0), _en_row(1), _en_row(2)])
    responses = _pairs(3)
    responses[_en_q(1)] = tr.TranslatorConfigError("forbidden")

    with pytest.raises(tr.TranslatorConfigError):
        tr.translate_raw(out, FakeTranslator(responses))

    result = _read_raw(out)
    assert result[0]["user_input"] == _vi_q(0)
    assert result[1]["user_input"] == _en_q(1)
    assert "translation_review" not in result[1]
    assert result[2]["user_input"] == _en_q(2)


def test_translate_raw_ghi_nguyen_tu_khi_ghi_dut_giua_chung_raw_van_hop_le(
    out: Path, monkeypatch: pytest.MonkeyPatch
):
    path = _write_raw(out, [_en_row(0), _en_row(1)])
    real_replace = os.replace
    replaces: list[int] = []

    def flaky_replace(src: Any, dst: Any) -> None:
        replaces.append(1)
        if len(replaces) == 2:
            raise OSError("đĩa đầy")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)

    with pytest.raises(OSError, match="đĩa đầy"):
        tr.translate_raw(out, FakeTranslator(_pairs(2)))

    # File chính luôn là bản hoàn chỉnh sau mẫu thứ nhất, không có trạng thái nửa vời.
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert rows[0]["user_input"] == _vi_q(0)
    assert rows[0]["original_en"] == {"user_input": _en_q(0), "reference": _en_a(0)}
    assert rows[1] == _en_row(1)


def test_translate_raw_limit_gioi_han_so_mau_can_dich(out: Path):
    _write_raw(out, [_en_row(0), _en_row(1), _en_row(2)])
    translator = FakeTranslator(_pairs(3))

    report = tr.translate_raw(out, translator, limit=2)

    result = _read_raw(out)
    assert [r["user_input"] for r in result] == [_vi_q(0), _vi_q(1), _en_q(2)]
    assert report.translated_samples == 2


def test_translate_raw_limit_khong_tinh_mau_bi_co_bo_qua(out: Path):
    flagged = _en_row(0, translation_review="too_long")
    _write_raw(out, [flagged, _en_row(1), _en_row(2)])

    report = tr.translate_raw(out, FakeTranslator(_pairs(3)), limit=1)

    result = _read_raw(out)
    assert result[0] == flagged
    assert result[1]["user_input"] == _vi_q(1)
    assert result[2]["user_input"] == _en_q(2)
    assert report.skipped_flagged == 1
    assert report.translated_samples == 1


def test_translate_raw_mau_bi_co_mac_dinh_bi_bo_qua_va_thoat_ma_3(out: Path):
    flagged = _en_row(0, translation_review="too_long")
    path = _write_raw(out, [flagged])
    before = path.read_bytes()
    translator = FakeTranslator({})

    report = tr.translate_raw(out, translator)

    assert translator.calls == []
    assert path.read_bytes() == before
    assert report.skipped_flagged == 1
    assert report.flagged_total == 1
    assert report.exit_code == 3


def test_translate_raw_retry_flagged_dich_lai_va_xoa_co_khi_thanh_cong(out: Path):
    flagged = _en_row(0, translation_review="translate_error")
    _write_raw(out, [flagged])

    report = tr.translate_raw(out, FakeTranslator(_pairs(1)), retry_flagged=True)

    row = _read_raw(out)[0]
    assert row["user_input"] == _vi_q(0)
    assert row["translation_review"] is None
    assert report.skipped_flagged == 0
    assert report.flagged_total == 0
    assert report.exit_code == 0


def test_translate_raw_retry_flagged_chi_dem_truong_moi_dich(out: Path):
    row = _row(
        _vi_q(1),
        _en_a(1),
        original_en={"user_input": _en_q(1)},
        translation_review="citation_mismatch",
    )
    _write_raw(out, [row])

    report = tr.translate_raw(out, FakeTranslator(_pairs(2)), retry_flagged=True)

    saved = _read_raw(out)[0]
    assert saved["original_en"] == {"user_input": _en_q(1), "reference": _en_a(1)}
    assert saved["translation_review"] is None
    assert report.translated_fields == 1


def test_translate_raw_dung_sau_3_mau_lien_tiep_translate_error_thoat_ma_1(out: Path):
    rows = [_en_row(i) for i in range(5)]
    _write_raw(out, rows)
    errors: dict[str, str | Exception] = {
        _en_q(i): tr.TranslateError("x") for i in range(5)
    }
    translator = FakeTranslator(errors)

    report = tr.translate_raw(out, translator)

    assert translator.calls == [_en_q(0), _en_q(1), _en_q(2)]
    assert report.stopped_by_consecutive_errors
    assert report.exit_code == 1
    assert report.flagged_this_run == {"translate_error": 3}
    saved = _read_raw(out)
    assert [r["translation_review"] for r in saved[:3]] == ["translate_error"] * 3
    assert all(r["user_input"] == _en_q(i) for i, r in enumerate(saved[:3]))
    assert saved[3:] == rows[3:]  # chưa đụng tới


@pytest.mark.parametrize("third_row", ["ok", "citation_mismatch"])
def test_translate_raw_mau_khong_phai_translate_error_reset_bo_dem_lien_tiep(
    out: Path, third_row: str
):
    _write_raw(out, [_en_row(i) for i in range(5)])
    responses = _pairs(5)
    for i in (0, 1, 3, 4):
        responses[_en_q(i)] = tr.TranslateError("x")
    if third_row == "citation_mismatch":
        responses[_en_q(2)] = _vi_q(2).replace("Điều 2", "Điều 99")
    translator = FakeTranslator(responses)

    report = tr.translate_raw(out, translator)

    assert not report.stopped_by_consecutive_errors
    assert _en_q(4) in translator.calls  # đã đi hết 5 mẫu
    assert report.flagged_total == (4 if third_row == "ok" else 5)
    assert report.exit_code == 3


def test_translate_raw_dem_co_theo_ly_do_va_so_truong_da_dich(out: Path):
    _write_raw(out, [_en_row(0), _en_row(1), _en_row(2)])
    responses = _pairs(3)
    responses[_en_q(1)] = _vi_q(1).replace("Điều 1", "Điều 99")
    responses[_en_q(2)] = tr.TranslateError("x")

    report = tr.translate_raw(out, FakeTranslator(responses))

    assert report.flagged_this_run == {"citation_mismatch": 1, "translate_error": 1}
    assert report.translated_fields == 3  # (user_input + reference) + reference
    assert report.flagged_total == 2
    assert report.exit_code == 3


def test_translate_raw_mau_chi_bi_co_khong_dich_duoc_truong_nao_thi_khong_tinh_la_da_dich(
    out: Path,
):
    _write_raw(out, [_en_row(0), _en_row(1)])
    responses = _pairs(2)
    responses[_en_q(0)] = _vi_q(0).replace("Điều 0", "Điều 99")
    responses[_en_a(0)] = _vi_a(0).replace("Điều 0", "Điều 99")

    report = tr.translate_raw(out, FakeTranslator(responses))

    assert (
        report.translated_samples == 1
    )  # chỉ mẫu 1; mẫu 0 bị cờ, không trường nào dịch
    assert report.translated_fields == 2
    assert report.flagged_this_run == {"citation_mismatch": 1}


def test_translate_raw_ma_thoat_3_tinh_theo_moi_dong_raw_con_translation_review(
    out: Path,
):
    # Dòng tiếng Việt nhưng còn cờ từ lần trước/soát tay chưa xoá: vẫn tính.
    _write_raw(out, [_vi_row(0, translation_review="citation_mismatch"), _vi_row(1)])
    translator = FakeTranslator({})

    report = tr.translate_raw(out, translator)

    assert translator.calls == []
    assert report.translated_samples == 0
    assert report.not_needed == 2
    assert report.flagged_total == 1
    assert report.exit_code == 3


def test_translation_report_exit_code():
    assert tr.TranslationReport().exit_code == 0
    assert tr.TranslationReport(flagged_total=2).exit_code == 3
    assert tr.TranslationReport(stopped_by_consecutive_errors=True).exit_code == 1
    both = tr.TranslationReport(stopped_by_consecutive_errors=True, flagged_total=2)
    assert both.exit_code == 1  # dừng vì lỗi liên tiếp thắng "còn mẫu cần soát"


def test_translate_raw_ap_bang_thuat_ngu_tu_output_dir_len_ban_dich(out: Path):
    _write_raw(out, [_en_row(0)])
    glossary = {"lương hưu xã hội": "trợ cấp hưu trí xã hội"}
    (out / tr.GLOSSARY_FILENAME).write_text(
        json.dumps(glossary, ensure_ascii=False), encoding="utf-8"
    )
    responses = _pairs(1)
    responses[_en_a(0)] = "Lương hưu xã hội của người lao động 0 theo Khoản 2 Điều 0."

    tr.translate_raw(out, FakeTranslator(responses))

    row = _read_raw(out)[0]
    assert row["reference"].startswith("Trợ cấp hưu trí xã hội của")
    assert row["original_en"]["reference"] == _en_a(0)


def test_translate_raw_bang_thuat_ngu_hong_la_loi_dau_vao_truoc_khi_dich(out: Path):
    path = _write_raw(out, [_en_row(0)])
    before = path.read_bytes()
    (out / tr.GLOSSARY_FILENAME).write_text("{hỏng", encoding="utf-8")
    translator = FakeTranslator({})

    with pytest.raises(tg.EvalInputError):
        tr.translate_raw(out, translator)

    assert translator.calls == []
    assert path.read_bytes() == before


def test_translate_raw_thieu_raw_la_loi_dau_vao_nen_chay_generate(out: Path):
    with pytest.raises(tg.EvalInputError, match="generate"):
        tr.translate_raw(out, FakeTranslator({}))


def test_translate_raw_dong_khong_hop_le_la_loi_dau_vao_nen_so_dong(out: Path):
    _write_raw(out, [_vi_row(0), {"user_input": "thiếu trường"}])

    with pytest.raises(tg.EvalInputError, match="dòng 2"):
        tr.translate_raw(out, FakeTranslator({}))


def test_translate_raw_khong_phai_danh_sach_la_loi_dau_vao(out: Path):
    out.mkdir()
    (out / tg.RAW_TESTSET_FILENAME).write_text('{"a": 1}', encoding="utf-8")

    with pytest.raises(tg.EvalInputError):
        tr.translate_raw(out, FakeTranslator({}))


def test_translate_raw_log_chi_co_so_dem_khong_noi_dung_hay_thong_diep_loi(
    out: Path, caplog: pytest.LogCaptureFixture
):
    _write_raw(out, [_en_row(0), _en_row(1)])
    responses = _pairs(2)
    responses[_en_q(1)] = RuntimeError("SECRET-CONTENT-IN-ERROR")
    caplog.set_level(logging.INFO, logger=tr.__name__)

    with pytest.raises(RuntimeError):
        tr.translate_raw(out, FakeTranslator(responses))

    assert "Dịch dòng 1 xong" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "SECRET-CONTENT-IN-ERROR" not in caplog.text
    assert _en_q(0) not in caplog.text and _vi_q(0) not in caplog.text
    assert _en_q(1) not in caplog.text


# ---------------------------------------------------------------------------
# plan_translation (--dry-run)
# ---------------------------------------------------------------------------


def test_plan_translation_liet_ke_truong_tieng_anh_theo_dong_va_tong_ky_tu(out: Path):
    path = _write_raw(out, [_vi_row(0), _en_row(1), _vi_row(2)])
    before = path.read_bytes()

    plan = tr.plan_translation(out)

    main_table = plan.split("Vùng mơ hồ")[0]
    lines = [line for line in main_table.splitlines() if "user_input" in line]
    assert len(lines) == 1 and lines[0].split()[0] == "2"  # số dòng 1-based
    assert any(line.startswith("   2  reference") for line in main_table.splitlines())
    total_chars = len(_en_q(1)) + len(_en_a(1))
    assert "Tóm tắt: 1 mẫu, 2 trường tiếng Anh" in plan
    assert f"{total_chars} ký tự cần dịch" in plan
    assert "bỏ qua 0 mẫu đã bị cờ" in plan
    assert path.read_bytes() == before  # không ghi gì


def test_plan_translation_vung_mo_ho_ghi_ro_se_dich_hay_giu_nguyen(out: Path):
    keep = " ".join(["xã"] * 2 + ["abc"] * 8)  # 0,20: >= ngưỡng, giữ nguyên
    translate = " ".join(["xã"] * 2 + ["abc"] * 23)  # 0,08: < ngưỡng, sẽ dịch
    _write_raw(
        out,
        [_row(_vi_q(0), keep), _row(_vi_q(1), translate), _vi_row(2)],
    )

    plan = tr.plan_translation(out)

    main_table, ambiguous = plan.split("Vùng mơ hồ")
    assert "reference" in main_table and "0.08" in main_table
    assert "0.20" not in main_table
    ambiguous_lines = [line for line in ambiguous.splitlines() if "reference" in line]
    assert len(ambiguous_lines) == 2
    assert "(giữ nguyên)" in ambiguous_lines[0] and "0.20" in ambiguous_lines[0]
    assert "(sẽ dịch)" in ambiguous_lines[1] and "0.08" in ambiguous_lines[1]


def test_plan_translation_bo_qua_mau_bi_co_tru_khi_retry_flagged_va_ton_trong_limit(
    out: Path,
):
    _write_raw(
        out,
        [_en_row(0, translation_review="too_long"), _en_row(1), _en_row(2)],
    )

    default = tr.plan_translation(out)
    retried = tr.plan_translation(out, retry_flagged=True)
    limited = tr.plan_translation(out, limit=1)

    assert "Tóm tắt: 2 mẫu" in default and "bỏ qua 1 mẫu đã bị cờ" in default
    assert "Tóm tắt: 3 mẫu" in retried and "bỏ qua 0 mẫu đã bị cờ" in retried
    assert "Tóm tắt: 1 mẫu" in limited


def test_plan_translation_khong_co_mau_tieng_anh(out: Path):
    _write_raw(out, [_vi_row(0), _vi_row(1)])

    plan = tr.plan_translation(out)

    assert "Tóm tắt: 0 mẫu, 0 trường" in plan


def test_plan_translation_thieu_hoac_hong_raw_la_loi_dau_vao(out: Path):
    with pytest.raises(tg.EvalInputError):
        tr.plan_translation(out)
    _write_raw(out, [{"user_input": "thiếu"}])
    with pytest.raises(tg.EvalInputError, match="dòng 1"):
        tr.plan_translation(out)


# ---------------------------------------------------------------------------
# Model dữ liệu (12.5)
# ---------------------------------------------------------------------------


def test_golden_test_case_raw_cu_khong_co_hai_cot_moi_van_doc_duoc():
    old_row = {
        "user_input": _vi_q(0),
        "reference": _vi_a(0),
        "reference_contexts": [VI_CONTEXT],
        "synthesizer_name": "single_hop_specific_query_synthesizer",
    }

    case = GoldenTestCase.model_validate(old_row)

    assert case.original_en is None
    assert case.translation_review is None
    dumped = case.model_dump()
    assert dumped["original_en"] is None and dumped["translation_review"] is None


def test_golden_test_case_original_en_chua_chuoi_va_list_khoi_phuc_nguyen_ve():
    data = _row(
        _vi_q(0),
        _vi_a(0),
        original_en={
            "user_input": _en_q(0),
            "reference_contexts": [EN_CONTEXT, VI_CONTEXT],
        },
        translation_review="contexts_translated",
    )

    case = GoldenTestCase.model_validate(data)
    restored = GoldenTestCase.model_validate_json(case.model_dump_json())

    assert restored == case
    assert restored.original_en == data["original_en"]
    assert restored.translation_review == "contexts_translated"


@pytest.mark.parametrize(
    "extra",
    [
        {"original_en": "không phải dict"},
        {"original_en": {"user_input": 5}},
        {"original_en": {"reference_contexts": [1, 2]}},
        {"translation_review": ["citation_mismatch"]},
    ],
)
def test_golden_test_case_hai_cot_moi_sai_kieu_bi_tu_choi(extra: dict[str, Any]):
    with pytest.raises(ValidationError):
        GoldenTestCase.model_validate(_row(_vi_q(0), _vi_a(0), **extra))


# ---------------------------------------------------------------------------
# finalize giữ hai cột phụ (12.5)
# ---------------------------------------------------------------------------


def _finalize_rows(count: int, flagged: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(count):
        extra: dict[str, Any] = {"source_document": "a.md", "synthesizer_name": "s"}
        if i < flagged:
            extra["original_en"] = {"user_input": _en_q(i)}
            extra["translation_review"] = "citation_mismatch"
        rows.append(_row(_vi_q(i), _vi_a(i), **extra))
    return rows


def test_finalize_giu_hai_cot_original_en_va_translation_review(out: Path):
    _write_raw(out, _finalize_rows(3, flagged=1))

    cases = tg.finalize_golden_testset(out, target=3)

    saved = json.loads((out / tg.GOLDEN_TESTSET_FILENAME).read_text(encoding="utf-8"))
    assert len(cases) == len(saved) == 3
    assert saved[0]["original_en"] == {"user_input": _en_q(0)}
    assert saved[0]["translation_review"] == "citation_mismatch"
    assert saved[1]["original_en"] is None
    assert saved[1]["translation_review"] is None


def test_finalize_raw_cu_khong_co_hai_cot_ghi_ra_null(out: Path):
    _write_raw(out, [_vi_row(0), _vi_row(1)])

    tg.finalize_golden_testset(out, target=2)

    saved = json.loads((out / tg.GOLDEN_TESTSET_FILENAME).read_text(encoding="utf-8"))
    assert all(r["original_en"] is None for r in saved)
    assert all(r["translation_review"] is None for r in saved)


# ---------------------------------------------------------------------------
# CLI `translate` / `finalize`
# ---------------------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    """Module `tools.generate_testset`; cwd tạm và không có biến TRANSLATE_* (không đọc `.env`)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRANSLATE_URL", raising=False)
    monkeypatch.delenv("TRANSLATE_KEY", raising=False)
    from tools import generate_testset

    return generate_testset


def _invoke(cli: ModuleType, *args: str) -> Any:
    return CliRunner().invoke(cli.app, list(args))


def _stub_translator(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, translator: FakeTranslator
) -> None:
    settings = tr.TranslateSettings.model_construct(
        url=URL, key=None, timeout_seconds=60
    )
    monkeypatch.setattr(cli, "load_translate_settings", lambda: settings)
    monkeypatch.setattr(cli, "AppsScriptTranslator", lambda _settings: translator)


def _install_fake_requests(
    monkeypatch: pytest.MonkeyPatch, reply: Callable[[str], SimpleNamespace]
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, *, params: dict[str, str], timeout: float) -> Any:
        calls.append({"url": url, "params": dict(params), "timeout": timeout})
        return reply(params["text"])

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=fake_get))
    return calls


def _json_reply(payload: object, status: int = 200) -> SimpleNamespace:
    body = json.dumps(payload).encode("utf-8")
    return SimpleNamespace(status_code=status, content=body)


def test_cli_translate_dry_run_khong_can_url_khong_goi_mang_khong_ghi(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, out: Path
):
    path = _write_raw(out, [_vi_row(0), _en_row(1)])
    before = path.read_bytes()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("--dry-run không được nạp cấu hình hay dựng translator")

    monkeypatch.setattr(cli, "load_translate_settings", forbidden)
    monkeypatch.setattr(cli, "AppsScriptTranslator", forbidden)

    result = _invoke(cli, "translate", "--dry-run", "--output-dir", str(out))

    assert result.exit_code == 0, result.output
    assert "Tóm tắt: 1 mẫu, 2 trường" in result.output
    assert path.read_bytes() == before


def test_cli_translate_thieu_translate_url_thoat_ma_2_nen_ten_bien(
    cli: ModuleType, out: Path
):
    path = _write_raw(out, [_en_row(0)])
    before = path.read_bytes()

    result = _invoke(cli, "translate", "--output-dir", str(out))

    assert result.exit_code == 2
    assert "TRANSLATE_URL" in result.output
    assert path.read_bytes() == before


def test_cli_translate_thieu_raw_thoat_ma_2(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, out: Path
):
    _stub_translator(cli, monkeypatch, FakeTranslator({}))

    result = _invoke(cli, "translate", "--output-dir", str(out))

    assert result.exit_code == 2
    assert "generate" in result.output


def test_cli_translate_sach_thoat_ma_0(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, out: Path
):
    _write_raw(out, [_en_row(0), _vi_row(1)])
    _stub_translator(cli, monkeypatch, FakeTranslator(_pairs(2)))

    result = _invoke(cli, "translate", "--output-dir", str(out))

    assert result.exit_code == 0, result.output
    assert "Đã dịch 1 mẫu (2 trường)" in result.output
    assert _read_raw(out)[0]["user_input"] == _vi_q(0)


def test_cli_translate_in_ly_do_co_lan_nay_de_doc(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, out: Path
):
    _write_raw(out, [_en_row(0)])
    untouched = FakeTranslator({_en_q(0): _en_q(0), _en_a(0): _en_a(0)})
    _stub_translator(cli, monkeypatch, untouched)

    result = _invoke(cli, "translate", "--output-dir", str(out))

    assert "Đã dịch 0 mẫu (0 trường)" in result.output
    assert "cờ lần này: still_english=1;" in result.output


def test_cli_translate_khong_co_co_lan_nay_in_khong(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, out: Path
):
    _write_raw(out, [_en_row(0)])
    _stub_translator(cli, monkeypatch, FakeTranslator(_pairs(1)))

    result = _invoke(cli, "translate", "--output-dir", str(out))

    assert "cờ lần này: không;" in result.output


def test_cli_translate_con_mau_can_soat_thoat_ma_3(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, out: Path
):
    _write_raw(out, [_en_row(0)])
    untouched = FakeTranslator({_en_q(0): _en_q(0), _en_a(0): _en_a(0)})
    _stub_translator(cli, monkeypatch, untouched)

    result = _invoke(cli, "translate", "--output-dir", str(out))

    assert result.exit_code == 3
    assert "soát tay" in result.output
    assert _read_raw(out)[0]["translation_review"] == "still_english"


def test_cli_translate_3_mau_lien_tiep_loi_thoat_ma_1(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, out: Path
):
    _write_raw(out, [_en_row(i) for i in range(4)])
    errors: dict[str, str | Exception] = {
        _en_q(i): tr.TranslateError("x") for i in range(4)
    }
    _stub_translator(cli, monkeypatch, FakeTranslator(errors))

    result = _invoke(cli, "translate", "--output-dir", str(out))

    assert result.exit_code == 1
    assert "Dừng" in result.output
    reviews = [r.get("translation_review") for r in _read_raw(out)]
    assert reviews == ["translate_error"] * 3 + [None]


def test_cli_translate_loi_cau_hinh_giua_chung_thoat_ma_2_giu_mau_da_dich(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, out: Path
):
    _write_raw(out, [_en_row(0), _en_row(1)])
    responses = _pairs(2)
    responses[_en_q(1)] = tr.TranslatorConfigError("forbidden")
    _stub_translator(cli, monkeypatch, FakeTranslator(responses))

    result = _invoke(cli, "translate", "--output-dir", str(out))

    assert result.exit_code == 2
    saved = _read_raw(out)
    assert saved[0]["user_input"] == _vi_q(0)
    assert saved[1]["user_input"] == _en_q(1)


def test_cli_translate_limit_va_retry_flagged_duoc_chuyen_xuong(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, out: Path
):
    flagged = _en_row(0, translation_review="translate_error")
    _write_raw(out, [flagged, _en_row(1), _en_row(2)])
    _stub_translator(cli, monkeypatch, FakeTranslator(_pairs(3)))

    first = _invoke(cli, "translate", "--output-dir", str(out), "--limit", "1")

    assert first.exit_code == 3  # mẫu 0 vẫn còn cờ
    saved = _read_raw(out)
    assert saved[0] == flagged
    assert [r["user_input"] for r in saved[1:]] == [_vi_q(1), _en_q(2)]

    second = _invoke(cli, "translate", "--output-dir", str(out), "--retry-flagged")

    assert second.exit_code == 0, second.output
    saved = _read_raw(out)
    assert [r["user_input"] for r in saved] == [_vi_q(i) for i in range(3)]
    assert saved[0]["translation_review"] is None


def test_cli_translate_limit_khong_hop_le_bi_tu_choi(cli: ModuleType, out: Path):
    _write_raw(out, [_en_row(0)])

    result = _invoke(cli, "translate", "--output-dir", str(out), "--limit", "0")

    assert result.exit_code == 2


def test_cli_translate_dau_cuoi_qua_requests_gia_gui_key_va_khong_lo_url_key(
    cli: ModuleType, monkeypatch: pytest.MonkeyPatch, out: Path
):
    _write_raw(out, [_en_row(0)])
    monkeypatch.setenv("TRANSLATE_URL", URL)
    monkeypatch.setenv("TRANSLATE_KEY", KEY)
    mapping = _pairs(1)

    def reply(text: str) -> SimpleNamespace:
        translated = mapping[text]
        assert isinstance(translated, str)
        return _json_reply({"translatedText": translated})

    calls = _install_fake_requests(monkeypatch, reply)

    result = _invoke(cli, "translate", "--output-dir", str(out))

    assert result.exit_code == 0, result.output
    assert [c["params"]["text"] for c in calls] == [_en_q(0), _en_a(0)]
    assert all(c["url"] == URL and c["params"]["key"] == KEY for c in calls)
    assert all(c["params"]["source"] == "en" for c in calls)
    assert all(c["params"]["target"] == "vi" for c in calls)
    assert _read_raw(out)[0]["reference"] == _vi_a(0)
    assert URL not in result.output and KEY not in result.output


@pytest.mark.parametrize(
    "reply_factory",
    [
        lambda: SimpleNamespace(status_code=200, content=b"<html>Sign in</html>"),
        lambda: _json_reply({"error": "forbidden"}),
        lambda: SimpleNamespace(status_code=403, content=b""),
    ],
    ids=["html", "forbidden", "http_403"],
)
def test_cli_translate_loi_cau_hinh_http_thoat_ma_2_khong_retry_khong_lo_bi_mat(
    cli: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    out: Path,
    reply_factory: Callable[[], SimpleNamespace],
):
    path = _write_raw(out, [_en_row(0), _en_row(1)])
    before = path.read_bytes()
    monkeypatch.setenv("TRANSLATE_URL", URL)
    monkeypatch.setenv("TRANSLATE_KEY", KEY)
    calls = _install_fake_requests(monkeypatch, lambda _text: reply_factory())

    result = _invoke(cli, "translate", "--output-dir", str(out))

    assert result.exit_code == 2
    assert len(calls) == 1  # không retry, không đi tiếp sang trường/mẫu khác
    assert URL not in result.output and KEY not in result.output
    assert "SECRET-DEPLOY-ID" not in result.output
    assert path.read_bytes() == before


def test_cli_finalize_canh_bao_so_dong_con_translation_review_khong_chan(
    cli: ModuleType, out: Path
):
    _write_raw(out, _finalize_rows(tg.TARGET_SIZE, flagged=2))

    result = _invoke(cli, "finalize", "--output-dir", str(out))

    assert result.exit_code == 0, result.output
    assert "Cảnh báo: 2 câu còn translation_review" in result.output
    saved = json.loads((out / tg.GOLDEN_TESTSET_FILENAME).read_text(encoding="utf-8"))
    assert len(saved) == tg.TARGET_SIZE
    assert saved[0]["original_en"] == {"user_input": _en_q(0)}
    assert saved[0]["translation_review"] == "citation_mismatch"


def test_cli_finalize_khong_canh_bao_khi_het_translation_review(
    cli: ModuleType, out: Path
):
    _write_raw(out, _finalize_rows(tg.TARGET_SIZE, flagged=0))

    result = _invoke(cli, "finalize", "--output-dir", str(out))

    assert result.exit_code == 0, result.output
    assert "Cảnh báo" not in result.output


# ---------------------------------------------------------------------------
# Cấu hình đi kèm (.env.example, pyproject)
# ---------------------------------------------------------------------------


def test_env_example_khai_bao_translate_url_key_de_trong():
    lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()

    assert "TRANSLATE_URL=" in lines
    assert "TRANSLATE_KEY=" in lines  # không có giá trị thật trong file mẫu


def test_pyproject_khai_bao_requests_trong_nhom_eval():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    eval_group = data["dependency-groups"]["eval"]

    assert any(str(dep).split("[")[0].strip() == "requests" for dep in eval_group)
