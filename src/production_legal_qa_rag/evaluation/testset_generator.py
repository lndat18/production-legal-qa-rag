"""Điều phối sinh golden testset RAGAS Phase 1 theo từng đơn vị (evaluation_spec.md mục 4).

Module này KHÔNG import `ragas`: toàn bộ phần chạm `ragas`/Groq nằm ở
`ragas_runner.py` (import lười trong `build_unit_runner`) để lệnh `generate
--dry-run` và `finalize` chạy được trên venv thường, không cần dependency-group
`eval` và không cần key Groq.

Việc của module: chia văn bản thành đơn vị (`unit_splitter`), sắp theo thứ tự chạy
nhỏ -> lớn, phân bổ số câu theo đơn vị (`allocate_questions`), chạy từng đơn vị qua
`UnitRunner`, nối kết quả vào `golden_testset_raw.json` rồi ghi
`generation_progress.json` nguyên tử (mục 4.5), dừng ngay khi một đơn vị lỗi, và
chốt đúng `TARGET_SIZE` câu (`finalize_golden_testset`, mục 4.2).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import unicodedata
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

from pydantic import BaseModel, ValidationError

from production_legal_qa_rag.evaluation.corpus_loader import DEFAULT_MARKDOWN_DIR
from production_legal_qa_rag.evaluation.models import (
    GenerationProgress,
    GoldenTestCase,
    UnitFailure,
    UnitProgress,
)
from production_legal_qa_rag.evaluation.unit_splitter import EvalUnit, split_directory

if TYPE_CHECKING:
    from production_legal_qa_rag.config import TestsetGeneratorSettings

logger = logging.getLogger(__name__)

# Hằng số nội bộ module, KHÔNG phải biến môi trường (mục 4). Sinh dư 240 để trừ hao
# khi người dùng đọc lướt và xoá câu xấu, `finalize` chốt còn đúng 180.
GENERATE_SIZE: Final = 240
TARGET_SIZE: Final = 180
# Tỷ lệ loại câu 80/10/10 (mục 1, 10.10): single-hop / multi-hop abstract / multi-hop specific.
_MULTI_HOP_SHARE: Final = 0.1
# Quota token/ngày của 6 tài khoản Groq free x 200K (mục 3.1), chỉ để ước lượng số ngày.
TOKENS_PER_DAY: Final = 1_200_000

DEFAULT_OUTPUT_DIR: Final = Path("data/eval")
RAW_TESTSET_FILENAME: Final = "golden_testset_raw.json"
GOLDEN_TESTSET_FILENAME: Final = "golden_testset.json"
PROGRESS_FILENAME: Final = "generation_progress.json"
KNOWLEDGE_GRAPH_DIRNAME: Final = "knowledge_graph"

QUESTION_TYPES: Final = ("single_hop", "abstract", "specific")
# Tên synthesizer của ragas (`synthesizer_name` trong output) -> loại câu trong progress.
SYNTHESIZER_TYPES: Final = {
    "single_hop_specific_query_synthesizer": "single_hop",
    "multi_hop_abstract_query_synthesizer": "abstract",
    "multi_hop_specific_query_synthesizer": "specific",
}

_ERROR_MESSAGE_MAX_CHARS: Final = 200


class EvalInputError(ValueError):
    """Đầu vào/tệp trạng thái không hợp lệ (progress hỏng, `--only` sai, thiếu câu...)."""


class UnitGenerationError(RuntimeError):
    """Một đơn vị lỗi giữa chừng (quota, timeout...); chương trình đã dừng (mục 4.5)."""

    def __init__(self, unit_key: str, description: str) -> None:
        super().__init__(f"Đơn vị {unit_key} lỗi và đã dừng: {description}")
        self.unit_key = unit_key


class QuestionQuota(BaseModel):
    """Số câu cần sinh cho một đơn vị, theo từng loại."""

    single_hop: int = 0
    abstract: int = 0
    specific: int = 0

    @property
    def total(self) -> int:
        """Tổng số câu của đơn vị (cả 3 loại)."""
        return self.single_hop + self.abstract + self.specific


class UnitResult(BaseModel):
    """Kết quả một đơn vị do `UnitRunner` trả về (chưa gắn nguồn, chưa ghi đĩa)."""

    cases: list[GoldenTestCase]
    llm_calls: int


class UnitRunner(Protocol):
    """Sinh câu hỏi cho MỘT đơn vị; cài đặt thật ở `ragas_runner.RagasUnitRunner`."""

    def run_unit(
        self,
        unit: EvalUnit,
        quota: QuestionQuota,
        knowledge_graph_path: Path,
        *,
        reuse_knowledge_graph: bool,
    ) -> UnitResult:
        """Dựng (hoặc nạp lại) KG của đơn vị, sinh câu hỏi, lưu KG khi thành công."""
        ...


class GenerationReport(BaseModel):
    """Tóm tắt một lần chạy `generate_testset`."""

    generated_units: list[str]
    skipped_units: list[str]
    new_questions: int


# ---------------------------------------------------------------------------
# Khoá, thứ tự chạy, phân bổ số câu (mục 4, 4.5)
# ---------------------------------------------------------------------------


def unit_key(unit: EvalUnit) -> str:
    """Khoá đơn vị `<tên file .md>#<số thứ tự>` dùng trong progress và `--only`."""
    return f"{unit.source_document}#{unit.index}"


def knowledge_graph_path(output_dir: Path, unit: EvalUnit) -> Path:
    """Đường dẫn KG của đơn vị: `knowledge_graph/<văn bản>__<số thứ tự>.json`."""
    stem = Path(unit.source_document).stem
    return output_dir / KNOWLEDGE_GRAPH_DIRNAME / f"{stem}__{unit.index:02d}.json"


def order_units(units: Sequence[EvalUnit]) -> list[EvalUnit]:
    """Thứ tự chạy: số ký tự tăng dần, hoà thì theo (tên văn bản, số thứ tự)."""
    return sorted(units, key=lambda u: (u.char_count, u.source_document, u.index))


def split_question_mix(total: int) -> QuestionQuota:
    """Chia tổng số câu theo 80/10/10 (`240 -> 192/24/24`)."""
    multi_hop = round(total * _MULTI_HOP_SHARE)
    return QuestionQuota(
        single_hop=total - 2 * multi_hop, abstract=multi_hop, specific=multi_hop
    )


def _largest_remainder(total: int, weights: Sequence[int]) -> list[int]:
    """Chia `total` tỷ lệ theo `weights`, phần dư lớn nhất; tổng luôn đúng `total`."""
    weight_sum = sum(weights)
    floors = [total * w // weight_sum for w in weights]
    remainders = [total * w % weight_sum for w in weights]
    order = sorted(range(len(weights)), key=lambda i: (-remainders[i], i))
    for index in order[: total - sum(floors)]:
        floors[index] += 1
    return floors


def allocate_questions(
    units: Sequence[EvalUnit], total: int = GENERATE_SIZE
) -> dict[str, QuestionQuota]:
    """Phân bổ `total` câu (80/10/10) cho các đơn vị, tỷ lệ theo số ký tự (mục 4).

    Mỗi loại phân bổ riêng bằng phương pháp phần dư lớn nhất nên tổng từng loại
    đúng bằng phần của nó (192/24/24 khi `total=240`).

    Returns:
        Khoá đơn vị (`unit_key`) -> quota. Đơn vị nhận 0 câu multi-hop chỉ chạy single-hop.
    """
    if not units:
        return {}
    mix = split_question_mix(total)
    weights = [u.char_count for u in units]
    per_type = {
        kind: _largest_remainder(getattr(mix, kind), weights) for kind in QUESTION_TYPES
    }
    return {
        unit_key(unit): QuestionQuota(
            **{kind: per_type[kind][i] for kind in QUESTION_TYPES}
        )
        for i, unit in enumerate(units)
    }


# ---------------------------------------------------------------------------
# Ghi/đọc file trạng thái (mục 4.5)
# ---------------------------------------------------------------------------


def _write_text_atomic(path: Path, content: str) -> None:
    """Ghi file tạm rồi đổi tên: người đọc không bao giờ thấy file dở dang."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(content, encoding="utf-8")
    os.replace(temp_path, path)


def load_progress(path: Path) -> GenerationProgress:
    """Đọc `generation_progress.json`; chưa có file thì là tiến độ rỗng.

    Raises:
        EvalInputError: File hỏng/không parse được/không phải UTF-8 — không coi như
            "chưa làm gì" (sẽ đốt lại toàn bộ quota, mục 8).
    """
    if not path.exists():
        return GenerationProgress()
    try:
        return GenerationProgress.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, UnicodeDecodeError) as error:
        raise EvalInputError(
            f"{path} hỏng hoặc sai định dạng; sửa/xoá tay rồi chạy lại: {error}"
        ) from error


def save_progress(path: Path, progress: GenerationProgress) -> None:
    """Ghi progress nguyên tử."""
    _write_text_atomic(path, progress.model_dump_json(indent=2) + "\n")


def read_raw_rows(path: Path) -> list[dict[str, Any]]:
    """Đọc `golden_testset_raw.json` thành danh sách dict nguyên trạng (giữ mọi trường).

    Raises:
        EvalInputError: File không phải UTF-8/JSON hợp lệ dạng danh sách các đối tượng.
    """
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise EvalInputError(f"{path} không phải JSON hợp lệ: {error}") from error
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise EvalInputError(f"{path} phải là danh sách các đối tượng câu hỏi.")
    return rows


def append_raw_cases(
    path: Path, cases: Sequence[GoldenTestCase]
) -> list[GoldenTestCase]:
    """Nối `cases` vào cuối file raw, bỏ câu có `user_input` trùng y hệt câu đã có.

    Không bao giờ ghi đè/sắp xếp lại các dòng sẵn có (người dùng đã sửa tay, mục 8).

    Returns:
        Các câu thực sự được nối (sau khi bỏ trùng).
    """
    rows = read_raw_rows(path)
    seen = {row.get("user_input") for row in rows}
    added: list[GoldenTestCase] = []
    for case in cases:
        if case.user_input in seen:
            continue
        seen.add(case.user_input)
        added.append(case)
    rows.extend(case.model_dump() for case in added)
    _write_text_atomic(path, json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    return added


def _count_by_type(synthesizer_names: Sequence[str | None]) -> dict[str, int]:
    counts = dict.fromkeys(QUESTION_TYPES, 0)
    for name in synthesizer_names:
        kind = SYNTHESIZER_TYPES.get(name or "")
        if kind is not None:
            counts[kind] += 1
    return counts


# ---------------------------------------------------------------------------
# Chuẩn bị lần chạy: chia đơn vị, --only, kiểm tra khớp, khôi phục (mục 4.5, 8)
# ---------------------------------------------------------------------------


def _normalize(name: str) -> str:
    return unicodedata.normalize("NFC", name)


def select_units(units: Sequence[EvalUnit], only: Sequence[str]) -> list[EvalUnit]:
    """Lọc đơn vị theo `--only <tên>[#<số>]` (tên có/không đuôi `.md`); rỗng = tất cả.

    Raises:
        EvalInputError: Tên văn bản hoặc số thứ tự không tồn tại (liệt kê giá trị hợp lệ).
    """
    if not only:
        return list(units)
    documents = sorted({u.source_document for u in units})
    chosen: set[str] = set()
    for item in only:
        name, _, index_text = _normalize(item).partition("#")
        matches = [
            d for d in documents if name in (_normalize(d), _normalize(Path(d).stem))
        ]
        if not matches:
            raise EvalInputError(
                f"--only {item!r}: không có văn bản này. Hợp lệ: {', '.join(documents)}"
            )
        document = matches[0]
        indexes = [u.index for u in units if u.source_document == document]
        if index_text and not (index_text.isdigit() and int(index_text) in indexes):
            raise EvalInputError(
                f"--only {item!r}: {document} chỉ có đơn vị số {indexes[0]}-{indexes[-1]}."
            )
        chosen.update(
            unit_key(u)
            for u in units
            if u.source_document == document
            and (not index_text or u.index == int(index_text))
        )
    return [u for u in units if unit_key(u) in chosen]


def _check_progress_matches_source(
    units: Sequence[EvalUnit], progress: GenerationProgress
) -> None:
    """Dừng nếu đơn vị đã xong không còn khớp nguồn (số thứ tự có thể trỏ sang đơn vị khác)."""
    by_key = {unit_key(u): u for u in units}
    for key, done in progress.units.items():
        unit = by_key.get(key)
        if unit is None or unit.char_count != done.chars:
            current = (
                "không còn tồn tại" if unit is None else f"nay {unit.char_count} ký tự"
            )
            raise EvalInputError(
                f"Đơn vị {key} đã xong với {done.chars} ký tự nhưng {current}: "
                "văn bản nguồn hoặc quy tắc chia đã đổi. Không tự bỏ qua/ghi đè — "
                "kiểm tra rồi sửa/xoá dòng đó trong generation_progress.json."
            )


def _rows_of_unit(
    rows: Sequence[dict[str, Any]], unit: EvalUnit
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("source_document") == unit.source_document
        and row.get("source_section") == unit.title
    ]


def _recoverable_units(
    units: Sequence[EvalUnit],
    progress: GenerationProgress,
    rows: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Đơn vị có dòng trong raw nhưng chưa có trong progress (chết giữa hai lần ghi)."""
    recoverable = {}
    for unit in units:
        key = unit_key(unit)
        unit_rows = [] if key in progress.units else _rows_of_unit(rows, unit)
        if unit_rows:
            recoverable[key] = unit_rows
    return recoverable


def _new_unit_progress(
    unit: EvalUnit, questions: dict[str, int], llm_calls: int, seconds: float
) -> UnitProgress:
    return UnitProgress(
        title=unit.title,
        chars=unit.char_count,
        estimated_tokens=unit.estimated_tokens,
        questions=questions,
        llm_calls=llm_calls,
        seconds=seconds,
        completed_at=datetime.now().astimezone(),
    )


def _load_units(markdown_dir: Path) -> list[EvalUnit]:
    """Chia `markdown_dir` thành đơn vị; fail fast nếu thiếu/rỗng/sai mã hoá (mục 8)."""
    try:
        units = split_directory(markdown_dir)
    except (FileNotFoundError, UnicodeDecodeError) as error:
        raise EvalInputError(f"Không đọc được văn bản nguồn: {error}") from error
    for unit in units:
        if not unit.text.strip():
            raise EvalInputError(f"File markdown trống: {unit.source_document}")
    return units


class _RunState(BaseModel):
    """Trạng thái đã đọc + đã kiểm tra, dùng chung cho `generate_testset` và dry-run."""

    all_units: list[EvalUnit]
    selected: list[EvalUnit]  # theo thứ tự chạy
    quotas: dict[str, QuestionQuota]
    progress: GenerationProgress
    raw_rows: list[dict[str, Any]]
    recoverable: dict[str, list[dict[str, Any]]]


def _require_only_with_append(only: Sequence[str], append: bool) -> None:
    """`--append` chạy lại cả đơn vị đã xong nên phải chỉ rõ đơn vị (tránh chạy lại cả 50)."""
    if append and not only:
        raise EvalInputError(
            "--append phải đi kèm --only: nó chạy lại cả đơn vị đã xong, không có --only "
            "sẽ chạy lại mọi đơn vị và đốt quota nhiều ngày."
        )


def _prepare_run(
    markdown_dir: Path, output_dir: Path, only: Sequence[str], testset_size: int | None
) -> _RunState:
    all_units = _load_units(markdown_dir)
    selected = order_units(select_units(all_units, only))
    progress = load_progress(output_dir / PROGRESS_FILENAME)
    _check_progress_matches_source(all_units, progress)
    raw_rows = read_raw_rows(output_dir / RAW_TESTSET_FILENAME)
    # Không truyền --testset-size: quota theo cả corpus để chạy từng phần cho kết quả
    # giống chạy một lượt. Có --testset-size (sinh bù): chia N cho đúng các đơn vị đã chọn.
    quota_units = all_units if testset_size is None else selected
    return _RunState(
        all_units=all_units,
        selected=selected,
        quotas=allocate_questions(quota_units, testset_size or GENERATE_SIZE),
        progress=progress,
        raw_rows=raw_rows,
        recoverable=_recoverable_units(all_units, progress, raw_rows),
    )


# ---------------------------------------------------------------------------
# Kế hoạch / dry-run (không gọi LLM, không ghi file)
# ---------------------------------------------------------------------------


def _format_time(moment: datetime) -> str:
    return moment.strftime("%d/%m %H:%M")


def _unit_status(unit: EvalUnit, state: _RunState) -> str:
    key = unit_key(unit)
    done = state.progress.units.get(key)
    if done is not None:
        return f"xong {_format_time(done.completed_at)}"
    if key in state.recoverable:
        return "xong (có dòng trong raw, chưa ghi progress)"
    failure = state.progress.last_failure
    if failure is not None and failure.unit == key:
        return f"lỗi lần cuối {_format_time(failure.at)}"
    return "chưa"


def _pending_units(state: _RunState, append: bool) -> list[EvalUnit]:
    return [
        u
        for u in state.selected
        if append
        or (
            unit_key(u) not in state.progress.units
            and unit_key(u) not in state.recoverable
        )
    ]


def _render_plan(state: _RunState, *, append: bool) -> str:
    """Bảng kế hoạch theo thứ tự chạy + tóm tắt tiến độ (mục 4.5)."""
    pending_keys = {unit_key(u) for u in _pending_units(state, append)}
    next_key = next(
        (unit_key(u) for u in state.selected if unit_key(u) in pending_keys), None
    )
    lines = [
        "| Thứ tự | Khoá đơn vị | Đơn vị | Ký tự | Ước lượng token | Số câu | Trạng thái |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for rank, unit in enumerate(state.selected, start=1):
        key = unit_key(unit)
        marker = "  <- CHẠY TIẾP THEO" if key == next_key else ""
        lines.append(
            f"| {rank} | {key} | {unit.title} | {unit.char_count:,} "
            f"| {unit.estimated_tokens // 1000}K | {state.quotas[key].total} "
            f"| {_unit_status(unit, state)}{marker} |"
        )
    remaining = [u for u in state.selected if unit_key(u) in pending_keys]
    lines.append("")
    lines.append(_summary_line(len(state.selected), remaining))
    return "\n".join(lines)


def _summary_line(total_units: int, remaining: Sequence[EvalUnit]) -> str:
    tokens = sum(u.estimated_tokens for u in remaining)
    return (
        f"Đã xong {total_units - len(remaining)}/{total_units} đơn vị, còn {len(remaining)} đơn vị "
        f"ước lượng ~{tokens / 1e6:.2f}M token (≈ {tokens / TOKENS_PER_DAY:.1f} ngày "
        f"ở {TOKENS_PER_DAY / 1e6:.1f}M token/ngày)."
    )


def plan_generation(
    markdown_dir: Path = DEFAULT_MARKDOWN_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    only: Sequence[str] = (),
    append: bool = False,
    testset_size: int | None = None,
) -> str:
    """Kế hoạch chạy (`generate --dry-run`): không gọi LLM, không cần key Groq, không ghi file."""
    _require_only_with_append(only, append)
    return _render_plan(
        _prepare_run(markdown_dir, output_dir, only, testset_size), append=append
    )


def summarize_progress(
    markdown_dir: Path = DEFAULT_MARKDOWN_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> str:
    """Một dòng tóm tắt tiến độ toàn corpus (in sau khi dừng giữa chừng)."""
    state = _prepare_run(markdown_dir, output_dir, (), None)
    return _summary_line(len(state.all_units), _pending_units(state, append=False))


# ---------------------------------------------------------------------------
# Chạy sinh câu hỏi (mục 4.5)
# ---------------------------------------------------------------------------


def _describe_error(error: BaseException) -> str:
    """Mô tả lỗi cho `last_failure`/log: chỉ kèm thông điệp khi là lỗi HTTP từ Groq.

    Thông điệp lỗi của ragas (parse output LLM...) có thể chứa nội dung câu hỏi/context,
    mà chính sách log của project cấm ghi nội dung đó (mục 8).
    """
    status = getattr(error, "status_code", None)
    if status is None:
        return type(error).__name__
    message = (
        str(error).splitlines()[0][:_ERROR_MESSAGE_MAX_CHARS] if str(error) else ""
    )
    return f"{type(error).__name__} (HTTP {status}): {message}"


def _record_failure(
    progress_path: Path, progress: GenerationProgress, key: str, error: BaseException
) -> str:
    description = _describe_error(error)
    progress.last_failure = UnitFailure(
        unit=key, error=description, at=datetime.now().astimezone()
    )
    save_progress(progress_path, progress)
    return description


def _recover_unfinished(state: _RunState, progress_path: Path) -> None:
    """Ghi bổ sung progress cho đơn vị đã có dòng trong raw mà chưa được ghi nhận."""
    by_key = {unit_key(u): u for u in state.all_units}
    for key, rows in state.recoverable.items():
        logger.warning(
            "Đơn vị %s có %d dòng trong raw nhưng chưa có trong progress: coi là đã xong, ghi bổ sung.",
            key,
            len(rows),
        )
        counts = _count_by_type([row.get("synthesizer_name") for row in rows])
        state.progress.units[key] = _new_unit_progress(by_key[key], counts, 0, 0.0)
    if state.recoverable:
        save_progress(progress_path, state.progress)


def _record_unit_done(
    progress: GenerationProgress,
    unit: EvalUnit,
    added: Sequence[GoldenTestCase],
    llm_calls: int,
    seconds: float,
) -> None:
    key = unit_key(unit)
    counts = _count_by_type([case.synthesizer_name for case in added])
    previous = progress.units.get(key)
    if previous is not None:  # --append: cộng dồn
        counts = {k: counts[k] + previous.questions.get(k, 0) for k in QUESTION_TYPES}
        llm_calls += previous.llm_calls
        seconds += previous.seconds
    progress.units[key] = _new_unit_progress(unit, counts, llm_calls, seconds)
    if progress.last_failure is not None and progress.last_failure.unit == key:
        progress.last_failure = None


def _process_unit(
    runner: UnitRunner,
    unit: EvalUnit,
    quota: QuestionQuota,
    output_dir: Path,
    *,
    reuse_knowledge_graph: bool,
) -> tuple[list[GoldenTestCase], int]:
    """Chạy `runner` cho một đơn vị và nối kết quả vào raw; trả (câu đã nối, số lượt gọi)."""
    result = runner.run_unit(
        unit,
        quota,
        knowledge_graph_path(output_dir, unit),
        reuse_knowledge_graph=reuse_knowledge_graph,
    )
    cases = [
        case.model_copy(
            update={
                "source_document": unit.source_document,
                "source_section": unit.title,
            }
        )
        for case in result.cases
    ]
    return append_raw_cases(output_dir / RAW_TESTSET_FILENAME, cases), result.llm_calls


def build_unit_runner(
    settings: TestsetGeneratorSettings | None = None,
) -> UnitRunner:
    """Dựng runner thật (ragas + 6 tài khoản Groq); import lười để module này không cần ragas.

    Raises:
        EvalInputError: Thiếu/sai biến môi trường Groq. Thông báo chỉ nêu TÊN biến, không
            kèm giá trị (lỗi gốc của pydantic in đầu/đuôi key).
    """
    from production_legal_qa_rag.evaluation.ragas_runner import RagasUnitRunner

    try:
        return RagasUnitRunner(settings)
    except ValidationError as error:
        names = sorted(
            {".".join(str(part) for part in e["loc"]) for e in error.errors()}
        )
        raise EvalInputError(
            "Thiếu hoặc sai cấu hình Groq trong .env (cần GROQ_API_KEY và "
            f"GROQ_API_KEY_2 ... GROQ_API_KEY_6): {', '.join(names)}"
        ) from None


def _run_one_unit(
    runner: UnitRunner,
    unit: EvalUnit,
    state: _RunState,
    output_dir: Path,
    *,
    reuse_knowledge_graph: bool,
) -> int:
    """Chạy + checkpoint một đơn vị; lỗi thì ghi `last_failure` và dừng. Trả số câu mới."""
    key = unit_key(unit)
    progress_path = output_dir / PROGRESS_FILENAME
    # Đơn vị chưa xong có thể còn KG hoàn chỉnh từ lần lỗi trước (lưu ngay sau khi dựng,
    # mục 4.5): dùng lại thay vì tốn lại 30-160K token. Đơn vị đã xong (--append) chỉ dùng
    # lại khi người dùng đòi bằng --reuse-knowledge-graph.
    reuse = reuse_knowledge_graph or key not in state.progress.units
    started = time.monotonic()
    try:
        added, llm_calls = _process_unit(
            runner,
            unit,
            state.quotas[key],
            output_dir,
            reuse_knowledge_graph=reuse,
        )
    except (Exception, KeyboardInterrupt) as error:
        description = _record_failure(progress_path, state.progress, key, error)
        logger.error("Đơn vị %s lỗi, dừng: %s", key, description)
        if isinstance(error, KeyboardInterrupt):
            raise
        raise UnitGenerationError(key, description) from error
    _record_unit_done(
        state.progress, unit, added, llm_calls, time.monotonic() - started
    )
    save_progress(progress_path, state.progress)
    logger.info("Xong %s: +%d câu, %d lượt gọi LLM.", key, len(added), llm_calls)
    return len(added)


def generate_testset(
    markdown_dir: Path = DEFAULT_MARKDOWN_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    only: Sequence[str] = (),
    reuse_knowledge_graph: bool = False,
    append: bool = False,
    testset_size: int | None = None,
    unit_runner: UnitRunner | None = None,
    settings: TestsetGeneratorSettings | None = None,
) -> GenerationReport:
    """Sinh testset theo từng đơn vị, checkpoint sau mỗi đơn vị (mục 4, 4.5).

    Args:
        markdown_dir: Thư mục `data/markdown/*.md`.
        output_dir: Thư mục `data/eval` (raw, progress, knowledge_graph/).
        only: Bộ lọc `<tên>[#<số>]`; rỗng = mọi đơn vị theo thứ tự nhỏ -> lớn.
        reuse_knowledge_graph: Nạp lại KG đã lưu của đơn vị ĐÃ XONG (dùng với `append`);
            đơn vị chưa xong luôn tự dùng lại KG còn sót từ lần lỗi trước.
        append: Chạy lại cả đơn vị đã xong, nối thêm dòng và cộng dồn progress; bắt buộc
            đi kèm `only`.
        testset_size: Ghi đè tổng số câu (mặc định `GENERATE_SIZE`); chia cho các đơn vị đã chọn.
        unit_runner: Cài đặt thay thế cho test; `None` thì dựng runner ragas thật khi cần.
        settings: Cấu hình 6 key Groq, chỉ dùng khi `unit_runner` là `None`.

    Raises:
        EvalInputError: `--only` sai, `append` thiếu `only`, thiếu key Groq, progress
            hỏng/lệch nguồn, file raw hỏng.
        UnitGenerationError: Một đơn vị lỗi giữa chừng; `last_failure` đã được ghi,
            đơn vị đó KHÔNG được ghi vào raw/progress (KG hoàn chỉnh nếu đã dựng xong thì
            được giữ), các đơn vị sau không chạy.
    """
    _require_only_with_append(only, append)
    state = _prepare_run(markdown_dir, output_dir, only, testset_size)
    _recover_unfinished(state, output_dir / PROGRESS_FILENAME)
    pending = {unit_key(u) for u in _pending_units(state, append)}
    to_run = [
        u
        for u in state.selected
        if unit_key(u) in pending and state.quotas[unit_key(u)].total > 0
    ]
    to_run_keys = {unit_key(u) for u in to_run}
    report = GenerationReport(generated_units=[], skipped_units=[], new_questions=0)
    for unit in state.selected:
        key = unit_key(unit)
        if key not in to_run_keys:
            logger.info(
                "Bỏ qua %s (%s).", key, "quota 0 câu" if key in pending else "đã xong"
            )
            report.skipped_units.append(key)
    if not to_run:
        return report

    # Dựng runner (tạo client, đọc 6 key) TRƯỚC vòng lặp và ngoài `try` của từng đơn vị:
    # thiếu key là lỗi cấu hình, không phải lỗi của một đơn vị cụ thể.
    runner = unit_runner or build_unit_runner(settings)
    for unit in to_run:
        added = _run_one_unit(
            runner, unit, state, output_dir, reuse_knowledge_graph=reuse_knowledge_graph
        )
        report.generated_units.append(unit_key(unit))
        report.new_questions += added
    return report


# ---------------------------------------------------------------------------
# Chốt đúng TARGET_SIZE câu (mục 4.2)
# ---------------------------------------------------------------------------


def finalize_testset(
    raw_cases: Sequence[GoldenTestCase], target: int = TARGET_SIZE
) -> list[GoldenTestCase]:
    """Chọn đúng `target` câu, cắt phân tầng theo (`source_document`, `synthesizer_name`).

    Mỗi dòng nhận khoá `(thứ hạng trong nhóm theo thứ tự file + 0,5) / kích thước nhóm`;
    lấy `target` dòng có khoá nhỏ nhất, hoà thì theo thứ tự file. Tất định; giữ tỷ lệ theo
    văn bản và loại câu sau khi người dùng đã xoá, không dồn vào vài văn bản đầu file.
    Kết quả giữ thứ tự file.

    Raises:
        EvalInputError: Còn ít hơn `target` câu (kèm gợi ý lệnh sinh bù).
    """
    if len(raw_cases) < target:
        shortfall = target - len(raw_cases)
        raise EvalInputError(
            f"Chỉ còn {len(raw_cases)} câu, thiếu {shortfall} câu so với đích {target}. "
            "Sinh bù rồi finalize lại: generate --only <tên văn bản> "
            f"--reuse-knowledge-graph --append --testset-size {math.ceil(shortfall * 1.3)}"
        )
    groups: dict[tuple[str | None, str | None], list[int]] = {}
    for position, case in enumerate(raw_cases):
        groups.setdefault((case.source_document, case.synthesizer_name), []).append(
            position
        )
    keys: dict[int, float] = {}
    for positions in groups.values():
        for rank, position in enumerate(positions):
            keys[position] = (rank + 0.5) / len(positions)
    chosen = sorted(sorted(keys, key=lambda p: (keys[p], p))[:target])
    return [raw_cases[position] for position in chosen]


def _load_raw_cases(path: Path) -> list[GoldenTestCase]:
    """Đọc + validate raw; lỗi nêu vị trí dòng (1-based), không âm thầm bỏ qua (mục 8)."""
    if not path.exists():
        raise EvalInputError(f"Không thấy {path}; chạy `generate` trước.")
    cases: list[GoldenTestCase] = []
    for position, row in enumerate(read_raw_rows(path), start=1):
        try:
            case = GoldenTestCase.model_validate(row)
        except ValidationError as error:
            raise EvalInputError(
                f"{path}: dòng {position} không hợp lệ: {error}"
            ) from error
        if missing := case.empty_required_fields():
            raise EvalInputError(f"{path}: dòng {position} rỗng: {', '.join(missing)}.")
        cases.append(case)
    return cases


def finalize_golden_testset(
    output_dir: Path = DEFAULT_OUTPUT_DIR, target: int = TARGET_SIZE
) -> list[GoldenTestCase]:
    """Đọc raw đã review, chốt đúng `target` câu và ghi `golden_testset.json` (mục 4.2).

    Raises:
        EvalInputError: Raw thiếu/hỏng, có dòng thiếu trường, hoặc còn ít hơn `target` câu
            (khi đó KHÔNG ghi `golden_testset.json`).
    """
    cases = finalize_testset(_load_raw_cases(output_dir / RAW_TESTSET_FILENAME), target)
    _write_text_atomic(
        output_dir / GOLDEN_TESTSET_FILENAME,
        json.dumps([c.model_dump() for c in cases], ensure_ascii=False, indent=2)
        + "\n",
    )
    return cases
