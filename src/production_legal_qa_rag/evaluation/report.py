"""Pure aggregation of paired retrieval and verified answer evaluation scores."""

from __future__ import annotations

import functools
import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from production_legal_qa_rag.evaluation.jsonl_store import JsonlStore
from production_legal_qa_rag.evaluation.models import GoldenTestCase
from production_legal_qa_rag.evaluation.run_models import (
    CONFIGS,
    AnswerRecord,
    EvalConfig,
    HydeRecord,
    RetrievalConfig,
    RetrievalRecord,
    ScoreRecord,
    StageSummary,
    case_id,
    load_testset,
)
from production_legal_qa_rag.evaluation.testset_generator import _write_text_atomic

INTERPRETATION_NOTES = [
    "Điểm đo mức khớp với reference do LLM sinh, không phải xác nhận đúng luật tuyệt đối.",
    "Judge RAGAS cùng họ model với generator; faithfulness chỉ chấm câu đã qua Evidence Judge.",
    "Single-hop là chỉ số chính; multi-hop specific (testset chuẩn n=15) chỉ đọc như xu hướng.",
    "Testset không có multi-hop abstract; guardrail không được đo.",
]


def _mean(values: Sequence[float | None]) -> dict[str, int | float | None]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return {
        "mean": sum(finite) / len(finite) if finite else None,
        "n": len(finite),
        "null": len(values) - len(finite),
    }


def _slices(
    cases: list[GoldenTestCase], aggregate: Callable[[list[str]], Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "overall": aggregate([case_id(c.user_input) for c in cases])
    }
    for dimension in ("synthesizer_name", "source_document"):
        groups: dict[str, list[str]] = {}
        for case in cases:
            label = getattr(case, dimension) or "unknown"
            groups.setdefault(label, []).append(case_id(case.user_input))
        result[dimension] = {
            label: aggregate(identifiers) for label, identifiers in groups.items()
        }
    return result


def build_report(
    cases: list[GoldenTestCase],
    recall: Sequence[ScoreRecord],
    precision: Sequence[ScoreRecord],
    answer_scores: Sequence[ScoreRecord],
    answers: Sequence[AnswerRecord],
    hyde: Sequence[HydeRecord] = (),
    retrieval: Sequence[RetrievalRecord] = (),
) -> dict[str, Any]:
    """Aggregate latest records; never compare unpaired recall or count NaN as zero."""
    recall_by = {(r.case_id, r.config): r for r in recall}
    precision_by = {(r.case_id, r.config): r for r in precision}
    scores_by = {(r.case_id, r.config): r for r in answer_scores}
    answers_by = {(r.case_id, r.config): r for r in answers}
    ids = {case_id(c.user_input) for c in cases}

    def paired(identifiers: list[str]) -> dict[str, Any]:
        pairs: list[tuple[ScoreRecord, ScoreRecord]] = []
        for identifier in identifiers:
            left, right = (
                recall_by.get((identifier, "mmr_on")),
                recall_by.get((identifier, "mmr_off")),
            )
            if (
                left is not None
                and right is not None
                and left.error is None
                and right.error is None
                and all(
                    (value := r.scores.get("context_recall")) is not None
                    and math.isfinite(value)
                    for r in (left, right)
                )
            ):
                pairs.append((left, right))
        on = [r.scores["context_recall"] for r, _ in pairs]
        off = [r.scores["context_recall"] for _, r in pairs]
        deltas = [
            left - right
            for left, right in zip(on, off, strict=True)
            if left is not None and right is not None
        ]
        return {
            "mmr_on": _mean(on),
            "mmr_off": _mean(off),
            "wins": sum(d > 0 for d in deltas),
            "losses": sum(d < 0 for d in deltas),
            "ties": sum(d == 0 for d in deltas),
            "identical_chunks": sum(
                left.reused_from is not None or right.reused_from is not None
                for left, right in pairs
            ),
            "paired": len(pairs),
            "excluded": len(identifiers) - len(pairs),
        }

    def score_slices(
        mapping: dict[tuple[str, RetrievalConfig], ScoreRecord],
        choice: RetrievalConfig,
        metric: str,
    ) -> dict[str, Any]:
        return _slices(
            cases,
            lambda identifiers: _mean(
                [
                    row.scores.get(metric) if row is not None else None
                    for identifier in identifiers
                    for row in [mapping.get((identifier, choice))]
                ]
            ),
        )

    def answer_group(identifiers: list[str], choice: RetrievalConfig) -> dict[str, Any]:
        rows = [
            answers_by[(identifier, choice)]
            for identifier in identifiers
            if (identifier, choice) in answers_by
        ]
        outcomes = Counter(row.outcome for row in rows)
        conditional: dict[str, Any] = {}
        end_to_end: dict[str, Any] = {}
        for metric in ("faithfulness", "answer_relevancy"):
            answered_values: list[float | None] = []
            all_values: list[float | None] = []
            for row in rows:
                if row.outcome == "error":
                    continue
                if row.outcome == "answered":
                    score = scores_by.get((row.case_id, choice))
                    value = score.scores.get(metric) if score is not None else None
                    answered_values.append(value)
                    all_values.append(value)
                else:
                    all_values.append(0.0)
            conditional[metric] = _mean(answered_values)
            end_to_end[metric] = _mean(all_values)
        count = len(rows)
        refusals = outcomes["insufficient_evidence"] + outcomes["unable_to_verify"]
        return {
            "n": count,
            "pending": len(identifiers) - count,
            "outcomes": {
                name: {
                    "n": outcomes[name],
                    "rate": outcomes[name] / count if count else None,
                }
                for name in (
                    "answered",
                    "insufficient_evidence",
                    "unable_to_verify",
                    "error",
                )
            },
            "refusal_rate": refusals / count if count else None,
            "repair_rate": sum(row.repair_used for row in rows) / count
            if count
            else None,
            "conditional": conditional,
            "end_to_end": end_to_end,
            "excluded_errors": outcomes["error"],
        }

    return {
        "testset_size": len(cases),
        "retrieval_comparison": _slices(cases, paired),
        "context_precision": {
            choice: score_slices(precision_by, choice, "context_precision")
            for choice in CONFIGS
        },
        "answers": {
            choice: _slices(
                cases,
                functools.partial(answer_group, choice=choice),
            )
            for choice in CONFIGS
        },
        "operations": {
            "hyde_null": sum(
                row.case_id in ids
                and row.error is None
                and row.hypothetical_document is None
                for row in hyde
            ),
            "hyde_errors": sum(
                row.case_id in ids and row.error is not None for row in hyde
            ),
            "retrieval_errors": {
                choice: sum(
                    row.case_id in ids
                    and row.config == choice
                    and row.error is not None
                    for row in retrieval
                )
                for choice in CONFIGS
            },
            "rerank_fallback": sum(
                row.case_id in ids and row.error == "rerank_fallback"
                for row in retrieval
            ),
            "prompt_versions": sorted(
                {row.prompt_version for row in answers if row.case_id in ids}
            ),
        },
        "interpretation_notes": INTERPRETATION_NOTES,
    }


def write_report(config: EvalConfig) -> dict[str, Any]:
    """Read stage stores and atomically save report.json without external calls."""
    cases = load_testset(config)

    def records(filename: str, model: Any) -> list[Any]:
        return list(JsonlStore(config.output_dir / filename, model).records.values())

    report = build_report(
        cases,
        records("recall_scores.jsonl", ScoreRecord),
        records("precision_scores.jsonl", ScoreRecord),
        records("answer_scores.jsonl", ScoreRecord),
        records("answers.jsonl", AnswerRecord),
        records("hyde.jsonl", HydeRecord),
        [
            row
            for choice in CONFIGS
            for row in records(f"retrieved_{choice}.jsonl", RetrievalRecord)
        ],
    )
    _write_text_atomic(
        config.output_dir / "report.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return report


def stage_status(config: EvalConfig) -> dict[str, StageSummary]:
    """Count latest checkpoints within the selected testset and configurations."""
    from production_legal_qa_rag.evaluation.run_models import EmbeddingRecord

    identifiers = [case_id(case.user_input) for case in load_testset(config)]
    result: dict[str, StageSummary] = {}
    for name, filename, model in (
        ("hyde", "hyde.jsonl", HydeRecord),
        ("embed", "embeddings.jsonl", EmbeddingRecord),
    ):
        result[name] = JsonlStore(config.output_dir / filename, model).summary(
            identifiers
        )
    for choice in CONFIGS:
        for name, filename, configured_model in (
            ("retrieve", f"retrieved_{choice}.jsonl", RetrievalRecord),
            ("score-recall", "recall_scores.jsonl", ScoreRecord),
            ("generate", "answers.jsonl", AnswerRecord),
            ("score-answers", "answer_scores.jsonl", ScoreRecord),
            ("score-precision", "precision_scores.jsonl", ScoreRecord),
        ):
            store = JsonlStore(config.output_dir / filename, configured_model)
            eligible = identifiers
            if name == "score-answers":
                answers = JsonlStore(config.output_dir / "answers.jsonl", AnswerRecord)
                eligible = [
                    identifier
                    for identifier in identifiers
                    if (answer := answers.get(identifier, choice)) is None
                    or answer.outcome in {"answered", "error"}
                ]
            result[f"{name}:{choice}"] = store.summary(eligible, choice)
    return result
