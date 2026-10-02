"""Protect RAGAS score scheduling and checkpoint semantics with offline evaluators."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("ragas", reason="Requires the isolated eval dependency group")

from production_legal_qa_rag.evaluation import scoring
from production_legal_qa_rag.evaluation.groq_round_robin import (
    DailyQuotaExhaustedError,
)
from production_legal_qa_rag.evaluation.jsonl_store import JsonlStore
from production_legal_qa_rag.evaluation.run_models import (
    AnswerRecord,
    RetrievalRecord,
    ScoreRecord,
    case_id,
)
from production_legal_qa_rag.generation.generator import build_context
from production_legal_qa_rag.generation.models import Citation
from tests.test_evaluation_phase2 import (
    make_chunk,
    make_config,
    seed_retrieval,
)


@pytest.mark.parametrize("same_order,expected_calls", [(True, 1), (False, 2)])
def test_recall_reuses_only_identical_ordered_chunk_ids(
    tmp_path: Path, same_order: bool, expected_calls: int
) -> None:
    config, cases = make_config(tmp_path, 1)
    identifier = case_id(cases[0].user_input)
    chunks = [make_chunk("c1"), make_chunk("c2")]
    for choice in ("mmr_on", "mmr_off"):
        ordered = chunks if choice == "mmr_on" or same_order else list(reversed(chunks))
        JsonlStore(
            config.output_dir / f"retrieved_{choice}.jsonl", RetrievalRecord
        ).append(RetrievalRecord(case_id=identifier, config=choice, chunks=ordered))
    calls: list[list[dict[str, Any]]] = []

    def evaluate(
        rows: list[dict[str, Any]], names: list[str]
    ) -> list[dict[str, float]]:
        calls.append(rows)
        assert names == ["context_recall"]
        assert rows[0]["retrieved_contexts"][0] == build_context(
            [chunks[0] if len(calls) == 1 or same_order else chunks[1]]
        ).removeprefix("[1] ")
        return [{"context_recall": 0.7}]

    result = scoring.run_scoring(config, "recall", evaluate)
    assert all(summary.done == 1 for summary in result.values())
    assert len(calls) == expected_calls
    store = JsonlStore(config.output_dir / "recall_scores.jsonl", ScoreRecord)
    off = store.get(identifier, "mmr_off")
    assert off.reused_from == ("mmr_on" if same_order else None)
    assert off.scores == {"context_recall": 0.7}


def test_nan_is_error_retry_is_opt_in_and_scoring_batches_at_ten(
    tmp_path: Path,
) -> None:
    config, cases = make_config(tmp_path, 12)
    seed_retrieval(config, cases)
    sizes: list[int] = []

    def evaluate(
        rows: list[dict[str, Any]], names: list[str]
    ) -> list[dict[str, float]]:
        sizes.append(len(rows))
        return [
            {
                "context_recall": float("nan")
                if row["user_input"] == cases[0].user_input
                else 0.8
            }
            for row in rows
        ]

    result = scoring.run_scoring(config, "recall", evaluate)
    assert sizes == [10, 2, 1]
    assert result["mmr_on"].errors == result["mmr_off"].errors == 1
    before = sizes.copy()
    scoring.run_scoring(config, "recall", evaluate)
    assert sizes == before

    def healthy(rows: list[dict[str, Any]], names: list[str]) -> list[dict[str, float]]:
        return [{"context_recall": 0.9} for _ in rows]

    retried = scoring.run_scoring(
        config.model_copy(update={"retry_failed": True}), "recall", healthy
    )
    assert all(
        summary.done == 12 and summary.errors == 0 for summary in retried.values()
    )


@pytest.mark.parametrize("daily", [False, True])
def test_scoring_unexpected_error_records_failure_daily_leaves_pending(
    tmp_path: Path, daily: bool
) -> None:
    config, cases = make_config(tmp_path, 2)
    seed_retrieval(config, cases)

    def evaluate(
        rows: list[dict[str, Any]], names: list[str]
    ) -> list[dict[str, float]]:
        if daily:
            raise DailyQuotaExhaustedError("fake daily")
        raise TimeoutError("private provider text")

    result = scoring.run_scoring(config, "recall", evaluate)
    store = JsonlStore(config.output_dir / "recall_scores.jsonl", ScoreRecord)
    if daily:
        assert not store.records
        assert all(
            summary.pending == 2 and summary.quota_exhausted
            for summary in result.values()
        )
    else:
        assert all(summary.errors == 2 for summary in result.values())
        assert {row.error for row in store.records.values()} == {"TimeoutError"}
        assert "private provider" not in store.path.read_text()


def test_quota_breaker_keeps_finite_partial_results_and_nan_pending(
    tmp_path: Path,
) -> None:
    config, cases = make_config(tmp_path, 2)
    seed_retrieval(config, cases)

    def evaluate(
        rows: list[dict[str, Any]], names: list[str]
    ) -> list[dict[str, float]]:
        return [{"context_recall": 0.8}, {"context_recall": float("nan")}]

    result = scoring.run_scoring(config, "recall", evaluate, quota_check=lambda: True)
    on = result["mmr_on"]
    assert (on.done, on.errors, on.pending, on.quota_exhausted) == (1, 0, 1, True)
    store = JsonlStore(config.output_dir / "recall_scores.jsonl", ScoreRecord)
    assert store.get(case_id(cases[1].user_input), "mmr_on") is None
    assert result["mmr_off"].done == 1
    assert result["mmr_off"].pending == 1


def test_answer_scoring_strips_only_valid_citations_skips_refusal_and_keeps_raw(
    tmp_path: Path,
) -> None:
    config, cases = make_config(tmp_path, 3)
    seed_retrieval(config, cases)
    answers = JsonlStore(config.output_dir / "answers.jsonl", AnswerRecord)
    raw = "Quyền lợi [1], luật ghi [99], thêm [1]."
    for i, case in enumerate(cases):
        answers.append(
            AnswerRecord(
                case_id=case_id(case.user_input),
                config="mmr_on",
                outcome="answered"
                if i == 0
                else "insufficient_evidence"
                if i == 1
                else "error",
                error="timeout" if i == 2 else None,
                response=raw if i == 0 else None,
                citations=[
                    Citation(
                        n=1,
                        chunk_id="c1",
                        source_document="Luật.md",
                        breadcrumb="Điều 1",
                    )
                ],
                prompt_version="v11",
            )
        )
    captured: list[dict[str, Any]] = []

    def evaluate(
        rows: list[dict[str, Any]], names: list[str]
    ) -> list[dict[str, float]]:
        assert names == ["faithfulness", "answer_relevancy"]
        captured.extend(rows)
        return [{"faithfulness": 1.0, "answer_relevancy": 0.8} for _ in rows]

    summary = scoring.run_scoring(config, "answers", evaluate)["mmr_on"]
    assert (summary.done, summary.pending, summary.missing_upstream) == (1, 1, 1)
    assert len(captured) == 1
    assert captured[0]["response"] == "Quyền lợi , luật ghi [99], thêm ."
    assert (
        JsonlStore(answers.path, AnswerRecord)
        .get(case_id(cases[0].user_input), "mmr_on")
        .response
        == raw
    )


def test_ragas_uses_standard_metrics_strictness_three_segmented_embeddings_and_no_nested_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = scoring.RagasScorer.__new__(scoring.RagasScorer)
    scorer.llm = SimpleNamespace()
    scorer.run_config = SimpleNamespace(max_workers=3)
    scorer._embedding = None
    captured: dict[str, Any] = {}

    class Metric:
        def __init__(self, llm: Any) -> None:
            self.llm = llm

    for name in (
        "Faithfulness",
        "ResponseRelevancy",
        "LLMContextRecall",
        "LLMContextPrecisionWithReference",
    ):
        monkeypatch.setattr(scoring, name, Metric)

    def adapter(*, segment: bool) -> Any:
        captured["segment"] = segment
        return "segmented-adapter"

    def evaluate(dataset: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(scores=[{"faithfulness": 1.0, "answer_relevancy": 0.8}])

    monkeypatch.setattr(scoring, "RagasEmbeddingsAdapter", adapter)
    monkeypatch.setattr(scoring, "LangchainEmbeddingsWrapper", lambda value: value)
    monkeypatch.setattr(scoring, "evaluate", evaluate)
    rows = [
        {
            "user_input": "q",
            "response": "r",
            "retrieved_contexts": ["c"],
            "reference": "ref",
        }
    ]
    assert (
        scorer.evaluate_batch(rows, ["faithfulness", "answer_relevancy"])[0][
            "faithfulness"
        ]
        == 1.0
    )
    assert captured["segment"] is True
    assert captured["metrics"][1].strictness == 3
    assert captured["metrics"][1].embeddings == "segmented-adapter"
    assert captured["raise_exceptions"] is False
    assert captured["allow_nest_asyncio"] is False
