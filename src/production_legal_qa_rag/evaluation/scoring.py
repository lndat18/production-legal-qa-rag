"""RAGAS-only boundary for recall, answer quality and reranker precision."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any, Literal

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from production_legal_qa_rag.evaluation.embeddings_adapter import RagasEmbeddingsAdapter
from production_legal_qa_rag.evaluation.groq_round_robin import GroqRoundRobinChatModel
from production_legal_qa_rag.evaluation.jsonl_store import JsonlStore
from production_legal_qa_rag.evaluation.key_pool import (
    api_keys,
    error_code,
    is_daily_quota,
)
from production_legal_qa_rag.evaluation.models import GoldenTestCase
from production_legal_qa_rag.evaluation.ragas_runner import build_run_config
from production_legal_qa_rag.evaluation.run_models import (
    CONFIGS,
    SCORING_BATCH_SIZE,
    AnswerRecord,
    EvalConfig,
    RetrievalConfig,
    RetrievalRecord,
    ScoreRecord,
    StageSummary,
    case_id,
    load_testset,
)
from production_legal_qa_rag.evaluation.testset_generator import EvalInputError
from production_legal_qa_rag.generation.generator import build_context
from production_legal_qa_rag.retrieval.models import RetrievedChunk

type ScoreStage = Literal["recall", "answers", "precision"]
type EvaluateBatch = Callable[[list[dict[str, Any]], list[str]], list[dict[str, float]]]


def retrieved_contexts(chunks: list[RetrievedChunk]) -> list[str]:
    """Use exact generation chunk text while retaining rerank order for metrics."""
    return [build_context([chunk]).removeprefix("[1] ") for chunk in chunks]


def strip_citations(answer: AnswerRecord) -> str:
    """Remove only markers belonging to the answer's validated citation map."""
    valid = {citation.n for citation in answer.citations}
    return re.sub(
        r"\[(\d+)\]",
        lambda match: "" if int(match[1]) in valid else match[0],
        answer.response or "",
    )


def _finite_score(value: object) -> float | None:
    """Treat malformed or non-finite metric output as a retriable failure."""
    try:
        score = float(value)  # type: ignore[arg-type]
    except TypeError, ValueError, OverflowError:
        return None
    return score if math.isfinite(score) else None


class RagasScorer:
    """Reuse Phase 1 wrapper/run config, with no SDK or RAGAS retry multiplier."""

    def __init__(self, workers: int) -> None:
        clients = [
            ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=key,
                model="openai/gpt-oss-120b",
                timeout=60.0,
                max_retries=0,
            )
            for key in api_keys()
        ]
        self.router = GroqRoundRobinChatModel(clients=clients)
        self.run_config = build_run_config()
        self.run_config.max_workers = workers
        self.run_config.timeout = 600
        self.llm = LangchainLLMWrapper(self.router, run_config=self.run_config)
        self._embedding: Any = None

    def evaluate_batch(
        self, rows: list[dict[str, Any]], names: list[str]
    ) -> list[dict[str, float]]:
        """Evaluate at most ten samples; RAGAS NaN is handled by checkpoint code."""
        metrics: list[Any] = []
        constructors = {
            "context_recall": LLMContextRecall,
            "context_precision": LLMContextPrecisionWithReference,
            "faithfulness": Faithfulness,
            "answer_relevancy": ResponseRelevancy,
        }
        for name in names:
            metric = constructors[name](llm=self.llm)
            # RAGAS đặt tên cột theo class (precision → llm_context_precision_with_reference).
            metric.name = name
            if name == "answer_relevancy":
                if self._embedding is None:
                    self._embedding = LangchainEmbeddingsWrapper(
                        RagasEmbeddingsAdapter(segment=True)
                    )
                metric.embeddings = self._embedding
                metric.strictness = 3
            metrics.append(metric)
        result = evaluate(
            EvaluationDataset.from_list(rows),
            metrics=metrics,
            llm=self.llm,
            run_config=self.run_config,
            raise_exceptions=False,
            show_progress=False,
            allow_nest_asyncio=False,
        )
        return result.scores  # type: ignore[union-attr]


def run_scoring(
    config: EvalConfig,
    stage: ScoreStage,
    evaluator: EvaluateBatch | None = None,
    quota_check: Callable[[], bool] | None = None,
) -> dict[str, StageSummary]:
    """Score available successes; daily-exhausted/unfinished rows stay pending."""
    if stage != "recall" and config.config is None:
        raise EvalInputError("--config bắt buộc cho score-answers/score-precision.")
    cases = load_testset(config)
    filename = {
        "recall": "recall_scores.jsonl",
        "answers": "answer_scores.jsonl",
        "precision": "precision_scores.jsonl",
    }[stage]
    store = JsonlStore(config.output_dir / filename, ScoreRecord)
    answers = JsonlStore(config.output_dir / "answers.jsonl", AnswerRecord)
    names = {
        "recall": ["context_recall"],
        "precision": ["context_precision"],
        "answers": ["faithfulness", "answer_relevancy"],
    }[stage]
    choices = CONFIGS if stage == "recall" else (config.config,)
    result: dict[str, StageSummary] = {}
    exhausted = False
    for choice in choices:
        assert choice is not None
        retrieval = JsonlStore(
            config.output_dir / f"retrieved_{choice}.jsonl", RetrievalRecord
        )
        jobs, missing = _prepare_jobs(
            cases, config, stage, choice, retrieval, answers, store
        )
        for offset in range(0, len(jobs), SCORING_BATCH_SIZE):
            if exhausted:
                break
            batch = jobs[offset : offset + SCORING_BATCH_SIZE]
            if evaluator is None:
                scorer = RagasScorer(config.workers)
                evaluator = scorer.evaluate_batch
                quota_check = lambda scorer=scorer: scorer.router.daily_quota_exhausted
            try:
                scored = evaluator([row for _, row in batch], names)
                if len(scored) != len(batch):
                    raise ValueError("Wrong RAGAS result count")
            except Exception as error:  # noqa: BLE001 - checkpoint unexpected external failures.
                if is_daily_quota(error):
                    exhausted = True
                    break
                scored = [{} for _ in batch]
                failure = error_code(error)
            else:
                failure = "non_finite_score"
            exhausted = bool(quota_check and quota_check())
            for (identifier, _), values in zip(batch, scored, strict=True):
                scores = {name: _finite_score(values.get(name)) for name in names}
                incomplete = any(value is None for value in scores.values())
                if exhausted and incomplete:
                    continue
                store.append(
                    ScoreRecord(
                        case_id=identifier,
                        config=choice,
                        scores=scores,
                        error=failure if incomplete else None,
                    )
                )
        result[choice] = store.summary(
            [
                case_id(c.user_input)
                for c in cases
                if stage != "answers"
                or (answer := answers.get(case_id(c.user_input), choice)) is None
                or answer.outcome in {"answered", "error"}
            ],
            choice,
            missing_upstream=missing,
            quota_exhausted=exhausted,
        )
    return result


def _prepare_jobs(
    cases: list[GoldenTestCase],
    config: EvalConfig,
    stage: ScoreStage,
    choice: RetrievalConfig,
    retrieval: JsonlStore[RetrievalRecord],
    answers: JsonlStore[AnswerRecord],
    store: JsonlStore[ScoreRecord],
) -> tuple[list[tuple[str, dict[str, Any]]], int]:
    jobs: list[tuple[str, dict[str, Any]]] = []
    missing = 0
    other: RetrievalConfig = "mmr_off" if choice == "mmr_on" else "mmr_on"
    other_retrieval = JsonlStore(
        config.output_dir / f"retrieved_{other}.jsonl", RetrievalRecord
    )
    for case in cases:
        identifier = case_id(case.user_input)
        if not store.should_run(
            identifier, config=choice, retry_failed=config.retry_failed
        ):
            continue
        row = retrieval.get(identifier, choice)
        if row is None or row.error is not None or not row.chunks:
            missing += 1
            continue
        sample: dict[str, Any] = {
            "user_input": case.user_input,
            "reference": case.reference,
            "retrieved_contexts": retrieved_contexts(row.chunks),
        }
        if stage == "answers":
            answer = answers.get(identifier, choice)
            if answer is None or answer.outcome == "error":
                missing += 1
                continue
            if answer.outcome != "answered":
                continue
            sample["response"] = strip_citations(answer)
        elif stage == "recall":
            previous = store.get(identifier, other)
            other_row = other_retrieval.get(identifier, other)
            if (
                previous is not None
                and previous.error is None
                and other_row is not None
                and other_row.error is None
                and [c.chunk_id for c in row.chunks]
                == [c.chunk_id for c in other_row.chunks]
            ):
                store.append(
                    ScoreRecord(
                        case_id=identifier,
                        config=choice,
                        scores=previous.scores,
                        reused_from=other,
                    )
                )
                continue
        jobs.append((identifier, sample))
    return jobs, missing
