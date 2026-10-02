"""Separate malformed Judge policy output from legitimate generation refusals."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from production_legal_qa_rag.evaluation.generate_stage import run_generate
from production_legal_qa_rag.evaluation.jsonl_store import JsonlStore
from production_legal_qa_rag.evaluation.key_pool import GenerationWorker
from production_legal_qa_rag.evaluation.run_models import AnswerRecord, case_id
from production_legal_qa_rag.generation.generator import GeneratedAnswer
from production_legal_qa_rag.generation.judge import EvidenceJudge
from production_legal_qa_rag.generation.models import (
    Citation,
    JudgeIssue,
    JudgeVerdict,
    VerificationIssue,
)
from production_legal_qa_rag.retrieval.models import RetrievedChunk
from tests.test_evaluation_phase2 import make_config, seed_retrieval


def _issue(number: int | None = None) -> JudgeIssue:
    """Produce schema-valid evidence that may violate the context-bound policy."""
    return JudgeIssue(
        code="unsupported_claim",
        claim="quyền lợi",
        detail="Thiếu evidence",
        evidence_numbers=[] if number is None else [number],
    )


def _bind_provider(
    monkeypatch: pytest.MonkeyPatch,
    worker: GenerationWorker,
    verdicts: dict[str, JudgeVerdict],
    calls: list[str],
) -> None:
    """Fake provider boundaries while retaining the eval wrapper and real pipeline."""

    async def draft(query: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        return GeneratedAnswer(
            text="Người lao động có quyền lợi [1].",
            fragments=["Người lao động có quyền lợi [1]."],
            finish_reason="stop",
        )

    async def repair(
        query: str,
        chunks: list[RetrievedChunk],
        previous: str,
        issues: list[VerificationIssue],
    ) -> GeneratedAnswer:
        return await draft(query, chunks)

    async def provider(
        self: EvidenceJudge,
        query: str,
        chunks: list[RetrievedChunk],
        response: str,
        citations: list[Citation],
    ) -> JudgeVerdict:
        calls.append(query)
        return verdicts[query]

    monkeypatch.setattr(worker.generator, "draft", draft)
    monkeypatch.setattr(worker.generator, "repair", repair)
    monkeypatch.setattr(EvidenceJudge, "judge", provider)


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(JudgeVerdict(verdict="repair"), id="repair-without-issues"),
        pytest.param(
            JudgeVerdict(verdict="insufficient_evidence"),
            id="insufficient-without-issues",
        ),
        pytest.param(
            JudgeVerdict(verdict="pass", issues=[_issue()]), id="pass-with-issues"
        ),
        pytest.param(
            JudgeVerdict(verdict="repair", issues=[_issue(0)]), id="evidence-zero"
        ),
        pytest.param(
            JudgeVerdict(verdict="repair", issues=[_issue(-1)]),
            id="evidence-negative",
        ),
        pytest.param(
            JudgeVerdict(verdict="insufficient_evidence", issues=[_issue(2)]),
            id="evidence-outside-context",
        ),
    ],
)
def test_policy_invalid_judge_is_error_and_only_failed_retry_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: JudgeVerdict
) -> None:
    config, cases = make_config(tmp_path, 1)
    seed_retrieval(config, cases)
    query = cases[0].user_input
    worker = GenerationWorker("fake-policy-key")
    calls: list[str] = []
    verdicts = {query: invalid}
    _bind_provider(monkeypatch, worker, verdicts, calls)

    first = asyncio.run(run_generate(config, [worker]))
    assert (first.done, first.errors) == (0, 1)
    assert isinstance(worker.judge.last_error, ValueError)
    store = JsonlStore(config.output_dir / "answers.jsonl", AnswerRecord)
    record = store.get(case_id(query), "mmr_on")
    assert record is not None
    assert (record.outcome, record.error, record.error_code) == (
        "error",
        "ValueError",
        "ValueError",
    )
    assert record.response is None
    assert calls == [query]

    verdicts[query] = JudgeVerdict(verdict="pass")
    skipped = asyncio.run(run_generate(config, [worker]))
    assert (skipped.done, skipped.errors) == (0, 1)
    assert calls == [query]
    retried = asyncio.run(
        run_generate(config.model_copy(update={"retry_failed": True}), [worker])
    )
    assert (retried.done, retried.errors) == (1, 0)
    assert calls == [query, query]
    assert worker.judge.last_error is None
    recovered = JsonlStore(store.path, AnswerRecord).get(case_id(query), "mmr_on")
    assert recovered is not None
    assert (recovered.outcome, recovered.error) == ("answered", None)
    assert recovered.response == "Người lao động có quyền lợi [1]."


def test_valid_refusal_and_exhausted_repair_remain_successful_and_skip_failed_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cases = make_config(tmp_path, 2)
    seed_retrieval(config, cases)
    worker = GenerationWorker("fake-policy-key")
    queries = [case.user_input for case in cases]
    verdicts = {
        queries[0]: JudgeVerdict(verdict="insufficient_evidence", issues=[_issue(1)]),
        queries[1]: JudgeVerdict(verdict="repair", issues=[_issue(1)]),
    }
    calls: list[str] = []
    _bind_provider(monkeypatch, worker, verdicts, calls)

    result = asyncio.run(run_generate(config, [worker]))
    assert (result.done, result.errors) == (2, 0)
    assert worker.judge.last_error is None
    store = JsonlStore(config.output_dir / "answers.jsonl", AnswerRecord)
    refused, exhausted = [store.get(case_id(query), "mmr_on") for query in queries]
    assert refused is not None and exhausted is not None
    assert (refused.outcome, refused.error) == ("insufficient_evidence", None)
    assert (exhausted.outcome, exhausted.error) == ("unable_to_verify", None)
    assert exhausted.repair_used
    assert calls.count(queries[0]) == 1
    assert calls.count(queries[1]) == 2
    before = calls.copy()
    asyncio.run(run_generate(config, [worker]))
    asyncio.run(
        run_generate(config.model_copy(update={"retry_failed": True}), [worker])
    )
    assert calls == before
