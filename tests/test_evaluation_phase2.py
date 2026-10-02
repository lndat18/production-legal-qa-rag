"""Protect Phase 2 contracts, durable stages and production generation semantics."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from production_legal_qa_rag.config import JudgeSettings
from production_legal_qa_rag.evaluation.embed_stage import run_embed
from production_legal_qa_rag.evaluation.generate_stage import run_generate
from production_legal_qa_rag.evaluation.groq_round_robin import DailyQuotaExhaustedError
from production_legal_qa_rag.evaluation.hyde_stage import run_hyde
from production_legal_qa_rag.evaluation.jsonl_store import JsonlStore
from production_legal_qa_rag.evaluation.key_pool import (
    EvalEvidenceJudge,
    EvalRateLimitError,
    GenerationWorker,
    run_key_queue,
)
from production_legal_qa_rag.evaluation.models import GoldenTestCase
from production_legal_qa_rag.evaluation.report import build_report, stage_status
from production_legal_qa_rag.evaluation.retrieve_stage import run_retrieve
from production_legal_qa_rag.evaluation.run_models import (
    AnswerRecord,
    EmbeddingRecord,
    EvalConfig,
    HydeRecord,
    RetrievalRecord,
    ScoreRecord,
    case_id,
    load_testset,
)
from production_legal_qa_rag.evaluation.testset_generator import EvalInputError
from production_legal_qa_rag.generation.generator import GeneratedAnswer
from production_legal_qa_rag.generation.judge import EvidenceJudge, JudgeError
from production_legal_qa_rag.generation.models import (
    JudgeIssue,
    JudgeVerdict,
    Usage,
)
from production_legal_qa_rag.retrieval.llm_throttle import ThrottleTimeout
from production_legal_qa_rag.retrieval.models import PrecomputedQuery, RetrievedChunk


def make_cases(count: int) -> list[GoldenTestCase]:
    """Build unique legal questions without loading the checked-in real corpus."""
    return [
        GoldenTestCase(
            user_input=f"Quyền của người lao động trường hợp {i}?",
            reference="Người lao động có quyền lợi.",
            reference_contexts=["Quyền lợi theo pháp luật."],
            synthesizer_name="single_hop_specific_query_synthesizer"
            if i % 2 == 0
            else "multi_hop_specific_query_synthesizer",
            source_document="A.md" if i % 2 == 0 else "B.md",
        )
        for i in range(count)
    ]


def make_config(
    tmp_path: Path, count: int = 4
) -> tuple[EvalConfig, list[GoldenTestCase]]:
    """Create a testset and independent checkpoint directory."""
    cases = make_cases(count)
    testset = tmp_path / "testset.json"
    testset.write_text(json.dumps([c.model_dump() for c in cases]), encoding="utf-8")
    return EvalConfig(
        testset=testset, output_dir=tmp_path / "out", config="mmr_on"
    ), cases


def make_chunk(identifier: str = "c1", score: float | None = 1.0) -> RetrievedChunk:
    """Return one ranked chunk suitable for the unchanged hard gate."""
    return RetrievedChunk(
        chunk_id=identifier,
        source_document="Luật.md",
        breadcrumb="Điều 1",
        content="Người lao động có quyền lợi.",
        rerank_score=score,
    )


def seed_retrieval(config: EvalConfig, cases: list[GoldenTestCase]) -> None:
    """Prepare both configurations with the same ordered context."""
    for choice in ("mmr_on", "mmr_off"):
        store = JsonlStore(
            config.output_dir / f"retrieved_{choice}.jsonl", RetrievalRecord
        )
        for case in cases:
            store.append(
                RetrievalRecord(
                    case_id=case_id(case.user_input),
                    config=choice,
                    chunks=[make_chunk()],
                )
            )


def test_case_identity_uses_unchanged_utf8_and_testset_prefix(tmp_path: Path) -> None:
    config, cases = make_config(tmp_path)
    query = "  Quyền lợi tiếng Việt?  "
    assert case_id(query) == hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
    assert case_id(query) != case_id(query.strip())
    assert load_testset(config.model_copy(update={"limit": 2})) == cases[:2]


@pytest.mark.parametrize(
    "problem",
    ["duplicate", "empty_question", "empty_reference", "empty_context", "object"],
)
def test_testset_rejects_invalid_input_before_clients(
    tmp_path: Path, problem: str
) -> None:
    config, cases = make_config(tmp_path)
    rows: Any = [c.model_dump() for c in cases]
    if problem == "duplicate":
        rows[1] = rows[0]
    elif problem == "object":
        rows = {"rows": rows}
    else:
        field = {
            "empty_question": "user_input",
            "empty_reference": "reference",
            "empty_context": "reference_contexts",
        }[problem]
        rows[0][field] = [] if field == "reference_contexts" else "  "
    config.testset.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(EvalInputError):
        load_testset(config)


@pytest.mark.parametrize(
    "values", [{"limit": 0}, {"workers": 0}, {"workers": 10}, {"config": "other"}]
)
def test_eval_scope_validation(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EvalConfig(**values)


def test_jsonl_repairs_crash_tail_retry_latest_and_config_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scores.jsonl"
    old = ScoreRecord(case_id="0123456789ab", config="mmr_on", error="timeout")
    path.write_bytes(old.model_dump_json().encode() + b'\n{"case_id":')
    store = JsonlStore(path, ScoreRecord)
    assert not store.should_run(old.case_id, config="mmr_on")
    assert store.should_run(old.case_id, config="mmr_on", retry_failed=True)
    success = old.model_copy(update={"error": None, "scores": {"context_recall": 0.8}})
    store.append(success)
    store.append(success.model_copy(update={"config": "mmr_off"}))
    reread = JsonlStore(path, ScoreRecord)
    assert len(path.read_text().splitlines()) == 3
    assert reread.get(old.case_id, "mmr_on") == success
    assert not reread.should_run(old.case_id, config="mmr_on", retry_failed=True)
    assert reread.summary([old.case_id, "111111111111"], "mmr_off").model_dump() == {
        "done": 1,
        "errors": 0,
        "pending": 1,
        "missing_upstream": 0,
        "quota_exhausted": False,
    }


def test_jsonl_valid_unterminated_line_gets_separator(tmp_path: Path) -> None:
    path = tmp_path / "hyde.jsonl"
    first = HydeRecord(case_id="0123456789ab", hypothetical_document=None)
    second = HydeRecord(case_id="111111111111", error="TimeoutError")
    path.write_text(first.model_dump_json(), encoding="utf-8")
    JsonlStore(path, HydeRecord).append(second)
    assert list(JsonlStore(path, HydeRecord).records.values()) == [first, second]


def test_jsonl_rejects_corrupt_middle_and_nan_schema(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    row = ScoreRecord(case_id="0123456789ab", config="mmr_on")
    path.write_text("broken\n" + row.model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(EvalInputError, match="dòng 1"):
        JsonlStore(path, ScoreRecord)
    with pytest.raises(ValidationError):
        ScoreRecord(
            case_id="0123456789ab",
            config="mmr_on",
            scores={"context_recall": float("nan")},
        )


def test_shared_key_queue_reassigns_depleted_case_and_stops_when_all_depleted() -> None:
    received: list[int] = []

    async def process(worker: str, item: int) -> None:
        if worker == "depleted":
            await asyncio.sleep(0)
            raise DailyQuotaExhaustedError("fake daily quota")
        received.append(item)
        await asyncio.sleep(0)

    assert not asyncio.run(run_key_queue([0, 1, 2], ["depleted", "healthy"], process))
    assert sorted(received) == [0, 1, 2]
    assert asyncio.run(run_key_queue([3, 4], ["depleted"], process))
    assert received == received[:3]


def test_hyde_null_is_done_failure_is_retryable_daily_is_pending(
    tmp_path: Path,
) -> None:
    config, cases = make_config(tmp_path, 3)

    class Worker:
        async def generate(self, query: str) -> str | None:
            if query == cases[1].user_input:
                raise TimeoutError("provider payload must not be written")
            if query == cases[2].user_input:
                raise DailyQuotaExhaustedError("fake daily quota")
            return None

    summary = asyncio.run(run_hyde(config, [Worker()]))
    assert (summary.done, summary.errors, summary.pending, summary.quota_exhausted) == (
        1,
        1,
        1,
        True,
    )
    store = JsonlStore(config.output_dir / "hyde.jsonl", HydeRecord)
    assert store.get(case_id(cases[0].user_input)).error is None
    assert store.get(case_id(cases[1].user_input)).error == "TimeoutError"
    assert "provider payload" not in store.path.read_text()
    assert store.get(case_id(cases[2].user_input)) is None

    class Healthy:
        async def generate(self, query: str) -> str:
            return "Giả định"

    retried = asyncio.run(
        run_hyde(config.model_copy(update={"retry_failed": True}), [Healthy()])
    )
    assert (retried.done, retried.errors, retried.pending) == (3, 0, 0)
    assert (
        JsonlStore(store.path, HydeRecord)
        .get(case_id(cases[0].user_input))
        .hypothetical_document
        is None
    )


def test_embedding_batches_25_preserve_hypothesis_query_order_and_resume(
    tmp_path: Path,
) -> None:
    config, cases = make_config(tmp_path, 26)
    hyde = JsonlStore(config.output_dir / "hyde.jsonl", HydeRecord)
    expected: list[str] = []
    for i, case in enumerate(cases):
        hypo = f"hypo-{i}" if i % 2 == 0 else None
        hyde.append(
            HydeRecord(case_id=case_id(case.user_input), hypothetical_document=hypo)
        )
        expected.extend(
            [hypo, case.user_input] if hypo is not None else [case.user_input]
        )

    class Embedder:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls.append(texts)
            return [[float(expected.index(text)), 1.0] for text in texts]

    embedder = Embedder()
    summary = asyncio.run(run_embed(config, embedder))
    assert summary.done == 26
    assert [text for batch in embedder.calls for text in batch] == expected
    assert max(map(len, embedder.calls)) <= 25
    rows = JsonlStore(config.output_dir / "embeddings.jsonl", EmbeddingRecord)
    for i, case in enumerate(cases):
        precomputed = rows.get(case_id(case.user_input)).precomputed
        assert precomputed.query_embedding == [
            float(expected.index(case.user_input)),
            1.0,
        ]
        assert precomputed.hypothetical_embedding == (
            [float(expected.index(f"hypo-{i}")), 1.0] if i % 2 == 0 else None
        )
    before = len(embedder.calls)
    assert asyncio.run(run_embed(config, embedder)).done == 26
    assert len(embedder.calls) == before


@pytest.mark.parametrize("failure", ["empty", "fallback", "exception"])
def test_retrieval_is_sequential_both_configs_and_fallback_is_error(
    tmp_path: Path, failure: str
) -> None:
    config, cases = make_config(tmp_path, 2)
    embeddings = JsonlStore(config.output_dir / "embeddings.jsonl", EmbeddingRecord)
    for case in cases:
        embeddings.append(
            EmbeddingRecord(
                case_id=case_id(case.user_input),
                precomputed=PrecomputedQuery(query_embedding=[1.0, 0.0]),
            )
        )
    calls: list[tuple[str, bool]] = []

    class Pipeline:
        async def retrieve(
            self, query: str, *, use_mmr: bool, precomputed: PrecomputedQuery
        ) -> list[RetrievedChunk]:
            calls.append((query, use_mmr))
            assert precomputed.query_embedding == [1.0, 0.0]
            if failure == "exception":
                raise TimeoutError("private provider message")
            return [] if failure == "empty" else [make_chunk(score=None)]

    summaries = asyncio.run(run_retrieve(config, Pipeline()))
    assert calls == [
        (case.user_input, choice) for choice in (True, False) for case in cases
    ]
    assert all(s.errors == 2 and s.done == 0 for s in summaries.values())
    expected = {
        "empty": "no_context",
        "fallback": "rerank_fallback",
        "exception": "TimeoutError",
    }[failure]
    for choice in ("mmr_on", "mmr_off"):
        rows = JsonlStore(
            config.output_dir / f"retrieved_{choice}.jsonl", RetrievalRecord
        )
        assert all(row.error == expected for row in rows.records.values())


def test_generation_actual_pipeline_distinguishes_refusal_exhaustion_and_parse_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cases = make_config(tmp_path)
    seed_retrieval(config, cases)
    worker = GenerationWorker("fake-generation-key")
    calls: list[str] = []

    async def draft(query: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        return GeneratedAnswer(
            text="Người lao động có quyền lợi [1].",
            fragments=["Người lao động có quyền lợi [1]."],
            finish_reason="stop",
            usage=Usage(prompt_tokens=8, completion_tokens=6),
        )

    async def judge(
        query: str, chunks: list[RetrievedChunk], answer: str, citations: list[Any]
    ) -> JudgeVerdict:
        calls.append(query)
        if query == cases[1].user_input:
            return JudgeVerdict(
                verdict="insufficient_evidence",
                issues=[
                    JudgeIssue(
                        code="context_insufficient", claim="", detail="Thiếu evidence"
                    )
                ],
            )
        if query == cases[2].user_input:
            return JudgeVerdict(
                verdict="repair",
                issues=[
                    JudgeIssue(
                        code="unsupported_claim",
                        claim="quyền lợi",
                        detail="Không đủ evidence",
                    )
                ],
            )
        if query == cases[3].user_input:
            error = ValueError("private malformed judge reply")
            worker.judge.last_error = error
            raise error
        return JudgeVerdict(verdict="pass")

    async def repair(
        query: str, chunks: list[RetrievedChunk], old: str, issues: list[Any]
    ) -> GeneratedAnswer:
        return await draft(query, chunks)

    monkeypatch.setattr(worker.generator, "draft", draft)
    monkeypatch.setattr(worker.generator, "repair", repair)
    monkeypatch.setattr(worker.judge, "judge", judge)
    result = asyncio.run(run_generate(config, [worker]))
    assert (result.done, result.errors) == (3, 1)
    store = JsonlStore(config.output_dir / "answers.jsonl", AnswerRecord)
    rows = [store.get(case_id(case.user_input), "mmr_on") for case in cases]
    assert [r.outcome for r in rows] == [
        "answered",
        "insufficient_evidence",
        "unable_to_verify",
        "error",
    ]
    assert rows[0].response == "Người lao động có quyền lợi [1]."
    assert rows[0].citations[0].chunk_id == "c1"
    assert rows[0].usage.prompt_tokens == 8
    assert rows[2].repair_used
    assert calls.count(cases[2].user_input) == 2
    assert rows[3].error == "ValueError"
    assert "private malformed" not in store.path.read_text()
    before = len(calls)
    asyncio.run(run_generate(config, [worker]))
    assert len(calls) == before
    asyncio.run(
        run_generate(config.model_copy(update={"retry_failed": True}), [worker])
    )
    assert len(calls) == before + 1


def test_judge_wrapped_throttle_timeout_is_429_and_tracked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = EvalEvidenceJudge(JudgeSettings(GROQ_API_KEY_2="fake", _env_file=None))

    async def timeout(*args: Any, **kwargs: Any) -> Any:
        raise JudgeError("wrapped") from ThrottleTimeout("fake timeout")

    monkeypatch.setattr(EvidenceJudge, "judge", timeout)
    with pytest.raises(EvalRateLimitError) as caught:
        asyncio.run(judge.judge("q", [], "d", []))
    assert caught.value.status_code == 429
    assert isinstance(judge.last_error, JudgeError)


def test_generation_daily_exhaustion_does_not_write_answer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cases = make_config(tmp_path, 1)
    seed_retrieval(config, cases)
    worker = GenerationWorker("fake-generation-key")

    async def exhausted(*args: Any, **kwargs: Any) -> GeneratedAnswer:
        error = DailyQuotaExhaustedError("fake quota")
        worker.generator.last_error = error
        raise error

    monkeypatch.setattr(worker.generator, "draft", exhausted)
    summary = asyncio.run(run_generate(config, [worker]))
    assert (summary.pending, summary.errors, summary.quota_exhausted) == (1, 0, True)
    assert not JsonlStore(config.output_dir / "answers.jsonl", AnswerRecord).records


def test_report_uses_paired_recall_refusal_zero_and_null_error_counts(
    tmp_path: Path,
) -> None:
    config, cases = make_config(tmp_path, 5)
    ids = [case_id(c.user_input) for c in cases]
    recall = [
        ScoreRecord(case_id=ids[0], config="mmr_on", scores={"context_recall": 0.9}),
        ScoreRecord(case_id=ids[0], config="mmr_off", scores={"context_recall": 0.5}),
        ScoreRecord(case_id=ids[1], config="mmr_on", scores={"context_recall": 0.8}),
        ScoreRecord(
            case_id=ids[1],
            config="mmr_off",
            scores={"context_recall": 0.8},
            reused_from="mmr_on",
        ),
        ScoreRecord(case_id=ids[2], config="mmr_on", scores={"context_recall": 0.2}),
        ScoreRecord(
            case_id=ids[2],
            config="mmr_off",
            scores={"context_recall": None},
            error="nan",
        ),
    ]
    answers = [
        AnswerRecord(
            case_id=identifier,
            config="mmr_on",
            outcome=outcome,
            repair_used=i == 2,
            error="timeout" if outcome == "error" else None,
            prompt_version="v11",
        )
        for i, (identifier, outcome) in enumerate(
            zip(
                ids,
                [
                    "answered",
                    "insufficient_evidence",
                    "unable_to_verify",
                    "error",
                    "answered",
                ],
                strict=True,
            )
        )
    ]
    scores = [
        ScoreRecord(
            case_id=ids[0],
            config="mmr_on",
            scores={"faithfulness": 0.9, "answer_relevancy": 0.6},
        ),
        ScoreRecord(
            case_id=ids[4],
            config="mmr_on",
            scores={"faithfulness": None, "answer_relevancy": 0.3},
            error="non_finite_score",
        ),
    ]
    report = build_report(
        cases,
        recall,
        [],
        scores,
        answers,
        [HydeRecord(case_id=ids[0]), HydeRecord(case_id=ids[1], error="timeout")],
        [RetrievalRecord(case_id=ids[2], config="mmr_on", error="rerank_fallback")],
    )
    paired = report["retrieval_comparison"]["overall"]
    assert (
        paired["paired"],
        paired["excluded"],
        paired["wins"],
        paired["ties"],
        paired["identical_chunks"],
    ) == (2, 3, 1, 1, 1)
    assert paired["mmr_on"]["mean"] == pytest.approx(0.85)
    group = report["answers"]["mmr_on"]["overall"]
    assert group["conditional"]["faithfulness"] == {"mean": 0.9, "n": 1, "null": 1}
    assert group["end_to_end"]["faithfulness"] == {"mean": 0.3, "n": 3, "null": 1}
    assert group["excluded_errors"] == 1
    assert group["refusal_rate"] == 0.4
    assert group["repair_rate"] == 0.2
    assert report["answers"]["mmr_on"]["source_document"]["A.md"]["n"] == 3
    assert len(report["answers"]["mmr_on"]["synthesizer_name"]) == 2
    assert report["operations"] == {
        "hyde_null": 1,
        "hyde_errors": 1,
        "retrieval_errors": {"mmr_on": 1, "mmr_off": 0},
        "rerank_fallback": 1,
        "prompt_versions": ["v11"],
    }
    assert len(report["interpretation_notes"]) == 4
    for answer in answers:
        JsonlStore(config.output_dir / "answers.jsonl", AnswerRecord).append(answer)
    status = stage_status(config)
    assert status["score-answers:mmr_on"].pending == 3
