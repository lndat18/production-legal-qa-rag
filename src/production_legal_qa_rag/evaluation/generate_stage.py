"""S5: reduce the actual verified generation event stream into answer records."""

from __future__ import annotations

from collections.abc import Sequence

from production_legal_qa_rag.evaluation.jsonl_store import JsonlStore
from production_legal_qa_rag.evaluation.key_pool import (
    GenerationWorker,
    api_keys,
    error_code,
    is_daily_quota,
    run_key_queue,
)
from production_legal_qa_rag.evaluation.models import GoldenTestCase
from production_legal_qa_rag.evaluation.run_models import (
    AnswerRecord,
    EvalConfig,
    RetrievalRecord,
    StageSummary,
    case_id,
    load_testset,
)
from production_legal_qa_rag.evaluation.testset_generator import EvalInputError
from production_legal_qa_rag.generation.generator import PROMPT_VERSION


async def run_generate(
    config: EvalConfig, workers: Sequence[GenerationWorker] | None = None
) -> StageSummary:
    """Run no guardrail/cache/retrieval; daily failures return to the key queue."""
    if config.config is None:
        raise EvalInputError("--config bắt buộc cho generate.")
    choice = config.config
    cases = load_testset(config)
    retrieval = JsonlStore(
        config.output_dir / f"retrieved_{choice}.jsonl", RetrievalRecord
    )
    store = JsonlStore(config.output_dir / "answers.jsonl", AnswerRecord)
    pending: list[GoldenTestCase] = []
    missing = 0
    for case in cases:
        identifier = case_id(case.user_input)
        if not store.should_run(
            identifier, config=choice, retry_failed=config.retry_failed
        ):
            continue
        row = retrieval.get(identifier, choice)
        if row is None or row.error is not None or not row.chunks:
            missing += 1
        else:
            pending.append(case)
    if pending:
        workers = workers or [
            GenerationWorker(key) for key in api_keys()[: config.workers]
        ]
    exhausted = False

    async def process(worker: GenerationWorker, case: GoldenTestCase) -> None:
        identifier = case_id(case.user_input)
        row = retrieval.get(identifier, choice)
        assert row is not None
        worker.reset()
        record = AnswerRecord(
            case_id=identifier,
            config=choice,
            outcome="error",
            prompt_version=PROMPT_VERSION,
        )
        fragments: list[str] = []
        try:
            async for event in worker.pipeline.generate(case.user_input, row.chunks):
                if event.type == "token":
                    fragments.append(event.text)
                    record.outcome = "answered"
                elif event.type == "citations":
                    record.citations = event.citations
                elif event.type == "warning":
                    record.warning_codes.append(event.code)
                elif event.type == "status" and event.stage == "repairing":
                    record.repair_used = True
                elif event.type == "refusal":
                    if event.reason not in {
                        "insufficient_evidence",
                        "unable_to_verify",
                    }:
                        raise ValueError("Unexpected generation refusal")
                    record.outcome = event.reason
                elif event.type == "error":
                    record.outcome = "error"
                    record.error = record.error_code = event.code
                elif event.type == "done":
                    record.usage = event.usage
            worker.raise_daily_quota()
            unexpected = worker.generator.last_error or worker.judge.last_error
            if unexpected is not None:
                raise unexpected
            if record.outcome == "answered":
                record.response = "".join(fragments)
            elif record.outcome == "error" and record.error is None:
                record.error = record.error_code = "incomplete_generation"
        except Exception as error:
            if is_daily_quota(error):
                raise
            record.outcome = "error"
            record.error = record.error_code = error_code(error)
        store.append(record)

    if pending:
        assert workers is not None
        exhausted = await run_key_queue(pending, workers, process)
    return store.summary(
        [case_id(c.user_input) for c in cases],
        choice,
        missing_upstream=missing,
        quota_exhausted=exhausted,
    )
