"""S1: prepare production HyDE through per-key workers and shared queue."""

from __future__ import annotations

from collections.abc import Sequence

from production_legal_qa_rag.evaluation.jsonl_store import JsonlStore
from production_legal_qa_rag.evaluation.key_pool import (
    api_keys,
    build_hyde_workers,
    error_code,
    is_daily_quota,
    run_key_queue,
)
from production_legal_qa_rag.evaluation.models import GoldenTestCase
from production_legal_qa_rag.evaluation.run_models import (
    EvalConfig,
    HydeRecord,
    StageSummary,
    case_id,
    load_testset,
)
from production_legal_qa_rag.retrieval.hyde import HydeGenerator


async def run_hyde(
    config: EvalConfig, workers: Sequence[HydeGenerator] | None = None
) -> StageSummary:
    """Append one HyDE checkpoint per processed case, leaving exhausted cases pending."""
    cases = load_testset(config)
    store = JsonlStore(config.output_dir / "hyde.jsonl", HydeRecord)
    pending = [
        c
        for c in cases
        if store.should_run(case_id(c.user_input), retry_failed=config.retry_failed)
    ]
    if not pending:
        return store.summary([case_id(c.user_input) for c in cases])
    workers = workers or build_hyde_workers(api_keys()[: config.workers])

    async def process(worker: HydeGenerator, case: GoldenTestCase) -> None:
        identifier = case_id(case.user_input)
        try:
            hypo = await worker.generate(case.user_input)
            record = HydeRecord(case_id=identifier, hypothetical_document=hypo)
        except Exception as error:
            if is_daily_quota(error):
                raise
            record = HydeRecord(case_id=identifier, error=error_code(error))
        store.append(record)

    exhausted = await run_key_queue(pending, workers, process)
    return store.summary(
        [case_id(c.user_input) for c in cases], quota_exhausted=exhausted
    )
