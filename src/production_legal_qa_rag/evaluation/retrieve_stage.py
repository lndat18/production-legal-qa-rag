"""S3: sequential MMR-on then MMR-off retrieval sharing query embeddings."""

from __future__ import annotations

from production_legal_qa_rag.evaluation.jsonl_store import JsonlStore
from production_legal_qa_rag.evaluation.key_pool import error_code
from production_legal_qa_rag.evaluation.run_models import (
    CONFIGS,
    EmbeddingRecord,
    EvalConfig,
    RetrievalRecord,
    StageSummary,
    case_id,
    load_testset,
)
from production_legal_qa_rag.retrieval.pipeline import RetrievalPipeline

RETRIEVE_CONCURRENCY = 1


async def run_retrieve(
    config: EvalConfig, pipeline: RetrievalPipeline | None = None
) -> dict[str, StageSummary]:
    """Use one GPU pipeline; rerank fallback and empty retrieval are errors."""
    cases = load_testset(config)
    embeddings = JsonlStore(config.output_dir / "embeddings.jsonl", EmbeddingRecord)
    result: dict[str, StageSummary] = {}
    for choice in CONFIGS:
        store = JsonlStore(
            config.output_dir / f"retrieved_{choice}.jsonl", RetrievalRecord
        )
        missing = 0
        for case in cases:
            identifier = case_id(case.user_input)
            if not store.should_run(
                identifier, config=choice, retry_failed=config.retry_failed
            ):
                continue
            row = embeddings.get(identifier)
            if row is None or row.error is not None or row.precomputed is None:
                missing += 1
                continue
            pipeline = pipeline or RetrievalPipeline()
            try:
                chunks = await pipeline.retrieve(
                    case.user_input,
                    use_mmr=choice == "mmr_on",
                    precomputed=row.precomputed,
                )
                failure = (
                    "no_context"
                    if not chunks
                    else "rerank_fallback"
                    if any(c.rerank_score is None for c in chunks)
                    else None
                )
                record = RetrievalRecord(
                    case_id=identifier, config=choice, chunks=chunks, error=failure
                )
            except Exception as error:  # noqa: BLE001 - checkpoint unexpected external failures.
                record = RetrievalRecord(
                    case_id=identifier, config=choice, error=error_code(error)
                )
            store.append(record)
        result[choice] = store.summary(
            [case_id(c.user_input) for c in cases], choice, missing_upstream=missing
        )
    return result
