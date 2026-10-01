"""S2: batch query/HyDE texts into at most 25 texts per HF request."""

from __future__ import annotations

from production_legal_qa_rag.evaluation.jsonl_store import JsonlStore
from production_legal_qa_rag.evaluation.key_pool import error_code
from production_legal_qa_rag.evaluation.run_models import (
    EmbeddingRecord,
    EvalConfig,
    HydeRecord,
    StageSummary,
    case_id,
    load_testset,
)
from production_legal_qa_rag.retrieval.models import PrecomputedQuery
from production_legal_qa_rag.retrieval.query_embedder import QueryEmbedder

EMBED_BATCH_SIZE = 25


async def run_embed(
    config: EvalConfig, embedder: QueryEmbedder | None = None
) -> StageSummary:
    """Preserve text/vector order across shared batching and resume checkpoints."""
    cases = load_testset(config)
    hyde = JsonlStore(config.output_dir / "hyde.jsonl", HydeRecord)
    store = JsonlStore(config.output_dir / "embeddings.jsonl", EmbeddingRecord)
    jobs: list[tuple[str, str | None, list[str]]] = []
    missing = 0
    for case in cases:
        identifier = case_id(case.user_input)
        if not store.should_run(identifier, retry_failed=config.retry_failed):
            continue
        row = hyde.get(identifier)
        if row is None or row.error is not None:
            missing += 1
            continue
        texts = (
            [case.user_input]
            if row.hypothetical_document is None
            else [row.hypothetical_document, case.user_input]
        )
        jobs.append((identifier, row.hypothetical_document, texts))
    if jobs:
        embedder = embedder or QueryEmbedder()
    batch: list[tuple[str, str | None, list[str]]] = []
    for job in jobs:
        if sum(len(j[2]) for j in batch) + len(job[2]) > EMBED_BATCH_SIZE:
            assert embedder is not None
            await _embed_batch(batch, embedder, store)
            batch = []
        batch.append(job)
    if batch:
        assert embedder is not None
        await _embed_batch(batch, embedder, store)
    return store.summary(
        [case_id(c.user_input) for c in cases], missing_upstream=missing
    )


async def _embed_batch(
    jobs: list[tuple[str, str | None, list[str]]],
    embedder: QueryEmbedder,
    store: JsonlStore[EmbeddingRecord],
) -> None:
    texts = [text for _, _, values in jobs for text in values]
    try:
        vectors = await embedder.embed(texts)
        if len(vectors) != len(texts):
            raise ValueError("Wrong HF vector count")
        offset = 0
        records: list[EmbeddingRecord] = []
        for identifier, hypo, values in jobs:
            count = len(values)
            precomputed = PrecomputedQuery(
                hypothetical_document=hypo,
                query_embedding=vectors[offset + count - 1],
                hypothetical_embedding=vectors[offset] if count == 2 else None,
            )
            records.append(EmbeddingRecord(case_id=identifier, precomputed=precomputed))
            offset += count
    except Exception as error:  # noqa: BLE001 - checkpoint unexpected external failures.
        records = [
            EmbeddingRecord(case_id=identifier, error=error_code(error))
            for identifier, _, _ in jobs
        ]
    for record in records:
        store.append(record)
