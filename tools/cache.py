"""Kiểm tra thủ công Redis cache mà không khởi tạo toàn bộ API."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Annotated, Final

import typer
from redis.asyncio import Redis

from production_legal_qa_rag.cache.keys import compute_corpus_version
from production_legal_qa_rag.cache.models import CachedAnswer
from production_legal_qa_rag.cache.singleflight import SingleFlight
from production_legal_qa_rag.cache.store import AnswerCache, RetrievalCache
from production_legal_qa_rag.generation.generator import PROMPT_VERSION
from production_legal_qa_rag.generation.models import Citation
from production_legal_qa_rag.retrieval.models import RetrievedChunk

app = typer.Typer(add_completion=False)

_DEFAULT_REDIS_URL: Final = "redis://localhost:6379/0"
_DEFAULT_MODEL_NAME: Final = "openai/gpt-oss-120b"
_SAMPLE_QUERY: Final = "Khoản 1 Điều 113 quy định gì?"


@app.command()
def smoke(
    redis_url: Annotated[str, typer.Option(envvar="REDIS_URL")] = _DEFAULT_REDIS_URL,
    corpus_version: Annotated[
        str | None, typer.Option(envvar="CACHE_CORPUS_VERSION")
    ] = None,
    model_name: Annotated[str, typer.Option()] = _DEFAULT_MODEL_NAME,
) -> None:
    """Kiểm tra answer cache, retrieval cache và lock trên Redis hiện tại."""
    asyncio.run(_smoke(redis_url, corpus_version, model_name))


async def _smoke(redis_url: str, corpus_override: str | None, model_name: str) -> None:
    """Chạy round-trip cache trực tiếp, không cần dựng API/Groq/Pinecone."""
    redis = Redis.from_url(redis_url)
    corpus_version = compute_corpus_version(override=corpus_override)
    answer_cache = AnswerCache(
        redis,
        corpus_version=corpus_version,
        prompt_version=PROMPT_VERSION,
        model_name=model_name,
    )
    retrieval_cache = RetrievalCache(redis, corpus_version=corpus_version)
    try:
        await redis.ping()
        answer = _sample_answer()
        chunks = [_sample_chunk()]
        await answer_cache.set(_SAMPLE_QUERY, answer)
        await retrieval_cache.set(_SAMPLE_QUERY, chunks)
        answer_hit = await answer_cache.get(_SAMPLE_QUERY)
        retrieval_hit = await retrieval_cache.get(_SAMPLE_QUERY)
        async with SingleFlight(redis, answer_cache).acquire(_SAMPLE_QUERY) as flight:
            typer.echo(f"single_flight_leader={flight.is_leader}")
        typer.echo(f"corpus_version={corpus_version}")
        typer.echo(f"answer_key={answer_cache.key_for(_SAMPLE_QUERY)}")
        typer.echo(f"answer_round_trip={answer_hit == answer}")
        typer.echo(f"retrieval_round_trip={retrieval_hit == chunks}")
    finally:
        await redis.aclose()


def _sample_answer() -> CachedAnswer:
    """Tạo answer tối thiểu có citation hợp lệ cho CLI smoke test."""
    return CachedAnswer(
        text="Nội dung mẫu có căn cứ [1].",
        citations=[
            Citation(
                n=1,
                chunk_id="cache-smoke-chunk",
                source_document="manual",
                breadcrumb="Điều mẫu",
            )
        ],
        created_at=datetime.now(UTC),
    )


def _sample_chunk() -> RetrievedChunk:
    """Tạo chunk tối thiểu để xác nhận serialization retrieval cache."""
    return RetrievedChunk(
        chunk_id="cache-smoke-chunk",
        source_document="manual",
        breadcrumb="Điều mẫu",
        content="Nội dung mẫu.",
        rerank_score=1.0,
    )


if __name__ == "__main__":
    app()
