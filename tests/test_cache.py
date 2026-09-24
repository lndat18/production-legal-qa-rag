"""Kiểm thử contract Redis cache, replay và single-flight."""

from __future__ import annotations

import asyncio
import importlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from production_legal_qa_rag.cache.keys import (
    UNKNOWN_CORPUS_VERSION,
    answer_key,
    compute_corpus_version,
    lock_key,
    retrieval_key,
)
from production_legal_qa_rag.cache.models import CachedAnswer
from production_legal_qa_rag.cache.normalize import normalize_query
from production_legal_qa_rag.cache.singleflight import SingleFlight
from production_legal_qa_rag.cache.store import (
    ANSWER_TTL_SECONDS,
    RETRIEVAL_TTL_SECONDS,
    AnswerCache,
    RetrievalCache,
)
from production_legal_qa_rag.config import CacheSettings, RedisSettings
from production_legal_qa_rag.generation.models import (
    Citation,
    CitationsEvent,
    DoneEvent,
    TokenEvent,
)
from production_legal_qa_rag.retrieval.models import RetrievedChunk


class _MemoryRedis:
    """Redis async tối thiểu, ghi nhận TTL để test không cần service thật."""

    def __init__(self) -> None:
        self.values: dict[str, str | bytes] = {}
        self.expirations: dict[str, int] = {}
        self.should_fail = False

    async def get(self, key: str) -> str | bytes | None:
        self._raise_if_needed()
        return self.values.get(key)

    async def set(
        self,
        key: str,
        value: str | bytes,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool:
        self._raise_if_needed()
        if nx and key in self.values:
            return False
        self.values[key] = value
        if ex is not None:
            self.expirations[key] = ex
        return True

    async def eval(self, _script: str, _numkeys: int, key: str, value: str) -> int:
        self._raise_if_needed()
        if self.values.get(key) != value:
            return 0
        del self.values[key]
        self.expirations.pop(key, None)
        return 1

    def _raise_if_needed(self) -> None:
        if self.should_fail:
            raise ConnectionError("Redis unavailable")


def _answer() -> CachedAnswer:
    return CachedAnswer(
        text="Nội dung có căn cứ [1].",
        citations=[
            Citation(
                n=1,
                chunk_id="chunk-1",
                source_document="source",
                breadcrumb="Điều 1",
            )
        ],
        created_at=datetime.now(UTC),
    )


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        source_document="source",
        breadcrumb="Điều 1",
        content="Nội dung.",
        rerank_score=0.9,
    )


def _answer_cache(redis: Any) -> AnswerCache:
    return AnswerCache(
        redis, corpus_version="corpus", prompt_version="v1", model_name="model"
    )


def test_normalize_query_giu_dau_tieng_viet_va_so() -> None:
    assert normalize_query("Khoản 2  Điều 113?!  ") == "khoản 2 điều 113"
    assert normalize_query("Điều 113") != normalize_query("Điều 114")


def test_keys_doi_theo_query_va_versions() -> None:
    first = answer_key(
        "Khoản 1 Điều 113?",
        corpus_version="corpus-a",
        prompt_version="v1",
        model_name="model-a",
    )
    assert first == answer_key(
        "khoản 1 điều 113",
        corpus_version="corpus-a",
        prompt_version="v1",
        model_name="model-a",
    )
    assert first != answer_key(
        "Khoản 2 Điều 113",
        corpus_version="corpus-a",
        prompt_version="v1",
        model_name="model-a",
    )
    assert first != answer_key(
        "Khoản 1 Điều 113",
        corpus_version="corpus-b",
        prompt_version="v1",
        model_name="model-a",
    )
    assert first != answer_key(
        "Khoản 1 Điều 113",
        corpus_version="corpus-a",
        prompt_version="v2",
        model_name="model-a",
    )
    assert first != answer_key(
        "Khoản 1 Điều 113",
        corpus_version="corpus-a",
        prompt_version="v1",
        model_name="model-b",
    )
    assert lock_key(first) == f"{first}:lock"
    assert retrieval_key("Khoản 1 Điều 113", corpus_version="corpus-a") != retrieval_key(
        "Khoản 1 Điều 113", corpus_version="corpus-b"
    )


def test_compute_corpus_version_dung_override_va_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    params = tmp_path / "bm25_params.json"
    params.write_text('{"k": 1}', encoding="utf-8")

    assert len(compute_corpus_version(params)) == 12
    assert compute_corpus_version(params, override="manual") == "manual"

    with caplog.at_level(logging.WARNING):
        assert compute_corpus_version(tmp_path / "missing.json") == UNKNOWN_CORPUS_VERSION
    assert "CACHE_CORPUS_VERSION" in caplog.text


def test_answer_and_retrieval_cache_round_trip_va_ttl() -> None:
    async def scenario() -> None:
        redis: Any = _MemoryRedis()
        answer_cache = _answer_cache(redis)
        retrieval_cache = RetrievalCache(redis, corpus_version="corpus")
        answer = _answer()
        chunks = [_chunk()]

        await answer_cache.set("Câu hỏi", answer)
        await retrieval_cache.set("Câu hỏi", chunks)

        assert await answer_cache.get("câu hỏi?") == answer
        assert await retrieval_cache.get("câu hỏi") == chunks
        assert redis.expirations[answer_cache.key_for("câu hỏi")] == ANSWER_TTL_SECONDS
        assert (
            redis.expirations[retrieval_cache.key_for("câu hỏi")]
            == RETRIEVAL_TTL_SECONDS
        )

    asyncio.run(scenario())


def test_cache_loi_redis_luon_degrade_thanh_miss() -> None:
    async def scenario() -> None:
        redis: Any = _MemoryRedis()
        answer_cache = _answer_cache(redis)
        retrieval_cache = RetrievalCache(redis, corpus_version="corpus")
        redis.should_fail = True

        await answer_cache.set("q", _answer())
        await retrieval_cache.set("q", [_chunk()])
        assert await answer_cache.get("q") is None
        assert await retrieval_cache.get("q") is None

    asyncio.run(scenario())


def test_retrieval_cache_khong_luu_danh_sach_rong() -> None:
    async def scenario() -> None:
        redis: Any = _MemoryRedis()
        cache = RetrievalCache(redis, corpus_version="corpus")
        await cache.set("q", [])
        assert redis.values == {}

    asyncio.run(scenario())


def test_replay_giu_nguyen_text_va_chi_phat_event_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> tuple[list[Any], list[float]]:
        replay_module = importlib.import_module(
            "production_legal_qa_rag.cache.replay"
        )
        pauses: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            pauses.append(seconds)

        monkeypatch.setattr(replay_module.asyncio, "sleep", fake_sleep)
        answer = _answer().model_copy(
            update={"text": "  Một hai\nba bốn  năm sáu  "}
        )
        events = [event async for event in replay_module.replay(answer)]
        return events, pauses

    events, pauses = asyncio.run(scenario())
    token_text = "".join(event.text for event in events if isinstance(event, TokenEvent))
    assert token_text == "  Một hai\nba bốn  năm sáu  "
    assert isinstance(events[-2], CitationsEvent)
    assert isinstance(events[-1], DoneEvent) and events[-1].usage is None
    assert pauses == [0.015]


def test_singleflight_follower_doc_cache_va_leader_nha_lock() -> None:
    async def scenario() -> None:
        redis: Any = _MemoryRedis()
        answer_cache = _answer_cache(redis)
        flight_manager = SingleFlight(redis, answer_cache)
        query = "Câu hỏi"
        lock = lock_key(answer_cache.key_for(query))
        answer = _answer()

        async with flight_manager.acquire(query, "leader") as leader:
            assert leader.is_leader
            async with flight_manager.acquire(query, "follower") as follower:
                assert not follower.is_leader
                await answer_cache.set(query, answer)
                assert await follower.wait_for_answer() == answer
        assert lock not in redis.values

    asyncio.run(scenario())


def test_singleflight_follower_take_over_khi_lock_bien_mat() -> None:
    async def scenario() -> None:
        redis: Any = _MemoryRedis()
        answer_cache = _answer_cache(redis)
        flight_manager = SingleFlight(redis, answer_cache)
        query = "Câu hỏi"
        lock = lock_key(answer_cache.key_for(query))
        await redis.set(lock, "old-leader", nx=True, ex=60)

        async with flight_manager.acquire(query, "follower") as follower:
            assert not follower.is_leader
            del redis.values[lock]
            assert await follower.wait_for_answer() is None
            assert follower.is_leader
        assert lock not in redis.values

    asyncio.run(scenario())


def test_redis_and_cache_settings_doc_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://cache.example:6379/2")
    monkeypatch.setenv("CACHE_CORPUS_VERSION", "manual-corpus")
    monkeypatch.setattr(
        RedisSettings,
        "model_config",
        {**RedisSettings.model_config, "env_file": None},
    )
    monkeypatch.setattr(
        CacheSettings,
        "model_config",
        {**CacheSettings.model_config, "env_file": None},
    )

    assert RedisSettings().redis_url == "redis://cache.example:6379/2"
    assert CacheSettings().corpus_version == "manual-corpus"
