"""Verify vector validation, segmented embedding and real router reservations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from production_legal_qa_rag.config import EmbeddingSettings
from production_legal_qa_rag.evaluation import embeddings_adapter
from production_legal_qa_rag.evaluation.embeddings_adapter import RagasEmbeddingsAdapter
from production_legal_qa_rag.evaluation.groq_round_robin import GroqRoundRobinChatModel
from production_legal_qa_rag.retrieval.llm_throttle import TokenWindowThrottle
from production_legal_qa_rag.retrieval.models import PrecomputedQuery


@pytest.mark.parametrize(
    "values",
    [
        {"query_embedding": []},
        {"query_embedding": [float("nan")]},
        {"query_embedding": [float("inf")]},
        {"query_embedding": [1.0], "hypothetical_document": "hypo"},
        {"query_embedding": [1.0], "hypothetical_embedding": [1.0]},
        {
            "query_embedding": [1.0],
            "hypothetical_document": "hypo",
            "hypothetical_embedding": [1.0, 2.0],
        },
        {
            "query_embedding": [1.0],
            "hypothetical_document": "hypo",
            "hypothetical_embedding": [float("nan")],
        },
    ],
)
def test_precomputed_query_rejects_unusable_vectors(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PrecomputedQuery(**values)


@pytest.mark.parametrize("segment", [False, True])
def test_adapter_query_and_documents_share_optional_segmentation(
    segment: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    class Client:
        def feature_extraction(self, texts: list[str]) -> list[list[float]]:
            seen.append(texts)
            return [[float(i), 1.0] for i in range(len(texts))]

    monkeypatch.setattr(
        embeddings_adapter.ViTokenizer, "tokenize", lambda text: "SEG:" + text
    )
    settings = EmbeddingSettings(HF_TOKEN="fake", _env_file=None)
    adapter = RagasEmbeddingsAdapter(settings, Client(), segment=segment)
    texts = [f"quyền lợi {i}" for i in range(26)]
    vectors = adapter.embed_documents(texts)
    assert len(vectors) == 26
    adapter.embed_query(texts[0])
    prefix = "SEG:" if segment else ""
    assert seen == [
        [prefix + text for text in texts[:25]],
        [prefix + texts[25]],
        [prefix + texts[0]],
    ]
    assert texts[0] == "quyền lợi 0"


def test_router_reserves_before_http_settles_actual_usage_and_waits_tpm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    waits: list[float] = []
    events: list[str] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    class RecordingThrottle(TokenWindowThrottle):
        async def acquire(self, estimated_tokens: int, max_wait_seconds: float) -> Any:
            events.append("acquire")
            assert estimated_tokens == 1001
            return await super().acquire(estimated_tokens, max_wait_seconds)

        def settle(self, reservation: Any, actual_tokens: int | None) -> None:
            events.append("settle")
            assert actual_tokens == 1300
            super().settle(reservation, actual_tokens)

    throttle = RecordingThrottle(2500, 30, 1.0, clock=lambda: now[0], sleep=sleep)
    client = ChatOpenAI(api_key="fake-key", model="openai/gpt-oss-120b")

    def generate(*args: Any, **kwargs: Any) -> ChatResult:
        events.append("http")
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="ok"))],
            llm_output={
                "token_usage": {"prompt_tokens": 1200, "completion_tokens": 100}
            },
        )

    monkeypatch.setattr(client, "_generate", generate)
    identities: list[tuple[str, str]] = []

    def factory(model: str, key: str) -> TokenWindowThrottle:
        identities.append((model, key))
        return throttle

    router = GroqRoundRobinChatModel(clients=[client], throttle_factory=factory)
    for _ in range(3):
        assert router.invoke([HumanMessage(content="abc")]).content == "ok"
    assert events == ["acquire", "http", "settle"] * 3
    assert waits == [60.0]
    assert identities == [("openai/gpt-oss-120b", "fake-key")] * 3
    assert router.call_counts == [3]


def test_real_token_throttle_rpm_and_missing_usage_keep_reservation() -> None:
    now = [0.0]
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    throttle = TokenWindowThrottle(100, 1, 1.0, clock=lambda: now[0], sleep=sleep)

    async def requests() -> None:
        reservation = await throttle.acquire(40, 300)
        throttle.settle(reservation, None)
        await throttle.acquire(40, 300)

    asyncio.run(requests())
    assert waits == [60.0]


def test_fake_clock_factory_keeps_models_and_keys_in_separate_real_buckets(
    eval_throttle_factory: Callable[..., Any],
) -> None:
    assert eval_throttle_factory("model", "key") is eval_throttle_factory(
        "model", "key"
    )
    assert eval_throttle_factory("model", "key") is not eval_throttle_factory(
        "other", "key"
    )
    assert eval_throttle_factory("model", "key") is not eval_throttle_factory(
        "model", "other"
    )
