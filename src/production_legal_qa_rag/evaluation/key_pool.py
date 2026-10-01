"""Shared work queues and per-key production clients for offline evaluation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from types import SimpleNamespace
from typing import Any

from groq import AsyncGroq

from production_legal_qa_rag.config import (
    GenerationSettings,
    HydeSettings,
    JudgeSettings,
    TestsetGeneratorSettings,
    ThrottleSettings,
)
from production_legal_qa_rag.evaluation.groq_round_robin import DailyQuotaExhaustedError
from production_legal_qa_rag.generation.generator import (
    AnswerGenerator,
    GeneratedAnswer,
    build_messages,
    build_repair_messages,
)
from production_legal_qa_rag.generation.judge import EvidenceJudge
from production_legal_qa_rag.generation.models import (
    Citation,
    JudgeVerdict,
    VerificationIssue,
)
from production_legal_qa_rag.generation.pipeline import GenerationPipeline
from production_legal_qa_rag.retrieval.hyde import HydeGenerator
from production_legal_qa_rag.retrieval.llm_throttle import (
    ThrottleTimeout,
    estimate_tokens,
    get_throttle,
)
from production_legal_qa_rag.retrieval.models import RetrievedChunk

THROTTLE_MAX_WAIT = 300.0
CHARS_PER_TOKEN = 3.0
EXPECTED_COMPLETION_TOKENS = 1000


def error_chain(error: BaseException) -> list[BaseException]:
    """Walk wrapped causes without printing provider/user content."""
    chain: list[BaseException] = []
    while error not in chain:
        chain.append(error)
        successor = error.__cause__ or error.__context__
        if successor is None:
            break
        error = successor
    return chain


def is_daily_quota(error: BaseException) -> bool:
    """Recognize both SDK daily limits and router circuit breakers."""
    return any(
        isinstance(item, DailyQuotaExhaustedError)
        or (
            getattr(item, "status_code", None) == 429
            and any(
                marker in str(item).lower() for marker in ("per day", "(tpd)", "(rpd)")
            )
        )
        for item in error_chain(error)
    )


def error_code(error: BaseException) -> str:
    """Return a safe exception classification, never provider messages."""
    root = error_chain(error)[-1]
    status = getattr(root, "status_code", None)
    return (
        f"{type(root).__name__}:{status}"
        if isinstance(status, int)
        else type(root).__name__
    )


def api_keys(settings: TestsetGeneratorSettings | None = None) -> list[str]:
    """Load the mandatory nine credentials through existing settings only."""
    settings = settings or TestsetGeneratorSettings()
    return [settings.api_key] + [
        getattr(settings, f"api_key_{i}") for i in range(2, 10)
    ]


async def run_key_queue[ItemT, WorkerT](
    items: Sequence[ItemT],
    workers: Sequence[WorkerT],
    process: Callable[[WorkerT, ItemT], Awaitable[None]],
) -> bool:
    """Share a queue; daily-exhausted workers return their case to live workers."""
    queue: asyncio.Queue[ItemT] = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)
    exhausted = 0

    async def run_worker(worker: WorkerT) -> None:
        nonlocal exhausted
        while True:
            item = await queue.get()
            try:
                await process(worker, item)
            except Exception as error:
                if not is_daily_quota(error):
                    raise
                queue.put_nowait(item)
                exhausted += 1
                return
            finally:
                queue.task_done()

    tasks = {asyncio.create_task(run_worker(worker)) for worker in workers}
    joined = asyncio.create_task(queue.join())
    try:
        live = tasks.copy()
        while live:
            finished, _ = await asyncio.wait(
                live | {joined}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in finished - {joined}:
                task.result()
            live -= finished
            if joined in finished:
                break
    finally:
        for task in tasks | {joined}:
            task.cancel()
        await asyncio.gather(*tasks, joined, return_exceptions=True)
    return bool(exhausted and not queue.empty())


class EvalRateLimitError(RuntimeError):
    """Make throttle timeouts retriable through production's 429 detection."""

    status_code = 429


class TrackedHydeGenerator(HydeGenerator):
    """Keep production HyDE behavior while observing swallowed HTTP failures."""

    def __init__(self, settings: HydeSettings) -> None:
        self.last_error: Exception | None = None
        client = AsyncGroq(
            api_key=settings.api_key,
            max_retries=0,
            timeout=float(settings.timeout_seconds),
        )
        original = client.chat.completions.create

        async def tracked_create(**kwargs: Any) -> Any:
            try:
                return await original(**kwargs)
            except Exception as error:  # noqa: BLE001 - preserve swallowed provider failures.
                self.last_error = error
                raise RuntimeError("HyDE provider failure") from None

        proxy = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=tracked_create))
        )
        super().__init__(
            settings,
            client=proxy,  # type: ignore[arg-type]
            throttle_settings=ThrottleSettings(
                optional_step_max_wait_seconds=THROTTLE_MAX_WAIT
            ),
        )

    async def generate(self, query: str) -> str | None:
        """Raise tracked failures after production fallback returns null."""
        self.last_error = None
        result = await super().generate(query)
        if self.last_error is not None:
            raise self.last_error
        return result


class ThrottledAnswerGenerator(AnswerGenerator):
    """Reserve expected tokens before every draft/repair and settle actual usage."""

    def __init__(self, settings: GenerationSettings) -> None:
        super().__init__(settings)
        self.last_error: Exception | None = None

    async def _call(
        self,
        messages: list[dict[str, str]],
        operation: Callable[[], Awaitable[GeneratedAnswer]],
    ) -> GeneratedAnswer:
        settings = self._get_settings()
        throttle = get_throttle(settings.model_name, settings.api_key)
        reservation = await throttle.acquire(
            estimate_tokens(
                sum(len(m["content"]) for m in messages),
                CHARS_PER_TOKEN,
                EXPECTED_COMPLETION_TOKENS,
            ),
            THROTTLE_MAX_WAIT,
        )
        try:
            answer = await operation()
        except Exception as error:
            self.last_error = error
            raise
        usage = answer.usage
        total = (
            None
            if usage is None
            or usage.prompt_tokens is None
            or usage.completion_tokens is None
            else usage.prompt_tokens + usage.completion_tokens
        )
        throttle.settle(reservation, total)
        return answer

    async def draft(self, query: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        """Throttle the unchanged production draft prompt."""
        return await self._call(
            build_messages(query, chunks),
            lambda: super(ThrottledAnswerGenerator, self).draft(query, chunks),
        )

    async def repair(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        draft: str,
        issues: list[VerificationIssue],
    ) -> GeneratedAnswer:
        """Throttle the complete production repair prompt including prior draft."""
        return await self._call(
            build_repair_messages(query, chunks, draft, issues),
            lambda: super(ThrottledAnswerGenerator, self).repair(
                query, chunks, draft, issues
            ),
        )


class EvalEvidenceJudge(EvidenceJudge):
    """Retain Judge semantics; expose daily exhaustion and throttle timeouts."""

    def __init__(self, settings: JudgeSettings) -> None:
        super().__init__(settings)
        self.last_error: Exception | None = None

    async def judge(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        draft: str,
        citations: list[Citation],
    ) -> JudgeVerdict:
        """Translate wrapped throttle errors before the pipeline sees them."""
        try:
            return await super().judge(query, chunks, draft, citations)
        except Exception as error:
            self.last_error = error
            if any(isinstance(item, ThrottleTimeout) for item in error_chain(error)):
                raise EvalRateLimitError("Judge throttle timeout") from error
            raise


def build_hyde_workers(keys: Sequence[str]) -> list[TrackedHydeGenerator]:
    """Bind one 20b client to each independent key."""
    return [
        TrackedHydeGenerator(HydeSettings(GROQ_API_KEY_1=key, max_retries=0))
        for key in keys
    ]


class GenerationWorker:
    """Own one pipeline and its tracked dependencies for a single credential."""

    def __init__(self, key: str) -> None:
        self.generator = ThrottledAnswerGenerator(
            GenerationSettings(GROQ_API_KEY_3=key, GROQ_API_KEY_4=None, max_retries=0)
        )
        self.judge = EvalEvidenceJudge(
            JudgeSettings(
                GROQ_API_KEY_2=key,
                max_retries=0,
                timeout_seconds=int(THROTTLE_MAX_WAIT),
            )
        )
        self.pipeline = GenerationPipeline(generator=self.generator, judge=self.judge)

    def reset(self) -> None:
        """Clear errors between independent cases."""
        self.generator.last_error = self.judge.last_error = None

    def raise_daily_quota(self) -> None:
        """Recover daily provider failures swallowed by pipeline event mapping."""
        for error in (self.generator.last_error, self.judge.last_error):
            if error is not None and is_daily_quota(error):
                raise error
