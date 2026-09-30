"""Cài đặt `UnitRunner` bằng `ragas` + 9 tài khoản Groq round-robin (evaluation_spec.md mục 3, 4.3).

Toàn bộ code chạm `ragas` nằm ở đây (import ở mức module) để `testset_generator.py`
không cần dependency-group `eval`; chỉ import module này qua `build_unit_runner`.

Chữ ký ragas đã đọc trực tiếp trên `ragas==0.4.3` cài thật: `prepare_combinations`
của single-hop nhận `(node, terms, personas, persona_concepts)`, còn multi-hop nhận
`(nodes, combinations, personas, persona_item_mapping, property_name)` và đều trả
`list[dict]` có khoá `"styles"`. Multi-hop dùng chữ ký tường minh để lọc mapping
persona không khớp trước khi gọi API Ragas.

Bước sinh câu KHÔNG gọi `TestsetGenerator.generate` (huỷ cả lô khi một sample lỗi, mà
`raise_exceptions=False` làm ragas 0.4.3 crash trên `NaN`) mà tự lặp theo API công khai của
synthesizer để giữ phần đã sinh khi lỗi giữa đơn vị (evaluation_spec.md mục 3.3).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pydantic import ValidationError
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona, generate_personas_from_kg
from ragas.testset.synthesizers.base import BaseScenario, BaseSynthesizer, QueryStyle
from ragas.testset.synthesizers.multi_hop import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.utils import calculate_split_values
from ragas.testset.transforms import Parallel, apply_transforms, default_transforms
from ragas.testset.transforms.base import LLMBasedExtractor

from production_legal_qa_rag.config import TestsetGeneratorSettings
from production_legal_qa_rag.evaluation.embeddings_adapter import (
    RagasEmbeddingsAdapter,
)
from production_legal_qa_rag.evaluation.groq_round_robin import (
    DailyQuotaExhaustedError,
    GroqRoundRobinChatModel,
    TokenTotals,
)
from production_legal_qa_rag.evaluation.models import GoldenTestCase
from production_legal_qa_rag.evaluation.testset_generator import (
    QuestionQuota,
    UnitGenerationError,
    UnitResult,
    unit_key,
)
from production_legal_qa_rag.evaluation.unit_splitter import EvalUnit

logger = logging.getLogger(__name__)

# Groq free: TPM 8K/tài khoản, mặc định ragas 32.000 token/lượt bị 413 (mục 3.1).
MAX_TOKEN_LIMIT: Final = 4_000
MAX_WORKERS: Final = 4
LANGUAGE: Final = "vietnamese"
# Số persona sinh một lần cho mỗi đơn vị (giá trị mặc định của `TestsetGenerator.generate`).
NUM_PERSONAS: Final = 3
# Token suy luận của gpt-oss tính vào TPM/TPD; KG (summary/themes/NER/headlines) là trích xuất
# đơn giản nên hạ xuống `low`. Sinh câu hỏi/đáp án giữ mức mặc định để chất lượng testset
# không đổi (mục 3.2 C).
_KG_REASONING_EFFORT: Final = "low"

# Retry HTTP chỉ do `GroqRoundRobinChatModel` sở hữu. SDK/RAGAS retry đồng thời sẽ nhân
# request/token khi một phản hồi đã lỗi sau khi Groq nhận prompt (mục 3.4).
MAX_RETRIES: Final = 0
MAX_WAIT_SECONDS: Final = 0

# Groq công bố endpoint OpenAI-compatible chính thức (generation_spec.md mục 8),
# cùng pattern với generation/generator.py.
_GROQ_OPENAI_BASE_URL: Final = "https://api.groq.com/openai/v1"


def _force_perfect_grammar(combinations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ragas mặc định trộn 4 `QueryStyle`, 3/4 là nhiễu (sai chính tả/ngữ pháp); ép sạch."""
    for combination in combinations:
        combination["styles"] = [QueryStyle.PERFECT_GRAMMAR]
    return combinations


def _filter_persona_item_mapping(
    personas: list[Persona], persona_item_mapping: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Chỉ giữ mapping cho persona thực sự đã được Ragas sinh.

    Prompt ghép theme-persona đôi khi trả một key persona không thuộc `personas`.
    Ragas 0.4.3 tra key đó qua `PersonaList.__getitem__` và ném `KeyError`, làm hỏng
    cả đơn vị dù các mapping còn lại dùng được.
    """
    persona_names = {persona.name for persona in personas}
    return {
        name: concepts
        for name, concepts in persona_item_mapping.items()
        if name in persona_names
    }


def _exception_chain(error: BaseException) -> list[BaseException]:
    """Lấy `cause`/`context` không lặp để phân biệt lỗi HTTP với lỗi semantic RAGAS."""
    pending = [error]
    seen: set[int] = set()
    chain: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        chain.append(current)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return chain


def _has_transport_error(error: BaseException) -> bool:
    """True khi request đã là lỗi API/network, không được RAGAS retry lần nữa."""
    transport_types = (
        DailyQuotaExhaustedError,
        APIStatusError,
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
    )
    return any(isinstance(item, transport_types) for item in _exception_chain(error))


def _is_semantic_ragas_error(error: BaseException) -> bool:
    """Chỉ các lỗi parse/validation sau HTTP 200 mới đủ điều kiện retry tầng RAGAS."""
    if _has_transport_error(error):
        return False
    return any(
        isinstance(item, (ValueError, KeyError, ValidationError))
        for item in _exception_chain(error)
    )


def _is_classified_ragas_error(error: BaseException) -> bool:
    """Lỗi RAGAS/API biết trước; `OSError`/`TypeError`/bug phải fail-fast."""
    return _has_transport_error(error) or _is_semantic_ragas_error(error)


def _error_type(error: BaseException | None) -> str:
    """Tên nguyên nhân sâu nhất có ích cho checkpoint, không chứa message người dùng."""
    if error is None:
        return "InvalidSample"
    return type(_exception_chain(error)[-1]).__name__


@dataclass
class CleanSingleHopSynthesizer(SingleHopSpecificQuerySynthesizer):
    """Single-hop luôn dùng `QueryStyle.PERFECT_GRAMMAR` (mục 4.3)."""

    def prepare_combinations(
        self,
        node: Node,
        terms: list[str],
        personas: list[Persona],
        persona_concepts: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        """Ép grammar và bỏ persona lạ trước lookup `PersonaList` của RAGAS 0.4.3."""
        return _force_perfect_grammar(
            super().prepare_combinations(
                node,
                terms,
                personas,
                _filter_persona_item_mapping(personas, persona_concepts),
            )
        )


@dataclass
class CleanMultiHopAbstractSynthesizer(MultiHopAbstractQuerySynthesizer):
    """Multi-hop abstract luôn dùng `QueryStyle.PERFECT_GRAMMAR`."""

    def prepare_combinations(
        self,
        nodes: Any,
        combinations: list[list[str]],
        personas: list[Persona],
        persona_item_mapping: dict[str, list[str]],
        property_name: str,
    ) -> list[dict[str, Any]]:
        """Ép grammar sạch và bỏ mapping persona không khớp trước khi gọi Ragas."""
        return _force_perfect_grammar(
            super().prepare_combinations(
                nodes,
                combinations,
                personas,
                _filter_persona_item_mapping(personas, persona_item_mapping),
                property_name,
            )
        )


@dataclass
class CleanMultiHopSpecificSynthesizer(MultiHopSpecificQuerySynthesizer):
    """Multi-hop specific luôn dùng `QueryStyle.PERFECT_GRAMMAR`."""

    def prepare_combinations(
        self,
        nodes: Any,
        combinations: list[list[str]],
        personas: list[Persona],
        persona_item_mapping: dict[str, list[str]],
        property_name: str,
    ) -> list[dict[str, Any]]:
        """Ép grammar sạch và bỏ mapping persona không khớp trước khi gọi Ragas."""
        return _force_perfect_grammar(
            super().prepare_combinations(
                nodes,
                combinations,
                personas,
                _filter_persona_item_mapping(personas, persona_item_mapping),
                property_name,
            )
        )


def build_run_config() -> RunConfig:
    """`RunConfig` không retry: router là chủ sở hữu retry HTTP duy nhất."""
    return RunConfig(
        max_workers=MAX_WORKERS,
        max_retries=MAX_RETRIES,
        max_wait=MAX_WAIT_SECONDS,
        exception_types=(RateLimitError,),
    )


def cap_token_limit(transforms: Any, limit: int) -> int:
    """Hạ `max_token_limit` của mọi `LLMBasedExtractor` trong `transforms`; trả số extractor đã đổi."""
    if isinstance(transforms, Parallel):
        transforms = transforms.transformations
    if isinstance(transforms, LLMBasedExtractor):
        transforms.max_token_limit = limit
        return 1
    if isinstance(transforms, list):
        return sum(cap_token_limit(item, limit) for item in transforms)
    return 0


def _build_groq_clients(settings: TestsetGeneratorSettings) -> list[ChatOpenAI]:
    """Khởi tạo 9 `ChatOpenAI` (Groq) độc lập tài khoản cho round-robin (mục 3.1)."""
    api_keys = (
        settings.api_key,
        settings.api_key_2,
        settings.api_key_3,
        settings.api_key_4,
        settings.api_key_5,
        settings.api_key_6,
        settings.api_key_7,
        settings.api_key_8,
        settings.api_key_9,
    )
    return [
        ChatOpenAI(
            base_url=_GROQ_OPENAI_BASE_URL,
            api_key=api_key,
            model=settings.model_name,
            max_retries=0,
            timeout=float(settings.timeout_seconds),
        )
        for api_key in api_keys
    ]


def _has_clusters(
    synthesizer: BaseSynthesizer[Any], graph: KnowledgeGraph, n: int
) -> bool:
    """Multi-hop cần cụm đoạn có quan hệ; KG nhỏ có thể không có -> ragas raise ValueError."""
    try:
        if isinstance(synthesizer, MultiHopAbstractQuerySynthesizer):
            return bool(synthesizer.get_node_clusters(graph, n))
        if isinstance(synthesizer, MultiHopSpecificQuerySynthesizer):
            return bool(synthesizer.get_node_clusters(graph))
    except ValueError:
        # KnowledgeGraph.find_n_indirect_clusters raise thay vì trả rỗng khi không có
        # quan hệ nào khớp điều kiện — coi như "không có cụm" (mục 4/4.5: bỏ, không bù).
        return False
    return True


class RagasUnitRunner:
    """Sinh câu hỏi cho một đơn vị bằng ragas: KG riêng của đơn vị + 3 synthesizer."""

    def __init__(
        self,
        settings: TestsetGeneratorSettings | None = None,
        embeddings_adapter: RagasEmbeddingsAdapter | None = None,
    ) -> None:
        settings = settings or TestsetGeneratorSettings()  # type: ignore[call-arg]
        self.router = GroqRoundRobinChatModel(clients=_build_groq_clients(settings))
        # Ragas chỉ đọc retry từ `llm.run_config` (không từ `run_config` truyền vào
        # `apply_transforms`, chỉ dùng cho `max_workers`), và `TestsetGenerator.generate` chỉ
        # đặt lại nó ở bước sinh câu. Gán ngay đây để dựng KG + `adapt_prompts` cũng dùng.
        self.run_config = build_run_config()
        self.llm = LangchainLLMWrapper(self.router, run_config=self.run_config)
        self.embeddings = LangchainEmbeddingsWrapper(
            embeddings_adapter or RagasEmbeddingsAdapter()
        )
        self._synthesizers: dict[str, BaseSynthesizer[Any]] | None = None

    def _get_synthesizers(self) -> dict[str, BaseSynthesizer[Any]]:
        """3 synthesizer đã `adapt_prompts` tiếng Việt — gọi LLM 1 lần rồi dùng lại mọi đơn vị."""
        if self._synthesizers is None:
            synthesizers: dict[str, BaseSynthesizer[Any]] = {
                "single_hop": CleanSingleHopSynthesizer(llm=self.llm),
                "abstract": CleanMultiHopAbstractSynthesizer(llm=self.llm),
                "specific": CleanMultiHopSpecificSynthesizer(llm=self.llm),
            }
            for synthesizer in synthesizers.values():
                adapted = asyncio.run(synthesizer.adapt_prompts(LANGUAGE, llm=self.llm))
                synthesizer.set_prompts(**adapted)
            self._synthesizers = synthesizers
        return self._synthesizers

    def _build_knowledge_graph(self, unit: EvalUnit) -> KnowledgeGraph:
        """Dựng KG riêng cho đơn vị (không nối quan hệ chéo đơn vị, mục 1).

        Chạy trong `reasoning_effort=low` (mục 3.2 C); thoát khỏi context dù có lỗi để
        bước sinh câu sau đó dùng lại mức mặc định.
        """
        with self.router.reasoning_effort(_KG_REASONING_EFFORT):
            document = Document(
                page_content=unit.text, metadata={"source": unit.source_document}
            )
            transforms = default_transforms(
                documents=[document], llm=self.llm, embedding_model=self.embeddings
            )
            capped = cap_token_limit(transforms, MAX_TOKEN_LIMIT)
            logger.info(
                "Hạ max_token_limit=%d cho %d extractor.", MAX_TOKEN_LIMIT, capped
            )
            graph = KnowledgeGraph(
                nodes=[
                    Node(
                        type=NodeType.DOCUMENT,
                        properties={
                            "page_content": document.page_content,
                            "document_metadata": document.metadata,
                        },
                    )
                ]
            )
            apply_transforms(graph, transforms, run_config=self.run_config)
        return graph

    def _query_distribution(
        self, graph: KnowledgeGraph, quota: QuestionQuota
    ) -> tuple[list[tuple[BaseSynthesizer[Any], float]], int]:
        """Danh sách (synthesizer, trọng số) + tổng số câu, bỏ loại có quota 0 hoặc không có cụm.

        Trọng số `(n - 0,5) / tổng`: ragas tính số câu bằng `ceil(tổng * trọng số)`, nên
        `n / tổng` có thể vượt `n` một đơn vị do sai số dấu phẩy động.
        """
        synthesizers = self._get_synthesizers()
        wanted: dict[str, int] = {}
        for kind in ("single_hop", "abstract", "specific"):
            count: int = getattr(quota, kind)
            if count > 0 and not _has_clusters(synthesizers[kind], graph, count):
                logger.warning(
                    "KG không có cụm cho loại %s: bỏ %d câu (không bù).", kind, count
                )
                continue
            if count > 0:
                wanted[kind] = count
        total = sum(wanted.values())
        return [
            (synthesizers[kind], (n - 0.5) / total) for kind, n in wanted.items()
        ], total

    def _obtain_knowledge_graph(
        self, unit: EvalUnit, path: Path, *, reuse: bool
    ) -> KnowledgeGraph:
        """Nạp KG đã lưu (khi `reuse`) hoặc dựng mới và lưu NGAY; trả KG.

        KG dựng xong là hoàn chỉnh (~30-160K token) nên lưu trước bước sinh câu: lỗi ở bước
        sinh câu không làm mất nó, lần sau đơn vị dở dùng lại thay vì dựng lại.
        """
        if reuse and path.exists():
            graph = _load_matching_graph(path, unit)
            if graph is not None:
                logger.info("Tái dùng KG đã lưu: %s", path)
                return graph
        graph = self._build_knowledge_graph(unit)
        _save_graph_atomic(graph, path)
        return graph

    def _generate_cases(
        self,
        unit: EvalUnit,
        graph: KnowledgeGraph,
        distribution: list[tuple[BaseSynthesizer[Any], float]],
        total: int,
    ) -> _Generation:
        """Sinh câu cho đơn vị, giữ phần đã xong khi lỗi giữa chừng (mục 3.3).

        Thay cho `TestsetGenerator.generate`: persona một lần, rồi từng loại một
        (scenario -> từng sample bọc try/except, đồng thời tối đa `MAX_WORKERS`).

        Raises:
            UnitGenerationError: Không sinh được câu nào mà có lỗi.
        """
        splits, _ = calculate_split_values(
            [weight for _, weight in distribution], total
        )
        planned = [
            (synthesizer, count)
            for (synthesizer, _), count in zip(distribution, splits, strict=True)
        ]
        try:
            personas = _generate_personas_once_more(graph, self.llm)
        except DailyQuotaExhaustedError:
            raise
        except Exception as error:
            if not _is_classified_ragas_error(error):
                raise
            skipped_types = {_question_type(synthesizer) for synthesizer, _ in planned}
            logger.warning(
                "Bỏ %d loại câu vì generate_personas không dùng được: %s.",
                len(skipped_types),
                type(error).__name__,
            )
            return _finish_generation(
                unit,
                _Sampling(
                    skipped_question_types=skipped_types,
                    last_error=error,
                ),
            )
        sampling = asyncio.run(_sample_all(planned, graph, personas))
        return _finish_generation(unit, sampling)

    def run_unit(
        self,
        unit: EvalUnit,
        quota: QuestionQuota,
        knowledge_graph_path: Path,
        *,
        reuse_knowledge_graph: bool,
    ) -> UnitResult:
        """Sinh câu hỏi cho `unit`; KG được lưu ngay sau khi dựng xong (trước khi sinh câu)."""
        self._get_synthesizers()  # lần đầu gọi LLM (adapt_prompts); không tính vào đơn vị
        calls_before = sum(self.router.call_counts)
        tokens_before = self.router.token_totals
        try:
            try:
                graph = self._obtain_knowledge_graph(
                    unit, knowledge_graph_path, reuse=reuse_knowledge_graph
                )
            except DailyQuotaExhaustedError:
                raise
            except Exception as error:
                if not _is_classified_ragas_error(error):
                    raise
                usage = _token_delta(tokens_before, self.router.token_totals)
                raise UnitGenerationError(
                    unit_key(unit),
                    "dựng knowledge graph thất bại",
                    stage="knowledge_graph",
                    error_type=type(error).__name__,
                    attempts=sum(self.router.call_counts) - calls_before,
                    tokens=sum(item.total_tokens for item in usage),
                    reasoning_tokens=sum(item.reasoning_tokens for item in usage),
                ) from error
            distribution, total = self._query_distribution(graph, quota)
            if total > 0:
                generation = self._generate_cases(unit, graph, distribution, total)
            elif quota.total > 0:
                skipped_types = {
                    kind
                    for kind in ("single_hop", "abstract", "specific")
                    if getattr(quota, kind) > 0
                }
                generation = _finish_generation(
                    unit,
                    _Sampling(skipped_question_types=skipped_types),
                )
            else:
                generation = _Generation()
        except UnitGenerationError as error:
            usage = _token_delta(tokens_before, self.router.token_totals)
            enriched = UnitGenerationError(
                error.unit_key,
                "không sinh được câu hợp lệ",
                stage=error.stage,
                error_type=error.error_type,
                attempts=sum(self.router.call_counts) - calls_before,
                tokens=sum(item.total_tokens for item in usage),
                reasoning_tokens=sum(item.reasoning_tokens for item in usage),
                skipped_samples=error.skipped_samples,
                skipped_question_types=error.skipped_question_types,
            )
            raise enriched from error
        finally:
            # Ghi cả khi đơn vị lỗi: token đã đốt vẫn cần thấy được trong log.
            usage = _token_delta(tokens_before, self.router.token_totals)
            _log_token_usage(unit, usage)
        return UnitResult(
            cases=generation.cases,
            llm_calls=sum(self.router.call_counts) - calls_before,
            tokens=sum(item.total_tokens for item in usage),
            reasoning_tokens=sum(item.reasoning_tokens for item in usage),
            skipped_samples=generation.skipped,
            skipped_question_types=generation.skipped_question_types,
            interruption=generation.interruption,
        )


@dataclass
class _Sampling:
    """Tích luỹ kết quả bước sinh sample của một đơn vị (chỉ ghi từ event loop, không cần lock)."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    skipped: int = 0
    skipped_question_types: set[str] = field(default_factory=set)
    last_error: BaseException | None = None
    quota_error: DailyQuotaExhaustedError | None = None
    interruption: BaseException | None = None  # lỗi sinh scenario của một loại


@dataclass
class _Generation:
    """Kết quả bước sinh câu đã chuyển thành `GoldenTestCase`."""

    cases: list[GoldenTestCase] = field(default_factory=list)
    skipped: int = 0
    skipped_question_types: set[str] = field(default_factory=set)
    interruption: BaseException | None = None


async def _sample_one(
    synthesizer: BaseSynthesizer[Any],
    scenario: BaseScenario,
    semaphore: asyncio.Semaphore,
    sampling: _Sampling,
) -> dict[str, Any] | None:
    """Sinh một sample; lỗi thì bỏ (chỉ log tên loại lỗi, không log nội dung — mục 8)."""
    async with semaphore:
        if sampling.quota_error is not None:
            return None  # breaker đã bật: không đưa sample mới
        for attempt in range(2):
            try:
                sample = await synthesizer.generate_sample(scenario)
            except DailyQuotaExhaustedError as error:
                sampling.quota_error = error
                return None
            except Exception as error:
                if _is_semantic_ragas_error(error) and attempt == 0:
                    continue
                if not _is_classified_ragas_error(error):
                    raise
                sampling.skipped += 1
                sampling.last_error = error
                logger.warning(
                    "Bỏ 1 sample của %s sau lỗi đã phân loại: %s.",
                    synthesizer.name,
                    type(error).__name__,
                )
                return None
            else:
                break
    row: dict[str, Any] = sample.model_dump(exclude_none=True)
    row["synthesizer_name"] = synthesizer.name
    return row


async def _sample_all(
    planned: list[tuple[BaseSynthesizer[Any], int]],
    graph: KnowledgeGraph,
    personas: list[Persona],
) -> _Sampling:
    """Từng loại: sinh scenario rồi sinh sample đồng thời (tối đa `MAX_WORKERS`).

    Dừng ngay khi breaker báo hết quota ngày (giữ sample đã xong) hoặc khi sinh scenario của
    một loại lỗi (giữ các loại trước đó).
    """
    sampling = _Sampling()
    semaphore = asyncio.Semaphore(MAX_WORKERS)
    for synthesizer, count in planned:
        try:
            scenarios = await _generate_scenarios_once_more(
                synthesizer, count, graph, personas
            )
        except DailyQuotaExhaustedError as error:
            sampling.quota_error = error
            break
        except Exception as error:
            if not _is_classified_ragas_error(error):
                raise
            kind = _question_type(synthesizer)
            sampling.skipped_question_types.add(kind)
            sampling.last_error = error
            logger.warning(
                "Bỏ loại câu %s sau 2 lần tạo scenario lỗi: %s.",
                kind,
                type(error).__name__,
            )
            continue
        rows = await asyncio.gather(
            *(_sample_one(synthesizer, item, semaphore, sampling) for item in scenarios)
        )
        sampling.rows.extend(row for row in rows if row is not None)
        if sampling.quota_error is not None:
            break
    return sampling


def _finish_generation(unit: EvalUnit, sampling: _Sampling) -> _Generation:
    """Đổi sample thành câu; quota dương không bao giờ được checkpoint `done` rỗng."""
    cases = _to_cases(sampling.rows)
    skipped = sampling.skipped + len(sampling.rows) - len(cases)
    interruption = sampling.quota_error or sampling.interruption
    if not cases and (
        skipped
        or sampling.skipped_question_types
        or isinstance(interruption, DailyQuotaExhaustedError)
    ):
        raise UnitGenerationError(
            unit_key(unit),
            f"không sinh được câu nào ({skipped} sample bị bỏ)",
            stage="generation",
            error_type=_error_type(sampling.last_error or interruption),
            skipped_samples=skipped,
            skipped_question_types=sampling.skipped_question_types,
        ) from (interruption or sampling.last_error)
    return _Generation(
        cases=cases,
        skipped=skipped,
        skipped_question_types=sampling.skipped_question_types,
        interruption=interruption,
    )


def _question_type(synthesizer: BaseSynthesizer[Any]) -> str:
    """Ánh xạ synthesizer nội bộ sang khoá progress, không dựa vào text exception RAGAS."""
    name = getattr(synthesizer, "name", "")
    if "multi_hop_abstract" in name:
        return "abstract"
    if "multi_hop_specific" in name:
        return "specific"
    if isinstance(synthesizer, CleanMultiHopAbstractSynthesizer):
        return "abstract"
    if isinstance(synthesizer, CleanMultiHopSpecificSynthesizer):
        return "specific"
    return "single_hop"


def _generate_personas_once_more(graph: KnowledgeGraph, llm: Any) -> list[Persona]:
    """Sinh persona tối đa hai lần; chỉ lỗi RAGAS sau HTTP 200 mới đến retry thứ hai."""
    for attempt in range(2):
        try:
            return generate_personas_from_kg(
                kg=graph, llm=llm, num_personas=NUM_PERSONAS
            )
        except Exception as error:
            if not _is_semantic_ragas_error(error) or attempt == 1:
                raise
    raise AssertionError("vòng retry persona phải return hoặc raise")


async def _generate_scenarios_once_more(
    synthesizer: BaseSynthesizer[Any],
    count: int,
    graph: KnowledgeGraph,
    personas: list[Persona],
) -> list[BaseScenario]:
    """Tạo scenario tối đa hai lần; không nhân retry ở tầng SDK/RAGAS."""
    for attempt in range(2):
        try:
            return await synthesizer.generate_scenarios(count, graph, personas)
        except Exception as error:
            if not _is_semantic_ragas_error(error) or attempt == 1:
                raise
    raise AssertionError("vòng retry scenario phải return hoặc raise")


def _token_delta(
    before: list[TokenTotals], after: list[TokenTotals]
) -> list[TokenTotals]:
    """Token của riêng đơn vị theo từng tài khoản (hiệu số sau - trước, như `llm_calls`)."""
    return [
        TokenTotals(
            prompt_tokens=new.prompt_tokens - old.prompt_tokens,
            completion_tokens=new.completion_tokens - old.completion_tokens,
            reasoning_tokens=new.reasoning_tokens - old.reasoning_tokens,
        )
        for old, new in zip(before, after, strict=True)
    ]


def _log_token_usage(unit: EvalUnit, usage: list[TokenTotals]) -> None:
    """Log INFO tổng token và token theo tài khoản (chỉ số thứ tự và số, không có key)."""
    logger.info(
        "Token %s: tổng %d (vào %d, ra %d, suy luận %d); theo tài khoản: %s.",
        unit_key(unit),
        sum(item.total_tokens for item in usage),
        sum(item.prompt_tokens for item in usage),
        sum(item.completion_tokens for item in usage),
        sum(item.reasoning_tokens for item in usage),
        ", ".join(f"#{n}={item.total_tokens}" for n, item in enumerate(usage, start=1)),
    )


def _load_matching_graph(path: Path, unit: EvalUnit) -> KnowledgeGraph | None:
    """Nạp KG; `None` (kèm log) nếu file hỏng hoặc dựng từ văn bản khác `unit.text`."""
    try:
        graph = KnowledgeGraph.load(path)
    except ValueError, KeyError, TypeError, OSError:
        # ValueError gồm JSONDecodeError/UnicodeDecodeError; KeyError/TypeError: JSON hợp lệ
        # nhưng sai cấu trúc (`{}`, `[]`). Chỉ log đường dẫn, không log nội dung KG.
        logger.warning("KG %s không đọc được: dựng lại.", path)
        return None
    if not any(
        node.type == NodeType.DOCUMENT
        and node.properties.get("page_content") == unit.text
        for node in graph.nodes
    ):
        logger.warning("KG %s không khớp văn bản đơn vị hiện tại: dựng lại.", path)
        return None
    return graph


def _save_graph_atomic(graph: KnowledgeGraph, path: Path) -> None:
    """Ghi file tạm rồi đổi tên: chết giữa chừng không để lại KG dở dang."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    graph.save(temp_path)
    os.replace(temp_path, path)


def _to_cases(samples: list[dict[str, Any]]) -> list[GoldenTestCase]:
    """`Testset.to_list()` -> `GoldenTestCase`; bỏ (có log số lượng) mẫu thiếu cột bắt buộc."""
    cases: list[GoldenTestCase] = []
    skipped = 0
    for sample in samples:
        try:
            case = GoldenTestCase.model_validate(sample)
        except ValidationError:
            skipped += 1
            continue
        if case.empty_required_fields():
            skipped += 1
            continue
        cases.append(case)
    if skipped:
        logger.warning(
            "Bỏ %d mẫu ragas thiếu user_input/reference/reference_contexts.", skipped
        )
    return cases
