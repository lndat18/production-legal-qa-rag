"""Cài đặt `UnitRunner` bằng `ragas` + 6 tài khoản Groq round-robin (evaluation_spec.md mục 3, 4.3).

Toàn bộ code chạm `ragas` nằm ở đây (import ở mức module) để `testset_generator.py`
không cần dependency-group `eval`; chỉ import module này qua `build_unit_runner`.

Chữ ký ragas đã đọc trực tiếp trên `ragas==0.4.3` cài thật: cả 3 synthesizer
(`prepare_combinations` của single-hop nhận `(node, terms, personas, persona_concepts)`,
của multi-hop nhận `(nodes, combinations, personas, persona_item_mapping,
property_name)`) đều trả `list[dict]` có khoá `"styles"` — nên một hàm ép
`PERFECT_GRAMMAR` dùng chung, bọc bằng `*args, **kwargs`, đúng cho cả ba.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, InternalServerError, RateLimitError
from pydantic import ValidationError
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.executor import Executor
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.synthesizers.base import BaseSynthesizer, QueryStyle
from ragas.testset.synthesizers.multi_hop import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.transforms import Parallel, apply_transforms, default_transforms
from ragas.testset.transforms.base import LLMBasedExtractor

from production_legal_qa_rag.config import TestsetGeneratorSettings
from production_legal_qa_rag.evaluation.embeddings_adapter import (
    RagasEmbeddingsAdapter,
)
from production_legal_qa_rag.evaluation.groq_round_robin import (
    GroqRoundRobinChatModel,
)
from production_legal_qa_rag.evaluation.models import GoldenTestCase
from production_legal_qa_rag.evaluation.testset_generator import (
    QuestionQuota,
    UnitResult,
)
from production_legal_qa_rag.evaluation.unit_splitter import EvalUnit

logger = logging.getLogger(__name__)

# Groq free: TPM 8K/tài khoản, mặc định ragas 32.000 token/lượt bị 413 (mục 3.1).
MAX_TOKEN_LIMIT: Final = 4_000
MAX_WORKERS: Final = 4
LANGUAGE: Final = "vietnamese"

# Retry của ragas (mục 3.1, 8). Mặc định `RunConfig` là 10 lần với MỌI `Exception`: lỗi tất
# định (400/401/413) bị thử lại vô ích và khi hết quota mỗi lượt gọi đốt hàng trăm request
# (mỗi lượt đã là 6 tài khoản x retry SDK). Chỉ thử lại lỗi tạm thời, ít lần, chờ ngắn.
MAX_RETRIES: Final = 3
MAX_WAIT_SECONDS: Final = 30
# 429 (`RateLimitError`, theo phút), timeout + lỗi kết nối (`APIConnectionError`), 5xx
# (`InternalServerError`). Cố ý không có `DailyQuotaExhaustedError`, 400/401/403/413.
RETRYABLE_EXCEPTIONS: Final = (RateLimitError, APIConnectionError, InternalServerError)

# Groq công bố endpoint OpenAI-compatible chính thức (generation_spec.md mục 8),
# cùng pattern với generation/generator.py.
_GROQ_OPENAI_BASE_URL: Final = "https://api.groq.com/openai/v1"


def _force_perfect_grammar(combinations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ragas mặc định trộn 4 `QueryStyle`, 3/4 là nhiễu (sai chính tả/ngữ pháp); ép sạch."""
    for combination in combinations:
        combination["styles"] = [QueryStyle.PERFECT_GRAMMAR]
    return combinations


@dataclass
class CleanSingleHopSynthesizer(SingleHopSpecificQuerySynthesizer):
    """Single-hop luôn dùng `QueryStyle.PERFECT_GRAMMAR` (mục 4.3)."""

    def prepare_combinations(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return _force_perfect_grammar(super().prepare_combinations(*args, **kwargs))


@dataclass
class CleanMultiHopAbstractSynthesizer(MultiHopAbstractQuerySynthesizer):
    """Multi-hop abstract luôn dùng `QueryStyle.PERFECT_GRAMMAR`."""

    def prepare_combinations(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return _force_perfect_grammar(super().prepare_combinations(*args, **kwargs))


@dataclass
class CleanMultiHopSpecificSynthesizer(MultiHopSpecificQuerySynthesizer):
    """Multi-hop specific luôn dùng `QueryStyle.PERFECT_GRAMMAR`."""

    def prepare_combinations(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return _force_perfect_grammar(super().prepare_combinations(*args, **kwargs))


def build_run_config() -> RunConfig:
    """`RunConfig` dùng chung cho dựng KG, `adapt_prompts` và sinh câu (chỉ retry lỗi tạm thời)."""
    return RunConfig(
        max_workers=MAX_WORKERS,
        max_retries=MAX_RETRIES,
        max_wait=MAX_WAIT_SECONDS,
        exception_types=RETRYABLE_EXCEPTIONS,
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
    """Khởi tạo 6 `ChatOpenAI` (Groq) độc lập tài khoản cho round-robin (mục 3.1)."""
    api_keys = (
        settings.api_key,
        settings.api_key_2,
        settings.api_key_3,
        settings.api_key_4,
        settings.api_key_5,
        settings.api_key_6,
    )
    return [
        ChatOpenAI(
            base_url=_GROQ_OPENAI_BASE_URL,
            api_key=api_key,
            model=settings.model_name,
            max_retries=settings.max_retries,
            timeout=float(settings.timeout_seconds),
        )
        for api_key in api_keys
    ]


def _has_clusters(
    synthesizer: BaseSynthesizer[Any], graph: KnowledgeGraph, n: int
) -> bool:
    """Multi-hop cần cụm đoạn có quan hệ; KG nhỏ có thể không có -> ragas sẽ raise ValueError."""
    if isinstance(synthesizer, MultiHopAbstractQuerySynthesizer):
        return bool(synthesizer.get_node_clusters(graph, n))
    if isinstance(synthesizer, MultiHopSpecificQuerySynthesizer):
        return bool(synthesizer.get_node_clusters(graph))
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
        """Dựng KG riêng cho đơn vị (không nối quan hệ chéo đơn vị, mục 1)."""
        document = Document(
            page_content=unit.text, metadata={"source": unit.source_document}
        )
        transforms = default_transforms(
            documents=[document], llm=self.llm, embedding_model=self.embeddings
        )
        capped = cap_token_limit(transforms, MAX_TOKEN_LIMIT)
        logger.info("Hạ max_token_limit=%d cho %d extractor.", MAX_TOKEN_LIMIT, capped)
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
        graph: KnowledgeGraph,
        distribution: list[tuple[BaseSynthesizer[Any], float]],
        total: int,
    ) -> list[GoldenTestCase]:
        generator = TestsetGenerator(llm=self.llm, embedding_model=self.embeddings)
        generator.knowledge_graph = graph
        testset = generator.generate(
            testset_size=total,
            query_distribution=distribution,
            run_config=self.run_config,
        )
        if isinstance(testset, Executor):
            raise TypeError("RagasUnitRunner không hỗ trợ return_executor=True.")
        return _to_cases(testset.to_list())

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
        graph = self._obtain_knowledge_graph(
            unit, knowledge_graph_path, reuse=reuse_knowledge_graph
        )
        distribution, total = self._query_distribution(graph, quota)
        cases = self._generate_cases(graph, distribution, total) if total > 0 else []
        return UnitResult(
            cases=cases, llm_calls=sum(self.router.call_counts) - calls_before
        )


def _load_matching_graph(path: Path, unit: EvalUnit) -> KnowledgeGraph | None:
    """Nạp KG; `None` (kèm log) nếu file hỏng hoặc dựng từ văn bản khác `unit.text`."""
    try:
        graph = KnowledgeGraph.load(path)
    except ValueError:
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
