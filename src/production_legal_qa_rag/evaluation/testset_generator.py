"""Điều phối sinh golden testset RAGAS Phase 1 (evaluation_spec.md mục 4).

Khởi tạo `ragas.testset.TestsetGenerator` với `generator_llm` round-robin 3
tài khoản Groq (`GroqRoundRobinChatModel`, mục 3.1) và `generator_embeddings`
qua `RagasEmbeddingsAdapter` (mục 3, 6), chạy `generate_with_langchain_docs`
trên toàn bộ `data/markdown/*.md`, rồi lưu `KnowledgeGraph` + `golden_testset.json`
(mục 2). Không tách bước build KG và sinh câu hỏi thành 2 lệnh CLI riêng — chấp
nhận chạy lại toàn bộ nếu lỗi giữa chừng (mục 4, 8).

Chữ ký `TestsetGenerator(llm=..., embedding_model=...)` và
`generate_with_langchain_docs(documents, testset_size=...)` đã được xác nhận
bằng `inspect.signature`/đọc trực tiếp mã nguồn `ragas==0.4.3` cài thật
(mục 10.4) — không suy diễn từ tài liệu web khác version. Tương tự,
`generator.generate(testset_size=...)` (không truyền `documents`) dùng lại
đúng `self.knowledge_graph` đã có sẵn, không build lại — đây là cơ chế tái
dùng `KnowledgeGraph` đã lưu (mục 9.5).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

from langchain_openai import ChatOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.executor import Executor
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph

from production_legal_qa_rag.config import TestsetGeneratorSettings
from production_legal_qa_rag.evaluation.corpus_loader import (
    DEFAULT_MARKDOWN_DIR,
    load_markdown_documents,
)
from production_legal_qa_rag.evaluation.embeddings_adapter import (
    RagasEmbeddingsAdapter,
)
from production_legal_qa_rag.evaluation.groq_round_robin import (
    GroqRoundRobinChatModel,
)
from production_legal_qa_rag.evaluation.models import GoldenTestCase

logger = logging.getLogger(__name__)

# Hằng số nội bộ module, KHÔNG phải biến môi trường (mục 4) -- build KnowledgeGraph
# là chi phí cố định theo corpus, chỉ bước sinh câu hỏi tỉ lệ theo số này.
TESTSET_SIZE: Final = 360

DEFAULT_OUTPUT_DIR: Final = Path("data/eval")
GOLDEN_TESTSET_FILENAME: Final = "golden_testset.json"
KNOWLEDGE_GRAPH_FILENAME: Final = "knowledge_graph.json"

# Groq công bố endpoint OpenAI-compatible chính thức (generation_spec.md mục 8),
# cùng pattern với generation/generator.py.
_GROQ_OPENAI_BASE_URL: Final = "https://api.groq.com/openai/v1"


def _build_groq_clients(settings: TestsetGeneratorSettings) -> list[ChatOpenAI]:
    """Khởi tạo 3 `ChatOpenAI` (Groq) độc lập tài khoản cho round-robin (mục 3.1)."""
    return [
        ChatOpenAI(
            base_url=_GROQ_OPENAI_BASE_URL,
            api_key=api_key,
            model=settings.model_name,
            max_retries=settings.max_retries,
            timeout=float(settings.timeout_seconds),
        )
        for api_key in (settings.api_key, settings.api_key_2, settings.api_key_3)
    ]


def build_testset_generator(
    settings: TestsetGeneratorSettings | None = None,
    embeddings_adapter: RagasEmbeddingsAdapter | None = None,
) -> TestsetGenerator:
    """Dựng `TestsetGenerator` với `generator_llm` round-robin và HF embeddings.

    Args:
        settings: Cấu hình 3 key Groq round-robin; `None` thì đọc từ `.env`.
        embeddings_adapter: Adapter embeddings; `None` thì tạo mới bằng
            `EmbeddingSettings` mặc định (mục 6) -- tham số này chỉ phục vụ
            dependency injection cho test, không phải cấu hình mới.

    Returns:
        `TestsetGenerator` sẵn sàng gọi `generate_with_langchain_docs`/`generate`.
    """
    settings = settings or TestsetGeneratorSettings()  # type: ignore[call-arg]
    clients = _build_groq_clients(settings)
    generator_llm = LangchainLLMWrapper(GroqRoundRobinChatModel(clients=clients))
    generator_embeddings = LangchainEmbeddingsWrapper(
        embeddings_adapter or RagasEmbeddingsAdapter()
    )
    return TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)


def generate_golden_testset(
    markdown_dir: Path = DEFAULT_MARKDOWN_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    settings: TestsetGeneratorSettings | None = None,
    reuse_knowledge_graph: bool = False,
    testset_size: int = TESTSET_SIZE,
) -> list[GoldenTestCase]:
    """Sinh golden testset từ corpus markdown và lưu ra `output_dir` (mục 4, 8, 9).

    Args:
        markdown_dir: Thư mục chứa `data/markdown/*.md`.
        output_dir: Thư mục lưu `golden_testset.json` + `knowledge_graph.json`.
        settings: Cấu hình Groq round-robin; `None` thì đọc từ `.env`.
        reuse_knowledge_graph: `True` thì nạp lại `knowledge_graph.json` đã lưu
            (nếu tồn tại) và gọi `generate()` thay vì `generate_with_langchain_docs()`
            để không build lại đồ thị (mục 9.5) -- tiết kiệm lượt gọi Groq.
        testset_size: Số câu hỏi cần sinh; mặc định `TESTSET_SIZE=360` (mục 1, 4).

    Returns:
        Danh sách `GoldenTestCase` đã sinh, cùng thứ tự với `testset.to_list()`.

    Raises:
        FileNotFoundError: `markdown_dir` thiếu hoặc không có file `.md` (fail
            fast trước khi gọi Groq, mục 8).
        ValueError: Một file markdown trống (mục 8).
        Exception: Bất kỳ lỗi nào từ Groq/ragas trong lúc build KnowledgeGraph
            hoặc sinh câu hỏi được log rồi raise nguyên vẹn -- không lưu file
            output dở dang (mục 8).
    """
    documents = load_markdown_documents(markdown_dir)
    generator = build_testset_generator(settings)

    knowledge_graph_path = output_dir / KNOWLEDGE_GRAPH_FILENAME
    try:
        if reuse_knowledge_graph and knowledge_graph_path.exists():
            logger.info("Tái dùng KnowledgeGraph đã lưu: %s", knowledge_graph_path)
            generator.knowledge_graph = KnowledgeGraph.load(knowledge_graph_path)
            testset = generator.generate(testset_size=testset_size)
        else:
            testset = generator.generate_with_langchain_docs(
                documents, testset_size=testset_size
            )
    except Exception:
        # Không log nội dung câu hỏi/context (mục 8) -- chỉ log rằng đã lỗi và
        # dừng, không lưu file output dở dang.
        logger.exception(
            "Lỗi khi build KnowledgeGraph hoặc sinh câu hỏi, không lưu output."
        )
        raise

    if isinstance(testset, Executor):
        # Không thể xảy ra: không truyền return_executor=True ở trên. Guard
        # này chỉ để mypy hẹp kiểu `Testset | Executor` và fail rõ nếu code
        # sau này đổi mà quên cập nhật.
        raise TypeError("generate_golden_testset() không hỗ trợ return_executor=True.")

    golden_cases = [GoldenTestCase(**sample) for sample in testset.to_list()]

    output_dir.mkdir(parents=True, exist_ok=True)
    generator.knowledge_graph.save(knowledge_graph_path)
    testset_path = output_dir / GOLDEN_TESTSET_FILENAME
    testset_path.write_text(
        json.dumps(
            [case.model_dump() for case in golden_cases],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Đã lưu %d câu hỏi vào %s, KnowledgeGraph vào %s.",
        len(golden_cases),
        testset_path,
        knowledge_graph_path,
    )
    return golden_cases
