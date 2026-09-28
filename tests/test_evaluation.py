"""Unit test cho package `evaluation/` — điều phối `ragas` (Phase 1 sinh golden testset).

Mọi interaction Groq/HuggingFace/`ragas.testset.TestsetGenerator` trong file
này dùng fake đã bị monkeypatch — không gọi dịch vụ ngoài, không build
`KnowledgeGraph`/sinh câu hỏi thật (evaluation_spec.md mục 8, 10). Chữ ký thật
của `TestsetGenerator`/`generate_with_langchain_docs` đã được xác nhận trực
tiếp trên `ragas==0.4.3` cài thật bằng `inspect.signature` lúc implement
(mục 10.4), không phải qua test này.

`ragas` chỉ nằm trong dependency-group `eval` (`uv sync` mặc định — venv
production — không cài, xem [dependency-groups] trong `pyproject.toml`), nên
toàn bộ module này được skip nếu chạy trên venv không có `eval`
(`uv run --group eval --no-group production pytest` để chạy thật). Các test
không cần `ragas` (models, corpus_loader, embeddings_adapter, groq_round_robin,
`TestsetGeneratorSettings`) nằm ở `test_evaluation_components.py` và luôn chạy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "ragas",
    reason="ragas chỉ có trong dependency-group `eval` (`uv sync --group eval`)",
)

from typer.testing import CliRunner

from production_legal_qa_rag.config import EmbeddingSettings, TestsetGeneratorSettings
from production_legal_qa_rag.evaluation import testset_generator
from production_legal_qa_rag.evaluation.embeddings_adapter import RagasEmbeddingsAdapter
from production_legal_qa_rag.evaluation.groq_round_robin import GroqRoundRobinChatModel
from production_legal_qa_rag.evaluation.models import GoldenTestCase

# ==========================================================================
# testset_generator.py -- điều phối TestsetGenerator (mục 4, 8, 9); toàn bộ
# tương tác ragas.testset.TestsetGenerator được fake, không build KG/sinh câu
# hỏi thật.
# ==========================================================================


def _testset_settings() -> TestsetGeneratorSettings:
    return TestsetGeneratorSettings(  # type: ignore[call-arg]
        GROQ_API_KEY="key-1", GROQ_API_KEY_2="key-2", GROQ_API_KEY_3="key-3"
    )


def _sample(index: int) -> dict[str, Any]:
    return {
        "user_input": f"Câu hỏi {index}?",
        "reference": f"Đáp án {index}.",
        "reference_contexts": [f"Ngữ cảnh {index}"],
        "synthesizer_name": "single_hop_specific_query_synthesizer",
    }


class _FakeKnowledgeGraph:
    """Ghi lại đường dẫn `save()` được gọi để kiểm tra thay vì I/O ragas thật."""

    def __init__(self, marker: str = "built") -> None:
        self.marker = marker
        self.saved_to: Path | None = None

    def save(self, path: Path) -> None:
        self.saved_to = Path(path)


class _FakeTestset:
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self._samples = samples

    def to_list(self) -> list[dict[str, Any]]:
        return self._samples


class _FakeTestsetGenerator:
    """Fake thay cho `ragas.testset.TestsetGenerator` -- không gọi Groq/HF thật."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        *,
        error: Exception | None = None,
    ) -> None:
        self.knowledge_graph: Any = _FakeKnowledgeGraph()
        self._samples = samples
        self._error = error
        self.generate_with_langchain_docs_calls: list[tuple[Any, int]] = []
        self.generate_calls: list[int] = []

    def generate_with_langchain_docs(
        self, documents: Any, testset_size: int
    ) -> _FakeTestset:
        self.generate_with_langchain_docs_calls.append((documents, testset_size))
        if self._error is not None:
            raise self._error
        return _FakeTestset(self._samples)

    def generate(self, testset_size: int) -> _FakeTestset:
        self.generate_calls.append(testset_size)
        if self._error is not None:
            raise self._error
        return _FakeTestset(self._samples)


def test_build_groq_clients_tao_dung_3_client_doc_lap_tai_khoan():
    clients = testset_generator._build_groq_clients(_testset_settings())

    assert [client.openai_api_key.get_secret_value() for client in clients] == [
        "key-1",
        "key-2",
        "key-3",
    ]
    assert all(client.model_name == "openai/gpt-oss-120b" for client in clients)


def test_build_groq_clients_tro_dung_groq_base_url_va_forward_retry_timeout():
    settings = TestsetGeneratorSettings(  # type: ignore[call-arg]
        GROQ_API_KEY="key-1",
        GROQ_API_KEY_2="key-2",
        GROQ_API_KEY_3="key-3",
        max_retries=5,
        timeout_seconds=90,
    )

    clients = testset_generator._build_groq_clients(settings)

    assert all(
        client.openai_api_base == testset_generator._GROQ_OPENAI_BASE_URL
        for client in clients
    )
    assert all(client.max_retries == 5 for client in clients)
    assert all(client.request_timeout == 90.0 for client in clients)


def test_build_testset_generator_wire_dung_llm_va_embeddings(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    adapter = RagasEmbeddingsAdapter(EmbeddingSettings(), client=object())  # type: ignore[arg-type]

    generator = testset_generator.build_testset_generator(
        _testset_settings(), embeddings_adapter=adapter
    )

    assert isinstance(generator.llm.langchain_llm, GroqRoundRobinChatModel)
    assert len(generator.llm.langchain_llm.clients) == 3
    assert generator.embedding_model.embeddings is adapter


def test_generate_golden_testset_fail_fast_truoc_khi_dung_groq(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def _khong_duoc_goi(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("build_testset_generator không được gọi khi corpus lỗi")

    monkeypatch.setattr(testset_generator, "build_testset_generator", _khong_duoc_goi)

    with pytest.raises(FileNotFoundError):
        testset_generator.generate_golden_testset(
            tmp_path / "khong-ton-tai", tmp_path / "out"
        )


def test_generate_golden_testset_luu_ca_2_file_va_tra_ve_golden_test_case(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    (markdown_dir / "luat.md").write_text("Nội dung luật", encoding="utf-8")
    output_dir = tmp_path / "out"

    fake_generator = _FakeTestsetGenerator([_sample(1), _sample(2)])
    monkeypatch.setattr(
        testset_generator, "build_testset_generator", lambda *_a, **_kw: fake_generator
    )

    cases = testset_generator.generate_golden_testset(
        markdown_dir, output_dir, testset_size=2
    )

    assert cases == [GoldenTestCase(**_sample(1)), GoldenTestCase(**_sample(2))]
    assert len(fake_generator.generate_with_langchain_docs_calls) == 1
    assert fake_generator.generate_with_langchain_docs_calls[0][1] == 2
    assert fake_generator.knowledge_graph.saved_to == (
        output_dir / "knowledge_graph.json"
    )

    saved_testset = json.loads(
        (output_dir / "golden_testset.json").read_text(encoding="utf-8")
    )
    assert saved_testset == [_sample(1), _sample(2)]


def test_generate_golden_testset_tai_dung_knowledge_graph_khi_co_flag_va_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    (markdown_dir / "luat.md").write_text("Nội dung luật", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "knowledge_graph.json").write_text("{}", encoding="utf-8")

    fake_generator = _FakeTestsetGenerator([_sample(1)])
    loaded_graph = _FakeKnowledgeGraph(marker="loaded")
    monkeypatch.setattr(
        testset_generator, "build_testset_generator", lambda *_a, **_kw: fake_generator
    )
    monkeypatch.setattr(
        testset_generator.KnowledgeGraph, "load", lambda _path: loaded_graph
    )

    testset_generator.generate_golden_testset(
        markdown_dir, output_dir, reuse_knowledge_graph=True, testset_size=1
    )

    assert fake_generator.generate_with_langchain_docs_calls == []
    assert fake_generator.generate_calls == [1]
    assert fake_generator.knowledge_graph is loaded_graph
    assert loaded_graph.saved_to == output_dir / "knowledge_graph.json"


def test_generate_golden_testset_khong_co_file_kg_thi_van_build_lai(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    (markdown_dir / "luat.md").write_text("Nội dung luật", encoding="utf-8")
    output_dir = tmp_path / "out"  # chưa có knowledge_graph.json

    fake_generator = _FakeTestsetGenerator([_sample(1)])
    monkeypatch.setattr(
        testset_generator, "build_testset_generator", lambda *_a, **_kw: fake_generator
    )

    testset_generator.generate_golden_testset(
        markdown_dir, output_dir, reuse_knowledge_graph=True, testset_size=1
    )

    assert fake_generator.generate_calls == []
    assert len(fake_generator.generate_with_langchain_docs_calls) == 1


def test_generate_golden_testset_loi_groq_khong_luu_file_do_dang(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    (markdown_dir / "luat.md").write_text("Nội dung luật", encoding="utf-8")
    output_dir = tmp_path / "out"

    fake_generator = _FakeTestsetGenerator([], error=RuntimeError("Groq lỗi"))
    monkeypatch.setattr(
        testset_generator, "build_testset_generator", lambda *_a, **_kw: fake_generator
    )

    with pytest.raises(RuntimeError, match="Groq lỗi"):
        testset_generator.generate_golden_testset(markdown_dir, output_dir)

    assert not output_dir.exists()


def test_generate_golden_testset_tu_choi_executor_tra_ve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    markdown_dir = tmp_path / "markdown"
    markdown_dir.mkdir()
    (markdown_dir / "luat.md").write_text("Nội dung luật", encoding="utf-8")

    class _FakeExecutorMarker:
        """Đứng thay cho `ragas.executor.Executor` trong test isinstance."""

    fake_generator = _FakeTestsetGenerator([])

    def _generate_with_langchain_docs(
        _documents: Any, testset_size: int
    ) -> _FakeExecutorMarker:
        del testset_size
        return _FakeExecutorMarker()

    fake_generator.generate_with_langchain_docs = _generate_with_langchain_docs  # type: ignore[method-assign]
    monkeypatch.setattr(
        testset_generator, "build_testset_generator", lambda *_a, **_kw: fake_generator
    )
    monkeypatch.setattr(testset_generator, "Executor", _FakeExecutorMarker)

    with pytest.raises(TypeError, match="return_executor"):
        testset_generator.generate_golden_testset(markdown_dir, tmp_path / "out")


# ==========================================================================
# tools/generate_testset.py -- CLI mỏng gọi testset_generator (mục 7)
# ==========================================================================


def test_cli_goi_dung_generate_golden_testset_va_in_so_luong(
    monkeypatch: pytest.MonkeyPatch,
):
    from tools import generate_testset

    captured: dict[str, Any] = {}

    def _fake_generate(
        markdown_dir: Path,
        output_dir: Path,
        *,
        reuse_knowledge_graph: bool = False,
    ) -> list[GoldenTestCase]:
        captured["args"] = (markdown_dir, output_dir, reuse_knowledge_graph)
        return [GoldenTestCase(**_sample(1))]

    monkeypatch.setattr(generate_testset, "generate_golden_testset", _fake_generate)

    result = CliRunner().invoke(generate_testset.app, ["--reuse-knowledge-graph"])

    assert result.exit_code == 0
    assert captured["args"][2] is True
    assert "Đã sinh 1 câu hỏi" in result.output


def test_cli_mac_dinh_khong_bat_reuse_knowledge_graph_va_dung_thu_muc_mac_dinh(
    monkeypatch: pytest.MonkeyPatch,
):
    from tools import generate_testset

    captured: dict[str, Any] = {}

    def _fake_generate(
        markdown_dir: Path,
        output_dir: Path,
        *,
        reuse_knowledge_graph: bool = False,
    ) -> list[GoldenTestCase]:
        captured["args"] = (markdown_dir, output_dir, reuse_knowledge_graph)
        return []

    monkeypatch.setattr(generate_testset, "generate_golden_testset", _fake_generate)

    result = CliRunner().invoke(generate_testset.app, [])

    assert result.exit_code == 0
    assert captured["args"] == (
        generate_testset.DEFAULT_MARKDOWN_DIR,
        generate_testset.DEFAULT_OUTPUT_DIR,
        False,
    )
    assert "Đã sinh 0 câu hỏi" in result.output
