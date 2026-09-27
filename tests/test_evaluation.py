"""Unit test cho package `evaluation/` — Phase 1 sinh golden testset RAGAS.

Mọi interaction Groq/HuggingFace/`ragas.testset.TestsetGenerator` trong file
này dùng fake đã bị monkeypatch — không gọi dịch vụ ngoài, không build
`KnowledgeGraph`/sinh câu hỏi thật (evaluation_spec.md mục 8, 10). Chữ ký thật
của `TestsetGenerator`/`generate_with_langchain_docs` đã được xác nhận trực
tiếp trên `ragas==0.4.3` cài thật bằng `inspect.signature` lúc implement
(mục 10.4), không phải qua test này.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from openai import APIStatusError, RateLimitError
from pydantic import ValidationError
from typer.testing import CliRunner

from production_legal_qa_rag.config import EmbeddingSettings, TestsetGeneratorSettings
from production_legal_qa_rag.evaluation import testset_generator
from production_legal_qa_rag.evaluation.corpus_loader import load_markdown_documents
from production_legal_qa_rag.evaluation.embeddings_adapter import (
    RagasEmbeddingsAdapter,
    _coerce_embeddings,
)
from production_legal_qa_rag.evaluation.groq_round_robin import GroqRoundRobinChatModel
from production_legal_qa_rag.evaluation.models import GoldenTestCase

# ==========================================================================
# models.py -- GoldenTestCase (mục 5)
# ==========================================================================


def test_golden_test_case_chap_nhan_du_3_cot_bat_buoc_va_synthesizer_tuy_chon():
    case = GoldenTestCase(
        user_input="Thời gian thử việc tối đa là bao lâu?",
        reference="Tối đa 60 ngày với công việc cần trình độ cao đẳng.",
        reference_contexts=["Điều 25 Bộ luật Lao động quy định..."],
    )

    assert case.synthesizer_name is None
    assert case.reference_contexts == ["Điều 25 Bộ luật Lao động quy định..."]


def test_golden_test_case_giu_synthesizer_name_khi_co():
    case = GoldenTestCase(
        user_input="Câu hỏi",
        reference="Đáp án",
        reference_contexts=["Ngữ cảnh"],
        synthesizer_name="single_hop_specific_query_synthesizer",
    )

    assert case.synthesizer_name == "single_hop_specific_query_synthesizer"


def test_golden_test_case_thieu_field_bat_buoc_bi_tu_choi():
    with pytest.raises(ValidationError):
        GoldenTestCase(user_input="Câu hỏi")  # type: ignore[call-arg]


# ==========================================================================
# corpus_loader.py -- đọc data/markdown/*.md thành Document (mục 2, 4, 8)
# ==========================================================================


def test_load_markdown_documents_doc_dung_noi_dung_va_metadata_source(
    tmp_path: Path,
):
    (tmp_path / "b.md").write_text("Nội dung B", encoding="utf-8")
    (tmp_path / "a.md").write_text("Nội dung A", encoding="utf-8")

    documents = load_markdown_documents(tmp_path)

    assert [document.metadata["source"] for document in documents] == ["a.md", "b.md"]
    assert [document.page_content for document in documents] == [
        "Nội dung A",
        "Nội dung B",
    ]


def test_load_markdown_documents_bao_loi_khi_thu_muc_khong_ton_tai(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_markdown_documents(tmp_path / "khong-ton-tai")


def test_load_markdown_documents_bao_loi_khi_khong_co_file_md(tmp_path: Path):
    (tmp_path / "khong-phai-md.txt").write_text("noi dung", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_markdown_documents(tmp_path)


def test_load_markdown_documents_bao_loi_khi_file_rong(tmp_path: Path):
    (tmp_path / "rong.md").write_text("   \n", encoding="utf-8")

    with pytest.raises(ValueError, match="trống"):
        load_markdown_documents(tmp_path)


# ==========================================================================
# embeddings_adapter.py -- Embeddings interface quanh InferenceClient (mục 3, 6, 8)
# ==========================================================================


class _FakeHFClient:
    """Ghi request HF và trả response đã lập trình sẵn, giống test_embedding.py."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls: list[list[str]] = []

    def feature_extraction(self, inputs: list[str]) -> object:
        self.calls.append(inputs)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return response(inputs)
        return response


def _embedding_settings(monkeypatch: pytest.MonkeyPatch) -> EmbeddingSettings:
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    return EmbeddingSettings()


def test_embed_documents_khong_word_segment_va_giu_dung_thu_tu(
    monkeypatch: pytest.MonkeyPatch,
):
    client = _FakeHFClient(
        [lambda inputs: [[float(index)] for index, _ in enumerate(inputs)]]
    )
    adapter = RagasEmbeddingsAdapter(_embedding_settings(monkeypatch), client)  # type: ignore[arg-type]

    embeddings = adapter.embed_documents(["Điều 1", "Điều 2"])

    # Không áp pyvi.ViTokenizer.tokenize (mục 3) -- input gửi HF phải nguyên văn.
    assert client.calls == [["Điều 1", "Điều 2"]]
    assert embeddings == [[0.0], [1.0]]


def test_embed_documents_chia_batch_theo_dung_kich_thuoc(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "production_legal_qa_rag.evaluation.embeddings_adapter._BATCH_SIZE", 2
    )
    client = _FakeHFClient(
        [
            lambda inputs: [[1.0] for _ in inputs],
            lambda inputs: [[2.0] for _ in inputs],
        ]
    )
    adapter = RagasEmbeddingsAdapter(_embedding_settings(monkeypatch), client)  # type: ignore[arg-type]

    embeddings = adapter.embed_documents(["a", "b", "c"])

    assert [len(call) for call in client.calls] == [2, 1]
    assert embeddings == [[1.0], [1.0], [2.0]]


def test_embed_query_tra_ve_dung_1_vector(monkeypatch: pytest.MonkeyPatch):
    client = _FakeHFClient([[[3.0, 4.0]]])
    adapter = RagasEmbeddingsAdapter(_embedding_settings(monkeypatch), client)  # type: ignore[arg-type]

    assert adapter.embed_query("Điều 1") == [3.0, 4.0]
    assert client.calls == [["Điều 1"]]


def test_coerce_embeddings_tu_response_tolist_va_validate_shape():
    response = SimpleNamespace(tolist=lambda: [[1, 2.5]])

    assert _coerce_embeddings(response, expected_count=1) == [[1.0, 2.5]]
    with pytest.raises(ValueError, match="đúng số vector"):
        _coerce_embeddings([], expected_count=1)
    with pytest.raises(TypeError, match="không phải list"):
        _coerce_embeddings(["không phải vector"], expected_count=1)


# ==========================================================================
# groq_round_robin.py -- GroqRoundRobinChatModel (mục 3.1)
# ==========================================================================


def _chat_result(text: str) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def _fake_response(status_code: int) -> SimpleNamespace:
    return SimpleNamespace(
        request=SimpleNamespace(method="POST", url="https://api.groq.com/openai/v1"),
        status_code=status_code,
        headers={},
    )


def _rate_limit_error() -> RateLimitError:
    return RateLimitError(
        "rate limited",
        response=_fake_response(429),  # type: ignore[arg-type]
        body=None,
    )


def _fake_client(*, name: str) -> ChatOpenAI:
    """Trả về `ChatOpenAI` thật (chưa gọi mạng) để field `list[ChatOpenAI]` validate."""
    return ChatOpenAI(api_key=f"key-{name}", model="openai/gpt-oss-120b")


def test_generate_luan_phien_dung_thu_tu_qua_nhieu_lan_goi():
    calls: list[str] = []
    clients = [_fake_client(name=name) for name in ("a", "b", "c")]
    for name, client in zip(("a", "b", "c"), clients, strict=True):

        def _make(label: str) -> object:
            def _generate(messages: object, **kwargs: object) -> ChatResult:
                calls.append(label)
                return _chat_result(label)

            return _generate

        client._generate = _make(name)  # type: ignore[method-assign]

    router = GroqRoundRobinChatModel(clients=clients)

    results = [router._generate(messages=[]) for _ in range(5)]

    assert calls == ["a", "b", "c", "a", "b"]
    assert [result.generations[0].text for result in results] == [
        "a",
        "b",
        "c",
        "a",
        "b",
    ]


def test_generate_fallback_sang_client_ke_tiep_khi_rate_limit():
    client_a = _fake_client(name="a")
    client_b = _fake_client(name="b")
    calls: list[str] = []

    def _generate_a(messages: object, **kwargs: object) -> ChatResult:
        calls.append("a")
        raise _rate_limit_error()

    def _generate_b(messages: object, **kwargs: object) -> ChatResult:
        calls.append("b")
        return _chat_result("ok-from-b")

    client_a._generate = _generate_a  # type: ignore[method-assign]
    client_b._generate = _generate_b  # type: ignore[method-assign]

    router = GroqRoundRobinChatModel(clients=[client_a, client_b])

    result = router._generate(messages=[])

    assert calls == ["a", "b"]
    assert result.generations[0].text == "ok-from-b"


def test_generate_het_vong_van_rate_limit_thi_raise_loi_cuoi():
    clients = [_fake_client(name=name) for name in ("a", "b")]
    for client in clients:

        def _generate(messages: object, **kwargs: object) -> ChatResult:
            raise _rate_limit_error()

        client._generate = _generate  # type: ignore[method-assign]

    router = GroqRoundRobinChatModel(clients=clients)

    with pytest.raises(RateLimitError):
        router._generate(messages=[])


def test_generate_khong_bat_loi_khac_rate_limit():
    client = _fake_client(name="a")

    def _generate(messages: object, **kwargs: object) -> ChatResult:
        raise APIStatusError(
            "loi khac",
            response=_fake_response(500),  # type: ignore[arg-type]
            body=None,
        )

    client._generate = _generate  # type: ignore[method-assign]
    router = GroqRoundRobinChatModel(clients=[client])

    with pytest.raises(APIStatusError):
        router._generate(messages=[])


def test_router_bao_loi_khi_khong_co_client_nao():
    with pytest.raises(ValueError, match="ít nhất 1 client"):
        GroqRoundRobinChatModel(clients=[])


def test_router_llm_type_co_ten_rieng():
    router = GroqRoundRobinChatModel(clients=[_fake_client(name="a")])
    assert router._llm_type == "groq-round-robin"


# ==========================================================================
# config.py -- TestsetGeneratorSettings (mục 6): cả 3 key BẮT BUỘC
# ==========================================================================


def test_testset_generator_settings_doc_dung_ca_3_key_va_default(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "key-1")
    monkeypatch.setenv("GROQ_API_KEY_2", "key-2")
    monkeypatch.setenv("GROQ_API_KEY_3", "key-3")

    settings = TestsetGeneratorSettings()  # type: ignore[call-arg]

    assert (settings.api_key, settings.api_key_2, settings.api_key_3) == (
        "key-1",
        "key-2",
        "key-3",
    )
    assert settings.model_name == "openai/gpt-oss-120b"
    assert settings.max_retries == 2
    assert settings.timeout_seconds == 60


@pytest.mark.parametrize(
    "missing_env",
    ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"],
)
def test_testset_generator_settings_bao_loi_khi_thieu_bat_ky_key_nao(
    monkeypatch: pytest.MonkeyPatch, missing_env: str
):
    monkeypatch.setenv("GROQ_API_KEY", "key-1")
    monkeypatch.setenv("GROQ_API_KEY_2", "key-2")
    monkeypatch.setenv("GROQ_API_KEY_3", "key-3")
    monkeypatch.delenv(missing_env, raising=False)
    monkeypatch.setattr(
        TestsetGeneratorSettings,
        "model_config",
        {**TestsetGeneratorSettings.model_config, "env_file": None},
    )

    with pytest.raises(ValidationError):
        TestsetGeneratorSettings()  # type: ignore[call-arg]


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
