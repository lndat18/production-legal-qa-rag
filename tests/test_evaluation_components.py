"""Unit test cho các thành phần `evaluation/` KHÔNG phụ thuộc `ragas` (Phase 1).

Gồm `models`, `corpus_loader`, `embeddings_adapter`, `groq_round_robin` và
`TestsetGeneratorSettings` (evaluation_spec.md mục 3, 5, 6, 8) -- các module này
chỉ import `langchain_core`/`langchain_openai`/`openai` nên chạy được trên venv mặc
định (không cần dependency-group `eval`), khác `test_evaluation.py` (điều phối
`ragas.testset.TestsetGenerator`, cần `ragas`). Mọi interaction Groq/HuggingFace
dùng fake, không gọi dịch vụ ngoài.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from openai import APIStatusError, RateLimitError
from pydantic import ValidationError

from production_legal_qa_rag.config import EmbeddingSettings, TestsetGeneratorSettings
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
