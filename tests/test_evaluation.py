"""Unit test cho `evaluation/ragas_runner.py` — phần chạm `ragas` (Phase 1 sinh golden testset).

Mọi interaction Groq/HuggingFace/`TestsetGenerator.generate` dùng fake hoặc monkeypatch —
không gọi dịch vụ ngoài, không dựng KG/sinh câu hỏi thật (evaluation_spec.md mục 4.3, 8).
Chữ ký `prepare_combinations` của 3 synthesizer và `calculate_split_values` là của
`ragas==0.4.3` cài thật (test gọi thẳng code ragas để phát hiện khi nâng version).

`ragas` chỉ nằm trong dependency-group `eval` (venv mặc định không cài, xem
`[dependency-groups]` trong `pyproject.toml`), nên module này được skip nếu thiếu
(`uv run --group eval --no-group production pytest` để chạy thật). Logic điều phối thuần
(thứ tự, progress, finalize, CLI) nằm ở `test_evaluation_testset_generator.py` và luôn chạy.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip(
    "ragas",
    reason="ragas chỉ có trong dependency-group `eval` (`uv sync --group eval`)",
)

from langchain_core.documents import Document
from langchain_core.prompt_values import StringPromptValue
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from ragas.prompt.mixin import PromptMixin
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.base import QueryStyle
from ragas.testset.synthesizers.utils import calculate_split_values
from ragas.testset.transforms import default_transforms

from production_legal_qa_rag.config import EmbeddingSettings, TestsetGeneratorSettings
from production_legal_qa_rag.evaluation import ragas_runner, testset_generator
from production_legal_qa_rag.evaluation.embeddings_adapter import RagasEmbeddingsAdapter
from production_legal_qa_rag.evaluation.groq_round_robin import (
    DailyQuotaExhaustedError,
    GroqRoundRobinChatModel,
)
from production_legal_qa_rag.evaluation.testset_generator import QuestionQuota
from production_legal_qa_rag.evaluation.unit_splitter import EvalUnit


def _settings(**overrides: Any) -> TestsetGeneratorSettings:
    return TestsetGeneratorSettings(  # type: ignore[call-arg]
        GROQ_API_KEY="key-1",
        GROQ_API_KEY_2="key-2",
        GROQ_API_KEY_3="key-3",
        GROQ_API_KEY_4="key-4",
        GROQ_API_KEY_5="key-5",
        GROQ_API_KEY_6="key-6",
        **overrides,
    )


def _runner(monkeypatch: pytest.MonkeyPatch) -> ragas_runner.RagasUnitRunner:
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    adapter = RagasEmbeddingsAdapter(EmbeddingSettings(), client=object())  # type: ignore[arg-type]
    return ragas_runner.RagasUnitRunner(_settings(), embeddings_adapter=adapter)


def _unit() -> EvalUnit:
    return EvalUnit(
        source_document="A.md",
        index=1,
        title="Chương I",
        text="## Chương I. X\n\n#### Điều 1. Y\n\n" + "nội dung " * 500,
    )


# ==========================================================================
# Client Groq, transforms
# ==========================================================================


def test_build_groq_clients_tao_dung_6_client_doc_lap_tai_khoan():
    clients = ragas_runner._build_groq_clients(_settings())

    assert [client.openai_api_key.get_secret_value() for client in clients] == [
        f"key-{n}" for n in range(1, 7)
    ]
    assert all(client.model_name == "openai/gpt-oss-120b" for client in clients)


def test_build_groq_clients_tro_dung_groq_base_url_va_forward_retry_timeout():
    clients = ragas_runner._build_groq_clients(
        _settings(max_retries=5, timeout_seconds=90)
    )

    assert all(
        client.openai_api_base == ragas_runner._GROQ_OPENAI_BASE_URL
        for client in clients
    )
    assert all(client.max_retries == 5 for client in clients)
    assert all(client.request_timeout == 90.0 for client in clients)


def test_runner_wire_dung_llm_round_robin_6_key_embeddings_va_max_workers(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner(monkeypatch)

    assert isinstance(runner.llm.langchain_llm, GroqRoundRobinChatModel)
    assert len(runner.llm.langchain_llm.clients) == 6
    assert runner.run_config.max_workers == ragas_runner.MAX_WORKERS == 4


def test_cap_token_limit_ha_dung_4_extractor_cua_transforms_mac_dinh(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner(monkeypatch)
    unit = _unit()
    document = Document(
        page_content=unit.text, metadata={"source": unit.source_document}
    )
    transforms = default_transforms(
        documents=[document], llm=runner.llm, embedding_model=runner.embeddings
    )

    changed = ragas_runner.cap_token_limit(transforms, 4000)

    assert changed == 4  # số đã đo ở pilot (mục 4.3)


def test_cap_token_limit_bo_qua_transform_khong_phai_extractor():
    assert ragas_runner.cap_token_limit([object(), []], 4000) == 0


def test_build_unit_runner_dung_runner_ragas_that(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")

    runner = testset_generator.build_unit_runner(_settings())

    assert isinstance(runner, ragas_runner.RagasUnitRunner)


# ==========================================================================
# Ép PERFECT_GRAMMAR cho CẢ 3 synthesizer (mục 4.3): gọi prepare_combinations thật của ragas
# ==========================================================================

_PERSONAS = [Persona(name="An", role_description="Người lao động")]


def _node(**properties: Any) -> Node:
    return Node(type=NodeType.CHUNK, properties={"page_content": "x", **properties})


def test_single_hop_ep_perfect_grammar(monkeypatch: pytest.MonkeyPatch):
    runner = _runner(monkeypatch)
    synthesizer = ragas_runner.CleanSingleHopSynthesizer(llm=runner.llm)

    combinations = synthesizer.prepare_combinations(
        _node(), ["thử việc"], _PERSONAS, {"An": ["thử việc"]}
    )

    assert combinations
    assert all(c["styles"] == [QueryStyle.PERFECT_GRAMMAR] for c in combinations)
    assert all(
        c["personas"] == _PERSONAS for c in combinations
    )  # phần còn lại giữ nguyên


@pytest.mark.parametrize(
    ("synthesizer_class", "property_name"),
    [
        (ragas_runner.CleanMultiHopAbstractSynthesizer, "themes"),
        (ragas_runner.CleanMultiHopSpecificSynthesizer, "entities"),
    ],
)
def test_multi_hop_ep_perfect_grammar_voi_chu_ky_keyword_cua_ragas(
    monkeypatch: pytest.MonkeyPatch, synthesizer_class: type, property_name: str
):
    runner = _runner(monkeypatch)
    synthesizer = synthesizer_class(llm=runner.llm)
    nodes = [
        _node(**{property_name: ["thử việc"]}),
        _node(**{property_name: ["lương"]}),
    ]

    combinations = synthesizer.prepare_combinations(
        nodes,
        [["thử việc", "lương"]],
        personas=_PERSONAS,
        persona_item_mapping={"An": ["thử việc"]},
        property_name=property_name,
    )

    assert len(combinations) == 1
    assert combinations[0]["styles"] == [QueryStyle.PERFECT_GRAMMAR]
    assert len(combinations[0]["nodes"]) == 2


# ==========================================================================
# query_distribution: trọng số cho ra ĐÚNG số câu từng loại theo ragas
# ==========================================================================


class _FakeSynthesizer:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.mark.parametrize(
    "counts",
    [
        (4, 1, 1),
        (192, 24, 24),
        (3, 0, 2),
        (1, 1, 0),
        (7, 3, 3),
        (17, 2, 1),
        (0, 0, 5),
    ],
)
def test_query_distribution_cho_ra_dung_so_cau_theo_calculate_split_values(
    monkeypatch: pytest.MonkeyPatch, counts: tuple[int, int, int]
):
    runner = _runner(monkeypatch)
    runner._synthesizers = {
        "single_hop": _FakeSynthesizer("single_hop"),  # type: ignore[dict-item]
        "abstract": _FakeSynthesizer("abstract"),  # type: ignore[dict-item]
        "specific": _FakeSynthesizer("specific"),  # type: ignore[dict-item]
    }
    monkeypatch.setattr(ragas_runner, "_has_clusters", lambda *_args: True)
    quota = QuestionQuota(single_hop=counts[0], abstract=counts[1], specific=counts[2])

    distribution, total = runner._query_distribution(KnowledgeGraph(), quota)

    assert total == sum(counts)
    splits, _ = calculate_split_values([prob for _, prob in distribution], total)
    assert splits == [n for n in counts if n > 0]
    assert [s.name for s, _ in distribution] == [  # type: ignore[attr-defined]
        kind
        for kind, n in zip(("single_hop", "abstract", "specific"), counts, strict=True)
        if n > 0
    ]


def test_query_distribution_bo_loai_khong_co_cum_va_khong_bu(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner(monkeypatch)
    runner._synthesizers = {
        "single_hop": _FakeSynthesizer("single_hop"),  # type: ignore[dict-item]
        "abstract": _FakeSynthesizer("abstract"),  # type: ignore[dict-item]
        "specific": _FakeSynthesizer("specific"),  # type: ignore[dict-item]
    }
    monkeypatch.setattr(
        ragas_runner,
        "_has_clusters",
        lambda synthesizer, *_args: synthesizer.name != "abstract",
    )

    distribution, total = runner._query_distribution(
        KnowledgeGraph(), QuestionQuota(single_hop=5, abstract=2, specific=1)
    )

    assert total == 6
    assert [s.name for s, _ in distribution] == ["single_hop", "specific"]  # type: ignore[attr-defined]


# ==========================================================================
# run_unit: KG chỉ lưu khi sinh xong, tái dùng KG, đếm lượt gọi
# ==========================================================================


class _FakeTestset:
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self._samples = samples

    def to_list(self) -> list[dict[str, Any]]:
        return self._samples


def _sample(index: int, **overrides: Any) -> dict[str, Any]:
    sample = {
        "user_input": f"Câu hỏi {index}?",
        "reference": f"Đáp án {index}.",
        "reference_contexts": [f"Ngữ cảnh {index}"],
        "synthesizer_name": "single_hop_specific_query_synthesizer",
    }
    sample.update(overrides)
    return sample


class _FakeGraph:
    """KG giả: có 1 node DOCUMENT mang `page_content` để kiểm khớp văn bản khi nạp lại."""

    def __init__(self, page_content: str | None = None) -> None:
        text = _unit().text if page_content is None else page_content
        self.nodes = [Node(type=NodeType.DOCUMENT, properties={"page_content": text})]
        self.saved_to: Path | None = None

    def save(self, path: Path) -> None:
        self.saved_to = Path(path)
        Path(path).write_text("{}", encoding="utf-8")


def _patch_generation(
    monkeypatch: pytest.MonkeyPatch,
    runner: ragas_runner.RagasUnitRunner,
    *,
    samples: list[dict[str, Any]],
    error: Exception | None = None,
) -> dict[str, Any]:
    """Thay dựng KG + `TestsetGenerator` bằng fake; trả dict ghi lại các lần gọi."""
    seen: dict[str, Any] = {"built": 0, "generate_calls": []}
    graph = _FakeGraph()

    def _build(unit: EvalUnit) -> _FakeGraph:
        seen["built"] += 1
        return graph

    class _FakeGenerator:
        def __init__(self, **_kwargs: Any) -> None:
            self.knowledge_graph: Any = None

        def generate(self, **kwargs: Any) -> _FakeTestset:
            seen["generate_calls"].append(kwargs)
            seen["kg_used"] = self.knowledge_graph
            if error is not None:
                raise error
            return _FakeTestset(samples)

    monkeypatch.setattr(runner, "_build_knowledge_graph", _build)
    monkeypatch.setattr(ragas_runner, "TestsetGenerator", _FakeGenerator)
    runner._synthesizers = {
        "single_hop": _FakeSynthesizer("single_hop"),  # type: ignore[dict-item]
        "abstract": _FakeSynthesizer("abstract"),  # type: ignore[dict-item]
        "specific": _FakeSynthesizer("specific"),  # type: ignore[dict-item]
    }
    monkeypatch.setattr(ragas_runner, "_has_clusters", lambda *_args: True)
    seen["graph"] = graph
    return seen


def test_run_unit_dung_kg_rieng_sinh_cau_va_luu_kg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    seen = _patch_generation(monkeypatch, runner, samples=[_sample(1), _sample(2)])
    kg_path = tmp_path / "knowledge_graph" / "A__01.json"

    result = runner.run_unit(
        _unit(),
        QuestionQuota(single_hop=2),
        kg_path,
        reuse_knowledge_graph=False,
    )

    assert [c.user_input for c in result.cases] == ["Câu hỏi 1?", "Câu hỏi 2?"]
    assert seen["built"] == 1
    assert seen["kg_used"] is seen["graph"]
    assert seen["generate_calls"][0]["testset_size"] == 2
    assert seen["generate_calls"][0]["run_config"] is runner.run_config
    assert kg_path.exists()


def test_run_unit_loi_sinh_cau_van_giu_kg_da_dung_xong_khong_de_lai_file_tam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    _patch_generation(monkeypatch, runner, samples=[], error=RuntimeError("429"))
    kg_path = tmp_path / "knowledge_graph" / "A__01.json"

    with pytest.raises(RuntimeError):
        runner.run_unit(
            _unit(), QuestionQuota(single_hop=2), kg_path, reuse_knowledge_graph=False
        )

    assert kg_path.exists()
    assert list(kg_path.parent.glob("*.tmp")) == []


def test_run_unit_loi_khi_dung_kg_thi_khong_de_lai_file_kg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    _patch_generation(monkeypatch, runner, samples=[])

    def _build_fails(unit: EvalUnit) -> Any:
        raise RuntimeError("429 giữa lúc dựng KG")

    monkeypatch.setattr(runner, "_build_knowledge_graph", _build_fails)
    kg_path = tmp_path / "knowledge_graph" / "A__01.json"

    with pytest.raises(RuntimeError):
        runner.run_unit(
            _unit(), QuestionQuota(single_hop=2), kg_path, reuse_knowledge_graph=False
        )

    assert not kg_path.exists()


def test_run_unit_reuse_nap_kg_da_luu_va_khong_dung_lai_khong_ghi_de(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    seen = _patch_generation(monkeypatch, runner, samples=[_sample(1)])
    kg_path = tmp_path / "A__01.json"
    loaded_graph = _FakeGraph()
    loaded_from: list[Path] = []
    kg_path.write_text("nguyên trạng", encoding="utf-8")

    def _load(path: Path) -> _FakeGraph:
        loaded_from.append(path)
        return loaded_graph

    monkeypatch.setattr(KnowledgeGraph, "load", staticmethod(_load))

    runner.run_unit(
        _unit(), QuestionQuota(single_hop=1), kg_path, reuse_knowledge_graph=True
    )

    assert loaded_from == [kg_path]
    assert seen["built"] == 0
    assert seen["kg_used"] is loaded_graph
    assert kg_path.read_text("utf-8") == "nguyên trạng"


def test_run_unit_reuse_nhung_kg_dung_tu_van_ban_khac_thi_dung_lai_va_ghi_de(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    seen = _patch_generation(monkeypatch, runner, samples=[_sample(1)])
    kg_path = tmp_path / "A__01.json"
    kg_path.write_text("cũ", encoding="utf-8")
    monkeypatch.setattr(
        KnowledgeGraph,
        "load",
        staticmethod(lambda _path: _FakeGraph(page_content="văn bản đã bị sửa")),
    )

    runner.run_unit(
        _unit(), QuestionQuota(single_hop=1), kg_path, reuse_knowledge_graph=True
    )

    assert seen["built"] == 1
    assert kg_path.read_text("utf-8") == "{}"


def test_run_unit_reuse_nhung_file_kg_hong_thi_dung_lai(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    seen = _patch_generation(monkeypatch, runner, samples=[_sample(1)])
    kg_path = tmp_path / "A__01.json"
    kg_path.write_text(
        "{hỏng", encoding="utf-8"
    )  # KnowledgeGraph.load thật -> ValueError

    runner.run_unit(
        _unit(), QuestionQuota(single_hop=1), kg_path, reuse_knowledge_graph=True
    )

    assert seen["built"] == 1


def test_run_unit_reuse_nhung_chua_co_file_kg_thi_van_dung_kg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    seen = _patch_generation(monkeypatch, runner, samples=[_sample(1)])
    kg_path = tmp_path / "A__01.json"

    runner.run_unit(
        _unit(), QuestionQuota(single_hop=1), kg_path, reuse_knowledge_graph=True
    )

    assert seen["built"] == 1
    assert kg_path.exists()


def test_run_unit_quota_bang_0_khong_goi_generate_van_luu_kg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    seen = _patch_generation(monkeypatch, runner, samples=[_sample(1)])

    result = runner.run_unit(
        _unit(), QuestionQuota(), tmp_path / "A__01.json", reuse_knowledge_graph=False
    )

    assert result.cases == []
    assert seen["generate_calls"] == []
    assert (tmp_path / "A__01.json").exists()


def test_run_unit_llm_calls_la_hieu_so_dem_cua_router(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    _patch_generation(monkeypatch, runner, samples=[_sample(1)])
    router = runner.router
    router._next_client()  # lượt gọi từ trước (vd. adapt_prompts) không tính vào đơn vị này
    original_build = runner._build_knowledge_graph

    def _build_and_call(unit: EvalUnit) -> Any:
        for _ in range(3):
            router._next_client()
        return original_build(unit)

    monkeypatch.setattr(runner, "_build_knowledge_graph", _build_and_call)

    result = runner.run_unit(
        _unit(),
        QuestionQuota(single_hop=1),
        tmp_path / "A__01.json",
        reuse_knowledge_graph=False,
    )

    assert result.llm_calls == 3


def test_to_cases_bo_mau_thieu_cot_bat_buoc_va_giu_synthesizer_name():
    samples = [
        _sample(1),
        _sample(2, reference=""),
        _sample(3, reference_contexts=[]),
        {"user_input": "thiếu cột"},
        _sample(5, synthesizer_name="multi_hop_abstract_query_synthesizer"),
    ]

    cases = ragas_runner._to_cases(samples)

    assert [c.user_input for c in cases] == ["Câu hỏi 1?", "Câu hỏi 5?"]
    assert cases[1].synthesizer_name == "multi_hop_abstract_query_synthesizer"
    assert all(c.source_document is None for c in cases)  # do testset_generator gắn


# ==========================================================================
# Retry của ragas (mục 3.1, 8): chỉ lỗi tạm thời, ít lần, và có hiệu lực TRƯỚC generate()
# ==========================================================================


def _status_error(cls: type[Any], status: int, message: str = "lỗi") -> Any:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return cls(message, response=httpx.Response(status, request=request), body=None)


def test_run_config_chi_retry_loi_tam_thoi_voi_so_lan_va_thoi_gian_cho_han_che():
    config = ragas_runner.build_run_config()

    assert config.max_retries == ragas_runner.MAX_RETRIES == 3
    assert config.max_wait == ragas_runner.MAX_WAIT_SECONDS == 30
    assert config.max_workers == ragas_runner.MAX_WORKERS
    types = config.exception_types
    assert isinstance(types, tuple)
    for transient in (
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
    ):
        assert issubclass(transient, types)
    for deterministic in (
        BadRequestError,
        AuthenticationError,
        DailyQuotaExhaustedError,
    ):
        assert not issubclass(deterministic, types)


def test_llm_dung_chung_run_config_cua_runner_ngay_tu_khoi_tao(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner(monkeypatch)

    assert runner.llm.run_config is runner.run_config
    assert runner.llm.run_config.exception_types == ragas_runner.RETRYABLE_EXCEPTIONS
    assert runner.llm.run_config.max_retries == ragas_runner.MAX_RETRIES


def _raise_from_all_clients(
    runner: ragas_runner.RagasUnitRunner, error_factory: Any
) -> None:
    def _generate(messages: object, **kwargs: object) -> Any:
        raise error_factory()

    for client in runner.router.clients:
        client._generate = _generate  # type: ignore[method-assign]
    runner.llm.run_config.max_wait = 0  # test không được ngủ thật giữa các lần thử


def _ask(runner: ragas_runner.RagasUnitRunner) -> None:
    asyncio.run(runner.llm.generate(StringPromptValue(text="x")))


def test_llm_loi_tat_dinh_400_khong_bi_ragas_thu_lai_ngay_ca_khi_chua_goi_generate(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner(monkeypatch)
    _raise_from_all_clients(runner, lambda: _status_error(BadRequestError, 400))

    with pytest.raises(BadRequestError):
        _ask(runner)

    assert sum(runner.router.call_counts) == 1


def test_llm_429_theo_phut_bi_thu_lai_dung_max_retries_lan_moi_lan_du_6_tai_khoan(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner(monkeypatch)
    _raise_from_all_clients(
        runner,
        lambda: _status_error(
            RateLimitError, 429, "Rate limit reached on tokens per minute (TPM)"
        ),
    )

    with pytest.raises(RateLimitError):
        _ask(runner)

    assert sum(runner.router.call_counts) == ragas_runner.MAX_RETRIES * 6


def test_llm_het_quota_ngay_dung_sau_mot_vong_va_tu_choi_moi_luot_sau_do(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner(monkeypatch)
    _raise_from_all_clients(
        runner,
        lambda: _status_error(
            RateLimitError, 429, "Rate limit reached on tokens per day (TPD)"
        ),
    )

    with pytest.raises(DailyQuotaExhaustedError):
        _ask(runner)
    after_first = sum(runner.router.call_counts)
    with pytest.raises(DailyQuotaExhaustedError):
        _ask(runner)

    assert after_first == 6
    assert sum(runner.router.call_counts) == 6  # lượt sau không gửi request nào


# ==========================================================================
# adapt_prompts đúng 1 lần mỗi synthesizer, dùng lại qua nhiều đơn vị (mục 4, 4.3b)
# ==========================================================================


def test_get_synthesizers_adapt_prompts_dung_1_lan_moi_synthesizer_qua_nhieu_don_vi(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner(monkeypatch)
    adapt_calls: list[tuple[str, str, Any]] = []
    set_calls: list[tuple[str, set[str]]] = []
    original_set_prompts = PromptMixin.set_prompts

    async def _fake_adapt(self: Any, language: str, llm: Any, *_a: Any) -> Any:
        adapt_calls.append((type(self).__name__, language, llm))
        return dict(self.get_prompts())

    def _spy_set_prompts(self: Any, **prompts: Any) -> None:
        set_calls.append((type(self).__name__, set(prompts)))
        original_set_prompts(self, **prompts)

    monkeypatch.setattr(PromptMixin, "adapt_prompts", _fake_adapt)
    monkeypatch.setattr(PromptMixin, "set_prompts", _spy_set_prompts)
    monkeypatch.setattr(ragas_runner, "_has_clusters", lambda *_args: True)

    for _ in range(3):  # ba đơn vị liên tiếp
        runner._query_distribution(KnowledgeGraph(), QuestionQuota(single_hop=2))

    names = sorted(name for name, _language, _llm in adapt_calls)
    assert names == sorted(
        [
            "CleanSingleHopSynthesizer",
            "CleanMultiHopAbstractSynthesizer",
            "CleanMultiHopSpecificSynthesizer",
        ]
    )
    assert all(language == "vietnamese" for _n, language, _l in adapt_calls)
    assert all(llm is runner.llm for _n, _lang, llm in adapt_calls)
    assert [name for name, _keys in set_calls] == [name for name, _l, _m in adapt_calls]
    assert all(keys for _name, keys in set_calls)  # prompt đã dịch được áp lại


def test_build_unit_runner_thieu_key_chi_bao_ten_bien_khong_lo_gia_tri(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)  # tránh đọc .env thật của repo
    for name in ("GROQ_API_KEY_5", "GROQ_API_KEY_6"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_BI_MAT_KEY_1_xyz")
    for n in (2, 3, 4):
        monkeypatch.setenv(f"GROQ_API_KEY_{n}", f"gsk_BI_MAT_KEY_{n}_xyz")

    with pytest.raises(testset_generator.EvalInputError) as excinfo:
        testset_generator.build_unit_runner()

    message = str(excinfo.value)
    assert "GROQ_API_KEY_5" in message
    assert "GROQ_API_KEY_6" in message
    assert "gsk_BI_MAT" not in message
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__
