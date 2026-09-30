"""Unit test cho `evaluation/ragas_runner.py` — phần chạm `ragas` (Phase 1 sinh golden testset).

Mọi interaction Groq/HuggingFace, dựng KG, sinh persona/scenario/sample đều dùng fake hoặc
monkeypatch — không gọi dịch vụ ngoài, không dựng KG/sinh câu hỏi thật (evaluation_spec.md
mục 3.3, 4.3, 8). Chữ ký `prepare_combinations` của 3 synthesizer, `calculate_split_values`,
`generate_personas_from_kg`, `generate_scenarios`/`generate_sample` là của `ragas==0.4.3`
cài thật (test gọi thẳng code ragas để phát hiện khi nâng version).

`ragas` chỉ nằm trong dependency-group `eval` (venv mặc định không cài, xem
`[dependency-groups]` trong `pyproject.toml`), nên module này được skip nếu thiếu
(`uv run --group eval --no-group production pytest` để chạy thật). Logic điều phối thuần
(thứ tự, progress, finalize, CLI) nằm ở `test_evaluation_testset_generator.py` và luôn chạy.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip(
    "ragas",
    reason="ragas chỉ có trong dependency-group `eval` (`uv sync --group eval`)",
)

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompt_values import StringPromptValue
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)
from ragas.dataset_schema import SingleTurnSample
from ragas.prompt.mixin import PromptMixin
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona, generate_personas_from_kg
from ragas.testset.synthesizers.base import BaseSynthesizer, QueryStyle
from ragas.testset.synthesizers.utils import calculate_split_values
from ragas.testset.transforms import default_transforms

from production_legal_qa_rag.config import EmbeddingSettings, TestsetGeneratorSettings
from production_legal_qa_rag.evaluation import ragas_runner, testset_generator
from production_legal_qa_rag.evaluation.embeddings_adapter import RagasEmbeddingsAdapter
from production_legal_qa_rag.evaluation.groq_round_robin import (
    DailyQuotaExhaustedError,
    GroqRoundRobinChatModel,
)
from production_legal_qa_rag.evaluation.models import GoldenTestCase
from production_legal_qa_rag.evaluation.testset_generator import (
    QuestionQuota,
    UnitGenerationError,
    UnitResult,
)
from production_legal_qa_rag.evaluation.unit_splitter import EvalUnit


def _settings(**overrides: Any) -> TestsetGeneratorSettings:
    return TestsetGeneratorSettings(  # type: ignore[call-arg]
        GROQ_API_KEY_1="key-1",
        GROQ_API_KEY_2="key-2",
        GROQ_API_KEY_3="key-3",
        GROQ_API_KEY_4="key-4",
        GROQ_API_KEY_5="key-5",
        GROQ_API_KEY_6="key-6",
        GROQ_API_KEY_7="key-7",
        GROQ_API_KEY_8="key-8",
        GROQ_API_KEY_9="key-9",
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


def test_build_groq_clients_tao_dung_9_client_doc_lap_tai_khoan():
    clients = ragas_runner._build_groq_clients(_settings())

    assert [client.openai_api_key.get_secret_value() for client in clients] == [
        f"key-{n}" for n in range(1, 10)
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


def test_runner_wire_dung_llm_round_robin_9_key_embeddings_va_max_workers(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner(monkeypatch)

    assert isinstance(runner.llm.langchain_llm, GroqRoundRobinChatModel)
    assert len(runner.llm.langchain_llm.clients) == 9
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


_NAME_OF_KIND = {
    "single_hop": "single_hop_specific_query_synthesizer",
    "abstract": "multi_hop_abstract_query_synthesizer",
    "specific": "multi_hop_specific_query_synthesizer",
}


@dataclass(frozen=True)
class _FakeScenario:
    kind: str
    index: int


class _FakeSample:
    """Giống `SingleTurnSample` ở đúng phần runner đọc: `model_dump(exclude_none=True)`."""

    def __init__(self, fields: dict[str, Any]) -> None:
        self._fields = fields

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        # `synthesizer_name` không do sample của ragas trả: runner tự gắn từ `synthesizer.name`.
        fields = {k: v for k, v in self._fields.items() if k != "synthesizer_name"}
        if exclude_none:
            fields = {k: v for k, v in fields.items() if v is not None}
        return fields


def _default_sample(name: str, index: int) -> dict[str, Any]:
    return {
        "user_input": f"Câu hỏi {name} {index}?",
        "reference": f"Đáp án {index}.",
        "reference_contexts": [f"Ngữ cảnh {index}"],
        # Trường thừa có thật ở sample của ragas; GoldenTestCase bỏ qua.
        "persona_name": "An",
    }


class _FakeSynthesizer:
    """Synthesizer giả theo đúng API công khai runner dùng (mục 3.3).

    `outcomes[i]` là kết quả của scenario thứ `i`: dict các trường của sample, hoặc exception
    để ném; thiếu thì sinh sample mặc định. Mặc định không nhường event loop nên các sample
    chạy tuần tự, tất định; đặt `yield_in_sample` để đo mức đồng thời.
    """

    def __init__(
        self,
        name: str,
        outcomes: list[Any] | None = None,
        *,
        scenario_error: BaseException | None = None,
        probe: Callable[[], Any] | None = None,
    ) -> None:
        self.name = name
        self.outcomes = outcomes or []
        self.scenario_error = scenario_error
        self.probe = probe
        self.scenario_requests: list[int] = []
        self.graphs: list[Any] = []
        self.persona_lists: list[Any] = []
        self.sampled: list[_FakeScenario] = []
        self.probed: list[Any] = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self.yield_in_sample = False

    def _record_probe(self) -> None:
        if self.probe is not None:
            self.probed.append(self.probe())

    async def generate_scenarios(
        self, n: int, knowledge_graph: Any, persona_list: Any, callbacks: Any = None
    ) -> list[_FakeScenario]:
        self.scenario_requests.append(n)
        self.graphs.append(knowledge_graph)
        self.persona_lists.append(persona_list)
        self._record_probe()
        if self.scenario_error is not None:
            raise self.scenario_error
        return [_FakeScenario(self.name, index) for index in range(n)]

    async def generate_sample(
        self, scenario: _FakeScenario, callbacks: Any = None
    ) -> _FakeSample:
        self.sampled.append(scenario)
        self._record_probe()
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            if self.yield_in_sample:
                await asyncio.sleep(0)
        finally:
            self.in_flight -= 1
        outcome = self._outcome(scenario.index)
        if isinstance(outcome, BaseException):
            raise outcome
        return _FakeSample(outcome)

    def _outcome(self, index: int) -> Any:
        if index < len(self.outcomes):
            return self.outcomes[index]
        return _default_sample(self.name, index)


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


@pytest.mark.parametrize(
    "make_synthesizer",
    [
        lambda llm: ragas_runner.CleanMultiHopAbstractSynthesizer(llm=llm),
        lambda llm: ragas_runner.CleanMultiHopSpecificSynthesizer(llm=llm),
    ],
)
def test_has_clusters_tra_false_khi_kg_khong_co_quan_he_thay_vi_raise(
    monkeypatch: pytest.MonkeyPatch, make_synthesizer: Any
) -> None:
    """`KnowledgeGraph.find_n_indirect_clusters` (ragas thật) raise `ValueError` khi KG
    không có quan hệ nào khớp điều kiện — `_has_clusters` phải bắt và trả `False` (bỏ loại
    multi-hop đó, không bù, mục 4/4.5), không để lỗi lan lên `_query_distribution`.
    """
    runner = _runner(monkeypatch)
    synthesizer = make_synthesizer(runner.llm)
    empty_graph = KnowledgeGraph()  # không node, không relationship -> ragas raise thật

    assert ragas_runner._has_clusters(synthesizer, empty_graph, 2) is False


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
    samples: list[Any] | None = None,
    error: BaseException | None = None,
    outcomes: dict[str, list[Any]] | None = None,
    scenario_errors: dict[str, BaseException] | None = None,
) -> dict[str, Any]:
    """Thay dựng KG, persona và 3 synthesizer bằng fake; trả dict ghi lại các lần gọi.

    `samples`/`error` là viết tắt cho loại single-hop: kết quả từng scenario / lỗi khi sinh
    scenario. `outcomes`/`scenario_errors` chỉ định theo loại (`single_hop`/`abstract`/
    `specific`). Mỗi phần tử `outcomes` là dict trường của sample hoặc exception để ném.
    """
    seen: dict[str, Any] = {"built": 0, "persona_calls": []}
    graph = _FakeGraph()
    per_kind = dict(outcomes or {})
    if samples is not None:
        per_kind["single_hop"] = samples
    errors_per_kind = dict(scenario_errors or {})
    if error is not None:
        errors_per_kind["single_hop"] = error
    personas = list(_PERSONAS)

    def _build(unit: EvalUnit) -> _FakeGraph:
        seen["built"] += 1
        return graph

    def _personas(**kwargs: Any) -> list[Persona]:
        effort = runner.router.current_reasoning_effort
        seen["persona_calls"].append({**kwargs, "effort": effort})
        seen["kg_used"] = kwargs["kg"]
        return personas

    synthesizers = {
        kind: _FakeSynthesizer(
            _NAME_OF_KIND[kind],
            per_kind.get(kind),
            scenario_error=errors_per_kind.get(kind),
            probe=lambda: runner.router.current_reasoning_effort,
        )
        for kind in ("single_hop", "abstract", "specific")
    }
    monkeypatch.setattr(runner, "_build_knowledge_graph", _build)
    monkeypatch.setattr(ragas_runner, "generate_personas_from_kg", _personas)
    monkeypatch.setattr(ragas_runner, "_has_clusters", lambda *_args: True)
    runner._synthesizers = synthesizers  # type: ignore[assignment]
    seen.update(graph=graph, personas=personas, synthesizers=synthesizers)
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

    single = seen["synthesizers"]["single_hop"]
    persona_call = seen["persona_calls"][0]
    assert [c.user_input for c in result.cases] == ["Câu hỏi 1?", "Câu hỏi 2?"]
    assert seen["built"] == 1
    assert seen["kg_used"] is seen["graph"]
    assert persona_call["llm"] is runner.llm
    assert persona_call["num_personas"] == ragas_runner.NUM_PERSONAS
    assert single.scenario_requests == [2]
    assert single.graphs == [seen["graph"]]
    assert kg_path.exists()


def test_run_unit_loi_sinh_cau_van_giu_kg_da_dung_xong_khong_de_lai_file_tam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    failure = RuntimeError("429")
    _patch_generation(monkeypatch, runner, error=failure)
    kg_path = tmp_path / "knowledge_graph" / "A__01.json"

    with pytest.raises(UnitGenerationError) as excinfo:
        runner.run_unit(
            _unit(), QuestionQuota(single_hop=2), kg_path, reuse_knowledge_graph=False
        )

    assert excinfo.value.__cause__ is failure
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


def test_run_unit_quota_bang_0_khong_sinh_persona_hay_scenario_van_luu_kg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    seen = _patch_generation(monkeypatch, runner, samples=[_sample(1)])

    result = runner.run_unit(
        _unit(), QuestionQuota(), tmp_path / "A__01.json", reuse_knowledge_graph=False
    )

    assert result.cases == []
    assert seen["persona_calls"] == []
    assert all(s.scenario_requests == [] for s in seen["synthesizers"].values())
    assert (tmp_path / "A__01.json").exists()


def test_run_unit_llm_calls_la_hieu_so_dem_cua_router(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    _patch_generation(monkeypatch, runner, samples=[_sample(1)])
    router = runner.router
    # lượt gọi từ trước (vd. adapt_prompts) không tính vào đơn vị này
    router._record_attempt(0)
    original_build = runner._build_knowledge_graph

    def _build_and_call(unit: EvalUnit) -> Any:
        for _ in range(3):
            router._record_attempt(0)
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
    # Cả 9 tài khoản cùng cooldown phút thì router chờ (mục 3.2 A): không ngủ thật ở đây.
    runner.router.sleep = lambda _seconds: None


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


def test_llm_429_theo_phut_bi_thu_lai_dung_max_retries_lan_moi_lan_du_9_tai_khoan(
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

    assert sum(runner.router.call_counts) == ragas_runner.MAX_RETRIES * 9


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

    assert after_first == 9
    assert sum(runner.router.call_counts) == 9  # lượt sau không gửi request nào


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
    for name in (
        "GROQ_API_KEY_5",
        "GROQ_API_KEY_6",
        "GROQ_API_KEY_7",
        "GROQ_API_KEY_8",
        "GROQ_API_KEY_9",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GROQ_API_KEY_1", "gsk_BI_MAT_KEY_1_xyz")
    for n in (2, 3, 4):
        monkeypatch.setenv(f"GROQ_API_KEY_{n}", f"gsk_BI_MAT_KEY_{n}_xyz")

    with pytest.raises(testset_generator.EvalInputError) as excinfo:
        testset_generator.build_unit_runner()

    message = str(excinfo.value)
    assert "GROQ_API_KEY_5" in message
    assert "GROQ_API_KEY_6" in message
    assert "GROQ_API_KEY_9" in message
    assert "gsk_BI_MAT" not in message
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__


def test_build_unit_runner_key_de_trong_bi_tu_choi_chi_bao_ten_bien_khong_lo_gia_tri(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)  # tránh đọc .env thật của repo
    monkeypatch.setenv("GROQ_API_KEY_1", "gsk_BI_MAT_KEY_1_xyz")
    for n in (2, 3, 4, 6, 7, 8, 9):
        monkeypatch.setenv(f"GROQ_API_KEY_{n}", f"gsk_BI_MAT_KEY_{n}_xyz")
    monkeypatch.setenv("GROQ_API_KEY_5", "")  # `GROQ_API_KEY_5=` như .env.example

    with pytest.raises(testset_generator.EvalInputError) as excinfo:
        testset_generator.build_unit_runner()

    message = str(excinfo.value)
    assert "GROQ_API_KEY_5" in message
    assert "gsk_BI_MAT" not in message
    assert excinfo.value.__cause__ is None


@pytest.mark.parametrize(
    ("cls", "status"),
    [
        (AuthenticationError, 401),
        (PermissionDeniedError, 403),
        (APIStatusError, 413),  # tất định
    ],
)
def test_llm_loi_tat_dinh_401_403_413_khong_bi_ragas_thu_lai(
    monkeypatch: pytest.MonkeyPatch, cls: type[Any], status: int
):
    runner = _runner(monkeypatch)
    _raise_from_all_clients(runner, lambda: _status_error(cls, status))

    with pytest.raises(cls):
        _ask(runner)

    assert sum(runner.router.call_counts) == 1


def _connection_error() -> Any:
    return APIConnectionError(
        request=httpx.Request("POST", "https://api.groq.com/openai/v1")
    )


def _timeout_error() -> Any:
    return APITimeoutError(
        request=httpx.Request("POST", "https://api.groq.com/openai/v1")
    )


@pytest.mark.parametrize(
    "error_factory",
    [
        lambda: _status_error(InternalServerError, 500),
        lambda: _status_error(InternalServerError, 503),
        _connection_error,
        _timeout_error,
    ],
    ids=["500", "503", "loi-ket-noi", "timeout"],
)
def test_llm_loi_tam_thoi_5xx_ket_noi_timeout_bi_thu_lai_dung_max_retries_lan(
    monkeypatch: pytest.MonkeyPatch, error_factory: Any
):
    runner = _runner(monkeypatch)
    _raise_from_all_clients(runner, error_factory)

    with pytest.raises((InternalServerError, APIConnectionError)):
        _ask(runner)

    assert sum(runner.router.call_counts) == ragas_runner.MAX_RETRIES


# ==========================================================================
# KG lưu ngay sau khi dựng: lần sau tái dùng cho đơn vị dở (mục 4.5, 8)
# ==========================================================================


def test_run_unit_loi_o_buoc_sinh_cau_thi_lan_sau_tai_dung_kg_khong_dung_lai(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    kg_path = tmp_path / "knowledge_graph" / "A__01.json"
    first = _patch_generation(monkeypatch, runner, error=RuntimeError("429"))
    with pytest.raises(UnitGenerationError):
        runner.run_unit(
            _unit(), QuestionQuota(single_hop=2), kg_path, reuse_knowledge_graph=True
        )
    saved_after_failure = kg_path.read_text("utf-8")
    reloaded = _FakeGraph()
    monkeypatch.setattr(KnowledgeGraph, "load", staticmethod(lambda _p: reloaded))
    second = _patch_generation(monkeypatch, runner, samples=[_sample(1), _sample(2)])

    result = runner.run_unit(
        _unit(), QuestionQuota(single_hop=2), kg_path, reuse_knowledge_graph=True
    )

    assert first["built"] == 1
    assert second["built"] == 0  # không tốn lại 30-160K token dựng KG
    assert second["kg_used"] is reloaded
    assert [c.user_input for c in result.cases] == ["Câu hỏi 1?", "Câu hỏi 2?"]
    assert kg_path.read_text("utf-8") == saved_after_failure


def _document_graph(page_content: str) -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes=[
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": page_content,
                    "document_metadata": {"source": "A.md"},
                },
            )
        ]
    )


def test_luu_kg_nguyen_tu_roi_nap_lai_that_thi_khop_van_ban_moi_duoc_tai_dung(
    tmp_path: Path,
):
    unit = _unit()
    path = tmp_path / "knowledge_graph" / "A__01.json"

    ragas_runner._save_graph_atomic(_document_graph(unit.text), path)
    same = ragas_runner._load_matching_graph(path, unit)
    edited = ragas_runner._load_matching_graph(
        path, unit.model_copy(update={"text": unit.text + " sửa"})
    )

    assert list(path.parent.glob("*.tmp")) == []
    assert same is not None
    assert [n.properties["page_content"] for n in same.nodes] == [unit.text]
    assert edited is None  # văn bản đơn vị đã đổi -> phải dựng lại


def test_nap_kg_khong_co_node_document_thi_khong_tai_dung(tmp_path: Path):
    unit = _unit()
    path = tmp_path / "A__01.json"
    graph = KnowledgeGraph(
        nodes=[Node(type=NodeType.CHUNK, properties={"page_content": unit.text})]
    )
    ragas_runner._save_graph_atomic(graph, path)

    assert ragas_runner._load_matching_graph(path, unit) is None


@pytest.mark.parametrize(
    "content",
    [
        "{hỏng",  # JSON cắt dở
        "",
        "[]",  # JSON hợp lệ nhưng sai hình dạng
        "{}",
        '{"nodes": []}',  # thiếu "relationships"
        '{"nodes": [{"properties": 5}], "relationships": []}',
    ],
    ids=["cat-do", "rong", "mang", "doi-tuong-rong", "thieu-relationships", "node-sai"],
)
def test_file_kg_hong_bat_ke_dang_hong_thi_dung_lai_thay_vi_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str
):
    runner = _runner(monkeypatch)
    seen = _patch_generation(monkeypatch, runner, samples=[_sample(1)])
    kg_path = tmp_path / "A__01.json"
    kg_path.write_text(content, encoding="utf-8")

    runner.run_unit(
        _unit(), QuestionQuota(single_hop=1), kg_path, reuse_knowledge_graph=True
    )

    assert seen["built"] == 1
    assert kg_path.read_text("utf-8") == "{}"  # đã ghi đè bằng KG mới dựng


# ==========================================================================
# Sinh từng sample, giữ phần đã xong khi lỗi giữa đơn vị (mục 3.3)
# ==========================================================================

_SINGLE = _NAME_OF_KIND["single_hop"]
_ABSTRACT = _NAME_OF_KIND["abstract"]
_SPECIFIC = _NAME_OF_KIND["specific"]


def _run(
    runner: ragas_runner.RagasUnitRunner, tmp_path: Path, quota: QuestionQuota
) -> UnitResult:
    return runner.run_unit(
        _unit(), quota, tmp_path / "A__01.json", reuse_knowledge_graph=False
    )


def test_sinh_dung_so_scenario_tung_loai_theo_quota_va_giu_thu_tu_loai(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    seen = _patch_generation(monkeypatch, runner)
    quota = QuestionQuota(single_hop=7, abstract=3, specific=3)

    result = _run(runner, tmp_path, quota)

    synthesizers = seen["synthesizers"]
    names = [case.synthesizer_name for case in result.cases]
    expected = [_SINGLE] * 7 + [_ABSTRACT] * 3 + [_SPECIFIC] * 3
    assert synthesizers["single_hop"].scenario_requests == [7]
    assert synthesizers["abstract"].scenario_requests == [3]
    assert synthesizers["specific"].scenario_requests == [3]
    assert names == expected
    assert result.interruption is None
    assert result.skipped_samples == 0


def test_persona_sinh_mot_lan_va_kg_cung_persona_duoc_truyen_cho_moi_loai(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    seen = _patch_generation(monkeypatch, runner)
    quota = QuestionQuota(single_hop=2, abstract=1, specific=1)

    _run(runner, tmp_path, quota)

    assert len(seen["persona_calls"]) == 1
    for synthesizer in seen["synthesizers"].values():
        assert synthesizer.graphs == [seen["graph"]]
        assert synthesizer.persona_lists == [seen["personas"]]


def test_breaker_bat_giua_luc_sinh_thi_giu_sample_da_xong_va_dung_ngay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    quota_error = DailyQuotaExhaustedError("Hết quota ngày cả 9 tài khoản Groq")
    outcomes = [_sample(0), _sample(1), quota_error, _sample(3)]
    seen = _patch_generation(monkeypatch, runner, samples=outcomes)
    quota = QuestionQuota(single_hop=4, abstract=2)

    result = _run(runner, tmp_path, quota)

    synthesizers = seen["synthesizers"]
    assert [c.user_input for c in result.cases] == ["Câu hỏi 0?", "Câu hỏi 1?"]
    assert result.interruption is quota_error
    assert result.skipped_samples == 0
    # Sample thứ tư không được đưa thêm sau khi breaker bật, và không sang loại kế tiếp.
    assert len(synthesizers["single_hop"].sampled) == 3
    assert synthesizers["abstract"].scenario_requests == []
    assert (tmp_path / "A__01.json").exists()  # KG đã dựng xong vẫn được giữ


def test_loi_sinh_scenario_cua_mot_loai_giu_cac_loai_da_xong_truoc_do(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    failure = RuntimeError("lỗi sinh scenario")
    errors = {"abstract": failure}
    seen = _patch_generation(monkeypatch, runner, scenario_errors=errors)
    quota = QuestionQuota(single_hop=4, abstract=2, specific=2)

    result = _run(runner, tmp_path, quota)

    synthesizers = seen["synthesizers"]
    assert len(result.cases) == 4
    assert {case.synthesizer_name for case in result.cases} == {_SINGLE}
    assert result.interruption is failure
    assert synthesizers["abstract"].scenario_requests == [2]
    assert synthesizers["specific"].scenario_requests == []


def test_sample_loi_bi_bo_va_dem_dung_ke_ca_sample_thieu_cot_bat_buoc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    outcomes = [
        _sample(0),
        ValueError("nội dung bí mật"),
        _sample(2, reference=""),
        {"user_input": "thiếu cột"},
        _sample(4),
    ]
    _patch_generation(monkeypatch, runner, samples=outcomes)

    result = _run(runner, tmp_path, QuestionQuota(single_hop=5))

    assert [c.user_input for c in result.cases] == ["Câu hỏi 0?", "Câu hỏi 4?"]
    # 1 sample ném lỗi + 2 sample thiếu cột bắt buộc (reference rỗng, thiếu hẳn cột).
    assert result.skipped_samples == 3
    assert result.interruption is None


def test_moi_sample_deu_loi_thi_raise_unit_generation_error_va_van_giu_kg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    outcomes = [ValueError("sample 0"), ValueError("sample 1")]
    _patch_generation(monkeypatch, runner, samples=outcomes)
    kg_path = tmp_path / "A__01.json"

    with pytest.raises(UnitGenerationError, match="2 sample bị bỏ") as excinfo:
        runner.run_unit(
            _unit(), QuestionQuota(single_hop=2), kg_path, reuse_knowledge_graph=False
        )

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert kg_path.exists()


def test_khong_sinh_duoc_cau_nao_thi_loi_goc_uu_tien_loi_quota_hon_loi_parse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    quota_error = DailyQuotaExhaustedError("Hết quota ngày cả 9 tài khoản Groq")
    outcomes = [ValueError("parse"), quota_error, _sample(2)]
    _patch_generation(monkeypatch, runner, samples=outcomes)

    with pytest.raises(UnitGenerationError) as excinfo:
        _run(runner, tmp_path, QuestionQuota(single_hop=3))

    assert excinfo.value.__cause__ is quota_error


def test_chi_con_sample_thieu_cot_bat_buoc_thi_van_raise_nhung_khong_co_loi_goc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    _patch_generation(monkeypatch, runner, samples=[_sample(0, reference="")])

    with pytest.raises(UnitGenerationError) as excinfo:
        _run(runner, tmp_path, QuestionQuota(single_hop=1))

    assert excinfo.value.__cause__ is None
    assert "1 sample bị bỏ" in str(excinfo.value)


def test_log_loi_sample_chi_co_ten_synthesizer_va_ten_loai_loi_khong_co_noi_dung(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    runner = _runner(monkeypatch)
    outcomes = [ValueError("NỘI DUNG BÍ MẬT: Điều 5"), _sample(1)]
    _patch_generation(monkeypatch, runner, samples=outcomes)

    with caplog.at_level(logging.INFO, logger=ragas_runner.logger.name):
        _run(runner, tmp_path, QuestionQuota(single_hop=2))

    skipped = [r for r in caplog.records if "Bỏ 1 sample" in r.getMessage()]
    assert len(skipped) == 1
    assert _SINGLE in skipped[0].getMessage()
    assert "ValueError" in skipped[0].getMessage()
    assert "BÍ MẬT" not in caplog.text
    assert "Điều 5" not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_sample_chay_dong_thoi_nhieu_hon_1_nhung_khong_qua_max_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    seen = _patch_generation(monkeypatch, runner)
    single = seen["synthesizers"]["single_hop"]
    single.yield_in_sample = True

    result = _run(runner, tmp_path, QuestionQuota(single_hop=12))

    assert len(result.cases) == 12
    assert 2 <= single.peak_in_flight <= ragas_runner.MAX_WORKERS


# ==========================================================================
# Token thật theo đơn vị (mục 3.2 B) và reasoning_effort=low chỉ khi dựng KG (mục 3.2 C)
# ==========================================================================


def _usage_result(prompt: int, completion: int, reasoning: int) -> ChatResult:
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    }
    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content="ok"))],
        llm_output={"token_usage": usage},
    )


def _spend_tokens_while_building(
    monkeypatch: pytest.MonkeyPatch, runner: ragas_runner.RagasUnitRunner
) -> None:
    """Đơn vị tiêu 60 token (vào 50, ra 10, suy luận 5) ngay lúc dựng KG: 35 ở tài khoản 1, 25 ở 2."""
    router = runner.router
    # Token tiêu từ trước đơn vị này không được tính vào đơn vị.
    router._record_usage(0, _usage_result(100, 50, 10))
    original_build = runner._build_knowledge_graph

    def _build_and_spend(unit: EvalUnit) -> Any:
        router._record_usage(1, _usage_result(20, 5, 2))
        router._record_usage(0, _usage_result(30, 5, 3))
        return original_build(unit)

    monkeypatch.setattr(runner, "_build_knowledge_graph", _build_and_spend)


def test_run_unit_tokens_la_hieu_so_token_that_cua_router_khong_tinh_luot_truoc_do(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    _patch_generation(monkeypatch, runner)
    _spend_tokens_while_building(monkeypatch, runner)

    result = _run(runner, tmp_path, QuestionQuota(single_hop=1))

    assert result.tokens == 60
    assert result.reasoning_tokens == 5


@pytest.mark.parametrize("unit_fails", [False, True], ids=["ok", "don-vi-loi"])
def test_log_token_cuoi_don_vi_co_tong_va_theo_tai_khoan_ke_ca_khi_don_vi_loi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    unit_fails: bool,
):
    runner = _runner(monkeypatch)
    failure = RuntimeError("429") if unit_fails else None
    _patch_generation(monkeypatch, runner, error=failure)
    _spend_tokens_while_building(monkeypatch, runner)

    with caplog.at_level(logging.INFO, logger=ragas_runner.logger.name):
        if unit_fails:
            with pytest.raises(UnitGenerationError):
                _run(runner, tmp_path, QuestionQuota(single_hop=1))
        else:
            _run(runner, tmp_path, QuestionQuota(single_hop=1))

    messages = [record.getMessage() for record in caplog.records]
    records = [message for message in messages if "Token A.md#1" in message]
    assert len(records) == 1
    assert "tổng 60 (vào 50, ra 10, suy luận 5)" in records[0]
    assert "theo tài khoản: #1=35, #2=25, #3=0" in records[0]
    assert "key-" not in caplog.text  # chỉ số thứ tự và số, không lộ key


def test_dung_kg_chay_trong_reasoning_effort_low_con_persona_va_sample_thi_khong(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    runner = _runner(monkeypatch)
    seen = _patch_generation(monkeypatch, runner)
    # Bỏ bản fake để `run_unit` dùng hàm dựng KG thật (chỉ thay transforms).
    monkeypatch.delattr(runner, "_build_knowledge_graph")
    build_efforts: list[str | None] = []

    def _spy_apply(graph: Any, transforms: Any, **_kwargs: Any) -> None:
        build_efforts.append(runner.router.current_reasoning_effort)

    monkeypatch.setattr(ragas_runner, "default_transforms", lambda **_kwargs: [])
    monkeypatch.setattr(ragas_runner, "apply_transforms", _spy_apply)

    _run(runner, tmp_path, QuestionQuota(single_hop=2))

    single = seen["synthesizers"]["single_hop"]
    assert build_efforts == ["low"]
    assert seen["persona_calls"][0]["effort"] is None
    assert single.probed == [None, None, None]  # 1 lần sinh scenario + 2 sample
    assert runner.router.current_reasoning_effort is None


def test_dung_kg_loi_giua_chung_van_go_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
):
    runner = _runner(monkeypatch)

    def _failing_apply(graph: Any, transforms: Any, **_kwargs: Any) -> None:
        raise RuntimeError("429 giữa lúc dựng KG")

    monkeypatch.setattr(ragas_runner, "default_transforms", lambda **_kwargs: [])
    monkeypatch.setattr(ragas_runner, "apply_transforms", _failing_apply)

    with pytest.raises(RuntimeError):
        runner._build_knowledge_graph(_unit())

    assert runner.router.current_reasoning_effort is None


def test_adapt_prompts_khong_ha_reasoning_effort(monkeypatch: pytest.MonkeyPatch):
    runner = _runner(monkeypatch)
    efforts: list[str | None] = []

    async def _fake_adapt(self: Any, language: str, llm: Any, *_a: Any) -> Any:
        efforts.append(runner.router.current_reasoning_effort)
        return dict(self.get_prompts())

    monkeypatch.setattr(PromptMixin, "adapt_prompts", _fake_adapt)

    runner._get_synthesizers()

    assert efforts == [None, None, None]


# ==========================================================================
# Khoá hành vi ragas 0.4.3 mà bước sinh câu tự lặp phụ thuộc (mục 3.3)
# ==========================================================================


def test_khoa_api_ragas_ma_buoc_sinh_persona_scenario_sample_phu_thuoc():
    persona_params = inspect.signature(generate_personas_from_kg).parameters
    scenario_params = inspect.signature(BaseSynthesizer.generate_scenarios).parameters
    sample_params = inspect.signature(BaseSynthesizer.generate_sample).parameters

    assert {"kg", "llm", "num_personas"} <= set(persona_params)
    assert not inspect.iscoroutinefunction(generate_personas_from_kg)
    scenario_names = list(scenario_params)[:4]
    assert scenario_names == ["self", "n", "knowledge_graph", "persona_list"]
    assert list(sample_params)[:2] == ["self", "scenario"]
    assert inspect.iscoroutinefunction(BaseSynthesizer.generate_scenarios)
    assert inspect.iscoroutinefunction(BaseSynthesizer.generate_sample)


@pytest.mark.parametrize(
    "synthesizer_class",
    [
        ragas_runner.CleanSingleHopSynthesizer,
        ragas_runner.CleanMultiHopAbstractSynthesizer,
        ragas_runner.CleanMultiHopSpecificSynthesizer,
    ],
)
def test_ba_synthesizer_thuc_te_giu_coroutine_va_ten_khop_khoa_progress(
    monkeypatch: pytest.MonkeyPatch, synthesizer_class: type
):
    runner = _runner(monkeypatch)

    synthesizer = synthesizer_class(llm=runner.llm)

    assert inspect.iscoroutinefunction(synthesizer.generate_scenarios)
    assert inspect.iscoroutinefunction(synthesizer.generate_sample)
    assert synthesizer.name in testset_generator.SYNTHESIZER_TYPES


def test_ten_synthesizer_trong_fake_khop_bang_synthesizer_types_cua_generator():
    types = testset_generator.SYNTHESIZER_TYPES
    inverse = {kind: name for name, kind in types.items()}

    assert _NAME_OF_KIND == inverse


def test_calculate_split_values_lam_tron_len_tich_tong_voi_trong_so():
    splits, _ = calculate_split_values([0.75, 0.25], 4)

    assert splits == [3, 1]


def test_sample_that_cua_ragas_dump_exclude_none_hop_le_voi_golden_test_case():
    sample = SingleTurnSample(
        user_input="Câu hỏi?", reference="Đáp án.", reference_contexts=["Ngữ cảnh"]
    )
    row = sample.model_dump(exclude_none=True)
    row["synthesizer_name"] = _SINGLE

    case = GoldenTestCase.model_validate(row)

    assert (case.user_input, case.reference) == ("Câu hỏi?", "Đáp án.")
    assert case.reference_contexts == ["Ngữ cảnh"]
    assert case.empty_required_fields() == []
