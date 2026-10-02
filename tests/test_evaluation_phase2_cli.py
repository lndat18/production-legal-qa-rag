"""Exercise real Typer option propagation and exit codes without service clients."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from production_legal_qa_rag.evaluation.run_models import EvalConfig, StageSummary
from production_legal_qa_rag.evaluation.testset_generator import EvalInputError
from tools import run_eval


@pytest.mark.parametrize(
    "command",
    [
        "hyde",
        "embed",
        "retrieve",
        "score-recall",
        "generate",
        "score-answers",
        "score-precision",
        "status",
        "report",
    ],
)
@pytest.mark.parametrize("placement", ["before", "after"])
def test_cli_all_common_options_reach_stage(
    command: str, placement: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[EvalConfig] = []
    prefix = "production_legal_qa_rag.evaluation."

    async def async_stage(config: EvalConfig) -> StageSummary:
        seen.append(config)
        return StageSummary(done=1)

    def sync_stage(config: EvalConfig, *args: Any) -> dict[str, StageSummary]:
        seen.append(config)
        return {command: StageSummary(done=1)}

    def report(config: EvalConfig) -> dict[str, Any]:
        seen.append(config)
        return {
            "retrieval_comparison": {
                "overall": {
                    "mmr_on": {"mean": 0.8, "n": 1},
                    "mmr_off": {"mean": 0.7, "n": 1},
                    "wins": 1,
                    "losses": 0,
                    "ties": 0,
                    "excluded": 0,
                }
            }
        }

    if command in {"hyde", "embed", "retrieve", "generate"}:
        monkeypatch.setitem(
            sys.modules,
            prefix + command + "_stage",
            SimpleNamespace(**{"run_" + command: async_stage}),
        )
    elif command.startswith("score-"):
        monkeypatch.setitem(
            sys.modules, prefix + "scoring", SimpleNamespace(run_scoring=sync_stage)
        )
    else:
        monkeypatch.setitem(
            sys.modules,
            prefix + "report",
            SimpleNamespace(stage_status=sync_stage, write_report=report),
        )
    flags = [
        "--testset",
        str(tmp_path / "custom.json"),
        "--output-dir",
        str(tmp_path / "results"),
        "--limit",
        "2",
        "--retry-failed",
        "--workers",
        "3",
        "--config",
        "mmr_off",
    ]
    args = [*flags, command] if placement == "before" else [command, *flags]
    result = CliRunner().invoke(run_eval.app, args)
    assert result.exit_code == 0, result.output
    assert len(seen) == 1
    assert seen[0].model_dump() == {
        "testset": tmp_path / "custom.json",
        "output_dir": tmp_path / "results",
        "limit": 2,
        "retry_failed": True,
        "workers": 3,
        "config": "mmr_off",
    }


@pytest.mark.parametrize(
    "summary,expected",
    [
        (StageSummary(done=1), 0),
        (StageSummary(errors=1), 1),
        (StageSummary(pending=1, quota_exhausted=True), 1),
    ],
)
def test_cli_stage_exit_codes(
    summary: StageSummary, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def stage(config: EvalConfig) -> StageSummary:
        return summary

    monkeypatch.setitem(
        sys.modules,
        "production_legal_qa_rag.evaluation.hyde_stage",
        SimpleNamespace(run_hyde=stage),
    )
    assert CliRunner().invoke(run_eval.app, ["hyde"]).exit_code == expected


def test_cli_invalid_input_exit_two_without_provider_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stage(config: EvalConfig) -> StageSummary:
        raise EvalInputError("private input payload")

    monkeypatch.setitem(
        sys.modules,
        "production_legal_qa_rag.evaluation.hyde_stage",
        SimpleNamespace(run_hyde=stage),
    )
    result = CliRunner().invoke(run_eval.app, ["hyde"])
    assert result.exit_code == 2
    assert "private input" not in result.output
