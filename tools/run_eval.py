"""Run resumable Phase 2 stages with optional dependencies loaded only for scoring."""

from __future__ import annotations

import asyncio
import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from production_legal_qa_rag.evaluation.run_models import EvalConfig
from production_legal_qa_rag.evaluation.testset_generator import EvalInputError

app = typer.Typer(add_completion=False, no_args_is_help=True)
Testset = Annotated[Path, typer.Option(help="Golden testset JSON.")]
OutputDir = Annotated[Path, typer.Option(help="Directory of stage JSONL checkpoints.")]
Limit = Annotated[int | None, typer.Option(min=1, help="Use the first N test cases.")]
RetryFailed = Annotated[
    bool, typer.Option(help="Retry failed checkpoints only; keep successes.")
]
Workers = Annotated[
    int, typer.Option(min=1, max=9, help="Concurrency for Groq stages.")
]


class ConfigChoice(StrEnum):
    """Supported CLI retrieval configurations."""

    MMR_ON = "mmr_on"
    MMR_OFF = "mmr_off"


Choice = Annotated[
    ConfigChoice | None,
    typer.Option(
        "--config", help="Required for generate and answer/precision scoring."
    ),
]


class _SafeLogFilter(logging.Filter):
    """Remove third-party exception payloads that can contain prompts."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.exc_info = None
        record.exc_text = None
        return True


@app.callback()
def options(
    ctx: typer.Context,
    testset: Testset = Path("data/eval/golden_testset.json"),
    output_dir: OutputDir = Path("data/eval/phase2"),
    limit: Limit = None,
    retry_failed: RetryFailed = False,
    workers: Workers = 9,
    config: Choice = None,
) -> None:
    """Select common scope before the stage command."""
    ctx.obj = EvalConfig(
        testset=testset,
        output_dir=output_dir,
        limit=limit,
        retry_failed=retry_failed,
        workers=workers,
        config=config,
    )
    for name in (
        "production_legal_qa_rag.retrieval.hyde",
        "production_legal_qa_rag.retrieval.query_embedder",
    ):
        logging.getLogger(name).addFilter(_SafeLogFilter())
    logging.getLogger("ragas.executor").setLevel(logging.CRITICAL)


def _execute(ctx: typer.Context, stage: str) -> None:
    config: EvalConfig = ctx.obj
    try:
        if stage == "hyde":
            from production_legal_qa_rag.evaluation.hyde_stage import run_hyde

            result = asyncio.run(run_hyde(config))
        elif stage == "embed":
            from production_legal_qa_rag.evaluation.embed_stage import run_embed

            result = asyncio.run(run_embed(config))
        elif stage == "retrieve":
            from production_legal_qa_rag.evaluation.retrieve_stage import run_retrieve

            result = asyncio.run(run_retrieve(config))
        elif stage == "generate":
            from production_legal_qa_rag.evaluation.generate_stage import run_generate

            result = asyncio.run(run_generate(config))
        elif stage.startswith("score-"):
            from production_legal_qa_rag.evaluation.scoring import run_scoring

            result = run_scoring(config, stage.removeprefix("score-"))  # type: ignore[arg-type]
        elif stage == "status":
            from production_legal_qa_rag.evaluation.report import stage_status

            result = stage_status(config)
        else:
            from production_legal_qa_rag.evaluation.report import write_report

            report = write_report(config)
            comparison = report["retrieval_comparison"]["overall"]
            typer.echo("Config\tcontext_recall\tn")
            for choice in ("mmr_on", "mmr_off"):
                typer.echo(
                    f"{choice}\t{comparison[choice]['mean']}\t{comparison[choice]['n']}"
                )
            typer.echo(
                f"Wins/losses/ties: {comparison['wins']}/{comparison['losses']}/{comparison['ties']}; excluded={comparison['excluded']}"
            )
            typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
            return
        summaries = result if isinstance(result, dict) else {stage: result}
        typer.echo(
            json.dumps(
                {name: value.model_dump() for name, value in summaries.items()},
                ensure_ascii=False,
                indent=2,
            )
        )
        if stage != "status" and any(
            value.quota_exhausted or value.errors for value in summaries.values()
        ):
            raise typer.Exit(1)
    except EvalInputError, ValidationError:
        typer.echo(
            "Đầu vào/cấu hình sai; kiểm tra testset, JSONL, --config và biến môi trường bắt buộc.",
            err=True,
        )
        raise typer.Exit(2) from None
    except ImportError:
        typer.echo("Stage này cần uv run --group eval --no-group production.", err=True)
        raise typer.Exit(2) from None
    except (OSError, RuntimeError) as error:
        typer.echo(f"Stage dừng: {type(error).__name__}.", err=True)
        raise typer.Exit(1) from None


def _register(name: str) -> None:
    def command(
        ctx: typer.Context,
        testset: Testset = Path("data/eval/golden_testset.json"),
        output_dir: OutputDir = Path("data/eval/phase2"),
        limit: Limit = None,
        retry_failed: RetryFailed = False,
        workers: Workers = 9,
        config: Choice = None,
    ) -> None:
        """Run the selected stage and print resumable checkpoint statistics."""
        updates = {
            key: value
            for key, value in {
                "testset": testset,
                "output_dir": output_dir,
                "limit": limit,
                "retry_failed": retry_failed,
                "workers": workers,
                "config": config.value if config is not None else None,
            }.items()
            if (source := ctx.get_parameter_source(key)) is not None
            and source.name == "COMMANDLINE"
        }
        ctx.obj = ctx.obj.model_copy(update=updates)
        _execute(ctx, name)

    app.command(name)(command)


for _stage in (
    "hyde",
    "embed",
    "retrieve",
    "score-recall",
    "generate",
    "score-answers",
    "score-precision",
    "report",
    "status",
):
    _register(_stage)


if __name__ == "__main__":
    """
    uv run --group eval --no-group production tools/run_eval.py <param>
    """
    app()
