"""CLI sinh golden testset RAGAS Phase 1 (evaluation_spec.md mục 4, 7, 9).

Chỉ gọi thẳng `testset_generator.py`, không chứa business logic. Hai lệnh:

- `generate`: sinh câu hỏi theo từng đơn vị (Chương/Mục), nhỏ -> lớn, checkpoint sau mỗi
  đơn vị. Chạy lại đúng lệnh đó hôm sau để làm tiếp khi hết quota Groq (mục 4.5).
  `--dry-run` chỉ in kế hoạch + tiến độ, không gọi LLM, không cần key Groq, không cần
  `ragas` (chạy được trên venv thường).
- `finalize`: chốt đúng 180 câu từ `golden_testset_raw.json` đã review -> `golden_testset.json`.

`ragas` (và `langchain-community`) chỉ nằm trong dependency-group `eval`
(`pyproject.toml`, không cài khi `uv sync` mặc định) — lệnh `generate` thật cần chạy với
`--group eval --no-group production` (xem evaluation_spec.md mục 3). `uv run` không cờ sẽ
re-sync về `default-groups` và kéo `openai` về bản production trong khi `ragas` vẫn còn
trong venv, nên mọi lệnh `uv run` liên quan tới `eval` đều cần đủ cờ:
    uv sync --group eval --no-group production
    uv run --group eval --no-group production tools/generate_testset.py generate --dry-run
    uv run --group eval --no-group production tools/generate_testset.py generate
    uv run --group eval --no-group production tools/generate_testset.py generate \\
        --only "Luật bảo hiểm y tế.md#6" --append
    uv run --group eval --no-group production tools/generate_testset.py finalize
Xong việc, chạy `uv sync` (không cờ) để trả venv về profile mặc định.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from production_legal_qa_rag.evaluation.corpus_loader import DEFAULT_MARKDOWN_DIR
from production_legal_qa_rag.evaluation.testset_generator import (
    DEFAULT_OUTPUT_DIR,
    TARGET_SIZE,
    EvalInputError,
    UnitGenerationError,
    finalize_golden_testset,
    generate_testset,
    plan_generation,
    summarize_progress,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def generate(
    markdown_dir: Path = typer.Option(
        DEFAULT_MARKDOWN_DIR, help="Thư mục chứa văn bản pháp luật .md nguồn."
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        help="Thư mục lưu golden_testset_raw.json, generation_progress.json, knowledge_graph/.",
    ),
    only: list[str] = typer.Option(
        [],
        "--only",
        help='Chỉ chạy văn bản "<tên>" hoặc đơn vị "<tên>#<số thứ tự>"; lặp lại được.',
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="In kế hoạch theo thứ tự chạy + trạng thái + ước lượng token/ngày; không gọi LLM.",
    ),
    reuse_knowledge_graph: bool = typer.Option(
        False,
        help=(
            "Nạp lại KG đã lưu của đơn vị ĐÃ XONG (dùng với --append). Đơn vị chưa xong "
            "luôn tự dùng lại KG còn sót từ lần lỗi trước, không cần cờ này."
        ),
    ),
    append: bool = typer.Option(
        False,
        help=(
            "Chạy lại cả đơn vị đã xong, nối thêm câu vào raw và cộng dồn progress "
            "(sinh bù). Bắt buộc đi kèm --only."
        ),
    ),
    testset_size: int | None = typer.Option(
        None,
        help="Ghi đè tổng số câu (mặc định 240), chia cho các đơn vị đã chọn.",
    ),
) -> None:
    """Sinh golden testset theo từng đơn vị.

    Mã thoát: 0 xong; 1 một đơn vị lỗi giữa chừng (quota...), chạy lại để làm tiếp;
    2 đầu vào/cấu hình sai (không gọi LLM).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        if dry_run:
            typer.echo(
                plan_generation(
                    markdown_dir,
                    output_dir,
                    only=only,
                    append=append,
                    testset_size=testset_size,
                )
            )
            return
        report = generate_testset(
            markdown_dir,
            output_dir,
            only=only,
            reuse_knowledge_graph=reuse_knowledge_graph,
            append=append,
            testset_size=testset_size,
        )
    except EvalInputError as error:
        typer.echo(f"Lỗi đầu vào: {error}", err=True)
        raise typer.Exit(2) from error
    except UnitGenerationError as error:
        typer.echo(f"{error}", err=True)
        typer.echo(summarize_progress(markdown_dir, output_dir), err=True)
        typer.echo(
            "Chạy lại đúng lệnh này khi quota Groq được reset để làm tiếp.", err=True
        )
        raise typer.Exit(1) from error
    typer.echo(
        f"Xong {len(report.generated_units)} đơn vị, +{report.new_questions} câu "
        f"(bỏ qua {len(report.skipped_units)}). {summarize_progress(markdown_dir, output_dir)}"
    )


@app.command()
def finalize(
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, help="Thư mục chứa golden_testset_raw.json (đã review)."
    ),
) -> None:
    """Chốt đúng 180 câu từ raw đã review -> golden_testset.json (mã thoát 2 nếu đầu vào sai)."""
    try:
        cases = finalize_golden_testset(output_dir)
    except EvalInputError as error:
        typer.echo(f"Lỗi đầu vào: {error}", err=True)
        raise typer.Exit(2) from error
    typer.echo(
        f"Đã ghi {len(cases)}/{TARGET_SIZE} câu vào {output_dir / 'golden_testset.json'}."
    )


if __name__ == "__main__":
    """
    uv sync --group eval --no-group production
    uv run --group eval --no-group production tools/generate_testset.py generate --dry-run
    uv run --group eval --no-group production tools/generate_testset.py generate

    # Chạy nền, tắt terminal vẫn chạy tiếp (tiến độ được checkpoint theo từng đơn vị):
    setsid nohup uv run --group eval --no-group production tools/generate_testset.py generate > data/eval/generate.log 2>&1 &
    
    # Xem log (Ctrl+C chỉ dừng tail, không dừng generate)
    tail -f data/eval/generate.log
    
    # Dừng hẳn khi cần
    pkill -f generate_testset.py     
    """
    app()
