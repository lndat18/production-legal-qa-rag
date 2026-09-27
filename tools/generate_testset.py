"""CLI sinh golden testset RAGAS Phase 1 (evaluation_spec.md mục 4, 7, 9).

Chỉ gọi thẳng `testset_generator.py`, không chứa business logic. Đây là job
offline không gấp -- chấp nhận chạy lâu (có thể nhiều giờ) do `testset_size=360`.

`ragas` (và `langchain-community`) chỉ nằm trong dependency-group `eval`
(`pyproject.toml`, không cài khi `uv sync` mặc định) -- luôn chạy script này với
`--group eval --no-group production` (xem evaluation_spec.md mục 3).

Cách dùng:
    uv sync --group eval --no-group production
    uv run tools/generate_testset.py
    uv run tools/generate_testset.py --reuse-knowledge-graph
"""

from __future__ import annotations

from pathlib import Path

import typer

from production_legal_qa_rag.evaluation.corpus_loader import DEFAULT_MARKDOWN_DIR
from production_legal_qa_rag.evaluation.testset_generator import (
    DEFAULT_OUTPUT_DIR,
    generate_golden_testset,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    markdown_dir: Path = typer.Option(
        DEFAULT_MARKDOWN_DIR, help="Thư mục chứa văn bản pháp luật .md nguồn."
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        help="Thư mục lưu golden_testset.json và knowledge_graph.json.",
    ),
    reuse_knowledge_graph: bool = typer.Option(
        False,
        help=(
            "Nạp lại knowledge_graph.json đã lưu (nếu có) thay vì build lại từ "
            "đầu, tiết kiệm lượt gọi Groq (mục 9.5)."
        ),
    ),
) -> None:
    """Sinh golden testset RAGAS từ `data/markdown/*.md`, lưu ra `output_dir`."""
    golden_cases = generate_golden_testset(
        markdown_dir,
        output_dir,
        reuse_knowledge_graph=reuse_knowledge_graph,
    )
    typer.echo(f"Đã sinh {len(golden_cases)} câu hỏi vào {output_dir}.")


if __name__ == "__main__":
    app()
