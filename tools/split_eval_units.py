"""CLI chia văn bản pháp luật thành đơn vị sinh testset RAGAS (evaluation_spec.md mục 4.4).

Chỉ chia và ghi ra đĩa để người dùng duyệt — KHÔNG gọi LLM, không cần `ragas`/key Groq.
Mỗi lần chạy xoá và ghi lại toàn bộ thư mục `units/` (kết quả xác định, không giữ trạng thái).

Đầu ra trong `--output-dir` (mặc định `data/eval`):
    units/<văn bản>__<số thứ tự>.md   nguyên văn từng đơn vị
    units_plan.md                     bảng số thứ tự, tên, ký tự, ước lượng token

Cách dùng:
    uv run python tools/split_eval_units.py
    uv run python tools/split_eval_units.py --markdown-dir data/markdown --output-dir data/eval
"""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

from production_legal_qa_rag.evaluation.unit_splitter import (
    MAX_UNIT_CHARS,
    EvalUnit,
    split_directory,
)

DEFAULT_MARKDOWN_DIR = Path("data/markdown")
DEFAULT_OUTPUT_DIR = Path("data/eval")

app = typer.Typer(add_completion=False)


def _unit_path(units_dir: Path, unit: EvalUnit) -> Path:
    stem = Path(unit.source_document).stem
    return units_dir / f"{stem}__{unit.index:02d}.md"


def _render_plan(units: list[EvalUnit]) -> str:
    total_chars = sum(u.char_count for u in units)
    total_tokens = sum(u.estimated_tokens for u in units)
    lines = [
        "# Kế hoạch đơn vị sinh testset",
        "",
        (
            f"{len(units)} đơn vị, {total_chars:,} ký tự, ước lượng ~{total_tokens / 1e6:.2f}M "
            "token (5,5 token/ký tự + 4K mỗi đơn vị, sai số ±30%)."
        ),
        "",
        (
            "Sắp xếp từ nhỏ đến lớn (thứ tự ưu tiên chạy); `#` là số thứ tự đơn vị trong văn bản "
            "(khoá của file `units/` và của `--only`), không đổi khi sắp xếp."
        ),
        "",
        "| Thứ tự chạy | Văn bản | # | Đơn vị | Ký tự | Ước lượng token | Token cộng dồn |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    cumulative = 0
    for rank, unit in enumerate(sorted(units, key=lambda u: u.char_count), start=1):
        cumulative += unit.estimated_tokens
        lines.append(
            f"| {rank} | {unit.source_document} | {unit.index} | {unit.title} "
            f"| {unit.char_count:,} | {unit.estimated_tokens // 1000}K | {cumulative / 1e6:.2f}M |"
        )
    return "\n".join(lines) + "\n"


@app.command()
def main(
    markdown_dir: Path = typer.Option(DEFAULT_MARKDOWN_DIR, help="Thư mục .md nguồn."),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, help="Thư mục ghi units/ và units_plan.md."
    ),
) -> None:
    """Chia `markdown_dir` thành đơn vị Chương/Mục và ghi ra `output_dir`."""
    units = split_directory(markdown_dir)

    units_dir = output_dir / "units"
    if units_dir.exists():
        shutil.rmtree(units_dir)
    units_dir.mkdir(parents=True)
    for unit in units:
        _unit_path(units_dir, unit).write_text(unit.text, encoding="utf-8")
    (output_dir / "units_plan.md").write_text(_render_plan(units), encoding="utf-8")

    oversized = [u for u in units if u.char_count > MAX_UNIT_CHARS]
    typer.echo(
        f"{len(units)} đơn vị -> {units_dir}; kế hoạch: {output_dir / 'units_plan.md'}"
    )
    for unit in oversized:
        typer.echo(
            f"CẢNH BÁO: {unit.source_document}#{unit.index} dài {unit.char_count:,} ký tự "
            f"(> {MAX_UNIT_CHARS:,})"
        )


if __name__ == "__main__":
    app()
