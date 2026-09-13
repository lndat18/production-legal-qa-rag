"""CLI mỏng gọi `formatting.pipeline.convert_directory`.

Chuyển toàn bộ file `.docx` trong thư mục đầu vào sang Markdown có cấu trúc
trong thư mục đầu ra. Không giữ trạng thái giữa các lần chạy: mỗi lần chạy
xử lý lại toàn bộ input và ghi đè output.

Cách dùng:
    uv run python tools/format_documents.py
    uv run python tools/format_documents.py --raw-dir data/raw --out-dir data/markdown
"""

from __future__ import annotations

from pathlib import Path

import typer

from production_legal_qa_rag.formatting.pipeline import convert_directory

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUT_DIR = Path("data/markdown")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    raw_dir: Path = typer.Option(DEFAULT_RAW_DIR, help="Thư mục .docx đầu vào."),
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, help="Thư mục .md đầu ra."),
) -> None:
    """Chuyển toàn bộ `.docx` trong `raw_dir` sang Markdown trong `out_dir`."""
    exit_code = convert_directory(raw_dir, out_dir)
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
