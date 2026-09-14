"""CLI mỏng gọi `chunking.pipeline.convert_directory`.

Chuyển toàn bộ file `.md` trong thư mục đầu vào (output của `formatting/`)
thành chunk JSONL trong thư mục đầu ra. Không giữ trạng thái giữa các lần
chạy: mỗi lần chạy xử lý lại toàn bộ input và ghi đè output.

Cách dùng:
    uv run python tools/chunk_documents.py
    uv run python tools/chunk_documents.py --markdown-dir data/markdown --out-dir data/chunks
"""

from __future__ import annotations

from pathlib import Path

import typer

from production_legal_qa_rag.chunking.pipeline import convert_directory

DEFAULT_MARKDOWN_DIR = Path("data/markdown")
DEFAULT_OUT_DIR = Path("data/chunks")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    markdown_dir: Path = typer.Option(DEFAULT_MARKDOWN_DIR, help="Thư mục .md đầu vào."),
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, help="Thư mục .jsonl đầu ra."),
) -> None:
    """Chuyển toàn bộ `.md` trong `markdown_dir` thành chunk JSONL trong `out_dir`."""
    exit_code = convert_directory(markdown_dir, out_dir)
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
