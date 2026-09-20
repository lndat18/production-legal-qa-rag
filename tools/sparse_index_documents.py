"""CLI build Pinecone sparse index (BM25) từ chunk JSON.

Fit BM25 trên toàn bộ corpus, ghi `bm25_params.json`, rồi xoá và upsert lại
toàn bộ sparse index. Chạy lại sau mỗi lần `embedding/` build lại dense index.
`.env` cần có `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` và
`PINECONE_SPARSE_INDEX_NAME` (dùng chung cloud/region của `VectorDBSettings`).

Cách dùng:
    uv run python tools/sparse_index_documents.py
    uv run python tools/sparse_index_documents.py --chunks-dir data/chunks --params-out data/bm25/bm25_params.json
"""

from __future__ import annotations

import traceback
from pathlib import Path

import typer

from production_legal_qa_rag.retrieval.sparse_index import build_index

DEFAULT_CHUNKS_DIR = Path("data/chunks")
DEFAULT_PARAMS_OUT = Path("data/bm25/bm25_params.json")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    chunks_dir: Path = typer.Option(
        DEFAULT_CHUNKS_DIR, help="Thư mục .json Chunk đầu vào."
    ),
    params_out: Path = typer.Option(
        DEFAULT_PARAMS_OUT, help="Đường dẫn ghi bm25_params.json."
    ),
) -> None:
    """Fit BM25, ghi tham số và full-refresh Pinecone sparse index."""
    try:
        total = build_index(chunks_dir, params_out)
    except Exception:  # noqa: BLE001 - CLI báo lỗi bằng exit code
        traceback.print_exc()
        raise typer.Exit(code=1) from None
    print(f"Đã index {total} chunk vào sparse index; params: {params_out}")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
