"""CLI chạy tuần tự pha embedding checkpoint rồi upsert Pinecone.

Cách dùng:
    uv run python tools/embed_documents.py
    uv run python tools/embed_documents.py --chunks-dir data/chunks --embeddings-dir data/embeddings
"""

from __future__ import annotations

from pathlib import Path

import typer

from production_legal_qa_rag.embedding.pipeline import embed, upsert

DEFAULT_CHUNKS_DIR = Path("data/chunks")
DEFAULT_EMBEDDINGS_DIR = Path("data/embeddings")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    chunks_dir: Path = typer.Option(
        DEFAULT_CHUNKS_DIR, help="Thư mục .json Chunk đầu vào."
    ),
    embeddings_dir: Path = typer.Option(
        DEFAULT_EMBEDDINGS_DIR, help="Thư mục checkpoint EmbeddedChunk."
    ),
) -> None:
    """Chạy pha 1 embed xong rồi pha 2 upsert Pinecone."""
    embed_exit_code = embed(chunks_dir, embeddings_dir)
    if embed_exit_code != 0:
        raise typer.Exit(code=embed_exit_code)
    raise typer.Exit(code=upsert(embeddings_dir))


if __name__ == "__main__":
    app()
