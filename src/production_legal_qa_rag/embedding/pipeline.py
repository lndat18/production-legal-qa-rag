"""Điều phối hai pha checkpoint embedding rồi full-refresh Pinecone."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from pydantic import BaseModel

from production_legal_qa_rag.chunking.models import Chunk
from production_legal_qa_rag.embedding.hf_client import (
    HFRequestLimitExceeded,
    HuggingFaceEmbedder,
)
from production_legal_qa_rag.embedding.models import EmbeddedChunk
from production_legal_qa_rag.embedding.pinecone_client import upsert_embedded_chunks


def embed(chunks_dir: Path, embeddings_dir: Path) -> int:
    """Pha 1: đọc chunk JSON, gọi HF và ghi checkpoint JSON atomically.

    Args:
        chunks_dir: Thư mục chứa các mảng ``Chunk`` JSON.
        embeddings_dir: Thư mục checkpoint ``EmbeddedChunk`` JSON.

    Returns:
        0 khi mọi file nguồn đọc/ghi thành công, 1 nếu có file lỗi. Batch HF
        lỗi là partial success và không làm thay đổi exit code.
    """
    sources = _scan_json_files(chunks_dir)
    if not sources:
        print(f"Không tìm thấy .json nào trong {chunks_dir}")
        return 0

    embedder = HuggingFaceEmbedder()
    total_chunks = 0
    total_embedded = 0
    skipped_chunk_ids: list[str] = []
    failed: list[tuple[str, str]] = []

    for source_path in sources:
        try:
            chunks = _read_chunks(source_path)
            embedded_chunks, skipped = embedder.embed_chunks(chunks)
            write_atomic(
                _output_path_for(source_path, chunks_dir, embeddings_dir),
                embedded_chunks,
            )
        except HFRequestLimitExceeded:
            raise
        except Exception as error:  # noqa: BLE001 - lỗi một file không chặn corpus
            traceback.print_exc()
            failed.append((str(source_path), repr(error)))
            continue

        total_chunks += len(chunks)
        total_embedded += len(embedded_chunks)
        skipped_chunk_ids.extend(skipped)
        print(
            f"✓ {source_path.name} ({len(embedded_chunks)}/{len(chunks)} chunk embedded)"
        )

    _print_embed_summary(total_chunks, total_embedded, skipped_chunk_ids, failed)
    return 1 if failed else 0


def upsert(embeddings_dir: Path) -> int:
    """Pha 2: đọc toàn bộ checkpoint và full-refresh Pinecone index.

    Args:
        embeddings_dir: Thư mục chứa các mảng ``EmbeddedChunk`` JSON.

    Returns:
        0 khi index được full-refresh thành công, 1 khi đọc checkpoint hoặc
        thao tác Pinecone thất bại.
    """
    sources = _scan_json_files(embeddings_dir)
    if not sources:
        print(f"Không tìm thấy .json nào trong {embeddings_dir}")
        return 0

    try:
        embedded_chunks = [
            chunk
            for source_path in sources
            for chunk in _read_embedded_chunks(source_path)
        ]
        upsert_embedded_chunks(embedded_chunks)
    except Exception:  # noqa: BLE001 - CLI phải trả exit code thay vì crash
        traceback.print_exc()
        return 1

    print(f"✓ Đã upsert {len(embedded_chunks)} vector lên Pinecone")
    return 0


def write_atomic(path: str | Path, embedded_chunks: list[EmbeddedChunk]) -> None:
    """Ghi checkpoint JSON nguyên tử bằng file tạm cùng filesystem.

    Args:
        path: Đường dẫn checkpoint cần ghi.
        embedded_chunks: Các chunk embed thành công của một file nguồn.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                [chunk.model_dump(mode="json") for chunk in embedded_chunks],
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_chunks(path: Path) -> list[Chunk]:
    """Đọc và validate một mảng ``Chunk`` JSON."""
    return _validate_json_array(path, Chunk)


def _read_embedded_chunks(path: Path) -> list[EmbeddedChunk]:
    """Đọc và validate một mảng ``EmbeddedChunk`` JSON."""
    return _validate_json_array(path, EmbeddedChunk)


def _validate_json_array[ModelT: BaseModel](
    path: Path, model_type: type[ModelT]
) -> list[ModelT]:
    """Parse mảng JSON thành Pydantic model được chỉ định."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError(f"{path} phải là một mảng JSON.")
    return [model_type.model_validate(item) for item in payload]


def _scan_json_files(directory: Path) -> list[Path]:
    """Trả về file JSON trực tiếp trong thư mục theo thứ tự ổn định."""
    if not directory.exists():
        return []
    return sorted(
        (path for path in directory.glob("*.json") if path.is_file()),
        key=lambda path: path.name.casefold(),
    )


def _output_path_for(source_path: Path, chunks_dir: Path, embeddings_dir: Path) -> Path:
    """Ánh xạ tên checkpoint 1-1 theo file chunk nguồn."""
    return embeddings_dir / source_path.relative_to(chunks_dir)


def _print_embed_summary(
    total_chunks: int,
    total_embedded: int,
    skipped_chunk_ids: list[str],
    failed: list[tuple[str, str]],
) -> None:
    """In tổng kết pha 1 cùng QC warning cho batch bị bỏ qua."""
    print("\nKết quả embedding")
    print("─" * 36)
    print(f"Tổng số chunk       : {total_chunks}")
    print(f"Embed thành công    : {total_embedded}")
    print(f"Bị bỏ qua           : {len(skipped_chunk_ids)}")
    print(f"File lỗi            : {len(failed)}")
    if skipped_chunk_ids:
        print(f"QC warning chunk_id : {', '.join(skipped_chunk_ids)}")
    for source_path, message in failed:
        print(f"\n✗ {Path(source_path).name}\n  {message}")
