"""Orchestration mức trên: Markdown -> Khoản -> chunk, ghi atomic, gom summary.

Stateless, không DB tracking — cùng triết lý với `formatting/pipeline.py`:
mỗi lần chạy xử lý lại toàn bộ `data/markdown/` và ghi đè `data/chunks/`.
Chạy tuần tự (không multiprocessing); lỗi ở 1 file không chặn các file còn
lại trong batch (mục 9).
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

from production_legal_qa_rag.chunking.models import Chunk, ChunkingResult
from production_legal_qa_rag.chunking.parser import parse_markdown
from production_legal_qa_rag.chunking.splitter import split_khoan
from production_legal_qa_rag.config import EmbeddingSettings


def convert_markdown_to_chunks(path: str | Path) -> ChunkingResult:
    """Chuyển 1 file `.md` (`data/markdown/*.md`) thành `ChunkingResult`.

    Args:
        path: Đường dẫn tới file `.md` nguồn.

    Returns:
        Toàn bộ chunk sinh ra từ file, kèm thống kê số Khoản/số Khoản bị cắt.
    """
    max_tokens = EmbeddingSettings().max_tokens
    tree = parse_markdown(path)

    chunks: list[Chunk] = []
    split_khoan_count = 0
    for khoan in tree.khoans:
        khoan_chunks = split_khoan(
            khoan, source_document=tree.source_document, max_tokens=max_tokens
        )
        chunks.extend(khoan_chunks)
        if len(khoan_chunks) > 1:
            split_khoan_count += 1

    return ChunkingResult(
        source_path=str(path),
        chunks=chunks,
        khoan_count=len(tree.khoans),
        split_khoan_count=split_khoan_count,
    )


def write_atomic(path: str | Path, chunks: list[Chunk]) -> None:
    """Ghi danh sách `Chunk` ra JSONL bằng file tạm + rename nguyên tử.

    Giống `formatting/pipeline.py::write_atomic`: không bao giờ để lại file
    dở dang nếu tiến trình bị ngắt giữa chừng.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for chunk in chunks:
                handle.write(chunk.model_dump_json())
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _scan_markdown_files(markdown_dir: Path) -> list[Path]:
    """Quét đệ quy, trả về danh sách `.md` theo thứ tự ổn định."""
    if not markdown_dir.exists():
        return []
    return sorted(
        (path for path in markdown_dir.rglob("*.md") if path.is_file()),
        key=lambda path: str(path).casefold(),
    )


def _output_path_for(source: Path, markdown_dir: Path, out_dir: Path) -> Path:
    """Giữ nguyên cấu trúc thư mục con, đổi đuôi thành `.jsonl`."""
    relative = source.relative_to(markdown_dir)
    return (out_dir / relative).with_suffix(".jsonl")


def _print_summary(
    successful: list[str],
    failed: list[tuple[str, str]],
    total_chunks: int,
    total_split_khoan: int,
) -> None:
    print("\nKết quả")
    print("─" * 36)
    print(f"✓ Thành công        : {len(successful)}")
    print(f"✗ Thất bại          : {len(failed)}")
    print(f"Tổng số chunk       : {total_chunks}")
    print(f"Số Khoản bị cắt nhỏ : {total_split_khoan}")

    for source_path, message in failed:
        print(f"\n✗ {Path(source_path).name}\n  {message}")


def convert_directory(markdown_dir: Path, out_dir: Path) -> int:
    """Chunk hoá toàn bộ `.md` trong `markdown_dir`, ghi JSONL vào `out_dir`.

    Xử lý tuần tự, lỗi ở 1 file không chặn các file còn lại. In summary (số
    file thành công/lỗi, tổng số chunk, số Khoản bị cắt) khi kết thúc.

    Args:
        markdown_dir: Thư mục chứa file `.md` nguồn.
        out_dir: Thư mục ghi file `.jsonl` đầu ra.

    Returns:
        0 nếu tất cả thành công, 1 nếu có ít nhất một file lỗi.
    """
    sources = _scan_markdown_files(markdown_dir)
    if not sources:
        print(f"Không tìm thấy .md nào trong {markdown_dir}")
        return 0

    successful: list[str] = []
    failed: list[tuple[str, str]] = []
    total_chunks = 0
    total_split_khoan = 0

    for source_path in sources:
        output_path = _output_path_for(source_path, markdown_dir, out_dir)
        try:
            result = convert_markdown_to_chunks(source_path)
            write_atomic(output_path, result.chunks)
        except Exception as error:  # noqa: BLE001 - lỗi 1 file không được chặn cả batch
            traceback.print_exc()
            failed.append((str(source_path), repr(error)))
            continue

        successful.append(str(source_path))
        total_chunks += len(result.chunks)
        total_split_khoan += result.split_khoan_count
        print(f"✓ {source_path.name} ({len(result.chunks)} chunk)")

    _print_summary(successful, failed, total_chunks, total_split_khoan)
    return 1 if failed else 0
