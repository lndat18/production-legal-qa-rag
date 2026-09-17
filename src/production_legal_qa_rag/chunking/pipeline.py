"""Orchestration mức trên: Markdown -> Khoản -> chunk, ghi atomic, gom summary.

Stateless, không DB tracking — cùng triết lý với `formatting/pipeline.py`:
mỗi lần chạy xử lý lại toàn bộ `data/markdown/` và ghi đè `data/chunks/`.
Chạy tuần tự (không multiprocessing); lỗi ở 1 file không chặn các file còn
lại trong batch (mục 9).
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from production_legal_qa_rag.chunking.models import Chunk, ChunkingResult
from production_legal_qa_rag.chunking.parser import parse_markdown
from production_legal_qa_rag.chunking.splitter import split_implicit_khoan, split_khoan
from production_legal_qa_rag.config import EmbeddingSettings

_BACKMATTER_SUFFIX = "Chú thích sửa đổi (cuối văn bản)"


def _ensure_unique_chunk_ids(chunks: list[Chunk], source_document: str) -> None:
    """Invariant chung: `chunk_id` phải duy nhất trong 1 file (mục 2).

    Pinecone upsert theo `chunk_id` (mục 2) -- 2 chunk cùng id khiến dòng ghi
    sau âm thầm đè dòng trước, mất dữ liệu vĩnh viễn, không có cảnh báo nào.
    Đây là lớp bảo vệ chung đặt tại điểm sinh ra TOÀN BỘ chunk của 1 file,
    bất kể nguyên nhân trùng lặp là gì (vd. 2 heading cấp 5 không khớp dạng
    nào đã biết trong cùng 1 Điều -> 2 Khoản ngầm định cùng
    `breadcrumb_prefix`, xem `parser.py`) -- fail loud (raise) thay vì âm
    thầm ghi đè, thay vì chỉ vá riêng từng nguyên nhân cụ thể mỗi lần phát
    hiện. Đúng triết lý mục 9 "lỗi 1 file không chặn cả batch": raise ở đây
    khiến `convert_directory` coi file này là lỗi (catch `Exception`) nhưng
    vẫn tiếp tục xử lý các file còn lại.
    """
    seen_breadcrumbs: dict[str, str] = {}
    for chunk in chunks:
        previous_breadcrumb = seen_breadcrumbs.get(chunk.chunk_id)
        if previous_breadcrumb is not None:
            raise ValueError(
                f"chunk_id trùng lặp trong {source_document!r}: "
                f"{chunk.chunk_id!r} được sinh bởi 2 breadcrumb khác nhau "
                f"({previous_breadcrumb!r} và {chunk.breadcrumb!r}) -- sẽ "
                "ghi đè âm thầm khi upsert lên Pinecone, từ chối ghi file này."
            )
        seen_breadcrumbs[chunk.chunk_id] = chunk.breadcrumb


def convert_markdown_to_chunks(
    path: str | Path, *, max_tokens: int | None = None
) -> ChunkingResult:
    """Chuyển 1 file `.md` (`data/markdown/*.md`) thành `ChunkingResult`.

    Args:
        path: Đường dẫn tới file `.md` nguồn.
        max_tokens: Ngân sách token tối đa cho 1 chunk. Mặc định `None` ->
            đọc từ `EmbeddingSettings().max_tokens` (đọc lại `.env` mỗi lần
            gọi) -- dùng khi gọi hàm này độc lập (vd. test, script rời).
            `convert_directory` load `EmbeddingSettings()` 1 lần rồi truyền
            xuống đây cho mọi file trong batch, tránh đọc lại `.env` mỗi file.

    Returns:
        Toàn bộ chunk sinh ra từ file, kèm thống kê số Khoản/số Khoản bị cắt.

    Raises:
        ValueError: 2 chunk trong cùng file có `chunk_id` trùng nhau (xem
            `_ensure_unique_chunk_ids`) -- lỗi dữ liệu nguồn, chặn file này
            nhưng không chặn `convert_directory` xử lý các file khác.
    """
    if max_tokens is None:
        max_tokens = EmbeddingSettings().max_tokens
    tree = parse_markdown(path)

    chunks: list[Chunk] = []

    # 2 "Khoản ngầm định cấp văn bản" (mục 4.6) -- cắt riêng bằng
    # `split_implicit_khoan`, không tính vào `split_khoan_count` (thống kê
    # đó chỉ đếm Khoản thật, mục 9).
    if tree.frontmatter_content:
        chunks.extend(
            split_implicit_khoan(
                tree.frontmatter_content,
                breadcrumb_prefix=tree.source_document,
                source_document=tree.source_document,
                max_tokens=max_tokens,
            )
        )

    split_khoan_count = 0
    for khoan in tree.khoans:
        khoan_chunks = split_khoan(
            khoan, source_document=tree.source_document, max_tokens=max_tokens
        )
        chunks.extend(khoan_chunks)
        if len(khoan_chunks) > 1:
            split_khoan_count += 1

    if tree.backmatter_content:
        chunks.extend(
            split_implicit_khoan(
                tree.backmatter_content,
                breadcrumb_prefix=f"{tree.source_document} - {_BACKMATTER_SUFFIX}",
                source_document=tree.source_document,
                max_tokens=max_tokens,
            )
        )

    _ensure_unique_chunk_ids(chunks, tree.source_document)

    return ChunkingResult(
        source_path=str(path),
        chunks=chunks,
        khoan_count=len(tree.khoans),
        split_khoan_count=split_khoan_count,
    )


def write_atomic(path: str | Path, chunks: list[Chunk]) -> None:
    """Ghi danh sách `Chunk` ra 1 file JSON (mảng) bằng file tạm + rename nguyên tử.

    Giống `formatting/pipeline.py::write_atomic`: không bao giờ để lại file
    dở dang nếu tiến trình bị ngắt giữa chừng.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                [chunk.model_dump(mode="json") for chunk in chunks],
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


def _scan_markdown_files(markdown_dir: Path) -> list[Path]:
    """Quét đệ quy, trả về danh sách `.md` theo thứ tự ổn định."""
    if not markdown_dir.exists():
        return []
    return sorted(
        (path for path in markdown_dir.rglob("*.md") if path.is_file()),
        key=lambda path: str(path).casefold(),
    )


def _output_path_for(source: Path, markdown_dir: Path, out_dir: Path) -> Path:
    """Giữ nguyên cấu trúc thư mục con, đổi đuôi thành `.json`."""
    relative = source.relative_to(markdown_dir)
    return (out_dir / relative).with_suffix(".json")


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
    """Chunk hoá toàn bộ `.md` trong `markdown_dir`, ghi JSON vào `out_dir`.

    Xử lý tuần tự, lỗi ở 1 file không chặn các file còn lại. In summary (số
    file thành công/lỗi, tổng số chunk, số Khoản bị cắt) khi kết thúc.

    Args:
        markdown_dir: Thư mục chứa file `.md` nguồn.
        out_dir: Thư mục ghi file `.json` đầu ra.

    Returns:
        0 nếu tất cả thành công, 1 nếu có ít nhất một file lỗi.
    """
    sources = _scan_markdown_files(markdown_dir)
    if not sources:
        print(f"Không tìm thấy .md nào trong {markdown_dir}")
        return 0

    max_tokens = EmbeddingSettings().max_tokens

    successful: list[str] = []
    failed: list[tuple[str, str]] = []
    total_chunks = 0
    total_split_khoan = 0

    for source_path in sources:
        output_path = _output_path_for(source_path, markdown_dir, out_dir)
        try:
            result = convert_markdown_to_chunks(source_path, max_tokens=max_tokens)
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
