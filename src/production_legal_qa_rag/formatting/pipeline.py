"""Orchestration mức trên: DOCX -> Markdown, ghi atomic, gom summary.

Stateless, không có DB tracking — mỗi lần chạy xử lý lại toàn bộ
``data/raw/`` và ghi đè ``data/markdown/``. Chạy tuần tự (không
multiprocessing): với quy mô corpus hiện tại (6 file), xử lý tuần tự đủ
nhanh và đơn giản hơn hẳn so với phối hợp ProcessPoolExecutor. Lỗi ở một
file không chặn các file còn lại trong batch.
"""

from __future__ import annotations

import collections
import os
import traceback
from pathlib import Path

from production_legal_qa_rag.formatting import (
    docx_reader,
    emitter,
    footnotes,
    frontmatter,
    tables,
    validator,
)
from production_legal_qa_rag.formatting.footnotes import Footnote
from production_legal_qa_rag.formatting.models import FormattingResult, QcWarning


def convert_docx_to_markdown(path: str | Path) -> FormattingResult:
    """Chuyển một file `.docx` thành Markdown có cấu trúc.

    Args:
        path: Đường dẫn tới file `.docx` nguồn.

    Returns:
        Kết quả chuyển đổi: markdown, front matter và danh sách cảnh báo QC.

    Raises:
        ValueError: Khi không trích xuất được nội dung nào từ file.
    """
    path = Path(path)
    blocks = docx_reader.read_docx(path)
    if not blocks:
        raise ValueError(f"Không trích xuất được nội dung: {path}")

    warnings: list[QcWarning] = []

    # S0 — triage bảng, phải chạy trước mọi thứ khác.
    quoc_hieu_block, body, table_warnings = tables.triage_tables(blocks)
    warnings.extend(table_warnings)

    # S1 — cắt vùng chú thích TRƯỚC khi nhận diện heading.
    region_start, region_warnings = footnotes.find_region_start(body)
    warnings.extend(region_warnings)
    footnote_map: dict[int, Footnote]
    if region_start is None:
        footnote_map = {}
    else:
        footnote_map, parse_warnings = footnotes.resolve_region(body[region_start:])
        warnings.extend(parse_warnings)
        body = body[:region_start]

    # S2 — gỡ marker TRƯỚC khi nhận diện heading.
    body, refs, strip_warnings = footnotes.strip_all(body)
    warnings.extend(strip_warnings)

    # S3 — biên vùng dẫn nhập.
    preamble_end = emitter.find_preamble_end(body)

    # S5 — emit (chạy trước front matter vì is_phu_luc do vòng emit xác định).
    parts, deferred, is_phu_luc, emit_warnings = emitter.emit(
        body, refs, footnote_map, preamble_end
    )
    warnings.extend(emit_warnings)

    # S4 — front matter, dựng trên body đã gỡ marker.
    front_matter, fm_warnings = frontmatter.build_frontmatter(
        body,
        quoc_hieu_block,
        source_path=str(path),
        is_phu_luc=is_phu_luc,
    )
    warnings.extend(fm_warnings)

    # S6 — ghép.
    sections = [front_matter.to_yaml(), *parts]
    if deferred:
        sections.extend(deferred)
    markdown = "\n\n".join(section for section in sections if section) + "\n"

    # S7 — QC.
    warnings.extend(validator.validate(markdown))

    return FormattingResult(
        markdown=markdown, front_matter=front_matter, warnings=warnings
    )


def write_atomic(path: str | Path, content: str) -> None:
    """Ghi nội dung ra file bằng file tạm + rename nguyên tử.

    Ghi ra file ``.tmp`` cùng thư mục rồi ``os.replace`` để không bao giờ để
    lại markdown hỏng một nửa nếu tiến trình bị ngắt giữa chừng. ``os.replace``
    là nguyên tử khi nguồn và đích nằm trên cùng filesystem, nên file tạm phải
    cùng thư mục với đích.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _scan_docx_files(raw_dir: Path) -> list[Path]:
    """Quét đệ quy, trả về danh sách DOCX theo thứ tự ổn định.

    Bỏ qua file tạm của Word (tên bắt đầu bằng ``~$``) — mở một file đó bằng
    python-docx sẽ lỗi.
    """
    if not raw_dir.exists():
        return []
    return sorted(
        (
            path
            for path in raw_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() == ".docx"
            and not path.name.startswith("~$")
        ),
        key=lambda path: str(path).casefold(),
    )


def _output_path_for(source: Path, raw_dir: Path, out_dir: Path) -> Path:
    """Giữ nguyên cấu trúc thư mục con, đổi đuôi thành ``.md``."""
    relative = source.relative_to(raw_dir)
    return (out_dir / relative).with_suffix(".md")


def _print_summary(
    successful: list[str],
    failed: list[tuple[str, str]],
    warning_counter: collections.Counter[str],
) -> None:
    print("\nKết quả")
    print("─" * 36)
    print(f"✓ Thành công : {len(successful)}")
    print(f"✗ Thất bại   : {len(failed)}")

    if warning_counter:
        print("\nCảnh báo theo mã")
        print("─" * 36)
        width = max(len(code) for code in warning_counter)
        for code, count in warning_counter.most_common():
            print(f"  {code:<{width}}  {count}")

    for source_path, message in failed:
        print(f"\n✗ {Path(source_path).name}\n  {message}")


def convert_directory(raw_dir: Path, out_dir: Path) -> int:
    """Chuyển toàn bộ `.docx` trong ``raw_dir`` sang Markdown trong ``out_dir``.

    Xử lý tuần tự, lỗi ở một file không chặn các file còn lại. In summary
    (số file thành công/lỗi, đếm cảnh báo QC theo mã) khi kết thúc.

    Args:
        raw_dir: Thư mục chứa file `.docx` nguồn.
        out_dir: Thư mục ghi file `.md` đầu ra.

    Returns:
        0 nếu tất cả thành công, 1 nếu có ít nhất một file lỗi.
    """
    sources = _scan_docx_files(raw_dir)
    if not sources:
        print(f"Không tìm thấy .docx nào trong {raw_dir}")
        return 0

    successful: list[str] = []
    failed: list[tuple[str, str]] = []
    warning_counter: collections.Counter[str] = collections.Counter()

    for source_path in sources:
        output_path = _output_path_for(source_path, raw_dir, out_dir)
        try:
            result = convert_docx_to_markdown(source_path)
            write_atomic(output_path, result.markdown)
        except Exception as error:  # noqa: BLE001 - lỗi 1 file không được chặn cả batch
            traceback.print_exc()
            failed.append((str(source_path), repr(error)))
            continue

        successful.append(str(source_path))
        warning_counter.update(warning.code for warning in result.warnings)
        print(f"✓ {source_path.name}")

    _print_summary(successful, failed, warning_counter)
    return 1 if failed else 0
