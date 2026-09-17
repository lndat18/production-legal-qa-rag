"""Orchestration mức trên: DOCX -> Markdown, ghi atomic, gom summary.

Stateless, không có DB tracking — mỗi lần chạy xử lý lại toàn bộ
``data/raw/`` và ghi đè ``data/markdown/``. Chạy tuần tự (không
multiprocessing): với quy mô corpus hiện tại (6 file), xử lý tuần tự đủ
nhanh và đơn giản hơn hẳn so với phối hợp ProcessPoolExecutor. Lỗi ở một
file không chặn các file còn lại trong batch.

``convert_docx_to_markdown`` ghép 3 phần theo mục 1.1, 2 spec: front matter
(Groq, theo chunk) + nội dung ở giữa (pipeline hiện có, không đổi) + back
matter (Groq, theo chunk, nếu có, ngăn cách bằng dòng ``---``). Rate limiter
dùng chung xuyên suốt cả lần chạy (mục 1.2, 5 spec) là singleton module-level
trong ``llm_client.py`` — tự động dùng chung cho mọi lệnh gọi trong tiến
trình, không cần khởi tạo/truyền tham chiếu tường minh qua ``pipeline.py``.

**[Mới 2026-09-17]** ``convert_docx_to_markdown`` dựng job qua
``frontmatter.build_prompts``/``backmatter.build_prompts`` (thuần, không gọi
Groq), nối ``before_prompts + after_prompts + chunk_prompts`` thành 1 danh
sách duy nhất, gọi **1 lần duy nhất**
``llm_client.convert_chunks_concurrently`` (mục 1.3 spec — 2 worker đồng
thời nếu có ``GROQ_API_KEY_2``, tuần tự nếu không), rồi cắt kết quả theo
ranh giới đã nhớ trước khi gọi ``frontmatter.assemble``/``backmatter.assemble``.
"""

from __future__ import annotations

import collections
import os
import traceback
from pathlib import Path

from production_legal_qa_rag.formatting import (
    backmatter,
    docx_reader,
    emitter,
    frontmatter,
    llm_client,
    tables,
    validator,
)
from production_legal_qa_rag.formatting.models import FormattingResult, QcWarning


def _compose_markdown(
    front_markdown: str, middle_markdown: str, back_markdown: str
) -> str:
    """Ghép front matter + nội dung ở giữa + back matter (mục 1.1, 2 spec).

    Không có YAML, không có heading gán riêng cho front/back matter — cả hai
    là văn bản thường nối trực tiếp. Back matter (nếu có) ngăn cách bằng một
    dòng ``---``.
    """
    sections = [section for section in (front_markdown, middle_markdown) if section]
    markdown = "\n\n".join(sections)
    if back_markdown:
        markdown = f"{markdown}\n\n---\n\n{back_markdown}"
    return f"{markdown}\n"


def convert_docx_to_markdown(path: str | Path) -> FormattingResult:
    """Chuyển một file `.docx` thành Markdown có cấu trúc.

    Args:
        path: Đường dẫn tới file `.docx` nguồn.

    Returns:
        Kết quả chuyển đổi: markdown và danh sách cảnh báo QC.

    Raises:
        ValueError: Khi không trích xuất được nội dung nào từ file.
    """
    path = Path(path)
    blocks = docx_reader.read_docx(path)
    if not blocks:
        raise ValueError(f"Không trích xuất được nội dung: {path}")

    warnings: list[QcWarning] = []

    # S0 — biên front matter: block đầu tiên khớp regex heading cấu trúc.
    fm_boundary = frontmatter.find_boundary(blocks)
    front_blocks = blocks[:fm_boundary]
    rest = blocks[fm_boundary:]

    # S1 — biên back matter: sau bảng chữ ký cuối cùng, nếu có.
    middle_raw, back_blocks, split_warnings = backmatter.split_backmatter(rest)
    warnings.extend(split_warnings)

    # S2 — lọc bảng đính kèm khỏi vùng nội dung ở giữa.
    middle_blocks, table_warnings = tables.filter_middle_tables(middle_raw)
    warnings.extend(table_warnings)

    # S3 — dựng job Groq (thuần, không I/O) cho front matter/back matter
    # (mục 1.2), gộp chung thành 1 danh sách và gọi Groq đúng 1 lần (mục 1.3
    # spec — 2 worker đồng thời nếu có GROQ_API_KEY_2, tuần tự nếu không).
    front_jobs = frontmatter.build_prompts(front_blocks)
    back_prompts = backmatter.build_prompts(back_blocks)

    before_count = len(front_jobs.before_prompts)
    after_count = len(front_jobs.after_prompts)
    all_prompts = [
        *front_jobs.before_prompts,
        *front_jobs.after_prompts,
        *back_prompts,
    ]
    all_results = llm_client.convert_chunks_concurrently(all_prompts)

    before_results = all_results[:before_count]
    after_results = all_results[before_count : before_count + after_count]
    back_results = all_results[before_count + after_count :]

    # S3b — ghép kết quả (thuần, không I/O) — không có fallback khi lỗi.
    front_markdown, fm_warnings = frontmatter.assemble(
        front_jobs.title_text, before_results, after_results
    )
    warnings.extend(fm_warnings)

    back_markdown, bm_warnings = backmatter.assemble(back_results)
    warnings.extend(bm_warnings)

    # S4 — emit nội dung ở giữa (không đổi so với bản cũ).
    middle_markdown = "\n\n".join(emitter.emit(middle_blocks))
    warnings.extend(validator.validate(middle_markdown))

    # S5 — ghép.
    markdown = _compose_markdown(front_markdown, middle_markdown, back_markdown)

    return FormattingResult(markdown=markdown, warnings=warnings)


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
