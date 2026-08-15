from __future__ import annotations

import os
import re
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
from multiprocessing import freeze_support
from pathlib import Path

from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from tqdm import tqdm

INPUT_DIRECTORY = Path("data/raw")
OUTPUT_DIRECTORY = Path("data/markdown")
REGISTRY_PATH = Path("data/ingestion.db")

MAX_WORKERS = 2

_REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT UNIQUE NOT NULL,
    source_hash TEXT NOT NULL,
    markdown_path TEXT,
    status TEXT NOT NULL,
    processed_at DATETIME,
    error_message TEXT
)
"""


@dataclass(frozen=True)
class PendingDocument:
    """Một DOCX đã được main process chọn để gửi sang worker."""

    source_path: Path
    source_hash: str
    output_directory: Path


def calculate_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Tính SHA-256 theo từng khối để không nạp toàn bộ DOCX vào bộ nhớ."""
    digest = sha256()
    with file_path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scan_docx_files(input_directory: Path) -> list[Path]:
    """Quét đệ quy và trả về danh sách DOCX theo thứ tự ổn định."""
    if not input_directory.exists():
        return []
    return sorted(
        (
            path
            for path in input_directory.rglob("*")
            if path.is_file() and path.suffix.lower() == ".docx"
        ),
        key=lambda path: str(path).casefold(),
    )


def open_registry(registry_path: Path) -> sqlite3.Connection:
    """Mở registry và bảo đảm schema tồn tại.

    Hàm này chỉ được gọi ở main process. Worker không nhận connection hoặc
    đường dẫn SQLite.
    """
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(registry_path)
    connection.row_factory = sqlite3.Row
    connection.execute(_REGISTRY_SCHEMA)
    connection.commit()
    return connection


def recover_interrupted_documents(connection: sqlite3.Connection) -> int:
    """Đưa các lần chạy bị ngắt về ``failed`` để lần chạy mới có thể retry."""
    cursor = connection.execute(
        """
        UPDATE documents
        SET status = 'failed',
            error_message = 'Lần xử lý trước bị gián đoạn'
        WHERE status = 'processing'
        """
    )
    connection.commit()
    return cursor.rowcount


def get_document(
    connection: sqlite3.Connection,
    source_path: Path,
) -> sqlite3.Row | None:
    """Lấy trạng thái ingestion hiện tại của một đường dẫn nguồn."""
    return connection.execute(
        "SELECT * FROM documents WHERE source_path = ?",
        (str(source_path.resolve()),),
    ).fetchone()


def should_process(record: sqlite3.Row | None, current_hash: str) -> bool:
    """Áp dụng đúng decision logic: new, changed hoặc failed thì xử lý."""
    return (
        record is None
        or record["status"] == "failed"
        or record["source_hash"] != current_hash
    )


def mark_processing(
    connection: sqlite3.Connection,
    source_path: Path,
    source_hash: str,
) -> None:
    """Insert/update tài liệu trước khi main process submit worker."""
    connection.execute(
        """
        INSERT INTO documents (
            source_path,
            source_hash,
            markdown_path,
            status,
            processed_at,
            error_message
        )
        VALUES (?, ?, NULL, 'processing', NULL, NULL)
        ON CONFLICT(source_path) DO UPDATE SET
            source_hash = excluded.source_hash,
            status = 'processing',
            processed_at = NULL,
            error_message = NULL
        """,
        (str(source_path.resolve()), source_hash),
    )


def mark_completed(
    connection: sqlite3.Connection,
    source_path: Path,
    source_hash: str,
    markdown_path: Path,
) -> None:
    """Ghi kết quả thành công; chỉ main process gọi hàm này."""
    connection.execute(
        """
        UPDATE documents
        SET source_hash = ?,
            markdown_path = ?,
            status = 'completed',
            processed_at = ?,
            error_message = NULL
        WHERE source_path = ?
        """,
        (
            source_hash,
            str(markdown_path.resolve()),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            str(source_path.resolve()),
        ),
    )


def mark_failed(
    connection: sqlite3.Connection,
    source_path: Path,
    error_message: str,
) -> None:
    """Lưu lỗi worker để tài liệu được retry ở lần chạy sau."""
    connection.execute(
        """
        UPDATE documents
        SET status = 'failed',
            processed_at = NULL,
            error_message = ?
        WHERE source_path = ?
        """,
        (error_message, str(source_path.resolve())),
    )


_HEADING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "part",
        re.compile(
            r"^Phần\s+(?:thứ\s+)?(?:[IVXLCDM]+|\d+)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "chapter",
        re.compile(r"^Chương\s+(?:[IVXLCDM]+|\d+)\b", re.IGNORECASE),
    ),
    (
        "section",
        re.compile(r"^Mục\s+(?:[IVXLCDM]+|\d+)\b", re.IGNORECASE),
    ),
    (
        "article",
        re.compile(r"^Điều\s+\d+[a-z]?\b", re.IGNORECASE),
    ),
    (
        "appendix",
        re.compile(r"^Phụ\s+lục\b", re.IGNORECASE),
    ),
)

_DOCUMENT_TYPE_PATTERN = re.compile(
    r"^(?:BỘ\s+LUẬT|LUẬT|NGHỊ\s+ĐỊNH|NGHỊ\s+QUYẾT|"
    r"QUYẾT\s+ĐỊNH|THÔNG\s+TƯ)$",
    re.IGNORECASE,
)


def _split_paragraphs(content: str) -> list[str]:
    """Tách nội dung text thành các đoạn, bỏ các đoạn trống."""
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", content)
        if paragraph.strip()
    ]


def _cell_lines(cell) -> list[str]:
    """Lấy các dòng có nội dung trong một ô Word."""
    return [
        " ".join(line.split())
        for paragraph in cell.paragraphs
        for line in paragraph.text.splitlines()
        if line.strip()
    ]


def _cell_to_markdown(cell) -> str:
    """Chuyển nội dung một ô Word thành nội dung hợp lệ trong bảng Markdown."""
    return "<br>".join(_cell_lines(cell)).replace("|", r"\|")


def _single_row_table_to_html(table: Table) -> str:
    """Xuất bảng một hàng không có header bằng HTML hợp lệ trong Markdown."""
    cells = [
        f"      <td>{'<br>'.join(escape(line) for line in _cell_lines(cell))}</td>"
        for cell in table.rows[0].cells
    ]
    return "\n".join(
        [
            "<table>",
            "  <tbody>",
            "    <tr>",
            *cells,
            "    </tr>",
            "  </tbody>",
            "</table>",
        ]
    )


def table_to_markdown(table: Table) -> str:
    """Chuyển bảng DOCX sang cú pháp bảng phù hợp trong Markdown."""
    rows = [
        [_cell_to_markdown(cell) for cell in row.cells]
        for row in table.rows
    ]

    if not rows:
        return ""

    # Bảng Markdown luôn có một hàng header. Với bảng Word chỉ có một hàng
    # (thường là công thức hoặc khu vực ký tên), dùng HTML để tránh sinh header
    # rỗng ở phía trên dữ liệu.
    if len(rows) == 1:
        return _single_row_table_to_html(table)

    column_count = max(len(row) for row in rows)
    normalized_rows = [
        row + [""] * (column_count - len(row))
        for row in rows
    ]

    header = normalized_rows[0]
    body_rows = normalized_rows[1:]

    separator = ["---"] * column_count
    markdown_rows = [header, separator, *body_rows]
    return "\n".join(
        f"| {' | '.join(row)} |"
        for row in markdown_rows
    )


def extract_docx_blocks(input_path: Path) -> list[tuple[str, str]]:
    """Trích xuất paragraph và table theo đúng thứ tự xuất hiện trong DOCX."""
    document = Document(str(input_path))
    blocks: list[tuple[str, str]] = []

    for element in document.element.body.iterchildren():
        if isinstance(element, CT_P):
            paragraph = Paragraph(element, document).text.strip()
            if paragraph:
                blocks.append(("paragraph", paragraph))
        elif isinstance(element, CT_Tbl):
            markdown_table = table_to_markdown(Table(element, document))
            if markdown_table:
                blocks.append(("table", markdown_table))

    return blocks


def _is_uppercase_title(paragraph: str) -> bool:
    """Kiểm tra một đoạn có phải dòng tiêu đề viết hoa hay không."""
    letters = "".join(character for character in paragraph if character.isalpha())
    return len(letters) >= 2 and letters.isupper()


def _heading_kind(paragraph: str) -> str | None:
    """Trả về loại heading pháp lý được nhận diện từ nội dung đoạn."""
    for kind, pattern in _HEADING_PATTERNS:
        if pattern.match(paragraph):
            return kind
    return None


def _heading_level(
    kind: str,
    *,
    part_level: int | None,
    chapter_level: int | None,
    section_level: int | None,
) -> int:
    """Xác định cấp Markdown từ quan hệ phân cấp của văn bản pháp luật."""
    if kind == "part":
        return 2

    if kind == "chapter":
        return (part_level or 1) + 1

    if kind == "section":
        return (chapter_level or part_level or 1) + 1

    if kind == "article":
        return (section_level or chapter_level or part_level or 1) + 1

    # Phụ lục là một nhánh độc lập ở cấp cao nhất dưới tên file.
    return 2


def convert_legal_blocks_to_markdown(
    blocks: list[tuple[str, str]],
    document_title: str,
) -> str:
    """Tạo Markdown có heading từ paragraph và table đã trích xuất.

    Các DOCX đầu vào dùng style ``Normal`` cho mọi đoạn nên không thể dựa vào
    paragraph style. Thay vào đó, hàm nhận diện các nhãn cấu trúc cố định như
    ``Chương``, ``Mục``, ``Điều`` và ``Phụ lục``. Các bảng đã được chuyển sang
    Markdown được giữ nguyên vị trí, không đưa vào nhận diện heading.
    """
    markdown_parts = []

    part_level: int | None = None
    chapter_level: int | None = None
    section_level: int | None = None
    index = 0

    while index < len(blocks):
        block_type, paragraph = blocks[index]

        if block_type == "table":
            markdown_parts.append(paragraph)
            index += 1
            continue

        kind = _heading_kind(paragraph)

        # Tên loại văn bản (LUẬT, NGHỊ ĐỊNH, ...) thường nằm ở phần đầu và có
        # dòng viết hoa ngay sau nó là tên chính thức của văn bản.
        is_document_title = (
            index < 20
            and _DOCUMENT_TYPE_PATTERN.fullmatch(paragraph) is not None
        )

        if kind is None and not is_document_title:
            markdown_parts.append(paragraph)
            index += 1
            continue

        if is_document_title:
            level = 2
        else:
            level = _heading_level(
                kind,
                part_level=part_level,
                chapter_level=chapter_level,
                section_level=section_level,
            )

        # Trong các văn bản này, tên Chương/Phần và tên văn bản thường nằm ở
        # đoạn in hoa kế tiếp. Gộp chúng để heading đầy đủ và dễ dùng khi chunk.
        heading_text = paragraph
        next_index = index + 1
        if (
            next_index < len(blocks)
            and blocks[next_index][0] == "paragraph"
            and _is_uppercase_title(blocks[next_index][1])
            and _heading_kind(blocks[next_index][1]) is None
        ):
            heading_text = f"{heading_text} — {blocks[next_index][1]}"
            next_index += 1

        markdown_parts.append(f"{'#' * level} {heading_text}")

        if kind == "part":
            part_level = level
            chapter_level = None
            section_level = None
        elif kind == "chapter":
            chapter_level = level
            section_level = None
        elif kind == "section":
            section_level = level
        elif kind == "appendix":
            part_level = None
            chapter_level = None
            section_level = None

        index = next_index

    return "\n\n".join(markdown_parts) + "\n"


def convert_legal_content_to_markdown(content: str, document_title: str) -> str:
    """Chuyển text thuần sang Markdown, giữ tương thích với cách gọi cũ."""
    blocks = [("paragraph", paragraph) for paragraph in _split_paragraphs(content)]
    return convert_legal_blocks_to_markdown(blocks, document_title)


def convert_docx_to_markdown(
    file_path: str,
    output_directory: str,
) -> tuple[str, str, int]:
    """
    Đọc một file DOCX và tạo file Markdown tương ứng.

    Hàm này được chạy bên trong worker process nên phải:
    - Được khai báo ở cấp module.
    - Chỉ nhận và trả về dữ liệu có thể serialize.
    """
    input_path = Path(file_path)
    output_dir = Path(output_directory)

    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    if input_path.suffix.lower() != ".docx":
        raise ValueError(f"File không phải DOCX: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    blocks = extract_docx_blocks(input_path)

    if not blocks:
        raise ValueError(f"Không trích xuất được nội dung: {input_path}")

    output_path = output_dir / f"{input_path.stem}.md"

    markdown_content = convert_legal_blocks_to_markdown(
        blocks,
        document_title=input_path.stem,
    )

    # Giữ nguyên tiếng Việt bằng UTF-8.
    output_path.write_text(
        markdown_content,
        encoding="utf-8",
    )

    return str(input_path), str(output_path), os.getpid()


def main() -> None:
    input_directory = INPUT_DIRECTORY.resolve()
    output_directory = OUTPUT_DIRECTORY.resolve()
    successful_files: list[tuple[str, str]] = []
    failed_files: list[tuple[str, str]] = []
    skipped_files: list[str] = []
    pending_documents: list[PendingDocument] = []

    with open_registry(REGISTRY_PATH.resolve()) as connection:
        recovered_count = recover_interrupted_documents(connection)
        if recovered_count:
            print(f"Khôi phục {recovered_count} tài liệu bị gián đoạn để retry.")

        for source_path in scan_docx_files(input_directory):
            try:
                current_hash = calculate_sha256(source_path)
            except Exception as error:
                failed_files.append((str(source_path), str(error)))
                tqdm.write(
                    f"✗ {source_path.name} | Lỗi: {type(error).__name__}"
                )
                continue

            record = get_document(connection, source_path)
            if not should_process(record, current_hash):
                skipped_files.append(str(source_path))
                continue

            relative_parent = source_path.relative_to(input_directory).parent
            task = PendingDocument(
                source_path=source_path,
                source_hash=current_hash,
                output_directory=output_directory / relative_parent,
            )
            mark_processing(connection, source_path, current_hash)
            pending_documents.append(task)

        # Trạng thái processing phải bền vững trước khi tạo worker.
        connection.commit()

        if pending_documents:
            worker_count = min(MAX_WORKERS, len(pending_documents))
            if worker_count < 1:
                raise ValueError("MAX_WORKERS phải lớn hơn hoặc bằng 1")

            filename_width = max(
                len(document.source_path.name)
                for document in pending_documents
            )

            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_document = {
                    executor.submit(
                        convert_docx_to_markdown,
                        str(document.source_path),
                        str(document.output_directory),
                    ): document
                    for document in pending_documents
                }

                with tqdm(
                    total=len(future_to_document),
                    desc="Converting DOCX",
                    unit="file",
                    dynamic_ncols=True,
                    bar_format=(
                        "{desc}: {percentage:3.0f}%|{bar}| "
                        "{n_fmt}/{total_fmt}"
                    ),
                ) as progress_bar:
                    for future in as_completed(future_to_document):
                        document = future_to_document[future]
                        terminal_message: str
                        try:
                            input_path, output_path, worker_pid = future.result()
                            mark_completed(
                                connection,
                                document.source_path,
                                document.source_hash,
                                Path(output_path),
                            )
                            successful_files.append((input_path, output_path))
                            terminal_message = (
                                f"✓ {Path(input_path).name:<{filename_width}} "
                                f"PID {worker_pid}"
                            )
                        except Exception as error:
                            error_message = str(error)
                            mark_failed(
                                connection,
                                document.source_path,
                                error_message,
                            )
                            failed_files.append(
                                (str(document.source_path), error_message)
                            )
                            terminal_message = (
                                f"✗ {document.source_path.name:<{filename_width}} "
                                f"Lỗi: {type(error).__name__}"
                            )
                        finally:
                            # Commit từng kết quả để không mất trạng thái của
                            # các file đã xong nếu tiến trình chính bị ngắt.
                            connection.commit()
                            progress_bar.update(1)
                        tqdm.write(terminal_message)

    print("\nKết quả")
    print("─" * 36)
    print(f"✓ Thành công : {len(successful_files)}")
    print(f"✗ Thất bại   : {len(failed_files)}")
    print(f"↷ Bỏ qua     : {len(skipped_files)}")


if __name__ == "__main__":
    freeze_support()
    main()
