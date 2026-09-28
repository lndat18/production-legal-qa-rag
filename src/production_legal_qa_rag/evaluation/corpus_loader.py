"""Đọc corpus markdown pháp luật thành tài liệu LangChain (evaluation_spec.md mục 2, 4).

`TestsetGenerator` tự chunk/build `KnowledgeGraph` nội bộ từ tài liệu gốc —
module này chỉ đọc nguyên văn từng file `.md`, không cắt nhỏ hay biến đổi nội
dung (khác `chunking/`, vốn phục vụ mục đích embedding sản xuất).
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

DEFAULT_MARKDOWN_DIR: Path = Path("data/markdown")


def load_markdown_documents(
    markdown_dir: Path = DEFAULT_MARKDOWN_DIR,
) -> list[Document]:
    """Đọc toàn bộ file `.md` trong `markdown_dir` thành `Document` LangChain.

    Args:
        markdown_dir: Thư mục chứa các văn bản pháp luật `.md` hoàn chỉnh.

    Returns:
        Danh sách `Document`, mỗi phần tử có `page_content` là nội dung
        nguyên văn của một file và `metadata={"source": <tên file>}`.

    Raises:
        FileNotFoundError: `markdown_dir` không tồn tại hoặc không có file `.md`.
        ValueError: Một file `.md` rỗng (fail fast trước khi gọi `TestsetGenerator`,
            evaluation_spec.md mục 8).
    """
    if not markdown_dir.is_dir():
        raise FileNotFoundError(f"Thư mục markdown không tồn tại: {markdown_dir}")

    markdown_paths = sorted(markdown_dir.glob("*.md"))
    if not markdown_paths:
        raise FileNotFoundError(f"Không tìm thấy file .md nào trong {markdown_dir}")

    documents: list[Document] = []
    for path in markdown_paths:
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"File markdown trống: {path}")
        documents.append(Document(page_content=content, metadata={"source": path.name}))

    return documents
