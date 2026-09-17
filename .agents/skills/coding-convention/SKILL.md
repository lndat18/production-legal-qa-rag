---
name: coding-convention
description: Áp dụng quy ước Python production của production-legal-qa-rag khi tạo hoặc sửa code, cấu trúc package, hay *_spec.md; gồm naming, docstring, Pydantic, Typer, Ruff và mypy. Không dùng cho thay đổi dữ liệu/chunks thuần túy.
---

# Coding convention chuẩn production

Áp dụng các quy ước này cho source code trong repository. Tôn trọng cấu trúc và cấu hình hiện có trước khi đề xuất thư viện hoặc layout mới.

## Kiến trúc thư mục và package

- Toàn bộ source code import được nằm trong `src/production_legal_qa_rag/`.
- Tách mỗi logic nghiệp vụ thành một package riêng. Ví dụ logic chunking thuộc `src/production_legal_qa_rag/chunking/`.
- Chia package thành các module theo trách nhiệm; không gom toàn bộ logic vào một file.
- Với spec mới, đặt `<package>_spec.md` cạnh package nó mô tả, không tạo thư mục `specs/` ở root.
- Khi viết hoặc sửa spec, không đề cập đến việc viết tests; agent CI/CD chịu trách nhiệm này.

## Naming và cấu trúc mã

- Module/file dùng `snake_case.py`; class dùng `PascalCase`; function và variable dùng `snake_case`; constant dùng `UPPER_SNAKE_CASE`.
- Biến hoặc hàm nội bộ của module dùng tiền tố `_`.
- Chọn tên mô tả hành vi rõ ràng; tránh viết tắt tối nghĩa, ví dụ dùng `chunk_legal_document` thay cho `proc_doc`.
- Bắt buộc type hint cho mọi function signature, gồm tham số và return type.
- Mỗi function chỉ nên làm một việc và thường không dài quá khoảng 40–50 dòng. Tách hàm khi logic phức tạp.
- Thứ tự import: standard library, third-party, local; mỗi nhóm cách nhau một dòng trống.

## Format, lint và type checking

- Dùng cấu hình trong `pyproject.toml`; format bằng `ruff format` và kiểm tra bằng `ruff check`.
- Ruff là formatter/linter chuẩn của repo, thay cho Black, isort và flake8.
- Dùng mypy cho type checking vì repo đã cấu hình plugin `pydantic.mypy`. Chỉ đề xuất `ty` khi người dùng yêu cầu đánh giá hoặc thay đổi công cụ type checking.
- Dùng `uv` để quản lý dependency và môi trường. Không thay thế bằng pip, Poetry hoặc pyenv trừ khi người dùng yêu cầu.
- Khi cần audit dependency, ưu tiên `pip-audit`; chỉ thêm hoặc chạy công cụ này khi phạm vi yêu cầu bao gồm audit bảo mật.

## Comments và docstring

- Comment giải thích **vì sao**, không diễn giải lại điều code đã thể hiện.
- Không để comment thừa hoặc lặp tên hàm. `# TODO:` và `# FIXME:` phải có ngữ cảnh ngắn.
- Mỗi module, class và public function phải có docstring theo PEP 257.
- Mỗi file `.py` bắt đầu bằng module-level docstring, trước imports. Bắt đầu bằng tóm tắt ngắn; thêm đoạn mô tả chi tiết khi cần.
- Docstring của public function dùng đoạn tóm tắt, rồi `Args:`, `Returns:` và `Raises:` khi phù hợp.

```python
"""Tách văn bản pháp luật thành các đoạn nhỏ theo cấp Khoản/Điểm.

Module này xử lý bước chunking trong pipeline ingestion, nhận đầu vào là
Markdown đã chuẩn hóa và trả về danh sách đoạn văn bản sẵn sàng embedding.
"""

from production_legal_qa_rag.chunking.models import Chunk
```

```python
def split_by_khoan(text: str, max_tokens: int = 192) -> list[str]:
    """Tách văn bản pháp luật thành các đoạn theo cấp Khoản.

    Args:
        text: Nội dung văn bản đầu vào đã chuẩn hóa.
        max_tokens: Giới hạn token tối đa mỗi đoạn.

    Returns:
        Danh sách các đoạn văn bản đã tách.
    """
```

## Data validation và CLI

- Dữ liệu trao đổi giữa packages hoặc giữa các bước pipeline phải dùng Pydantic v2 `BaseModel`; không dùng `dict` thô hoặc `dataclass` trần cho contract dữ liệu.
- Dùng `pydantic-settings` cho cấu hình lấy từ môi trường.
- CLI và script trong `tools/` dùng Typer, không dùng `argparse`.

## Nguyên tắc chọn công cụ

Khi cần bổ sung thư viện mới, ưu tiên theo thứ tự:

1. Được cộng đồng dùng ổn định trong production.
2. Miễn phí, open-source và có hiệu năng tốt; ưu tiên lựa chọn Rust khi tính năng tương đương.
3. Tích hợp tốt với stack hiện có: Pydantic, LangChain, LlamaIndex, Celery hoặc Dagster.
4. Không đưa dependency mới vào project nếu thư viện hiện có đã giải quyết được yêu cầu một cách rõ ràng.
