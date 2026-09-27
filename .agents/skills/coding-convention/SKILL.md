---
name: coding-convention
description: Áp dụng quy ước Python production của production-legal-qa-rag khi tạo hoặc sửa code, CLI, cấu trúc package, hoặc *_spec.md.
---

# Coding convention

- Source importable nằm trong `src/production_legal_qa_rag/`; tách logic nghiệp vụ thành package và module theo trách nhiệm. Đặt `<package>_spec.md` cạnh package mô tả, không tạo `specs/` ở root.
- Dùng `snake_case.py` cho module, `PascalCase` cho class, `snake_case` cho function/variable, `UPPER_SNAKE_CASE` cho constant, và `_` cho nội bộ. Tên phải mô tả hành vi rõ ràng.
- Mọi function có type hints cho tham số và return. Một function chỉ làm một việc; tách khi logic phức tạp hoặc quá khoảng 40–50 dòng. Imports theo thứ tự stdlib, third-party, local.
- Dùng `uv`; format/lint bằng `ruff format` và `ruff check`; type-check bằng `mypy` theo `pyproject.toml`. Không thay thế tool hiện có nếu không được yêu cầu.
- Dùng Pydantic v2 `BaseModel` cho contract dữ liệu giữa packages/bước pipeline; dùng Typer, không dùng argparse, cho CLI trong `tools/`.
- Comment giải thích vì sao, không lặp lại điều code đã thể hiện. Mỗi module, class, public function có docstring PEP 257; module docstring đứng trước imports. `TODO`/`FIXME` phải có ngữ cảnh ngắn.
- Khi chọn dependency mới: ưu tiên giải pháp production ổn định, miễn phí/open-source, hiệu năng tốt, và tương thích Pydantic/LangChain/Celery/Dagster; không thêm dependency nếu stack hiện có giải quyết rõ ràng.
