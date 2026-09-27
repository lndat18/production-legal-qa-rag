---
name: coding-convention
description: Quy ước coding chuẩn production cho project — kiến trúc thư mục, naming, format, docstring, và bộ công cụ hiện đại (pydantic, typer, ruff)
---
# Coding convention chuẩn production

## Kiến trúc thư mục & package

- Toàn bộ source code có thể import được phải nằm trong `src/production_legal_qa_rag/`
- Mỗi logic nghiệp vụ tách thành 1 package riêng. Ví dụ: logic chunking → `src/production_legal_qa_rag/chunking/`
- Trong mỗi package, chia nhỏ thành nhiều module phục vụ cho logic đó (không gộp hết vào 1 file)
- Không cần đề cập đến việc viết tests trong spec — đã có agent CI/CD riêng đảm nhiệm

## Naming conventions

- Module/file: `snake_case.py`
- Class: `PascalCase`
- Function/variable: `snake_case`
- Constant: `UPPER_SNAKE_CASE`
- Private (nội bộ module): prefix `_underscore`
- Tên phải mô tả rõ hành vi, tránh viết tắt tối nghĩa (`chunk_legal_document` thay vì `proc_doc`)

## Định dạng & cấu trúc mã

- Format bằng **Ruff** (`ruff format` + `ruff check`) — thay thế Black/isort/flake8, chạy trong `pyproject.toml`
- Type hint bắt buộc cho mọi function signature (tham số + return type)
- Ưu tiên `src/` layout (đã đúng với cấu trúc hiện tại của project)
- Import order: stdlib → third-party → local, cách nhau 1 dòng trắng (Ruff tự sắp xếp)
- Mỗi function nên làm 1 việc, độ dài hợp lý (không quá ~40-50 dòng), tách nhỏ nếu logic phức tạp

## Comments

- Comment giải thích **tại sao** (why), không giải thích **cái gì** (what) — code đã tự nói cái gì
- Không để comment thừa/lặp lại tên hàm
- Đánh dấu rõ `# TODO:`, `# FIXME:` kèm ngữ cảnh ngắn nếu để lại việc chưa xong

## Docstring chuẩn PEP 8 / PEP 257

- Mọi module, class, public function đều có docstring
- **Module-level docstring**: mỗi file `.py` bắt đầu bằng docstring mô tả module này làm công việc gì, đặt ngay dòng đầu file, trước phần import
- Format: dòng tóm tắt ngắn → dòng trống → mô tả chi tiết (nếu cần) → `Args:` / `Returns:` / `Raises:` (với function)

Ví dụ module-level:

```python
"""Tách văn bản pháp luật thành các đoạn nhỏ theo cấp Khoản/Điểm.

Module này xử lý bước chunking trong pipeline ingestion, nhận đầu vào là
Markdown đã chuẩn hóa và trả về danh sách đoạn văn bản sẵn sàng embedding.
"""

from production_legal_qa_rag.chunking.models import Chunk
```

Ví dụ function-level:

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

## Data validation — dùng Pydantic

- Mọi cấu trúc dữ liệu trao đổi giữa các package (input/output của pipeline) định nghĩa bằng **Pydantic v2** `BaseModel`, không dùng `dict` thô hoặc `dataclass` trần
- Pydantic tự validate kiểu dữ liệu tại runtime, giảm lỗi ẩn khi dữ liệu từ nguồn ngoài (docx, API) không đúng format

## CLI — dùng Typer thay cho argparse

- Mọi script/CLI trong `tools/` dùng **Typer** thay vì `argparse`
- Typer tự sinh type hint validation, help text, và autocomplete từ function signature — ít boilerplate hơn argparse

## Ưu tiên công cụ hiện đại, miễn phí (2026 stack)

| Việc                        | Công cụ khuyến nghị                                                        | Thay thế cho                       |
| ---------------------------- | ------------------------------------------------------------------------------ | ----------------------------------- |
| Quản lý dependency         | `uv` (Astral)                                                                | pip, pip-tools, poetry, pyenv       |
| Lint + format                | `ruff`                                                                       | black, isort, flake8, pyupgrade     |
| Type checking                | `mypy` hoặc `ty` (beta, nhanh hơn nhưng chưa đủ plugin cho pydantic) | —                                  |
| Validate dữ liệu           | `pydantic` v2                                                                | dataclass thô, dict validation tay |
| CLI                          | `typer`                                                                      | argparse                            |
| Audit dependency (bảo mật) | `pip-audit`                                                                  | —                                  |

Khi cần chọn thư viện mới cho 1 tác vụ, ưu tiên: (1) đang được cộng đồng lớn dùng trong production 2026, (2) viết bằng Rust/tốc độ cao nếu có lựa chọn tương đương, (3) miễn phí/open-source, (4) tích hợp tốt với stack hiện tại (Pydantic, LangChain, Celery/Dagster).
