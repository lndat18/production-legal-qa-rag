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

Bảng trên là công cụ nền tảng, dùng xuyên suốt cả repo — không đổi theo từng bài toán.

**Ngoài bảng trên, chọn thư viện/kỹ thuật cho một bài toán cụ thể là quyết định mở theo
từng bài toán, không có danh sách cố định "luôn dùng X cho Y".** Ưu tiên phương án đo được
là hiệu quả nhất cho đúng bài toán đó — code ngắn gọn hơn, ít bề mặt lỗi hơn, ít
round-trip/I/O hơn, dễ test hơn — dựa trên bằng chứng cụ thể (tài liệu chính thức,
benchmark, so sánh số dòng/độ phức tạp thực tế), không phải thói quen hay "nghe quen tên".
Đồng thời cân nhắc: (1) cộng đồng lớn dùng trong production 2026, (2) hiệu năng cao nếu có
lựa chọn tương đương, (3) miễn phí/open-source, (4) tích hợp tốt với stack hiện tại
(Pydantic, Typer,...). Một lựa chọn tốt cho bài toán này (vd. ingest dữ liệu có cấu trúc
sẵn) có thể không phải lựa chọn tốt cho bài toán khác nhìn giống nó (vd. truy vấn ngôn ngữ
tự nhiên trên cùng dữ liệu) — đừng suy diễn một quyết định thành rule chung. Quyết định cụ
thể kèm lý do so sánh ghi trong `*_spec.md` của package liên quan (mục "Công cụ & công
nghệ"), không phải ở file quy ước chung này — vì lựa chọn tốt nhất có thể khác nhau giữa
các bài toán và đổi theo thời gian khi công nghệ mới xuất hiện.
