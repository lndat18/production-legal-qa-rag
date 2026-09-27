# Production-ready Legal QA RAG system for Vietnamese legal documents

## Cấu hình biến môi trường (`.env`)

Project có 2 cặp file `.env`/`.env.example` tách riêng, không dùng chung:

| Cặp file | Dùng khi | Cách tạo |
| --- | --- | --- |
| `.env` / `.env.example` (root) | Chạy code Python trực tiếp trên host (`uv run pytest`, `tools/`, `api` không qua Docker) | `cp .env.example .env` rồi điền |
| `deploy/.env` / `deploy/.env.example` | Chạy toàn bộ stack qua Docker Compose (`deploy_spec.md`) | `./deploy/up.sh` — tự tạo, tự điền sẵn key trùng với root `.env` |

Mỗi biến trong 2 file `.env.example` có chú thích `[BẮT BUỘC]` (kèm link lấy key),
`[TỰ SINH]` (kèm lệnh sinh giá trị) hoặc `[TÙY CHỌN]` (để trống dùng mặc định) — đọc
comment ngay phía trên từng biến trước khi điền.

**Kiểm tra `.env` có khớp danh sách biến với `.env.example` không** (chỉ so **tên** biến,
không in giá trị — an toàn để chạy/dán kết quả ra ngoài):

```bash
diff <(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' .env.example | sort) \
     <(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' .env | sort)
```

Không in gì ra nghĩa là khớp hoàn toàn. Dòng bắt đầu bằng `<` là biến có trong
`.env.example` nhưng thiếu trong `.env`; dòng bắt đầu bằng `>` là biến thừa trong `.env`
so với mẫu. Muốn kiểm tra cặp `deploy/.env`/`deploy/.env.example`, đổi `file_path` của cả
2 chỗ `.env.example` và `.env` trong lệnh trên thành `deploy/.env.example` và `deploy/.env`.
