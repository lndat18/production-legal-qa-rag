# Cấu hình và triển khai

- Dùng chung một file `.env` ở repo root cho APP, DEPLOY và OBSERVABILITY.
- Chạy các lệnh từ repo root. Không commit `.env`.
- Hướng dẫn tổng quan: [README](../README.md).

## 1. Chuẩn bị `.env`

1. Tạo file nếu chưa có: `cp .env.example .env`.
2. Điền key dịch vụ trong block APP:
   - `GROQ_API_KEY_1`: lấy tại [Groq](https://console.groq.com/keys); production bắt buộc key này.
   - `HF_TOKEN`: lấy tại [Hugging Face](https://huggingface.co/settings/tokens), cần quyền inference cho embedding.
   - `PINECONE_API_KEY`: lấy tại [Pinecone](https://app.pinecone.io).
   - `PINECONE_INDEX_NAME`, `PINECONE_SPARSE_INDEX_NAME`: tên index dense/sparse; tạo bằng `tools/embed_documents.py` và `tools/sparse_index_documents.py`.
3. Sinh `CHATBOT_API_KEY` để OpenWebUI gọi backend. Chạy lại lệnh cho từng secret cần sinh:

   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

### Phân vai key Groq

- Key 1: guardrail, condense, HyDE.
- Key 2: Judge; thiếu thì dùng key 1.
- Key 3–4: generation xoay vòng; thiếu key 3 thì dùng key 1, thiếu key 4 thì không xoay.
- Formatting dùng key 1, thêm key 2 thì chạy hai worker.
- Evaluation cần đủ key 1–9; production không dùng key 5–9.
- Hạn mức tính theo **(tài khoản, model)**. Muốn tăng ngân sách, dùng key từ các tài khoản khác nhau.

### Biến tùy chọn của APP

- `REDIS_URL=redis://localhost:6379/0`: dùng khi chạy trên host; cần tự chạy Redis cục bộ.
- `CACHE_CORPUS_VERSION`: tự tính từ `data/bm25/bm25_params.json`; chỉ bỏ comment khi muốn ghi đè.
- `RERANKER_MODEL_NAME`, `RERANKER_MAX_LENGTH`, `RERANKER_BATCH_SIZE`: để trống dùng mặc định `AITeamVN/Vietnamese_Reranker`, `512`, `16`.
- `RATE_LIMIT_PER_MINUTE=5`: request/phút/user; `KEEPALIVE_SECONDS=15`: chu kỳ keep-alive SSE. Giữ giá trị mẫu hoặc bỏ dòng, không để trống.
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`: điền sau khi tạo project Langfuse; để trống thì không có trace.
- `LANGFUSE_BASE_URL=http://localhost:3001`: dành cho API trên host; Docker tự ghi đè khi nối stack observe.

## 2. Deploy production

1. Điền block DEPLOY:
   - `DEPLOY_POSTGRES_USER`: tự đặt; `DEPLOY_POSTGRES_PASSWORD`, `REDIS_PASSWORD`, `WEBUI_SECRET_KEY`: sinh riêng bằng lệnh ở trên.
   - Giữ `WEBUI_SECRET_KEY` ổn định; đổi key sẽ đăng xuất người dùng.
   - `COMPOSE_PROFILES=quick`: URL ngẫu nhiên `*.trycloudflare.com`; để trống `WEBUI_URL`, `TUNNEL_TOKEN`.
   - Muốn URL cố định: tạo named tunnel theo [deploy_spec.md](../deploy/deploy_spec.md) mục 3, đặt `COMPOSE_PROFILES=named`, điền `WEBUI_URL` và `TUNNEL_TOKEN`.
2. Nếu cần trace/metrics, hoàn thành mục 3 bên dưới trước. Sau đó bật production: `./deploy/up.sh`.
3. Khi cần tắt, chạy `./deploy/down.sh`.

- Container API dùng chung key trong APP; Compose tự dựng `REDIS_URL` từ `REDIS_PASSWORD`.
- Nếu chưa có `.env`, `up.sh` tạo từ mẫu rồi dừng để bạn điền; điền xong chạy lại.

## 3. Bật observability

1. Điền block OBSERVABILITY; có thể giữ nguyên user/database mẫu:
   - Sinh riêng từng mật khẩu `OBS_POSTGRES_PASSWORD`, `OBS_REDIS_AUTH`, `OBS_CLICKHOUSE_PASSWORD`, `OBS_MINIO_ROOT_PASSWORD`, `OBS_GRAFANA_ADMIN_PASSWORD` bằng `openssl rand -hex 16`.
   - Sinh riêng `OBS_SALT`, `OBS_ENCRYPTION_KEY`, `OBS_NEXTAUTH_SECRET` bằng `openssl rand -hex 32`; `OBS_ENCRYPTION_KEY` bắt buộc đúng 64 ký tự hex.
2. Bật stack: `./observability/up.sh`.
3. Mở `http://localhost:3001`, đăng ký tài khoản Langfuse, tạo project và API key; điền `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` vào block APP.
4. Chạy lại `./deploy/up.sh` để nối API vào stack observe và nhận cấu hình mới.
5. Hỏi thử chatbot, kiểm tra trace trong Langfuse và metrics:
   - Grafana: `http://localhost:3002`, đăng nhập bằng `OBS_GRAFANA_ADMIN_USER`/`OBS_GRAFANA_ADMIN_PASSWORD`.
   - Prometheus: `http://localhost:9092`.

- Chạy `./observability/down.sh` để tắt và giữ dữ liệu.
- Production vẫn chạy khi observe tắt, nhưng không lưu nhật ký lượt hỏi đáp.
- Chi tiết: [observability_spec.md](../src/production_legal_qa_rag/observability/observability_spec.md).

## Kiểm tra tên biến

- Lệnh dưới chỉ so tên biến, không in secret:

  ```bash
  diff <(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' .env.example | sort) \
       <(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' .env | sort)
  ```

- Không có output: khớp; `<`: thiếu trong `.env`; `>`: thừa so với mẫu.
