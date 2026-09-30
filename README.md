# Production-ready Legal QA RAG system for Vietnamese legal documents

## Cấu hình biến môi trường (`.env`)

**Đúng 1 cặp file `.env`/`.env.example` cho toàn bộ project, ở repo root** (chốt
2026-09-29 — trước đây có thêm `deploy/.env` riêng, nay gộp lại để dễ kiểm soát). File này
dùng cho cả 3 việc:

- Chạy code Python trực tiếp trên host (`uv run pytest`, `tools/`, `api` không qua Docker).
- Chạy toàn bộ stack production qua Docker Compose (`deploy/docker-compose.yml`,
  `./deploy/up.sh`) — container `api` đọc nguyên file `.env` này qua `env_file: ../.env`.
- Chạy stack Langfuse/Prometheus/Grafana (`observability/`).

`cp .env.example .env` rồi điền. `.env.example` chia 3 block bằng comment — **APP** (biến
app Python, dùng chung cho cả dev-trên-host lẫn container `api`), **DEPLOY** (chỉ container
production dùng — 2 biến `POSTGRES_USER`/`POSTGRES_PASSWORD` có prefix `DEPLOY_` vì trùng
tên với Postgres riêng của block dưới), **OBSERVABILITY** (chỉ stack Langfuse/Prometheus/
Grafana dev — mọi biến có prefix `OBS_`). Không commit `.env` (chứa key thật) — đã nằm
trong `.gitignore`.

Chạy `api`/tools trên host cần Redis cục bộ (`REDIS_URL` mặc định trỏ `localhost`) — tự dựng, repo không kèm compose cho việc này.

### Key dịch vụ ngoài `[BẮT BUỘC]`

| Biến                                                | Lấy ở đâu                                                                                    | Ghi chú                                                                                                      |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `GROQ_API_KEY_1` … `_9`                             | [console.groq.com/keys](https://console.groq.com/keys)                                       | Xem bảng phân vai key Groq bên dưới                                                                          |
| `HF_TOKEN`                                          | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)                     | Token quyền "Read" là đủ; dùng để embed văn bản                                                              |
| `PINECONE_API_KEY`                                  | [app.pinecone.io](https://app.pinecone.io) → "API Keys"                                      | —                                                                                                            |
| `PINECONE_INDEX_NAME`, `PINECONE_SPARSE_INDEX_NAME` | Tên index tạo trên Pinecone dashboard                                                        | PHẢI khớp đúng tên thật. Sparse index (BM25) tạo + build bằng `tools/sparse_index_documents.py`              |

### Key Groq: phân vai theo tài khoản

Groq giới hạn rate limit (TPM/RPM/TPD) theo **(tài khoản, model)**, không theo từng API key —
tạo thêm key từ cùng 1 tài khoản không có thêm ngân sách. Muốn thêm ngân sách, lấy key từ
tài khoản Groq khác (email khác, đăng ký riêng). Chỉ `GROQ_API_KEY_1` là bắt buộc cho production; key nào không
set thì bước tương ứng dùng lại `GROQ_API_KEY_1`.

| Biến             | Nhóm | Bước dùng                                                                                        | Không set thì |
| ---------------- | ---- | ------------------------------------------------------------------------------------------------ | ------------- |
| `GROQ_API_KEY_1` | Nhẹ  | Guardrail, condense, HyDE (model 20b)                                                            | Bắt buộc      |
| `GROQ_API_KEY_2` | Nhẹ  | Evidence Judge (model 20b)                                                                       | Dùng key 1    |
| `GROQ_API_KEY_3` | Nặng | Generation (model 120b), xoay vòng luân phiên với key 4 theo từng lượt draft/repair              | Dùng key 1    |
| `GROQ_API_KEY_4` | Nặng | Generation, xoay vòng với key 3 để giãn TPD (nút thắt nhất pipeline) ra 2 tài khoản              | Không xoay    |

Ngoại lệ: `formatting/` (offline) dùng `GROQ_API_KEY_1` (+ `GROQ_API_KEY_2` nếu có, chạy 2 worker);
`evaluation/` (RAGAS, `tools/generate_testset.py`) bắt buộc đủ 9 key `GROQ_API_KEY_1` … `_9` (9 tài khoản độc lập; production không dùng `_5`–`_9`) — thiếu
key nào, pydantic báo lỗi ngay lúc khởi tạo. Thiết kế đầy đủ:
`src/production_legal_qa_rag/conversation/conversation_spec.md` mục 12.1.

### Biến tự sinh `[TỰ SINH]`

`CHATBOT_API_KEY` là secret riêng giữa OpenWebUI và backend `api` (không phải key của dịch vụ
ngoài). Sinh bằng:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Biến tùy chọn `[TÙY CHỌN]`

Để trống là dùng mặc định.

| Biến                                                         | Mặc định                                                       | Ghi chú                                                                                                       |
| ------------------------------------------------------------ | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `REDIS_URL`                                                  | `redis://localhost:6379/0`                                     | Cache câu trả lời/retrieval và rate limit API. Redis dev chạy không mật khẩu, chỉ nghe localhost              |
| `CACHE_CORPUS_VERSION`                                       | Tự tính từ `data/bm25/bm25_params.json`                        | Hiếm khi cần; muốn ghi đè thì bỏ dấu `#` ở dòng trong `.env.example`                                          |
| `RERANKER_MODEL_NAME`, `RERANKER_MAX_LENGTH`, `RERANKER_BATCH_SIZE` | `AITeamVN/Vietnamese_Reranker`, `512`, `16`             | Reranker chạy in-process                                                                                      |
| `RATE_LIMIT_PER_MINUTE`, `KEEPALIVE_SECONDS`                 | `5`, `15`                                                      | Request/phút/user; giây giữa các keep-alive SSE khi đang chờ LLM                                              |

### Kiểm tra `.env` khớp `.env.example`

Chỉ so **tên** biến, không in giá trị — an toàn để chạy/dán kết quả ra ngoài:

```bash
diff <(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' .env.example | sort) \
     <(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' .env | sort)
```

Không in gì ra nghĩa là khớp hoàn toàn. Dòng bắt đầu bằng `<` là biến có trong `.env.example`
nhưng thiếu trong `.env`; dòng bắt đầu bằng `>` là biến thừa trong `.env` so với mẫu.

## Deploy Docker (`.env`)

Chạy toàn bộ stack (api, OpenWebUI, Redis, Postgres, Cloudflare Tunnel):

```bash
./deploy/up.sh
```

Lần đầu chưa có `.env` ở root: script tự tạo từ `.env.example` rồi dừng để bạn điền giá trị
thật (biến DEPLOY_* + các biến dùng chung đã điền ở phần trên nếu chạy code trên host trước
đó). Điền xong chạy lại `./deploy/up.sh`. Chi tiết: `deploy/deploy_spec.md` mục 7.

**Chế độ tunnel (`COMPOSE_PROFILES`):**

| Giá trị           | Điều kiện                                       | Kết quả                                                                                             |
| ----------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `quick` (mặc định) | Không cần domain, miễn phí                     | URL ngẫu nhiên `*.trycloudflare.com`, chỉ đổi khi container `cloudflared-quick` restart              |
| `named`           | Domain riêng đã trỏ về Cloudflare               | URL cố định. Dựng named tunnel theo `deploy_spec.md` mục 3 trước, rồi điền `TUNNEL_TOKEN`, `WEBUI_URL` |

**Biến `[BẮT BUỘC]` lấy từ dịch vụ ngoài:** `GROQ_API_KEY_*`, `HF_TOKEN`, `PINECONE_*` — đã
điền ở block APP phía trên, container `api` đọc chung, không cần điền lại.

**Biến `[TỰ SINH]`, riêng cho block DEPLOY** (sinh bằng
`python3 -c "import secrets; print(secrets.token_urlsafe(32))"` cho các secret dạng chuỗi):

| Biến                                                | Ghi chú                                                                                                     |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `REDIS_PASSWORD`                                     | Mật khẩu cho Redis container production do compose tạo mới                                                  |
| `DEPLOY_POSTGRES_USER`, `DEPLOY_POSTGRES_PASSWORD`  | Tự đặt tuỳ ý; compose tạo Postgres mới, chỉ phục vụ OpenWebUI (database `openwebui`). Prefix `DEPLOY_` vì Postgres của stack observability (dưới) dùng cùng tên biến gốc |
| `WEBUI_SECRET_KEY`                                   | Key ký session OpenWebUI. Điền 1 lần rồi giữ nguyên — đổi sau khi đã có người đăng nhập sẽ đăng xuất tất cả |

`docker-compose.yml` tự dựng `REDIS_URL` thật cho container `api` từ
`REDIS_PASSWORD` — không cần tự điền.

**`[TÙY CHỌN]`:** `WEBUI_URL` (URL cố định của named tunnel, vd. `https://chat.tenban.com`) và
`TUNNEL_TOKEN` (lấy từ Cloudflare Zero Trust dashboard) chỉ cần với named tunnel; quick tunnel
để trống.

## Observability (Langfuse + Prometheus + Grafana)

Chạy cạnh production trên cùng máy (`observability_spec.md`), dùng chung `.env` root (block `OBS_*`).

```bash
./observability/up.sh     # bật (giữ dữ liệu, chạy lại an toàn)
./observability/down.sh   # tắt, giữ volume
```

**Lần đầu:** mở `http://localhost:3001`, đăng ký tài khoản Langfuse local, tạo project, lấy
Public/Secret key rồi điền `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` vào `.env` (block
APP). Grafana: `http://localhost:3002` (`OBS_GRAFANA_ADMIN_USER`/`OBS_GRAFANA_ADMIN_PASSWORD`).
Prometheus: `http://localhost:9092`.

**Trace/track production (traffic end-user thật):** bật stack trên trước, điền key Langfuse,
rồi chạy `./deploy/up.sh` — script thấy network `legal-qa-observe` thì tự nối `api` vào
(`deploy/docker-compose.observe.yml`), trace gắn `environment=production` trong Langfuse,
Prometheus scrape `api:8000` (label `env=production`). Không bật stack observe thì
`up.sh` bỏ qua, production chạy như thường.
