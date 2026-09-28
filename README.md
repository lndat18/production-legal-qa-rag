# Production-ready Legal QA RAG system for Vietnamese legal documents

## Cấu hình biến môi trường (`.env`)

Project có 2 cặp file `.env`/`.env.example` tách riêng, không dùng chung:

| Cặp file                              | Dùng khi                                                                                 | Cách tạo                                                         |
| ------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `.env` / `.env.example` (root)        | Chạy code Python trực tiếp trên host (`uv run pytest`, `tools/`, `api` không qua Docker) | `cp .env.example .env` rồi điền                                  |
| `deploy/.env` / `deploy/.env.example` | Chạy toàn bộ stack qua Docker Compose (`deploy_spec.md`)                                 | `./deploy/up.sh` — tự tạo, tự điền sẵn key trùng với root `.env` |

Không commit `.env` (chứa key thật) — đảm bảo `.env` và `deploy/.env` nằm trong `.gitignore`.

Chạy code trên host cần Redis + Postgres cục bộ:

```bash
docker compose -f deploy/dev/docker-compose.yml up -d
```

### Key dịch vụ ngoài `[BẮT BUỘC]`

| Biến                                                | Lấy ở đâu                                                                                    | Ghi chú                                                                                                      |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `GROQ_API_KEY` (và `_2`, `_3`, `_4`)                | [console.groq.com/keys](https://console.groq.com/keys)                                       | Xem bảng phân vai key Groq bên dưới                                                                          |
| `HF_TOKEN`                                          | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)                     | Token quyền "Read" là đủ; dùng để embed văn bản                                                              |
| `PINECONE_API_KEY`                                  | [app.pinecone.io](https://app.pinecone.io) → "API Keys"                                      | —                                                                                                            |
| `PINECONE_INDEX_NAME`, `PINECONE_SPARSE_INDEX_NAME` | Tên index tạo trên Pinecone dashboard                                                        | PHẢI khớp đúng tên thật. Sparse index (BM25) tạo + build bằng `tools/sparse_index_documents.py`              |

### Key Groq: phân vai theo tài khoản

Groq giới hạn rate limit (TPM/RPM/TPD) theo **(tài khoản, model)**, không theo từng API key —
tạo thêm key từ cùng 1 tài khoản không có thêm ngân sách. Muốn thêm ngân sách, lấy key từ
tài khoản Groq khác (email khác, đăng ký riêng). Chỉ `GROQ_API_KEY` là bắt buộc; key nào không
set thì bước tương ứng dùng lại `GROQ_API_KEY`.

| Biến             | Nhóm | Bước dùng                                                                                        | Không set thì |
| ---------------- | ---- | ------------------------------------------------------------------------------------------------ | ------------- |
| `GROQ_API_KEY`   | Nhẹ  | Guardrail, condense, HyDE (model 20b)                                                            | Bắt buộc      |
| `GROQ_API_KEY_2` | Nhẹ  | Evidence Judge (model 20b)                                                                       | Dùng key 1    |
| `GROQ_API_KEY_3` | Nặng | Generation (model 120b), xoay vòng luân phiên với key 4 theo từng lượt draft/repair              | Dùng key 1    |
| `GROQ_API_KEY_4` | Nặng | Generation, xoay vòng với key 3 để giãn TPD (nút thắt nhất pipeline) ra 2 tài khoản              | Không xoay    |

Ngoại lệ: `formatting/` (offline) dùng `GROQ_API_KEY` (+ `GROQ_API_KEY_2` nếu có, chạy 2 worker);
`evaluation/` (RAGAS, `tools/generate_testset.py`) bắt buộc cả `GROQ_API_KEY`, `_2`, `_3` — thiếu
key nào, pydantic báo lỗi ngay lúc khởi tạo. Thiết kế đầy đủ:
`src/production_legal_qa_rag/conversation/conversation_spec.md` mục 12.1.

### Biến tự sinh `[TỰ SINH]`

`CHATBOT_API_KEY` là secret riêng giữa OpenWebUI và backend `api` (không phải key của dịch vụ
ngoài). Sinh bằng:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Biến tùy chọn `[TÙY CHỌN]`

Giữ nguyên giá trị mặc định nếu chạy `deploy/dev/docker-compose.yml`; để trống là dùng mặc định.

| Biến                                                         | Mặc định                                                       | Ghi chú                                                                                                       |
| ------------------------------------------------------------ | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `REDIS_URL`                                                  | `redis://localhost:6379/0`                                     | Cache câu trả lời/retrieval và rate limit API. Redis dev chạy không mật khẩu, chỉ nghe localhost              |
| `CACHE_CORPUS_VERSION`                                       | Tự tính từ `data/bm25/bm25_params.json`                        | Hiếm khi cần; muốn ghi đè thì bỏ dấu `#` ở dòng trong `.env.example`                                          |
| `RERANKER_MODEL_NAME`, `RERANKER_MAX_LENGTH`, `RERANKER_BATCH_SIZE` | `AITeamVN/Vietnamese_Reranker`, `512`, `16`             | Reranker chạy in-process                                                                                      |
| `CHATLOG_DATABASE_URL`                                       | `postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot` | Khớp user/password của `deploy/dev/docker-compose.yml`. Driver PHẢI là `postgresql+asyncpg`                          |
| `CHATLOG_RETENTION_DAYS`                                     | `90`                                                           | Số ngày `tools/purge_chatlog.py` giữ chatlog                                                                  |
| `RATE_LIMIT_PER_MINUTE`, `KEEPALIVE_SECONDS`                 | `5`, `15`                                                      | Request/phút/user; giây giữa các keep-alive SSE khi đang chờ LLM                                              |

### Kiểm tra `.env` khớp `.env.example`

Chỉ so **tên** biến, không in giá trị — an toàn để chạy/dán kết quả ra ngoài:

```bash
diff <(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' .env.example | sort) \
     <(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' .env | sort)
```

Không in gì ra nghĩa là khớp hoàn toàn. Dòng bắt đầu bằng `<` là biến có trong `.env.example`
nhưng thiếu trong `.env`; dòng bắt đầu bằng `>` là biến thừa trong `.env` so với mẫu. Muốn
kiểm tra cặp `deploy/.env`/`deploy/.env.example`, đổi cả 2 đường dẫn trong lệnh thành
`deploy/.env.example` và `deploy/.env`.

## Deploy Docker (`deploy/.env`)

Chạy toàn bộ stack (api, OpenWebUI, Redis, Postgres, Cloudflare Tunnel):

```bash
./deploy/up.sh
```

Lần đầu chưa có `deploy/.env`: script tự tạo từ `deploy/.env.example`, tự điền sẵn các key trùng
với root `.env` (`GROQ_API_KEY*`, `HF_TOKEN`, `PINECONE_*`, `CHATBOT_API_KEY`) nếu root `.env` đã
có giá trị, rồi dừng để bạn điền nốt phần còn thiếu. Điền xong chạy lại `./deploy/up.sh`. File
`deploy/.env` là file riêng, không dùng chung với root `.env` — chi tiết:
`deploy/deploy_spec.md` mục 7.

**Chế độ tunnel (`COMPOSE_PROFILES`):**

| Giá trị           | Điều kiện                                       | Kết quả                                                                                             |
| ----------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `quick` (mặc định) | Không cần domain, miễn phí                     | URL ngẫu nhiên `*.trycloudflare.com`, chỉ đổi khi container `cloudflared-quick` restart              |
| `named`           | Domain riêng đã trỏ về Cloudflare               | URL cố định. Dựng named tunnel theo `deploy_spec.md` mục 3 trước, rồi điền `TUNNEL_TOKEN`, `WEBUI_URL` |

**Biến `[BẮT BUỘC]` lấy từ dịch vụ ngoài:** `GROQ_API_KEY*`, `HF_TOKEN`, `PINECONE_*` — cùng giá
trị và cùng cách phân vai key Groq với mục trên. `PINECONE_INDEX_NAME`/`PINECONE_SPARSE_INDEX_NAME`
phải khớp tên index thật.

**Biến `[TỰ SINH]`** (chỉ container dùng, không ai gõ tay): sinh bằng
`python3 -c "import secrets; print(secrets.token_urlsafe(32))"`.

| Biến                                | Ghi chú                                                                                                                  |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `CHATBOT_API_KEY`                   | Secret giữa OpenWebUI và backend `api`. Dùng lại giá trị của root `.env` được                                            |
| `REDIS_PASSWORD`                    | Mật khẩu cho Redis container do compose tạo mới                                                                          |
| `POSTGRES_USER`, `POSTGRES_PASSWORD` | Tự đặt tùy ý; compose tạo Postgres mới, dùng chung cho 2 database `openwebui` và `chatbot`                              |
| `WEBUI_SECRET_KEY`                  | Key ký session OpenWebUI. Điền 1 lần rồi giữ nguyên — đổi sau khi đã có người đăng nhập sẽ đăng xuất tất cả              |

`REDIS_URL` và `CHATLOG_DATABASE_URL` trong `deploy/.env.example` là placeholder dạng `${VAR}`:
giữ nguyên, đừng sửa hay xoá — `docker-compose.yml` tự dựng giá trị thật từ `REDIS_PASSWORD`,
`POSTGRES_USER`, `POSTGRES_PASSWORD`.

**`[TÙY CHỌN]`:** `WEBUI_URL` (URL cố định của named tunnel, vd. `https://chat.tenban.com`) và
`TUNNEL_TOKEN` (lấy từ Cloudflare Zero Trust dashboard) chỉ cần với named tunnel; quick tunnel
để trống.

Kiểm tra `deploy/.env` khớp mẫu: dùng lệnh `diff` ở mục trên với 2 đường dẫn
`deploy/.env.example` và `deploy/.env`.
