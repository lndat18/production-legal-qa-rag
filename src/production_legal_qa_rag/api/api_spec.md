# API — FastAPI (OpenAI-compatible) + OpenWebUI + Redis + Postgres

## 1. Mục tiêu & phạm vi

Đưa lõi RAG (`conversation/` → `generation/` → `retrieval/`) thành **chatbot pháp luật
phục vụ nhiều người dùng qua web**, theo hướng sát production nhưng gọn (dự án cá nhân,
public). Đây là spec tổng: mô tả kiến trúc toàn hệ thống, lớp HTTP, triển khai và thứ tự
implement 4 spec (`conversation/`, `cache/`, `chatlog/`, `api/`).

**Kiến trúc:**

```
Người dùng ─HTTPS─► Cloudflare Tunnel ─► OpenWebUI (UI, đăng nhập, lịch sử) ──┐
                                   │ Postgres (DB `openwebui`)      │ /v1/chat/completions
                                   ▼                                 ▼   (mạng nội bộ, Bearer key)
                              [Postgres]◄── chatlog ──────  FastAPI `api/`
                                                               │ ChatOrchestrator (conversation/)
                              [Redis] ◄── cache, quota, rl ────┤   guardrail ‖ condense → cache
                                                               │   → admission → retrieve → generate
                       Groq · HF · Pinecone · Reranker ◄───────┘
```

**Trong phạm vi:**

- Endpoint tương thích OpenAI (`POST /v1/chat/completions` stream và không stream,
  `GET /v1/models`) để OpenWebUI coi backend như một model.
- Xác thực OpenWebUI ↔ backend, nhận danh tính user từ header, rate limit theo user.
- Chuyển `GenerationEvent` sang định dạng OpenAI (mục 6).
- Health/readiness, lifespan (khởi tạo client 1 lần), logging có `request_id`.
- Triển khai bằng docker compose (mục 8) và cấu hình OpenWebUI (mục 9).

**Không làm:**

- **Không Kafka** (nút thắt là hạn mức LLM, không phải truyền tin; Kafka không giúp chat
  stream). Chỉ đáng cân nhắc cho pipeline audit/analytics phía sau, ngoài phạm vi.
- Không tự viết UI (OpenWebUI); không auth/quản lý user riêng (OpenWebUI lo).
- Không tự host LLM (vLLM); LLM vẫn là Groq qua config.
- Không Prometheus/Grafana/Langfuse/tracing (làm ở phase cuối, sau RAGAS: tracing,
  tracking & CI — xem hình `docs/images/architecture.png`), không autoscale/Kubernetes, không nhiều worker (bản
  đầu 1 worker, mục 8), không WebSocket, không API riêng cho client khác OpenWebUI.

**Tiêu chí quan trọng nhất:** người dùng đăng nhập OpenWebUI, hỏi nhiều lượt, thấy câu trả
lời stream kèm nguồn; nhiều người dùng đồng thời không làm sập hệ thống hay cạn hạn mức
Groq (bị giới hạn/xếp hàng rõ ràng thay vì lỗi ngẫu nhiên); dữ liệu chỉ đi qua backend
qua mạng nội bộ có xác thực.

## 2. Input & Output (endpoint)

| Endpoint                     | Mô tả                                                                 |
| ---------------------------- | --------------------------------------------------------------------- |
| `POST /v1/chat/completions`  | Body OpenAI: `model`, `messages`, `stream`, `stream_options`; các tham số khác (temperature…) **bị bỏ qua** |
| `GET /v1/models`             | Trả đúng 1 model: `{"id": "legal-qa", "object": "model", "owned_by": "production-legal-qa-rag"}` |
| `GET /healthz`               | Liveness: process sống → 200                                          |
| `GET /readyz`                | Readiness: ping Redis + Postgres; lỗi → 503 (không ping Groq/Pinecone) |

Schema request/response là pydantic v2 (`api/schemas.py`). Response stream là
`text/event-stream`: mỗi chunk `data: {chat.completion.chunk JSON}`; chunk đầu có
`delta.role="assistant"`; chunk cuối có `finish_reason` (`stop`, hoặc `length` khi có
`warning(truncated)`); tuỳ chọn chunk `usage` khi `stream_options.include_usage`; kết thúc
`data: [DONE]`. Không stream: gom toàn bộ event thành 1 `chat.completion`.

Lỗi **trước khi stream bắt đầu** trả HTTP chuẩn với body OpenAI
`{"error": {"message", "type", "code"}}`: 401 (sai/thiếu key), 422 (`messages` không hợp
lệ: `InvalidConversationError`), 429 (rate limit theo phút, kèm `Retry-After`), 503.
Lỗi **sau khi stream bắt đầu** đi qua nội dung câu trả lời (mục 6), HTTP vẫn 200.

## 3. Công cụ

| Việc            | Công cụ                                                          |
| --------------- | ---------------------------------------------------------------- |
| Web framework   | `fastapi` + `uvicorn[standard]`                                  |
| SSE             | `StreamingResponse` (`text/event-stream`), không thêm thư viện   |
| Redis           | `redis` (asyncio) — rate limit, cache, quota                     |
| Postgres        | `sqlalchemy[asyncio]` + `asyncpg` (chatlog)                      |
| UI              | Open WebUI (image Docker, **ghim phiên bản**)                    |
| Truy cập public | Cloudflare Tunnel (`cloudflared`): HTTPS do Cloudflare lo, không mở cổng |
| Đóng gói        | Docker + docker compose, `uv` trong Dockerfile                   |

Dependency mới (`pyproject.toml`): `fastapi`, `uvicorn[standard]`, `redis`,
`sqlalchemy[asyncio]`, `asyncpg`, `alembic`. Kiểm tra lại bằng `pip-audit`.

## 4. Xác thực & danh tính (`auth.py`)

- OpenWebUI gọi backend với `Authorization: Bearer <CHATBOT_API_KEY>`. Backend so sánh bằng
  `secrets.compare_digest`; sai/thiếu → 401. Key chỉ nằm trong `.env` của compose, đưa cho
  OpenWebUI qua biến môi trường.
- **Backend không expose ra internet**, chỉ nằm trong mạng compose; người dùng chỉ thấy
  OpenWebUI. Key là lớp phòng thủ thứ hai, không thay thế cô lập mạng.
- Danh tính người dùng: bật `ENABLE_FORWARD_USER_INFO_HEADERS=true` ở OpenWebUI → backend
  đọc `X-OpenWebUI-User-Id` và `X-OpenWebUI-Chat-Id` (chỉ tin khi Bearer key hợp lệ).
  **Chỉ dùng id, không đọc/lưu email/tên.** Thiếu `User-Id` → dùng `"anonymous"` (rate
  limit chung cho nhóm này). Tên header xác nhận theo phiên bản OpenWebUI đã ghim.
- Kết quả: `RequestContext(user_id, chat_id, request_id)` (`request_id` = UUID sinh mỗi
  request, đưa vào log và header phản hồi `X-Request-Id`).

## 5. Rate limit theo phút (`rate_limit.py`)

Chặn ở tầng HTTP, **trước khi** stream, cho **mọi** request (kể cả cache hit) để chống
lạm dụng:

- Cửa sổ cố định qua Redis: `INCR rl:{user_id}:{phút}` + `EXPIRE 70`. Vượt
  `RATE_LIMIT_PER_MINUTE` (mặc định 5) → 429 + `Retry-After`.
- Quota theo **ngày** và **hàng đợi đồng thời** nằm ở `AdmissionController` của
  `conversation/` (chỉ tính khi thực sự gọi LLM, cache hit không tốn quota).
- Redis lỗi → **fail-open** (log warning): hàng phòng thủ cuối là semaphore in-process của
  admission.

## 6. Chuyển event → OpenAI (`openai_format.py`)

OpenAI chỉ có `delta.content` (và `reasoning_content` mà OpenWebUI hiển thị dạng "đang
suy nghĩ"), nên:

| `GenerationEvent` | Chuyển thành                                                                                   |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| `status`          | `delta.reasoning_content` một dòng: "Đang kiểm tra câu hỏi…" / "Đang tra cứu văn bản pháp luật…" / "Đang soạn câu trả lời…" (không vào `content`, không lưu vào câu trả lời) |
| `token`           | `delta.content = text`                                                                         |
| `refusal`         | `delta.content = message`, kết thúc `stop`                                                     |
| `citations`       | Sau token cuối, nối khối nguồn: `SOURCES_FOOTER_MARKER` (`"\n\n---\n**Nguồn**\n"`, hằng số ở `conversation/history.py`) rồi mỗi dòng `[n] {breadcrumb} — {source_document}` |
| `warning`         | Nối dưới khối nguồn: `⚠️ {message}`; `truncated` → `finish_reason="length"`                    |
| `error`           | `delta.content = message` (nếu đã có chữ thì xuống dòng trước, thêm `⚠️`); HTTP vẫn 200; `rate_limited` kèm "thử lại sau N giây" |
| `done`            | Chunk kết thúc (+ `usage` nếu được yêu cầu) rồi `data: [DONE]`                                  |

- **Disclaimer thời điểm dữ liệu (kế hoạch, chưa implement — `api/` chưa có code):** nối
  `DATA_SNAPSHOT_DISCLAIMER` (hằng số ở `conversation/history.py`, cùng chỗ với
  `SOURCES_FOOTER_MARKER`) vào `delta.content` đúng 1 lần, ở phần cuối cùng của luồng
  trả lời — sau khối nguồn (`citations`), và sau cả khối `warning` nếu có (luôn là phần
  đuôi cuối cùng trước `done`/`[DONE]`) — xem `conversation_spec.md` mục 19.3.2.
- **Keep-alive:** khi chờ hàng đợi admission hoặc reasoning lâu, gửi 1 dòng comment SSE
  (`: keep-alive`) mỗi `KEEPALIVE_SECONDS = 15` để proxy/trình duyệt không cắt kết nối.
  Nếu OpenWebUI không chấp nhận comment thì thay bằng chunk `delta` rỗng — xác nhận khi
  nghiệm thu.
- **Việc ẩn** `status` phụ thuộc OpenWebUI hỗ trợ `reasoning_content`; nếu không, bỏ `status`
  (mất "đang tra cứu…" nhưng không hỏng).
- Client ngắt kết nối: Starlette huỷ generator; `finally` ghi `chatlog` với
  `outcome="client_disconnected"`.

## 7. Workflow & lifecycle (`app.py`, `routes.py`)

`create_app()` (dùng `uvicorn --factory`), khởi tạo **1 lần** trong `lifespan`, đóng khi tắt:

1. `Settings` (mục 10), engine Postgres, `Redis`, `corpus_version`.
2. `InputGuardrail`, `AnswerGenerator`, `QueryCondenser`, `RetrievalPipeline`
   (`GenerationPipeline` gom guardrail + generator), `AnswerCache`/`RetrievalCache`/
   `SingleFlight`, `AdmissionController`, `ChatLogRepository`, rồi `ChatOrchestrator`.
   Lưu vào `app.state`; route lấy qua dependency.

Route `POST /v1/chat/completions`:

```
1. xác thực (mục 4) → RequestContext; rate limit phút (mục 5)
2. messages hợp lệ? (build_window, InvalidConversationError -> 422)
3. trace = TurnTrace(...); events = orchestrator.stream(messages, ctx, trace)
4. stream=true : StreamingResponse(openai_format(events)) ; finally -> ghi chatlog
   stream=false: gom events -> 1 JSON ; finally -> ghi chatlog
```

**Audit event loop (đã đo 2026-09-21):** đoạn CPU đồng bộ chạy trong process FastAPI sẽ
chặn event loop và làm đứng mọi người dùng. Trong retrieval chỉ có `ViTokenizer.tokenize`
(pyvi) gọi đồng bộ ở `query_embedder.embed` và `BM25Encoder.encode_query`; đo thực tế
~0.3 ms (câu hỏi 140 ký tự) đến ~2.3 ms (đoạn HyDE 840 ký tự) mỗi lần, cỡ vài ms mỗi câu
hỏi → **không cần `asyncio.to_thread` ở bản đầu**. Quy tắc giữ lại: mọi đoạn CPU đồng bộ
mới thêm vào đường xử lý mà đo > ~10 ms phải bọc `asyncio.to_thread`. Pinecone/HF đã
chạy qua `to_thread`; reranker là server riêng.

## 8. Triển khai (`deploy/`, ngoài package)

Toàn bộ đóng gói và vận hành nằm ở **`deploy/deploy_spec.md`** (spec riêng, đặt cạnh
`deploy/` ở root repo vì đó không phải package import được): Dockerfile, docker compose
(`cloudflared`, `open-webui`, `api`, `redis`, `postgres`), Cloudflare Tunnel, bí mật,
backup. **Hosting đã chốt (2026-09-21): chạy trên máy cá nhân + Cloudflare Tunnel**, không
VPS, không Caddy.

Ràng buộc từ phía `api/` mà deploy phải tôn trọng:

- **1 worker** ở bản đầu vì semaphore của admission là in-process
  (`conversation_spec.md` mục 8); lệnh chạy:
  `uvicorn production_legal_qa_rag.api.app:create_app --factory --host 0.0.0.0 --port 8000
  --workers 1`. Muốn nhiều worker/replica: chuyển semaphore sang Redis trước.
- `api` không publish cổng; chỉ `open-webui` gọi tới qua mạng nội bộ compose.
- Healthcheck của `api` là `GET /readyz` (mục 2).

## 9. Cấu hình OpenWebUI

Biến môi trường. Đã đối chiếu docs/GitHub OpenWebUI (2026-09-21): `ENABLE_FORWARD_USER_INFO_HEADERS`
(gửi `X-OpenWebUI-User-Id/-Name/-Email/-Role` và `X-OpenWebUI-Chat-Id`), `ENABLE_TITLE_GENERATION`,
`ENABLE_FOLLOW_UP_GENERATION`, `ENABLE_AUTOCOMPLETE_GENERATION` (mặc định False), `ENABLE_SIGNUP`,
`DEFAULT_USER_ROLE`, `OPENAI_API_BASE_URL`, `OPENAI_API_KEY`, `ENABLE_OLLAMA_API`, `WEBUI_URL`;
`delta.reasoning_content` được hiển thị dạng khối "Thinking". **Chưa xác nhận được** qua docs:
`ENABLE_TAGS_GENERATION`, `ENABLE_RETRIEVAL_QUERY_GENERATION`, `ENABLE_SEARCH_QUERY_GENERATION`,
`DATABASE_URL`, `WEBUI_SECRET_KEY` — kiểm tra lại với phiên bản đã ghim.
**Bẫy PersistentConfig:** nhiều cài đặt (vd. `ENABLE_FOLLOW_UP_GENERATION`) chỉ đọc từ env ở
lần khởi động đầu, sau đó nằm trong DB và chỉnh qua Admin Panel; đặt đúng ngay lần chạy đầu
hoặc bật `ENABLE_PERSISTENT_CONFIG=false` (xác nhận biến này) để compose luôn là nguồn chuẩn.

- **Kết nối backend:** `OPENAI_API_BASE_URL=http://api:8000/v1`,
  `OPENAI_API_KEY=<CHATBOT_API_KEY>`; tắt Ollama (`ENABLE_OLLAMA_API=false`); chỉ 1 model
  `legal-qa` (không cho người dùng chọn model khác).
- **Tắt các lời gọi phụ tới model** (nếu để mặc định chúng vào pipeline, bị guardrail chặn
  và đốt hạn mức): `ENABLE_TITLE_GENERATION`, `ENABLE_TAGS_GENERATION`,
  `ENABLE_FOLLOW_UP_GENERATION`, `ENABLE_AUTOCOMPLETE_GENERATION`,
  `ENABLE_RETRIEVAL_QUERY_GENERATION`, `ENABLE_SEARCH_QUERY_GENERATION` = `false`. Tiêu đề
  chat dùng cắt câu hỏi đầu, không cần LLM.
- **Tắt tính năng thừa:** RAG/tải tệp của OpenWebUI, web search, code execution, image
  generation, tool/function của người dùng.
- **Danh tính:** `ENABLE_FORWARD_USER_INFO_HEADERS=true` (mục 4).
- **Đăng ký:** mở hẳn (`ENABLE_SIGNUP=true`, `DEFAULT_USER_ROLE=user`) — đã chốt
  2026-09-21; hạn mức được bảo vệ bằng quota/ngân sách (`deploy_spec.md` mục 5). Tài khoản
  đầu tiên là admin: đăng ký trước khi công bố URL.
- **Lưu trữ:** `DATABASE_URL=postgresql://…/openwebui` (thay SQLite mặc định), `WEBUI_SECRET_KEY`
  cố định.
- **Thông báo người dùng:** banner "Hội thoại được lưu để cải thiện dịch vụ. Nội dung chỉ
  mang tính tham khảo, không thay thế tư vấn pháp lý." Có thể thêm system prompt mặc định
  cho UI, nhưng **backend bỏ qua `system` message từ client**.
- Kiểm tra giấy phép OpenWebUI nếu đổi thương hiệu/quy mô lớn (bản mới có điều khoản
  branding).

## 10. Config (`config.py`)

Thêm (cùng pattern `pydantic-settings`):

- `ApiSettings`: `chatbot_api_key` (`CHATBOT_API_KEY`, `SecretStr`), `rate_limit_per_minute
  = 5`, `keepalive_seconds = 15`.
- `RedisSettings`: `redis_url` (`REDIS_URL`) — dùng chung với `AdmissionSettings`
  (`conversation_spec.md` mục 10) và `cache/`; gộp thành 1 class nếu trùng.
- `DatabaseSettings`: `database_url` (`CHATLOG_DATABASE_URL`) — `chatlog_spec.md` mục 6.

Module `api/` không đọc `.env` trực tiếp. Cập nhật `.env.example` (mọi biến mới).

## 11. Module (`src/production_legal_qa_rag/api/`)

| Module             | Trách nhiệm                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| `schemas.py`       | Request/response OpenAI (chat completion, chunk, models, error)          |
| `auth.py`          | Xác thực Bearer, dựng `RequestContext` từ header                         |
| `rate_limit.py`    | Rate limit theo phút (Redis, fail-open)                                  |
| `openai_format.py` | `GenerationEvent` → chunk OpenAI; khối nguồn; keep-alive; gom cho non-stream |
| `routes.py`        | Route chat/models/health; ghi chatlog trong `finally`                    |
| `app.py`           | `create_app`, `lifespan`, exception handler trả lỗi dạng OpenAI          |

## 12. Xử lý lỗi

| Tình huống                              | Xử lý                                                       |
| --------------------------------------- | ----------------------------------------------------------- |
| Sai/thiếu Bearer key                    | 401 (OpenAI error body)                                     |
| `messages` không hợp lệ                 | 422                                                         |
| Vượt rate limit theo phút               | 429 + `Retry-After`                                         |
| Redis/Postgres chết                     | `/readyz` 503; chat vẫn chạy degrade (cache/quota/log bỏ qua) |
| Quota ngày / quá tải / lỗi LLM          | Nằm trong luồng stream (mục 6), HTTP 200                    |
| Ngoại lệ không lường trước trong route  | Bắt ở `app.py`: 500 body OpenAI; stream đã bắt đầu thì phát nội dung lỗi chung rồi `[DONE]`; log kèm `request_id`, không log nội dung |

## 13. Nghiệm thu thủ công

1. `docker compose up -d` → mở URL Cloudflare Tunnel (hoặc `127.0.0.1:3000`), đăng ký,
   đăng nhập, hỏi câu hỏi luật: chữ stream ra, có "Đang tra cứu…" dạng suy nghĩ, có khối
   Nguồn. (Nghiệm thu hạ tầng riêng: `deploy_spec.md` mục 9.)
2. Hội thoại nhiều lượt: follow-up ngắn trả đúng chủ đề (điều kiện nghiệm thu của
   `conversation_spec.md` mục 13).
3. Không có lời gọi phụ nào từ OpenWebUI vào `/v1/chat/completions` (tạo chat mới không
   sinh request tiêu đề/tags): kiểm tra qua `chatlog`.
4. `curl` thẳng vào `api` từ ngoài mạng compose → không tới được; từ trong mạng thiếu key →
   401.
5. Tải: ~20 người dùng đồng thời (script) → không có request nào treo, phần vượt hạn mức
   nhận thông báo rõ ràng; event loop không bị chặn (độ trễ `/healthz` vẫn thấp khi đang
   tải).
6. Client đóng tab giữa chừng → không rò slot admission/khoá; `chatlog` ghi
   `client_disconnected`.
7. Tắt Redis rồi Postgres → chatbot vẫn trả lời, `/readyz` 503, log warning.

## 14. Thứ tự implement & điểm mở

**Thứ tự (mỗi bước một vòng develop-cycle):**

1. `generation/` — thay đổi nhỏ theo `generation_spec.md` mục 16 (`generate` public,
   guardrail có ngữ cảnh, `PROMPT_VERSION`, mã lỗi `quota_exceeded`).
2. `conversation/` — history, condense, orchestrator (chưa cache/admission — inject
   `None`). Không làm `run_for_evaluation` (phase RAGAS sau). Nghiệm thu multi-turn ngay
   ở bước này qua script.
3. `cache/` + `AdmissionController` (Redis).
4. `chatlog/` + Alembic.
5. `api/` + `deploy/` + cấu hình OpenWebUI, nghiệm thu mục 13.

**Điểm mở (cần chốt trước/khi bước 5):**

1. **Đã chốt (2026-09-21):** hosting trên máy cá nhân + Cloudflare Tunnel (quick tunnel
   trước, named tunnel khi có domain) — `deploy/deploy_spec.md`.
2. **Đã chốt (2026-09-21):** đăng ký OpenWebUI mở hẳn (mục 9).
3. **Đã chốt (2026-09-21):** giữ reranker trên LightningAI (GPU free) qua ngrok, không
   đổi chỗ host. Chấp nhận rủi ro tunnel/Studio sleep: khi reranker lỗi, retrieval degrade
   (xen kẽ các nhánh, `rerank_score=None`, `retrieval_spec.md`) nên chatbot vẫn trả lời
   nhưng chất lượng xếp hạng giảm. `/readyz` không ping reranker; cần theo dõi tỉ lệ
   degrade qua `chatlog`.
4. Hạn mức Groq free (30 RPM, 1K RPD, 8K TPM, **200K TPD** mỗi model/org) chỉ đủ khoảng
   50–60 câu cache-miss/ngày cho generation; quota mặc định ở `AdmissionSettings`
   (user 5/ngày, toàn cục 50/ngày) suy ra từ đó, chỉnh sau khi có số liệu `usage` từ
   `chatlog`; cần nhiều hơn thì nâng cấp gói Groq trả phí (không cần đổi code).

## 15. Rủi ro

1. OpenWebUI đổi hành vi giữa các phiên bản (tên biến, header, `reasoning_content`) →
   ghim phiên bản, ghi lại phiên bản đã nghiệm thu.
2. `messages[]` do client cung cấp (kiểu A) không đáng tin: xem `conversation_spec.md`
   mục 3, 14.
3. Ngân sách Groq free không đủ cho public: cache + quota giảm nhẹ, không xoá hẳn rủi ro.
4. Ảnh Docker nặng vì `transformers`/`pyvi` (tokenizer): nếu retrieval không cần, cân nhắc
   tách dependency (ngoài phạm vi spec này).
5. Một máy, một worker: không có HA, máy tắt là dịch vụ tắt; chấp nhận cho dự án cá nhân,
   đường nâng cấp đã nêu (Redis semaphore → nhiều replica, VPS thật).
