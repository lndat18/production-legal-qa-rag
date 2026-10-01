# API — FastAPI (OpenAI-compatible) + OpenWebUI + Redis (+ Postgres cho OpenWebUI)

> Giữ nguyên số mục vì code/spec khác tham chiếu.

## 1. Mục tiêu & phạm vi

Đưa lõi RAG (`conversation/` → `generation/` → `retrieval/`) thành **chatbot pháp luật phục vụ nhiều người dùng qua web**. Đây là spec tổng: kiến trúc toàn hệ thống, lớp HTTP, triển khai. Nhật ký từng lượt hỏi-đáp nằm ở Langfuse (`observability_spec.md` mục 4.5).

```
Người dùng ─HTTPS─► Cloudflare Tunnel ─► OpenWebUI (UI, đăng nhập, lịch sử) ─► Postgres (DB openwebui)
                                              │ /v1/chat/completions (mạng nội bộ, Bearer key)
                                              ▼
                                        FastAPI `api/` ──trace──► Langfuse
                                              │ ChatOrchestrator (conversation/): guardrail ‖ condense → cache
                                              │   → admission → retrieve → generate      [Redis: cache, quota, rl]
                                              ▼
                              Groq · HF · Pinecone · Reranker (in-process)
```

**Làm:** endpoint tương thích OpenAI (`POST /v1/chat/completions` stream/không stream, `GET /v1/models`) để OpenWebUI coi backend là một model; xác thực OpenWebUI ↔ backend, nhận danh tính user từ header, rate limit theo user; chuyển `GenerationEvent` sang định dạng OpenAI (mục 6); health/readiness, lifespan (khởi tạo client 1 lần), log có `request_id`; docker compose (mục 8) + cấu hình OpenWebUI (mục 9).

**Không làm:** Kafka; UI/auth/quản lý user riêng (OpenWebUI lo); tự host LLM; autoscale/Kubernetes, nhiều worker, WebSocket, API riêng cho client khác OpenWebUI.

**Tiêu chí số 1:** người dùng đăng nhập OpenWebUI, hỏi nhiều lượt, thấy câu trả lời stream kèm nguồn; nhiều người đồng thời không làm sập hệ thống hay cạn hạn mức Groq (bị giới hạn/xếp hàng rõ ràng); dữ liệu chỉ đi qua backend qua mạng nội bộ có xác thực.

## 2. Input & Output (endpoint)

| Endpoint | Mô tả |
| --- | --- |
| `POST /v1/chat/completions` | Body OpenAI: `model`, `messages`, `stream`, `stream_options`; tham số khác **bị bỏ qua** |
| `GET /v1/models` | Đúng 1 model: `{"id": "legal-qa", "object": "model", "owned_by": "production-legal-qa-rag"}` |
| `GET /healthz` | Liveness: process sống → 200 |
| `GET /readyz` | Readiness: ping Redis; lỗi → 503 (không ping Groq/Pinecone) |

Schema pydantic v2 (`api/schemas.py`). Stream `text/event-stream`: `data: {chat.completion.chunk}`; chunk đầu `delta.role="assistant"`; chunk cuối có `finish_reason` (`stop`, hoặc `length` khi `warning(truncated)`); tuỳ chọn chunk `usage` khi `stream_options.include_usage`; kết thúc `data: [DONE]`. Không stream: gom event thành 1 `chat.completion`. Lỗi **trước khi stream** trả HTTP chuẩn body OpenAI `{"error":{"message","type","code"}}`: 401 (sai/thiếu key), 422 (`InvalidConversationError`), 429 (rate limit phút, kèm `Retry-After`), 503. Lỗi **sau khi stream bắt đầu** đi qua nội dung câu trả lời (mục 6), HTTP vẫn 200.

## 3. Công cụ

`fastapi` + `uvicorn[standard]`; SSE bằng `StreamingResponse`; `redis` asyncio (rate limit, cache, quota); Langfuse cho trace; Open WebUI (image Docker, **ghim phiên bản**); Cloudflare Tunnel (`cloudflared`, không mở cổng); Docker + compose, `uv` trong Dockerfile. `api` không kết nối Postgres (chỉ OpenWebUI dùng). Kiểm bằng `pip-audit`.

## 4. Xác thực & danh tính (`auth.py`)

- OpenWebUI gọi backend với `Authorization: Bearer <CHATBOT_API_KEY>`, so bằng `secrets.compare_digest`; sai/thiếu → 401. Key chỉ nằm trong `.env` của compose.
- **Backend không expose ra internet**, chỉ nằm trong mạng compose; key là lớp phòng thủ thứ hai.
- Danh tính: `ENABLE_FORWARD_USER_INFO_HEADERS=true` → đọc `X-OpenWebUI-User-Id`, `X-OpenWebUI-Chat-Id` (chỉ tin khi Bearer hợp lệ). **Chỉ dùng id, không đọc/lưu email/tên.** Thiếu `User-Id` → `"anonymous"` (rate limit chung).
- Kết quả `RequestContext(user_id, chat_id, request_id)` (`request_id` = UUID mỗi request, vào log và header `X-Request-Id`).

## 5. Rate limit theo phút (`rate_limit.py`)

Chặn ở tầng HTTP, **trước khi stream**, cho **mọi** request (kể cả cache hit): cửa sổ cố định qua Redis `INCR rl:{user_id}:{phút}` + `EXPIRE 70`; vượt `RATE_LIMIT_PER_MINUTE` (mặc định 5) → 429 + `Retry-After`. Quota theo **ngày** và **hàng đợi đồng thời** nằm ở `AdmissionController` của `conversation/` (chỉ tính khi thật sự gọi LLM). Redis lỗi → **fail-open**; hàng phòng thủ cuối là semaphore in-process của admission.

## 6. Chuyển event → OpenAI (`openai_format.py`)

OpenAI chỉ có `delta.content` (và `reasoning_content` mà OpenWebUI hiển thị dạng "đang suy nghĩ"):

| `GenerationEvent` | Chuyển thành |
| --- | --- |
| `status` | `delta.reasoning_content` một dòng ("Đang kiểm tra câu hỏi…" / "Đang tra cứu văn bản pháp luật…" / "Đang soạn câu trả lời…"); không vào `content` |
| `token` | `delta.content = text` |
| `refusal` | `delta.content = message`, kết thúc `stop` |
| `citations` | Sau token cuối nối `SOURCES_FOOTER_MARKER` (`"\n\n---\n**Nguồn**\n"`, hằng ở `conversation/history.py`) rồi mỗi dòng `[n] {breadcrumb} — {source_document}` |
| `warning` | Nối dưới khối nguồn `⚠️ {message}`; `truncated` → `finish_reason="length"` |
| `error` | `delta.content = message` (đã có chữ thì xuống dòng, thêm `⚠️`); HTTP vẫn 200; `rate_limited` kèm "thử lại sau N giây" |
| `done` | Chunk kết thúc (+ `usage` nếu yêu cầu) rồi `[DONE]` |

- **Disclaimer thời điểm dữ liệu:** nối `DATA_SNAPSHOT_DISCLAIMER` (hằng ở `conversation/history.py`) đúng 1 lần, ở đuôi cuối cùng — sau khối nguồn và `warning`, ngay trước `done` (`conversation_spec.md` mục 4).
- **Keep-alive:** khi chờ admission/reasoning lâu, gửi comment SSE `: keep-alive` mỗi `KEEPALIVE_SECONDS = 15` (nếu OpenWebUI không chấp nhận thì dùng chunk `delta` rỗng). Nếu OpenWebUI không hỗ trợ `reasoning_content` thì bỏ `status`.
- Client ngắt kết nối: Starlette huỷ generator; `finally` cập nhật trace Langfuse với `outcome="client_disconnected"`.

## 7. Workflow & lifecycle (`app.py`, `routes.py`)

`create_app()` (`uvicorn --factory`), khởi tạo **một lần** trong `lifespan`: `Settings`, `Redis`, `corpus_version` (cùng `PROMPT_VERSION`/`model_name` gom vào `RuntimeVersions`, gắn vào metadata trace); `InputGuardrail`, `AnswerGenerator`, `QueryCondenser`, `RetrievalPipeline` (`GenerationPipeline` gom guardrail + generator), `AnswerCache`/`RetrievalCache`/`SingleFlight`, `AdmissionController`, `ChatOrchestrator` → `app.state`, route lấy qua dependency. Route chat: xác thực → `RequestContext`; rate limit phút → `build_window` (`InvalidConversationError` → 422) → `TurnTrace` + `orchestrator.stream(...)` → stream hoặc gom JSON; `finally` cập nhật trace Langfuse + metrics.

**Quy tắc event loop:** đoạn CPU đồng bộ mới mà đo > ~10 ms phải bọc `asyncio.to_thread` (Pinecone/HF đã qua `to_thread`; reranker chạy in-process trong executor 1 worker, `retrieval_spec.md` mục 6.1; `pyvi` ~0.3–2.3 ms/lần nên không cần).

## 8. Triển khai (`deploy/`, ngoài package)

Đóng gói/vận hành ở **`deploy/deploy_spec.md`**: máy cá nhân + Cloudflare Tunnel. Ràng buộc từ `api/`: **1 worker** vì semaphore admission là in-process (`uvicorn production_legal_qa_rag.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 1`; nhiều worker/replica thì chuyển semaphore sang Redis trước); `api` không publish cổng, chỉ `open-webui` gọi qua mạng nội bộ; healthcheck là `GET /readyz`.

## 9. Cấu hình OpenWebUI

**Bẫy PersistentConfig:** nhiều cài đặt chỉ đọc env ở lần khởi động đầu, sau đó nằm trong DB và chỉnh qua Admin Panel → đặt đúng ngay lần chạy đầu hoặc `ENABLE_PERSISTENT_CONFIG=false` để compose luôn là nguồn chuẩn. Tên biến kiểm với phiên bản ghim.

- **Kết nối:** `OPENAI_API_BASE_URL=http://api:8000/v1`, `OPENAI_API_KEY=<CHATBOT_API_KEY>`, `ENABLE_OLLAMA_API=false`, chỉ 1 model `legal-qa`.
- **Tắt các lời gọi phụ tới model** (nếu để mặc định chúng vào pipeline, bị guardrail chặn và đốt hạn mức): `ENABLE_TITLE_GENERATION`, `ENABLE_TAGS_GENERATION`, `ENABLE_FOLLOW_UP_GENERATION`, `ENABLE_AUTOCOMPLETE_GENERATION`, `ENABLE_RETRIEVAL_QUERY_GENERATION`, `ENABLE_SEARCH_QUERY_GENERATION` = `false`. Tắt tính năng thừa: RAG/tải tệp, web search, code execution, image generation, tool/function của người dùng.
- **Danh tính:** `ENABLE_FORWARD_USER_INFO_HEADERS=true`. **Đăng ký mở** (`ENABLE_SIGNUP=true`, `DEFAULT_USER_ROLE=user`; hạn mức bảo vệ bằng quota/ngân sách, `deploy_spec.md` mục 5); tài khoản đầu tiên là admin → đăng ký trước khi công bố URL.
- **Lưu trữ:** `DATABASE_URL=postgresql://…/openwebui`, `WEBUI_SECRET_KEY` cố định. Banner: "Hội thoại được lưu để cải thiện dịch vụ. Nội dung chỉ mang tính tham khảo, không thay thế tư vấn pháp lý." **Backend bỏ qua `system` message từ client.**

## 10. Config (`config.py`)

`ApiSettings`: `chatbot_api_key` (`CHATBOT_API_KEY`, `SecretStr`), `rate_limit_per_minute = 5`, `keepalive_seconds = 15`. `RedisSettings`: `redis_url` (`REDIS_URL`) dùng chung với `AdmissionSettings` và `cache/`. `api/` không đọc `.env` trực tiếp; cập nhật `.env.example` mọi biến mới.

## 11. Module (`src/production_legal_qa_rag/api/`)

`schemas.py` (request/response OpenAI), `auth.py` (Bearer + `RequestContext`), `rate_limit.py` (phút, Redis, fail-open), `openai_format.py` (event → chunk, khối nguồn, keep-alive, gom non-stream), `routes.py` (chat/models/health; trace + metrics trong `finally`), `app.py` (`create_app`, `lifespan`, exception handler dạng OpenAI).

## 12. Xử lý lỗi

Sai/thiếu Bearer → 401; `messages` không hợp lệ → 422; vượt rate limit phút → 429 + `Retry-After`; Redis chết → `/readyz` 503, chat vẫn chạy degrade (cache/quota bỏ qua); Langfuse chết/không cấu hình → chat vẫn chạy, mất nhật ký lượt đó; quota ngày/quá tải/lỗi LLM → trong luồng stream (mục 6), HTTP 200; ngoại lệ không lường trước → `app.py` trả 500 body OpenAI (stream đã bắt đầu thì phát nội dung lỗi chung rồi `[DONE]`), log kèm `request_id`, **không log nội dung**.

## 13. Nghiệm thu thủ công

1. `docker compose up -d` → mở URL tunnel (hoặc `127.0.0.1:3000`), đăng ký, đăng nhập, hỏi câu luật: stream ra, có "Đang tra cứu…", có khối Nguồn. 2. Follow-up ngắn trả đúng chủ đề. 3. Không có lời gọi phụ nào từ OpenWebUI: mỗi lượt đúng 1 trace Langfuse. 4. `curl` vào `api` từ ngoài mạng compose → không tới được; từ trong mạng thiếu key → 401. 5. ~20 người dùng đồng thời → không request nào treo, phần vượt hạn mức nhận thông báo rõ. 6. Đóng tab giữa chừng → không rò slot admission/khoá; trace có `outcome:client_disconnected`. 7. Tắt Redis → vẫn trả lời, `/readyz` 503, log warning.

## 14. Thứ tự implement & điểm mở

Đã chốt: hosting máy cá nhân + Cloudflare Tunnel; đăng ký OpenWebUI mở hẳn; **reranker chạy in-process trong `api`** (`retrieval_spec.md` mục 6.1). Hạn mức Groq free (30 RPM, 1K RPD, 8K TPM, **200K TPD** mỗi model/tài khoản) chỉ đủ khoảng 50–60 câu cache-miss/ngày cho generation; quota mặc định `AdmissionSettings` (user 5/ngày, toàn cục 50/ngày) suy ra từ đó, chỉnh sau khi có số liệu `usage` từ Langfuse.

## 15. Rủi ro

1. OpenWebUI đổi hành vi giữa phiên bản (tên biến, header, `reasoning_content`) → ghim phiên bản, ghi phiên bản đã nghiệm thu. 2. `messages[]` do client cung cấp không đáng tin (`conversation_spec.md` mục 3, 14). 3. Ngân sách Groq free không đủ cho public: cache + quota giảm nhẹ, không xoá hẳn rủi ro. 4. Một máy, một worker, không HA: chấp nhận.
