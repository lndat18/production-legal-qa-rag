# API — FastAPI (OpenAI-compatible) + OpenWebUI + Redis (+ Postgres cho OpenWebUI)

- Giữ nguyên số mục để không làm hỏng tham chiếu từ code/spec khác.
- Spec liên quan: [deploy_spec.md](../../../deploy/deploy_spec.md), [conversation_spec.md](../conversation/conversation_spec.md), [observability_spec.md](../observability/observability_spec.md).

## 1. Mục tiêu & phạm vi

- Đưa conversation → generation → retrieval thành chatbot nhiều người dùng qua web.
- Spec tổng cho kiến trúc/HTTP; deploy chi tiết ở deploy spec, nhật ký Langfuse ở observability 4.5.
- Luồng: HTTPS → Cloudflare Tunnel → OpenWebUI (login/history, DB Postgres) → API nội bộ Bearer → ChatOrchestrator.
- Orchestrator: guardrail ‖ condense → cache → admission → retrieve → generate; Redis cache/rate limit; Groq/HF/Pinecone và reranker in-process.
- Làm: OpenAI chat/models, auth/header identity/rate limit, event adapter, health/readiness, lifespan, request ID, compose/OpenWebUI config.
- Không làm: Kafka, UI/auth/user management riêng, host LLM, autoscale/Kubernetes/nhiều worker/WebSocket, API riêng ngoài OpenAI cho client khác.
- Acceptance chính: login/hỏi nhiều lượt, answer có nguồn; tải đồng thời có giới hạn/hàng đợi; backend chỉ mạng nội bộ có xác thực.

## 2. Input & Output (endpoint)

- `POST /v1/chat/completions`: `model`, `messages`, `stream`, `stream_options`; bỏ qua tham số khác.
- `GET /v1/models`: một model `legal-qa`, object `model`, owned_by `production-legal-qa-rag`.
- `GET /healthz`: process sống → 200; `GET /readyz`: Redis ping → 200, lỗi → 503; không ping Groq/Pinecone.
- Schema Pydantic v2 tại `schemas.py`.
- Stream `text/event-stream`: `data: {chat.completion.chunk}`; đầu role assistant, cuối finish_reason stop/length khi warning truncated.
- Có `stream_options.include_usage`: thêm usage chunk; kết thúc `data: [DONE]`.
- Adapter nhận `GenerationEvent`; non-stream gom event thành một `chat.completion`.
- Trước stream: HTTP 401/422/429/503, body OpenAI `{"error":{"message","type","code"}}`; 429 có Retry-After.
- Sau stream: lỗi thành nội dung (mục 6), HTTP vẫn 200.

## 3. Công cụ

- FastAPI/Uvicorn, `StreamingResponse` SSE, Redis asyncio, Langfuse.
- OpenWebUI/cloudflared image ghim phiên bản; Docker Compose/uv; kiểm dependency bằng pip-audit.
- API không kết nối Postgres; DB chỉ phục vụ OpenWebUI.

## 4. Xác thực & danh tính (`auth.py`)

- `Authorization: Bearer <CHATBOT_API_KEY>`; `secrets.compare_digest`; thiếu/sai → 401.
- Key trong `.env` compose; backend không expose internet.
- `ENABLE_FORWARD_USER_INFO_HEADERS=true`: đọc User-Id/Chat-Id từ `X-OpenWebUI-*`, chỉ tin sau Bearer hợp lệ.
- Chỉ dùng ID, không email/tên; thiếu User-Id → anonymous, rate limit chung.
- `RequestContext(user_id, chat_id, request_id)`; UUID mỗi request vào log và header `X-Request-Id`.

## 5. Rate limit theo phút (`rate_limit.py`)

- Trước stream, áp mọi request kể cả cache hit.
- Fixed window: `INCR rl:{user_id}:{phút}`, `EXPIRE 70`; mặc định `RATE_LIMIT_PER_MINUTE=5`; vượt → 429 + Retry-After.
- Redis lỗi → fail-open; admission in-process là giới hạn đồng thời cuối.
- Không có quota ngày ở admission; hạn mức ngày do 429 thật của provider (conversation 9/12.1).

## 6. Chuyển event → OpenAI (`openai_format.py`)

- Status → `delta.reasoning_content` một dòng tiến độ, không content; client không hỗ trợ thì bỏ.
- Token → `delta.content=text`; refusal → message và stop.
- Citations → nối `SOURCES_FOOTER_MARKER="\n\n---\n**Nguồn**\n"`, mỗi dòng `[n] {breadcrumb} — {source_document}`.
- Warning → `⚠️ {message}` dưới nguồn; truncated → finish_reason length.
- Error → message, xuống dòng/thêm ⚠️ nếu đã có text; rate_limited thêm “thử lại sau N giây”; HTTP 200.
- Done → final chunk, usage nếu yêu cầu, `[DONE]`.
- Disclaimer tĩnh `DATA_SNAPSHOT_DISCLAIMER`: đúng một lần sau nguồn/warnings, trước done (conversation 4).
- Chờ admission/reasoning: SSE comment `: keep-alive` mỗi `KEEPALIVE_SECONDS=15`; không tương thích thì delta rỗng.
- Client ngắt: Starlette hủy generator; finally ghi outcome client_disconnected.

## 7. Workflow & lifecycle (`app.py`, `routes.py`)

- `create_app()` qua `uvicorn --factory`; lifespan tạo một lần rồi inject qua app.state/dependency.
- Thành phần: `Settings`, `Redis`, corpus version + `PROMPT_VERSION`/model trong `RuntimeVersions`.
- Pipeline: `InputGuardrail`, `AnswerGenerator`, `EvidenceJudge`, `QueryCondenser`, `RetrievalPipeline`, `GenerationPipeline`.
- Điều phối: `AnswerCache`, `RetrievalCache`, `SingleFlight`, `AdmissionController`, `ChatOrchestrator`.
- Chat route: auth/context → rate limit → `build_window` (`InvalidConversationError` → 422 trước stream) → TurnTrace/orchestrator → SSE hoặc JSON; finally Langfuse/metrics.
- CPU đồng bộ mới đo >~10ms phải đưa `asyncio.to_thread`; Pinecone/HF đã offload, reranker executor một worker (retrieval 6.1).
- PyVi khoảng 0.3–2.3ms/lần nên chưa cần offload.

## 8. Triển khai (`deploy/`, ngoài package)

- Theo `deploy/deploy_spec.md`: máy cá nhân, Cloudflare Tunnel, API không publish cổng, readyz healthcheck.
- Uvicorn: `production_legal_qa_rag.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 1`.
- Một worker vì admission in-process; nhiều worker/replica phải đổi semaphore Redis trước.

## 9. Cấu hình OpenWebUI

- PersistentConfig trong DB có thể đè env sau lần đầu; cấu hình đúng lúc tạo hoặc `ENABLE_PERSISTENT_CONFIG=false`; kiểm tên biến theo tag ghim.
- `OPENAI_API_BASE_URL=http://api:8000/v1`, `OPENAI_API_KEY=<CHATBOT_API_KEY>`; `ENABLE_OLLAMA_API=false`; chỉ model legal-qa.
- Tắt `ENABLE_TITLE_GENERATION`, `ENABLE_TAGS_GENERATION`, `ENABLE_FOLLOW_UP_GENERATION`, `ENABLE_AUTOCOMPLETE_GENERATION`, `ENABLE_RETRIEVAL_QUERY_GENERATION`, `ENABLE_SEARCH_QUERY_GENERATION` để tránh gọi phụ/đốt quota.
- Tắt RAG/upload/web search/code/image/tool-function user.
- Forward user headers; đăng ký mở `ENABLE_SIGNUP=true`, `DEFAULT_USER_ROLE=user`; đăng ký admin đầu tiên trước công bố URL.
- `DATABASE_URL=postgresql://…/openwebui`; WEBUI_SECRET_KEY cố định.
- Banner: “Hội thoại được lưu để cải thiện dịch vụ. Nội dung chỉ mang tính tham khảo, không thay thế tư vấn pháp lý.”
- Backend bỏ system message từ client.

## 10. Config (`config.py`)

- `ApiSettings`: `chatbot_api_key` SecretStr, `rate_limit_per_minute=5`, `keepalive_seconds=15`.
- `RedisSettings.redis_url` (`REDIS_URL`): dùng chung cache/rate limit; `AdmissionSettings` không dùng Redis hiện tại.
- Module không đọc `.env` trực tiếp; biến mới thêm `.env.example`.

## 11. Module (`src/production_legal_qa_rag/api/`)

- `schemas.py`: OpenAI request/response; `auth.py`: Bearer/context; `rate_limit.py`: Redis fixed window/fail-open.
- `openai_format.py`: event/chunk/nguồn/keepalive/non-stream; `routes.py`: chat/models/health/trace/metrics.
- `app.py`: factory/lifespan/OpenAI exception handler.

## 12. Xử lý lỗi

- Bearer sai/thiếu → 401; messages sai → 422; phút vượt → 429 + Retry-After.
- Redis lỗi → readyz 503, chat degrade cache/rate limit; Langfuse lỗi/thiếu → chat chạy, mất nhật ký lượt đó.
- Provider hết hạn mức/quá tải/LLM lỗi sau stream → nội dung lỗi, HTTP 200.
- Unexpected trước stream → 500 body OpenAI; sau stream → lỗi chung + `[DONE]`.
- Log request ID, không nội dung user/answer/LLM exception message.

## 13. Nghiệm thu thủ công

- Compose → URL tunnel hoặc port loopback dev 127.0.0.1:3000 đã cấu hình → đăng ký/login/hỏi luật, tiến độ + answer + nguồn.
- Follow-up đúng chủ đề; mỗi lượt một trace, không OpenWebUI call phụ.
- Ngoài compose không gọi API; trong compose thiếu key → 401.
- Khoảng 20 user đồng thời: không treo, quá tải thông báo rõ.
- Đóng tab: nhả slot/lock, trace client_disconnected; tắt Redis: chat chạy, readyz 503, warning log.

## 14. Thứ tự implement & điểm mở

- Đã chốt: máy cá nhân/Cloudflare, đăng ký mở, reranker in-process (retrieval 6.1).
- Hạn mức Groq đã quan sát: 30 RPM/1K RPD/8K TPM/200K TPD mỗi model/tài khoản; generation từng khoảng 50–60 cache-miss/ngày.
- Bảo vệ hiện tại: cache, rate limit phút, admission, throttle bước nhẹ, 429 thật; không quota ngày user 5/toàn cục 50 như thiết kế cũ.
- Đo usage Langfuse trước quyết định đổi hạn mức/kiến trúc.

## 15. Rủi ro

- OpenWebUI đổi env/header/reasoning_content → pin và ghi bản nghiệm thu.
- Messages client không đáng tin (conversation 3/14).
- Public có thể cạn Groq quota; cache/rate limit không bảo đảm ngân sách ngày.
- Một host/worker, không HA: chấp nhận.
