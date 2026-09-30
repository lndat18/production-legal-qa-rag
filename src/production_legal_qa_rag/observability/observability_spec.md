# Observability — Langfuse tracing + Prometheus/Grafana metrics

> Cô đọng 2026-09-30 (giữ số mục vì code/spec khác tham chiếu, đặc biệt 4.3–4.5). Bản đầy đủ: git history.

## 1. Mục tiêu & phạm vi

Chốt 2026-09-26: chạy **Langfuse self-host + Prometheus + Grafana trên local trước**. Spec này chỉ làm **observability** (CD là spec riêng
sau). Hai công cụ trả lời hai câu hỏi khác nhau, không chồng chéo:
- **Langfuse:** "1 lượt hỏi cụ thể đã chạy qua bước nào, bước nào chậm/lỗi, model nào dùng key nào, tốn bao nhiêu token" → trace chi tiết.
- **Prometheus/Grafana:** "hệ thống đang khoẻ không ngay lúc này" → vài chỉ số thô mức vận hành (tỉ lệ lỗi/từ chối, độ trễ, HTTP).

**Làm:** trace 1 lượt hỏi (`chat_turn`), span/generation con cho từng bước chính (guardrail, condense, retrieve [gồm HyDE], admission, generate
[gồm Judge]) — mục 4; **Langfuse là nơi duy nhất lưu nhật ký từng lượt hỏi-đáp** (đã gỡ `chatlog/`, 2026-09-29), trace mang đủ các trường chatlog
từng lưu (mục 4.5); `GET /metrics` ở `api` (HTTP mặc định của `prometheus-fastapi-instrumentator` + `chat_turns_total`, `turn_latency_seconds`,
`time_to_first_token_seconds`) — mục 5; wiring dev để Prometheus scrape `api` — mục 7.
**Ngoài phạm vi:** CD; dashboard Grafana as-code (datasource đã provision, panel tự tạo tay); alerting; trace bước offline/batch (chỉ đường online);
Langfuse Cloud (giữ self-host: dữ liệu hỏi-đáp pháp lý có thể chứa tình huống cá nhân, không rời máy).

**Tiêu chí số 1:** instrumentation **không bao giờ** làm chậm hay hỏng câu trả lời — thiếu/lỗi Langfuse hoặc Prometheus thì chatbot vẫn chạy, chỉ mất
khả năng quan sát.

## 2. Input & Output

Không đổi logic nghiệp vụ; chỉ bọc span/generation quanh lời gọi đã có. Output: trace cây trên Langfuse UI (`http://localhost:3001`, dev), `GET /metrics`
trên `api`, và mỗi trace `chat_turn` chứa đủ dữ liệu nhật ký lượt hỏi (mục 4.5).

## 3. Công cụ & công nghệ

`langfuse` Python SDK **v4** (`get_client()`, OTel-based; **không dùng v3** — SDK v3 bị Langfuse cutover 16/11/2026; platform self-host pin `:4`, yêu cầu
tối thiểu self-host ≥ 3.63.0); `prometheus-fastapi-instrumentator` (`Instrumentator().instrument(app).expose(app)`); `prometheus_client` trực tiếp cho
`Counter`/`Histogram` (không thêm thư viện); Grafana (compose + datasource provisioning có sẵn). Dependency mới nhóm `production`: `langfuse`,
`prometheus-fastapi-instrumentator`; chạy lại `pip-audit`.

## 4. Trace Langfuse (`observability/tracing.py`)

### 4.1 Client & fail-safe

`get_langfuse_client()` = singleton `get_client()`. SDK v4 đọc `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_BASE_URL` từ env; **thiếu key →
disabled mode tự động**, mọi span/generation thành no-op, không raise, không log rác. Đây là toàn bộ cơ chế fail-safe — **không** thêm biến
`OBSERVABILITY_ENABLED`. Production không set `LANGFUSE_*` → tự tắt.

### 4.2 Hai loại observation

- **`span(name, *, user_id=None, session_id=None, **kwargs)`**: bước không gọi LLM (cache lookup, container). `user_id`/`session_id` chỉ cho root span:
  `span` bọc `propagate_attributes(user_id=..., session_id=...)` để đặt ở cấp trace và lan xuống observation con.
- **`generation(name, *, model, **kwargs)`**: bước gọi LLM — Langfuse hiển thị riêng model/usage/cost; cần cho các bước dùng nhiều model/key khác nhau
  (`conversation_spec.md` mục 12.1). Cả hai dựng trên `start_as_current_observation(as_type=...)`.

`observability/` là package hạ tầng cross-cutting (như logging): **được phép import trực tiếp** từ bất kỳ module nào gọi LLM, không đi qua tầng điều
phối. OTel context tự propagate qua `await`/`asyncio.gather` (context copy khi tạo `Task`), nên span/generation tạo bên trong tự lồng đúng dưới span cha
đang active — không truyền tay span qua tham số hàm.

### 4.3 Cây trace mục tiêu (1 lượt hỏi)

```
chat_turn                                       (root — api/routes.py)
├── guardrail    [generation]  generation/guardrail.py  InputGuardrail.check_input
├── condense     [generation]  conversation/condenser.py  QueryCondenser.condense
├── cache_lookup [span]        conversation/orchestrator.py  answer_cache.get
└── admission    [span]        conversation/orchestrator.py  quanh admission.slot(...)
    ├── retrieve [span]        orchestrator quanh retrieve()
    │   └── hyde [generation]  retrieval/hyde.py
    └── generate [span]        orchestrator quanh generation.generate()
        ├── answer [generation] generation/generator.py
        └── judge  [generation] generation/judge.py
```

`guardrail` và `condense` chạy song song (`asyncio.gather`) — cả hai vẫn là con trực tiếp của `chat_turn`, không lồng nhau (nghiệm thu 9.3). Cache hit
thì `admission`/`retrieve`/`generate` không được tạo. Mỗi `generation` gắn `metadata={"key_bucket": ...}` từ `retrieval/llm_throttle.py` (bucket
`(model, key)`) để nhìn trên UI model/key nào đang bị dùng nhiều — **không** tạo thêm Prometheus metric trùng lặp. Nội dung tối thiểu: `verdict`
(guardrail), `standalone_query` (condense), `num_chunks` + `chunk_ids` + `top_rerank_score` (retrieve), `usage` (answer, judge). Chỉ lưu những gì chatlog
từng lưu (câu hỏi/trả lời/chunk id).

### 4.4 Root trace (`api/routes.py`)

Root span `chat_turn` bọc đúng đoạn stream 1 lượt (`stream_chat_turn`): `input=root_query`, `metadata={"request_id"}`, `user_id=context.user_id`,
`session_id=context.chat_id`; `except asyncio.CancelledError` đặt `trace.outcome = "client_disconnected"` rồi raise; `finally` gọi
`update_turn_trace(root_span, trace, request_id=..., versions=...)` (mục 4.5), rồi ngoài cùng `metrics.record_turn(trace)`. `user_id`/`session_id`
(= `chat_id` OpenWebUI) đặt ở cấp trace bằng thuộc tính chuẩn của Langfuse (`propagate_attributes`, không phải metadata tự chế) để lọc theo người
dùng và gom các lượt cùng hội thoại; thiếu thì bỏ trống, không bịa. Lifespan (`api/app.py`) gọi `get_langfuse_client().flush()` lúc shutdown (chặn tối
đa vài giây) để không mất trace các lượt cuối.

### 4.5 Trace thay bảng `chat_turns` (chốt 2026-09-29, **đã implement**)

Lý do: chỉ 1 nơi lưu nhật ký, tránh hai nguồn trùng nội dung; dữ liệu hỏi-đáp thật để chấm RAGAS Phase 2 lấy từ Langfuse (API/export). `chatlog/`,
`alembic/`, `tools/chatlog.py`, `tools/purge_chatlog.py`, `DatabaseSettings`/`CHATLOG_*`, dependency `alembic`/`sqlalchemy`/`asyncpg` và DB `chatbot`
trong compose/initdb đã gỡ; `GET /readyz` chỉ còn ping Redis; `TurnTrace.langfuse_trace_id` bỏ. DB `chatbot` cũ trong volume có sẵn không tự biến
mất — drop tay.

**Chi tiết SDK v4 (kiểm trên 4.15.x):** `user_id`/`session_id` đặt bằng `propagate_attributes` (bọc trong `tracing.span`); `output` và metadata là
`.update(output=..., metadata=...)` trên root observation (dict được SDK flatten thành `langfuse.observation.metadata.<key>`, list/dict lồng nhau
serialize JSON); `tags` chỉ biết cuối lượt nên đặt bằng `propagate_attributes(tags=[...])` ngay lúc root span còn active; `set_trace_io` bị SDK
deprecate nên không dùng (input/output của observation gốc là input/output của trace). `prompt_version`/`corpus_version`/`model_name` do
`RuntimeVersions` (`turn_trace.py`) giữ, tạo 1 lần ở lifespan (`app.state.runtime_versions`).

Ánh xạ trường chatlog cũ → Langfuse (`observability/turn_trace.py::update_turn_trace`, cuối lượt trên root span):

| Trường chatlog cũ | Vị trí trên Langfuse |
| --- | --- |
| `raw_query` | `input` của trace |
| `answer_text` | `output` của trace (rỗng nếu từ chối/lỗi) |
| `user_id`, `chat_id` | `user_id`, `session_id` của trace |
| `request_id` | metadata |
| `standalone_query`, `verdict`, `chunk_ids` | có ở span con (4.3), lặp vào metadata root để lọc |
| `outcome` (`answered`/`refused`/`error`/`client_disconnected`), `error_code`, `cache_status`, `citations`, `warnings` | metadata root |
| `usage` (token) | ở generation con; tổng lượt ở metadata root nếu có trên `TurnTrace` |
| `time_to_first_token_ms`, `latency_ms` | metadata root |
| `prompt_version`, `corpus_version`, `model_name` | metadata root — để so trước/sau khi đổi cấu hình |
| Lượt lỗi/từ chối/cache hit/ngắt kết nối | vẫn phải có đúng 1 trace (không mất vì đi nhánh sớm); `outcome` phân biệt |

Đặt `tags` `outcome:<x>`, `cache:<y>` để lọc nhanh. **Ràng buộc:** ghi trace không bao giờ chặn/làm hỏng câu trả lời. Khác chatlog: Langfuse không chạy
thì **mất** nhật ký lượt đó (không có hàng đợi bền) — chấp nhận cho dự án cá nhân, và là lý do stack observe nên luôn chạy cùng production.

## 5. Metrics Prometheus (`observability/metrics.py`)

Chỉ 2 nhóm, cố tình **không** đo lặp lại thứ Langfuse trả lời tốt hơn (per-model/key token, per-step latency): Prometheus/Grafana là "đèn báo sức khoẻ",
không phải nơi debug một lượt cụ thể. (1) **HTTP mặc định** của instrumentator (`http_requests_total`, `http_request_duration_seconds`…), instrument
nguyên app. (2) **Custom mức lượt hỏi**, nguồn `TurnTrace` đã điền đủ (cùng điểm cuối lượt với cập nhật trace — không thêm import vào `conversation/`/
`retrieval/`): `chat_turns_total{outcome,cache_status}` (Counter), `turn_latency_seconds{outcome}` (Histogram, buckets 0.5…120), `time_to_first_token_seconds`
(Histogram, buckets 0.2…20). `record_turn(trace)` bọc try/except trong chính hàm (lỗi ghi metric không lan ra, chỉ log warning); `instrument_app(app)`
gọi 1 lần trong `create_app()`, `record_turn` gọi ở `finally` cuối lượt trong `routes.py`. **Lưu ý SSE:** route stream lâu nên histogram HTTP mặc định đo
cả thời gian chờ token, số trùng phần lớn `turn_latency_seconds` — chấp nhận (đơn giản hơn loại trừ route), hai metric khác góc nhìn (route HTTP vs `outcome`
nghiệp vụ).

## 6. Config (`config.py`)

`LangfuseSettings` (`env_prefix="LANGFUSE_"`, `env_ignore_empty=True`): `public_key`/`secret_key: SecretStr | None = None`, `base_url = "http://localhost:3001"`.
Để trống key → SDK disabled (4.1), không cần cờ riêng. `api` khi dev chạy trực tiếp trên host nên `base_url` mặc định trỏ vào cổng `langfuse-web` publish ra
host (`127.0.0.1:3001:3000`), không phải tên service Docker. `api/`, `conversation/`, `generation/`, `retrieval/` không đọc `LangfuseSettings`; package
`observability/` tự đọc khi khởi tạo client. 3 biến `LANGFUSE_*` nằm trong `.env.example` root (block APP; 1 cặp `.env` duy nhất, `deploy_spec.md` mục 7).

## 7. Triển khai (`dev/observability/` + `deploy/docker-compose.observe.yml`)

Mục tiêu (chốt 2026-09-29): quan sát **traffic end-user thật** trên `api` production. Stack observe (Langfuse/Prometheus/Grafana) là compose riêng ở
`dev/observability/`, tách khỏi `deploy/` (không phục vụ end-user); `api` production nối vào bằng 1 network chung `legal-qa-observe` (tên cố định) do
`dev/observability/docker-compose.yml` tạo (`langfuse-web`, `prometheus` join). `deploy/docker-compose.observe.yml` (override, cùng kiểu `docker-compose.gpu.yml`)
khai network `external`, thêm `api` vào, ghi đè `LANGFUSE_BASE_URL=http://langfuse-web:3000` và `LANGFUSE_TRACING_ENVIRONMENT=production`; `deploy/up.sh` tự ghép
override khi network đã tồn tại (stack đang chạy), chưa chạy thì bỏ qua. `dev/observability/prometheus.yml` scrape `api:8000` (label `env=production`).
`LANGFUSE_PUBLIC_KEY`/`SECRET_KEY` điền vào `.env` root, tạo tay qua Langfuse UI sau lần `docker compose -f dev/observability/docker-compose.yml up -d` đầu
(cần symlink `dev/observability/.env` → `../../.env`); điền xong chạy lại `./deploy/up.sh`. Stack observe chạy cùng máy với production (chưa chốt tách VM).

## 8. Module

`observability/`: `tracing.py` (`get_langfuse_client`, `span`, `generation`), `turn_trace.py` (`RuntimeVersions`, `update_turn_trace`), `metrics.py`
(3 metric, `record_turn`, `instrument_app`). Nơi khác được sửa: `conversation/orchestrator.py` (bọc `cache_lookup`/`admission`/`retrieve`/`generate` bằng
`tracing.span`), `generation/guardrail.py`/`conversation/condenser.py`/`retrieval/hyde.py`/`generation/generator.py`/`generation/judge.py` (bọc lời gọi LLM bằng
`tracing.generation`), `api/routes.py` (root span + metrics), `api/app.py` (`instrument_app`, `flush()` lúc shutdown).

## 9. Nghiệm thu thủ công

1. Bật stack (`dev/observability`), hỏi vài câu qua OpenWebUI/`tools/conversation.py`. 2. Langfuse UI: mỗi lượt đúng 1 trace `chat_turn`, cây span khớp 4.3 (span
   nào chạy tuỳ nhánh cache hit/miss/refused/error). 3. `guardrail` + `condense` (song song) đều là con trực tiếp của `chat_turn`, không lẫn ngữ cảnh. 4. `curl
   api:8000/metrics` thấy `chat_turns_total`, `turn_latency_seconds`, `http_requests_total`; Prometheus (`localhost:9092/targets`) báo `legal-qa-api` `UP`. 5. Trace có
   đủ nhật ký (4.5): `input`, `output`, `user_id`/`session_id`, metadata `outcome`/`cache_status`/`chunk_ids`; lượt từ chối/lỗi/cache hit vẫn có trace riêng; lọc `session_id`
   gom đúng hội thoại. 6. Bỏ trống `LANGFUSE_*` (hoặc tắt stack) → chatbot vẫn chạy, không lỗi/warning lặp. 7. Tắt Langfuse giữa chừng → chatbot vẫn trả lời (SDK buffer/retry rồi bỏ,
   không chặn). 8. Tổng `mem_limit` observability stack (~3.4GB ClickHouse+MinIO) + load rerank không làm OOM. 9. `docker compose down` → trace các lượt cuối không mất (`flush()` chạy).

## 10. Rủi ro / điểm mở

1. OTel context qua `asyncio.gather`/`Task` tự propagate về lý thuyết; nghiệm thu 9.3 là bằng chứng thực tế duy nhất — sai thì cần `contextvars.copy_context().run(...)` quanh từng
   coroutine trong `gather`. 2. Compose observability nặng (~3.4GB): không bật cùng lúc load model trên máy yếu. 3. `langfuse` + `prometheus-fastapi-instrumentator` kéo
   `opentelemetry-*`: kiểm xung đột (`uv sync` + `pip-audit`), đặc biệt nhóm `production` vs `eval` đã có `[tool.uv.conflicts]` cho `openai`. 4. Tách observability sang VM riêng
   (chưa chốt): chỉ đổi `LANGFUSE_BASE_URL`/network, code không đổi. 5. Chưa đo tác động hiệu năng của bọc span/generation (vài ms/lời gọi LLM); nếu chậm rõ rệt thì batch export/giảm
   granularity thay vì bỏ hẳn.
