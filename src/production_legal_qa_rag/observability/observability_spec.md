# Observability — Langfuse tracing + Prometheus/Grafana metrics

> Giữ nguyên số mục vì code/spec khác tham chiếu (đặc biệt 4.3–4.5).

## 1. Mục tiêu & phạm vi

Chạy **Langfuse self-host + Prometheus + Grafana trên local**. Hai công cụ trả lời hai câu hỏi khác nhau:
- **Langfuse:** "1 lượt hỏi cụ thể đã chạy qua bước nào, bước nào chậm/lỗi, model nào dùng key nào, tốn bao nhiêu token" → trace chi tiết.
- **Prometheus/Grafana:** "hệ thống đang khoẻ không ngay lúc này" → vài chỉ số thô mức vận hành.

**Làm:** trace 1 lượt hỏi (`chat_turn`), span/generation con cho từng bước chính (guardrail, condense, retrieve [gồm HyDE], admission, generate [gồm Judge]) — mục 4; **Langfuse là nơi duy nhất lưu nhật ký từng lượt hỏi-đáp**, trace mang đủ các trường nhật ký (mục 4.5); `GET /metrics` ở `api` (HTTP mặc định của `prometheus-fastapi-instrumentator` + `chat_turns_total`, `turn_latency_seconds`, `time_to_first_token_seconds`) — mục 5; wiring để Prometheus scrape `api` — mục 7.
**Ngoài phạm vi:** CD; dashboard Grafana as-code (datasource đã provision, panel tự tạo tay); alerting; trace bước offline/batch; Langfuse Cloud (dữ liệu hỏi-đáp có thể chứa tình huống cá nhân, không rời máy).

**Tiêu chí số 1:** instrumentation **không bao giờ** làm chậm hay hỏng câu trả lời — thiếu/lỗi Langfuse hoặc Prometheus thì chatbot vẫn chạy, chỉ mất khả năng quan sát.

## 2. Input & Output

Không đổi logic nghiệp vụ; chỉ bọc span/generation quanh lời gọi đã có. Output: trace cây trên Langfuse UI (`http://localhost:3001`, dev), `GET /metrics` trên `api`, và mỗi trace `chat_turn` chứa đủ dữ liệu nhật ký lượt hỏi (mục 4.5).

## 3. Công cụ & công nghệ

`langfuse` Python SDK **v4** (`get_client()`, OTel-based; **không dùng v3**; platform self-host pin `:4`, yêu cầu ≥ 3.63.0); `prometheus-fastapi-instrumentator` (`Instrumentator().instrument(app).expose(app)`); `prometheus_client` trực tiếp cho `Counter`/`Histogram`; Grafana (compose + datasource provisioning). Dependency nhóm `production`: `langfuse`, `prometheus-fastapi-instrumentator`; chạy lại `pip-audit`.

## 4. Trace Langfuse (`observability/tracing.py`)

### 4.1 Client & fail-safe

`get_langfuse_client()` = singleton `get_client()`. SDK v4 đọc `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_BASE_URL` từ env; **thiếu key → disabled mode tự động**, mọi span/generation thành no-op. Đây là toàn bộ cơ chế fail-safe — **không** thêm biến `OBSERVABILITY_ENABLED`. Production không set `LANGFUSE_*` → tự tắt.

### 4.2 Hai loại observation

- **`span(name, *, user_id=None, session_id=None, **kwargs)`**: bước không gọi LLM. `user_id`/`session_id` chỉ cho root span: `span` bọc `propagate_attributes(user_id=..., session_id=...)` để đặt ở cấp trace và lan xuống observation con.
- **`generation(name, *, model, **kwargs)`**: bước gọi LLM (Langfuse hiển thị riêng model/usage/cost). Cả hai dựng trên `start_as_current_observation(as_type=...)`.

`observability/` là package hạ tầng cross-cutting: **được phép import trực tiếp** từ bất kỳ module nào gọi LLM. OTel context tự propagate qua `await`/`asyncio.gather`, nên span/generation tạo bên trong tự lồng đúng dưới span cha đang active — không truyền tay span qua tham số hàm.

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

`guardrail` và `condense` chạy song song (`asyncio.gather`) — cả hai là con trực tiếp của `chat_turn`, không lồng nhau. Cache hit thì `admission`/`retrieve`/`generate` không được tạo. Mỗi `generation` gắn `metadata={"key_bucket": ...}` từ `retrieval/llm_throttle.py` (bucket `(model, key)`) — **không** tạo thêm Prometheus metric trùng lặp. Nội dung tối thiểu: `verdict` (guardrail), `standalone_query` (condense), `num_chunks` + `chunk_ids` + `top_rerank_score` (retrieve), `usage` (answer, judge).

### 4.4 Root trace (`api/routes.py`)

Root span `chat_turn` bọc đúng đoạn stream 1 lượt (`stream_chat_turn`): `input=root_query`, `metadata={"request_id"}`, `user_id=context.user_id`, `session_id=context.chat_id`; `except asyncio.CancelledError` đặt `trace.outcome = "client_disconnected"` rồi raise; `finally` gọi `update_turn_trace(root_span, trace, request_id=..., versions=...)` (mục 4.5), rồi ngoài cùng `metrics.record_turn(trace)`. `user_id`/`session_id` đặt ở cấp trace bằng `propagate_attributes` (không phải metadata tự chế); thiếu thì bỏ trống. Lifespan (`api/app.py`) gọi `get_langfuse_client().flush()` lúc shutdown để không mất trace các lượt cuối.

### 4.5 Trace là nhật ký lượt hỏi

Một nơi lưu nhật ký duy nhất; dữ liệu hỏi-đáp thật để chấm RAGAS lấy từ Langfuse (API/export). DB `chatbot` cũ trong volume có sẵn không tự mất — drop tay.

**Chi tiết SDK v4 (4.15.x):** `user_id`/`session_id` đặt bằng `propagate_attributes` (bọc trong `tracing.span`); `output` và metadata là `.update(output=..., metadata=...)` trên root observation (dict được flatten thành `langfuse.observation.metadata.<key>`, list/dict lồng nhau serialize JSON); `tags` đặt bằng `propagate_attributes(tags=[...])` khi root span còn active; không dùng `set_trace_io` (deprecated). `prompt_version`/`corpus_version`/`model_name` do `RuntimeVersions` (`turn_trace.py`) giữ, tạo 1 lần ở lifespan (`app.state.runtime_versions`).

Ánh xạ trường nhật ký → Langfuse (`observability/turn_trace.py::update_turn_trace`, cuối lượt trên root span):

| Trường | Vị trí trên Langfuse |
| --- | --- |
| `raw_query` | `input` của trace |
| `answer_text` | `output` của trace (rỗng nếu từ chối/lỗi) |
| `user_id`, `chat_id` | `user_id`, `session_id` của trace |
| `request_id` | metadata |
| `standalone_query`, `verdict`, `chunk_ids` | có ở span con (4.3), lặp vào metadata root để lọc |
| `outcome` (`answered`/`refused`/`error`/`client_disconnected`), `error_code`, `cache_status`, `citations`, `warnings` | metadata root |
| `usage` (token) | ở generation con; tổng lượt ở metadata root nếu có trên `TurnTrace` |
| `time_to_first_token_ms`, `latency_ms` | metadata root |
| `prompt_version`, `corpus_version`, `model_name` | metadata root |
| Lượt lỗi/từ chối/cache hit/ngắt kết nối | vẫn phải có đúng 1 trace; `outcome` phân biệt |

Đặt `tags` `outcome:<x>`, `cache:<y>` để lọc nhanh. **Ràng buộc:** ghi trace không bao giờ chặn/làm hỏng câu trả lời. Langfuse không chạy thì **mất** nhật ký lượt đó (không có hàng đợi bền) → stack observe nên luôn chạy cùng production.

## 5. Metrics Prometheus (`observability/metrics.py`)

Chỉ 2 nhóm, **không** đo lặp lại thứ Langfuse trả lời tốt hơn (per-model/key token, per-step latency). (1) **HTTP mặc định** của instrumentator (`http_requests_total`, `http_request_duration_seconds`…). (2) **Custom mức lượt hỏi**, nguồn `TurnTrace` (cùng điểm cuối lượt với cập nhật trace): `chat_turns_total{outcome,cache_status}` (Counter), `turn_latency_seconds{outcome}` (Histogram, buckets 0.5…120), `time_to_first_token_seconds` (Histogram, buckets 0.2…20). `record_turn(trace)` bọc try/except trong chính hàm (lỗi ghi metric chỉ log warning); `instrument_app(app)` gọi 1 lần trong `create_app()`, `record_turn` gọi ở `finally` cuối lượt trong `routes.py`. Histogram HTTP mặc định đo cả thời gian chờ token (route SSE) nên số trùng phần lớn `turn_latency_seconds` — chấp nhận.

## 6. Config (`config.py`)

`LangfuseSettings` (`env_prefix="LANGFUSE_"`, `env_ignore_empty=True`): `public_key`/`secret_key: SecretStr | None = None`, `base_url = "http://localhost:3001"`. Để trống key → SDK disabled (4.1). `api/`, `conversation/`, `generation/`, `retrieval/` không đọc `LangfuseSettings`; package `observability/` tự đọc khi khởi tạo client. 3 biến `LANGFUSE_*` nằm trong `.env.example` root (block APP; 1 cặp `.env` duy nhất, `deploy_spec.md` mục 7).

## 7. Triển khai (`observability/` + `deploy/docker-compose.observe.yml`)

Quan sát **traffic end-user thật** trên `api` production. Stack observe (Langfuse/Prometheus/Grafana) là compose riêng ở `observability/`, tách khỏi `deploy/`; `api` production nối vào bằng 1 network chung `legal-qa-observe` (tên cố định) do `observability/docker-compose.yml` tạo. `deploy/docker-compose.observe.yml` (override) khai network `external`, thêm `api` vào, ghi đè `LANGFUSE_BASE_URL=http://langfuse-web:3000` và `LANGFUSE_TRACING_ENVIRONMENT=production`; `deploy/up.sh` tự ghép override khi network đã tồn tại. `observability/prometheus.yml` scrape `api:8000` (label `env=production`). `LANGFUSE_PUBLIC_KEY`/`SECRET_KEY` điền vào `.env` root, tạo tay qua Langfuse UI sau lần `./observability/up.sh` đầu (script bọc `docker compose --env-file ../.env`; `down.sh` giữ volume); điền xong chạy lại `./deploy/up.sh`. Stack observe chạy cùng máy với production.

## 8. Module

`observability/`: `tracing.py` (`get_langfuse_client`, `span`, `generation`), `turn_trace.py` (`RuntimeVersions`, `update_turn_trace`), `metrics.py` (3 metric, `record_turn`, `instrument_app`). Nơi khác được sửa: `conversation/orchestrator.py` (bọc `cache_lookup`/`admission`/`retrieve`/`generate` bằng `tracing.span`), `generation/guardrail.py`/`conversation/condenser.py`/`retrieval/hyde.py`/`generation/generator.py`/`generation/judge.py` (bọc lời gọi LLM bằng `tracing.generation`), `api/routes.py` (root span + metrics), `api/app.py` (`instrument_app`, `flush()` lúc shutdown).

## 9. Nghiệm thu thủ công

1. Bật stack, hỏi vài câu: mỗi lượt đúng 1 trace `chat_turn`, cây span khớp 4.3. 2. `guardrail` + `condense` (song song) đều là con trực tiếp của `chat_turn`. 3. `curl api:8000/metrics` thấy `chat_turns_total`, `turn_latency_seconds`, `http_requests_total`; Prometheus (`localhost:9092/targets`) báo `legal-qa-api` `UP`. 4. Trace có đủ nhật ký (4.5): `input`, `output`, `user_id`/`session_id`, metadata `outcome`/`cache_status`/`chunk_ids`; lượt từ chối/lỗi/cache hit vẫn có trace riêng; lọc `session_id` gom đúng hội thoại. 5. Bỏ trống `LANGFUSE_*` hoặc tắt Langfuse giữa chừng → chatbot vẫn trả lời, không lỗi/warning lặp. 6. Tổng `mem_limit` observability stack (~3.4GB) + load rerank không làm OOM. 7. `docker compose down` → trace các lượt cuối không mất (`flush()` chạy).

## 10. Rủi ro / điểm mở

1. OTel context qua `asyncio.gather` tự propagate; nghiệm thu 9.2 là bằng chứng thực tế duy nhất — sai thì cần `contextvars.copy_context().run(...)` quanh từng coroutine trong `gather`. 2. Compose observability nặng (~3.4GB): không bật cùng lúc load model trên máy yếu. 3. `langfuse` + `prometheus-fastapi-instrumentator` kéo `opentelemetry-*`: kiểm xung đột (`uv sync` + `pip-audit`), đặc biệt nhóm `production` vs `eval` đã có `[tool.uv.conflicts]` cho `openai`.
