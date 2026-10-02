# Observability — Langfuse tracing + Prometheus/Grafana metrics

- Giữ nguyên số mục, đặc biệt 4.3–4.5.
- Spec liên quan: [api_spec.md](../api/api_spec.md), [conversation_spec.md](../conversation/conversation_spec.md).

## 1. Mục tiêu & phạm vi

- Langfuse self-host: từng lượt chạy bước nào, model/key bucket nào, chậm/lỗi/token/cost.
- Prometheus/Grafana local: sức khỏe/tải hệ thống, chỉ số vận hành thô.
- Làm: chat_turn trace và observation bước chính; Langfuse là nhật ký hỏi-đáp duy nhất; API /metrics + Prometheus scrape.
- Không làm: CD, dashboard panel as-code (datasource provision, panel thủ công), alerting, offline/batch trace, Langfuse Cloud.
- Instrumentation không chặn/làm hỏng answer; thiếu dịch vụ chỉ mất quan sát.

## 2. Input & Output

- Bọc span/generation quanh lời gọi có sẵn, không đổi nghiệp vụ.
- Output: cây chat_turn trên Langfuse (`http://localhost:3001`), API `GET /metrics`, nhật ký đủ trường mục 4.5.

## 3. Công cụ & công nghệ

- Langfuse Python SDK v4 (`get_client()`, OTel); không v3; self-host pin `:4`, platform ≥3.63.0.
- `prometheus-fastapi-instrumentator`: instrument/expose; `prometheus_client`: Counter/Histogram; Grafana + datasource provisioning.
- Production dependency: langfuse/instrumentator; kiểm pip-audit.

## 4. Trace Langfuse (`observability/tracing.py`)

### 4.1 Client & fail-safe

- `get_langfuse_client()` singleton `get_client()`; SDK đọc `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`.
- Thiếu key → disabled/no-op; không thêm OBSERVABILITY_ENABLED.
- Không set LANGFUSE_* production → tự tắt.

### 4.2 Hai loại observation

- `span(name, *, user_id=None, session_id=None, **kwargs)`: bước không LLM.
- `generation(name, *, model, **kwargs)`: bước LLM, model/usage/cost riêng.
- Cả hai dùng `start_as_current_observation(as_type=...)`.
- Root span bọc `propagate_attributes(user_id=..., session_id=...)` để đặt danh tính cấp trace và propagate; con không tự đặt lại.
- Observability là hạ tầng cross-cutting, được import trực tiếp tại module LLM.
- OTel context qua await/gather; không truyền tay span qua tham số.

### 4.3 Cây trace mục tiêu (1 lượt hỏi)

```text
chat_turn                                # root, api/routes.py
├── guardrail [generation]               # generation/guardrail.py
├── condense [generation]                # conversation/condenser.py
├── cache_lookup [span]                  # conversation/orchestrator.py
└── admission [span]                     # quanh admission.slot
    ├── retrieve [span]                  # quanh retrieve
    │   └── hyde [generation]            # retrieval/hyde.py
    └── generate [span]                  # quanh generation.generate
        ├── answer [generation]          # generation/generator.py
        └── judge [generation]           # generation/judge.py
```

- Guardrail/condense gather song song, cùng là con root; cache hit không có admission/retrieve/generate.
- Generation metadata key_bucket từ llm_throttle `(model, key)`; không lộ key hoặc thêm metric Prometheus trùng.
- Nội dung tối thiểu: `InputGuardrail.check_input` verdict, `QueryCondenser.condense` standalone_query, retrieve/HyDE num_chunks/chunk_ids/top_rerank_score, answer/Judge usage.

### 4.4 Root trace (`api/routes.py`)

- Root chat_turn bọc `stream_chat_turn`: input root_query, request_id metadata, user_id/session_id từ context.
- Thiếu identity thì bỏ trống; dùng propagate_attributes, không metadata tự chế.
- CancelledError → outcome client_disconnected rồi raise.
- Finally: `update_turn_trace(root_span, trace, request_id=..., versions=...)`, ngoài cùng `metrics.record_turn(trace)`.
- API lifespan shutdown gọi client.flush(), giữ trace lượt cuối.

### 4.5 Trace là nhật ký lượt hỏi

- Dữ liệu QA thật cho RAGAS eval lấy Langfuse API/export; không DB chatbot riêng.
- DB chatbot cũ trong volume không tự mất; người vận hành drop tay.
- SDK v4 4.15.x: root `.update(output=..., metadata=...)`; dict flatten `langfuse.observation.metadata.<key>`, list/dict lồng serialize JSON.
- User/session/tags qua propagate_attributes khi root active; không dùng deprecated set_trace_io.
- `RuntimeVersions` trong turn_trace.py: prompt_version/corpus_version/model_name, tạo một lần lifespan, app.state.runtime_versions.
- Mapping nhật ký:
  - raw_query → trace input; answer_text → output, rỗng nếu lỗi/từ chối.
  - user_id/chat_id → trace user_id/session_id; request_id → root metadata.
  - standalone_query/verdict/chunk_ids → observation con + root metadata để lọc.
  - outcome (answered/refused/error/client_disconnected), error_code/cache_status/citations/warnings → root metadata.
  - usage → generation con; tổng lượt → root metadata khi TurnTrace có.
  - `time_to_first_token_ms`/`latency_ms`, prompt/corpus version/model → root metadata.
- Mọi lượt lỗi/refusal/cache hit/ngắt kết nối vẫn đúng một trace; tags outcome:<x>, cache:<y>.
- Trace không cản answer; không có hàng đợi bền, Langfuse ngừng thì mất nhật ký lượt đó; nên chạy observe cùng production.

## 5. Metrics Prometheus (`observability/metrics.py`)

- Nhóm HTTP mặc định instrumentator: http_requests_total/http_request_duration_seconds…
- Nhóm TurnTrace: chat_turns_total{outcome,cache_status} Counter; turn_latency_seconds{outcome} Histogram buckets 0.5…120; time_to_first_token_seconds Histogram buckets 0.2…20.
- Không metric token theo model/key hoặc latency theo bước, đã có Langfuse.
- `instrument_app(app)` một lần create_app; record_turn ở finally route, tự bắt lỗi/log warning.
- HTTP histogram gồm thời gian chờ SSE, trùng phần turn latency: chấp nhận.

## 6. Config (`config.py`)

- `LangfuseSettings`: `env_prefix="LANGFUSE_"`, `env_ignore_empty=True`; `public_key`/`secret_key: SecretStr | None = None`; `base_url="http://localhost:3001"`.
- Observability tự đọc config khi tạo client; api/conversation/generation/retrieval không đọc LangfuseSettings.
- Ba biến LANGFUSE_* nằm block APP của `.env.example` root; duy nhất một cặp env (deploy 7).

## 7. Triển khai (`observability/` + `deploy/docker-compose.observe.yml`)

- Chỉ quan sát API production/end-user; compose observe riêng ở root observability, cùng host.
- Observe tạo network cố định legal-qa-observe; deploy override khai external, nối API.
- Override: LANGFUSE_BASE_URL=http://langfuse-web:3000, LANGFUSE_TRACING_ENVIRONMENT=production.
- deploy/up.sh tự ghép khi network tồn tại; Prometheus scrape api:8000 với env=production.
- Bật `./observability/up.sh` → tạo project/key bằng UI → điền LANGFUSE_PUBLIC_KEY/SECRET_KEY ở root env → chạy `./deploy/up.sh`.
- Observe script dùng compose --env-file ../.env; down giữ volume.

## 8. Module

- tracing.py: singleton/span/generation; turn_trace.py: RuntimeVersions/update_turn_trace; metrics.py: record_turn/instrument_app/custom metrics.
- Orchestrator bọc cache/admission/retrieve/generate; module LLM bọc generation observation.
- API routes root/metrics; app instrument/flush shutdown.

## 9. Nghiệm thu thủ công

- **Ca 1–2:** mỗi lượt một chat_turn, cây đúng 4.3; guardrail/condense là hai con root.
- **Ca 3:** API /metrics có custom/HTTP metrics; localhost:9092/targets báo legal-qa-api UP.
- **Ca 4:** trace đủ 4.5, identity/session lọc đúng; lỗi/refusal/cache hit có trace riêng.
- **Ca 5:** key trống/Langfuse tắt giữa chừng vẫn trả lời, không warning lặp.
- **Ca 6–7:** tổng observe khoảng 3.4GB + rerank không OOM; shutdown flush không mất lượt cuối.

## 10. Rủi ro / điểm mở

- OTel gather phải nghiệm thu 9.2; sai thì copy_context().run quanh coroutine.
- Stack khoảng 3.4GB, cân nhắc RAM khi load model.
- OTel dependencies cần uv sync/pip-audit; production/eval xung đột OpenAI được tách bằng tool.uv.conflicts.
