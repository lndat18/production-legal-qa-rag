# Observability — Langfuse tracing + Prometheus/Grafana metrics

## 1. Mục tiêu & phạm vi

Roadmap (README mục "Tiến độ", memory `roadmap-post-chatlog`) đã chốt: sau RAGAS Phase 1,
làm CI/CD + observability, chạy **Langfuse self-host + Prometheus + Grafana trên local
trước** (quyết định 2026-09-26). Spec này chỉ làm phần **observability**; CD (GitHub
Actions build/push image) brainstorm ở spec riêng sau, không thuộc phạm vi ở đây.

Hai công cụ, hai câu hỏi khác nhau, không chồng chéo:

- **Langfuse** (đã có khung compose ở `dev/observability/`) — trả lời "1 lượt hỏi cụ
  thể đã chạy qua những bước nào, bước nào chậm/lỗi, model nào dùng key nào, tốn bao nhiêu
  token" → trace chi tiết, xem trên UI Langfuse.
- **Prometheus/Grafana** — trả lời "hệ thống đang khoẻ không ngay lúc này" → vài chỉ số thô
  mức vận hành (tỉ lệ lỗi/từ chối, độ trễ, HTTP), xem trên dashboard.

**Trong phạm vi:**

- Trace 1 lượt hỏi (`chat_turn`) trên Langfuse: 1 trace/lượt, span/generation con cho từng
  bước chính (guardrail, condense, retrieve [gồm HyDE], admission, generate [gồm Judge]) —
  mục 4.
- **Gỡ hẳn package `chatlog/`** (chốt 2026-09-29): Langfuse trở thành nơi duy nhất lưu nhật
  ký từng lượt hỏi-đáp thay cho bảng `chat_turns` — trace phải mang đủ các trường chatlog từng
  lưu (mục 4.5). Không còn cột/migration `langfuse_trace_id`.
- `GET /metrics` (Prometheus) ở `api`: mặc định của `prometheus-fastapi-instrumentator`
  (HTTP) + 2 custom metric ở mức lượt hỏi (`chat_turns_total`, `turn_latency_seconds`,
  `time_to_first_token_seconds`) — mục 5.
- Wiring dev: join network để Prometheus scrape được `api`, bật scrape target trong
  `prometheus.yml` (đã có sẵn khung, đang comment) — mục 7.
- Gỡ toàn bộ hạ tầng chatlog (package, bảng, Alembic, CLI, DB `chatbot`) — mục 4.5.

**Ngoài phạm vi (chốt rõ để không lấn sang CD):**

- CD (GitHub Actions build + push image lên registry) — spec riêng sau, đã có hướng chọn sơ
  bộ (CI mở rộng, không SSH tự động vào máy nhà) nhưng chưa brainstorm chi tiết.
- Đưa observability stack vào **production** (`deploy/docker-compose.yml`, máy cá nhân +
  Cloudflare Tunnel): giữ nguyên quyết định 2026-09-26, chỉ chạy dev. Code instrumentation
  viết ra vẫn chạy được ở prod (no-op nếu thiếu env, mục 4.1) nhưng compose prod không đổi.
- Tự thiết kế dashboard Grafana bằng JSON provisioning — datasource Prometheus đã provision
  sẵn; panel cụ thể tự tạo tay trên UI khi cần, không đáng làm as-code ở quy mô 1 người dùng.
- Alerting (Alertmanager, ngưỡng cảnh báo) — chưa cần khi chưa có traffic ổn định.
- Trace cho các bước offline/batch (`formatting/`, `chunking/`, `embedding/`,
  `evaluation/`) — chỉ trace đường online (1 lượt hỏi qua `conversation/`).
- Đổi sang Langfuse Cloud — giữ self-host, cùng lý do đã chốt trước đây với `chatlog` (riêng tư, dữ
  liệu là câu hỏi/trả lời pháp lý có thể chứa tình huống cá nhân, không rời khỏi máy).

**Tiêu chí quan trọng nhất:** như nhật ký `chatlog` trước đây, instrumentation **không bao giờ** làm
chậm hay làm hỏng câu trả lời — thiếu/lỗi Langfuse hoặc Prometheus thì chatbot vẫn chạy
bình thường, chỉ mất khả năng quan sát.

## 2. Input & Output

**Input:** các bước đã tồn tại trong `conversation/`, `generation/`, `retrieval/` — spec
này không đổi logic nghiệp vụ, chỉ bọc thêm span/generation quanh lời gọi đã có.

**Output:**

- Trace cây trên Langfuse UI (`http://localhost:3001`, dev) cho mỗi lượt hỏi.
- `GET /metrics` (Prometheus text format) trên `api`.
- Mỗi trace `chat_turn` chứa đủ dữ liệu nhật ký của lượt hỏi (mục 4.5) — thay `chat_turns`.

## 3. Công cụ & công nghệ

| Việc | Công cụ | Ghi chú |
| --- | --- | --- |
| Tracing LLM | `langfuse` Python SDK **v4** (`get_client()`, OTel-based) | Không dùng v3: SDK v3 bị Langfuse cutover 16/11/2026. Platform self-host đã pin `:4` trong compose khung — khớp yêu cầu tối thiểu (self-host ≥ 3.63.0). |
| Metrics HTTP | `prometheus-fastapi-instrumentator` | Chủ động maintain, hỗ trợ async instrumentation function; expose `/metrics` qua `Instrumentator().instrument(app).expose(app)`. |
| Metrics custom | `prometheus_client` (dependency của instrumentator, dùng trực tiếp `Counter`/`Histogram`) | Không thêm thư viện mới. |
| Dashboard | Grafana (đã có compose + datasource provisioning) | Không đổi. |

Dependency mới (`pyproject.toml`, nhóm `production`): `langfuse`, `prometheus-fastapi-instrumentator`. Chạy `pip-audit` lại (CI đã có bước này, `continue-on-error: true`).

## 4. Trace Langfuse (`observability/tracing.py`)

### 4.1 Client & fail-safe

```python
from langfuse import Langfuse, get_client


def get_langfuse_client() -> Langfuse:
    """Singleton Langfuse client; SDK tự no-op nếu thiếu public/secret key."""
    return get_client()
```

SDK v4 đọc `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_BASE_URL` từ env; thiếu
key → client rơi vào **disabled mode tự động**, mọi `start_as_current_span`/`generation`
trở thành no-op, không raise, không log rác. Đây là toàn bộ cơ chế "fail-safe" đã chọn —
**không** thêm biến `OBSERVABILITY_ENABLED` riêng. Production (`deploy/docker-compose.yml`)
không set `LANGFUSE_*` → tự động tắt, không cần đổi gì thêm ở đó.

### 4.2 Hai loại observation

- **`span(name, *, user_id=None, session_id=None, **kwargs)`**: bước không gọi LLM (cache
  lookup, container bao ngoài). `user_id`/`session_id` chỉ dùng cho root span (mục 4.4):
  `span` bọc `propagate_attributes(user_id=..., session_id=...)` của SDK để đặt chúng ở cấp
  trace và lan xuống mọi observation con.
- **`generation(name, *, model, **kwargs)`**: bước gọi LLM — Langfuse hiển thị
  riêng model/usage/cost cho loại này, cần cho đúng các bước dùng `conversation_spec.md`
  mục 12.1 (nhiều model/key khác nhau: `gpt-oss-120b` key 3/4, `gpt-oss-20b` key 1/2,
  `gpt-oss-safeguard-20b`).

```python
@contextmanager
def span(
    name: str,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    **kwargs: Any,
) -> Iterator[Any]:
    with get_langfuse_client().start_as_current_observation(
        name=name, as_type="span", **kwargs
    ) as s:
        if user_id is None and session_id is None:
            yield s
            return
        with propagate_attributes(user_id=user_id, session_id=session_id):
            yield s


@contextmanager
def generation(name: str, *, model: str, **kwargs: Any) -> Iterator[Any]:
    with get_langfuse_client().start_as_current_observation(
        name=name, as_type="generation", model=model, **kwargs
    ) as g:
        yield g
```

`observability/` là package hạ tầng cross-cutting (giống logging) — **được phép import
trực tiếp** từ bất kỳ module nào gọi LLM, không đi qua tầng điều phối như
`cache/`. OTel context tự propagate qua `await`/`asyncio.gather` (context được copy khi
tạo `Task`), nên span/generation tạo bên trong các module dưới đây tự lồng đúng dưới span
cha đang active — không cần truyền tay đối tượng span qua tham số hàm.

### 4.3 Cây trace mục tiêu (1 lượt hỏi)

```
chat_turn                                            (root — api/routes.py)
├── guardrail            [generation]                (generation/guardrail.py: InputGuardrail.check_input)
├── condense             [generation]                (conversation/condenser.py: QueryCondenser.condense)
├── cache_lookup         [span]                       (conversation/orchestrator.py: answer_cache.get)
└── admission            [span]                       (conversation/orchestrator.py: quanh admission.slot(...))
    ├── retrieve         [span]                       (conversation/orchestrator.py: quanh retrieve())
    │   └── hyde         [generation]                (retrieval/hyde.py: HydeGenerator)
    └── generate         [span]                       (conversation/orchestrator.py: quanh generation.generate())
        ├── answer       [generation]                (generation/generator.py: AnswerGenerator/generate())
        └── judge        [generation]                (generation/judge.py: EvidenceJudge.judge)
```

`guardrail` và `condense` chạy song song qua `asyncio.gather` (mục 7 `conversation_spec.md`)
— cả hai vẫn là con trực tiếp của `chat_turn` (không lồng nhau); nghiệm thu mục 8.6 xác
nhận không bị lẫn ngữ cảnh. Cache hit (`answer_cache` trả sớm) thì `admission`/`retrieve`/
`generate` không được tạo — khớp hành vi `status(retrieval|generation)` chỉ phát khi thực
sự chạy.

Mỗi `generation` gắn `metadata={"key_bucket": ...}` lấy từ `retrieval/llm_throttle.py`
(bucket `(model, key)` đã có sẵn theo mục 12.1) — mục đích: nhìn trực quan trên Langfuse UI
model/key nào đang bị dùng nhiều, hỗ trợ tinh chỉnh chính sách rate limit sau này, **không**
tạo thêm Prometheus metric trùng lặp cho việc này (mục 5 chỉ giữ Prometheus ở mức thô).

`span`/`generation` cập nhật `output`/`metadata` tối thiểu: `verdict` (guardrail),
`standalone_query` (condense), `num_chunks` + `chunk_ids` + `top_rerank_score` (retrieve),
`usage` (answer, judge — Langfuse tự tính cost nếu model có bảng giá, không bắt buộc).
Nội dung lưu chỉ gồm những gì chatlog từng lưu (câu hỏi/trả lời/chunk id; self-host, cùng
mức riêng tư — mục 1 "Ngoài phạm vi").

### 4.4 Root trace (`api/routes.py`)

Root span bọc quanh đúng đoạn stream 1 lượt hỏi (`api_spec.md` mục 7/12), trong
`stream_chat_turn` (`api/routes.py`):

```python
with tracing.span(
    "chat_turn",
    input=root_query,
    metadata={"request_id": context.request_id},
    user_id=context.user_id,
    session_id=context.chat_id,
) as root_span:
    try:
        async for event in orchestrator.stream(messages, context, trace):
            yield event
    except asyncio.CancelledError:
        trace.outcome = "client_disconnected"
        raise
    finally:
        update_turn_trace(root_span, trace, request_id=..., versions=...)  # mục 4.5
# finally ngoài cùng: metrics.record_turn(trace)
```

- `user_id` / `session_id` (= `chat_id` của OpenWebUI) đặt ở cấp trace để Langfuse UI lọc
  theo người dùng và gom các lượt cùng cuộc hội thoại (dùng thuộc tính chuẩn của Langfuse
  qua `langfuse.propagate_attributes`, không phải metadata tự chế). `chat_id`/`user_id`
  thiếu thì bỏ trống, không bịa giá trị.
- `TurnTrace.langfuse_trace_id` và `tracing.current_trace_id()` đã **xoá** (không còn bảng
  nào để đối chiếu).
- Lifespan (`api/app.py`) gọi `get_langfuse_client().flush()` lúc shutdown (chặn tối đa vài
  giây) để không mất trace của các lượt hỏi cuối cùng trước khi container dừng.

### 4.5 Gỡ `chatlog/`: trace thay bảng `chat_turns` (chốt 2026-09-29)

> **Trạng thái: đã implement (2026-09-29)** — `chatlog/`, `alembic/`, CLI và dependency liên
> quan đã gỡ; `observability/turn_trace.py::update_turn_trace` gắn output/metadata/tags cuối
> lượt lên root span. Chi tiết SDK v4 (đã kiểm tra trên bản cài 4.15.x):
> `user_id`/`session_id` đặt bằng `propagate_attributes` (bọc trong `tracing.span`); `output`
> và metadata là `.update(output=..., metadata=...)` trên root observation (dict được SDK
> flatten thành `langfuse.observation.metadata.<key>`, list/dict lồng nhau serialize JSON);
> `tags` chỉ biết cuối lượt nên đặt bằng `propagate_attributes(tags=[...])` ngay lúc root span
> còn là span active (SDK ghi tag lên span hiện tại). `set_trace_io` bị SDK đánh dấu
> deprecated nên không dùng — input/output của observation gốc đã là input/output của trace.
> `prompt_version`/`corpus_version`/`model_name` do `RuntimeVersions` (`turn_trace.py`) giữ,
> tạo 1 lần ở lifespan, lưu `app.state.runtime_versions`.

Lý do: chỉ giữ 1 nơi lưu nhật ký (Langfuse), tránh 2 nguồn trùng nội dung; dữ liệu hỏi-đáp
thật để chấm RAGAS phase 2 lấy từ Langfuse (API/export) thay vì query SQL `chat_turns`.

**Cập nhật cuối lượt trên root span `chat_turn`** (đúng điểm `finally` cũ ghi chatlog, dùng
`TurnTrace` đã điền đủ) — mọi trường chatlog từng có phải còn truy cập được trên trace:

| Trường chatlog cũ | Vị trí trên Langfuse |
| --- | --- |
| `raw_query` | `input` của trace |
| `answer_text` | `output` của trace (rỗng nếu từ chối/lỗi) |
| `user_id`, `chat_id` | `user_id`, `session_id` của trace |
| `request_id` | metadata |
| `standalone_query`, `verdict`, `chunk_ids` | đã có ở span con (mục 4.3), lặp thêm vào metadata root để lọc |
| `outcome` (`answered`/`refused`/`error`/`client_disconnected`), `error_code`, `cache_status`, `citations`, `warnings` | metadata root |
| `usage` (token) | có ở generation con; tổng lượt thì metadata root nếu có sẵn trên `TurnTrace` |
| `time_to_first_token_ms`, `latency_ms` | metadata root (latency còn có sẵn từ span) |
| `prompt_version`, `corpus_version`, `model_name` | metadata root — để so trước/sau khi đổi cấu hình |
| Lượt lỗi / từ chối / cache hit / ngắt kết nối | vẫn phải có đúng 1 trace (không được mất vì đi nhánh sớm) — `outcome` trong metadata phân biệt |

Đặt `tags` trên trace (`outcome:<x>`, `cache:<y>`) để lọc nhanh trên UI Langfuse.

**Đã gỡ khỏi repo (developer làm, tester cập nhật test):**

- Package `src/production_legal_qa_rag/chatlog/` (models, tables, repository, `chatlog_spec.md`).
- `ChatLogTaskManager` + `ChatLogRepository` protocol trong `api/routes.py`; engine/repo trong
  lifespan `api/app.py` (giữ `flush()` Langfuse); `metrics.record_turn` giữ, gọi ở chỗ cũ.
  `GET /readyz` chỉ còn ping Redis (bỏ ping Postgres vì `api` không còn engine DB).
- `alembic/` toàn bộ (thư mục `alembic/`, cả `0001`, `0002`) và `alembic.ini` nếu có; dependency
  `alembic`, `sqlalchemy[asyncio]`, `asyncpg` trong `pyproject.toml` (+ `uv lock`).
- `tools/chatlog.py`, `tools/purge_chatlog.py`; biến `CHATLOG_DATABASE_URL`, `CHATLOG_RETENTION_DAYS`
  và `DatabaseSettings` trong `config.py`, `.env.example`, README.
- Compose production: `api` không còn dùng Postgres — bỏ `CHATLOG_DATABASE_URL` và
  `depends_on: postgres` khỏi service `api`; `deploy/initdb/001_create_databases.sh` chỉ tạo
  `openwebui`. Postgres vẫn giữ cho OpenWebUI. (DB `chatbot` cũ trong volume đã có không tự
  biến mất — người dùng drop tay, ngoài phạm vi code.)
- Test `tests/test_chatlog.py`; các test khác phụ thuộc chatlog (`test_api_routes.py`,
  `test_deploy_compose.py`, `test_deploy_migration.py`, `test_observability_metrics.py`) sửa
  hoặc xoá phần liên quan chatlog.
- Doc: `api_spec.md`, `conversation_spec.md`, `cache_spec.md`, `deploy_spec.md`, `README.md`
  đã sửa. Phần người dùng tự cập nhật ngoài lượt develop đó đã xong (2026-09-29):
  `evaluation_spec.md`, `CLAUDE.md` (bảng package, roadmap: RAGAS lấy mẫu từ Langfuse); riêng
  `docs/online_flow.md` đã bị xoá, sơ đồ tổng quan nay là `docs/architecture.png`.

**Ràng buộc còn nguyên:** ghi trace không bao giờ chặn/làm hỏng câu trả lời (mục 1). Khác
chatlog: Langfuse không chạy thì **mất** nhật ký lượt đó (không có hàng đợi bền) — chấp nhận
được cho dự án cá nhân, và là lý do stack observe nên luôn chạy cùng production.

## 5. Metrics Prometheus (`observability/metrics.py`)

Chỉ 2 nhóm, cố tình **không** đo lặp lại những gì Langfuse đã trả lời tốt hơn (per-model/
key token, per-step latency chi tiết) — Prometheus/Grafana ở đây là "đèn báo sức khoẻ",
không phải nơi debug 1 lượt hỏi cụ thể:

1. **HTTP mặc định** của `prometheus-fastapi-instrumentator` (`http_requests_total`,
   `http_request_duration_seconds`, …) — instrument nguyên `api` app, không cấu hình thêm.
2. **Custom, mức lượt hỏi**, nguồn dữ liệu là `TurnTrace` đã điền đầy đủ (cùng điểm cuối lượt
   với việc cập nhật trace — không thêm import mới vào `conversation/`/`retrieval/`):

```python
from prometheus_client import Counter, Histogram

CHAT_TURNS_TOTAL = Counter(
    "chat_turns_total", "Tổng lượt hỏi", ["outcome", "cache_status"]
)
TURN_LATENCY_SECONDS = Histogram(
    "turn_latency_seconds",
    "Độ trễ toàn bộ 1 lượt hỏi (giây)",
    ["outcome"],
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 60, 120),
)
TIME_TO_FIRST_TOKEN_SECONDS = Histogram(
    "time_to_first_token_seconds",
    "Độ trễ tới token đầu tiên (giây)",
    buckets=(0.2, 0.5, 1, 2, 5, 10, 20),
)


def record_turn(trace: TurnTrace) -> None:
    """Bọc try/except ngay trong hàm — lỗi ghi metric không bao giờ lan ra
    ngoài (cùng tinh thần `update_turn_trace`, chỉ log warning)."""
    try:
        CHAT_TURNS_TOTAL.labels(
            outcome=trace.outcome, cache_status=trace.cache_status
        ).inc()
        TURN_LATENCY_SECONDS.labels(outcome=trace.outcome).observe(
            trace.latency_ms / 1000
        )
        if trace.time_to_first_token_ms is not None:
            TIME_TO_FIRST_TOKEN_SECONDS.observe(trace.time_to_first_token_ms / 1000)
    except Exception:
        _logger.warning("Không thể ghi metrics cho lượt hỏi")


def instrument_app(app: FastAPI) -> None:
    Instrumentator().instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False
    )
```

`instrument_app(app)` gọi 1 lần trong `create_app()` (`api/app.py`). `record_turn(trace)`
gọi ở đúng chỗ `finally` cuối lượt trong `api/routes.py` — cùng nguồn `TurnTrace` với
việc cập nhật trace (mục 4.5), không tính toán lại.

**Lưu ý route SSE:** `POST /v1/chat/completions` stream lâu (chờ token) — histogram HTTP
mặc định của instrumentator sẽ đo cả thời gian đó vào `http_request_duration_seconds`, số
sẽ trùng phần lớn với `turn_latency_seconds`. Chấp nhận trùng lặp này (đơn giản hơn loại
trừ route), 2 metric khác góc nhìn: 1 cái theo route HTTP, 1 cái theo `outcome` nghiệp vụ.

## 6. Config (`config.py`)

```python
class LangfuseSettings(BaseSettings):
    """Kết nối Langfuse self-host (observability_spec.md mục 4).

    Để trống public_key/secret_key -> SDK tự chuyển sang chế độ disabled
    (mục 4.1), không cần cờ bật/tắt riêng. `api` khi dev luôn chạy trực tiếp
    trên host (`uv run uvicorn`), không nằm cùng network Docker với
    `langfuse-web` — base_url mặc định trỏ vào cổng `langfuse-web` publish ra
    host (`127.0.0.1:3001:3000`), không phải tên service nội bộ Docker.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="LANGFUSE_", extra="ignore", env_ignore_empty=True
    )

    public_key: SecretStr | None = None
    secret_key: SecretStr | None = None
    base_url: str = "http://localhost:3001"
```

`api/`, `conversation/`, `generation/`, `retrieval/` không đọc `LangfuseSettings` trực
tiếp — package `observability/` tự đọc khi khởi tạo client (giống cách `RerankerSettings`
chỉ được đọc trong `retrieval/`). 3 biến `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/
`LANGFUSE_BASE_URL` nằm trong `.env.example` (root, block APP — duy nhất 1 cặp `.env`/
`.env.example` cho toàn project, chốt 2026-09-29, xem `deploy_spec.md` mục 7), để trống
mặc định.

## 7. Triển khai (`dev/observability/` + `deploy/docker-compose.observe.yml`, ngoài package)

Mục tiêu (chốt 2026-09-29): quan sát **traffic end-user thật** trên `api` production. Stack
observe (Langfuse/Prometheus/Grafana) là compose riêng ở `dev/observability/`, tách khỏi
`deploy/` (không phục vụ end-user); `api` production nối vào bằng 1 network chung:

- Network `legal-qa-observe` (tên cố định) do `dev/observability/docker-compose.yml` tạo,
  `langfuse-web` và `prometheus` join. `deploy/docker-compose.observe.yml` (override, cùng
  kiểu `docker-compose.gpu.yml`) khai network này `external` và thêm `api` vào, đồng thời
  ghi đè `LANGFUSE_BASE_URL=http://langfuse-web:3000` và
  `LANGFUSE_TRACING_ENVIRONMENT=production`. `deploy/up.sh` tự ghép override khi network
  đã tồn tại (stack observe đang chạy); chưa chạy thì bỏ qua, production dựng như cũ.
- `dev/observability/prometheus.yml`: scrape `api:8000` (label `env=production`). Không còn
  target cho `api` chạy trên host.
- `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`: điền vào `.env` root (không phải file riêng)
  — tạo tay qua Langfuse UI sau khi `docker compose -f dev/observability/docker-compose.yml
  up -d` lần đầu (cần symlink `dev/observability/.env` → `../../.env` để không gõ
  `--env-file`); bước thủ công 1 lần khi setup. Điền xong chạy lại `./deploy/up.sh`.
- Stack observe chạy trên cùng máy với production (chưa chốt tách VM riêng).

## 8. Module (`src/production_legal_qa_rag/observability/`)

| Module | Trách nhiệm |
| --- | --- |
| `tracing.py` | `get_langfuse_client`, `span` (+ `user_id`/`session_id`), `generation` |
| `turn_trace.py` | `RuntimeVersions`, `update_turn_trace` (output/metadata/tags cuối lượt, mục 4.5) |
| `metrics.py` | `CHAT_TURNS_TOTAL`, `TURN_LATENCY_SECONDS`, `TIME_TO_FIRST_TOKEN_SECONDS`, `record_turn`, `instrument_app` |

Nơi khác trong repo được sửa (không phải package mới, liệt kê để triển khai):

| File | Thay đổi |
| --- | --- |
| `conversation/models.py` | Đã bỏ `TurnTrace.langfuse_trace_id` (mục 4.4); docstring `TurnTrace` không còn nhắc chatlog |
| `conversation/orchestrator.py` | Bọc `cache_lookup`/`admission`/`retrieve`/`generate` bằng `tracing.span` |
| `generation/guardrail.py` | Bọc lời gọi LLM trong `check_input` bằng `tracing.generation("guardrail", ...)` |
| `conversation/condenser.py` | Bọc lời gọi LLM trong `condense` bằng `tracing.generation("condense", ...)` |
| `retrieval/hyde.py` | Bọc lời gọi LLM trong `HydeGenerator` bằng `tracing.generation("hyde", ...)` |
| `generation/generator.py` | Bọc lời gọi LLM chính bằng `tracing.generation("answer", ...)` |
| `generation/judge.py` | Bọc `EvidenceJudge.judge` bằng `tracing.generation("judge", ...)` |
| `api/routes.py` | Root span `chat_turn` (+ `user_id`/`session_id`, cập nhật output/metadata/tags cuối lượt theo mục 4.5), gọi `metrics.record_turn(trace)`; bỏ `ChatLogTaskManager` |
| `api/app.py` | `metrics.instrument_app(app)` lúc tạo app; `get_langfuse_client().flush()` lúc lifespan shutdown; bỏ engine/repository chatlog |
| `chatlog/`, `alembic/`, `tools/chatlog.py`, `tools/purge_chatlog.py` | **Xoá** (mục 4.5) |
| `pyproject.toml`, `.env.example` (root) | Dependency + biến môi trường mới |

## 9. Nghiệm thu thủ công

1. `docker compose -f dev/observability/docker-compose.yml up -d`
   (network join theo mục 7) → hỏi vài câu qua OpenWebUI/`tools/conversation.py`.
2. Mở Langfuse UI (`localhost:3001`) → mỗi lượt hỏi có đúng 1 trace `chat_turn`, cây span
   khớp mục 4.3 (đúng span nào chạy tuỳ nhánh cache hit/miss/refused/error).
3. `guardrail` + `condense` (chạy song song) đều là con trực tiếp của `chat_turn`, không bị
   lẫn ngữ cảnh hay tách thành trace riêng.
4. `curl` nội bộ `api:8000/metrics` → thấy `chat_turns_total`, `turn_latency_seconds`,
   `http_requests_total`. Prometheus (`localhost:9092/targets`) báo target `legal-qa-api`
   là `UP`.
5. Trace của 1 lượt hỏi có đủ dữ liệu nhật ký (mục 4.5): `input` = câu hỏi, `output` = câu
   trả lời, `user_id`/`session_id` đúng, metadata `outcome`/`cache_status`/`chunk_ids`…; các
   lượt từ chối, lỗi, cache hit đều vẫn có trace riêng với `outcome` tương ứng. Lọc theo
   `session_id` gom đúng các lượt cùng cuộc hội thoại.
6. Bỏ trống `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` (hoặc tắt hẳn stack observability)
   → chatbot vẫn trả lời bình thường, không lỗi, không log warning lặp lại.
7. Tắt Langfuse container giữa chừng (không tắt `api`) → chatbot vẫn trả lời (SDK tự buffer/
   retry rồi bỏ qua khi hết hạn, không chặn response).
8. Tổng `mem_limit` khi bật observability stack + load
   model rerank không làm máy OOM (bài học đã ghi `deploy_spec.md`; observability compose đã
   tự ghi chú ClickHouse+MinIO ~3.4GB).
9. Tắt server (`docker compose down`) → không có trace nào của các lượt hỏi cuối bị mất
   trên Langfuse UI (xác nhận `flush()` hoạt động).

## 10. Rủi ro / điểm mở

1. OTel context qua `asyncio.gather`/`Task` tự propagate về lý thuyết; nghiệm thu mục 9.3
   là bằng chứng thực tế duy nhất — nếu sai, cần `contextvars.copy_context().run(...)` thủ
   công quanh từng coroutine trong `gather`.
2. Compose observability nặng (~3.4GB, ClickHouse+MinIO) — không bật cùng lúc load model
   trên máy yếu; đã ghi chú sẵn trong file, không phải rủi ro mới.
3. 2 dependency mới (`langfuse`, `prometheus-fastapi-instrumentator`) kéo theo
   `opentelemetry-*` — kiểm tra xung đột version với dependency hiện có (`uv sync` +
   `pip-audit` sẽ báo nếu có), đặc biệt nhóm `production` vs `eval` đã có `[tool.uv.conflicts]`
   cho `openai` (`.github/workflows/ci.yml` đã ghi lý do).
4. Nếu sau này tách hạ tầng observability sang VM riêng (chưa chốt, memory
   `roadmap-post-chatlog`) thì chỉ cần đổi `LANGFUSE_BASE_URL`/network — code
   `observability/` không đổi.
5. Chưa đo tác động hiệu năng của việc bọc span/generation (thêm 1 vài ms mỗi lời gọi LLM
   do SDK serialize + gửi async) — nếu nghiệm thu mục 9 phát hiện chậm rõ rệt, cân nhắc batch
   export/tắt bớt granularity thay vì bỏ hẳn.
