# production-legal-qa-rag

RAG chatbot hỏi-đáp pháp luật Việt Nam, thiết kế theo hướng sát production nhưng gọn
(dự án cá nhân, public qua Cloudflare Tunnel). Python 3.14, quản lý bằng `uv`.

## Cấu trúc thư mục

```
src/production_legal_qa_rag/   Toàn bộ source code import được (packages theo nghiệp vụ)
tests/                         Test, đặt tên test_<package>_<phần>.py
tools/                         CLI (Typer) chạy từng bước pipeline độc lập, vd. tools/chunk_documents.py
alembic/                       Migration schema Postgres (chỉ còn chatlog — gỡ cùng chatlog, xem Tiến độ)
data/                          raw -> markdown -> chunks -> embeddings, bm25/ cho sparse index
models/                        Model tải local, vd. vietnamese-reranker (chạy in-process, không host tách rời)
deploy/                        Docker compose production (chỉ phục vụ end-user) + docs (deploy/deploy_spec.md), scripts/ (backup, reset cache); không phải code import được
dev/                           observability/ (Langfuse/Prometheus/Grafana compose, dev only) — tách khỏi deploy/ (2026-09-29) vì không phục vụ end-user
docs/                          Tài liệu tổng quan hệ thống (luồng xử lý 1 câu hỏi, kiến trúc)
.claude/                       Cấu hình Claude Code cho project: agents/, skills/, settings.json
```

Mỗi package trong `src/production_legal_qa_rag/` có một `<package>_spec.md` nằm ngay
cạnh nó, là nguồn sự thật cho thiết kế/quyết định của package đó — đọc trước khi sửa code
trong package tương ứng.

## Package map (theo thứ tự pipeline)

| Package           | Vai trò                                                                                                 | Spec                                                                                 |
| ----------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `formatting/`   | `.docx` pháp luật → Markdown có cấu trúc, giữ vị trí pháp lý (Điều/Khoản/Điểm)         | [formatting_spec.md](src/production_legal_qa_rag/formatting/formatting_spec.md)       |
| `chunking/`     | Markdown → chunk tự đủ nghĩa, sẵn sàng embedding                                                  | [chunking_spec.md](src/production_legal_qa_rag/chunking/chunking_spec.md)             |
| `embedding/`    | Chunk → vector, giữ citation cho bước sinh câu trả lời                                            | [embedding_spec.md](src/production_legal_qa_rag/embedding/embedding_spec.md)          |
| `retrieval/`    | Câu hỏi → tập`RetrievedChunk` liên quan (HyDE, hybrid, RRF, MMR, rerank local GPU)                | [retrieval_spec.md](src/production_legal_qa_rag/retrieval/retrieval_spec.md)          |
| `generation/`   | `RetrievedChunk` đã rerank → câu trả lời có citation, đã kiểm chứng                         | [generation_spec.md](src/production_legal_qa_rag/generation/generation_spec.md)       |
| `conversation/` | Điều phối 1 lượt hỏi đáp: condense → guardrail → cache → admission → retrieve → generate    | [conversation_spec.md](src/production_legal_qa_rag/conversation/conversation_spec.md) |
| `cache/`        | Cache câu trả lời & kết quả retrieval bằng Redis, single-flight                                    | [cache_spec.md](src/production_legal_qa_rag/cache/cache_spec.md)                      |
| `chatlog/`      | Ghi mỗi lượt hỏi-đáp vào Postgres (**sẽ gỡ**, Langfuse thay thế — xem Tiến độ) | [chatlog_spec.md](src/production_legal_qa_rag/chatlog/chatlog_spec.md)                |
| `api/`          | FastAPI (OpenAI-compatible) + OpenWebUI + Redis + Postgres, spec tổng toàn hệ thống                  | [api_spec.md](src/production_legal_qa_rag/api/api_spec.md)                            |
| `observability/` | Langfuse trace 1 lượt hỏi + Prometheus `/metrics` cho `api`                                       | [observability_spec.md](src/production_legal_qa_rag/observability/observability_spec.md) |
| `evaluation/`   | Đánh giá bằng RAGAS: Phase 1 sinh golden testset (đã merge); Phase 2 chạy pipeline thật + chấm điểm (đã thiết kế, chưa implement) | [evaluation_spec.md](src/production_legal_qa_rag/evaluation/evaluation_spec.md)       |

## Tiến độ

Trạng thái tại **2026-09-29**.

**Đã xong**
- Pipeline `formatting/` → `chunking/` → `embedding/` → `retrieval/` → `generation/` →
  `conversation/` → `cache/` → `chatlog/` → `api/`: có spec, có test, chatbot chạy end-to-end
  qua API OpenAI-compatible + OpenWebUI + Redis + Postgres.
- **Deploy production** (`deploy/`, chỉ phục vụ end-user): entrypoint `deploy/up.sh` (tự dò GPU
  NVIDIA), public qua Cloudflare quick tunnel, đã nghiệm thu thật. Lưu ý vận hành: `api` cần
  `mem_limit: 3g`; Groq giới hạn rate limit theo (tài khoản, model), không theo API key.
- **Chính sách model/key LLM** (`conversation_spec.md` mục 12.1, PR #58): generation `gpt-oss-120b`
  xoay key 3 ⇄ 4; condense/HyDE/Judge `gpt-oss-20b`; guardrail `gpt-oss-safeguard-20b`;
  throttle theo bucket `(model, key)`.
- **Observability code** (PR #61): Langfuse trace + Prometheus/Grafana; `deploy/docker-compose.observe.yml`
  nối `api` production vào stack observe (`dev/observability/`), `up.sh` tự ghép khi stack đang
  chạy. Prometheus chỉ scrape `api` production (`env=production`).
- **Hạ tầng gọn:** đúng 1 cặp `.env`/`.env.example` ở repo root (3 block APP/DEPLOY/OBSERVABILITY,
  prefix `DEPLOY_`/`OBS_` cho biến trùng tên); `deploy/` chỉ chứa thứ phục vụ end-user, stack
  dev nằm ở `dev/observability/` (chạy: `docker compose -f dev/observability/docker-compose.yml up -d`,
  cần symlink `dev/observability/.env` → `../../.env`).
- **Evaluation Phase 1** (PR #55, #60): sinh golden testset theo đơn vị, có checkpoint, chạy tiếp
  nhiều ngày (`tools/generate_testset.py`).

**Đã chốt thiết kế, chưa implement**
- **Gỡ `chatlog/`** (quyết định 2026-09-29): Langfuse là nơi duy nhất lưu nhật ký lượt hỏi-đáp
  (`observability_spec.md` mục 4.5, có bảng ánh xạ trường `chat_turns` → trace). Gỡ kèm `alembic/`,
  `sqlalchemy`/`asyncpg`, `tools/chatlog.py`, `tools/purge_chatlog.py`, biến `CHATLOG_*`, DB
  `chatbot` trong compose/initdb; test và các spec liên quan cập nhật theo. Dữ liệu `chat_turns`
  cũ trong Postgres production: người dùng tự drop. Hệ quả: Langfuse không chạy thì mất nhật ký
  lượt đó — stack observe nên chạy cùng production. Langfuse giữ self-host (riêng tư), không dùng cloud.
- **Evaluation Phase 2** (`evaluation_spec.md` mục 11): 6 stage (HyDE → embed → retrieve MMR bật/tắt
  → chấm retrieval → generation → chấm câu trả lời), file JSONL trung gian, resume theo `case_id`;
  generation chạy nguyên `GenerationPipeline` (phương án B), rải 6 key Groq; pilot `--limit 10–20`
  trước khi chạy full. Cần `precomputed` ở `RetrievalPipeline.retrieve` (`retrieval_spec.md` mục 2).
  Spec chưa commit.

**Đang dở / chờ người dùng**
- Nghiệm thu thủ công observe (bật stack, tạo project + key Langfuse, điền `LANGFUSE_*` vào
  `.env`, `./deploy/up.sh`, xem trace/metrics thật) — chưa làm xong.
- Golden testset: tiến trình sinh và duyệt tay chưa hoàn tất (`data/eval/*`).

Roadmap tiếp theo (thứ tự đề xuất):

1. Implement gỡ chatlog (`/develop-cycle` trên `observability_spec.md`, branch mới).
2. Nghiệm thu observe trên production thật.
3. Hoàn tất golden testset → implement Evaluation Phase 2 → lấy mẫu Q&A thật từ Langfuse.
4. **CD**: GitHub Actions build + push image lên GHCR (không SSH tự động vào máy nhà) — chưa
   brainstorm chi tiết. Tách observability sang VM riêng: chưa chốt.

## Quy trình phát triển (`.claude/`)

Project có agent/skill riêng cho vòng đời spec → implement → test → review:

| Agent         | Vai trò                                                                                                                                                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `architect` | Brainstorm và chốt`*_spec.md` cùng người dùng trước khi implement (logic/workflow lẫn lựa chọn công nghệ)                                                                                            |
| `developer` | Implement code từ spec đã chốt; chỉ commit local, không push/mở PR                                                                                                                                           |
| `tester`    | Viết Unit/Integration/Data-schema test theo spec; đảm nhiệm push + mở PR để CI chạy, tổng hợp feedback                                                                                                    |
| `reviewer`  | Review kiến trúc/logic/security/scalability đối chiếu spec + skill`coding-convention`; chạy local sau khi CI pass, PASS thì comment kết luận lên PR — không tự merge, người dùng merge thủ công |

Skill `develop-cycle` (`.claude/skills/develop-cycle/`) chạy trọn vòng lặp
developer → tester → reviewer cho một spec cụ thể (`argument-hint: <đường dẫn spec.md> <tên branch>`). Skill `coding-convention` (`.claude/skills/coding-convention/`) là quy ước
coding chuẩn production dùng chung (kiến trúc thư mục, naming, format, docstring, công cụ
pydantic/typer/ruff) — `reviewer` đối chiếu theo skill này.

`.claude/settings.json` allowlist các lệnh `git`/`gh`/đọc-file an toàn (status, log, diff,
pr view/list/diff/checks, grep/rg/find/cat/ls...) và deny các thao tác phá hoại
(`push --force`, `reset --hard`, `git clean`, `rm`, đọc `.env`, ...).

## Lệnh dev

```bash
uv sync                    # cài dependency (dev group gồm pytest, ruff, mypy)
uv run pytest              # chạy test; -m "not slow" để bỏ test nạp model thật
uv run ruff format .
uv run ruff check .
uv run mypy .
uv run alembic upgrade head   # bỏ khi gỡ chatlog
```
