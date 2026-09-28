# production-legal-qa-rag

RAG chatbot hỏi-đáp pháp luật Việt Nam, thiết kế theo hướng sát production nhưng gọn
(dự án cá nhân, public qua Cloudflare Tunnel). Python 3.14, quản lý bằng `uv`.

## Cấu trúc thư mục

```
src/production_legal_qa_rag/   Toàn bộ source code import được (packages theo nghiệp vụ)
tests/                         Test, đặt tên test_<package>_<phần>.py
tools/                         CLI (Typer) chạy từng bước pipeline độc lập, vd. tools/chunk_documents.py
alembic/                       Migration schema Postgres (chatlog, ...)
data/                          raw -> markdown -> chunks -> embeddings, bm25/ cho sparse index
models/                        Model tải local, vd. vietnamese-reranker (chạy in-process, không host tách rời)
deploy/                        Docker compose production + docs (deploy/deploy_spec.md); dev/ (compose dev + observability), scripts/ (backup, reset cache); không phải code import được
docs/                          Tài liệu tổng quan hệ thống (vd. online_flow.md — activity diagram 1 câu hỏi)
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
| `chatlog/`      | Ghi mỗi lượt hỏi-đáp vào Postgres để quan sát/đánh giá (không phải lịch sử hội thoại) | [chatlog_spec.md](src/production_legal_qa_rag/chatlog/chatlog_spec.md)                |
| `api/`          | FastAPI (OpenAI-compatible) + OpenWebUI + Redis + Postgres, spec tổng toàn hệ thống                  | [api_spec.md](src/production_legal_qa_rag/api/api_spec.md)                            |
| `evaluation/`   | Đánh giá bằng RAGAS: Phase 1 sinh golden testset tổng hợp từ corpus (Phase 2 chạy eval — chưa làm)  | [evaluation_spec.md](src/production_legal_qa_rag/evaluation/evaluation_spec.md)       |

## Tiến độ

Toàn bộ 9 package trong pipeline (`formatting/` → `chunking/` → `embedding/` →
`retrieval/` → `generation/` → `conversation/` → `cache/` → `chatlog/` → `api/`) đã
implement xong, có spec, có test — chatbot chạy được end-to-end qua API OpenAI-compatible
(#53), kèm OpenWebUI + Redis + Postgres.

**Deploy production đã xong (2026-09-27)** — `deploy/docker-compose.yml` (deploy_spec.md
mục 4) đã tạo, entrypoint duy nhất `deploy/up.sh` (tự dò GPU NVIDIA, build đúng biến thể
torch — `cpu` mặc định, `cu126` trở lên cho GPU vì `cu121`/`cu124` không có wheel Python
3.14 — rồi `up -d`). Đã nghiệm thu thật: public qua Cloudflare quick tunnel, end-user
hỏi-đáp multi-turn thành công (citation, Evidence Judge chạy đúng). Bài học vận hành đã ghi
vào deploy_spec.md: `api` cần `mem_limit: 3g` (1.5g cũ bị OOM-killer giết ngay lượt hỏi
retrieval+rerank đầu); Groq giới hạn rate limit theo (tài khoản, model) chứ không theo API
key — key chỉ tách được ngân sách thật nếu lấy từ tài khoản Groq khác.

**Cập nhật 2026-09-28:**

- **`deploy/` tổ chức lại** (#57): compose dev + `observability/` chuyển vào `deploy/dev/`
  (`deploy/dev/docker-compose.yml` vẫn phục vụ dev cục bộ: Redis + Postgres), `backup.sh` +
  `reset_cache.sh` vào `deploy/scripts/`. Dockerfile, compose và 2 file `.env.example` đã rút
  gọn; hướng dẫn cấu hình chi tiết nằm trong `README.md`.
- **Chốt spec model + key LLM để tránh rate limit** (chưa implement code — `conversation_spec.md`
  mục 12.1): generation giữ `gpt-oss-120b` xoay vòng `GROQ_API_KEY_3` ⇄ `_4`; condense/HyDE/
  Judge/guardrail dùng model 20b trên key 1, 2 (Judge riêng key 2), có throttle cửa sổ trượt
  dùng chung (`retrieval/llm_throttle.py`). `GROQ_JUDGE_API_KEY` bỏ, thay bằng
  `GROQ_API_KEY_4`. Việc còn lại cho developer: `config.py` (+`HydeSettings`,
  `ThrottleSettings`), `llm_throttle.py`, đo lại `MIN_RERANK_SCORE` với HyDE 20b, chạy lại bộ
  ca Judge 20b, bump `PROMPT_VERSION`; `.env.example` hiện mô tả trạng thái đích.
- **Evaluation Phase 1 đã merge** (#55): `evaluation/` + `tools/generate_testset.py` sinh
  golden testset (RAGAS, round-robin 3 tài khoản Groq). Chưa ghi nhận đã chạy sinh testset
  và duyệt tay; Phase 2 (chạy pipeline thật, tính metric) chưa làm.

Roadmap tiếp theo (đã chốt, xem thứ tự — không đảo ngược trừ khi có quyết định mới):

1. **Đánh giá chất lượng bằng RAGAS** — Phase 1 (code sinh golden testset tổng hợp) đã
   merge; còn chạy sinh + duyệt tay testset, rồi Phase 2 (chạy pipeline thật, tính metric)
   và lấy mẫu Q&A thật từ bảng `chat_turns` (chatlog) khi dữ liệu đã tích lũy đủ.
   **Trước đó/song song:** implement chính sách model + key LLM ở trên.
2. **CI/CD** (GitHub Actions cho CD) + **tracking/tracing/observability** (Prometheus +
   Grafana + Langfuse — Langfuse trace từng bước condense → retrieve → rerank → generate,
   Prometheus/Grafana cho metrics/ops thời gian thực). Quyết định gần nhất
   (2026-09-26): chạy Langfuse self-host + Prometheus + Grafana **trên local trước**;
   việc tách hạ tầng sang VM free-tier riêng (vd. Oracle Cloud, cho k8s/observability)
   để tính sau, chưa chốt. `deploy/dev/observability/` đã có compose khung cho
   Prometheus/Grafana.

Giữ nguyên quyết định: `chatlog` (Postgres tự host) vẫn là nguồn dữ liệu chính chủ, không
thay bằng Langfuse cloud (lý do riêng tư + cần query SQL trực tiếp lên schema nghiệp vụ).

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
uv run alembic upgrade head
```
