# Đồng bộ với `CLAUDE.md`

Khi bắt đầu **mỗi session**, trước mọi công việc khác, hãy kiểm tra phần nội dung được đồng bộ bên dưới với nội dung hiện tại của [`CLAUDE.md`](CLAUDE.md). Nếu khác content, hãy map toàn bộ content từ `CLAUDE.md` sang phần được đồng bộ trong `AGENTS.md`, giữ nguyên mục **Đồng bộ với `CLAUDE.md`** này.

<!-- BEGIN CLAUDE.md SYNC -->
# production-legal-qa-rag

RAG chatbot hỏi-đáp pháp luật Việt Nam, thiết kế theo hướng sát production nhưng gọn
(dự án cá nhân, public qua Cloudflare Tunnel). Python 3.14, quản lý bằng `uv`.

## Cấu trúc thư mục

```
src/production_legal_qa_rag/   Toàn bộ source code import được (packages theo nghiệp vụ)
tests/                         Test, đặt tên test_<package>_<phần>.py
tools/                         CLI (Typer) chạy từng bước pipeline độc lập, vd. tools/chunk_documents.py
data/                          raw -> markdown -> chunks -> embeddings, bm25/ cho sparse index
models/                        Model tải local, vd. vietnamese-reranker (chạy in-process, không host tách rời)
deploy/                        Docker compose production (chỉ phục vụ end-user) + docs (deploy/deploy_spec.md), up.sh/down.sh/backup.sh/reset_cache.sh; không phải code import được
observability/                 Langfuse/Prometheus/Grafana compose (chạy cạnh production để quan sát end-user) — tách khỏi deploy/ (2026-09-29) vì không phục vụ end-user
docs/                          Tài liệu tổng quan hệ thống (luồng xử lý 1 câu hỏi, kiến trúc)
.claude/                       Cấu hình Claude Code cho project: agents/, skills/, settings.json
```

Mỗi package trong `src/production_legal_qa_rag/` có một `<package>_spec.md` nằm ngay
cạnh nó, là nguồn sự thật cho thiết kế/quyết định của package đó — đọc trước khi sửa code
trong package tương ứng. Các spec đã **cô đọng 2026-09-30 và rút gọn lần 2 2026-10-01 (chỉ giữ phần bắt buộc)** (bản đầy đủ ở git history) và **giữ nguyên số
mục** vì code/spec khác tham chiếu (`conversation_spec.md` mục 12.1, `observability_spec.md` mục 4.5,
`evaluation_spec.md` mục 3.x/4.x…) — đổi số mục là làm hỏng các tham chiếu đó.

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
| `api/`          | FastAPI (OpenAI-compatible) + OpenWebUI + Redis + Postgres (chỉ cho OpenWebUI), spec tổng toàn hệ thống | [api_spec.md](src/production_legal_qa_rag/api/api_spec.md)                            |
| `observability/` | Langfuse trace 1 lượt hỏi + Prometheus `/metrics` cho `api`                                       | [observability_spec.md](src/production_legal_qa_rag/observability/observability_spec.md) |
| `evaluation/`   | Đánh giá bằng RAGAS: Phase 1 sinh golden testset (đã merge); Phase 2 đã implement code trên feat/gen-testset-ans, chưa chạy đánh giá thật | [evaluation_spec.md](src/production_legal_qa_rag/evaluation/evaluation_spec.md)       |

## Tiến độ

Trạng thái tại **2026-09-30**. Tóm tắt: phần lõi (pipeline → API → deploy end-user) đã xong và nghiệm
thu; **CD chưa làm**; các mục còn lại (observe end-user, golden testset, Evaluation Phase 2) đang hoàn thiện.

**Đã xong**
- Pipeline `formatting/` → `chunking/` → `embedding/` → `retrieval/` → `generation/` →
  `conversation/` → `cache/` → `api/`: có spec, có test, chatbot chạy end-to-end qua API
  OpenAI-compatible + OpenWebUI + Redis (Postgres chỉ còn phục vụ OpenWebUI).
- **Deploy production** (`deploy/`, chỉ phục vụ end-user): `deploy/up.sh` (tự dò GPU NVIDIA) / `deploy/down.sh`,
  public qua Cloudflare quick tunnel; `backup.sh` (pg_dump `openwebui`), `reset_cache.sh` (xoá cache Redis) nằm cạnh
  `up.sh`. Nghiệm thu thật ngày 2026-09-27 và **chạy lại sau #57–#62 ngày 2026-09-30** (đăng ký user thường,
  hỏi-đáp có citation, cache hoạt động) — phần observe vẫn chưa nghiệm thu, xem "Đang dở". Lưu ý vận hành: `api` cần
  `mem_limit: 3g`; Groq giới hạn rate limit theo (tài khoản, model), không theo API key.
  Bài học 2026-09-30: (1) OpenWebUI lưu cấu hình vào DB (PersistentConfig) — giá trị chỉnh ở Admin Panel
  (vd. New Sign Ups) đè `ENABLE_SIGNUP` trong compose ở các lần khởi động sau; (2) `docker compose down` phải
  có `--env-file ../.env --profile '*'` (dùng `down.sh`), thiếu thì cloudflared sót và giữ network; (3) Dockerfile dùng
  `--mount=type=cache` cho cache wheel của uv để đổi dependency không tải lại torch; đừng `docker builder prune`;
  (4) Postgres chỉ 1 DB nên `POSTGRES_DB=openwebui`, đã bỏ `deploy/initdb/`.
- **Chính sách model/key LLM** (`conversation_spec.md` mục 12.1, PR #58): generation `gpt-oss-120b`
  xoay key 3 ⇄ 4; condense/HyDE/Judge `gpt-oss-20b`; guardrail `gpt-oss-safeguard-20b`;
  throttle theo bucket `(model, key)`.
- **Observability code** (PR #61): Langfuse trace + Prometheus/Grafana; `deploy/docker-compose.observe.yml`
  nối `api` production vào stack observe (`observability/`), `up.sh` tự ghép khi stack đang
  chạy. Prometheus chỉ scrape `api` production (`env=production`).
- **Gỡ `chatlog/`** (PR #62): Langfuse (self-host, riêng tư) là nơi duy nhất lưu nhật ký lượt hỏi-đáp
  (`observability_spec.md` mục 4.5, bảng ánh xạ trường `chat_turns` → trace). Đã bỏ `alembic/`,
  `sqlalchemy`/`asyncpg`, `tools/chatlog.py`, biến `CHATLOG_*`. Hệ quả: Langfuse không chạy thì mất
  nhật ký lượt đó — stack observe nên chạy cùng production. Dữ liệu `chat_turns` cũ trong Postgres
  production: người dùng tự drop. Dự án hiện tập trung vào phục vụ end-user + observe end-user.
- **Hạ tầng gọn:** đúng 1 cặp `.env`/`.env.example` ở repo root (3 block APP/DEPLOY/OBSERVABILITY,
  prefix `DEPLOY_`/`OBS_` cho biến trùng tên); `deploy/` chỉ chứa thứ phục vụ end-user, stack
  observe nằm ở `observability/` (root) (chạy: `./observability/up.sh` / `./observability/down.sh`; bật observe TRƯỚC rồi mới `./deploy/up.sh`).
- **Evaluation Phase 1** (sinh golden testset theo đơn vị, có checkpoint, chạy tiếp nhiều ngày,
  `tools/generate_testset.py`; PR #55, #60, #63, #64): dùng **9 key Groq** `GROQ_API_KEY_1..9` xoay vòng
  (`groq_round_robin.py`); 429 theo phút cooldown theo `retry-after`, đếm token thật theo key/đơn vị,
  `reasoning_effort=low` chỉ khi dựng KG, và **giữ phần đã sinh khi lỗi giữa đơn vị** (đơn vị `partial`
  chạy tiếp phần thiếu, không mất sample đã xong) — `evaluation_spec.md` mục 3.2, 3.3.
- **Evaluation Phase 2 — spec** (`evaluation_spec.md` mục 11, chốt lại **2026-10-01**): stage HyDE → embed → retrieve
  (MMR bật/tắt, tuần tự) → `context_recall` cả hai cấu hình → chọn MMR → generation → `faithfulness` + `answer_relevancy` →
  `context_precision`; 4 metric chuẩn RAGAS, file JSONL trung gian, resume theo `case_id`; generation chạy nguyên
  `GenerationPipeline` (phương án B); hàng đợi chung 9 key Groq + throttle chủ động theo `(model, key)`; testset cuối
  **157 câu (142 single + 15 multi-hop specific)** = mẫu `keep` của review luna (`data/eval/golden_testset_review.json`),
  không random, không duyệt tay; không pilot bắt buộc.

**Đang nghiệm thu code**
- **Evaluation Phase 2 — code**: đã implement các stage CLI/resume/report, `precomputed` retrieval,
  throttle router/wrapper và `finalize` theo review trên nhánh `feat/gen-testset-ans`
  (`evaluation_spec.md` mục 11.12). Chưa chạy đánh giá thật 157 câu; đang qua vòng tester/reviewer.

**Chưa làm**
- **CD** (GitHub Actions build + push image lên GHCR; không SSH tự động vào máy nhà): chưa brainstorm chi tiết,
  hiện cập nhật thủ công bằng `git pull` → `./deploy/up.sh`.

**Đang hoàn thiện**
- **Golden testset** (`data/eval/`): job `tools/generate_testset.py generate` **đã chạy hết, không còn process** (kiểm tra 2026-09-30 ~22:10):
  **49/50 đơn vị `done`, 1 `skipped`**, raw có **203 câu** (180 single-hop, 23 multi-hop specific,
  **0 multi-hop abstract**) — đã vượt 180; **chốt 2026-10-01: luna (Codex) review 203 mẫu, testset cuối = 157 mẫu `keep` (142 single + 15 multi-hop specific) trong `golden_testset.json` (đã sinh), không random, không duyệt tay**. Đơn vị `skipped` duy nhất:
  `Văn bản hợp nhất bộ luật lao động.md#5` (Chương IV + V, ~29K ký tự) — lỗi ở stage KG (bước `ThemesExtractor`,
  model trả JSON hỏng/rỗng → `OutputParserException`; lần trước là `OpenAITimeoutError`), 96 lượt tích luỹ qua 2 lần thử, 0 câu.
  `Quy định mức lương tối thiểu.md#1` đã chạy lại thành công (`--retry-skipped`, +9 câu). **Đã chốt bỏ** đơn vị này
  (203 > 180, BLLĐ còn nhiều đơn vị khác); nếu muốn thử nữa: `generate --retry-skipped --only "Văn bản hợp nhất bộ luật lao động.md#5"`
  (`generate` không tự chạy lại unit `skipped`). Hệ số token đo được 3,94 token/ký tự (thấp hơn ước tính 5,5).
  Cảnh báo `KG không có cụm cho loại abstract: bỏ N câu` vẫn xuất hiện đều.
  abstract = 0: **đã chốt chấp nhận** (`evaluation_spec.md` mục 4.6); `golden_testset.json` đã sinh; `finalize` đã sửa theo review trên nhánh Phase 2.
- **Nghiệm thu thủ công observe** (bật stack, tạo project + key Langfuse, điền `LANGFUSE_*` vào `.env`,
  `./deploy/up.sh`, xem trace/metrics thật) — đang làm.

Roadmap tiếp theo (thứ tự đề xuất):

1. Nghiệm thu observe trên production (bật stack observe cùng lúc với `./deploy/up.sh` để xem trace/metrics thật).
2. Golden testset **xong** (49/50 đơn vị, 203 câu raw → luna review → `golden_testset.json` 157 mẫu; BLLĐ#5 bỏ, abstract = 0 chấp nhận).
3. Hoàn tất tester/reviewer Evaluation Phase 2 (`/develop-cycle` trên `evaluation_spec.md`, branch `feat/gen-testset-ans`) → chạy full (không pilot
   bắt buộc; S5 lần đầu `--limit 2`);
   sau đó lấy mẫu Q&A thật từ Langfuse.
4. **CD** (chưa làm): GitHub Actions build + push image lên GHCR (không SSH tự động vào máy nhà).
   Tách observability sang VM riêng: chưa chốt.

## Nguyên tắc & bài học xương máu (đúc kết, chi tiết ở từng spec)

- **Đo bằng số trước khi đổi/tin, pilot nhỏ trước khi chạy dài.** Mọi đổi prompt kèm vòng đo trên bộ ca; job tốn quota nhiều ngày phải pilot bằng CLI thật trước. Điều tra nghi vấn bằng dữ liệu
  thật (fetch chunk thật) trước khi kết luận là bug; tìm đúng bên gây lỗi rồi chỉ sửa bên đó.
- **Code deterministic sở hữu cấu trúc và mọi kiểm tra xác định được; LLM chỉ làm phần ngôn ngữ/ngữ nghĩa.** Đừng tin prompt tuyệt đối — sửa ở tầng code (chuẩn hoá citation `【n】`, guardrail setext).
  Gate kiểm chứng **fail-closed** (Judge); guardrail đầu vào **fail-open** (availability). Lỗi ngôn ngữ mơ hồ vá regex quá 3 vòng → chuyển sang LLM.
- **Hạn mức Groq là nút thắt thật:** rate limit theo `(tài khoản, model)`, TPD 200K là ràng buộc chính → cache bắt buộc; 429 theo ngày phải dừng ngay (circuit breaker), 429 theo phút cooldown theo
  `retry-after`; `reasoning_effort=low` cho bước nhẹ (medium tốn token gấp 3–5 lần); 413 không retry được.
- **Trạng thái dùng chung phải thread-safe từ đầu** (không "rủi ro chấp nhận được"); checkpoint theo đơn vị nhỏ, ghi nguyên tử, **ghi dữ liệu trước, ghi trạng thái sau**; "đã xong" là file trạng thái riêng, không suy từ dữ liệu.
- **Log không chứa nội dung người dùng/thông điệp lỗi LLM** (không `exc_info`/`logger.exception`), không log key; bí mật chỉ ở `.env` (không commit, agent không đọc).
- **Quyết định thiết kế ≠ đã implement:** đọc lại code thật trước khi coi spec là xong; đổi model/prompt Judge phải bump `PROMPT_VERSION` (khoá cache).
- **Test phụ thuộc thư viện tuỳ chọn (ragas) bị CI bỏ qua** → chạy cục bộ trong venv `eval` (`--group eval --no-group production`, `~/.cache/eval-venv-run`) trước khi tin; CI xanh chưa đủ.

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
(`push --force`, `reset --hard`, `git clean`, `rm`, đọc `.env`, ...). `develop-cycle` **chỉ người dùng gọi được**, Claude không tự gọi; chỉ `tester` push/mở PR.

**Lưu ý vận hành git/CI:** `main` bảo vệ, bắt buộc check `checks` xanh (`strict`) — **không dùng `[skip ci]`** (check treo pending, không merge được). CI chạy đủ khi diff **cả PR** đụng `src/`, `tests/`,
`tools/`, `pyproject.toml`, `uv.lock`, `.github/workflows/` (spec `.md` dưới `src/` cũng tính). Merge squash nên xoá branch cũ phải `git branch -D` (không phải `-d`); `git checkout main` khi còn sửa chưa commit
ở file mà `main` có bản khác sẽ bị chặn → `git stash` trước.

## Lệnh dev

```bash
uv sync                    # cài dependency (dev group gồm pytest, ruff, mypy)
uv run pytest              # chạy test; -m "not slow" để bỏ test nạp model thật
uv run ruff format .
uv run ruff check .
uv run mypy src            # đúng lệnh CI (`mypy .` báo lỗi tên module ở tools/, có sẵn từ trước)
```
<!-- END CLAUDE.md SYNC -->
