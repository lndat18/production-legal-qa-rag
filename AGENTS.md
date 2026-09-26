# production-legal-qa-rag

RAG chatbot hỏi-đáp pháp luật Việt Nam, thiết kế theo hướng sát production nhưng gọn. Đây là dự án cá nhân, public qua Cloudflare Tunnel; dùng Python 3.14 và `uv`.

## Repository map

- `src/production_legal_qa_rag/`: source importable, chia package theo nghiệp vụ.
- `tests/`: test, tên `test_<package>_<phần>.py`.
- `tools/`: CLI Typer cho từng bước pipeline.
- `alembic/`: migration Postgres.
- `data/`: raw → Markdown → chunks → embeddings; `bm25/` cho sparse index.
- `models/`: model tải local chạy in-process, ví dụ Vietnamese reranker.
- `deploy/`: Docker Compose và tài liệu triển khai; không phải source importable.
- `docs/`: tài liệu tổng quan, ví dụ activity diagram cho một lượt hỏi đáp.
- `.codex/`: cấu hình Codex theo project: custom subagent roles, MCP và command rules.
- `.agents/skills/`: skills Codex cho coding convention và develop cycle.

Mỗi package trong `src/production_legal_qa_rag/` có `<package>_spec.md` cạnh package; đọc spec đó trước khi sửa code thuộc package.

## Pipeline packages

| Package | Vai trò | Spec |
| --- | --- | --- |
| `formatting/` | `.docx` pháp luật → Markdown có cấu trúc, giữ Điều/Khoản/Điểm | `formatting_spec.md` |
| `chunking/` | Markdown → chunk tự đủ nghĩa, sẵn sàng embedding | `chunking_spec.md` |
| `embedding/` | Chunk → vector, giữ citation | `embedding_spec.md` |
| `retrieval/` | Câu hỏi → `RetrievedChunk` liên quan: HyDE, hybrid, RRF, MMR, rerank local GPU | `retrieval_spec.md` |
| `generation/` | Retrieved chunks đã rerank → câu trả lời có citation, được kiểm chứng | `generation_spec.md` |
| `conversation/` | Điều phối lượt hỏi đáp: condense → guardrail → cache → admission → retrieve → generate | `conversation_spec.md` |
| `cache/` | Redis cache cho câu trả lời/retrieval, single-flight | `cache_spec.md` |
| `chatlog/` | Postgres lưu lượt hỏi đáp để quan sát/eval, không phải lịch sử chat | `chatlog_spec.md` |
| `api/` | FastAPI OpenAI-compatible, OpenWebUI, Redis và Postgres; spec toàn hệ thống | `api_spec.md` |

Các file spec trên nằm trong package tương ứng dưới `src/production_legal_qa_rag/`.

## Current state and roadmap

- Cả 9 packages đã implement, có spec và tests; API OpenAI-compatible chạy end-to-end cùng OpenWebUI, Redis và Postgres.
- `deploy/docker-compose.dev.yml` chạy Redis/Postgres local. Docker Compose production đầy đủ chưa tạo.
- Thứ tự roadmap đã chốt, không đảo thứ tự nếu chưa có quyết định mới:
  1. Đánh giá chất lượng bằng RAGAS dựa trên Q&A thật từ `chat_turns`.
  2. CI/CD và observability: Langfuse self-host, Prometheus, Grafana chạy local trước; tách hạ tầng sang VM/k8s tính sau.
- `chatlog` Postgres tự host vẫn là nguồn dữ liệu chính chủ; không thay bằng Langfuse Cloud vì riêng tư và nhu cầu query SQL trực tiếp.

## Project conventions

- Dùng `$coding-convention` khi tạo/sửa Python, CLI, cấu trúc package, hoặc `*_spec.md`.
- Không thêm scope ngoài spec đã chốt. Hỏi người dùng nếu một quyết định thiết kế quan trọng chưa có trong spec.
- Các lệnh phát triển chuẩn: `uv sync`, `uv run pytest`, `uv run ruff format .`, `uv run ruff check .`, `uv run mypy .`, `uv run alembic upgrade head`.
- Không force-push, xoá branch cưỡng bức, `git reset --hard`, `git clean`, xoá bằng `rm`, hoặc đọc `.env`.
- Trong develop cycle, chỉ tester được push/mở PR; không agent nào được merge. Merge vào `main` do người dùng thực hiện thủ công sau reviewer `PASS`.

## Development workflow

- Codex có các custom subagent roles `architect`, `developer`, `tester`, `reviewer`; dùng roles/skills tương ứng cho vòng đời spec → implement → test → review.
- `architect` brainstorm và chốt `*_spec.md`; `developer` implement rồi chỉ commit local; `tester` chỉ sửa `tests/`, push/mở PR và theo dõi CI; `reviewer` review PR sau checks, comment `PASS` hoặc `REVISE`, không merge.
- Dùng `$develop-cycle <spec-path> <branch>` khi user yêu cầu toàn bộ lifecycle. Workflow tuần tự developer → tester → reviewer, dừng khi hard local gate/checks fail, bị block, hoặc một trong hai loại feedback (CI/design) đạt 3 vòng.
- `.codex/rules/settings.rules` là allowlist command tương ứng cho Git/GitHub CLI/local checks và chặn các thao tác phá hoại; tuân thủ giới hạn role ngay cả khi command được allow.

## OpenAI documentation

Always use the OpenAI developer documentation MCP server when work concerns the OpenAI API, plugins, ChatGPT, or Codex, unless the user explicitly asks not to browse.
