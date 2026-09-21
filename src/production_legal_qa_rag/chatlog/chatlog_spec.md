# Chatlog — Nhật ký từng lượt hỏi đáp trong Postgres

## 1. Mục tiêu & phạm vi

Ghi lại **mỗi lượt hỏi–đáp** ở phía backend để quan sát vận hành và làm nguyên liệu đánh
giá chất lượng bằng dữ liệu thật (RAGAS phase sau, đo tỉ lệ cache, phân tích lỗi/từ
chối). **Không phải lịch sử hội thoại**: lịch sử do OpenWebUI ghi vào Postgres của nó
(kiểu A, `api_spec.md` mục 9); chatlog không được dùng để dựng lại `messages[]`.

**Trong phạm vi:** một bảng `chat_turns`, ghi bất đồng bộ không chặn người dùng, migration
bằng Alembic, script xoá dữ liệu quá hạn.

**Không làm:** lưu lịch sử hội thoại; dashboard/metrics (Prometheus/Grafana — phase sau);
lưu `messages[]` đầy đủ; gửi log sang Kafka hay hệ thống bên ngoài; lưu email/tên người
dùng.

**Tiêu chí quan trọng nhất:** ghi log **không bao giờ** làm chậm hay làm hỏng câu trả lời
cho người dùng; mọi lượt (kể cả bị từ chối, lỗi, cache hit, ngắt kết nối) đều có 1 dòng.

## 2. Input & Output

`TurnRecord` (pydantic v2, `chatlog/models.py`) dựng từ `TurnTrace` + `RequestContext` của
`conversation/`:

| Cột                       | Kiểu             | Ghi chú                                                        |
| ------------------------- | ---------------- | -------------------------------------------------------------- |
| `id`                      | uuid PK          | sinh ở app                                                     |
| `created_at`              | timestamptz      | index                                                          |
| `request_id`              | text             | tương ứng header/log ứng dụng                                  |
| `user_id`                 | text             | id nội bộ OpenWebUI (từ header), index; không lưu email/tên    |
| `chat_id`                 | text null        | id cuộc hội thoại của OpenWebUI, index                         |
| `raw_query`               | text             | câu người dùng vừa gửi                                         |
| `standalone_query`        | text null        | sau condense; null nếu bị chặn trước đó                        |
| `outcome`                 | text             | `answered` \| `refused` \| `error` \| `client_disconnected`     |
| `verdict`                 | text             | `allow` \| `out_of_scope` \| `injection`                       |
| `error_code`              | text null        |                                                                |
| `cache_status`            | text             | `answer_hit` \| `retrieval_hit` \| `miss` \| `bypass`          |
| `chunk_ids`               | jsonb            | thứ tự như context                                             |
| `answer_text`             | text             | rỗng nếu refused/error                                         |
| `citations`, `warnings`   | jsonb            |                                                                |
| `usage`                   | jsonb null       | token prompt/completion/reasoning                              |
| `time_to_first_token_ms`  | int null         |                                                                |
| `latency_ms`              | int              |                                                                |
| `prompt_version`, `corpus_version`, `model_name` | text | để so sánh trước/sau khi đổi cấu hình                |

Giao diện: `ChatLogRepository.record(turn: TurnRecord) -> None` (không ném ngoại lệ).

## 3. Công cụ

| Việc       | Công cụ                                            |
| ---------- | -------------------------------------------------- |
| CSDL       | Postgres 17 (docker compose; chung instance với OpenWebUI, **database riêng** `chatbot`) |
| ORM/driver | `sqlalchemy[asyncio]` 2.x + `asyncpg`              |
| Migration  | `alembic` (thư mục `alembic/` ở root repo)         |
| CLI dọn dữ liệu | `typer` (`tools/purge_chatlog.py`)            |

Dependency mới: `sqlalchemy[asyncio]`, `asyncpg`, `alembic`. Engine tạo 1 lần ở lifespan của
API và inject vào repository.

## 4. Workflow ghi log

- Lớp API (`api/routes.py`) khởi tạo `TurnTrace`, truyền cho `orchestrator.stream`, và trong
  khối `finally` sau khi stream kết thúc (bình thường, lỗi hoặc client ngắt) dựng
  `TurnRecord` rồi `asyncio.create_task(repository.record(...))`. Giữ tham chiếu task
  trong 1 tập để tránh bị GC; task lỗi chỉ log warning (không nội dung).
- Ghi 1 lần/lượt (`INSERT`), không update. Timeout ghi `5 s`.
- Khi lifespan tắt: chờ các task ghi còn dang dở tối đa 5 s.

## 5. Quyền riêng tư & lưu trữ

- Câu hỏi và câu trả lời có thể chứa thông tin cá nhân (tình huống lao động cụ thể). Bảng
  chỉ truy cập được từ backend/nhà vận hành; **không** log nội dung ra stdout
  (`generation_spec.md` mục 9 vẫn đúng cho log ứng dụng — bảng này là kho có kiểm soát).
- Lưu `user_id` nội bộ của OpenWebUI, không lưu email/tên.
- **Thời hạn lưu `CHATLOG_RETENTION_DAYS = 90`**; `tools/purge_chatlog.py` xoá dòng quá hạn
  (chạy tay hoặc cron của host). Xoá khi người dùng yêu cầu: `--user-id`.
- Với triển khai public: hiển thị thông báo trên OpenWebUI (banner) rằng hội thoại được
  lưu để cải thiện dịch vụ (`api_spec.md` mục 9).

## 6. Config

`DatabaseSettings` (`config.py`): `database_url` (`CHATLOG_DATABASE_URL`, dạng
`postgresql+asyncpg://...`). `CHATLOG_RETENTION_DAYS` là tham số của script dọn (mặc định 90).
Module không đọc `.env` trực tiếp. Cập nhật `.env.example`.

## 7. Module (`src/production_legal_qa_rag/chatlog/`)

| Module          | Trách nhiệm                                                    |
| --------------- | -------------------------------------------------------------- |
| `models.py`     | `TurnRecord` (pydantic), hàm dựng từ `TurnTrace` + `RequestContext` |
| `tables.py`     | Bảng `chat_turns` (SQLAlchemy) + index                         |
| `repository.py` | `ChatLogRepository.record`, tạo engine/session factory         |

Ngoài package: `alembic/` + `alembic.ini` (revision đầu tạo `chat_turns`),
`tools/purge_chatlog.py`.

## 8. Nghiệm thu thủ công

1. Câu hỏi thường, câu bị từ chối, câu gặp lỗi, cache hit, ngắt kết nối giữa chừng → mỗi
   loại có đúng 1 dòng với `outcome`/`cache_status` đúng.
2. Tắt Postgres → chatbot vẫn trả lời bình thường, log ứng dụng có warning.
3. Truy vấn thử: tỉ lệ `cache_status`, phân bố `outcome`, top `error_code`, p50/p95
   `time_to_first_token_ms`.
4. `purge_chatlog.py` xoá đúng dòng quá hạn, không đụng dòng khác.

## 9. Rủi ro / điểm mở

1. Bảng tăng nhanh nếu nhiều người dùng — index theo `created_at`, retention 90 ngày;
   partition nếu sau này cần.
2. Kết hợp dữ liệu này với RAGAS (lấy mẫu câu thật, gán nhãn) là việc của spec đánh giá
   phase sau.
