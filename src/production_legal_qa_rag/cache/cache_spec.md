# Cache — Cache câu trả lời & kết quả retrieval bằng Redis

> Giữ nguyên số mục vì code/spec khác tham chiếu.

## 1. Mục tiêu & phạm vi

Giảm chi phí LLM/độ trễ cho câu hỏi lặp lại và bảo vệ hạn mức Groq.

**Làm:** cache câu trả lời hoàn chỉnh và kết quả retrieval theo `standalone_query`; single-flight (cache trống thì chỉ 1 request chạy); phát lại câu trả lời cache như stream thật; version trong khoá để tự vô hiệu khi corpus/prompt/model đổi.

**Không làm:** semantic cache ("Khoản 1 Điều 113" và "Khoản 2 Điều 113" embedding gần y hệt nhưng đáp án khác); cache embedding, guardrail, lỗi/từ chối; cache theo user; invalidation thủ công.

**Tiêu chí số 1:** không bao giờ trả câu trả lời sai/cũ; Redis chết không làm chatbot chết.

## 2. Input & Output

- `CachedAnswer`: `text`, `citations: list[Citation]`, `created_at` (không lưu `warnings`/`usage`). `CacheStatus = Literal["answer_hit","retrieval_hit","miss","bypass"]`.
- `AnswerCache.get/set`, `RetrievalCache.get/set` (theo `standalone_query`), `SingleFlight.acquire(key)` (`leader`/`follower`), `replay(CachedAnswer)` → `token*`, `citations`, `done(usage=None)`.
- **Mọi hàm nuốt lỗi Redis**: log warning, hành xử như miss/không khoá.

## 3. Công cụ

Redis 7, `redis-py` asyncio, `hashlib.sha256`, pydantic, Typer (`tools/cache.py`). Client `Redis` tạo một lần ở lifespan API rồi **inject**; `cache/` không đọc `.env`.

## 4. Khoá cache

Chuẩn hoá: NFC → chữ thường → gộp khoảng trắng → bỏ khoảng trắng/`?.!` cuối. **Không bỏ dấu, không bỏ số**.

```
answer    = rag:ans:{corpus_version}:{prompt_version}:{model_name}:{sha256(norm)[:32]}
retrieval = rag:ret:{corpus_version}:{sha256(norm)[:32]}
lock      = {answer key}:lock
```

- `corpus_version` = 12 ký tự đầu sha256 của `data/bm25/bm25_params.json`; thiếu file → `"unknown"` + cảnh báo; override bằng `CACHE_CORPUS_VERSION`.
- `prompt_version` = `PROMPT_VERSION` trong `generation/generator.py`, **bắt buộc tăng** khi đổi prompt generation hoặc quy tắc `output_check`. `model_name` = `GenerationSettings.model_name`.

## 5. Chính sách cache

| Cache | Ghi khi | TTL |
| --- | --- | --- |
| Câu trả lời | luồng `done` **không `error`, không `warning`**, có ≥1 `[n]` hợp lệ hoặc là câu "không tìm thấy quy định" | 7 ngày |
| Retrieval | `retrieve()` trả ≥1 chunk | 24 giờ |

Không cache khi `bypass`, `refusal`, `error`, luồng bị client ngắt. Lỗi ghi cache: bỏ qua.

## 6. Single-flight

- `SET lock <request_id> NX EX 60` thành công → **leader**: chạy pipeline, ghi cache, nhả khoá trong `finally` (chỉ xoá nếu giá trị còn là `request_id` của mình).
- Thất bại → **follower**: poll `AnswerCache.get` mỗi 200 ms, tối đa `FOLLOWER_WAIT_SECONDS = 45`; có kết quả → replay; hết hạn hoặc khoá biến mất mà chưa có cache → tự thử làm leader.
- Follower **không** chiếm slot admission. Redis lỗi → không khoá, mọi request tự chạy.

## 7. Phát lại

Chia `text` thành mẩu ~4 từ (giữ nguyên khoảng trắng/xuống dòng), `TokenEvent` mỗi ~15 ms, rồi `CitationsEvent`, `DoneEvent(usage=None)`; không phát `status`. Nối mẩu phải đúng từng ký tự với `text` gốc.

## 8. Config

`REDIS_URL` trong `RedisSettings` (`config.py`), dùng chung với quota/rate limit, mỗi thứ một tiền tố (`rag:`, `quota:`, `rl:`). TTL, `FOLLOWER_WAIT_SECONDS`, tốc độ replay là hằng số nội bộ. Env tuỳ chọn: `CACHE_CORPUS_VERSION`.

## 9. Module

`models.py`, `normalize.py` (`normalize_query`), `keys.py` (`compute_corpus_version`, dựng khoá), `store.py` (`AnswerCache`, `RetrievalCache`), `singleflight.py`, `replay.py`; `tools/cache.py` chạy tay các case mục 10.

## 10. Nghiệm thu thủ công

Cùng câu hỏi 2 lần → lần 2 không gọi Groq generation; hai câu chỉ khác số Khoản → hai khoá; đổi `PROMPT_VERSION` hoặc fit lại BM25 → cache cũ không dùng; 20 request đồng thời cùng câu → đúng 1 lần gọi generation; tắt Redis → chatbot vẫn trả lời, có warning.

## 11. Rủi ro / điểm mở

- **Bẫy re-index:** `corpus_version` suy từ file BM25; đổi index dense (Pinecone) mà không fit lại BM25 thì khoá không đổi → mỗi lần re-index phải fit lại BM25 hoặc set `CACHE_CORPUS_VERSION`.
- Nội dung câu trả lời nằm trong Redis: mạng nội bộ, có mật khẩu, không publish cổng ra ngoài.
