# Cache — Cache câu trả lời & kết quả retrieval bằng Redis

> Cô đọng 2026-09-30 (giữ số mục vì code/spec khác tham chiếu). Bản đầy đủ: git history.

## 1. Mục tiêu & phạm vi

Giảm chi phí LLM/độ trễ cho câu hỏi lặp lại và bảo vệ hạn mức Groq khi nhiều người hỏi cùng câu.

**Làm:** cache câu trả lời hoàn chỉnh và kết quả retrieval theo `standalone_query`; single-flight
(cache trống thì chỉ 1 request chạy); phát lại câu trả lời cache như stream thật; version trong khoá
để tự vô hiệu khi corpus/prompt/model đổi.

**Không làm:**
- **Không semantic cache.** "Khoản 1 Điều 113" và "Khoản 2 Điều 113" có embedding gần như y hệt
  nhưng đáp án khác; sai ở miền luật nguy hiểm hơn tốn thêm 1 call.
- Không cache embedding, kết quả guardrail (luôn chạy trên câu gốc), lỗi/từ chối; không cache theo user;
  không invalidation thủ công (đổi version trong khoá + TTL là đủ).

**Tiêu chí số 1:** không bao giờ trả câu trả lời sai/cũ (khác câu hỏi/corpus/prompt/model); Redis chết
không làm chatbot chết.

## 2. Input & Output

- `CachedAnswer`: `text`, `citations: list[Citation]`, `created_at` (không lưu `warnings`/`usage`: chỉ
  cache câu trả lời sạch). `CacheStatus = Literal["answer_hit","retrieval_hit","miss","bypass"]`.
- `AnswerCache.get/set`, `RetrievalCache.get/set` (theo `standalone_query`), `SingleFlight.acquire(key)`
  (cho biết `leader`/`follower`), `replay(CachedAnswer)` → `token*`, `citations`, `done(usage=None)`.
- **Mọi hàm nuốt lỗi Redis**: log warning, hành xử như miss/không khoá.

## 3. Công cụ

Redis 7, `redis-py` asyncio, `unicodedata`+`re`, `hashlib.sha256`, pydantic, Typer (`tools/cache.py`).
Client `Redis` tạo một lần ở lifespan API rồi **inject**; `cache/` không đọc `.env`.

## 4. Khoá cache

Chuẩn hoá: NFC → chữ thường → gộp khoảng trắng → bỏ khoảng trắng/`?.!` cuối. **Không bỏ dấu, không bỏ số**.

```
answer    = rag:ans:{corpus_version}:{prompt_version}:{model_name}:{sha256(norm)[:32]}
retrieval = rag:ret:{corpus_version}:{sha256(norm)[:32]}
lock      = {answer key}:lock
```

- `corpus_version` = 12 ký tự đầu sha256 của `data/bm25/bm25_params.json` (sinh lại mỗi lần fit BM25).
  Thiếu file → `"unknown"` + cảnh báo, không tự vô hiệu; override bằng `CACHE_CORPUS_VERSION`.
- `prompt_version` = `PROMPT_VERSION` trong `generation/generator.py`; **bắt buộc tăng** khi đổi prompt
  generation hoặc quy tắc `output_check`. `model_name` = `GenerationSettings.model_name`.

## 5. Chính sách cache

| Cache | Ghi khi | TTL |
| --- | --- | --- |
| Câu trả lời | luồng `done` **không `error`, không `warning`** (kể cả `truncated`), có ≥1 `[n]` hợp lệ hoặc là câu "không tìm thấy quy định" | 7 ngày |
| Retrieval | `retrieve()` trả ≥1 chunk | 24 giờ |

Không cache khi `bypass` (dành cho đường đánh giá; giữ trong `CacheStatus` để khỏi đổi schema), `refusal`,
`error`, luồng bị client ngắt. Lỗi ghi cache: bỏ qua.

## 6. Single-flight

Chống cache stampede: 50 người hỏi cùng câu lúc cache trống chỉ tốn 1 generation.
- `SET lock <request_id> NX EX 60` thành công → **leader**: chạy pipeline, ghi cache, nhả khoá trong
  `finally` (chỉ xoá nếu giá trị còn là `request_id` của mình — Lua/`WATCH`).
- Thất bại → **follower**: poll `AnswerCache.get` mỗi 200 ms, tối đa `FOLLOWER_WAIT_SECONDS = 45`. Có kết
  quả → replay; hết hạn hoặc khoá biến mất mà chưa có cache (leader lỗi) → tự thử làm leader.
- Khoá có TTL nên leader crash không kẹt. Follower **không** chiếm slot admission. Redis lỗi → không khoá,
  mọi request tự chạy (mất chống trùng, vẫn đúng).

## 7. Phát lại

Chia `text` thành mẩu ~4 từ (giữ nguyên khoảng trắng/xuống dòng), `TokenEvent` mỗi ~15 ms, rồi
`CitationsEvent`, `DoneEvent(usage=None)`; không phát `status`. Nối mẩu phải đúng từng ký tự với `text` gốc.

## 8. Config

`REDIS_URL` trong `RedisSettings` (`config.py`), dùng chung với quota/rate limit, mỗi thứ một tiền tố
(`rag:`, `quota:`, `rl:`). TTL, `FOLLOWER_WAIT_SECONDS`, tốc độ replay là hằng số nội bộ. Env tuỳ chọn:
`CACHE_CORPUS_VERSION`.

## 9. Module

`models.py`, `normalize.py` (`normalize_query`), `keys.py` (`compute_corpus_version`, dựng khoá), `store.py`
(`AnswerCache`, `RetrievalCache`), `singleflight.py`, `replay.py`; `tools/cache.py` chạy tay các case mục 10.

## 10. Nghiệm thu thủ công

1. Cùng câu hỏi 2 lần: lần 2 không gọi Groq generation, văn bản y hệt.
2. Hai câu chỉ khác số Khoản → hai khoá, hai câu trả lời khác nhau.
3. Đổi `PROMPT_VERSION` hoặc fit lại BM25 → cache cũ không dùng.
4. 20 request đồng thời cùng câu, cache trống → đúng 1 lần gọi generation.
5. Tắt Redis → chatbot vẫn trả lời, log có warning.
6. Đo tỉ lệ trúng qua Langfuse (`cache_status`) hoặc metric `chat_turns_total`.

## 11. Rủi ro / điểm mở

1. Exact-match phụ thuộc cách diễn đạt (và độ ổn định của condense): thấp hơn semantic nhưng an toàn; chỉ
   cân nhắc mạnh hơn khi số liệu `cache_status` cho thấy đáng.
2. Câu trả lời cache có thể chứa cảnh báo "tham khảo văn bản gốc" in sẵn — chấp nhận.
3. **Bẫy re-index:** `corpus_version` suy từ file BM25; đổi index dense (Pinecone) mà không fit lại BM25 thì
   khoá không đổi → mọi lần re-index phải fit lại BM25 hoặc set `CACHE_CORPUS_VERSION`.
4. Nội dung câu trả lời nằm trong Redis: mạng nội bộ, có mật khẩu, không publish cổng ra ngoài.
