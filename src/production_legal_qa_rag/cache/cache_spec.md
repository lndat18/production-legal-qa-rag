# Cache — Cache câu trả lời & kết quả retrieval bằng Redis

- Giữ nguyên số mục để không làm hỏng tham chiếu từ code/spec khác.
- Spec liên quan: [generation_spec.md](../generation/generation_spec.md),
  [conversation_spec.md](../conversation/conversation_spec.md).

## 1. Mục tiêu & phạm vi

- Giảm chi phí/độ trễ câu hỏi lặp lại, bảo vệ hạn mức Groq.
- Cache answer hoàn chỉnh/retrieval theo `standalone_query`; single-flight cache miss; replay stream; version
  key theo corpus/prompt/model.
- Không làm: semantic cache, embedding/guardrail/error/refusal cache, cache theo user, invalidation thủ công.
- Không semantic cache vì Khoản 1/2 cùng Điều có embedding gần nhau nhưng đáp án khác.
- Ưu tiên: không trả answer sai/cũ; Redis lỗi không làm chatbot dừng.

## 2. Input & Output

- `CachedAnswer(text, citations: list[Citation], created_at)`; không lưu warnings/usage.
- `CacheStatus`: `answer_hit`, `retrieval_hit`, `miss`, `bypass`.
- `AnswerCache.get/set`, `RetrievalCache.get/set`: theo standalone query.
- `SingleFlight.acquire(key)`: leader/follower; `replay(CachedAnswer)`: token* → citations → done(usage=None).
- Mọi hàm bắt lỗi Redis, log warning, xử lý như miss/không lock.

## 3. Công cụ

- Redis 7, redis-py asyncio, SHA-256, Pydantic, Typer (`tools/cache.py`).
- API lifespan tạo một Redis client rồi inject; cache không đọc `.env`.

## 4. Khoá cache

- Normalize: NFC → lowercase → gộp whitespace → bỏ whitespace/`?.!` cuối; giữ dấu và số.
- Key:

```text
answer    = rag:ans:{corpus_version}:{prompt_version}:{model_name}:{sha256(norm)[:32]}
retrieval = rag:ret:{corpus_version}:{sha256(norm)[:32]}
lock      = {answer key}:lock
```

- `corpus_version`: 12 ký tự đầu SHA-256 `data/bm25/bm25_params.json`; thiếu → `unknown` + warning; override
  `CACHE_CORPUS_VERSION`.
- `prompt_version`: `PROMPT_VERSION` trong `generation/generator.py`; tăng khi đổi prompt hoặc `output_check`.
- `model_name`: `GenerationSettings.model_name`.

## 5. Chính sách cache

- Answer TTL 7 ngày: luồng done không error/warning, có ≥1 citation `[n]` hợp lệ hoặc câu “không tìm thấy quy
  định”.
- Retrieval TTL 24 giờ: `retrieve()` có ≥1 chunk.
- Không ghi khi bypass/refusal/error/client ngắt; lỗi ghi bỏ qua.

## 6. Single-flight

- `SET lock <request_id> NX EX 60` thành công → leader chạy pipeline/ghi cache, nhả trong finally.
- Chỉ xóa lock còn mang request ID của mình.
- Follower poll answer cache mỗi 200ms, tối đa `FOLLOWER_WAIT_SECONDS=45`; có answer thì replay.
- Hết chờ hoặc lock mất chưa có cache → thử làm leader.
- Follower không chiếm admission; Redis lỗi → không lock, request tự chạy.

## 7. Phát lại

- Chia text khoảng 4 từ/mẩu, giữ whitespace/newline; `TokenEvent` mỗi khoảng 15ms.
- Sau token: `CitationsEvent`, `DoneEvent(usage=None)`; không status.
- Nối mẩu phải khớp từng ký tự text gốc.

## 8. Config

- `RedisSettings.redis_url`/`REDIS_URL` tại `config.py`; namespace riêng `rag:`, `quota:`, `rl:`.
- TTL/wait/replay là hằng nội bộ; env tùy chọn `CACHE_CORPUS_VERSION`.

## 9. Module

- `models.py`: contract; `normalize.py`: `normalize_query`; `keys.py`: `compute_corpus_version` và key.
- `store.py`: hai cache; `singleflight.py`: leader/follower; `replay.py`: event stream.
- `tools/cache.py`: nghiệm thu mục 10.

## 10. Nghiệm thu thủ công

- Cùng câu hai lần: lần hai không gọi generation; khác số Khoản: khác key.
- Bump prompt version/fit lại BM25: cache cũ không dùng.
- 20 request đồng thời cùng câu: một generation; Redis tắt: chatbot trả lời, log warning.

## 11. Rủi ro / điểm mở

- Re-index dense mà không fit BM25 không đổi corpus key; phải fit BM25 hoặc set `CACHE_CORPUS_VERSION`.
- Answer nằm trong Redis: mạng nội bộ, mật khẩu, không publish cổng.
