# Cache — Cache câu trả lời & kết quả retrieval bằng Redis

## 1. Mục tiêu & phạm vi

Giảm chi phí LLM và độ trễ cho câu hỏi lặp lại (câu hỏi luật lặp rất nhiều giữa người
dùng), đồng thời bảo vệ hạn mức Groq khi nhiều người hỏi cùng một câu.

**Trong phạm vi:**

- Cache **câu trả lời hoàn chỉnh** theo câu hỏi độc lập (`standalone_query`).
- Cache **kết quả retrieval** (danh sách chunk) theo câu hỏi độc lập.
- **Single-flight**: nhiều request cùng câu hỏi khi cache trống → chỉ 1 request chạy.
- Phát lại câu trả lời cache như một luồng stream thật.
- Vô hiệu hoá cache tự động khi corpus/prompt/model đổi (đưa version vào khoá).

**Không làm:**

- **Không semantic cache** (so khớp embedding): "Khoản 1 Điều 113" và "Khoản 2 Điều
  113" có embedding gần như y hệt nhưng đáp án khác; sai ở miền luật nguy hiểm hơn là
  tốn thêm 1 call.
- Không cache embedding riêng, không cache kết quả guardrail (guardrail luôn chạy trên câu
  gốc, `conversation_spec.md` mục 6–7), không cache lỗi/từ chối.
- Không cache theo user (câu trả lời không cá nhân hoá — prompt cấm tư vấn riêng).
- Không invalidation thủ công phức tạp: đổi version trong khoá là đủ; TTL dọn phần cũ.

**Tiêu chí quan trọng nhất:** cache không bao giờ trả câu trả lời sai/cũ (khác câu hỏi,
khác corpus, khác prompt, khác model), và Redis chết không làm chatbot chết.

## 2. Input & Output

Model (pydantic v2, `cache/models.py`):

- `CachedAnswer`: `text: str`, `citations: list[Citation]`, `created_at: datetime`.
  (Không lưu `warnings`/`usage`: chỉ cache câu trả lời sạch, mục 5.)
- `CacheStatus = Literal["answer_hit", "retrieval_hit", "miss", "bypass"]` — dùng ở
  `TurnTrace` của `conversation/`.

Giao diện:

- `AnswerCache.get(standalone_query) -> CachedAnswer | None`, `.set(standalone_query, CachedAnswer)`.
- `RetrievalCache.get(standalone_query) -> list[RetrievedChunk] | None`, `.set(...)`.
- `SingleFlight.acquire(key) -> async context manager` cho biết mình là `leader` hay
  `follower` (mục 6).
- `replay(CachedAnswer) -> AsyncIterator[GenerationEvent]` (mục 7): `token*`, `citations`,
  `done(usage=None)`.

Mọi hàm **không ném ngoại lệ Redis**: lỗi → log warning và hành xử như miss/không khoá.

## 3. Công cụ

| Việc            | Công cụ                                             |
| --------------- | --------------------------------------------------- |
| Kho cache       | Redis 7 (docker compose, `api_spec.md` mục 8)       |
| Client          | `redis` (redis-py) asyncio                          |
| Chuẩn hoá       | `unicodedata` + `re` thuần Python                   |
| Hash            | `hashlib.sha256`                                    |
| Serialize       | `pydantic` (`model_dump_json` / `model_validate_json`) |

Dependency mới: `redis`. Client `Redis` tạo 1 lần ở lifespan của API và **inject** vào
các lớp cache (module trong `cache/` không đọc `.env`).

## 4. Khoá cache (`normalize.py`, `keys.py`)

Chuẩn hoá `standalone_query`: Unicode NFC → hạ chữ thường → gộp mọi khoảng trắng thành 1
dấu cách → bỏ khoảng trắng/dấu `?.!` ở cuối. **Không bỏ dấu tiếng Việt, không bỏ số**
(đổi nghĩa).

```
answer key    = rag:ans:{corpus_version}:{prompt_version}:{model_name}:{sha256(norm)[:32]}
retrieval key = rag:ret:{corpus_version}:{sha256(norm)[:32]}
lock key      = {answer key}:lock
```

- `corpus_version`: 12 ký tự đầu của sha256 nội dung `data/bm25/bm25_params.json`, tính 1
  lần lúc khởi động (`compute_corpus_version`). File này sinh lại mỗi lần fit BM25 trên
  corpus → đổi corpus thì khoá tự đổi. (Nếu file thiếu: `corpus_version = "unknown"` và
  cảnh báo; cache vẫn chạy nhưng không tự vô hiệu — cần override thủ công qua
  `CACHE_CORPUS_VERSION`.)
- `prompt_version`: hằng số `PROMPT_VERSION` trong `generation/generator.py`; **bắt buộc
  tăng** khi đổi prompt generation hoặc quy tắc `output_check`.
- `model_name`: `GenerationSettings.model_name`.

## 5. Chính sách cache

| Cache      | Ghi khi                                                             | TTL        |
| ---------- | ------------------------------------------------------------------- | ---------- |
| Câu trả lời | Luồng kết thúc `done` **không có `error` và không có `warning`** (kể cả `truncated`), có ≥ 1 `[n]` hợp lệ hoặc là câu "không tìm thấy quy định" | 7 ngày     |
| Retrieval  | `retrieve()` trả ≥ 1 chunk                                          | 24 giờ     |

- Không cache khi `bypass` (dành cho đường đánh giá phase RAGAS sau; phase này chưa có
  luồng nào dùng `bypass`, giữ giá trị trong `CacheStatus` để khỏi đổi schema về sau).
- Không cache `refusal` (guardrail chạy mỗi lượt), `error`, luồng bị client ngắt.
- Lỗi khi ghi cache: bỏ qua, không ảnh hưởng người dùng.

## 6. Single-flight (`singleflight.py`)

Chống "cache stampede": 50 người hỏi cùng câu lúc cache trống chỉ tốn 1 lần generation.

- `SET {lock key} <request_id> NX EX 60` thành công → **leader**: chạy pipeline, ghi cache,
  nhả khoá (`finally`; chỉ xoá nếu giá trị còn là `request_id` của mình — script Lua
  nhỏ hoặc `WATCH`).
- Thất bại → **follower**: poll `AnswerCache.get` mỗi 200 ms tối đa `FOLLOWER_WAIT_SECONDS
  = 45`. Có kết quả → replay. Hết hạn hoặc khoá biến mất mà chưa có cache (leader lỗi/ngắt)
  → follower tự thử làm leader (không chờ vô hạn).
- Khoá có TTL nên leader crash không kẹt hệ thống.
- Follower **không** chiếm slot admission (chỉ leader gọi LLM).
- Redis lỗi → không khoá, mọi request tự chạy (mất tính năng chống trùng, vẫn đúng).

## 7. Phát lại (`replay.py`)

Cache hit phải cho cảm giác stream như thường: chia `text` thành mẩu ~4 từ (giữ nguyên
khoảng trắng/xuống dòng), phát `TokenEvent` mỗi ~15 ms, rồi `CitationsEvent`, rồi
`DoneEvent(usage=None)`. Không phát `status(retrieval|generation)`. Nối lại các mẩu phải
đúng bằng `text` gốc từng ký tự.

## 8. Config

- `REDIS_URL` nằm trong `AdmissionSettings`/`RedisSettings` của `config.py` (dùng chung
  với quota và rate limit, mỗi thứ 1 tiền tố khoá riêng: `rag:`, `quota:`, `rl:`).
- Hằng số nội bộ `cache/`: TTL, `FOLLOWER_WAIT_SECONDS`, tốc độ replay.
- Tuỳ chọn env `CACHE_CORPUS_VERSION` (ghi đè version tự tính).

## 9. Module (`src/production_legal_qa_rag/cache/`)

| Module          | Trách nhiệm                                                        |
| --------------- | ------------------------------------------------------------------ |
| `models.py`     | `CachedAnswer`, `CacheStatus`                                      |
| `normalize.py`  | `normalize_query`                                                  |
| `keys.py`       | `compute_corpus_version`, dựng khoá answer/retrieval/lock          |
| `store.py`      | `AnswerCache`, `RetrievalCache` (get/set, nuốt lỗi Redis)          |
| `singleflight.py` | `SingleFlight`                                                   |
| `replay.py`     | `replay`                                                           |

## 10. Nghiệm thu thủ công

1. Cùng câu hỏi 2 lần → lần 2 không gọi Groq generation, thời gian tới token đầu tiên
   giảm rõ, văn bản y hệt lần 1.
2. Hai câu chỉ khác số Khoản → hai khoá khác nhau, hai câu trả lời khác nhau.
3. Đổi `PROMPT_VERSION` hoặc chạy lại fit BM25 → cache cũ không được dùng.
4. 20 request đồng thời cùng câu hỏi khi cache trống → đúng 1 lần gọi generation.
5. Tắt Redis → chatbot vẫn trả lời (chậm hơn, không cache), log có warning.
6. Ghi lại tỉ lệ trúng cache qua bảng `chatlog` (`cache_status`) để biết cache có đáng giá.

## 11. Rủi ro / điểm mở

1. Tỉ lệ trúng của exact-match phụ thuộc cách người dùng diễn đạt (và độ ổn định của
   condense); thấp hơn semantic cache nhưng an toàn. Đo bằng `cache_status`; chỉ cân nhắc
   thứ gì mạnh hơn nếu số liệu cho thấy đáng.
2. Câu trả lời cache có thể chứa cảnh báo "tham khảo văn bản gốc" đã in sẵn — chấp nhận.
3. `corpus_version` suy ra từ file BM25: nếu index dense (Pinecone) đổi mà không fit lại
   BM25, khoá không đổi — quy trình re-index phải luôn fit lại BM25 hoặc set
   `CACHE_CORPUS_VERSION`.
4. Nội dung câu trả lời nằm trong Redis: đặt Redis trong mạng nội bộ, có mật khẩu, không
   publish cổng ra ngoài (`api_spec.md` mục 8).
