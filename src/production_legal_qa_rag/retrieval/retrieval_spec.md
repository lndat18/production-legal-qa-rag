# Retrieval — Legal Question → Evidence Chunks: Reference Spec

- Giữ nguyên số mục để không làm hỏng tham chiếu từ code/spec khác.
- Spec liên quan: [embedding_spec.md](../embedding/embedding_spec.md), [generation_spec.md](../generation/generation_spec.md), [conversation_spec.md](../conversation/conversation_spec.md), [evaluation_spec.md](../evaluation/evaluation_spec.md).

## 1. Mục đích

- Trả tập nhỏ `RetrievedChunk` liên quan, đủ citation cho generation.
- **Dense tìm theo nghĩa, sparse tìm định danh pháp lý, reranker chọn bằng chứng cuối.**
- Làm: hybrid dense/sparse, HyDE, citation-aware recall, RRF, MMR tùy chọn, rerank/fallback; build sparse offline từ chunk đã embed.
- Không làm: chunk/embed corpus/generate/hội thoại/cache, policy trả lời/từ chối, agentic decomposition.
- Baseline chất lượng: câu hỏi viện dẫn một Khoản phải có chunk đáp án trong candidate cuối.

## 2. Contract và ranh giới trách nhiệm

- `retrieve(query) -> list[RetrievedChunk]`: tối đa `FINAL_TOP_K`, giảm dần theo `rerank_score` khi rerank thành công.
- `RetrievedChunk` Pydantic v2: `chunk_id`, `source_document`, `breadcrumb`, `content` nguyên văn, `has_table`, `raw_table`, `rerank_score` (`None` khi fallback).
- Retrieval trả evidence tốt nhất tìm được; không lọc rỗng vì score thấp. Conversation/API quyết `no_context`, làm rõ hoặc tiếp tục.
- Eval (mục 11 của `evaluation_spec.md`): `RetrievalPipeline.retrieve(query, *, use_mmr=None, precomputed=None)`.
- `PrecomputedQuery`: `hypothetical_document`, `hypothetical_embedding`, `query_embedding`; hai trường hypothetical cùng `None` hoặc cùng có giá trị.
- Có `precomputed`: bỏ HyDE/embed, giữ nguyên mọi bước sau; `None`: hành vi production cũ. Hàm module-level không lộ tham số này.

## 3. Luồng online

- Nhánh A: HyDE best-effort → hypothetical text; nhánh B: query gốc, luôn có.
- Embed từng text cùng preprocessing index-time → dense/sparse song song mỗi nhánh → RRF → MMR tùy chọn.
- Union theo `chunk_id` → thêm citation extras từ sparse nhánh B → bổ sung metadata/vector thiếu → rerank bằng query gốc → top K.
- A tăng recall ngữ nghĩa; B giữ wording, số Điều/Khoản và tên văn bản.
- HyDE chỉ sinh ngữ nghĩa/văn phong; không bịa số Điều/Khoản, tên văn bản, năm hoặc mức số.
- HyDE lỗi/rỗng: bỏ A, giữ B; model `gpt-oss-20b`.
- Giữ `MIN_RERANK_SCORE` tại `relevance.py` theo `conversation_spec.md` mục 12.1.

## 4. Dense, sparse và fusion

- Dense: model/preprocessing query khớp index (`pyvi`); trả metadata, chỉ lấy vector khi MMR cần; chỉ đọc dense index.
- Sparse BM25: text = `breadcrumb + " " + content`; một tokenizer cho fit/document/query; không mặc định stopword/stemming tiếng Anh.
- Công thức/vocabulary/thứ tự ID deterministic; params có version, runtime từ chối version khác.
- RRF: không cộng dense score và BM25 score; dedupe theo ID, giới hạn pool trước rerank.

## 5. Citation-aware retrieval

- Chỉ parse citation từ query gốc; ưu tiên bỏ sót hơn nhận nhầm năm, tiền, tuổi, thời gian hoặc từ “điều kiện”.
- Văn bản mơ hồ/nhiều văn bản: không đoán document key.
- Structural terms ổn định ở document/query: `điều_113`, `khoản_1`, `điều_113_khoản_1`, `vb_blld_điều_113_khoản_1`.
- Thêm term vào sparse vector, không đổi text gốc/dense; chỉ B nhận term từ citation user.
- Mapping `source_document` → document key phải explicit/versioned; cảnh báo văn bản chưa map.
- Citation hợp lệ: thêm top sparse thô của B vào union trước rerank; dùng lại kết quả, không thêm round trip; chặn số extras.

## 6. Diversity và rerank

- MMR là cờ runtime/eval; có thể phạt oan các Khoản gần nhau nhưng cùng cần thiết. Tắt MMR giữ đầu RRF làm baseline.
- Rerank: query gốc, passage `breadcrumb + "\n" + content`; gửi toàn pool một lượt, chia batch nội bộ (6.1).
- Validate score hữu hạn/đúng thứ tự → sort giảm dần → top K.
- Không ghim kết quả sau rerank; bảo vệ citation recall ở candidate stage bằng extras.

### 6.1 Inference tại chỗ (local, không host API riêng)

- `AITeamVN/Vietnamese_Reranker` chạy in-process; load tokenizer/model một lần khi tạo pipeline, giữ suốt process.
- Tự chọn CUDA fp16 (`model.half()`), fallback CPU fp32; CPU hợp lệ; log INFO device một lần.
- `MAX_LENGTH=512`: tổng token query + passage + 4 special tokens; margin chưa đo trên corpus, đo lại khi truncation/chất lượng giảm.
- Pool tối đa `2 × (DENSE_TOP_N + SPARSE_TOP_N) = 80`; batch cố định `RERANK_BATCH_SIZE` khoảng 16 để giới hạn VRAM.
- Forward blocking chạy executor, không chặn event loop.
- **Đúng một inference request/process:** private module-scoped `ThreadPoolExecutor(max_workers=1)` bọc toàn `rerank()` và mọi batch.
- Không đưa concurrency vào `RerankerSettings`; không dùng `asyncio.Semaphore`/`LoopBoundClient` cho limiter này vì hai loop có thể cấp hai permit, gây CUDA OOM trên GPU 2GB.

## 7. Consistency và state offline

- Dense/sparse index và BM25 params phải thuộc cùng corpus snapshot; không best-effort khi lệch.
- Manifest sparse: corpus fingerprint, số chunk/ID/checksum, tokenizer/BM25 params, document mapping version, dense build nguồn.
- Runtime chỉ load params/index tương thích; sparse candidate không fetch được dense metadata thì bỏ và log dấu hiệu build lệch.
- Sparse build offline, full refresh.

## 8. Xử lý lỗi

- HyDE lỗi/rỗng/`ThrottleTimeout` → bỏ A, chạy B.
- Reranker lỗi/model/CUDA OOM → fallback deterministic từ các nhánh, `rerank_score=None`.
- Query embed, dense/sparse search hoặc metadata fetch lỗi → `RetrievalError`; không giả evidence.
- Corpus version mismatch → từ chối init/query, nêu cách rebuild.
- HF/Pinecone: timeout hữu hạn, chỉ retry lỗi tạm thời.
- Reranker: fallback ngay, không retry cùng input; log batch size/độ dài passage.

## 9. Module boundaries

- `models.py`: contract/`RetrievalError`; `hyde.py`: best-effort, `HydeSettings` riêng (`gpt-oss-20b`, `GROQ_API_KEY_1`).
- `llm_throttle.py`: `TokenWindowThrottle` dùng chung Condense/HyDE/Judge theo bucket (conversation 12.1).
- `query_embedder.py`: embed query; `bm25.py`: tokenizer/params; `citation.py`: parse/mapping/terms/extras.
- `dense_search.py`, `sparse_index.py`: index I/O; `fusion.py`, `mmr.py`: hàm thuần.
- `reranker.py`: load/batch/process-wide limiter/validate/fallback; `pipeline.py`: điều phối, sở hữu client.
- `relevance.py`: signal cho policy, không lọc chunk.
- `tools/retrieval.py`: Typer mỏng, corpus/index thật, in chunk, cảnh báo score `None`; `--query`, `--use-mmr/--no-use-mmr`.
- Contract Pydantic; `pydantic-settings` tập trung config.

## 10. Tiêu chí hoàn thành

- Literal/paraphrase/citation cùng contract; citation hợp lệ có đường recall độc lập HyDE/RRF/MMR.
- Dense index/query cùng preprocessing; sparse fit/document/query cùng tokenizer/structural terms.
- Output ≤ K, đủ metadata/citation, không candidate từ snapshot lệch.
- Rerank lỗi có fallback xác định; lỗi search không bị che; đúng một inference/process kể cả nhiều event loop.

## 11. Áp dụng cho bài toán mới

- Baseline: query gốc + dense/sparse song song + RRF + rerank query gốc.
- Chỉ thêm HyDE/MMR/extras cho failure mode đã quan sát; giữ baseline để so sánh.
