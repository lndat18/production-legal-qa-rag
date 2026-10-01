# Retrieval — Legal Question → Evidence Chunks: Reference Spec

> Giữ nguyên số mục vì code/spec khác tham chiếu.

## 1. Mục đích

Từ một câu hỏi, trả về tập nhỏ `RetrievedChunk` đủ liên quan và có citation để generation trả lời dựa trên bằng chứng.

> **Tìm theo nghĩa và theo định danh pháp lý, rồi để reranker chọn bằng chứng cuối cùng.** Dense mạnh về ý định nhưng không đáng tin với số Điều/Khoản; sparse mạnh về định danh nhưng yếu với paraphrase → giữ cả hai.

**Phạm vi:** `retrieve(query)` trả tối đa `FINAL_TOP_K` chunk; hybrid dense/sparse, citation-aware recall, rerank, fallback có chủ đích; build offline sparse index từ chunk đã embed. **Ngoài phạm vi:** chunking/embedding corpus/generation/hội thoại/cache; quyết định trả lời hay từ chối (policy của conversation/API); agentic decomposition. Baseline chất lượng: câu hỏi viện dẫn một Khoản phải đưa chunk đáp án vào candidate cuối.

## 2. Contract và ranh giới trách nhiệm

`retrieve(query) -> list[RetrievedChunk]` (≤ `FINAL_TOP_K`, xếp theo `rerank_score` khi rerank thành công). `RetrievedChunk` (Pydantic v2): `chunk_id`, `source_document`, `breadcrumb`, `content` (nguyên văn), `has_table`, `raw_table`, `rerank_score` (`None` khi fallback). Retrieval **luôn trả evidence tốt nhất tìm được, không lọc thành rỗng vì score thấp**; tầng policy dùng `rerank_score` để quyết `no_context`/làm rõ/tiếp tục.

**Đường vào cho evaluation (`evaluation_spec.md` mục 11):** `RetrievalPipeline.retrieve(query, *, use_mmr=None, precomputed=None)`; `PrecomputedQuery` (`models.py`): `hypothetical_document`, `hypothetical_embedding`, `query_embedding` (`hypothetical_document` là `None` khi và chỉ khi `hypothetical_embedding` là `None`). Có `precomputed` thì bỏ HyDE + embed và dùng giá trị đó; mọi bước sau giữ NGUYÊN code; mặc định `None` = hành vi cũ. Hàm module-level `retrieve()` không lộ tham số này.

## 3. Luồng online

Query gốc → (nhánh A: HyDE → hypothetical text, best-effort) + (nhánh B: query gốc, luôn có) → embed từng text bằng cùng preprocessing index-time → dense + sparse song song mỗi nhánh → RRF → MMR tuỳ chọn → union theo `chunk_id` → citation extras từ sparse nhánh B → bổ sung metadata/vector thiếu → rerank bằng query gốc → top K. Nhánh A tăng recall semantic; nhánh B là nguồn sự thật cho literal wording, số Điều/Khoản, tên văn bản.

HyDE chỉ sinh văn phong/ngữ nghĩa pháp lý, **không được bịa số Điều/Khoản, tên văn bản, năm, mức số**; hỏng/rỗng → bỏ nhánh A, không làm hỏng nhánh B. HyDE chạy `gpt-oss-20b`; `MIN_RERANK_SCORE` (`relevance.py`) giữ nguyên (`conversation_spec.md` mục 12.1).

## 4. Dense, sparse và fusion

- **Dense:** embed query bằng đúng model + preprocessing lúc index (word-segment `pyvi`); index-time/query-time là một contract. Dense query trả metadata; chỉ lấy vector khi MMR cần. `retrieval/` chỉ đọc dense index.
- **Sparse BM25:** text = `breadcrumb + " " + content`. Một tokenizer nhất quán cho fit/document/query; **không mặc định stopword/stemming tiếng Anh**. Công thức/vocabulary/thứ tự ID deterministic; params lưu versioned, runtime từ chối params khác version.
- **RRF:** không cộng thẳng dense score với BM25 score. Dedupe bằng `chunk_id`, cắt pool nhỏ trước rerank.

## 5. Citation-aware retrieval

**Parse thận trọng:** trích Điều/Khoản chỉ từ query gốc; ưu tiên false negative hơn false positive (năm, tiền, tuổi, đơn vị thời gian, từ "điều kiện" không thành Điều); tên văn bản mơ hồ/nhiều văn bản thì không đoán document key. **Structural terms:** token hiếm, ổn định ở cả document và query (`điều_113`, `khoản_1`, `điều_113_khoản_1`, `vb_blld_điều_113_khoản_1`) bổ sung vào sparse vector, không thay text gốc/dense; **chỉ nhánh B** nhận token từ citation người dùng. Mapping `source_document` → document key phải explicit, versioned, cảnh báo khi corpus có văn bản chưa map. **Citation extras:** khi query có citation hợp lệ, thêm top sparse thô của nhánh B vào union trước rerank (dùng lại kết quả sparse, không thêm round trip; số extras bị chặn).

## 6. Diversity và rerank

MMR có thể phạt oan các Khoản gần nhau vốn cùng cần cho câu hỏi luật → **là cờ runtime/evaluation, không phải "tối ưu" được tin sẵn**; tắt thì giữ đầu RRF làm baseline. Reranker là quyết định cuối: query là câu hỏi gốc (không phải HyDE); passage = `breadcrumb + "\n" + content`; gửi cả candidate pool một lượt (chia batch nội bộ, 6.1), validate score hữu hạn + đúng thứ tự, sort giảm dần, cắt top K. **Không ghim kết quả bằng rule sau rerank**; muốn bảo đảm citation recall thì làm ở candidate stage qua extras.

### 6.1 Inference tại chỗ (local, không host API riêng)

Model `AITeamVN/Vietnamese_Reranker` chạy in-process trong `RetrievalPipeline`.
- **Device:** tự phát hiện `cuda`, fallback `cpu`; GPU fp16 (`model.half()`), CPU fp32. CPU là đường hợp lệ. Log `INFO` device một lần lúc load.
- **Vòng đời:** load tokenizer + model một lần lúc khởi tạo pipeline, giữ suốt process.
- **`MAX_LENGTH = 512`** = giới hạn cho **tổng** `token(query) + token(breadcrumb + "\n" + content) + 4 token đặc biệt`. Là ước lượng có margin, **chưa đo trên corpus**; đo lại nếu có cảnh báo truncation hoặc chất lượng rerank giảm.
- **Batch:** pool tối đa `2 × (DENSE_TOP_N + SPARSE_TOP_N)` = 80 passage; chia batch cố định `RERANK_BATCH_SIZE` (~16) để chặn đỉnh VRAM.
- **Không chặn event loop:** forward blocking chạy trong executor.
- **Concurrency inference process-wide = 1:** nhiều forward đồng thời trên cùng model → VRAM cộng dồn → CUDA OOM (GPU 2GB). Bọc mỗi lần `rerank()` (gồm mọi batch) trong `ThreadPoolExecutor(max_workers=1)` private, module-scoped; cứng `1`, không đưa vào `RerankerSettings`. **Không dùng `asyncio.Semaphore`/`LoopBoundClient`** cho limiter này (loop-bound, hai loop song song sẽ có hai permit).

## 7. Consistency và state offline

Dense index, sparse index và BM25 params phải mô tả **cùng một corpus snapshot**. Mỗi build sparse ghi manifest/version (corpus fingerprint, số chunk + ID/checksum, tokenizer/BM25 params + mapping document version, dense build mà sparse dựa vào). Runtime chỉ load params/index tương thích dense snapshot, **không "best effort" khi lệch**; candidate có ở sparse mà không fetch được metadata từ dense bị bỏ và log như tín hiệu build lệch. Sparse build offline, full refresh.

## 8. Xử lý lỗi

| Sự cố | Hành vi |
| --- | --- |
| HyDE lỗi/rỗng/`ThrottleTimeout` | Bỏ nhánh A, chạy nhánh B |
| Reranker lỗi (runtime/model, ví dụ CUDA OOM) | Fallback deterministic từ các nhánh, `rerank_score=None` |
| Query embed, dense/sparse search, metadata fetch lỗi | `RetrievalError`; không giả vờ có evidence |
| Corpus version mismatch | Từ chối khởi tạo/query, nêu thao tác rebuild |

Retry chỉ cho lỗi tạm thời của dịch vụ ngoài (HF embedding, Pinecone), timeout hữu hạn. Reranker in-process: lỗi cùng input là deterministic → fallback ngay, log đủ (kích thước batch, độ dài passage).

## 9. Module boundaries

`models.py` (contract + `RetrievalError`), `hyde.py` (best-effort; `HydeSettings` riêng: `gpt-oss-20b`, `GROQ_API_KEY_1`), `llm_throttle.py` (`TokenWindowThrottle` chung cho bucket `gpt-oss-20b`: condense/HyDE/Judge, `conversation_spec.md` mục 12.1), `query_embedder.py`, `bm25.py`, `citation.py` (parse, mapping, structural terms, extras), `dense_search.py`/`sparse_index.py`, `fusion.py`/`mmr.py` (thuần, không I/O), `reranker.py` (load 1 lần, batch inference, giới hạn 1 request process-wide, validate, fallback), `pipeline.py` (điều phối, sở hữu client), `relevance.py` (chỉ cung cấp signal cho policy, không lọc), `tools/retrieval.py` (Typer mỏng: chạy `retrieve(query)` trên corpus/index thật, in từng chunk, **cảnh báo rõ nếu `rerank_score=None`**; có `--query`, `--use-mmr/--no-use-mmr`). Contract giữa module là Pydantic; config `pydantic-settings` tập trung.

## 10. Tiêu chí hoàn thành

Query literal/paraphrase/citation đi qua cùng contract; citation hợp lệ có đường recall độc lập HyDE/RRF/MMR; dense index/query cùng preprocessing, sparse fit/document/query cùng tokenizer + structural-term convention; output ≤ K chunk đủ citation/metadata, không chứa candidate từ snapshot lệch; rerank lỗi vẫn trả fallback có thứ tự xác định, lỗi search nền tảng không bị che giấu; reranker đúng 1 inference cùng lúc process-wide kể cả nhiều event loop.

## 11. Áp dụng cho bài toán mới

**Baseline nhỏ nhất đáng tin: query gốc + dense/sparse song song + RRF + rerank bằng query gốc.** Chỉ thêm HyDE, MMR, citation extras khi giải một failure mode đã quan sát, và luôn giữ baseline để so sánh.
