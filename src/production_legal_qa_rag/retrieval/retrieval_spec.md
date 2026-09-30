# Retrieval — Legal Question → Evidence Chunks: Reference Spec

> Cô đọng 2026-09-30 (giữ số mục vì code/spec khác tham chiếu). Bản đầy đủ: git history.

## 1. Mục đích

Từ một câu hỏi, trả về tập nhỏ `RetrievedChunk` đủ liên quan và có citation để generation trả lời dựa trên bằng chứng.

> **Tìm theo nghĩa và theo định danh pháp lý, rồi để reranker chọn bằng chứng cuối cùng.**

Luật có hai loại tín hiệu: **ý định** ("người lao động được nghỉ trong trường hợp nào?") và **định danh** ("Khoản 1 Điều 113
BLLĐ"). Dense mạnh về ý định nhưng không đáng tin với số Điều/Khoản; sparse mạnh về định danh nhưng yếu với paraphrase →
giữ cả hai, không ép một phương pháp giải toàn bộ.

**Phạm vi:** `retrieve(query)` trả tối đa `FINAL_TOP_K` chunk; hybrid dense/sparse, citation-aware recall, rerank, fallback có
chủ đích; build offline sparse index từ chunk đã embed. **Ngoài phạm vi:** chunking/embedding corpus/generation/hội thoại/cache;
quyết định trả lời hay từ chối (policy của conversation/API); agentic decomposition cho câu nhiều Điều/Điểm/ý định. Baseline
chất lượng: câu hỏi viện dẫn một Khoản phải đưa chunk đáp án vào candidate cuối; câu rộng hơn chạy cùng pipeline nhưng không hứa chất lượng chưa đo.

## 2. Contract và ranh giới trách nhiệm

`retrieve(query) -> list[RetrievedChunk]` (≤ `FINAL_TOP_K`, xếp theo `rerank_score` khi rerank thành công). `RetrievedChunk`
(Pydantic v2): `chunk_id`, `source_document`, `breadcrumb`, `content` (nguyên văn), `has_table`, `raw_table`, `rerank_score`
(`None` khi fallback). Retrieval **luôn trả evidence tốt nhất tìm được, không lọc thành rỗng vì score thấp**; tầng policy dùng
`rerank_score` đã hiệu chỉnh để quyết `no_context`/làm rõ/tiếp tục — nên primitive dùng lại được cho debug/research.

**Đường vào cho evaluation (2026-09-29, `evaluation_spec.md` mục 11):** `RetrievalPipeline.retrieve(query, *, use_mmr=None,
precomputed=None)`; `PrecomputedQuery` (`models.py`): `hypothetical_document`, `hypothetical_embedding`, `query_embedding`
(`hypothetical_document` là `None` khi và chỉ khi `hypothetical_embedding` là `None`). Có `precomputed` thì bỏ HyDE + embed và
dùng giá trị đó; mọi bước sau giữ NGUYÊN code; mặc định `None` = hành vi cũ. Mục đích duy nhất: eval tách HyDE/embed thành stage
lưu file nhưng vẫn đo đúng code production. Hàm module-level `retrieve()` không lộ tham số này.

## 3. Luồng online

Query gốc → (nhánh A: HyDE → hypothetical text, best-effort) + (nhánh B: query gốc, luôn có) → embed từng text bằng cùng
preprocessing index-time → dense + sparse song song mỗi nhánh → RRF → MMR tuỳ chọn → union theo `chunk_id` → citation extras từ
sparse nhánh B → bổ sung metadata/vector thiếu → rerank bằng query gốc → top K. Nhánh A tăng recall semantic; nhánh B là nguồn
sự thật cho literal wording, số Điều/Khoản, tên văn bản. Hai nhánh dùng cùng code search/fusion, khác text/vector/structural terms.

HyDE chỉ sinh văn phong/ngữ nghĩa pháp lý, **không được bịa số Điều/Khoản, tên văn bản, năm, mức số**. Hypo hỏng/rỗng → bỏ nhánh
A, không làm hỏng nhánh B. Từ 2026-09-28 HyDE chạy `gpt-oss-20b` (trước 120b) để nhường bucket 120b cho generation;
`MIN_RERANK_SCORE` (`relevance.py`) giữ nguyên, không đo lại (`conversation_spec.md` mục 12.1) — lệch chất lượng sẽ thấy ở RAGAS.

## 4. Dense, sparse và fusion

- **Dense:** embed query bằng đúng model + preprocessing lúc index (word-segment `pyvi`); lệch là lỗi semantic im lặng nên
  index-time/query-time là một contract. Dense query trả metadata; chỉ lấy vector khi MMR cần. `retrieval/` chỉ đọc dense index.
- **Sparse BM25:** text = `breadcrumb + " " + content` (breadcrumb đưa citation vào lexical index). Một tokenizer nhất quán cho
  fit/document/query; **không mặc định stopword/stemming tiếng Anh** (phá thuật ngữ pháp lý). BM25 tự viết được nhưng công
  thức/vocabulary/thứ tự ID phải deterministic; params lưu versioned, runtime từ chối params khác version.
- **RRF:** không cộng thẳng dense score với BM25 score (thang khác nhau). Dedupe bằng `chunk_id`, giữ candidate chỉ ở một nhánh,
  cắt pool nhỏ trước rerank.

## 5. Citation-aware retrieval

Cơ chế riêng cho định danh chính xác, không phải prompt trick. **Parse thận trọng:** trích Điều/Khoản chỉ từ query gốc; ưu tiên
false negative hơn false positive (năm, tiền, tuổi, đơn vị thời gian, từ "điều kiện" không thành Điều); tên văn bản mơ hồ/nhiều
văn bản thì không đoán document key. **Structural terms:** token hiếm, ổn định ở cả document và query (`điều_113`, `khoản_1`,
`điều_113_khoản_1`, `vb_blld_điều_113_khoản_1`) bổ sung vào sparse vector, không thay text gốc/dense; **chỉ nhánh B** nhận token
từ citation người dùng (HyDE không được sinh). Mapping `source_document` → document key phải explicit, versioned, cảnh báo khi
corpus có văn bản chưa map. **Citation extras:** RRF/MMR có thể làm rơi chunk citation đúng dù cao ở sparse → khi query có
citation hợp lệ, thêm top sparse thô của nhánh B vào union trước rerank (dùng lại kết quả sparse, không thêm round trip; số
extras bị chặn).

## 6. Diversity và rerank

MMR có thể phạt oan các Khoản gần nhau vốn cùng cần cho câu hỏi luật → **là cờ runtime/evaluation, không phải "tối ưu" được tin
sẵn**; tắt thì giữ đầu RRF làm baseline. Reranker là quyết định cuối: query là câu hỏi gốc (không phải HyDE); passage =
`breadcrumb + "\n" + content` (content công khai/text embed không đổi); gửi cả candidate pool một lượt (có thể chia batch nội bộ,
6.1), validate score hữu hạn + đúng thứ tự, sort giảm dần, cắt top K. **Không ghim kết quả bằng rule sau rerank**; muốn bắt buộc
citation recall thì làm ở candidate stage qua extras.

### 6.1 Inference tại chỗ (local, không host API riêng)

Model `AITeamVN/Vietnamese_Reranker` (fine-tune từ `bge-reranker-v2-m3`) chạy in-process trong `RetrievalPipeline` (bỏ hẳn
LightningAI/ngrok của bản cũ).
- **Device:** tự phát hiện `cuda`, fallback `cpu`; GPU dùng fp16 (`model.half()`), CPU fp32. CPU là đường hợp lệ, không phải
  lỗi. Log `INFO` device đúng một lần lúc load model.
- **Vòng đời:** load tokenizer + model một lần lúc khởi tạo pipeline, giữ suốt process.
- **`MAX_LENGTH = 512`** = giới hạn cho **tổng** `token(query) + token(breadcrumb + "\n" + content) + 4 token đặc biệt`, không
  riêng passage. Là **ước lượng có margin, KHÔNG đo trên corpus** (người dùng chủ động bỏ bước đo): `content` ≤ 236 token
  PhoBERT ×1.5 ≈ 400 (tokenizer SentencePiece đa ngôn ngữ của bge ra nhiều token hơn PhoBERT — **không suy thẳng 236 sang model
  này**) + breadcrumb ~40 + query 64 + 4 → ~508, làm tròn 512. Đo lại nếu log cảnh báo truncation hoặc chất lượng rerank giảm.
- **Batch:** pool tối đa `2 × (DENSE_TOP_N + SPARSE_TOP_N)` = 80 passage; chia batch cố định `RERANK_BATCH_SIZE` (~16) để chặn đỉnh VRAM.
- **Không chặn event loop:** forward là blocking → chạy trong executor.
- **Concurrency inference process-wide = 1 (bài học GPU 2GB):** nhiều request đồng thời → nhiều forward cùng lúc trên cùng
  `self._model` → VRAM cộng dồn → CUDA OOM. Bọc mỗi lần `rerank()` (gồm mọi batch) trong `ThreadPoolExecutor(max_workers=1)`
  private, module-scoped, dùng chung process; request khác await future nên không chặn loop/thread. Cứng `1`, không đưa vào
  `RerankerSettings`; áp đồng nhất cho CUDA và CPU (quy mô app tự giới hạn, đã có admission/quota ở trên). **Không dùng
  `asyncio.Semaphore`/`LoopBoundClient` cho limiter này**: semaphore là loop-bound và `LoopBoundClient` tạo semaphore riêng khi loop đổi,
  nên hai loop song song có hai permit và cùng forward một model; `ThreadPoolExecutor` là primitive thread-safe của process nên đúng
  qua nhiều `asyncio.run()` và nhiều loop.

## 7. Consistency và state offline

Dense index, sparse index và BM25 params phải mô tả **cùng một corpus snapshot**. Mỗi build sparse ghi manifest/version: corpus
fingerprint/version manifest embedding, số chunk + ID/checksum, tokenizer/BM25 params + mapping document version, dense build mà
sparse dựa vào. Runtime chỉ load params/index tương thích dense snapshot, **không "best effort" khi lệch**; candidate có ở sparse
mà không fetch được metadata từ dense bị bỏ và log như tín hiệu build lệch. Sparse build offline, full refresh; khi phục vụ
traffic dùng snapshot/namespace versioned rồi switch sau khi count khớp (`delete_all → upsert` không phải transaction).

## 8. Xử lý lỗi

| Sự cố | Hành vi |
| --- | --- |
| HyDE lỗi/rỗng/`ThrottleTimeout` | Bỏ nhánh A, chạy nhánh B |
| Reranker lỗi (runtime/model, ví dụ CUDA OOM) | Fallback deterministic từ các nhánh, `rerank_score=None` |
| Query embed, dense/sparse search, metadata fetch lỗi | `RetrievalError`; không giả vờ có evidence |
| Corpus version mismatch | Từ chối khởi tạo/query, nêu thao tác rebuild |

Retry chỉ cho lỗi tạm thời của dịch vụ ngoài (HF embedding, Pinecone), timeout hữu hạn. Reranker in-process: lỗi cùng input là
deterministic, retry vô nghĩa → fallback ngay, log đủ (kích thước batch, độ dài passage).

## 9. Module boundaries

`models.py` (contract + `RetrievalError`), `hyde.py` (best-effort; từ 2026-09-28 dùng `HydeSettings` riêng: `gpt-oss-20b`,
`GROQ_API_KEY_1`, không dùng `LLMSettings` của `formatting/`), `llm_throttle.py` (`TokenWindowThrottle` chung cho bucket
`gpt-oss-20b`: condense/HyDE/Judge, `conversation_spec.md` mục 12.1), `query_embedder.py`, `bm25.py`, `citation.py` (parse,
mapping, structural terms, extras), `dense_search.py`/`sparse_index.py`, `fusion.py`/`mmr.py` (thuần, không I/O), `reranker.py`
(load 1 lần, batch inference, giới hạn 1 request process-wide, validate, fallback), `pipeline.py` (điều phối, sở hữu client),
`relevance.py` (chỉ cung cấp signal cho policy, không lọc), `tools/retrieval.py` (Typer mỏng: chạy `retrieve(query)` trên
corpus/index thật để kiểm chứng reranker — in tổng thời gian, từng chunk, **cảnh báo rõ nếu `rerank_score=None`**; có bộ câu hỏi
preset, `--query`, `--use-mmr/--no-use-mmr`). Contract giữa module là Pydantic; config `pydantic-settings` tập trung.

## 10. Tiêu chí hoàn thành

Query literal/paraphrase/citation đi qua cùng contract; citation hợp lệ có đường recall độc lập HyDE/RRF/MMR, không thêm round
trip sparse; dense index/query cùng preprocessing, sparse fit/document/query cùng tokenizer + structural-term convention; output
≤ K chunk đủ citation/metadata, không chứa candidate từ snapshot lệch; rerank lỗi vẫn trả fallback có thứ tự xác định, lỗi search
nền tảng không bị che giấu; `MAX_LENGTH=512` là ước lượng (đo lại nếu có dấu hiệu truncation); reranker đúng 1 inference cùng
lúc process-wide kể cả nhiều event loop; tầng conversation áp policy score mà không làm đổi/mơ hồ contract `retrieve()`.

## 11. Áp dụng cho bài toán mới

Chốt trước khi code: nhiệm vụ retrieval đo được + loại query cần bảo đảm recall; tín hiệu semantic/literal/cấu trúc đặc thù domain
(chỉ thêm structural terms khi bù điểm mù có bằng chứng); contract preprocessing chung + corpus snapshot/version; candidate
budget, tiêu chí dùng MMR/reranker, latency thực tế; ranh giới degrade/fail-fast/policy. **Baseline nhỏ nhất đáng tin: query gốc +
dense/sparse song song + RRF + rerank bằng query gốc.** Chỉ thêm HyDE, MMR, citation extras khi giải một failure mode đã quan sát,
và luôn giữ baseline để so sánh.
