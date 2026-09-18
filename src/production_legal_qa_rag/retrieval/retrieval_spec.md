# Retrieval — HyDE → Hybrid Search → RRF → MMR → Rerank

## 1. Mục tiêu & phạm vi

Từ 1 câu hỏi tiếng Việt của người dùng, trả về danh sách top-K chunk pháp lý
liên quan nhất, sẵn sàng làm context cho bước generation (LLM sinh câu trả
lời — spec khác, ngoài phạm vi).

Pipeline chia 2 giai đoạn (đã chốt cùng người dùng, 2026-09-18):

**Pre-retrieval:**

- **HyDE** — sinh 1 đoạn văn bản giả định (hypothetical document) trả lời
  câu hỏi bằng LLM, dùng embedding của đoạn này **song song với** embedding
  câu hỏi gốc, cả 2 đều dùng để dense search (mục 3) — quyết định
  2026-09-18: chạy dense search 2 lần (HyDE + câu hỏi gốc) thay vì chỉ dùng
  HyDE, để không bỏ lỡ candidate mà HyDE có thể paraphrase lệch hướng.

**Post-retrieval:**

- **Hybrid search**: dense search (2 lượt — HyDE + câu hỏi gốc, Pinecone
  index hiện có, xem `embedding_spec.md` mục 5) + sparse/keyword search
  trên `breadcrumb` + `content` (Pinecone sparse index mới, tự fit BM25,
  mục 4.2).
- **RRF Fusion** hợp nhất 3 danh sách kết quả — dense-HyDE, dense-câu hỏi
  gốc, sparse (mục 4.3).
- **MMR** chọn tập con đa dạng từ danh sách đã fusion (mục 5).
- **Rerank** bằng cross-encoder `AITeamVN/Vietnamese_Reranker`, gọi qua HTTP
  tới server người dùng tự host (LightningAI, đã triển khai xong — mục 6).

**Trong phạm vi:**

- Toàn bộ pipeline `retrieve(query) -> list[RetrievedChunk]` (mục 2, 9).
- Xây dựng & duy trì **Pinecone sparse index mới** (fit BM25Encoder trên
  corpus, upsert) — hạ tầng bắt buộc để hybrid search hoạt động, nhưng
  **không** mở rộng package `embedding/` đã merge (PR #16) — `embedding/`
  chỉ sở hữu dense index, sparse index là hạ tầng riêng của `retrieval/`
  (mục 4.2, 9A).

**Ngoài phạm vi (chủ động không làm):**

- Generation/sinh câu trả lời cuối cùng — spec khác.
- Multi-turn/lịch sử hội thoại — mỗi lần gọi `retrieve()` xử lý đúng 1 câu
  hỏi độc lập, không giữ ngữ cảnh hội thoại trước đó.
- Cache kết quả retrieval giữa các lần gọi.
- Tự triển khai/vận hành server reranker trên LightningAI — thuộc hạ tầng
  người dùng tự thiết lập (đã triển khai xong, runbook tại mục 6.1);
  `retrieval/` chỉ implement HTTP client gọi tới endpoint đã cấu hình (mục
  6, 8).
- Incremental update sparse index — nhất quán quyết định "full re-embed mỗi
  lần" (`embedding_spec.md` mục 1): mỗi lần build lại là fit BM25 + upsert
  lại toàn bộ từ đầu (mục 9A).
- CLI test query độc lập — `retrieval/` chỉ expose function `retrieve()`
  (đã xác nhận với người dùng 2026-09-18); bước `generation/` sau này là nơi
  gọi thực tế.

## 2. Input & Output

- **Input**: 1 câu hỏi tiếng Việt thô (`query: str`), chưa qua xử lý gì.
- **Output**: `list[RetrievedChunk]` (tối đa `FINAL_TOP_K` phần tử, mục 6),
  mỗi phần tử = toàn bộ field của `Chunk` (tái dùng từ
  `production_legal_qa_rag.chunking.models.Chunk`, không định nghĩa lại
  schema) cộng `rerank_score: float` — điểm relevance cuối cùng từ reranker.

Đầu vào tĩnh khác (không phải input của `retrieve()`, nhưng pipeline cần
đọc):

- `data/chunks/*.json` — corpus nguồn để fit BM25Encoder + build sparse
  index (mục 4.2, 9A).
- `data/bm25/bm25_params.json` — tham số `BM25Encoder` đã fit (IDF, độ dài
  tài liệu trung bình...), ghi ra khi build sparse index (mục 9A), đọc lại
  1 lần lúc khởi tạo pipeline để encode query tại thời điểm truy vấn (không
  fit lại mỗi query).

## 3. Pre-retrieval — HyDE

- Sinh 1 đoạn văn bản giả định (hypothetical document) trả lời trực tiếp
  câu hỏi, bằng LLM Groq — **tái dùng `LLMSettings` hiện có trong
  `config.py`** (model `openai/gpt-oss-120b`), theo quyết định của người
  dùng (2026-09-18): không thêm provider/model riêng cho query-time.
- Prompt cần thiết kế riêng cho domain pháp luật Việt Nam (vd. yêu cầu LLM
  viết đoạn văn ngắn theo văn phong luật, như thể trích từ văn bản luật
  thật) — **[cần chốt nguyên văn prompt cùng người dùng khi implement, spec
  này chỉ định hướng]**.
- **Dense search (mục 4.1) chạy song song 2 lượt — dùng embedding của
  `hypothetical_document` VÀ embedding câu hỏi gốc** (quyết định
  2026-09-18, thay cho thiết kế ban đầu chỉ dùng HyDE). Lý do đổi: HyDE
  viết theo văn phong pháp lý nằm gần không gian vector với chunk luật thật
  hơn, nhưng LLM có thể paraphrase lệch ý người dùng — retrieve thêm bằng
  embedding câu hỏi gốc để không bỏ lỡ candidate mà HyDE paraphrase hỏng.
  Cả 2 danh sách được coi ngang hàng ở bước RRF Fusion (mục 4.3).
- **Sparse/keyword search (mục 4.2) dùng nguyên văn câu hỏi gốc**, KHÔNG qua
  HyDE — BM25 cần match từ khoá chính xác người dùng gõ (số Điều/Khoản,
  thuật ngữ cụ thể); hypothetical document có thể paraphrase khác đi, làm
  giảm hiệu quả match từ khoá.

**Rủi ro vận hành cần lưu ý** (ngoài phạm vi xử lý của spec này, ghi nhận để
biết khi có traffic thật): gọi HyDE tại query-time nghĩa là mỗi câu hỏi tốn
thêm 1 lượt gọi Groq + **2 lượt gọi HF Inference API** (embed
`hypothetical_document` VÀ embed câu hỏi gốc — tăng gấp đôi so với thiết kế
ban đầu do chạy dense search song song 2 lượt, mục 6 dùng chung
`EmbeddingSettings` với `embedding/`). HF Inference API dùng chung quota
RPD 1.000/ngày với batch embedding (`embedding_spec.md` mục 4) — nếu
traffic query lớn trùng thời điểm chạy lại batch embed, có thể cộng dồn
vượt quota nhanh hơn trước (gấp đôi số lượt gọi/câu hỏi). Chưa có cơ chế
theo dõi quota liên-package.

## 4. Post-retrieval — Hybrid search & RRF Fusion

### 4.1. Dense search

Query đúng Pinecone dense index đã có (`VectorDBSettings.index_name`,
`embedding_spec.md` mục 5) — tái dùng toàn bộ index đã build, `retrieval/`
không tạo thêm dense index nào. Chạy **2 lượt query song song** (quyết định
2026-09-18, mục 3), mỗi lượt lấy top `DENSE_TOP_N` (khởi điểm = 20) theo
cosine similarity:

- Lượt 1: bằng embedding của `hypothetical_document`.
- Lượt 2: bằng embedding câu hỏi gốc.

2 danh sách kết quả này giữ riêng biệt, đưa cả 2 vào RRF Fusion cùng sparse
(mục 4.3) — không gộp/dedupe trước.

### 4.2. Sparse/keyword search — Pinecone sparse index riêng

Pinecone có sparse index tích hợp sẵn (`pinecone-sparse-english-v0`) nhưng
model đó tối ưu cho tiếng Anh. Thay vào đó dùng thư viện `pinecone-text`
(`BM25Encoder`), tự fit trên corpus tiếng Việt — tự chủ tokenization, không
phụ thuộc model sparse có sẵn của Pinecone.

- **Text đưa vào BM25**: `breadcrumb + " " + content` của từng chunk (theo
  đúng yêu cầu người dùng — search trên cả breadcrumb lẫn content).
  Breadcrumb chứa viện dẫn cụ thể (Điều/Khoản) mà người dùng có thể gõ trực
  tiếp trong câu hỏi, dense search một mình khó match tốt trường hợp này.
- **Word-segment bằng `pyvi` trước khi fit/encode** — cùng lý do đã áp dụng
  ở `chunking/tokenizer.py` (`chunking_spec.md` mục 6) và
  `embedding/hf_client.py` (`embedding_spec.md` mục 3): BM25 mặc định tách
  theo khoảng trắng, không segment sẽ tách nhầm ở mức âm tiết thay vì từ,
  giảm chất lượng match. Segment 1 lần khi fit + build index (mục 9A), và
  lại 1 lần mỗi query trước khi `encode_queries()`.
- **Fit toàn bộ corpus mỗi lần build**, không incremental — nhất quán quyết
  định "full re-embed mỗi lần" (`embedding_spec.md` mục 1): BM25 IDF là
  thống kê tĩnh trên toàn corpus, đổi khi thêm/bớt chunk nên phải fit lại từ
  đầu mỗi lần corpus thay đổi.
- **Lưu tham số đã fit**: `BM25Encoder.dump("data/bm25/bm25_params.json")`
  — load lại params này khi encode query tại thời điểm retrieval (không fit
  lại mỗi query — tốn thời gian, và IDF phải nhất quán giữa lúc build và lúc
  query).
- **Pinecone sparse index riêng** (tên mới, field `sparse_index_name` trong
  `VectorDBSettings`, mục 8) — serverless, `metric="dotproduct"` (chuẩn cho
  sparse vector, khác `cosine` của dense index). Xoá sạch + upsert lại toàn
  bộ mỗi lần build (nhất quán mục 5 `embedding_spec.md`).
- Query sparse: `BM25Encoder.encode_queries([query_gốc_đã_segment])` →
  sparse vector → `sparse_index.query(sparse_vector=..., top_k=SPARSE_TOP_N)`
  (khởi điểm = 20).

### 4.3. RRF Fusion

Hợp nhất **3 danh sách** (dense-HyDE top-N, dense-câu hỏi gốc top-N,
sparse top-N — quyết định 2026-09-18, mục 3/4.1) theo `chunk_id`, công
thức RRF chuẩn mở rộng cho 3 chiều:

```
rrf_score(chunk) = Σ 1 / (k + rank_trong_mỗi_danh_sách)   # cộng dồn qua cả 3 danh sách
```

`k = 60` (hằng số gốc từ paper RRF, giá trị phổ biến, không cần tinh chỉnh
ban đầu). Chunk chỉ xuất hiện ở 1 hoặc 2 trong 3 danh sách vẫn được tính
(chỉ cộng số hạng của danh sách nó có mặt, không phạt vì thiếu danh sách
kia). Sort giảm dần theo `rrf_score`, cắt còn `FUSION_TOP_N` (khởi điểm =
20) đưa sang bước MMR (mục 5).

**Lưu ý kỹ thuật quan trọng**: candidate chỉ đến từ sparse (không nằm
trong top của CẢ 2 lượt dense) sẽ **không có sẵn dense vector** để tính
similarity ở bước MMR (mục 5) — cần `dense_index.fetch(ids=[...])` bổ sung
cho các `chunk_id` này trước khi chạy MMR (Pinecone hỗ trợ fetch vector
theo id trực tiếp, không cần embed lại).

## 5. Post-retrieval — MMR (Maximal Marginal Relevance)

Áp dụng ngay sau RRF fusion (mục 4.3), **trước** reranker (mục 6) — đúng
thứ tự người dùng chốt (2026-09-18): giảm số lượng candidate đưa vào
reranker (cross-encoder tốn compute theo từng cặp, mục 6) đồng thời loại
bớt trùng lặp ngữ nghĩa trước khi rerank.

Công thức chuẩn:

```
MMR = argmax_{d ∈ candidates \ selected} [ λ·sim(d, query) − (1−λ)·max_{d' ∈ selected} sim(d, d') ]
```

- `sim(d, query)`: cosine similarity giữa dense embedding của candidate và
  embedding **câu hỏi gốc** (quyết định 2026-09-18 — phản ánh đúng ý định
  thật của người dùng, không bị lệch bởi paraphrase của HyDE) — tái dùng
  vector đã có từ lượt dense search bằng câu hỏi gốc (mục 4.1), không gọi
  embed thêm.
- `sim(d, d')`: cosine similarity giữa dense embedding 2 candidate với
  nhau.
- `λ = 0.5` (khởi điểm, cân bằng relevance/diversity — giá trị phổ biến,
  điều chỉnh sau khi có dữ liệu đánh giá thật).
- Lặp chọn tuần tự tới khi đủ `MMR_TOP_N` (khởi điểm = 10) hoặc hết
  candidate.

## 6. Post-retrieval — Reranker

Model: `AITeamVN/Vietnamese_Reranker` (cross-encoder 0.6B tham số, base
`bge-reranker-v2-m3`, input tối đa 2304 token — 256 cho query, 2048 cho
passage). **Không có sẵn trên HF Inference API** — không Inference Provider
nào deploy model này (tra cứu 2026-09-18) — nên gọi qua **server nội bộ
người dùng tự host trên LightningAI**. Server đã triển khai và test thành
công end-to-end (2026-09-18) — code tại `reranker_server/` (thư mục riêng ở
root repo, xem mục 6.1 cho chi tiết vận hành).

- `retrieval/reranker_client.py`: HTTP client (thư viện `httpx`, đã chốt —
  có sẵn transitive qua `groq` SDK nên không tốn thêm dung lượng cài đặt,
  nhất quán với phần còn lại của dự án) gọi `POST {endpoint_url}` với header
  `X-API-Key: {api_key}` (cơ chế auth built-in của LitServe — framework
  server dùng), payload `{"query": câu_hỏi_gốc, "passages": [content của
  từng candidate sau MMR]}`, response `{"scores": [float, ...]}` cùng thứ
  tự. **Schema đã xác nhận bằng test thực tế** (`retrieval/test.py`, gọi
  thẳng server thật với 3 chunk mẫu, kết quả đúng kỳ vọng) — không còn là
  giả định.
- Input reranker chỉ dùng `content` (không kèm breadcrumb) — nhất quán lý
  do breadcrumb "không đưa vào chuỗi embed" (`chunking_spec.md` mục 2):
  breadcrumb là metadata định vị, không phải nội dung ngữ nghĩa cần đánh
  giá độ liên quan.
- Query đưa vào reranker là **câu hỏi gốc của người dùng** (không phải
  `hypothetical_document`) — reranker cần đánh giá độ liên quan giữa câu
  hỏi THẬT và passage; dùng hypothetical document ở đây sẽ đánh giá sai mục
  tiêu (HyDE chỉ hỗ trợ dense search, không phải câu hỏi thật, mục 3).
- Retry/timeout: cấu hình riêng `RerankerSettings` (mục 8), cùng tinh thần
  `max_retries`/`timeout_seconds` của `LLMSettings`.
- Sort giảm dần theo score trả về, cắt còn `FINAL_TOP_K` (khởi điểm = 5) —
  đây là output cuối cùng của `retrieve()` (mục 2).
- **[Cần chốt cùng người dùng]** Lỗi gọi server (hết retry): đề xuất
  fallback trả kết quả theo thứ tự MMR (bỏ qua rerank) kèm log warning —
  nhất quán tinh thần "1 phần lỗi không chặn toàn bộ" đã áp dụng ở
  `formatting/`/`embedding/` — nhưng đây là bước cuối ảnh hưởng trực tiếp
  UX nên cần xác nhận rõ thay vì mặc định.

### 6.1. Vận hành reranker server (runbook)

Ngoài phạm vi implement của `retrieval/` (mục 1) nhưng ghi lại ở đây vì
`retrieve()` phụ thuộc trực tiếp vào server này đang chạy — không có nó,
bước rerank (và cả `retrieve()`) không hoạt động được.

- **Code**: `reranker_server/` ở root repo — dự án `uv` hoàn toàn độc lập
  (`pyproject.toml`/`uv.lock` riêng, có `torch`/`transformers`/`litserve`),
  KHÔNG phải dependency của package `retrieval/` hay root `pyproject.toml`.
  Chỉ chạy trên LightningAI Studio, không chạy/cài trên máy local.
- **Hạ tầng**: 1 LightningAI Studio (CPU, free tier), SSH vào bằng
  `LIGHTNING_STUDIO_SSH` (mục 8, biến này không đổi khi Studio restart).
  Trong Studio, 2 tiến trình chạy nền bằng `tmux`:
  - session `reranker`: `cd reranker_server && uv run server.py` — chạy
    LitServe, lắng nghe cổng 8000, auth qua `LIT_SERVER_API_KEY` (đặt trong
    `reranker_server/.env`, phải trùng giá trị `RERANKER_API_KEY` ở `.env`
    gốc repo).
  - session `ngrok_tunnel`: `ngrok http 8000 --log stdout` — expose cổng
    8000 ra domain tĩnh free của tài khoản ngrok (không đổi mỗi lần
    restart, khác với tunnel ngẫu nhiên mặc định) → chính là
    `RERANKER_ENDPOINT_URL`.
  - Lý do dùng ngrok thay vì tính năng port-forward/Deploy có sẵn của
    Lightning: link "Forwarded Address" trong tab Ports của Studio chỉ là
    proxy giao diện web có xác thực (trả về trang Lightning, không phải
    reverse-proxy HTTP thật); còn "Lightning Deploy" (Autoscale) bị lỗi hạ
    tầng (replica crash-loop ngay ở bước Setup, không tới được bước chạy
    code) khi thử nghiệm 2026-09-18 — không debug được, dừng lại dùng
    ngrok cho use case 1 người dùng.
- **Rủi ro vận hành đã biết**: cả 2 tmux session chết khi Studio bị
  restart (auto-sleep do rảnh, hoặc chủ động tắt/mở lại) — cần SSH vào
  chạy lại cả 2 lệnh trên trước khi gọi `retrieve()`/`reranker_client.py`,
  không có cơ chế tự khởi động lại (chưa cần thiết cho use case cá nhân,
  low-traffic).

## 7. Tools & Integrations

| Việc | Công cụ |
| --- | --- |
| LLM sinh HyDE hypothetical document | `groq` SDK, tái dùng `LLMSettings` |
| Embed hypothetical document + câu hỏi gốc (2 lượt/query, mục 3) | `huggingface_hub.InferenceClient`, tái dùng `EmbeddingSettings` (cùng model `embedding/hf_client.py` dùng) |
| Sparse/keyword encode (BM25) | `pinecone-text` (`BM25Encoder`) — **dependency mới**, thêm vào `pyproject.toml` |
| Word-segment tiếng Việt trước BM25 | `pyvi` (đã có sẵn) |
| Dense + sparse vector search | `pinecone` SDK (đã có sẵn) |
| RRF Fusion, MMR | Tự cài thuật toán thuần Python — logic đơn giản (mục 4.3, 5), không cần thư viện ngoài |
| Rerank | HTTP client gọi server LightningAI — `httpx` (đã chốt, mục 6) |
| Data models | `pydantic` v2 |

## 8. Config tập trung (`src/production_legal_qa_rag/config.py`)

- `VectorDBSettings` thêm field mới:
  `sparse_index_name: str = Field(validation_alias="PINECONE_SPARSE_INDEX_NAME")`
  — tên Pinecone sparse index riêng (mục 4.2).
- `RerankerSettings` (class mới):

  ```python
  class RerankerSettings(BaseSettings):
      endpoint_url: str = Field(validation_alias="RERANKER_ENDPOINT_URL")
      max_retries: int = 2
      timeout_seconds: int = 30
  ```

  cùng pattern `LLMSettings`. `endpoint_url` bắt buộc, không có default —
  giá trị thật đọc từ `.env` (`RERANKER_ENDPOINT_URL`, cùng `RERANKER_API_KEY`
  gửi qua header `X-API-Key`, mục 6).
- `LIGHTNING_STUDIO_SSH` trong `.env` — **không** thuộc `RerankerSettings`
  hay bất kỳ Settings nào trong `config.py` (không phải config app đọc lúc
  chạy `retrieve()`), chỉ là ghi chú vận hành để biết cách SSH khởi động lại
  server khi cần (mục 6.1).
- **Không** đưa `DENSE_TOP_N`/`SPARSE_TOP_N`/`FUSION_TOP_N`/`RRF_K`/
  `MMR_LAMBDA`/`MMR_TOP_N`/`FINAL_TOP_K` vào `config.py` — hằng số nội bộ cơ
  chế riêng của `retrieval/`, nhất quán tinh thần mục 7 `embedding_spec.md`
  (chỉ field dùng chung nhiều package mới vào `config.py`).

## 9. Workflow & quản lý trạng thái

Chia 2 luồng độc lập:

### 9A. Setup/offline — build sparse index

Chạy 1 lần hoặc mỗi khi corpus (`data/chunks/*.json`) thay đổi, cùng tinh
thần batch job của `embedding/`:

```
tools/index_documents.py (Typer CLI)
  → retrieval.sparse_index.build_index(chunks_dir, params_out_path)
      1. Đọc data/chunks/*.json -> list[Chunk]
      2. Với mỗi chunk: text = breadcrumb + " " + content, segment bằng pyvi
      3. BM25Encoder().fit(all_texts) -> lưu data/bm25/bm25_params.json
      4. encode_documents(all_texts) -> sparse vectors
      5. Tạo Pinecone sparse index nếu chưa có (metric="dotproduct")
      6. Xoá sạch index cũ, upsert toàn bộ (id=chunk_id, sparse_values=...)
      7. In summary: tổng số chunk đã index
```

### 9B. Online — `retrieve(query)`

Library function, không CLI (mục 1):

```
retrieval.pipeline.retrieve(query: str) -> list[RetrievedChunk]
  1. hypothetical_document = hyde.generate(query)                        # mục 3, Groq
  2. hyde_embedding = query_embedder.embed(hypothetical_document)        # mục 3, HF API
  3. query_embedding = query_embedder.embed(query)                       # mục 3, HF API — song song với bước 2
  4. dense_results_hyde = dense_search.query(hyde_embedding, DENSE_TOP_N)    # mục 4.1, lượt 1
  5. dense_results_query = dense_search.query(query_embedding, DENSE_TOP_N) # mục 4.1, lượt 2
  6. sparse_results = sparse_index.query(query, SPARSE_TOP_N)            # mục 4.2, tự segment+encode query gốc
  7. fused = fusion.rrf(dense_results_hyde, dense_results_query, sparse_results, k=RRF_K)[:FUSION_TOP_N]  # mục 4.3, 3 chiều
  8. fetch dense vector còn thiếu cho candidate chỉ đến từ sparse         # mục 4.3
  9. selected = mmr.select(fused, query_embedding, MMR_LAMBDA, MMR_TOP_N) # mục 5, dùng embedding CÂU HỎI GỐC
  10. scores = reranker_client.rerank(query, [c.content for c in selected])  # mục 6, dùng query GỐC
  11. sort theo scores, trả về top FINAL_TOP_K dạng list[RetrievedChunk]
```

Stateless giữa các lần gọi `retrieve()` — không cache, không session. Chỉ
`bm25_params.json` (mục 2, 9A) là trạng thái persist, đọc 1 lần lúc khởi
tạo pipeline, dùng lại cho mọi query.

## 10. Cấu trúc module trong `src/production_legal_qa_rag/retrieval/`

| Module | Trách nhiệm |
| --- | --- |
| `models.py` | `RetrievedChunk` (`Chunk` + `rerank_score`), model trung gian nếu cần cho candidate sau fusion/MMR |
| `hyde.py` | Gọi Groq sinh hypothetical document (mục 3) |
| `query_embedder.py` | Embed 1 chuỗi qua HF Inference API, tái dùng `EmbeddingSettings` (mục 3) — tách riêng khỏi `embedding/hf_client.py` vì module đó gắn với luồng batch theo `Chunk`/file, không phải embed 1 chuỗi rời tại query-time (cùng lý do `embedding_spec.md` mục 3 không tái dùng `chunking/tokenizer.py`) |
| `sparse_index.py` | Fit/persist `BM25Encoder`, tạo/xoá/upsert Pinecone sparse index, encode+query tại runtime (mục 4.2, 9A) |
| `dense_search.py` | Query Pinecone dense index có sẵn (mục 4.1) |
| `fusion.py` | Thuật toán RRF (mục 4.3) |
| `mmr.py` | Thuật toán MMR (mục 5) |
| `reranker_client.py` | HTTP client gọi server LightningAI (mục 6) |
| `pipeline.py` | `retrieve(query) -> list[RetrievedChunk]` — điều phối toàn bộ (mục 9B) |
| `__init__.py` | |

`tools/index_documents.py` — Typer CLI mỏng gọi `sparse_index.build_index()`
(mục 9A), tách biệt hoàn toàn khỏi `pipeline.retrieve()` (mục 9B).

`retrieval/test.py` — script test thủ công gọi thẳng reranker server thật
(3 chunk mẫu từ `data/chunks/`), dùng để xác nhận end-to-end lúc server mới
host xong (mục 6.1) — không thuộc kiến trúc chính thức ở trên, xoá/thay thế
khi có test suite thật (`tester` agent) cho `reranker_client.py`.

## 11. Tiêu chí hoàn thành

- `tools/index_documents.py` chạy trên toàn bộ `data/chunks/*.json` hiện có
  không crash, sinh `data/bm25/bm25_params.json` và Pinecone sparse index
  chứa đúng số vector = tổng số chunk.
- `retrieve(query: str)` chạy end-to-end (HyDE → dense+sparse → RRF → MMR →
  rerank) trả về đúng tối đa `FINAL_TOP_K` phần tử `RetrievedChunk`, không
  crash với ít nhất 3 câu hỏi thật khác nhau — kể cả câu hỏi chứa viện dẫn
  cụ thể (vd. "Điều 3 khoản 1 quy định gì") để xác nhận sparse/keyword
  search hoạt động.
- Với câu hỏi chứa viện dẫn cụ thể, xác nhận thủ công candidate tương ứng
  xuất hiện trong top sparse search (breadcrumb match) dù dense search có
  thể xếp thấp candidate đó.
- Rerank server lỗi (giả lập timeout) không làm crash toàn bộ `retrieve()`
  — fallback đúng theo quyết định mục 6.
- `config.py` có đủ `sparse_index_name`/`RerankerSettings` mới; các module
  trong `retrieval/` không đọc `.env` trực tiếp.

## 12. Các điểm cần xác nhận thêm trước khi implement

Gom lại toàn bộ giả định/điểm mở đã đánh dấu rải rác trong spec, để rà soát
1 lần. (Đã chốt 2026-09-18: request/response schema reranker — test thành
công với server thật, mục 6; thư viện HTTP client — `httpx`, mục 6/7.)

1. Nguyên văn prompt HyDE (mục 3).
2. Chiến lược fallback khi reranker lỗi (mục 6) — đề xuất fallback về thứ
   tự MMR, cần xác nhận rõ vì ảnh hưởng UX trực tiếp.
3. Các hằng số khởi điểm — `DENSE_TOP_N=20`, `SPARSE_TOP_N=20`, `RRF_K=60`,
   `FUSION_TOP_N=20`, `MMR_LAMBDA=0.5`, `MMR_TOP_N=10`, `FINAL_TOP_K=5` —
   chưa có dữ liệu đánh giá thật, cần tinh chỉnh sau khi có eval set.
4. Rủi ro chia sẻ quota HF Inference API RPD giữa `embedding/` batch job và
   `retrieval/` query-time (mục 3) — chưa có cơ chế theo dõi chung.
