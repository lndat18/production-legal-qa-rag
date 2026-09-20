# Retrieval — HyDE → Hybrid Search (2 nhánh) → RRF → MMR (bật/tắt) → Union → Rerank

## 1. Mục tiêu & phạm vi

Từ 1 câu hỏi tiếng Việt của người dùng, trả về top-5 chunk pháp lý liên quan
nhất, sẵn sàng làm context cho bước generation (LLM sinh câu trả lời — spec
khác, ngoài phạm vi).

Flow đã chốt cùng người dùng (2026-09-19):

```
query → Groq (1 call) → hypothetical_document
      → HF embed [hypothetical_document, query] (1 call, 2 vector)
      → 2 nhánh song song (HyDE / câu hỏi gốc), mỗi nhánh: dense + sparse → RRF → MMR (bật/tắt được, mục 7)
      → union theo chunk_id
      → Reranker tự host (1 call; passage = breadcrumb + "\n" + content, mục 9)
      → top FINAL_TOP_K = 5
```

**Trong phạm vi:**

- Toàn bộ pipeline `retrieve(query)` (mục 3, 13B).
- Xây dựng & duy trì **Pinecone sparse index riêng** (fit BM25 tự viết trên
  corpus, upsert) — hạ tầng cho keyword search, **không** mở rộng package
  `embedding/` đã merge (PR #16): `embedding/` chỉ sở hữu dense index (mục
  6.2, 13A).
- Xử lý lỗi cho 2 điểm hay lỗi nhất: Groq và reranker tự host (mục 9, 10).

**Ngoài phạm vi (chủ động không làm):**

- Generation/sinh câu trả lời cuối cùng — spec khác.
- Multi-turn/lịch sử hội thoại — mỗi lần gọi `retrieve()` xử lý đúng 1 câu
  hỏi độc lập.
- Cache kết quả retrieval giữa các lần gọi.
- Tự triển khai/vận hành server reranker — thuộc hạ tầng người dùng tự thiết
  lập; `retrieval/` chỉ có HTTP client + runbook khởi động (mục 9.1).
- Incremental update sparse index — nhất quán "full re-embed mỗi lần"
  (`embedding_spec.md` mục 1): mỗi lần build là fit BM25 + upsert lại toàn bộ.
- Kiểm tra đồng bộ tự động giữa dense index, sparse index và `bm25_params.json` — quy ước vận hành: chạy lại `tools/sparse_index_documents.py` sau mỗi lần build lại dense index (mục 13A).
- CLI test query độc lập — `retrieval/` chỉ expose function `retrieve()`.
- **Tối ưu latency/API call nâng cao** (benchmark giới hạn batch của HF/Groq/
  Lightning, dynamic batching, circuit breaker, đo đường găng) — **bàn sau**
  khi flow này chạy đúng; spec này chỉ giữ các quyết định đã nằm trong flow
  (1 call Groq, 1 call HF gộp 2 text, 1 call rerank, 2 nhánh song song).

## 2. Input & Output

- **Input**: 1 câu hỏi tiếng Việt thô (`query: str`).
- **Output**: `list[RetrievedChunk]`, tối đa `FINAL_TOP_K` (=5) phần tử. Sắp
  xếp giảm dần theo độ liên quan.

`RetrievedChunk` (pydantic v2, định nghĩa trong `retrieval/models.py`) gồm
đúng các field lấy được từ Pinecone metadata
(`embedding.models.PineconeMetadata`), cộng điểm rerank:

| Field                                            | Ghi chú                                                                          |
| ------------------------------------------------ | --------------------------------------------------------------------------------- |
| `chunk_id`                                     | id vector Pinecone                                                                |
| `source_document`, `breadcrumb`, `content` | từ metadata                                                                      |
| `has_table`, `raw_table`                     | từ metadata                                                                      |
| `rerank_score: float \| None`                   | điểm reranker;`None` nếu bước rerank thất bại và đã fallback (mục 9) |

Metadata Pinecone **không** lưu `token_count`, `is_split`, `split_*`,
`standardization_table` nên `RetrievedChunk` không có các field này
(generation không cần chúng; chốt 2026-09-19).

Đầu vào tĩnh khác (không phải input của `retrieve()`, nhưng pipeline cần
đọc):

- `data/chunks/*.json` — corpus nguồn để fit BM25 + build sparse index (mục
  13A).
- `data/bm25/bm25_params.json` — tham số BM25 đã fit (vocabulary, IDF, độ dài
  trung bình; mục 6.2), ghi ra khi build sparse
  index, đọc 1 lần lúc khởi tạo pipeline để encode query (không fit lại mỗi
  query).

## 3. Flow tổng thể

```
                       ┌─ Nhánh A (text = hypothetical_document) ─┐
query → Groq → hypo ──►│   dense(emb_hypo) ┐                       │
        │              │   sparse(hypo)    ┴► RRF → MMR ──────────┤
        └► HF embed ──►│                                           ├► union → Rerank(query gốc, breadcrumb+"\n"+content) → top 5
          [hypo,query] │ Nhánh B (text = query gốc)                │
                       │   dense(emb_query) ┐                      │
                       └   sparse(query)    ┴► RRF → MMR ─────────┘
```

- Các bước tuần tự: Groq → HF embed → (2 nhánh song song) → union → rerank.
  Trong bước 2 nhánh, 2 nhánh chạy đồng thời, và trong mỗi nhánh dense +
  sparse cũng chạy đồng thời (`asyncio.gather`). Khởi động sớm nhánh B trước
  khi Groq xong là tối ưu, để sau.
- **Cả 2 nhánh dùng chung các hàm** (dense, sparse, RRF, MMR), chỉ khác text
  và embedding đầu vào — không nhân đôi code.
- **Lý do có 2 nhánh**: HyDE viết theo văn phong pháp lý nên gần không gian
  vector với chunk luật thật hơn, nhưng LLM có thể paraphrase lệch ý người
  dùng. Nhánh B (câu hỏi gốc) là lưới an toàn, và giữ được viện dẫn cụ thể
  (số Điều/Khoản) mà người dùng gõ.
- **Union**: gộp kết quả của 2 nhánh (sau MMR nếu bật, sau RRF nếu tắt) theo
  `chunk_id`, mỗi chunk 1 lần dù có ở cả 2 nhánh. Không tính lại điểm fusion
  giữa 2 nhánh — reranker quyết định thứ tự cuối cùng. Tối đa
  `2 × BRANCH_TOP_N` passage.
- **MMR là bước tuỳ chọn** (bật/tắt bằng công tắc, mục 7) để đánh giá tác động
  bằng RAGAS.

**Số lượt gọi API mỗi câu hỏi:**

| Dịch vụ                                                   | Số lượt                     |
| ----------------------------------------------------------- | ------------------------------ |
| Groq (HyDE)                                                 | 1                              |
| HF Inference API (embed 2 text trong 1 batch)               | 1                              |
| Pinecone dense query                                        | 2 (1/nhánh)                   |
| Pinecone sparse query                                       | 2 (1/nhánh)                   |
| Pinecone dense fetch (bổ sung vector/metadata, mục 6.4) | MMR bật: ≤ 2 (1/nhánh); MMR tắt: ≤ 1 (cho cả union) |
| Reranker tự host                                           | 1                              |

HF Inference API dùng chung quota RPD 1.000/ngày với batch embedding
(`embedding_spec.md` mục 4); gộp 2 text thành 1 request giúp giảm một nửa
mức tiêu thụ so với 2 request riêng. Spec này không xử lý thêm về quota.

## 4. Pre-retrieval — HyDE

- Sinh 1 `hypothetical_document` — đoạn văn ngắn trả lời trực tiếp câu hỏi,
  theo văn phong luật — bằng Groq, **tái dùng `LLMSettings` hiện có trong
  `config.py`** (model `openai/gpt-oss-120b`): không thêm provider/model
  riêng cho query-time.
- Prompt thiết kế riêng cho domain pháp luật Việt Nam — nguyên văn chốt khi
  implement. **Luật bắt buộc trong prompt (chốt 2026-09-19)**: chỉ viết nội
  dung quy định, **không nêu số Điều/Khoản hay tên văn bản cụ thể**. Lý do:
  sparse của nhánh A search trên `breadcrumb` (chứa số Điều/Khoản); nếu LLM bịa
  viện dẫn sai sẽ kéo các chunk khớp số sai vào nhánh A.
- `hypothetical_document` chỉ dùng cho **nhánh A** (embed + sparse query).
  MMR và rerank dùng câu hỏi gốc (mục 7, 9).
- Kiểm tra output: `hypothetical_document` rỗng (vd. model reasoning dùng hết
  `max_tokens` cho reasoning) → coi như Groq lỗi (mục 10).
- Retry/timeout theo `max_retries`/`timeout_seconds` của `LLMSettings` (đã
  có sẵn).

## 5. Embed query (HF Inference API)

- **1 request duy nhất** embed `[hypothetical_document, query]` → 2 vector
  (`emb_hypo`, `emb_query`), dùng `huggingface_hub.InferenceClient`, tái dùng
  `EmbeddingSettings` (cùng model với `embedding/hf_client.py`).
- **Bắt buộc tiền xử lý giống lúc index**: word-segment bằng
  `pyvi.ViTokenizer.tokenize()` trước khi gửi (như `embedding/hf_client.py`
  dòng 85, chỉ embed `content`, không kèm breadcrumb). Nếu bỏ bước này, vector
  query và vector chunk lệch không gian, dense search kém mà không báo lỗi.
- `emb_query` vừa dùng cho dense search nhánh B, vừa là `sim(d, query)` trong
  MMR của cả 2 nhánh khi MMR bật (mục 7).
- Không tái dùng `embedding/hf_client.py` — module đó gắn với luồng batch
  theo `Chunk`/file (cùng lý do `embedding_spec.md` mục 3 không tái dùng
  `chunking/tokenizer.py`). Logic embed query nằm trong
  `retrieval/query_embedder.py`; nhất quán tiền xử lý qua việc cả hai cùng
  dùng `ViTokenizer`.

## 6. Hybrid search & RRF (trong từng nhánh)

### 6.1. Dense search

Query Pinecone dense index đã có (`VectorDBSettings.index_name`,
`embedding_spec.md` mục 5, `metric="cosine"`) — tái dùng toàn bộ index đã
build, `retrieval/` không tạo dense index nào. Mỗi nhánh lấy top
`DENSE_TOP_N` (=20) theo cosine, bằng embedding của nhánh đó
(`emb_hypo` / `emb_query`). Query với `include_metadata=True` (cho `RetrievedChunk`) và
`include_values=True` chỉ khi MMR bật (vector phục vụ MMR, mục 7).

### 6.2. Sparse/keyword search — Pinecone sparse index riêng

**Đã chọn phương án Pinecone sparse index riêng** (2026-09-19). Các phương án
đã cân nhắc và loại:

- *Hybrid gốc 1 index (dense+sparse, `dotproduct`)*: phải build lại dense
  index và sửa `embedding/` đã merge — loại.
- *BM25 trong bộ nhớ (không thêm index)*: corpus hiện nhỏ (2.228 chunk, ~1MB)
  nên tiết kiệm được 2 lượt gọi/câu hỏi, nhưng người dùng chọn Pinecone sparse
  index riêng — chốt theo lựa chọn này. Ghi nhận để xét lại nếu latency/API
  call sparse thành điểm nghẽn.

Cấu hình:

**BM25 encoder tự viết** (`retrieval/bm25.py`, thuần Python, chốt
2026-09-19) — **không dùng `pinecone-text`**, vì:

- `pinecone-text 0.11.0` phụ thuộc `mmh3 4.1.0`, không có wheel cho Python
  3.13/3.14 (repo yêu cầu `>=3.14`) nên `uv sync` phải build từ source, lỗi
  khi không có `gcc` (đã kiểm chứng trên môi trường dev).
- `BM25Encoder` mặc định `remove_stopwords=True, stem=True,
  language="english"`: làm mất từ tiếng Việt trùng stopword tiếng Anh (đã
  kiểm chứng: "**do**" trong "do bị tai nạn" bị loại), và tự tải dữ liệu NLTK
  qua mạng khi import.
- Pinecone sparse index tích hợp (`pinecone-sparse-english-v0`) tối ưu cho
  tiếng Anh; Pinecone Document Index/full-text search không hỗ trợ tiếng Việt
  và không cho tự tokenize — đều loại.

Thiết kế encoder:

- **Text đưa vào BM25**: `breadcrumb + " " + content` của từng chunk.
  Breadcrumb chứa viện dẫn (Điều/Khoản) mà người dùng có thể gõ trực tiếp,
  dense search một mình khó match tốt trường hợp này.
- **Tokenize**: word-segment bằng `pyvi` (cùng lý do `chunking/tokenizer.py`,
  `embedding/hf_client.py`: không segment sẽ tách nhầm ở mức âm tiết), lowercase,
  tách theo khoảng trắng và giữ dấu `_` của pyvi (`lao_động`), bỏ dấu câu
  đứng riêng. **Không stemming, không stopword.** Cùng 1 hàm tokenize dùng cho
  fit, encode document và encode query (bắt buộc nhất quán).
- **`fit(texts)`** — chạy 1 lần trên toàn corpus mỗi lần build, không
  incremental (IDF là thống kê toàn corpus): tính `df` của từng term, `N`,
  `avgdl`; gán mỗi term 1 id số nguyên (`vocab: term → id`, sắp xếp cố định để
  tái lập được).
- **`encode_document(text)`** → sparse vector `{indices, values}`, với mỗi
  term: `w = tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))`, `k1=1.2`, `b=0.75`
  (mặc định thông dụng). Không nhân IDF ở phía document.
- **`encode_query(text)`** → sparse vector, mỗi term có trong vocabulary:
  `w = idf = ln((N − df + 0.5)/(df + 0.5) + 1)`; term ngoài vocabulary bị bỏ.
  Dot product query·document trong Pinecone chính là điểm BM25.
- **Lưu tham số**: `data/bm25/bm25_params.json` gồm `vocab`, `idf` (hoặc `df`
  + `N`), `avgdl`, `k1`, `b`. Load 1 lần khi khởi
  tạo pipeline.
- Encoder có unit test riêng (công thức, term ngoài vocabulary, lệch tokenizer)
  vì tự chịu trách nhiệm đúng công thức.

Sparse index:

- **Index riêng**: field `sparse_index_name` trong `VectorDBSettings` (mục
  12), serverless, `vector_type="sparse"`, `metric="dotproduct"` (khác
  `cosine` của dense), **dùng chung `cloud`/`region` với dense index** (gói
  Starter chỉ cho AWS `us-east-1`). Cú pháp tạo sparse index của SDK
  (`pinecone` ≥ 8, đang cài 10.0.0) cần xác nhận khi implement.
- **Record**: `id = chunk_id`, chỉ `sparse_values`, **không lưu metadata**
  (metadata lấy từ dense index qua fetch, mục 6.4, áp dụng cả khi MMR bật lẫn tắt). Mỗi vector tối đa 1.000 giá trị khác 0 (chunk ngắn nên
  không vượt).
- **Query**: nhánh A dùng `hypothetical_document`, nhánh B dùng câu hỏi gốc →
  `encode_query` → `sparse_index.query(sparse_vector=..., top_k=SPARSE_TOP_N)`
  (=20).
- **Xoá index trước khi upsert lại**: `index.delete(delete_all=True)`. Index
  mới tạo chưa có namespace mặc định nên lệnh này ném `NotFoundException`
  ("namespace not found") — bắt và coi như đã sạch, y hệt
  `embedding/pinecone_client.py` (`_delete_all_vectors`, commit `9c4e13d`).
  `retrieval/` **chép đoạn nhỏ này thành helper riêng** (không import hàm
  private của `embedding/`, giữ 2 package độc lập).
- **Khoảng trống khi rebuild (chấp nhận)**: xoá hết → upsert hết khiến sparse
  index rỗng vài giây đến khoảng chục giây (~23 batch × 100 với 2.228 chunk).
  Rebuild là thao tác thủ công, offline; quy ước **không chạy `retrieve()` khi
  đang rebuild**. Nếu có, `retrieve()` degrade theo mục 10 (sparse rỗng).

### 6.3. RRF Fusion (2 danh sách/nhánh)

Mỗi nhánh hợp nhất dense top-N và sparse top-N theo `chunk_id`:

```
rrf_score(chunk) = Σ 1 / (k + rank_trong_danh_sách)   # cộng qua 2 danh sách của nhánh
```

`k = 60` (hằng số gốc từ paper RRF). Chunk chỉ có ở 1 danh sách vẫn được tính
(chỉ cộng số hạng của danh sách nó có mặt). Sort giảm dần, cắt `FUSION_TOP_N`
(=20) đưa sang MMR.

### 6.4. Bổ sung vector/metadata cho candidate chỉ có ở sparse

Sparse index không lưu vector lẫn metadata, nên candidate chỉ đến từ sparse
(không nằm trong dense top-N) thiếu metadata (cho `RetrievedChunk`) và — khi
MMR bật — cả vector (cho MMR). Bổ sung bằng `dense_index.fetch(ids=[...])`
(trả cả values và metadata, không cần embed lại); bỏ qua nếu không thiếu id
nào:

- **MMR bật**: mỗi nhánh fetch 1 lần cho id thiếu, **trước** MMR — tối đa 2
  lượt/câu hỏi.
- **MMR tắt**: không cần vector; fetch **1 lần cho toàn bộ union** (chỉ lấy
  metadata của id thiếu) — tối đa 1 lượt/câu hỏi.
- Chunk có ở sparse nhưng fetch không trả về (không có ở dense index do lệch
  build) bị bỏ, log warning.

## 7. MMR (Maximal Marginal Relevance) — bật/tắt được

MMR là bước **tuỳ chọn**, mặc định bật theo flow đã chốt. Mục đích của công
tắc: khi đánh giá bằng RAGAS, chỉ cần đổi công tắc để so sánh có/không MMR ảnh
hưởng thế nào (chốt 2026-09-19). Bản thân việc chạy đánh giá RAGAS thuộc spec
khác.

**Công tắc:**

- Hằng số `USE_MMR = True` trong `retrieval/` là giá trị mặc định.
- `retrieve(query, *, use_mmr: bool | None = None)`: `None` → dùng `USE_MMR`;
  truyền `True`/`False` để ghi đè từng lần gọi. Script đánh giá chạy cùng bộ
  câu hỏi 2 lần (`use_mmr=True` rồi `False`) mà không sửa code.
- Hai chế độ chỉ khác đúng bước chọn candidate của mỗi nhánh; RRF,
  `FUSION_TOP_N`, `BRANCH_TOP_N`, union và rerank giữ nguyên để so sánh công
  bằng:

| | MMR bật | MMR tắt |
| --- | --- | --- |
| Chọn candidate mỗi nhánh | `mmr.select(fused, emb_query, MMR_LAMBDA, BRANCH_TOP_N)` | `fused[:BRANCH_TOP_N]` |
| Cần vector từ Pinecone | Có (`include_values`, fetch mỗi nhánh) | Không |
| Số lượt fetch (mục 6.4) | ≤ 2 (1/nhánh, trước MMR) | ≤ 1 (cho cả union) |

`BRANCH_TOP_N = 10` (số candidate mỗi nhánh đưa vào union, cả khi MMR tắt).

**MMR làm gì**: chạy trên **dense vector** (cosine giữa các embedding), không
chạy trên sparse. Sau RRF mỗi nhánh có tối đa `FUSION_TOP_N` candidate; MMR
chọn tuần tự `BRANCH_TOP_N` chunk, mỗi lượt lấy chunk tối đa hoá:

```
MMR = argmax_{d ∈ candidates \ selected} [ λ·sim(d, query) − (1−λ)·max_{d' ∈ selected} sim(d, d') ]
```

- `sim(d, query)`: cosine giữa vector của candidate và `emb_query` (embedding
  **câu hỏi gốc**, cho cả 2 nhánh — phản ánh ý định thật, không lệch theo
  paraphrase của HyDE; đã có sẵn từ mục 5, không gọi embed thêm).
- `sim(d, d')`: cosine giữa vector 2 candidate.
- `MMR_LAMBDA = 0.5`.

**Giả thuyết cần kiểm chứng bằng RAGAS**: MMR loại bớt chunk gần trùng nhau
(có lợi), nhưng văn bản luật có nhiều Khoản liền nhau ngữ nghĩa rất gần mà câu
trả lời thường cần cả nhóm — MMR có thể phạt oan các chunk này trước khi
reranker chấm điểm (có hại). So sánh 2 chế độ trên các metric context của
RAGAS; nếu MMR bật kém hơn thì tắt, hoặc tăng `MMR_LAMBDA` (~0.7) rồi đo lại.

## 8. Union

`union = dedupe_by_chunk_id(nhánh_a + nhánh_b)` — tối đa `2 × BRANCH_TOP_N`
chunk. Giữ thứ hạng của mỗi chunk trong nhánh của nó (sau MMR nếu bật, sau RRF
nếu tắt) để phục vụ fallback (mục 9). Khi MMR tắt, sau union bổ sung metadata
cho id còn thiếu bằng 1 lượt fetch (mục 6.4).

## 9. Reranker (server tự host)

Model: `AITeamVN/Vietnamese_Reranker` (cross-encoder 0.6B tham số, base
`bge-reranker-v2-m3`, input tối đa 2304 token — 256 cho query, 2048 cho
passage). Không có sẵn trên HF Inference API nên gọi qua **server tự host
trên LightningAI** (LitServe), code tại `reranker_server/` (thư mục riêng ở
root repo). Đã test thành công end-to-end (2026-09-18).

**Client** (`retrieval/reranker_client.py`, `httpx`):

- `POST {endpoint_url}` với header `X-API-Key: {api_key}` (auth built-in của
  LitServe), payload `{"query": câu_hỏi_gốc, "passages": [breadcrumb + "\n" + content của từng chunk trong union]}`, response `{"scores": [float, ...]}` cùng thứ tự.
  Schema đã xác nhận bằng `retrieval/test.py`.
- **1 request duy nhất cho toàn bộ union.**
- **Passage = `breadcrumb + "\n" + content`** (chốt 2026-09-20). Query là
  **câu hỏi gốc**, không phải `hypothetical_document`.
  - Lý do: cross-encoder cần thấy số Điều/Khoản và tiêu đề Điều (vd.
    "Điều 25. Thời gian thử việc") để khớp câu hỏi viện dẫn hoặc theo chủ đề;
    `content` một mình thường không chứa các thông tin này.
  - Phạm vi: breadcrumb vẫn là **metadata-only ở tầng embedding dense/chunking**
    (`chunking_spec.md` mục 2, `embedding/hf_client.py` chỉ embed `content`);
    quyết định này **chỉ áp dụng cho input của reranker**.
    `RetrievedChunk.content` không đổi — không nhét breadcrumb vào `content`;
    ghép chuỗi passage chỉ xảy ra ngay trước khi gọi reranker.
  - Độ dài: breadcrumb ngắn so với giới hạn passage 2048 token của reranker nên
    không cần cắt.
  - Bằng chứng thí nghiệm (2026-09-20, cùng tập candidate, chỉ đổi passage, 7
    câu, mỗi câu 1 lần): với 4 câu có chunk đáp án trong union, hạng đáp án khi
    chỉ `content` → khi có breadcrumb: "Điều 25 Bộ luật Lao động quy định gì?"
    8 → 2; "Thời gian thử việc tối đa là bao lâu?" 6 → 1; "nghỉ phép năm 12
    tháng" 1 → 1; "báo trước khi đơn phương chấm dứt HĐLĐ" 1 → 1. Không câu
    nào tệ đi về hạng; điểm tuyệt đối giảm nhẹ ở một số câu (vd. 4.74 → 3.31)
    nhưng thứ hạng giữ nguyên. 3 câu viện dẫn còn lại: chunk đáp án không nằm
    trong union nên breadcrumb không cứu được (điểm mở 5, mục 16).
  - **Caveat**: mẫu nhỏ (7 câu) — cần xác nhận lại bằng RAGAS ở phase đánh giá.
- Sort giảm dần theo score, cắt `FINAL_TOP_K` (=5), gán `rerank_score`.

**Server tự host rất dễ lỗi** (Studio sleep/restart, tmux/ngrok chết, model
đang load, CPU chậm, quá tải) — xử lý như sau:

- **Tách connect timeout và read timeout**: `RerankerSettings` thêm
  `connect_timeout_seconds` (ngắn, khoảng 5s) bên cạnh `timeout_seconds` (read,
  khoảng 30s). Nếu chỉ có 1 timeout dài, Studio đang sleep có thể khiến 1
  câu hỏi chờ ~`timeout × (1 + max_retries)` (≈ 90s với cấu hình mặc định).
- **Phân loại lỗi:**
  - **Retry** (tối đa `max_retries`, backoff ngắn): lỗi kết nối, timeout,
    HTTP 502/503/504.
  - **Không retry, log error rõ ràng, sang fallback ngay**: HTTP 401/403 (sai
    `RERANKER_API_KEY`), 422/400 (payload sai) — lỗi cấu hình/code, retry vô
    ích.
  - Với lỗi kết nối/5xx sau khi hết retry, log warning nhắc **kiểm tra Studio
    theo runbook mục 9.1**.
- **Validate response**: `scores` phải là list số hữu hạn, đúng độ dài
  `passages`; sai → coi như lỗi.
- **Fallback khi hết retry** (chốt 2026-09-19): không crash, log warning, trả
  top `FINAL_TOP_K` với `rerank_score=None`. Thứ tự: xen kẽ 2 nhánh theo thứ
  hạng (A1, B1, A2, B2, …), bỏ chunk trùng. Cách này không cần vector nên dùng
  được ở cả 2 chế độ MMR. Nếu chỉ có 1 nhánh (Groq lỗi) thì lấy theo thứ hạng
  của nhánh đó.

### 9.1. Runbook: khởi động reranker server (làm trước khi gọi `retrieve()`)

`retrieve()` phụ thuộc server này đang chạy. Studio free tier **tự sleep sau
10 phút không hoạt động** (hiện trên web: "The Studio auto slept after 10
minutes of inactivity"), và khi sleep thì cả 2 tiến trình dưới đây chết. Mỗi
lần Studio sleep/restart phải làm lại runbook. Nếu bỏ qua, mọi câu hỏi rơi vào
fallback mục 9 (không có `rerank_score`).

**Phân công (chốt 2026-09-19): người dùng chỉ bật Studio trên web; agent tự
SSH và chạy toàn bộ phần còn lại.**

**Bước 1 — Người dùng bật Studio (thủ công trên web, agent không làm được):**

1. Mở [https://lightning.ai](https://lightning.ai), vào Studio
   `production-legal-qa-rag`.
2. Bấm **Turn on**, chờ trạng thái **Running**, rồi báo cho agent.

**Bước 2 — Agent SSH vào Studio (qua Bash):**

3. Chạy lệnh SSH trong biến `LIGHTNING_STUDIO_SSH` của `.env` gốc repo (không
   đổi khi Studio restart). Agent chỉ đọc đúng biến này, không in các secret
   khác trong `.env`.

**Bước 3 — Agent chạy 2 tiến trình nền bằng `tmux` (tmux giúp tiến trình sống
tiếp khi phiên SSH đóng; nó KHÔNG ngăn Studio sleep, xem ghi chú):**

4. Reranker server (nếu session đã có thì `tmux attach -t reranker` và kiểm
   tra, không tạo trùng):

   ```bash
   tmux new -d -s reranker 'cd reranker_server && uv run server.py'
   ```

   LitServe lắng nghe cổng 8000, auth qua `LIT_SERVER_API_KEY` (trong
   `reranker_server/.env`, phải trùng `RERANKER_API_KEY` ở `.env` gốc repo).
5. Tunnel ngrok:

   ```bash
   tmux new -d -s ngrok_tunnel 'ngrok http 8000 --log stdout'
   ```

   Domain tĩnh free của tài khoản ngrok (không đổi mỗi lần restart) chính là
   `RERANKER_ENDPOINT_URL`.

**Bước 4 — Agent kiểm tra từ máy local:**

6. Chạy `retrieval/test.py` (gọi server thật với 3 chunk mẫu) — có `scores`
   trả về là server sẵn sàng. Model cần thời gian load sau khi start nên có
   thể phải thử lại vài lần trước khi thành công.

**Ghi chú:**

- **tmux không chống sleep.** tmux chỉ giữ tiến trình sống khi phiên SSH đóng.
  Việc một server đang chạy trong tmux có được Lightning tính là "hoạt động"
  hay không **chưa được xác minh** (tài liệu Lightning không nói rõ): cần thử
  thực tế — bật server trong tmux, đóng SSH, đợi >10 phút, xem Studio còn chạy
  không. Nếu vẫn sleep, chấp nhận chạy lại runbook mỗi lần dùng.
- **Ứng viên để bớt thao tác (chưa xác minh)**: tính năng *on-start actions*
  của Lightning (file `on_start.sh` chạy khi Studio bật). Nếu chạy được tmux ở
  đó thì người dùng chỉ cần bấm **Turn on**, bước 2–5 tự chạy. Cần thử trước
  khi đưa vào spec chính thức.
- `reranker_server/` là dự án `uv` độc lập (`pyproject.toml`/`uv.lock` riêng,
  có `torch`/`transformers`/`litserve`), KHÔNG phải dependency của package
  `retrieval/` hay root `pyproject.toml`. Chỉ chạy trên Lightning Studio.
- Lý do dùng ngrok: link "Forwarded Address" trong tab Ports của Studio chỉ
  là proxy giao diện web có xác thực; còn "Lightning Deploy" (Autoscale) bị
  lỗi hạ tầng (replica crash-loop ở bước Setup) khi thử nghiệm 2026-09-18.

## 10. Xử lý lỗi (chỉ degrade 2 điểm)

Chỉ degrade ở 2 điểm hay lỗi nhất và có phương án thay thế rẻ; các lỗi còn lại
raise rõ ràng, không cố xử lý từng trường hợp (giữ đơn giản).

| Lỗi (sau hết retry) | Xử lý |
| --- | --- |
| Groq lỗi/timeout hoặc trả rỗng | Bỏ nhánh A; chỉ embed câu hỏi gốc (batch 1) và chạy nhánh B; log warning |
| Reranker lỗi | Fallback mục 9 |
| HF embed hoặc Pinecone (dense/sparse/fetch) lỗi | Raise `RetrievalError` kèm nguyên nhân; bước generation quyết định xử lý |

Số lần retry theo settings sẵn có của từng dịch vụ (`LLMSettings`,
`EmbeddingSettings`, `RerankerSettings`).

## 11. Tools & Integrations

| Việc                                   | Công cụ                                                                                       |
| --------------------------------------- | ----------------------------------------------------------------------------------------------- |
| LLM sinh HyDE                           | `groq` SDK, tái dùng `LLMSettings`                                                        |
| Embed`[hypo, query]` (1 batch)        | `huggingface_hub.InferenceClient`, tái dùng `EmbeddingSettings`                           |
| Word-segment tiếng Việt               | `pyvi` (đã có) — cho cả embed query và BM25                                             |
| Sparse/keyword encode (BM25)            | Tự viết (`retrieval/bm25.py`, thuần Python) — **không dùng `pinecone-text`** (không cài được trên Python ≥3.14, mục 6.2); không thêm dependency mới |
| Dense + sparse vector search, fetch     | `pinecone` SDK (đã có)                                                                     |
| Chạy 2 nhánh / dense+sparse song song | `asyncio` (stdlib); dùng client async của từng SDK khi có                                 |
| RRF, MMR                                | Tự cài thuật toán thuần Python, không cần thư viện ngoài                              |
| Rerank                                  | HTTP client`httpx` gọi server LightningAI                                                    |
| Data models                             | `pydantic` v2                                                                                 |

`retrieve()` là `async def` (chốt 2026-09-19): pipeline nội bộ chạy song song,
bước generation sau cũng nên async.

## 12. Config tập trung (`src/production_legal_qa_rag/config.py`)

- `VectorDBSettings` thêm field:
  `sparse_index_name: str = Field(validation_alias="PINECONE_SPARSE_INDEX_NAME")`.
- `RerankerSettings` (class mới, cùng pattern `LLMSettings`):

  ```python
  class RerankerSettings(BaseSettings):
      endpoint_url: str = Field(validation_alias="RERANKER_ENDPOINT_URL")
      api_key: str = Field(validation_alias="RERANKER_API_KEY")
      max_retries: int = 2
      connect_timeout_seconds: int = 5
      timeout_seconds: int = 30
  ```

  `endpoint_url` và `api_key` bắt buộc, đọc từ `.env`; `api_key` gửi qua header
  `X-API-Key`.
- `LIGHTNING_STUDIO_SSH` trong `.env` — **không** thuộc Settings nào (chỉ là
  ghi chú vận hành cho runbook mục 9.1).
- **Không** đưa `DENSE_TOP_N`/`SPARSE_TOP_N`/`FUSION_TOP_N`/`RRF_K`/
  `MMR_LAMBDA`/`BRANCH_TOP_N`/`USE_MMR`/`FINAL_TOP_K` vào `config.py` — hằng số nội bộ của
  `retrieval/`, nhất quán `embedding_spec.md` mục 7 (chỉ field dùng chung nhiều
  package mới vào `config.py`).

## 13. Workflow & quản lý trạng thái

Chia 2 luồng độc lập:

### 13A. Setup/offline — build sparse index

Chạy 1 lần hoặc mỗi khi corpus (`data/chunks/*.json`) thay đổi, cùng tinh thần
batch job của `embedding/`:

```
tools/sparse_index_documents.py (Typer CLI)
  → retrieval.sparse_index.build_index(chunks_dir, params_out_path)
      1. Đọc data/chunks/*.json -> list[Chunk]
      2. Với mỗi chunk: text = breadcrumb + " " + content (corpus = list 2.228 string)
      3. bm25.fit(all_texts): tokenize (pyvi) -> vocab + df/idf + avgdl chung cho cả corpus
         -> lưu data/bm25/bm25_params.json
      4. bm25.encode_document(text) cho từng chunk riêng lẻ -> sparse vector {indices, values}
      5. Tạo Pinecone sparse index nếu chưa có (vector_type="sparse", metric="dotproduct")
      6. Xoá sạch index cũ (bỏ qua NotFoundException namespace), upsert toàn bộ theo batch
         (id=chunk_id, chỉ sparse_values, không metadata)
      7. In summary: tổng số chunk đã index
```

**Điểm chạy riêng để build sparse index** (chốt 2026-09-19): một script CLI
độc lập, không liên quan `retrieve()` và không cần chạy `embedding/` hay các
bước khác:

```bash
uv run python tools/sparse_index_documents.py
# tuỳ chọn: --chunks-dir data/chunks --params-out data/bm25/bm25_params.json
```

- Cùng khuôn với `tools/chunk_documents.py`, `tools/embed_documents.py`: Typer
  CLI mỏng (`add_completion=False`), docstring đầu file ghi cách dùng, mặc định
  `--chunks-dir=data/chunks`, `--params-out=data/bm25/bm25_params.json`,
  `raise typer.Exit(code=...)` (0 khi thành công, 1 khi lỗi).
- Chỉ cần `.env` có `PINECONE_API_KEY`, `PINECONE_SPARSE_INDEX_NAME` (cùng
  `cloud`/`region` của `VectorDBSettings`); không cần Groq/HF/reranker.
- Mỗi lần chạy build lại toàn bộ (fit + xoá + upsert), an toàn để chạy lại.
- Tên file dùng snake_case theo quy ước các script trong `tools/`.

Quy ước vận hành: sau mỗi lần `embedding/` build lại dense index, chạy lại
`tools/sparse_index_documents.py` để 2 index cùng corpus; không có bước tự kiểm tra.

### 13B. Online — `retrieve(query)`

Library function, không CLI:

```
retrieval.pipeline.retrieve(query: str, *, use_mmr: bool | None = None) -> list[RetrievedChunk]

  use_mmr = USE_MMR if use_mmr is None else use_mmr                  # mục 7
  1. hypo = await hyde.generate(query)                                # mục 4 (lỗi → chỉ nhánh B, mục 10)
  2. emb_hypo, emb_query = await embedder.embed([hypo, query])        # mục 5, 1 request HF
  3. branch_a, branch_b = await gather(
       run_branch(text=hypo,  dense_emb=emb_hypo),                    # nhánh A
       run_branch(text=query, dense_emb=emb_query))                   # nhánh B

     run_branch(text, dense_emb):
       a. dense, sparse = await gather(
            dense_search.query(dense_emb, DENSE_TOP_N, include_values=use_mmr),  # mục 6.1
            sparse_index.query(text, SPARSE_TOP_N))                   # mục 6.2
       b. fused = fusion.rrf(dense, sparse, k=RRF_K)[:FUSION_TOP_N]   # mục 6.3
       c. if use_mmr:
            fused = await fetch_missing(fused)                        # mục 6.4: vector+metadata
            return mmr.select(fused, emb_query, MMR_LAMBDA, BRANCH_TOP_N)   # mục 7
          return fused[:BRANCH_TOP_N]

  4. union = dedupe_by_chunk_id(branch_a + branch_b)                  # mục 8
     if not use_mmr: union = await fetch_missing_metadata(union)      # mục 6.4: 1 lượt cho cả union
  5. scores = await reranker_client.rerank(query, [c.breadcrumb + "\n" + c.content for c in union])  # mục 9 (lỗi → fallback)
  6. return top FINAL_TOP_K dạng list[RetrievedChunk]
```

Stateless giữa các lần gọi — không cache, không session. Trạng thái persist duy
nhất là `bm25_params.json`, đọc 1 lần lúc khởi tạo pipeline cùng các client
(khởi tạo 1 lần, dùng lại cho mọi query, không tạo mới mỗi query).

## 14. Cấu trúc module trong `src/production_legal_qa_rag/retrieval/`

| Module                 | Trách nhiệm                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| `models.py`          | `RetrievedChunk`, model trung gian cho candidate sau fusion/MMR                               |
| `hyde.py`            | Gọi Groq sinh hypothetical document (mục 4)                                                   |
| `query_embedder.py`  | Embed nhiều chuỗi trong 1 request HF, có`pyvi` segment (mục 5)                            |
| `bm25.py`            | BM25 encoder tự viết: tokenize, `fit`, `encode_document`, `encode_query`, lưu/đọc `bm25_params.json` (mục 6.2) |
| `sparse_index.py` | Tạo/xoá/upsert Pinecone sparse index (kèm helper xoá riêng), query runtime (mục 6.2, 13A) |
| `dense_search.py` | Query dense index có sẵn + fetch vector/metadata bổ sung (mục 6.1, 6.4) |
| `fusion.py`          | Thuật toán RRF (mục 6.3)                                                                     |
| `mmr.py` | Thuật toán MMR, bật/tắt bằng công tắc `USE_MMR` / `use_mmr` (mục 7) |
| `reranker_client.py` | HTTP client gọi server LightningAI, retry/validate/fallback (mục 9)                           |
| `pipeline.py`        | `retrieve(query)` — điều phối toàn bộ (mục 13B); sở hữu các client                  |
| `__init__.py`        |                                                                                                 |

`tools/sparse_index_documents.py` — Typer CLI mỏng gọi `sparse_index.build_index()`
(mục 13A), tách biệt hoàn toàn khỏi `pipeline.retrieve()`.

`retrieval/test.py` — script test thủ công gọi thẳng reranker server thật (3
chunk mẫu từ `data/chunks/`), dùng để xác nhận end-to-end (mục 9.1) — không
thuộc kiến trúc chính thức, xoá/thay thế khi có test suite thật (`tester`
agent) cho `reranker_client.py`.

## 15. Tiêu chí hoàn thành

- `tools/sparse_index_documents.py` chạy trên toàn bộ `data/chunks/*.json` hiện có
  không crash, sinh `data/bm25/bm25_params.json` và Pinecone sparse index
  chứa đúng số vector = tổng số chunk; chạy lại lần 2 (index đã có dữ liệu)
  và lần đầu trên index mới (chưa có namespace) đều không lỗi.
- `bm25.py` có unit test: điểm dot product query·document khớp công thức BM25
  tính tay trên corpus nhỏ; term ngoài vocabulary bị bỏ; từ như "do" không bị
  loại; câu hỏi chứa "Điều 3 khoản 1" xếp chunk có breadcrumb tương ứng lên
  đầu trên corpus mẫu.
- `retrieve(q, use_mmr=True)` và `retrieve(q, use_mmr=False)` đều trả kết quả hợp lệ (≤ 5 chunk, đủ metadata); số lượt fetch Pinecone đúng mục 6.4 cho từng chế độ.
- `retrieve(query)` chạy end-to-end (Groq → HF → 2 nhánh → union → rerank)
  trả về tối đa 5 `RetrievedChunk`, không crash với ít nhất 3 câu hỏi thật
  khác nhau — kể cả câu hỏi chứa viện dẫn cụ thể (vd. "Điều 3 khoản 1 quy
  định gì") để xác nhận sparse/keyword search hoạt động (candidate tương ứng
  xuất hiện trong top sparse dù dense có thể xếp thấp).
- Số lượt gọi API cho 1 câu hỏi khớp bảng mục 3 (Groq 1, HF 1, rerank 1).
- Vector query được embed qua cùng tiền xử lý `pyvi` như lúc index (test:
  câu hỏi trùng nguyên văn `content` 1 chunk phải trả về chunk đó ở top dense).
- Giả lập Groq lỗi và reranker lỗi (timeout, 401) → hành vi đúng mục 10 và mục 9, `retrieve()` không crash.
- Test: mỗi passage gửi tới reranker bắt đầu bằng `breadcrumb`, rồi xuống dòng
  (`"\n"`), rồi `content` (mục 9); `RetrievedChunk.content` trả ra vẫn là
  `content` nguyên gốc không kèm breadcrumb. Retry/lỗi/fallback không đổi so
  với mục 9.
- `config.py` có đủ `sparse_index_name`/`RerankerSettings` mới; các module
  trong `retrieval/` không đọc `.env` trực tiếp.

## 16. Điểm còn mở

Đã chốt (2026-09-19): luật "không nêu số Điều/Khoản" trong prompt HyDE;
`RetrievedChunk` thu gọn theo metadata Pinecone; fallback reranker xen kẽ 2
nhánh; `retrieve()` là `async def`; MMR có công tắc bật/tắt; runbook do agent
SSH + tmux, người dùng chỉ bật Studio.

Đã chốt (2026-09-20): passage gửi reranker = `breadcrumb + "\n" + content`
(mục 9; breadcrumb vẫn metadata-only ở embedding/chunking); cần xác nhận lại
bằng RAGAS vì thí nghiệm chỉ 7 câu.

1. Nguyên văn prompt HyDE — chốt khi implement (mục 4).
2. Xác minh tmux/`on_start.sh` có giúp Studio khỏi sleep không (mục 9.1).
3. MMR bật hay tắt làm mặc định production — quyết định sau khi đánh giá bằng
   RAGAS (mục 7). Các hằng số khởi điểm (`DENSE_TOP_N=20`, `SPARSE_TOP_N=20`,
   `RRF_K=60`, `FUSION_TOP_N=20`, `MMR_LAMBDA=0.5`, `BRANCH_TOP_N=10`,
   `FINAL_TOP_K=5`) tinh chỉnh cùng lúc đó.
4. **Bàn sau** (ngoài flow hiện tại): tối ưu latency/API call — benchmark giới
   hạn batch/concurrency của HF, Groq, Lightning; dynamic batching; circuit
   breaker cho reranker; khởi động sớm nhánh B trong lúc chờ Groq; theo dõi quota
   HF dùng chung.
5. **Recall cho câu hỏi viện dẫn Điều/Khoản** (chẩn đoán 2026-09-20): với 3
   câu viện dẫn (Điều 3 khoản 1 / Điều 36 khoản 2 / Khoản 1 Điều 113 Bộ luật
   Lao động), chunk đáp án nằm ở sparse top 5/10/16 (dense hạng 45 hoặc ngoài
   top 100), nhưng RRF (trọng số bằng nhau) đẩy nó xuống hạng 10/20/31 của
   danh sách fused, rồi bị cắt bởi `FUSION_TOP_N=20` và `BRANCH_TOP_N=10` (và
   MMR) nên không vào union — reranker (kể cả có breadcrumb) không cứu được.
   Hướng có thể xét sau (chưa quyết, chưa làm): tăng `BRANCH_TOP_N`/
   `FUSION_TOP_N` (đổi lại latency rerank vì tỉ lệ thuận số passage); đảm bảo
   top-k sparse luôn vào union; tăng trọng số sparse trong RRF; tắt MMR; hoặc
   nhận diện viện dẫn bằng regex.
