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
  xếp giảm dần theo độ liên quan (thứ tự sau rerank, không can thiệp thêm).

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
  `2 × BRANCH_TOP_N` passage; riêng câu hỏi viện dẫn (mục 8.1) thêm tối đa
  extras (mục 8.1: 10 với 1 Điều; tối đa 24 với 2-3 Điều) → tối đa
  `2 × BRANCH_TOP_N + 24` = 44.
- **MMR là bước tuỳ chọn** (bật/tắt bằng công tắc, mục 7) để đánh giá tác động
  bằng RAGAS.

**Số lượt gọi API mỗi câu hỏi:**

| Dịch vụ                                                   | Số lượt                     |
| ----------------------------------------------------------- | ------------------------------ |
| Groq (HyDE)                                                 | 1                              |
| HF Inference API (embed 2 text trong 1 batch)               | 1                              |
| Pinecone dense query                                        | 2 (1/nhánh)                   |
| Pinecone sparse query                                       | 2 (1/nhánh); câu viện dẫn ≥ 2 Điều thêm n lượt (mỗi Điều 1 lượt, n ≤ 3, chạy song song, mục 8.1) → ≤ 5 |
| Pinecone dense fetch (bổ sung vector/metadata, mục 6.4) | MMR bật: ≤ 2 (1/nhánh), câu viện dẫn thêm ≤ 1 (metadata extras, mục 8.1) → ≤ 3; MMR tắt: ≤ 1 (cho cả union, kể cả extras) |
| Reranker tự host                                           | 1                              |

HF Inference API dùng chung quota RPD 1.000/ngày với batch embedding
(`embedding_spec.md` mục 4); gộp 2 text thành 1 request giúp giảm một nửa
mức tiêu thụ so với 2 request riêng. Spec này không xử lý thêm về quota.

## 4. Pre-retrieval — HyDE

- Sinh 1 `hypothetical_document` — đoạn văn ngắn trả lời trực tiếp câu hỏi,
  theo văn phong luật — bằng Groq, **tái dùng `LLMSettings` hiện có trong
  `config.py`** (model `openai/gpt-oss-120b`): không thêm provider/model
  riêng cho query-time.
- Prompt thiết kế riêng cho domain pháp luật Việt Nam — **nguyên văn chốt
  2026-09-20 (theo lý thuyết, chưa đo bằng RAGAS; xem mục 16 điểm 1)** bên dưới.
  **Luật bắt buộc trong prompt (chốt 2026-09-19)**: chỉ viết nội
  dung quy định, **không nêu số Điều/Khoản hay tên văn bản cụ thể**. Lý do:
  sparse của nhánh A search trên `breadcrumb` (chứa số Điều/Khoản); nếu LLM bịa
  viện dẫn sai sẽ kéo các chunk khớp số sai vào nhánh A.

**Prompt (tách system/user; hằng số trong `retrieval/hyde.py`):**

<!-- tests/test_retrieval_clients.py đọc khối prompt bên dưới để so khớp với `retrieval/hyde.py`; đừng đổi cấu trúc khối (fence, nhãn [system]/[user]) mà không sửa test. -->

Tách system/user vì phần quy tắc cố định nằm ở system (model tuân thủ ổn định
hơn, và câu hỏi người dùng nằm riêng ở user nên khó "ghi đè" quy tắc bằng nội
dung câu hỏi).

```
[system]
Bạn là chuyên gia pháp luật Việt Nam. Nhiệm vụ: với mỗi câu hỏi của người dùng,
viết một đoạn văn giả định như thể trích từ văn bản quy phạm pháp luật Việt Nam
đang trả lời câu hỏi đó. Đoạn văn này chỉ dùng để tìm kiếm điều luật tương tự,
không phải câu trả lời cho người dùng.

Phạm vi pháp luật thường gặp: lao động và quan hệ lao động, bảo hiểm xã hội,
bảo hiểm y tế, thuế thu nhập cá nhân, tiền lương và mức lương tối thiểu.

Quy tắc:
1. Viết bằng văn phong văn bản quy phạm pháp luật (câu khẳng định, mang tính quy
   định: "được", "có quyền", "có trách nhiệm", "phải", "không được"...), dùng
   thuật ngữ pháp lý đúng lĩnh vực của câu hỏi (ví dụ: người lao động, người sử
   dụng lao động, người tham gia bảo hiểm, đóng và hưởng bảo hiểm, người nộp
   thuế, thu nhập chịu thuế, mức lương tối thiểu). Chỉ dùng thuật ngữ hợp với
   chủ đề câu hỏi, không nhồi thuật ngữ của lĩnh vực khác.
2. Độ dài: 3 đến 4 câu, khoảng 60-100 từ.
3. TUYỆT ĐỐI không nêu số Điều, Khoản, Điểm, Chương, Mục, tên hay số hiệu văn
   bản, năm ban hành — kể cả khi câu hỏi có nhắc tới.
4. Không nêu con số, mức tiền, tỉ lệ, thời hạn cụ thể, trừ khi bạn chắc chắn
   đúng; nếu không chắc, diễn đạt bằng từ chung ("mức tối thiểu theo quy định",
   "trong thời hạn theo quy định").
5. Câu hỏi mơ hồ, quá ngắn hoặc ngoài các lĩnh vực trên: vẫn viết một đoạn theo
   văn phong quy định về chủ đề pháp lý gần nhất mà câu hỏi gợi ra. Không từ
   chối, không hỏi lại.
6. Câu hỏi chỉ hỏi theo số Điều/Khoản mà không nêu chủ đề (ví dụ "Điều 36 khoản
   2 quy định gì?"): không đoán hay bịa chủ đề; viết một đoạn ngắn, chung chung
   về việc quy định các quyền, nghĩa vụ và trách nhiệm của các bên liên quan.
7. Đầu ra: chỉ một đoạn văn thuần tiếng Việt. Không markdown, không gạch đầu
   dòng, không tiêu đề, không lời mở đầu hay kết luận, không giải thích thêm.

[user]
Câu hỏi: {query}
```

Lý do từng quy tắc:

- **Trung lập lĩnh vực (1)**: corpus có 6 văn bản thuộc 5 lĩnh vực; prompt cũ
  chỉ gợi thuật ngữ lao động sẽ kéo vector/BM25 của nhánh A nghiêng về Bộ luật
  Lao động ngay cả với câu hỏi về thuế hay BHYT. Prompt mới liệt kê thuật ngữ
  đa lĩnh vực và dặn chỉ dùng thuật ngữ hợp chủ đề.
- **Độ dài 3-4 câu, ~60-100 từ (2)**: chunk tối đa 236 token (~1-2 Khoản). Hypo
  quá dài làm loãng vector dense (nhiều ý) và kéo thêm từ khoá không liên quan
  vào BM25; quá ngắn thiếu từ khoá để sparse/dense bám. 60-100 từ tiếng Việt
  (~100-200 token) nằm trong cỡ một chunk, chứa được 1-2 ý chính.
- **Cấm viện dẫn/tên văn bản/số hiệu/năm (3)**: giữ luật 2026-09-19 (sparse
  nhánh A search trên `breadcrumb + content`, chứa số Điều/Khoản và tên văn
  bản). Mở rộng thêm Chương/Mục/số hiệu/năm vì các token này cũng có trong
  breadcrumb và sẽ khớp nhầm.
- **Con số/mức/thời hạn (4)**: mặc định không nêu, chỉ nêu khi chắc chắn. Con
  số bịa sai (vd. sai tỉ lệ đóng bảo hiểm) tạo token nhiễu trong sparse và
  không giúp dense; HyDE chỉ cần đúng chủ đề và thuật ngữ, còn giá trị thật do
  chunk cung cấp. Không cấm tuyệt đối vì số chắc chắn (vd. "12 tháng") đôi khi
  là từ khoá khớp tốt.
- **Câu mơ hồ/ngoài phạm vi (5)**: từ chối hoặc hỏi lại làm hypo rỗng/vô nghĩa
  và nhánh A bị bỏ; sinh đoạn về chủ đề gần nhất giữ nhánh A hoạt động, còn
  nhánh B (câu gốc) và reranker là lưới an toàn nếu chủ đề đoán lệch.
- **Câu chỉ hỏi theo số Điều (6)**: phần viện dẫn đã do đường extras (mục 8.1)
  xử lý qua sparse nhánh B; HyDE không có căn cứ để biết chủ đề nên bịa chủ đề
  sẽ kéo nhánh A sang chunk sai. Đoạn chung chung là lựa chọn ít gây hại nhất
  (đóng góp trung tính cho nhánh A thay vì gây nhiễu).
- **Đầu ra thuần đoạn văn (7)**: markdown/gạch đầu dòng/tiêu đề/lời dẫn thêm
  token không thuộc văn phong luật vào embedding và BM25.

**Tham số gọi Groq (chốt 2026-09-20, hằng số nội bộ `retrieval/hyde.py`):**

| Tham số | Giá trị | Lý do |
| --- | --- | --- |
| `reasoning_effort` | `"low"` | Tác vụ viết một đoạn ngắn không cần suy luận sâu; giảm latency (Groq là bước đầu, nằm trên đường găng) và giảm rủi ro model dùng hết token cho reasoning dẫn tới output rỗng |
| `temperature` | `0.2` | Thấp để hypo ổn định giữa các lần gọi, kết quả retrieval tái lập được; không đặt 0 vì gpt-oss khuyến nghị tránh greedy hoàn toàn (dễ lặp) |
| `max_completion_tokens` | `2048` (giữ nguyên) | Output mục tiêu ~100-200 token; phần còn lại là dư cho reasoning. Chỉ trả tiền/latency theo token thực dùng nên giữ mức dư lớn là rẻ, còn giảm thấp làm tăng nguy cơ output rỗng |

- Developer **phải xác nhận** SDK `groq` đang cài và API Groq chấp nhận từng
  tham số trên cho `openai/gpt-oss-120b`. Tham số nào không được hỗ trợ thì
  **bỏ tham số đó** thay vì làm hỏng lời gọi, và ghi lại (comment trong
  `hyde.py` + cập nhật bảng này).
- **Đã xác nhận (2026-09-20)**: SDK `groq` 1.7.0 có đủ 3 tham số và API Groq chấp nhận
  cả `reasoning_effort="low"`, `temperature=0.2`, `max_completion_tokens=2048` cho
  `openai/gpt-oss-120b` (3 lời gọi thật đều trả output không rỗng, ~1s/lời gọi).
- Các tham số là hằng số nội bộ `retrieval/hyde.py`, **không** vào
  `LLMSettings`: chỉ HyDE dùng và chúng gắn với prompt này; `LLMSettings` dùng
  chung cho mọi tác vụ LLM (nhất quán mục 12). Nếu sau này có tác vụ LLM thứ
  hai cần tham số riêng thì mới xét lại.
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
- **`encode_query(text, extra_terms=())`** → sparse vector, mỗi term có trong
  vocabulary: `w = idf = ln((N − df + 0.5)/(df + 0.5) + 1)`; term ngoài
  vocabulary bị bỏ. Dot product query·document trong Pinecone chính là điểm
  BM25. `extra_terms` là các token cấu trúc (bên dưới), được nối vào danh sách
  token của query trước khi tính trọng số.
- **Token cấu trúc `điều_N`, `khoản_M`, `điều_N_khoản_M`** (chốt 2026-09-20,
  theo dữ liệu chẩn đoán). Vấn đề: BM25 tách "Điều 3" thành token rời "điều",
  "3"; số 1/3 có ở hàng trăm chunk nên idf ~0, và tên văn bản có trong
  breadcrumb mọi chunk, nên BM25 không phân biệt được Điều nào (ví dụ chunk
  đáp án "Điều 3 khoản 1 luật thuế TNCN" chỉ ở BM25 hạng 17). Giải pháp: thêm
  token cấu trúc — có idf cao vì hiếm — vào danh sách token của chunk và của
  query:
  - **Phía document** (khi `fit` và `encode_document`, cùng một hàm dùng cho
    cả hai): parse breadcrumb `... - Điều N. Tên điều - Khoản M [- Điểm ...]` để
    lấy Điều N và Khoản M (chunk split "(phần i/n)" hay có "Điểm" vẫn lấy Điều
    và Khoản như thường), sinh `điều_N`, `khoản_M`, `điều_N_khoản_M` thêm vào
    danh sách token của chunk (mỗi token 1 lần). Breadcrumb không có Điều (vd.
    frontmatter/Phụ lục) hoặc không có Khoản thì chỉ sinh token tương ứng có
    trong breadcrumb. Hàm parse breadcrumb (`citation.py`) dùng chung với
    tiện ích khác cần Điều/Khoản của chunk để không lệch định nghĩa.
  - **Phía query**: chỉ từ **câu hỏi gốc** (không phải hypo), khi
    `extract_citation_numbers` ≠ []: `điều_N` cho mỗi Điều nhận diện được (mục
    8.1), `khoản_M` cho mỗi Khoản (`khoản\s*(\d{1,3})`, cùng NFC/không phân biệt
    hoa thường, dedupe giữ thứ tự, tối đa `MAX_CITATION_KHOANS = 3`), và
    `điều_N_khoản_M` cho mỗi cặp (Điều, Khoản). Trần: ≤ 3 Điều × ≤ 3 Khoản →
    tối đa 3 + 3 + 9 = 15 token. "Khoản M" đứng một mình (không có số Điều)
    không sinh token. Áp cho **mọi** lượt `encode_query` có từ câu hỏi gốc: sparse
    nhánh B, và các lượt sparse phụ cho từng Điều (mục 8.1) — sub-query mỗi Điều
    chỉ sinh token của **Điều đó** (`điều_N`, `điều_N_khoản_M` cho mọi Khoản
    của câu, `khoản_M` cho mọi Khoản của câu; chưa gán được Khoản nào thuộc
    Điều nào, cùng giới hạn đã chấp nhận ở mục 8.1). **Nhánh A**: hypo không có
    số Điều (luật HyDE cấm) nên không sinh token cấu trúc; nhánh A gọi
    `encode_query` không có `extra_terms`. Token không có trong vocab bị bỏ như
    hiện có.
  - **Token theo văn bản** (chốt 2026-09-20, `params_version = 3`). Vấn đề
    (chẩn đoán "Điều 36 và Điều 113 Bộ luật Lao động"): token `điều_N` khớp mọi
    văn bản (Nghị định, BHXH, BHYT, TNCN đều có Điều 36 và Điều 113), còn tên
    văn bản trong breadcrumb chỉ là các từ rất phổ biến ("bộ_luật",
    "lao_động", idf thấp) nên không lọc được; thêm chuẩn hoá độ dài BM25 bất lợi
    cho chunk dài (Điều 113 Khoản 1). Kết quả: BLLĐ Điều 36 có 4 chunk nhưng chỉ
    1 vào union, Điều 113 có 7 chunk thiếu Khoản 1 (chunk nội dung chính).
    Giải pháp: gắn token định danh văn bản vào chunk và query.
    - **Bảng `DOCUMENTS`** (hằng số trong `citation.py`): `source_document` thật
      trong `data/chunks` → key ASCII ngắn (làm token) + danh sách alias.
      `source_document` thật hiện có (6 giá trị, đếm chunk: 2.228 tổng):

      | `source_document` (nguyên văn) | key | Alias (khớp sau NFC, lowercase, bỏ dấu) |
      | --- | --- | --- |
      | `BỘ LUẬT LAO ĐỘNG` (704) | `blld` | "bộ luật lao động", "luật lao động", "blld" |
      | `LUẬT BẢO HIỂM XÃ HỘI` (579) | `bhxh` | "luật bảo hiểm xã hội", "luật bhxh" |
      | `LUẬT BẢO HIỂM Y TẾ` (328) | `bhyt` | "luật bảo hiểm y tế", "luật bhyt" |
      | `LUẬT THUẾ THU NHẬP CÁ NHÂN` (121) | `tncn` | "luật thuế thu nhập cá nhân", "luật thuế tncn", "thuế thu nhập cá nhân", "thuế tncn" |
      | `NGHỊ ĐỊNH QUY ĐỊNH MỨC LƯƠNG TỐI THIỂU ĐỐI VỚI NGƯỜI LAO ĐỘNG LÀM VIỆC THEO HỢP ĐỒNG LAO ĐỘNG` (60) | `nd_luong` | "nghị định lương tối thiểu", "nghị định mức lương tối thiểu", "nghị định quy định mức lương tối thiểu" |
      | `NGHỊ ĐỊNH QUY ĐỊNH CHI TIẾT VÀ HƯỚNG DẪN THI HÀNH MỘT SỐ ĐIỀU CỦA BỘ LUẬT LAO ĐỘNG VỀ ĐIỀU KIỆN LAO ĐỘNG VÀ QUAN HỆ LAO ĐỘNG` (436) | `nd_dklđ` (ASCII: `nd_dkld`) | "nghị định điều kiện lao động", "nghị định quan hệ lao động", "nghị định hướng dẫn bộ luật lao động", "nghị định về điều kiện lao động và quan hệ lao động" |

      Đếm chunk theo file trong `data/chunks` (mỗi file 1 giá trị
      `source_document` duy nhất, đã kiểm tra riêng với BLLĐ 704/704);
      developer xác nhận lại lúc implement. Alias chỉ gồm tên có tiền tố
      "luật/bộ luật/nghị định" hoặc chữ viết tắt; **tên chủ đề trơn** ("bảo hiểm
      xã hội", "mức lương tối thiểu", "lao động") **không** là alias vì thường
      chỉ chủ đề chứ không chỉ văn bản. Thêm/đổi văn bản trong corpus phải
      cập nhật bảng này; `source_document` chưa có trong bảng → không sinh
      `vb_*` cho chunk đó, log warning khi build (không lỗi).
    - **Phía document**: từ `source_document` của chunk (không parse
      breadcrumb) tra key X; thêm vào token của chunk `vb_X` (1 lần),
      `vb_X_điều_N`, và (nếu có Khoản) `vb_X_điều_N_khoản_M`, cùng hàm định
      dạng với phía query. Thêm ≤ 3 token/chunk nữa.
    - **Phía query** — `detect_document(query) -> str | None`: chuẩn hoá NFC +
      lowercase + bỏ dấu cả câu hỏi và alias, tìm mọi alias xuất hiện, **alias
      dài thắng alias ngắn khi các khoảng khớp chồng lấn** (vd. "nghị định
      hướng dẫn bộ luật lao động" thắng "bộ luật lao động" bên trong nó). Còn
      đúng 1 văn bản → dùng key đó; 0 văn bản → `None`; ≥ 2 văn bản khác
      nhau (vd. "Điều 3 luật BHXH và Điều 5 luật BHYT") hoặc "nghị định" trơn
      không đủ phân biệt → `None` (mơ hồ, **không sinh token văn bản**; gán
      văn bản theo vị trí sát từng Điều chưa làm). Khi có key X: thêm `vb_X`,
      `vb_X_điều_N` cho mỗi Điều nhận diện được, `vb_X_điều_N_khoản_M` cho mỗi
      cặp (Điều, Khoản); áp cho sparse nhánh B và **mọi** sub-query mỗi Điều
      (sub-query Điều i chỉ sinh token văn bản của Điều i), không áp cho nhánh
      A. Trần: 28 token/lượt (15 token Điều/Khoản + 1 + 3 + 9). Câu không nêu
      văn bản: hành vi như trước (chỉ token Điều/Khoản).
    - **Kỳ vọng** (chưa đo, cần đo sau rebuild): token kết hợp
      `vb_blld_điều_36`/`vb_blld_điều_113` có idf cao (df 4 và 7 trên 2.228)
      nên cả 4 chunk BLLĐ Điều 36 và cả 7 chunk BLLĐ Điều 113 đứng đầu sparse
      của sub-query tương ứng, kể cả chunk dài (điểm token này giảm theo độ
      dài nhưng vẫn vượt xa chunk không có token).
  - Ảnh hưởng: chi phí thêm ≤ 6 token/chunk (dl tăng không đáng kể). Va chạm
    với token pyvi dạng `điều_khoản` là không thể vì token cấu trúc luôn có
    hậu tố số hoặc tiền tố `vb_`; test kiểm tra không có va chạm.
  - **Cần build lại** sparse index và `bm25_params.json`: người dùng chạy
    `tools/sparse_index_documents.py` sau khi merge (mục 13A). Để nhận biết
    params cũ, `bm25_params.json` thêm field `params_version` (số nguyên; bản
    có token cấu trúc Điều/Khoản = 2, bản có thêm token theo văn bản = **3**
    (phiên bản kỳ vọng hiện tại), bản cũ không có field = 1); pipeline khi load thấy
    thiếu/khác phiên bản kỳ vọng thì báo lỗi rõ, **nêu nguyên văn lệnh
    rebuild** `uv run python tools/sparse_index_documents.py`, thay vì chạy với
    params cũ (tránh lệch im lặng giữa index và encoder). Developer **phải xác
    nhận với code `bm25.py` hiện có** (cách lưu/đọc `bm25_params.json`, chỗ
    load) trước khi thêm field và điều chỉnh cơ chế này cho khớp.
  - **Bằng chứng mô phỏng** (offline, 2026-09-20; 40 câu viện dẫn tự sinh, 10
    câu × 4 văn bản; chấm bằng BM25 chạy riêng trên toàn corpus, **chưa qua
    Pinecone**, cần đo lại sau khi rebuild):

    | | top1 | top3 | top10 |
    | --- | --- | --- | --- |
    | BM25 hiện tại | 35% | 65% | 85% |
    | + token cấu trúc | 90% | 100% | 100% |

    Hai câu chẩn đoán: "Điều 3 khoản 1 của luật thuế TNCN quy định gì?" hạng
    17 → 1; "Khoản 1 Điều 113 Bộ luật Lao động nói gì?" hạng 10 → 3.
- **Lưu tham số**: `data/bm25/bm25_params.json` gồm `vocab`, `idf` (hoặc `df`
  + `N`), `avgdl`, `k1`, `b`, `params_version`. Load 1 lần khi khởi
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
  lượt/câu hỏi. Câu hỏi viện dẫn: sau khi thêm extras (mục 8.1) fetch thêm tối
  đa 1 lần, chỉ metadata (không cần values), chỉ khi còn id thiếu → tổng tối
  đa 3 lượt.
- **MMR tắt**: không cần vector; fetch **1 lần cho toàn bộ union** (chỉ lấy
  metadata của id thiếu, đã gồm extras nếu là câu viện dẫn) — tối đa 1
  lượt/câu hỏi.
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
| Số lượt fetch (mục 6.4) | ≤ 2 (1/nhánh, trước MMR); câu viện dẫn thêm ≤ 1 sau union (mục 8.1) | ≤ 1 (cho cả union, gồm extras) |

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
chunk (câu viện dẫn: cộng extras, mục 8.1, tối đa `2 × BRANCH_TOP_N + 24`).
Giữ thứ hạng của mỗi chunk trong nhánh của nó (sau MMR nếu bật, sau RRF
nếu tắt) để phục vụ fallback (mục 9). Sau union (và sau extras nếu có), bổ
sung metadata cho id còn thiếu (`fill_missing`, mục 6.4): MMR tắt 1 lượt cho
cả union; MMR bật chỉ thêm 1 lượt khi câu viện dẫn có id thiếu.

### 8.1. Đảm bảo top-k sparse cho câu hỏi viện dẫn (chốt 2026-09-20)

Vấn đề (điểm mở 5 cũ): với câu hỏi viện dẫn Điều/Khoản, chunk đáp án thường
nằm cao ở sparse nhưng dense xếp thấp; RRF trọng số bằng nhau rồi cắt
`FUSION_TOP_N`/`BRANCH_TOP_N` (và MMR) khiến nó không vào union.

- **Nhận diện** (`retrieval/citation.py`, chốt 2026-09-20 theo lý thuyết):
  `extract_citation_numbers(query: str) -> list[int]` trả các số Điều theo
  thứ tự xuất hiện, dedupe giữ thứ tự, **tối đa `MAX_CITATION_ARTICLES = 3`**
  (phần dư bỏ qua và log info — chặn số lượt sparse phụ và latency rerank);
  `has_citation(query) = bool(extract_citation_numbers(query))`. Regex trên
  **câu hỏi gốc** (không phải hypo), chuẩn hoá Unicode NFC, không phân biệt
  hoa thường. Số Điều là tín hiệu chính; "khoản N" đứng một mình **không đủ**.
  Nguyên tắc: **ưu tiên không false positive hơn là bắt hết**; mỗi dạng hỗ
  trợ phải có ca dương và ca âm trong test (mục 15). Số Điều tối đa 3 chữ số
  (`\d{1,3}` không dính thêm chữ số, để "điều 2024" không khớp), đứng sau ranh
  giới từ (`(?<!\w)`). **Siết (chốt 2026-09-20)**: vì kết quả nhận diện còn
  kích hoạt token cấu trúc (mục 6.2) và extras (thêm passage vào union), nhận
  nhầm tốn thêm latency rerank — **số Điều ngay trước một đơn vị**
  (`tháng|ngày|năm|tuổi|lần|%|đồng|triệu`; **không** gồm "người" vì "Điều 36
  người lao động được quyền gì" là viện dẫn thật) **không** tính là viện dẫn,
  áp cho **mọi** số Điều (cả số đầu tiên, không chỉ số nối trong danh sách).
  Cũng trích số Khoản: `extract_citation_khoans(query) -> list[int]`
  (`khoản\s*(\d{1,3})`, dedupe giữ thứ tự, ≤ `MAX_CITATION_KHOANS = 3`), chỉ
  dùng khi đã có số Điều (mục 6.2).

  | Dạng | Quyết định | Ca dương / âm và lý do |
  | --- | --- | --- |
  | "Điều 36", "điều 3", "khoản 2 điều 36" | Hỗ trợ (hiện có) | "Điều 36 khoản 2 ..." → [36] |
  | "điều36" (liền) | Hỗ trợ (`\s*` thay cho `\s+`) | dương: "điều36 quy định gì" → [36]; "điều" liền chữ số gần như không xuất hiện trong văn nói thường, false positive thấp |
  | "Điều thứ 5" | Hỗ trợ (`điều\s*(?:thứ\s+)?\d`) | dương: "Điều thứ 5" → [5]; âm: "điều thứ hai" (không có số) |
  | "Điều 36.2" | Hỗ trợ, chỉ lấy 36 | dương: "Điều 36.2" → [36]; phần ".2" (Khoản) bỏ qua, vẫn nằm trong sub-query/BM25 |
  | Danh sách "Điều 3, 5 và 7", "các Điều 3, 5" | Hỗ trợ: sau một `điều N`, các số nối bằng `,` `;` `và` `hoặc` `hay` được tính là Điều | dương → [3, 5, 7]; âm: số nối mà ngay sau là đơn vị thì không tính, vd. "Điều 3 và 5 tháng" → [3]; dương: "Điều 3 và 5 người lao động" → [3, 5] |
  | "Điều 5 tháng", "điều 3 ngày", "điều 2 lần" (số Điều đứng ngay trước đơn vị) | **Không** khớp | âm → []; siết để tránh nhận nhầm kéo thêm passage vào union |
  | "Điều 36 người lao động được quyền gì" | Khớp | dương → [36]; "người" không nằm trong danh sách đơn vị loại trừ vì đây là cách hỏi viện dẫn thật |
  | Khoảng "Điều 3 đến Điều 5", "Điều 3 đến 5" | Hỗ trợ **chỉ hai đầu mút** (`đến|tới` là dấu nối như trên), **không** mở rộng các Điều ở giữa (khoảng có thể rất rộng, phá trần) | dương → [3, 5]; giới hạn ghi nhận |
  | "Đ.3", "Đ3", "đ 3" | **Không** hỗ trợ | "Đ" đơn lẻ dễ trùng ký hiệu/mã ngẫu nhiên; người dùng hiếm viết tắt vậy; âm: "Đ3", "mã Đ 3" → [] |
  | "Điều II", "Điều V" (La Mã) | **Không** hỗ trợ | corpus và breadcrumb dùng số Ả Rập ("Điều 36."); token La Mã không khớp BM25; "Điều I/V" dễ nhầm chữ thường; âm: "Điều II" → [] |
  | "Điều ba" (chữ) | **Không** hỗ trợ | không khớp token số trong corpus; "điều một/hai/ba" hay là từ thường ("điều ba người cần biết"); âm → [] |
  | "điều kiện", "điều khoản", "điều hành", "trong 3 điều kiện", "chiều 5" | Không khớp | không có chữ số ngay sau "điều" / không qua ranh giới từ |

  Điểm ("Điểm a") và tên/số hiệu văn bản không dùng để nhận diện.
- **Extras** — `citation_extras(...)`: extras luôn được thêm vào union (dedupe
  theo `chunk_id`), bất kể RRF/MMR/`BRANCH_TOP_N` đã cắt, và là **hit sparse
  thô** nên điểm của chúng là điểm BM25 (dot product sparse), không phải
  `rrf_score`. Không dùng hypo (luật HyDE cấm nêu số Điều). Hằng số nội bộ
  `retrieval/`, không vào `config.py` (mục 12): `CITATION_SPARSE_TOP_K = 10`
  (K), `MAX_CITATION_ARTICLES = 3`, `CITATION_EXTRAS_BUDGET = 24`. Gọi n là số
  Điều nhận diện được (1 ≤ n ≤ 3):
  - **n = 1 (giữ nguyên hành vi hiện tại)**: đúng K hit đầu của danh sách
    sparse thô của nhánh B (câu hỏi gốc). **Dùng lại kết quả
    `sparse_index.query` đã có — không thêm lượt gọi Pinecone sparse.**
  - **n ≥ 2 (chốt 2026-09-20)**: với mỗi Điều i chạy **1 truy vấn sparse
    riêng**, sub-query_i = câu hỏi gốc đã bỏ các số (kèm "điều"/"điều thứ"
    đứng trước nếu có) của **các Điều khác** — vd. "Điều 3 khoản 1 và Điều 5
    khoản 2" → Điều 3: "Điều 3 khoản 1 và khoản 2"; Điều 5: "khoản 1 và
    Điều 5 khoản 2" — rồi `encode_query` + `sparse_index.query(top_k=k_n)`.
    Các truy vấn này (n lượt, ≤ 3) chạy **song song** với 2 nhánh (`gather`
    ở bước 3 mục 13B). Với n ≥ 2 danh sách sparse thô của nhánh B **không**
    dùng cho extras (nhánh B vẫn chạy bình thường cho union).
  - **Quota** (nâng lên 2026-09-20 sau chẩn đoán "Điều 36 và Điều 113 Bộ luật
    Lao động"): chia tổng ngân sách cố định `CITATION_EXTRAS_BUDGET = 24` cho
    n ≥ 2: `k_n = floor(24 / n)` hit đầu của mỗi Điều — n=2 → 12/Điều (tổng 24),
    n=3 → 8/Điều (tổng 24); n=1 giữ K = 10 (hồi quy). Lý do: một Điều có thể
    có nhiều chunk (BLLĐ Điều 113 có 7 chunk Khoản 1-7, Điều 36 có 4; Khoản bị
    cắt "(phần i/n)" còn thêm chunk) nên quota cũ 8/5 mỗi Điều (ngân sách 16)
    không đủ chứa hết cho n=3 và sát trần cho n=2; 8/Điều (n=3) vẫn chứa được
    Điều 113 (7 chunk), 12/Điều (n=2) còn dư. Tổng ≤ 24 vẫn chặn trần: không cần
    trần tổng riêng, union tối đa `2 × BRANCH_TOP_N + 24` = 44 passage. Ưu tiên
    chất lượng/recall hơn latency: reranker đang chạy CPU nên câu nhiều Điều
    có thể ~45-50s/câu (~1s/passage); đo lại khi có GPU. Hạn chế đã biết: một
    Điều có > 10 chunk khi hỏi 1 Điều vẫn bị cắt ở K=10 (chưa nâng K, ghi ở mục
    16); ngân sách cố định nên có thể lẫn passage thừa khi Điều ít chunk (đề
    xuất sau: cắt theo khoảng cách điểm sparse, chưa làm).
  - **Thứ tự extras E**: xen kẽ theo Điều (Điều1#1, Điều2#1, Điều3#1,
    Điều1#2, …), bỏ trùng `chunk_id` (chunk ở nhiều danh sách giữ ở vị trí
    sớm nhất) — dùng làm danh sách E của fallback (mục 9).
  - **Lý do chọn hướng (a)** thay vì (b) (giữ danh sách thô nhánh B, fetch
    metadata để gán chunk cho Điều theo breadcrumb rồi chia quota): (b) không
    tốn lượt sparse nhưng chỉ chọn trong SPARSE_TOP_N=20 hit thô dùng chung nên
    một Điều có thể chiếm gần hết, Điều kia rơi ngoài top 20 (recall kém đúng ở
    ca cần cải thiện), và phải fetch metadata *trước* khi chia quota (thêm một
    lượt fetch vào đường chính). (a) tốn ≤ 3 lượt sparse (rẻ, song song, không
    phụ thuộc Groq/HF nên không nằm trên đường găng) đổi lấy quota công bằng và
    sub-query bớt nhiễu từ số Điều khác.
  - **Lỗi sparse của lượt phụ (chốt 2026-09-20): DEGRADE, không raise.**
    Lượt phụ của một Điều lỗi (sau hết retry) → log warning, bỏ extras của
    Điều đó, các Điều khác vẫn dùng; nếu **mọi** lượt phụ lỗi → lùi về hành vi
    n=1 (K=10 hit đầu của danh sách sparse thô nhánh B). Lỗi sparse ở 2 nhánh
    chính giữ nguyên (raise `RetrievalError`, mục 10). Đây là **ngoại lệ có
    chủ ý** của nguyên tắc mục 10 ("chỉ degrade 2 điểm"): extras chỉ là phần
    tăng cường, để một lượt phụ hỏng làm hỏng cả câu trong khi đường chính vẫn
    trả lời được là quá nặng tay (mục 10 ghi là điểm degrade thứ 3).
  - **Giới hạn đã biết (chấp nhận)**: sub-query chỉ xoá **số Điều** của các
    Điều khác, không xoá "khoản Y" đi kèm (không phân định chắc "khoản" thuộc
    Điều nào) nên khoản của Điều kia vẫn lẫn vào sub-query; cần đo lại.
- **Câu không viện dẫn**: pipeline giữ nguyên hoàn toàn (không thêm latency,
  không đổi kết quả). **Câu 1 Điều**: y hệt hành vi cũ.
- **Metadata**: chunk extras chỉ có ở sparse cần fetch metadata; bước
  `fill_missing` cho union chạy **sau** khi thêm extras (số lượt fetch: mục 6.4;
  không đổi theo n). Chunk fetch không trả về thì bỏ, log warning.
- **Số lượt gọi API**: Pinecone sparse query: 2 (n ≤ 1) hoặc 2 + n (n ≥ 2,
  tối đa 5); các lượt còn lại như mục 3.
- **Bằng chứng** (2026-09-20): 40 câu viện dẫn tự sinh từ corpus (10 câu × 4
  văn bản: Bộ luật Lao động, Luật BHXH, Luật BHYT, Luật thuế TNCN; dạng "Điều
  N khoản M <tên văn bản> quy định gì?"; đáp án chuẩn = chunk có đúng
  Điều/Khoản; chỉ đo nhánh B, không HyDE/reranker). Recall chunk đáp án nằm
  trong union khi thêm top-k sparse (k=0 là hiện trạng):

  | k | MMR bật | MMR tắt |
  | --- | --- | --- |
  | 0 | 25% | 65% |
  | 3 | 70% (+1,8 chunk/câu) | 72% (+0,4) |
  | 5 | 80% (+3,0) | 78% (+1,1) |
  | 10 | 85% (+6,0) | 85% (+5,5) |
  | 20 | 90% (+14,9) | 90% (+13,7) |

  **Quyết định chọn k=10** (2026-09-20): ưu tiên chất lượng/recall — k=5 đạt
  80%/78% còn k=10 đạt 85%/85% (MMR bật/tắt). Đánh đổi: thêm trung bình ~6
  passage (MMR bật 6,0; MMR tắt 5,5) cho mỗi câu viện dẫn, latency rerank tăng.
  Chấp nhận vì hiện reranker chạy trên CPU để đạt kết quả tốt nhất; sẽ bật GPU
  sau nên latency là vấn đề của phase khác, đo lại khi có GPU.
  - **Xác nhận thực tế** trên hệ thống thật sau PR #22 (khi k=5): "Điều 36
    khoản 2" được cứu (hạng 1); "Điều 3 khoản 1" (chunk đáp án ở sparse hạng
    16) và "Khoản 1 Điều 113" (sparse hạng 10) vẫn lỡ. Với k=10 kỳ vọng cứu
    được "Điều 113"; "Điều 3 khoản 1" vẫn cần k≥16.
  - Latency quan sát trên CPU (k=5): câu viện dẫn ~30s, câu thường ~17s.
  - Caveat: câu hỏi tổng hợp một dạng, chỉ đo nhánh B nên là cận dưới (nhánh
    A có thể thêm chút); k là tham số cần tinh chỉnh lại bằng RAGAS/latency
    khi có GPU.
  - **Sau khi có token cấu trúc (mục 6.2)**: mô phỏng BM25 top3 = 100% nên k
    nhỏ hơn có thể đủ (giảm số passage rerank và latency); nhưng **giữ k=10 lúc
    này**, cân nhắc giảm sau khi rebuild và đo lại qua Pinecone (mục 16).

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
- Sort giảm dần theo score, cắt `FINAL_TOP_K` (=5), gán `rerank_score`. Chỉ
  tin thứ tự sau rerank, **không ghim/can thiệp thứ tự** (mọi loại câu hỏi,
  kể cả viện dẫn; xem quyết định gỡ ghim ở mục 16 điểm 11).

**Server tự host rất dễ lỗi** (Studio sleep/restart, tmux/ngrok chết, model
đang load, CPU chậm, quá tải) — xử lý như sau:

- **Tách connect timeout và read timeout, read timeout tỉ lệ số passage**
  (chốt 2026-09-20, theo đo thực tế): `RerankerSettings` giữ
  `connect_timeout_seconds` (ngắn, 5s) và `timeout_seconds` (30s) — nay
  `timeout_seconds` là **sàn** của read timeout. Read timeout thực tế của mỗi
  request = `max(timeout_seconds, RERANK_SECONDS_PER_PASSAGE × n_passages)`,
  `RERANK_SECONDS_PER_PASSAGE = 2.5` (hằng số nội bộ `reranker_client.py`,
  không vào `config.py`). Lý do: đo trên CPU ~1s/passage (câu 1 Điều ~30
  passage hoàn thành ~33s cả pipeline), 2.5s/passage là biên an toàn ~2,5 lần;
  union tối đa 44 passage (câu ≥ 2 Điều, mục 8.1) → ~110s. Timeout cố định 30s đã làm union ≥ 25-36
  passage bị ReadTimeout 3 lần liên tiếp. Khi bật GPU reranker nhanh hơn nhưng
  giới hạn này vẫn đủ rộng (chỉ là mức chờ tối đa, không làm chậm khi server
  nhanh); đo lại và hạ hằng số khi có GPU.
- **Phân loại lỗi:**
  - **Retry** (tối đa `max_retries`, backoff ngắn): **chỉ** `ConnectError`,
    `ConnectTimeout` và HTTP 502/503/504 — những lỗi mà server chắc chắn chưa
    xử lý request.
  - **KHÔNG retry `ReadTimeout`** (và các `httpx.TransportError` khác không
    thuộc nhóm retry, vd. `WriteTimeout`): sang fallback ngay. Lý do: server có
    thể vẫn đang tính request trước; retry gửi lại toàn bộ passage làm hàng đợi
    phình, càng timeout (câu nhiều Điều từng mất ~100s rồi vẫn fallback). Log
    warning rõ: "reranker chậm với N passage (read timeout X s), có thể do
    CPU/quá tải; các yêu cầu trước có thể còn trong hàng đợi".
  - **Không retry, log error rõ ràng, sang fallback ngay**: HTTP 401/403 (sai
    `RERANKER_API_KEY`), 422/400 (payload sai) — lỗi cấu hình/code, retry vô
    ích.
  - Với lỗi kết nối/5xx sau khi hết retry, log warning nhắc **kiểm tra Studio
    theo runbook mục 9.1**.
  - Thời gian chờ tối đa/câu vì thế là 1 lần read timeout (không còn
    `timeout × (1 + max_retries)`); lỗi kết nối thất bại nhanh (connect
    timeout 5s × (1 + `max_retries`)).
- **Validate response**: `scores` phải là list số hữu hạn, đúng độ dài
  `passages`; sai → coi như lỗi.
- **Fallback khi hết retry** (chốt 2026-09-19): không crash, log warning, trả
  top `FINAL_TOP_K` với `rerank_score=None`. Thứ tự: xen kẽ round-robin các
  danh sách theo thứ hạng (A1, B1, E1, A2, B2, E2, …), bỏ chunk trùng; E là
  danh sách extras (mục 8.1), chỉ có khi câu hỏi viện dẫn; với nhiều Điều, E
  đã được xen kẽ theo Điều sẵn (Điều1#1, Điều2#1, …) nên fallback vẫn coi E là
  một danh sách duy nhất. Cách này không cần
  vector nên dùng được ở cả 2 chế độ MMR. Nếu nhánh A vắng (Groq lỗi) thì bỏ
  qua A (B1, E1, B2, E2, …).
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

## 10. Xử lý lỗi (degrade 2 điểm chính + 1 ngoại lệ extras)

Chỉ degrade ở 2 điểm hay lỗi nhất và có phương án thay thế rẻ, cộng 1 ngoại lệ
có chủ ý là lượt sparse phụ của extras viện dẫn (dòng thứ 3 bảng dưới); các lỗi
còn lại raise rõ ràng, không cố xử lý từng trường hợp (giữ đơn giản).

| Lỗi (sau hết retry) | Xử lý |
| --- | --- |
| Groq lỗi/timeout hoặc trả rỗng | Bỏ nhánh A; chỉ embed câu hỏi gốc (batch 1) và chạy nhánh B; log warning |
| Reranker lỗi | Fallback mục 9 |
| Sparse lượt phụ của extras (câu ≥ 2 Điều, mục 8.1) lỗi | Log warning, bỏ extras của Điều đó, các Điều khác vẫn dùng; mọi lượt phụ lỗi → lùi về hành vi n=1. Lý do: extras chỉ là phần tăng cường, đường chính vẫn trả lời được nên không đáng làm hỏng cả câu |
| HF embed hoặc Pinecone (dense/sparse của 2 nhánh chính/fetch) lỗi | Raise `RetrievalError` kèm nguyên nhân; bước generation quyết định xử lý |

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
  `MMR_LAMBDA`/`BRANCH_TOP_N`/`USE_MMR`/`FINAL_TOP_K`/`CITATION_SPARSE_TOP_K`/`MAX_CITATION_ARTICLES`/
  `CITATION_EXTRAS_BUDGET`/`MAX_CITATION_KHOANS`/
  `RERANK_SECONDS_PER_PASSAGE` vào `config.py` — hằng số nội bộ của
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
         + token cấu trúc điều_N / khoản_M / điều_N_khoản_M parse từ breadcrumb
         + token theo văn bản vb_X / vb_X_điều_N / vb_X_điều_N_khoản_M từ source_document (mục 6.2)
      3. bm25.fit(all_texts + token cấu trúc): tokenize (pyvi) -> vocab + df/idf + avgdl chung cho cả corpus
         -> lưu data/bm25/bm25_params.json (kèm params_version)
      4. bm25.encode_document(text, token cấu trúc) cho từng chunk riêng lẻ -> sparse vector {indices, values}
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
Chạy lại script này cũng bắt buộc **một lần sau khi merge PR thêm token theo
văn bản** (mục 6.2, `params_version = 3`): index và `bm25_params.json` cũ không
có token `vb_*` (và bản trước đó không có cả `điều_N…`), `retrieve()` sẽ báo lỗi
`params_version` cho tới khi build lại: `uv run python tools/sparse_index_documents.py`
(việc của người dùng).

### 13B. Online — `retrieve(query)`

Library function, không CLI:

```
retrieval.pipeline.retrieve(query: str, *, use_mmr: bool | None = None) -> list[RetrievedChunk]

  use_mmr = USE_MMR if use_mmr is None else use_mmr                  # mục 7
  1. hypo = await hyde.generate(query)                                # mục 4 (lỗi → chỉ nhánh B, mục 10)
  2. emb_hypo, emb_query = await embedder.embed([hypo, query])        # mục 5, 1 request HF
  numbers = extract_citation_numbers(query)                           # mục 8.1 (≤ MAX_CITATION_ARTICLES, [] nếu không viện dẫn)
  khoans  = extract_citation_khoans(query) if numbers else []         # mục 8.1
  doc     = detect_document(query) if numbers else None               # mục 6.2: key văn bản duy nhất trong câu hỏi, None nếu không nêu/mơ hồ
  terms   = structural_terms(numbers, khoans, doc)                    # mục 6.2 (điều_N, khoản_M, điều_N_khoản_M + vb_X, vb_X_điều_N, vb_X_điều_N_khoản_M nếu có doc); [] nếu không viện dẫn
  3. branch_a, branch_b, article_hits = await gather(
       run_branch(text=hypo,  dense_emb=emb_hypo,  extra_terms=[]),   # nhánh A (hypo không có số Điều → không token cấu trúc)
       run_branch(text=query, dense_emb=emb_query, extra_terms=terms),# nhánh B (trả thêm sparse hit thô)
       citation_article_hits(query, numbers) if len(numbers) >= 2 else none())
         # mục 8.1: mỗi Điều 1 sparse query riêng (sub-query bỏ số Điều khác), top k_n = floor(24/n), song song; lượt phụ lỗi → degrade (mục 8.1, 10)

     run_branch(text, dense_emb, extra_terms):
       a. dense, sparse = await gather(
            dense_search.query(dense_emb, DENSE_TOP_N, include_values=use_mmr),  # mục 6.1
            sparse_index.query(text, SPARSE_TOP_N, extra_terms))      # mục 6.2
       b. fused = fusion.rrf(dense, sparse, k=RRF_K)[:FUSION_TOP_N]   # mục 6.3
       c. if use_mmr:
            fused = await fetch_missing(fused)                        # mục 6.4: vector+metadata
            return mmr.select(fused, emb_query, MMR_LAMBDA, BRANCH_TOP_N)   # mục 7
          return fused[:BRANCH_TOP_N]

  4. union = dedupe_by_chunk_id(branch_a + branch_b)                  # mục 8
     if numbers:                                                      # = has_citation(query), mục 8.1
         union += citation_extras(numbers, branch_b_sparse_hits, article_hits)
           # 1 Điều: K=10 hit đầu sparse thô nhánh B (không thêm lượt sparse);
           # ≥2 Điều: top k_n mỗi Điều, xen kẽ theo Điều, dedupe; tổng ≤ 24
     union = await fill_missing_metadata(union)                       # mục 6.4: MMR tắt 1 lượt cho cả union;
                                                                      # MMR bật chỉ khi còn id thiếu (extras)
  5. scores = await reranker_client.rerank(query, [c.breadcrumb + "\n" + c.content for c in union])
       # mục 9: read timeout = max(timeout_seconds, 2.5 × len(union)); ReadTimeout không retry (→ fallback)
  6. ranked = sort theo scores (lỗi → thứ tự fallback mục 9)
  7. return ranked[:FINAL_TOP_K] dạng list[RetrievedChunk]            # không ghim, chỉ tin thứ tự rerank
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
| `citation.py` | Nhận diện & extras cho câu hỏi viện dẫn (mục 8.1): `extract_citation_numbers(query) -> list[int]` (≤ 3 số Điều, theo thứ tự, dedupe); `has_citation(query)` = `bool(...)`; `build_article_queries(query, numbers)` — sub-query mỗi Điều (bỏ số Điều khác); `citation_extras(numbers, branch_b_hits, article_hits)` — 1 Điều: K hit đầu sparse thô nhánh B; ≥ 2 Điều: top `k_n` mỗi Điều xen kẽ, dedupe; `extract_citation_khoans(query)`; `parse_breadcrumb(breadcrumb) -> (Điều, Khoản)`; bảng `DOCUMENTS` (source_document → key + alias, mục 6.2) và `detect_document(query) -> str | None`; `structural_terms(numbers, khoans, doc)` / `breadcrumb_structural_terms(breadcrumb, source_document)` — cùng một hàm định dạng token `điều_N`, `khoản_M`, `điều_N_khoản_M`, `vb_X`, `vb_X_điều_N`, `vb_X_điều_N_khoản_M` (mục 6.2) |
| `reranker_client.py` | HTTP client gọi server LightningAI, read timeout tỉ lệ số passage, retry (chỉ connect/502/503/504)/validate/fallback (mục 9) |
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
- `citation.py` có unit test cho `extract_citation_numbers`/`has_citation`, mỗi
  dạng ở mục 8.1 có ca dương và ca âm: "Điều 36 khoản 2 ..." → [36], "khoản 2
  điều 36 ..." → [36], "điều 3" → [3], "điều36" → [36], "Điều thứ 5" → [5],
  "Điều 36.2" → [36], "Điều 3, 5 và 7" → [3, 5, 7], "Điều 3 đến Điều 5" →
  [3, 5], "Điều 3 và 5 tháng" → [3], "Điều 36 người lao động được quyền gì" → [36]; dedupe/giữ thứ tự ("Điều 5 và Điều 3 và
  Điều 5" → [5, 3]); trần: 4 Điều → chỉ 3 số đầu; sai (→ []) với "điều kiện
  lao động", "trong 3 điều kiện", "điều khoản", "điều hành", "Đ3", "Đ.3",
  "điều 5 tháng", "Điều 3 ngày", "Điều 2 lần" (số Điều đứng ngay trước đơn vị),
  "Điều II", "Điều ba", "điều 2024", "khoản 2 quy định gì" và câu hỏi tự
  nhiên không viện dẫn.
- Nhiều Điều (n ≥ 2): `build_article_queries` cho ra sub-query mỗi Điều đã bỏ
  số Điều khác; mỗi Điều có extras riêng (chunk đáp án nằm trong top `k_n` sparse
  của sub-query Điều đó thì có trong union dù Điều kia chiếm cao ở danh sách
  chung); quota `k_n = floor(24/n)` (n=2 → 12, n=3 → 8) và tổng extras ≤ 24;
  thứ tự E xen kẽ theo Điều, dedupe; số lượt Pinecone sparse = 2 + n (n ≥ 2).
  Lượt phụ sparse lỗi (fake client ném lỗi) → degrade: bỏ extras của Điều đó,
  các Điều khác vẫn có, log warning, `retrieve()` không raise; mọi lượt phụ lỗi
  → extras = 10 hit đầu sparse thô nhánh B; lỗi sparse ở nhánh chính vẫn raise
  `RetrievalError`. 1 Điều: extras đúng 10 hit đầu sparse thô nhánh
  B, không thêm lượt sparse (không đổi so với hiện trạng).
- `extract_citation_khoans`: "khoản 1" → [1], "Điều 3 khoản 1 và khoản 2" →
  [1, 2], trần 3, dedupe giữ thứ tự; "khoản" không số → [].
- Token cấu trúc (mục 6.2): `parse_breadcrumb` lấy đúng Điều/Khoản từ breadcrumb
  có "Điểm" và hậu tố "(phần i/n)"; token phía document và phía query sinh từ
  cùng hàm định dạng nên khớp nhau (breadcrumb "… - Điều 3. … - Khoản 1" và câu
  "Điều 3 khoản 1" cùng cho `điều_3`, `khoản_1`, `điều_3_khoản_1`); trên corpus
  mẫu, chunk đáp án của "Điều 3 khoản 1" xếp trên chunk **cùng số 3 hoặc 1 nhưng
  khác Điều** và trên chunk chỉ trùng tên văn bản; nhánh A (hypo) không có
  `extra_terms`; sub-query mỗi Điều chỉ có token của Điều đó; trần ≤ 28 token;
  term ngoài vocab bị bỏ; `bm25_params.json` thiếu/khác `params_version` → lỗi
  rõ có chứa lệnh `uv run python tools/sparse_index_documents.py`, không chạy với params cũ; không va chạm với token pyvi có `_`.
- Token cấu trúc theo văn bản (mục 6.2): `detect_document` — "Điều 36 và Điều 113
  Bộ luật Lao động" → BLLĐ; "Điều 3 khoản 1 luật thuế TNCN"/"luật thuế thu nhập
  cá nhân" → TNCN; không dấu ("luat bhxh") vẫn khớp; "luật lao động" → BLLĐ;
  "Nghị định hướng dẫn Bộ luật Lao động" → Nghị định điều kiện lao động (alias
  dài thắng alias ngắn "bộ luật lao động"); câu không nêu văn bản, chỉ có
  "nghị định" trơn, hoặc nêu ≥ 2 văn bản → `None` (hành vi như trước khi có
  token văn bản, không sinh `vb_*`); tên chủ đề trơn ("bảo hiểm xã hội", "mức
  lương tối thiểu") không phải alias. Token document và query cùng hàm định
  dạng: chunk BLLĐ Điều 36 Khoản 1 cho `vb_blld`, `vb_blld_điều_36`,
  `vb_blld_điều_36_khoản_1`; `source_document` không có trong bảng `DOCUMENTS`
  → không sinh `vb_*` cho chunk đó + log warning khi build; trên corpus thật, mọi
  chunk BLLĐ Điều 113 (7 chunk) và Điều 36 (4 chunk) đứng đầu sparse của
  sub-query tương ứng khi câu có "Bộ luật Lao động" (đo lại sau rebuild); `params_version = 3`.
- Reranker client (fake transport `httpx.MockTransport`): read timeout gửi đi =
  `max(timeout_seconds, 2.5 × n_passages)` (n nhỏ → 30s, n=44 → 110s); `ReadTimeout`
  **không retry** (đúng 1 lần gọi) và sang fallback kèm log cảnh báo nêu số
  passage/timeout; `ConnectError`/`ConnectTimeout` và 502/503/504 vẫn retry tới
  `max_retries` (vd. 503 hai lần rồi 200 → có scores); 401/403/400/422 không
  retry.
- Câu viện dẫn 1 Điều: chunk đáp án nằm trong sparse top-10 của nhánh B thì có trong
  union dù RRF/MMR đã loại; câu không viện dẫn: union giống hệt hiện trạng,
  không có extras; không thêm lượt Pinecone sparse (1 Điều); số lượt fetch đúng mục 3 /
  6.4 cho cả hai chế độ MMR (MMR bật ≤ 3, MMR tắt ≤ 1 với câu viện dẫn);
  fallback rerank xen kẽ có danh sách E (A1, B1, E1, …).
- Vector query được embed qua cùng tiền xử lý `pyvi` như lúc index (test:
  câu hỏi trùng nguyên văn `content` 1 chunk phải trả về chunk đó ở top dense).
- Giả lập Groq lỗi và reranker lỗi (timeout, 401) → hành vi đúng mục 10 và mục 9, `retrieve()` không crash.
- Test HyDE (fake client Groq): system prompt chứa các quy tắc cấm (không nêu
  số Điều/Khoản/Điểm/Chương/Mục, tên/số hiệu văn bản, năm ban hành); câu hỏi đi
  vào message `user`; lời gọi truyền đúng `reasoning_effort="low"`,
  `temperature=0.2`, `max_completion_tokens=2048` (hoặc giá trị đã điều chỉnh
  theo xác nhận SDK, mục 4); output rỗng vẫn coi là lỗi → `None` (không đổi).
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

Đã chốt (2026-09-20, điểm mở 5): recall câu hỏi viện dẫn giải quyết bằng
hướng A+C — `has_citation` regex + luôn thêm top `CITATION_SPARSE_TOP_K=10`
sparse của nhánh B vào union (mục 8.1); k tăng từ 5 lên 10 cùng ngày, cần
tinh chỉnh lại bằng RAGAS/latency khi có GPU.

1. **[ĐÃ CHỐT 2026-09-20 — theo lý thuyết, chưa đo]** Nguyên văn prompt HyDE và
   tham số Groq (mục 4): tách system/user, trung lập lĩnh vực, 3-4 câu
   (~60-100 từ), cấm viện dẫn/tên văn bản/số hiệu/năm, hạn chế con số cụ thể,
   không từ chối câu mơ hồ, câu chỉ hỏi theo số Điều thì sinh đoạn chung;
   `reasoning_effort="low"`, `temperature=0.2`, `max_completion_tokens=2048`.
   Chốt bằng lập luận, **không dùng RAGAS ở phase này**; cần đo lại ở phase
   đánh giá: hiệu quả nhánh A có/không có HyDE, temperature, độ dài đoạn.
2. Xác minh tmux/`on_start.sh` có giúp Studio khỏi sleep không (mục 9.1).
3. MMR bật hay tắt làm mặc định production — quyết định sau khi đánh giá bằng
   RAGAS (mục 7). Các hằng số khởi điểm (`DENSE_TOP_N=20`, `SPARSE_TOP_N=20`,
   `RRF_K=60`, `FUSION_TOP_N=20`, `MMR_LAMBDA=0.5`, `BRANCH_TOP_N=10`,
   `FINAL_TOP_K=5`) tinh chỉnh cùng lúc đó.
4. **Bàn sau** (ngoài flow hiện tại): tối ưu latency/API call — benchmark giới
   hạn batch/concurrency của HF, Groq, Lightning; dynamic batching; circuit
   breaker cho reranker; khởi động sớm nhánh B trong lúc chờ Groq; theo dõi quota
   HF dùng chung.
5. **[ĐÃ CHỐT 2026-09-20 — hướng A+C, k=10, xem mục 8.1]** **Recall cho câu hỏi
   viện dẫn Điều/Khoản** (chẩn đoán 2026-09-20): với 3
   câu viện dẫn (Điều 3 khoản 1 / Điều 36 khoản 2 / Khoản 1 Điều 113 Bộ luật
   Lao động), chunk đáp án nằm ở sparse top 5/10/16 (dense hạng 45 hoặc ngoài
   top 100), nhưng RRF (trọng số bằng nhau) đẩy nó xuống hạng 10/20/31 của
   danh sách fused, rồi bị cắt bởi `FUSION_TOP_N=20` và `BRANCH_TOP_N=10` (và
   MMR) nên không vào union — reranker (kể cả có breadcrumb) không cứu được.
   Các hướng đã cân nhắc: tăng `BRANCH_TOP_N`/`FUSION_TOP_N` (đổi lại latency
   rerank vì tỉ lệ thuận số passage); đảm bảo top-k sparse luôn vào union
   (**chọn, hướng A**); tăng trọng số sparse trong RRF; tắt MMR; nhận diện
   viện dẫn bằng regex (**chọn, hướng C**, để chỉ câu viện dẫn bị ảnh hưởng).
   Kết quả đo 40 câu: recall trong union k=0 → k=10 tăng 25% → 85% (MMR bật),
   65% → 85% (MMR tắt); k=5 đạt 80%/78%. Trên hệ thống thật (k=5) "Điều 36
   khoản 2" được cứu, "Điều 3 khoản 1" (sparse hạng 16) và "Khoản 1 Điều 113"
   (sparse hạng 10) vẫn lỡ; k=10 kỳ vọng cứu "Điều 113", "Điều 3 khoản 1"
   cần k≥16.
6. **MMR làm giảm mạnh recall câu viện dẫn**: hiện trạng (k=0) recall chỉ 25%
   khi MMR bật so với 65% khi tắt (MMR dựa trên cosine dense, không có nghĩa
   với số Điều). Đưa vào so sánh RAGAS (điểm 3); cân nhắc tắt MMR mặc định
   hoặc bỏ MMR khi `has_citation`.
7. **[ĐÃ CHỐT 2026-09-20 — theo lý thuyết, chưa đo]** Mở rộng nhận diện viện
   dẫn và chia quota cho câu nhiều Điều (mục 8.1): nhận "điều36", "Điều thứ
   5", "Điều 36.2", danh sách/khoảng (chỉ hai đầu mút), trần 3 Điều; không nhận
   "Đ.3"/"Đ3", La Mã, chữ. n ≥ 2: 1 sparse query riêng mỗi Điều (sub-query bỏ
   số Điều khác), `k_n = floor(24/n)` (`CITATION_EXTRAS_BUDGET = 24`, nâng từ 16 ở
   điểm 12: 12/Điều với n=2, 8/Điều với n=3; tổng ≤ 24, union ≤ 44), xen kẽ
   theo Điều; n = 1 giữ nguyên. Lượt sparse phụ lỗi thì degrade (ngoại lệ có
   chủ ý, mục 10). Ưu tiên recall hơn latency: trên CPU câu nhiều Điều có
   thể ~45-50s, đo lại khi có GPU. Chấp nhận chưa đo: false positive "điều 5 tháng", chất lượng sub-query
   (chỉ xoá số Điều khác, không xoá "khoản Y"), quota. Cần đo lại bằng RAGAS/bộ câu hỏi nhiều Điều sau này
   (kể cả có nên tăng quota/trần, nhận thêm Điểm, tên/số hiệu văn bản). k
   (`CITATION_SPARSE_TOP_K`) cũng cần tinh chỉnh lại bằng RAGAS/latency khi có
   GPU.
8. Recall còn lỡ ~15% ở k=10 (~20% ở k=5) chưa điều tra nguyên nhân (nghi ngờ câu trùng
   Điều/Khoản giữa nhiều văn bản, hoặc nhiễu sparse).
9. Có nên bỏ HyDE (tiết kiệm 1 lượt Groq) khi câu hỏi chỉ thuần viện dẫn, không
   nêu chủ đề — chỉ ghi nhận, chưa làm; xét cùng đánh giá hiệu quả nhánh A.
10. **[ĐÃ CHỐT 2026-09-20 — việc 1, PR riêng làm trước]** Reranker timeout/retry
    (mục 9): read timeout = `max(timeout_seconds, 2.5 × n_passages)`; không retry
    `ReadTimeout` (sang fallback + log cảnh báo), vẫn retry ConnectError/
    ConnectTimeout/502/503/504. Chốt theo đo thực tế (~1s/passage trên CPU, 3
    lần ReadTimeout với union ≥ 25-36 passage, retry làm phình hàng đợi). Đo lại
    và hạ `RERANK_SECONDS_PER_PASSAGE` khi có GPU.
11. **[ĐÃ CHỐT 2026-09-20 — token cấu trúc; ĐÃ GỠ ghim cùng ngày]** Token cấu
    trúc BM25 (`điều_N`, `khoản_M`, `điều_N_khoản_M`, mục 6.2), kèm siết nhận
    diện (số Điều ngay trước đơn vị không tính). Chốt theo dữ liệu chẩn đoán và
    mô phỏng BM25 offline (top1 35% → 90%, top3 65% → 100%), **chưa qua
    Pinecone/RAGAS**. Cần làm sau khi merge: (a) người dùng chạy lại
    `tools/sparse_index_documents.py`; (b) đo lại recall qua Pinecone sau rebuild;
    (c) cân nhắc giảm `CITATION_SPARSE_TOP_K` (top3 = 100% trong mô phỏng → ít
    passage rerank, latency thấp hơn) — **giữ k=10 lúc này**; (e) gán Khoản cho
    đúng Điều trong câu nhiều Điều (hiện dùng chung mọi Khoản cho mỗi Điều).

    **Gỡ ghim (2026-09-20)**: cơ chế "ghim chunk khớp chính xác Điều/Khoản lên
    đầu sau rerank" (PR #28 phần C, từng là mục 8.2) đã **gỡ hoàn toàn**; chỉ
    tin thứ tự sau rerank (mục 2, 9). Lý do: (1) ghim từng đưa chunk sai văn bản
    lên đầu (vd. BHXH Điều 3 Khoản 1, điểm rerank -8,33, lên hạng 2 dù câu nêu
    luật thuế TNCN); (2) người dùng chỉ tin thứ tự rerank. Bằng chứng thí nghiệm
    về giới hạn của reranker `AITeamVN/Vietnamese_Reranker` (cross-encoder ngữ
    nghĩa, **không khớp chính xác định danh Điều/Khoản**): trên 5 chunk (đáp án
    + 4 chunk gây nhiễu), câu "Điều 3 khoản 1 của luật thuế TNCN quy định gì?":
    đáp án đúng 1,12 < Điều 2 K1 (2,47) và Điều 27 K1 (1,67) → đáp án hạng 3/5
    (2 chunk kia có *nội dung* nhắc "Điều 3" nên khớp bề mặt, còn đáp án đúng
    chỉ có "Điều 3" trong breadcrumb, nội dung là "Thu nhập từ kinh doanh, bao
    gồm..."); viết lại câu theo dạng breadcrumb ("LUẬT THUẾ TNCN - Điều 3 -
    Khoản 1") **không** giúp (hạng 4/5). "Khoản 1 Điều 113 Bộ luật Lao động nói
    gì?": đáp án hạng 2/5 sau BLLĐ Điều 5 K1 (1,50 so với 1,07); câu viết dạng
    breadcrumb lên hạng 1 nhưng dạng khác lại hạng 2 (không ổn định); chỉ nội
    dung không breadcrumb tệ hơn hẳn (đáp án -6,16, mọi điểm âm). Câu ngữ nghĩa
    ("Thu nhập chịu thuế ... gồm những khoản nào?"): đáp án hạng 1 (5,46 so với
    0,37) → reranker tốt với câu ngữ nghĩa, kém với tra cứu theo định danh.
    **Hệ quả chấp nhận có chủ ý**: sau khi gỡ ghim, chunk khớp chính xác có
    thể đứng hạng 2-4 với câu viện dẫn thuần; xem lại khi có reranker mạnh hơn/
    GPU hoặc sau RAGAS.
12. **[ĐÃ CHỐT 2026-09-20 — token theo văn bản + nâng ngân sách extras]** Chẩn
    đoán "Điều 36 và Điều 113 Bộ luật Lao động quy định gì?" (union 31 chunk):
    BLLĐ Điều 36 có 4 chunk nhưng chỉ K3 vào union (thiếu 2 chunk K1 và 1 chunk
    K2); Điều 113 có 7 chunk, thiếu K1 (chunk nội dung chính, dài); reranker
    chấm 6 chunk Điều 113 còn lại từ -4,83 đến -6,59. Nguyên nhân là **recall
    ở bước extras, không phải reranker**: `điều_N` khớp mọi văn bản, tên văn bản
    idf thấp không lọc được, chuẩn hoá độ dài BM25 bất lợi cho chunk dài. Xử lý:
    token theo văn bản `vb_X`, `vb_X_điều_N`, `vb_X_điều_N_khoản_M` (mục 6.2,
    bảng `DOCUMENTS`, `params_version = 3`, rebuild bắt buộc: người dùng chạy
    `uv run python tools/sparse_index_documents.py`) và nâng
    `CITATION_EXTRAS_BUDGET` 16 → 24 (`k_n = floor(24/n)`, mục 8.1) để chứa hết
    chunk một Điều (7 chunk Điều 113). Chốt theo lý thuyết, **chưa đo**: cần đo
    lại sau rebuild — cả 4 chunk Điều 36 và cả 7 chunk Điều 113 có đứng đầu sparse
    không; số passage thừa do ngân sách cố định (cân nhắc cắt theo khoảng cách
    điểm sparse); nhiều văn bản trong một câu chưa gán được theo vị trí; K=10
    cho câu 1 Điều có thể cắt Điều có > 10 chunk.
13. **CHỜ NGƯỜI DÙNG QUYẾT ĐỊNH (chưa làm)**: `FINAL_TOP_K = 5` quá nhỏ khi hỏi
    trọn một Điều (BLLĐ Điều 113 có 7 chunk + Điều 36 có 4 chunk = 11), và
    reranker không phân biệt được các Khoản của cùng một Điều khi câu hỏi không
    có nội dung ngữ nghĩa (Điều 113: K4 -4,83, K5 -5,03 đứng đầu còn K1 nội dung
    chính bị bỏ). Hướng đề xuất: với câu viện dẫn **trọn Điều** (không nêu
    Khoản) và có tên văn bản (`detect_document` ≠ None), mở rộng số chunk trả về
    (vd. tới ~10-12) theo thứ tự rerank thay vì cắt 5. Cần người dùng quyết vì
    ảnh hưởng độ dài ngữ cảnh của bước generation (và mục 2 hiện hứa tối đa
    `FINAL_TOP_K` = 5 phần tử).
