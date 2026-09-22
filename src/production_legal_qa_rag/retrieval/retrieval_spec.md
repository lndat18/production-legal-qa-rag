# Retrieval — HyDE → Hybrid Search (2 nhánh) → RRF → MMR (bật/tắt) → Union + Extras → Rerank

## 1. Mục tiêu & phạm vi

Từ 1 câu hỏi tiếng Việt, trả về top-5 chunk pháp lý làm context cho bước
generation (spec khác, ngoài phạm vi).

**Tiêu chí quan trọng nhất (chốt 2026-09-20):** câu hỏi viện dẫn chính xác **một
Khoản** (chỉ phạm vi <= Khoản; ví dụ "Khoản 1 Điều 113 Bộ luật Lao động nói
gì?"):

- Chunk đáp án **bắt buộc nằm trong top 5** sau rerank. Không yêu cầu thứ hạng
  hay ngưỡng điểm cụ thể (LLM đọc cả 5 chunk).
- Kiểm chứng bằng bộ câu viện dẫn Khoản mẫu chạy thật (`retrieval/test.py`) —
  tiêu chí nghiệm thu thủ công, mục 15. RAGAS/đo lường đầy đủ ở phase sau.

**Ngoài phạm vi tối ưu:** câu hỏi cả Điều (không nêu Khoản), Điểm trở lên, nhiều
Điều — chạy pipeline như câu thường, top 5 theo rerank, **best-effort, chờ
phương án Agentic** (spec riêng). Không có nhánh xử lý riêng, không nới
`FINAL_TOP_K`.

**Trong phạm vi:** toàn bộ `retrieve(query)`; xây/duy trì **Pinecone sparse
index riêng** (BM25 tự viết; không mở rộng `embedding/`, package đó chỉ sở hữu
dense index); xử lý lỗi cho Groq và reranker.

**Không làm:** generation; multi-turn; cache; vận hành server reranker (chỉ
client + runbook 9.1); update sparse index incremental (mỗi lần build là fit +
upsert lại toàn bộ); tự kiểm tra đồng bộ dense/sparse/`bm25_params.json` (quy
ước: chạy lại `tools/sparse_index_documents.py` sau mỗi lần build lại dense
index); CLI test query; cơ chế ghim (pin) thứ tự; tối ưu latency nâng cao.

## 2. Input & Output

- **Input**: `query: str` (1 câu hỏi độc lập).
- **Output**: `list[RetrievedChunk]`, tối đa `FINAL_TOP_K = 5`, giảm dần theo
  thứ tự rerank (không can thiệp thêm).

`RetrievedChunk` (pydantic v2, `retrieval/models.py`): `chunk_id`,
`source_document`, `breadcrumb`, `content`, `has_table`, `raw_table` (từ metadata
Pinecone, `embedding.models.PineconeMetadata`), `rerank_score: float | None`
(`None` khi rerank lỗi và đã fallback). Không có `token_count`/`is_split`/
`split_*` (metadata Pinecone không lưu).

Đầu vào tĩnh: `data/chunks/*.json` (corpus để fit BM25 + build sparse index);
`data/bm25/bm25_params.json` (tham số BM25, đọc 1 lần lúc khởi tạo pipeline).

## 3. Flow & số chunk từng bước

```
query → Groq (1 call) → hypo
      → HF embed [hypo, query] (1 call) → emb_hypo, emb_query
      → 2 nhánh song song, mỗi nhánh: dense + sparse (song song) → RRF → MMR (tuỳ chọn)
          A: text = hypo,  emb = emb_hypo
          B: text = query, emb = emb_query (+ token cấu trúc khi có viện dẫn)
      → union theo chunk_id (+ extras nếu có viện dẫn) → fill metadata thiếu
      → Reranker (query gốc; passage = breadcrumb + "\n" + content)
      → top FINAL_TOP_K = 5
```

| Bước                                     | Vào → ra (mỗi nhánh nếu nêu)                   |
| ------------------------------------------ | ---------------------------------------------------- |
| Dense / Sparse                             | mỗi loại top`DENSE_TOP_N = SPARSE_TOP_N = 20`    |
| RRF (`RRF_K = 60`)                       | 20 + 20 → cắt`FUSION_TOP_N = 20`                 |
| MMR bật (`MMR_LAMBDA = 0.5`) / tắt     | 20 →`BRANCH_TOP_N = 10` (tắt: 10 đầu của RRF) |
| Union                                      | A + B, dedupe → ≤ 20                               |
| Extras (chỉ khi có viện dẫn, mục 8.1) | +`CITATION_SPARSE_TOP_K = 10` → ≤ 30             |
| Rerank                                     | ≤ 30 passage → top 5                               |

- Cả 2 nhánh dùng chung code (dense, sparse, RRF, MMR), chỉ khác text/embedding.
  Nhánh B là lưới an toàn khi HyDE paraphrase lệch, và giữ viện dẫn người dùng gõ.
- Union không tính lại điểm fusion; reranker quyết định thứ tự cuối.

**Số lượt gọi API mỗi câu:** Groq 1; HF 1 (2 text/batch); Pinecone dense query 2;
Pinecone sparse query 2 (extras dùng lại kết quả nhánh B, không thêm lượt); dense
fetch bổ sung metadata/vector (mục 6.4): MMR bật ≤ 2 (+1 nếu có extras thiếu id),
MMR tắt ≤ 1; reranker 1.

## 4. Pre-retrieval — HyDE

- Groq SDK, tái dùng `LLMSettings` (`openai/gpt-oss-120b`), hằng số nội bộ
  `retrieval/hyde.py`. Sinh 1 đoạn văn giả định theo văn phong luật.
- **Luật bắt buộc:** không nêu số Điều/Khoản/Điểm/Chương/Mục, tên/số hiệu văn
  bản, năm (sparse nhánh A search trên breadcrumb; viện dẫn bịa sẽ kéo chunk sai).
- Prompt nguyên văn (chốt 2026-09-20, theo lý thuyết, chưa đo bằng RAGAS):

<!-- tests/test_retrieval_clients.py đọc khối prompt bên dưới để so khớp với `retrieval/hyde.py`; đừng đổi cấu trúc khối (fence, nhãn [system]/[user]) mà không sửa test. -->

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

- Tách system/user để quy tắc cố định nằm ở system, câu hỏi nằm riêng ở user.
- **Tham số Groq** (đã xác nhận với SDK `groq` 1.7.0 và API thật):
  `reasoning_effort="low"`, `temperature=0.2`, `max_completion_tokens=2048`. Tham
  số nào API không hỗ trợ thì bỏ và ghi lại trong `hyde.py`.
- `hypo` chỉ dùng cho **nhánh A** (embed + sparse). MMR và rerank dùng câu gốc.
- `hypo` rỗng = Groq lỗi (mục 10). Retry/timeout theo `LLMSettings`.

## 5. Embed query (HF Inference API)

- 1 request `[hypo, query]` → `emb_hypo`, `emb_query`; `huggingface_hub.InferenceClient`,
  tái dùng `EmbeddingSettings` cho `model_name`/`hf_token`. Retry (2 lần) và timeout
  (30s) là **hằng số nội bộ** `query_embedder.py`, không nằm trong
  `EmbeddingSettings`. Code trong `retrieval/query_embedder.py` (không tái dùng
  `embedding/hf_client.py`).
- **Bắt buộc** word-segment `pyvi.ViTokenizer.tokenize()` như lúc index (chỉ
  embed `content`); thiếu bước này dense lệch không gian mà không báo lỗi.
- `emb_query` dùng cho dense nhánh B và `sim(d, query)` trong MMR cả 2 nhánh.

## 6. Hybrid search & RRF (trong từng nhánh)

### 6.1. Dense

Query dense index sẵn có (`VectorDBSettings.index_name`, cosine), top 20 bằng
embedding của nhánh; `include_metadata=True`, `include_values=True` chỉ khi MMR
bật. `retrieval/` không tạo dense index.

### 6.2. Sparse — Pinecone sparse index riêng + BM25 tự viết

**BM25 tự viết** (`retrieval/bm25.py`, thuần Python), không dùng `pinecone-text`
(phụ thuộc `mmh3` không có wheel Python ≥ 3.14; mặc định loại stopword/stem tiếng
Anh làm mất từ tiếng Việt như "do", tải NLTK qua mạng).

- **Text**: `breadcrumb + " " + content`. **Tokenize**: `pyvi`, lowercase, giữ `_`,
  bỏ dấu câu đứng riêng, không stemming/stopword; cùng 1 hàm cho fit, document,
  query.
- **`fit`**: 1 lần trên toàn corpus mỗi lần build (df, N, avgdl, `vocab` sắp xếp
  cố định).
- **`encode_document`**: `w = tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))`,
  `k1 = 1.2`, `b = 0.75`. **`encode_query(text, extra_terms=())`**: mỗi term trong
  vocab `w = ln((N − df + 0.5)/(df + 0.5) + 1)`, term ngoài vocab bị bỏ;
  dot product = điểm BM25.
- **Token cấu trúc** (vì "Điều 3" tách thành "điều", "3" có idf ~0 nên BM25 không
  phân biệt được Điều): thêm token hiếm vào chunk và query — `điều_N`, `khoản_M`,
  `điều_N_khoản_M`, và theo văn bản `vb_X`, `vb_X_điều_N`, `vb_X_điều_N_khoản_M`
  (`X` = key trong bảng `DOCUMENTS`, `citation.py`).
  - **Phía chunk**: Điều/Khoản parse từ breadcrumb (`parse_breadcrumb`, chịu được
    "Điểm" và "(phần i/n)"); `X` tra từ `source_document`. Văn bản chưa có trong
    `DOCUMENTS` → không sinh `vb_*`, log warning khi build.
  - **Phía query**: chỉ từ **câu hỏi gốc**, chỉ **sparse nhánh B**, khi có viện
    dẫn (mục 8.1). Nhánh A không có `extra_terms`. "Khoản M" không kèm số Điều
    không sinh token. Khoản tối đa `MAX_CITATION_KHOANS = 3`.
  - **`detect_document(query)`**: chuẩn hoá NFC + lowercase + bỏ dấu; alias dài
    thắng alias ngắn khi chồng lấn; đúng 1 văn bản → key, 0 hoặc ≥ 2 văn bản hoặc
    "nghị định" trơn → `None` (không sinh `vb_*`). Alias chỉ gồm tên có tiền tố
    "luật/bộ luật/nghị định" hoặc viết tắt; tên chủ đề trơn không phải alias.
  - `DOCUMENTS` (6 văn bản): `blld` (Bộ luật Lao động), `bhxh`, `bhyt`, `tncn`,
    `nd_luong` (NĐ mức lương tối thiểu), `nd_dkld` (NĐ hướng dẫn điều kiện lao
    động). Thêm/đổi văn bản trong corpus phải cập nhật bảng này.
  - Bằng chứng mô phỏng offline (40 câu viện dẫn, chưa qua Pinecone): top1
    35% → 90%, top3 65% → 100% khi thêm token cấu trúc.
- **`bm25_params.json`**: `vocab`, `idf`/`df`+`N`, `avgdl`, `k1`, `b`,
  `params_version` (hiện tại **3**; 2 = có token Điều/Khoản; 1 = không có field).
  Load thấy thiếu/khác phiên bản, hoặc **không có file** → lỗi rõ, nêu nguyên văn
  lệnh `uv run python tools/sparse_index_documents.py`, không chạy với params cũ.
- **Sparse index**: field `sparse_index_name` (`VectorDBSettings`), serverless,
  `vector_type="sparse"`, `metric="dotproduct"`, cùng `cloud`/`region` với dense.
  Record: `id = chunk_id`, chỉ `sparse_values`, **không metadata** (lấy từ dense
  qua fetch, mục 6.4). Query top `SPARSE_TOP_N = 20`.
- **Rebuild**: `index.delete(delete_all=True)` rồi upsert; index mới chưa có
  namespace ném `NotFoundException` → bắt, coi như sạch (helper riêng trong
  `retrieval/`, không import hàm private của `embedding/`). Index rỗng vài giây
  khi rebuild là chấp nhận; quy ước không chạy `retrieve()` lúc rebuild (nếu có,
  degrade như mục 10).

### 6.3. RRF

`rrf_score = Σ 1/(60 + rank)` cộng qua dense + sparse của nhánh (chunk chỉ có ở
1 danh sách vẫn được tính), sort giảm dần, cắt `FUSION_TOP_N = 20`.

### 6.4. Bổ sung vector/metadata cho candidate chỉ có ở sparse

Sparse không lưu vector/metadata; id thiếu được `dense_index.fetch(ids=[...])`
(bỏ qua nếu không thiếu):

- **MMR bật**: mỗi nhánh fetch 1 lần **trước** MMR (vector + metadata); nếu có
  extras còn thiếu id thì thêm 1 lần chỉ metadata → tối đa 3.
- **MMR tắt**: 1 lần cho toàn union (gồm extras), chỉ metadata → tối đa 1.
- Chunk fetch không trả về (lệch build) bị bỏ, log warning.

## 7. MMR (bật/tắt)

- `USE_MMR = True` mặc định; `retrieve(query, *, use_mmr: bool | None = None)`
  ghi đè từng lần gọi (để RAGAS so sánh có/không MMR mà không sửa code).
- Chỉ khác bước chọn candidate mỗi nhánh: bật = `mmr.select(fused, emb_query, 0.5, 10)` trên **dense vector** (`MMR = argmax λ·sim(d, query) − (1−λ)·max sim(d, d')`, `sim(d, query)` dùng `emb_query` cả 2 nhánh); tắt = `fused[:10]`.
- Rủi ro cần đo: luật có nhiều Khoản liền kề rất gần nhau, MMR có thể phạt oan;
  MMR dựa cosine nên không có nghĩa với số Điều (mục 16).

## 8. Union & Extras

`union = dedupe_by_chunk_id(nhánh_a + nhánh_b)`, giữ thứ hạng trong nhánh của
mỗi chunk (phục vụ fallback, mục 9); sau đó thêm extras (nếu có) rồi
`fill_missing` metadata (mục 6.4).

### 8.1. Câu hỏi viện dẫn (`retrieval/citation.py`)

Vấn đề: chunk đáp án thường cao ở sparse nhưng thấp ở dense; RRF + cắt + MMR làm
nó rớt khỏi union.

- **Nhận diện**: `extract_citation_numbers(query) -> list[int]` — số Điều theo
  thứ tự, dedupe, không trần (không có `has_citation`; dùng kết quả rỗng/không
  rỗng). Regex trên **câu gốc**,
  NFC, không phân biệt hoa thường. Nguyên tắc: **ưu tiên không false positive**.
  - Nhận: "Điều 36", "điều36", "Điều thứ 5", "Điều 36.2" (lấy 36), danh sách
    nối bằng `, ; và hoặc hay` ("Điều 3, 5 và 7"), khoảng chỉ lấy hai đầu mút
    ("Điều 3 đến Điều 5" → [3, 5]), "Điều 36 người lao động được quyền gì" → [36].
  - Không nhận: số Điều ngay trước đơn vị (`tháng|ngày|năm|tuổi|lần|%|đồng|triệu`,
    áp cho mọi số, "người" không thuộc danh sách này), "Đ.3"/"Đ3", La Mã, chữ
    ("Điều ba"), "điều kiện/khoản/hành", "điều 2024", "khoản N" đứng một mình.
  - `extract_citation_khoans(query)`: `khoản\s*(\d{1,3})`, dedupe, ≤ 3, chỉ dùng
    khi đã có số Điều.
- **Extras** (`citation_extras`): khi `extract_citation_numbers` ≠ [] (mọi n ≥ 1), extras = đúng
  `CITATION_SPARSE_TOP_K = 10` hit đầu của danh sách sparse thô nhánh B (điểm là
  BM25, không phải `rrf_score`), luôn thêm vào union dù RRF/MMR đã cắt. Dùng lại
  kết quả sparse có sẵn — **không thêm lượt Pinecone**. Không dùng hypo.
- Câu không viện dẫn: pipeline không đổi. Câu nhiều Điều: dùng cùng cơ chế,
  best-effort (chấp nhận recall thấp từng Điều).
- Hạn chế chấp nhận: một Điều có > 10 chunk vẫn bị cắt ở 10.
- Bằng chứng (40 câu tự sinh, chỉ nhánh B, chưa qua Pinecone/rerank): recall
  chunk đáp án trong union k=0 → k=10: 25% → 85% (MMR bật), 65% → 85% (MMR tắt).

## 9. Reranker (server tự host)

Model `AITeamVN/Vietnamese_Reranker` (cross-encoder 0.6B, base
`bge-reranker-v2-m3`, query ≤ 256 token, passage ≤ 2048 token) qua **LitServe trên
LightningAI** (`reranker_server/`, dự án `uv` độc lập).

**Client** (`retrieval/reranker_client.py`, `httpx`):

- `POST {endpoint_url}`, header `X-API-Key`, payload `{"query": câu_gốc, "passages": [breadcrumb + "\n" + content, ...]}` → `{"scores": [float, ...]}`
  cùng thứ tự. **1 request cho cả union.** Query là câu gốc, không phải hypo.
- Passage kèm breadcrumb vì cross-encoder cần thấy số Điều/Khoản và tiêu đề Điều;
  chỉ áp cho input reranker (`RetrievedChunk.content` không đổi; embedding vẫn
  chỉ `content`). Thí nghiệm 7 câu: hạng đáp án không tệ đi, có câu 8 → 2 và
  6 → 1 (cần xác nhận lại bằng RAGAS).
- Sort giảm dần theo score, cắt 5, gán `rerank_score`. **Không ghim/can thiệp thứ
  tự** — chỉ tin thứ tự rerank.
- **Timeout**: connect `connect_timeout_seconds = 5`; read = `max(timeout_seconds (30), RERANK_SECONDS_PER_PASSAGE (2.5) × n_passages)` (hằng số nội bộ; union
  ≤ 30 → 75s). Đo CPU ~1s/passage; hạ hằng số khi có GPU.
- **Retry** (tối đa `max_retries`): chỉ `ConnectError`, `ConnectTimeout`, HTTP
  502/503/504; lỗi ngoài httpx (thường là lỗi lập trình) không retry, fallback ngay
  kèm traceback. **Không retry** `ReadTimeout`/`WriteTimeout` (server có thể còn
  đang tính, retry làm phình hàng đợi) → fallback ngay + warning nêu số passage
  và timeout. 401/403/400/422 không retry, log error, fallback. 404 (ngrok offline/Studio
  sleep) không retry, fallback, log error nhắc runbook 9.1. Lỗi kết nối/5xx
  hết retry → warning nhắc runbook 9.1. Log không lộ nội dung passage/secret.
- **Validate**: `scores` là list số hữu hạn, đúng độ dài `passages`; sai = lỗi.
  Không có passage → không gọi request.
- **Fallback**: không crash, log warning, trả top 5 với `rerank_score=None`, xen
  kẽ round-robin A1, B1, E1, A2, B2, E2… (E = extras) bỏ trùng; nhánh A vắng thì
  bỏ A.

### 9.1. Runbook khởi động reranker server

Studio free tier tự sleep sau 10 phút không hoạt động và khi sleep cả 2 tiến
trình chết; bỏ qua runbook thì mọi câu rơi vào fallback.

1. **Người dùng** bật Studio `production-legal-qa-rag` trên web (Turn on, chờ
   Running).
2. **Agent** SSH bằng lệnh trong `LIGHTNING_STUDIO_SSH` (`.env` gốc; chỉ đọc biến
   này).
3. Chạy 2 tiến trình nền bằng `tmux` (kiểm tra session sẵn có, không tạo trùng):
   - `tmux new -d -s reranker 'cd reranker_server && uv run server.py'` (cổng 8000,
     `LIT_SERVER_API_KEY` trong `reranker_server/.env` phải trùng
     `RERANKER_API_KEY` ở `.env` gốc).
   - `tmux new -d -s ngrok_tunnel 'ngrok http 8000 --log stdout'` (domain tĩnh
     free = `RERANKER_ENDPOINT_URL`).
4. Kiểm tra từ local bằng `retrieval/test.py`; có `scores` = sẵn sàng (model cần
   thời gian load, có thể phải thử lại).

Lưu ý: tmux không chống sleep; `on_start.sh` chưa xác minh (mục 16).

## 10. Xử lý lỗi (degrade đúng 2 điểm)

| Lỗi (sau hết retry)                             | Xử lý                                                         |
| ------------------------------------------------- | --------------------------------------------------------------- |
| Groq lỗi/timeout/trả rỗng                      | Bỏ nhánh A; embed chỉ câu gốc; chạy nhánh B; log warning |
| Reranker lỗi                                     | Fallback mục 9                                                 |
| HF embed hoặc Pinecone (dense/sparse/fetch) lỗi | Raise`RetrievalError` kèm nguyên nhân                      |

Retry: Groq theo `LLMSettings`, reranker theo `RerankerSettings`; embed query dùng
hằng số nội bộ (2 retry, timeout 30s, mục 5).

## 11. Tools & Integrations

| Việc                      | Công cụ                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| HyDE                       | `groq` SDK, `LLMSettings`                                                               |
| Embed query                | `huggingface_hub.InferenceClient`, `EmbeddingSettings`                                  |
| Word-segment               | `pyvi` (embed query và BM25)                                                             |
| BM25                       | Tự viết`retrieval/bm25.py`, không dependency mới                                      |
| Dense/sparse search, fetch | `pinecone` SDK (>= 8, đang 10.0.0; cú pháp tạo sparse index xác nhận khi implement) |
| Song song                  | `asyncio` (`retrieve()` là `async def`)                                              |
| RRF, MMR                   | Thuần Python                                                                               |
| Rerank                     | `httpx` → server LightningAI                                                             |
| Models                     | `pydantic` v2                                                                             |

## 12. Config

- `VectorDBSettings.sparse_index_name: str = Field(validation_alias="PINECONE_SPARSE_INDEX_NAME")`.
- `RerankerSettings` (cùng pattern `LLMSettings`): `endpoint_url`
  (`RERANKER_ENDPOINT_URL`), `api_key` (`RERANKER_API_KEY`), `max_retries = 2`,
  `connect_timeout_seconds = 5`, `timeout_seconds = 30` (sàn của read timeout).
- `LIGHTNING_STUDIO_SSH` chỉ là ghi chú vận hành, không thuộc Settings.
- **Hằng số nội bộ `retrieval/`, không vào `config.py`**: `DENSE_TOP_N = 20`,
  `SPARSE_TOP_N = 20`, `RRF_K = 60`, `FUSION_TOP_N = 20`, `BRANCH_TOP_N = 10`,
  `MMR_LAMBDA = 0.5`, `USE_MMR = True`, `FINAL_TOP_K = 5`,
  `CITATION_SPARSE_TOP_K = 10`, `MAX_CITATION_KHOANS = 3`,
  `RERANK_SECONDS_PER_PASSAGE = 2.5`; BM25 `k1 = 1.2`, `b = 0.75`.
- Module trong `retrieval/` không đọc `.env` trực tiếp.

## 13. Workflow & state

### 13A. Offline — build sparse index

```bash
uv run python tools/sparse_index_documents.py   # --chunks-dir data/chunks --params-out data/bm25/bm25_params.json
```

Typer CLI mỏng (`add_completion=False`, exit 0/1) gọi
`retrieval.sparse_index.build_index`: đọc chunk → text + token cấu trúc/`vb_*` →
`bm25.fit` → lưu `bm25_params.json` (kèm `params_version`) → encode từng chunk →
tạo index nếu chưa có → xoá sạch → upsert batch (id, sparse_values) → in tổng số
chunk. Chỉ cần `.env` có `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` và
`PINECONE_SPARSE_INDEX_NAME` (`VectorDBSettings` đòi cả ba). Chạy lại được,
build lại toàn bộ. Bắt buộc chạy lại sau mỗi lần build lại dense index và khi
`params_version` đổi (việc của người dùng).

### 13B. Online — `retrieve(query, *, use_mmr=None)`

```
1. hypo = generate(query)                          # lỗi -> chỉ nhánh B
2. emb_hypo, emb_query = embed([hypo, query])
   numbers = extract_citation_numbers(query); khoans = ... if numbers
   doc = detect_document(query) if numbers; terms = structural_terms(numbers, khoans, doc)
3. branch_a, branch_b = gather(run_branch(hypo, emb_hypo, []), run_branch(query, emb_query, terms))
   run_branch: gather(dense top 20, sparse top 20) -> RRF[:20]
               -> MMR bật: fetch thiếu, mmr.select(..., 10) | tắt: [:10]
4. union = dedupe(a + b); if numbers: += citation_extras(sparse thô nhánh B)   # ≤ 30
   union = fill_missing_metadata(union)
5. scores = rerank(query, [breadcrumb + "\n" + content ...])   # lỗi -> fallback mục 9
6. return top 5 theo scores, list[RetrievedChunk]
```

Stateless giữa các lần gọi (không cache/session). Client và `bm25_params.json`
khởi tạo 1 lần, dùng lại cho mọi query.

## 14. Module (`src/production_legal_qa_rag/retrieval/`)

| Module                    | Trách nhiệm                                                                                                                                                                                                                |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `models.py`             | `RetrievedChunk`, model candidate trung gian                                                                                                                                                                               |
| `hyde.py`               | Groq sinh hypo (mục 4)                                                                                                                                                                                                      |
| `query_embedder.py`     | Embed nhiều chuỗi 1 request,`pyvi` segment                                                                                                                                                                               |
| `bm25.py`               | Tokenize,`fit`, `encode_document`, `encode_query`, lưu/đọc params                                                                                                                                                   |
| `sparse_index.py`       | Tạo/xoá/upsert/query sparse index                                                                                                                                                                                          |
| `dense_search.py`       | Query dense + fetch bổ sung                                                                                                                                                                                                 |
| `fusion.py`, `mmr.py` | RRF, MMR                                                                                                                                                                                                                     |
| `citation.py`           | `extract_citation_numbers/khoans`, `citation_extras`, `parse_breadcrumb`, `DOCUMENTS`, `detect_document`, `structural_terms`/`breadcrumb_structural_terms` (chung 1 hàm định dạng token) |
| `reranker_client.py`    | HTTP client, timeout/retry/validate/fallback                                                                                                                                                                                 |
| `pipeline.py`           | `retrieve()` điều phối, sở hữu client                                                                                                                                                                                 |
| `relevance.py`          | `MIN_RERANK_SCORE`, `is_low_relevance(chunks)` — tín hiệu độ liên quan dựa trên `rerank_score`; **không** được `retrieve()` gọi, chỉ export cho `conversation/orchestrator.py` tự quyết định `no_context` (mục 16 điểm 8, `conversation_spec.md` mục 18.2.2) |

`tools/sparse_index_documents.py` tách biệt khỏi `retrieve()`. `retrieval/test.py`
là script thử tay gọi reranker/pipeline thật (không thuộc kiến trúc chính thức).

## 15. Tiêu chí hoàn thành

**Nghiệm thu thủ công (quan trọng nhất)** — chạy thật (`retrieval/test.py`, đã
rebuild sparse index `params_version = 3`, reranker server đang chạy) trên bộ câu
viện dẫn **một Khoản** mẫu (tối thiểu: "Khoản 1 Điều 113 Bộ luật Lao động nói
gì?", "Điều 3 khoản 1 của luật thuế TNCN quy định gì?", "Điều 36 khoản 2 Bộ luật
Lao động"; có và không nêu tên văn bản): với mỗi câu, chunk đáp án nằm trong
top 5. Ghi lại hạng/điểm từng câu để theo dõi (không phải điều kiện đạt).

**Kiểm thử tự động**

- Build: `tools/sparse_index_documents.py` chạy trên toàn `data/chunks/*.json`
  không crash, sinh `bm25_params.json` và sparse index đúng số vector = số chunk;
  chạy lại lần 2 và trên index mới (chưa có namespace) đều không lỗi.
- `bm25.py`: dot product khớp công thức tay; term ngoài vocab bị bỏ; "do" không
  bị loại; "Điều 3 khoản 1" xếp chunk đúng breadcrumb lên đầu trên corpus mẫu;
  `params_version` thiếu/khác → lỗi có lệnh rebuild; token cấu trúc/`vb_*` phía
  chunk và query cùng hàm định dạng, không va chạm token pyvi.
- `citation.py`: mỗi dạng ở 8.1 có ca dương và ca âm (kể cả "Điều 3 và 5 tháng"
  → [3], "Điều 36 người lao động được quyền gì" → [36], "điều 2024" → [], "Điều
  1, 2, 3 và 4" → [1, 2, 3, 4]); `extract_citation_khoans` trần 3;
  `detect_document` (alias dài thắng, không dấu vẫn khớp, mơ hồ/≥ 2 văn bản/
  "nghị định" trơn → `None`); `source_document` ngoài `DOCUMENTS` → không `vb_*`
  + warning.
- Extras: n = 1 và n = 3 đều đúng 2 lượt sparse, extras = 10 hit đầu nhánh B,
  union ≤ 30; câu không viện dẫn: union không có extras, không đổi so với hiện
  trạng.
- `retrieve(q, use_mmr=True/False)` trả ≤ 5 chunk đủ metadata, số lượt fetch đúng
  mục 6.4; chạy end-to-end với ≥ 3 câu thật khác nhau không crash; số lượt Groq 1,
  HF 1, rerank 1.
- Embed query qua `pyvi`: câu trùng nguyên văn `content` 1 chunk trả chunk đó ở
  top dense.
- HyDE (fake Groq): prompt khớp từng ký tự với spec mục 4; câu hỏi vào message
  `user`; đúng 3 tham số Groq; output rỗng → lỗi/`None`.
- Reranker (`httpx.MockTransport`): read timeout = `max(30, 2.5 × n)`; `ReadTimeout`
  không retry (1 lần gọi) + fallback + log; `ConnectError`/`ConnectTimeout`/
  502/503/504 retry tới `max_retries`; 401/403/400/422 không retry; passage bắt đầu
  bằng `breadcrumb` + `"\n"` + `content`, `RetrievedChunk.content` không kèm
  breadcrumb; fallback xen kẽ có E.
- Giả lập Groq lỗi, reranker lỗi (timeout, 401) → đúng mục 10, `retrieve()` không
  crash.
- `config.py` có `sparse_index_name`/`RerankerSettings`.
- Ngoài phạm vi (cả Điều, Điểm trở lên, nhiều Điều): chỉ yêu cầu chạy hợp lệ
  (≤ 5 chunk, không crash).

## 16. Rủi ro / điểm mở

1. Xác minh tmux/`on_start.sh` có giúp Studio khỏi sleep không (mục 9.1).
2. MMR bật hay tắt làm mặc định production (mô phỏng: MMR bật làm recall câu viện
   dẫn thấp hơn hẳn khi không có extras) — quyết định sau RAGAS; các hằng số
   khởi điểm (mục 12), đặc biệt `CITATION_SPARSE_TOP_K` (mô phỏng cho thấy top3
   đã 100% sau token cấu trúc, có thể giảm để bớt passage/latency) tinh chỉnh
   cùng lúc.
3. Đo lại sau rebuild qua Pinecone thật: recall extras, chunk đáp án của 1 Điều
   có đứng đầu sparse không; ~15% recall còn lỡ ở k=10 chưa điều tra nguyên nhân.
4. Phase RAGAS: hiệu quả HyDE nhánh A (và bỏ HyDE khi câu chỉ thuần viện dẫn),
   passage kèm breadcrumb, temperature/độ dài hypo.
5. Bàn sau: tối ưu latency/API call (benchmark batch HF/Groq/Lightning, khởi động
   sớm nhánh B, circuit breaker reranker, quota HF); hạ `RERANK_SECONDS_PER_PASSAGE`
   khi có GPU.
6. Phương án Agentic cho câu cả Điều/Điểm trở lên/nhiều Điều (spec riêng).
7. **Mở rộng truy vấn theo từ đồng nghĩa pháp lý — hoãn (2026-09-22):** ca follow-up đổi
   chủ thể giới tính ("Vậy chồng thì sao?" sau câu hỏi về "nghỉ thai sản") cho thấy câu
   hỏi đúng thuật ngữ nhưng ít từ khoá trùng corpus có thể vẫn trượt retrieval. Đã quyết
   định **không** sửa `retrieval/` cho ca này (nguyên nhân gốc nằm ở condense sinh sai
   thuật ngữ, không phải retrieval — xem `conversation/conversation_spec.md` mục 17.1.1,
   17.2.2); chỉ ghi lại làm phương án dự phòng nếu sau khi sửa condense mà vẫn trượt
   (`conversation_spec.md` mục 17.5.1).
8. **`relevance.py` (mới, 2026-09-22, `conversation_spec.md` mục 18.2.2) — gate độ liên
   quan bằng `rerank_score`:** module này **không đổi hợp đồng `retrieve()`** (vẫn luôn
   trả top `FINAL_TOP_K` theo rerank, không lọc gì — giữ nguyên mục 1 "không có nhánh xử
   lý riêng, không nới `FINAL_TOP_K`"); quyết định "coi 5 chunk là không đủ liên quan →
   từ chối" là chính sách của `conversation/orchestrator.py`, không phải của `retrieve()`.
   `rerank_score` là **logit thô** của `AITeamVN/Vietnamese_Reranker` (không qua sigmoid,
   xem `reranker_server/server.py`), chưa có ngưỡng nào được hiệu chỉnh trong dự án trước
   đây — `MIN_RERANK_SCORE` đo lần đầu ở 18.2.2 trên rất ít ca (~4-5), rủi ro chưa đủ dữ
   liệu để tin cậy cao, có thể cần hiệu chỉnh lại ở phase RAGAS hoặc khi có nhãn
   `expected_chunks` đã duyệt (mục 15.5 `conversation_spec.md`).
