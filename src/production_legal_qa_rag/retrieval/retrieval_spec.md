# Retrieval — Legal Question → Evidence Chunks: Reference Spec

## 1. Mục đích

Từ một câu hỏi, trả về một tập nhỏ `RetrievedChunk` đủ liên quan và có citation
để generation trả lời dựa trên bằng chứng.

Nguyên tắc cốt lõi:

> **Tìm theo nghĩa và theo định danh pháp lý, rồi để reranker chọn bằng chứng
> cuối cùng.**

Văn bản pháp luật có hai loại tín hiệu khác nhau:

- Ý định: “người lao động được nghỉ trong trường hợp nào?”
- Định danh: “Khoản 1 Điều 113 Bộ luật Lao động”.

Dense embedding mạnh về ý định nhưng không đáng tin cậy với số Điều/Khoản;
sparse lexical mạnh về định danh nhưng yếu với paraphrase. Retrieval phải giữ cả
hai, không cố ép một phương pháp giải toàn bộ bài toán.

### Phạm vi

- `retrieve(query)` trả tối đa `FINAL_TOP_K` evidence chunks.
- Hybrid dense/sparse, citation-aware recall, rerank và fallback có chủ đích.
- Build offline sparse index từ corpus chunk đã được embed.

### Ngoài phạm vi

- Chunking, embedding corpus, generation, hội thoại nhiều lượt và cache.
- Quyết định có trả lời người dùng hay từ chối; đó là policy của tầng
  conversation/API.
- Agentic decomposition cho câu nhiều Điều, cả Điều, Điểm hoặc nhiều ý định.

Chất lượng phải được tuyên bố theo một task cụ thể. Với legal QA, baseline tốt
là: câu hỏi viện dẫn một Khoản phải đưa chunk đáp án vào candidate cuối cùng.
Câu hỏi rộng hơn vẫn chạy cùng pipeline, nhưng không được hứa chất lượng chưa
đo.

## 2. Contract và ranh giới trách nhiệm

`retrieve(query: str) -> list[RetrievedChunk]` trả tối đa `FINAL_TOP_K`, xếp
theo `rerank_score` khi rerank thành công.

`RetrievedChunk` là Pydantic v2 contract:

| Field | Mục đích |
| --- | --- |
| `chunk_id` | Định danh deterministic từ chunking. |
| `source_document`, `breadcrumb` | Citation và provenance. |
| `content` | Bằng chứng nguyên văn gửi generation. |
| `has_table`, `raw_table` | Giữ dữ liệu bảng khi cần hiển thị/trả lời. |
| `rerank_score` | Điểm rerank, hoặc `None` khi dùng fallback. |

Retrieval luôn trả evidence tốt nhất nó tìm được. Nó không lọc thành rỗng chỉ
vì score thấp. Tầng sở hữu UX/policy có thể dùng `rerank_score` đã hiệu chỉnh để
quyết định `no_context`, xin làm rõ, hay tiếp tục generation. Nhờ vậy primitive
retrieval vẫn tái sử dụng được cho debug, research và các product policy khác.

## 3. Luồng online

```text
query gốc
  ├─ HyDE → hypothetical legal text (nhánh A, best-effort)
  └─ query gốc                         (nhánh B, luôn có)
       → embed từng text bằng cùng preprocessing index-time
       → dense + sparse song song trong mỗi nhánh
       → RRF → MMR tùy chọn → union theo chunk_id
       → citation extras từ sparse nhánh B (nếu có)
       → bổ sung metadata/vector thiếu
       → rerank bằng query gốc
       → top K RetrievedChunk
```

Nhánh A tăng recall semantic cho cách hỏi tự nhiên. Nhánh B là nguồn sự thật
cho literal wording, số Điều/Khoản và tên văn bản. Hai nhánh dùng cùng code
search/fusion; khác nhau duy nhất ở text, vector và structural terms.

HyDE chỉ sinh văn phong/ngữ nghĩa pháp lý, **không được bịa số Điều/Khoản, tên
văn bản, năm hay mức số cụ thể**. Hypo hỏng/rỗng thì bỏ nhánh A, không làm hỏng
nhánh B.

## 4. Dense, sparse và fusion

### Dense

Embed query bằng đúng model và preprocessing lúc index (ví dụ word-segment tiếng
Việt với `pyvi`). Lệch preprocessing là lỗi semantic im lặng, nên index-time và
query-time phải được xem là cùng một contract.

Dense query trả metadata; chỉ lấy vector khi MMR cần. `retrieval/` chỉ đọc dense
index, không tạo hay mutate nó.

### Sparse BM25

Sparse search dùng text:

```text
breadcrumb + " " + content
```

Breadcrumb đưa citation vào lexical index, còn content giữ thuật ngữ và quy
định. Dùng một tokenizer nhất quán cho fit, document và query; với tiếng Việt,
không mặc định áp stopword/stemming tiếng Anh vì có thể phá thuật ngữ pháp lý.

BM25 có thể tự viết khi dependency sẵn có không phù hợp. Khi làm vậy, công thức,
vocabulary và thứ tự ID phải deterministic; params được lưu versioned và runtime
từ chối dùng params khác version.

### RRF

Fusion theo Reciprocal Rank Fusion, không cộng trực tiếp dense score với BM25
score vì hai thang điểm không tương đương. Dedupe bằng `chunk_id`, giữ candidate
xuất hiện ở chỉ một nhánh, rồi cắt candidate pool về một giới hạn nhỏ trước
rerank.

## 5. Citation-aware retrieval

Đây là cơ chế riêng cho định danh pháp lý chính xác, không phải prompt trick.

### Parse thận trọng

Trích Điều/Khoản chỉ từ query gốc. Ưu tiên false negative hơn false positive:
số năm, tiền, tuổi, đơn vị thời gian, từ như “điều kiện” không được biến thành
Điều. Nếu tên văn bản mơ hồ hoặc có nhiều văn bản, không đoán document key.

### Structural terms

Sinh token hiếm, ổn định ở cả document và query, ví dụ:

```text
điều_113
khoản_1
điều_113_khoản_1
vb_blld_điều_113_khoản_1
```

Chúng bổ sung vào sparse vector, không thay thế text gốc hay dense embedding.
Chỉ nhánh B nhận token từ citation người dùng; HyDE không được sinh chúng.
Mapping `source_document` → document key phải explicit, versioned và cảnh báo
khi corpus có văn bản chưa được map.

### Citation extras

RRF/MMR có thể làm rơi chunk citation đúng dù nó đứng cao ở sparse. Khi query có
citation hợp lệ, thêm top sparse thô của nhánh B vào union trước rerank. Cơ chế
này dùng lại kết quả sparse đã có, nên không thêm round trip. Số extras phải bị
chặn để giữ latency và context budget hữu hạn.

## 6. Diversity và rerank

MMR có thể chọn evidence đa dạng hơn từ candidate pool dense, nhưng có thể phạt
oan các Khoản gần nhau vốn cùng cần thiết cho câu hỏi luật. Vì vậy MMR phải là
một cờ runtime/evaluation, không phải “tối ưu” được tin sẵn. Khi tắt, giữ đầu RRF
làm baseline so sánh.

Reranker là quyết định cuối:

- Query là câu hỏi gốc, không phải HyDE.
- Passage là `breadcrumb + "\n" + content`; reranker cần thấy citation, nhưng
  `content` công khai và text embedding không bị thay đổi.
- Gửi cả candidate pool trong một lượt gọi (có thể chia batch nội bộ, xem 6.1),
  validate số score hữu hạn và đúng thứ tự passage, sort giảm dần rồi cắt top K.
- Không ghim kết quả bằng rule sau rerank; nếu muốn bắt buộc citation recall,
  làm ở candidate stage qua extras, không bóp méo thứ tự cuối.

### 6.1 Inference tại chỗ (local, không host API riêng)

Model (`AITeamVN/Vietnamese_Reranker`, fine-tune từ `bge-reranker-v2-m3`) chạy
in-process trong `RetrievalPipeline`, không qua HTTP/microservice — bỏ hẳn
kiến trúc host tách rời trên LightningAI Studio/ngrok của bản cũ.

- **Device**: tự phát hiện `cuda` nếu có, fallback `cpu`. Trên GPU dùng fp16
  (`model.half()`); trên CPU giữ fp32 (fp16 không có lợi tốc độ trên CPU).
  Không giả định máy chạy luôn có GPU — CPU phải là đường chạy hợp lệ, không
  phải lỗi. Log `INFO` device đã chọn đúng một lần lúc load model, để vận
  hành/CLI test (mục `tools/retrieval.py`) thấy ngay đang chạy `cuda` hay
  fallback `cpu` mà không cần API riêng.
- **Vòng đời**: load tokenizer + model một lần lúc `RetrievalPipeline` khởi
  tạo, giữ suốt vòng đời process; không load lại mỗi request.
- **`MAX_LENGTH = 512`** (giảm mạnh so với 2304 = 256 query + 2048 passage của
  bản cũ). `MAX_LENGTH` là giới hạn cho **tổng** `token(query) + token(breadcrumb
  + "\n" + content) + 4 token đặc biệt` sau khi tokenizer nối cặp
  `[query, passage]` thành một chuỗi — không phải giới hạn riêng cho passage.
  512 là số ước lượng có margin, KHÔNG đo thực tế trên corpus (user chủ động
  chọn bỏ qua bước đo), tính từ:
  - `content` ≤ `MAX_TOKENS` = 236 (đếm bằng tokenizer PhoBERT sau `pyvi`
    segment — xem `chunking_spec.md`), nhân hệ số an toàn ×1.5 ≈ 400, vì
    tokenizer của `bge-reranker-v2-m3` là SentencePiece đa ngôn ngữ (không
    word-segment tiếng Việt sẵn như PhoBERT) nên cùng đoạn văn thường ra
    nhiều token hơn — KHÔNG được suy thẳng 236 sang model này.
  - `breadcrumb` ước lượng ~40 token, `query` chừa 64 token, cộng 4 token đặc
    biệt (`<s> query </s></s> passage </s>`) → tổng ~508, làm tròn lên bội 64.
  - Vì là ước lượng chứ không phải số đo, cần đo lại thực tế nếu về sau log
    cảnh báo truncation hoặc chất lượng rerank giảm bất thường — không phải
    một bước bắt buộc trước khi implement.
- **Batch**: candidate pool trước dedupe có thể lên tới
  `2 × (DENSE_TOP_N + SPARSE_TOP_N)` = 80 passage (2 nhánh, mỗi nhánh dense +
  sparse top 20). Chia batch cố định (`RERANK_BATCH_SIZE`, đề xuất 16) khi gọi
  model để chặn đỉnh VRAM không phụ thuộc số lượng candidate thực tế, thay vì
  luôn forward nguyên pool trong một lần.
- **Không chặn event loop**: forward pass là blocking call (CPU/GPU-bound),
  phải chạy trong `asyncio.to_thread` (hoặc executor tương đương) vì phần còn
  lại của pipeline là async.
- **Concurrency inference process-wide = 1**: `asyncio.to_thread` mỗi request
  chạy trên thread riêng, nhưng dùng chung một `self._model`. Nhiều request
  đồng thời → nhiều forward pass cùng lúc trên cùng model → VRAM cộng dồn
  theo số request đồng thời, rất dễ CUDA OOM trên card 2GB dùng chung 1 model.
  Bọc toàn bộ một lần gọi `rerank()` (gồm mọi batch của request đó) trong
  `asyncio.Semaphore(1)` process-wide: request khác phải đợi (coroutine tạm
  dừng, không chặn event loop, không tốn thread chờ) tới khi request đang
  chạy xong. Nhờ vậy đỉnh VRAM của riêng reranker bị chặn ở đúng 1 batch
  (`RERANK_BATCH_SIZE`) tại một thời điểm, bất kể có bao nhiêu request đồng
  thời.
  - **Cứng `1`, không đưa vào `RerankerSettings`**: đúng với ràng buộc phần
    cứng hiện có (GPU 2GB, 1 model dùng chung); không thêm biến cấu hình chưa
    ai cần. Đổi GPU lớn hơn sau này thì sửa hằng số, không phải việc thường
    xuyên.
  - **Áp dụng đồng nhất cho cả CUDA lẫn CPU**, không phân biệt theo device —
    giữ đơn giản, không thêm nhánh logic chỉ để tối ưu throughput CPU. Đổi lại
    rerank trên CPU cũng bị serialize khi nhiều request tới cùng lúc; chấp
    nhận được vì quy mô ứng dụng tự giới hạn (`deploy_spec.md` mục 5) và đã có
    admission/quota ở tầng trên (`conversation_spec.md` mục 8).
  - **Loop-bound**: `asyncio.Semaphore` (Python ≥3.10) bám vào event loop đang
    chạy lúc `await` đầu tiên, dùng lại ở loop khác sẽ `RuntimeError` — cùng
    vấn đề `httpx.AsyncClient`/`AsyncGroq` đã gặp và được giải bằng
    `LoopBoundClient` (`retrieval/loop_bound.py`). `LocalReranker` là singleton
    sống qua nhiều `asyncio.run()` (`_default_pipeline`, test suite), nên dùng
    lại `LoopBoundClient` bọc factory `lambda: asyncio.Semaphore(1)` thay vì
    tạo semaphore thô — không viết abstraction mới.

## 7. Consistency và state offline

Dense index, sparse index và BM25 params phải mô tả **cùng một corpus snapshot**.
Mỗi build sparse ghi manifest/version gồm ít nhất:

- corpus fingerprint hoặc version của manifest embedding;
- số chunk, danh sách/ID nguồn hoặc checksum tương đương;
- tokenizer/BM25 params version và mapping document version;
- dense build/version mà sparse build dựa vào.

Runtime chỉ load sparse params và index có version tương thích với dense snapshot;
không “best effort” khi phát hiện lệch. Candidate có ở sparse nhưng không fetch
được metadata từ dense cũng bị bỏ và log như tín hiệu build lệch, không phải kết
quả hợp lệ.

Build sparse là offline, full refresh cho corpus nhỏ. Khi phục vụ traffic, dùng
snapshot versioned/namespace mới rồi switch consumer sau khi count đã khớp;
`delete_all → upsert` không phải transaction.

## 8. Xử lý lỗi

Phân biệt lỗi có thể degrade và lỗi phá tính đúng đắn:

| Sự cố | Hành vi |
| --- | --- |
| HyDE lỗi/rỗng | Bỏ nhánh A, chạy nhánh B. |
| Reranker lỗi (runtime/model, ví dụ CUDA OOM) | Fallback deterministic từ các nhánh, `rerank_score=None`. |
| Query embed, dense/sparse search hoặc metadata fetch lỗi | Raise `RetrievalError`; không giả vờ có evidence đáng tin. |
| Corpus version mismatch | Từ chối khởi tạo/query và nêu lệnh hay thao tác rebuild. |

Retry chỉ dành cho lỗi tạm thời của dịch vụ ngoài (HF embedding, Pinecone),
timeout hữu hạn. Reranker chạy in-process (mục 6.1): lỗi runtime/model với
cùng input là deterministic, retry vô nghĩa — fallback ngay, log đủ để debug
(kích thước batch, độ dài passage lúc lỗi).

## 9. Module boundaries

| Module | Trách nhiệm duy nhất |
| --- | --- |
| `models.py` | Pydantic public/intermediate contracts và `RetrievalError`. |
| `hyde.py` | Sinh hypothetical legal text, best-effort. |
| `query_embedder.py` | Query preprocessing, API batch và validate embedding. |
| `bm25.py` | Tokenize, fit/load params, encode sparse vector. |
| `citation.py` | Parse citation, mapping document, structural terms, extras. |
| `dense_search.py`, `sparse_index.py` | Client từng index và lifecycle offline của sparse. |
| `fusion.py`, `mmr.py` | Thuật toán thuần, không I/O. |
| `reranker.py` | Load model in-process 1 lần, batch inference (CUDA/CPU) giới hạn 1 request cùng lúc process-wide, validate, fallback khi lỗi. |
| `pipeline.py` | Điều phối `retrieve()` và sở hữu client. |
| `relevance.py` | Chỉ cung cấp relevance signal cho tầng policy, không lọc retrieval. |
| `tools/retrieval.py` | Typer entrypoint mỏng, chỉ gọi `RetrievalPipeline.retrieve()`. |

Mọi contract trao đổi giữa module là Pydantic. Config dùng `pydantic-settings`
tập trung; constants chỉ thuộc một cơ chế để trong module đó. CLI offline đặt ở
`tools/` dùng Typer và không chứa business logic.

CLI thủ công `tools/retrieval.py` dùng để chạy thử `retrieve(query)` trên
corpus/index thật (cần `PINECONE_API_KEY`, `HF_TOKEN`, index Pinecone đã nạp
dữ liệu), chủ yếu để kiểm chứng thay đổi reranker (mục 6.1) — device đang
dùng thật là `cuda` hay fallback `cpu`, không silent lỗi. `reranker.py` log
`INFO` device (`cuda`/`cpu`) một lần lúc load model, nên chạy CLI ở log level
mặc định là thấy ngay không cần API riêng chỉ để phục vụ test. CLI in ra:

- Tổng thời gian `retrieve()` (wall-clock).
- Từng `RetrievedChunk`: `breadcrumb`, `content` (rút gọn), `rerank_score`,
  `has_table`.
- Cảnh báo rõ ràng nếu `rerank_score=None` (fallback đã kích hoạt — mục 8)
  thay vì lặng lẽ in kết quả như đang chạy bình thường.

Có bộ câu hỏi mẫu preset (câu viện dẫn cụ thể + câu paraphrase, cùng quy ước
với `tools/conversation.py`), chọn qua option; `--query` nhập câu tùy ý,
`--use-mmr/--no-use-mmr` ghi đè `USE_MMR` mặc định.

## 10. Tiêu chí hoàn thành

- Câu query literal, paraphrase và citation đều đi qua cùng public contract.
- Citation hợp lệ có đường recall độc lập với HyDE/RRF/MMR và không phát sinh
  round trip sparse mới.
- Dense index/query dùng cùng preprocessing; sparse fit/document/query dùng cùng
  tokenizer và structural-term convention.
- Output tối đa K chunk, có citation/metadata đầy đủ, không chứa candidate từ
  snapshot lệch.
- Rerank lỗi vẫn trả fallback có thứ tự xác định; lỗi nền tảng search không bị
  che giấu.
- `MAX_LENGTH = 512` (mục 6.1) là số ước lượng có margin, không phải số đo;
  nếu sau này log cảnh báo truncation hoặc chất lượng rerank giảm bất thường,
  đo lại thực tế bằng tokenizer `bge-reranker-v2-m3` trên corpus rồi cập nhật.
- Reranker chỉ chạy đúng 1 inference cùng lúc process-wide
  (`asyncio.Semaphore(1)` qua `LoopBoundClient`, mục 6.1); N request đồng thời
  không làm VRAM cộng dồn theo N.
- Tầng conversation có thể áp policy score mà không làm đổi hoặc làm mơ hồ
  contract của `retrieve()`.

## 11. Áp dụng cho bài toán mới

Chốt trước khi code:

1. Nhiệm vụ retrieval đo được và loại query cần bảo đảm recall.
2. Tín hiệu semantic, literal và cấu trúc đặc thù domain; chỉ thêm structural
   terms khi chúng bù được điểm mù có bằng chứng.
3. Contract preprocessing chung giữa index/query, cùng corpus snapshot/version.
4. Candidate budget, tiêu chí dùng MMR và reranker, latency budget thực tế.
5. Ranh giới degrade, fail-fast và policy sản phẩm.

Baseline nhỏ nhất đáng tin là: query gốc + dense/sparse song song + RRF + rerank
bằng query gốc. Chỉ thêm HyDE, MMR hoặc citation extras khi chúng giải một failure
mode đã quan sát, và luôn giữ baseline để so sánh.
