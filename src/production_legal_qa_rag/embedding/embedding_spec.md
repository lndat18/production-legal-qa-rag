# Embedding — Chunk → Vector → Pinecone

## 1. Mục tiêu & phạm vi

Sinh vector embedding cho từng `Chunk` (đầu ra của `chunking/`, xem
`chunking_spec.md` mục 2) bằng HuggingFace Inference API (model
`CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2`), rồi upsert lên Pinecone để
phục vụ bước retrieval sau này.

**Phạm vi**: đọc toàn bộ `data/chunks/*.json` hiện có (2026-09-17: 6 file,
**2.228 chunk** — 579 + 328 + 121 + 60 + 704 + 436) mỗi lần chạy, xoá sạch
index Pinecone rồi upsert lại toàn bộ từ đầu (xem mục 5, 8).

Pipeline chia làm **2 pha tách biệt** (xác nhận với người dùng 2026-09-17):
pha 1 gọi HF Inference API cho toàn bộ chunk và ghi kết quả ra file trung
gian (mục 2), pha 2 mới đọc file trung gian rồi upsert lên Pinecone. Lý do
tách: lệnh gọi HF là tài nguyên **bị giới hạn quota** (1.000 request/ngày),
trong khi Pinecone không bị giới hạn kiểu đó — nếu gộp chung 1 pha mà bước
upsert lỗi giữa chừng (sai config, mất mạng...), kết quả embed đã tốn quota
gọi HF sẽ mất, phải gọi lại từ đầu. Ghi checkpoint ra đĩa giữa 2 pha tránh
rủi ro này.

**Ngoài phạm vi (chủ động không làm)**:

- Retrieval/query logic — thuộc spec khác.
- Incremental/delta embedding (chỉ diff chunk mới/thay đổi) — đã xác nhận
  với người dùng (2026-09-17): chấp nhận full re-embed mỗi lần, đơn giản
  hơn, và với batch request (mục 4) tổng số request vẫn nằm rất xa dưới
  ngân sách 1.000 request/ngày nên không cấp thiết phải tối ưu.
- Idempotent tracking qua Postgres/DB khác — nhất quán quyết định đã có ở
  `formatting_spec.md` mục 1.2.
- Multi-namespace/multi-tenant Pinecone — 1 index, không namespace riêng.
- Resume pha 1 giữa các lần chạy khác nhau (vd. skip chunk đã embed từ lần
  chạy trước nếu bị crash giữa chừng) — đã xác nhận với người dùng
  (2026-09-17): không cần, mỗi lần chạy pha 1 luôn embed lại toàn bộ từ đầu
  và ghi đè file trung gian cũ (mục 2). File trung gian chỉ là checkpoint
  nội bộ giữa pha 1 và pha 2 của **cùng 1 lần chạy**, không dùng để skip gì
  giữa các lần chạy khác nhau.
- Dispatch nhiều HF token chạy đồng thời (như Groq 2-key ở
  `formatting_spec.md` mục 1.3) — chỉ 1 `HF_TOKEN`, xử lý batch tuần tự (mục
  4). Quy mô ~90 request/lần chạy (ước tính, mục 4) không cần tối ưu song
  song; nếu corpus tăng quy mô lớn, đây là chỗ đầu tiên cần xét lại.
- Tự host embedding model (vd. `sentence-transformers` chạy local) — dùng
  HuggingFace Inference API cloud-hosted theo yêu cầu ban đầu người dùng.

## 2. Input & Output

- **Input**: `data/chunks/*.json` (mỗi file 1 mảng JSON các `Chunk`, schema
  đầy đủ xem `chunking_spec.md` mục 2). Field dùng ở bước này: `chunk_id`,
  `content` (chuỗi đem đi embed), `breadcrumb`, `source_document`,
  `has_table`, `raw_table`. **Không ghi/sửa vào các file này** — `embedding/`
  chỉ đọc, output của pha 1 đi vào file trung gian riêng (không mutate
  output của `chunking/`, giữ đúng ranh giới sở hữu giữa 2 package, và tránh
  bị `chunking/` ghi đè mất khi chạy lại — `data/chunks/*.json` luôn được
  ghi atomic toàn bộ file mỗi lần `chunking/` chạy).

- **Output trung gian (pha 1 — embed)**: `data/embeddings/*.json`, ánh xạ
  1-1 theo tên file nguồn từ `data/chunks/` (giữ nguyên tên). Mỗi file là 1
  mảng JSON các `EmbeddedChunk` (`embedding/models.py`) — toàn bộ field của
  `Chunk` cộng thêm `embedding: list[float]`. Chỉ chứa chunk **embed thành
  công**; chunk thuộc batch lỗi (mục 4) không xuất hiện trong file này. Ghi
  atomic (file tạm + rename, giống `chunking/`/`formatting/`) ngay sau khi
  xử lý xong file nguồn tương ứng — không đợi hết toàn bộ corpus mới ghi.

- **Output cuối (pha 2 — upsert)**: Pinecone index (tên lấy từ
  `VectorDBSettings.index_name`, mục 7), mỗi `EmbeddedChunk` đọc từ
  `data/embeddings/*.json` → 1 vector:

  | Field Pinecone | Giá trị |
  | --- | --- |
  | `id` | `chunk_id` (deterministic sẵn từ chunking, đảm bảo upsert ghi đè đúng vị trí nếu chạy lại) |
  | `values` | `EmbeddedChunk.embedding` (dimension xác định qua model config, mục 5 — không hardcode) |
  | `metadata.content` | `EmbeddedChunk.content` |
  | `metadata.breadcrumb` | `EmbeddedChunk.breadcrumb` |
  | `metadata.source_document` | `EmbeddedChunk.source_document` |
  | `metadata.has_table` | `EmbeddedChunk.has_table` |
  | `metadata.raw_table` | `EmbeddedChunk.raw_table` — **chỉ set key này khi `has_table=True`**; Pinecone metadata không nhận giá trị `null`, nên khi `has_table=False` bỏ hẳn key thay vì gán `None` |

  Không lưu `token_count`/`is_split`/`split_index`/`split_total` vào
  metadata — không phục vụ retrieval/generation, tránh phình metadata không
  cần thiết (đã xác nhận với người dùng 2026-09-17). Không lưu
  `standardization_table` riêng vì đã nằm trong `content` (chunking_spec mục
  5.4).

## 3. Segment tiếng Việt trước khi embed

Model PhoBERT-based được huấn luyện trên văn bản đã word-segment — giống lý
do `chunking/tokenizer.py` phải segment trước khi đếm token (xem module đó,
mục 6 `chunking_spec.md`). Trước khi gửi `content` cho HF Inference API,
segment bằng `pyvi.ViTokenizer.tokenize()`.

Gọi trực tiếp `pyvi` tại đây (1 dòng), **không** tái dùng
`chunking/tokenizer.py` — module đó gắn với việc đếm token bằng
`AutoTokenizer`, không phải mối quan tâm của `embedding/`; tái dùng chỉ vì 1
lệnh gọi `pyvi` chung sẽ tạo coupling không cần thiết giữa 2 package.

## 4. HuggingFace Inference API — batch & rate limiting

**Free tier: 1.000 request/ngày** (theo người dùng cung cấp). `HF_TOKEN` bắt
buộc phải có để đạt được quota này (anonymous thấp hơn nhiều).

**[Giả định cần xác thực thực nghiệm khi implement — chưa test tại thời
điểm viết spec, 2026-09-17]**: 3 điều sau phải được xác nhận bằng cách gọi
thử API thật (vài batch nhỏ) trước khi chốt hằng số cuối cùng, cùng tinh
thần với cách `formatting_spec.md` mục 1.2 đã đo token corpus thật trước khi
chốt `CHUNK_TOKEN_LIMIT`:

1. Endpoint có nhận **list nhiều text trong 1 request** không (`inputs:
   [text1, text2, ...]` → trả về list vector cùng thứ tự) — đây là điều
   kiện tiên quyết để batch giảm số request.
2. Giới hạn payload/batch size tối đa trước khi bị lỗi (số text và/hoặc
   tổng ký tự mỗi request).
3. Quota RPD thực tế khi có `HF_TOKEN` — đúng 1.000/ngày như đã biết chưa,
   có giới hạn theo phút (RPM) riêng không.

**Chiến lược (giá trị khởi điểm, điều chỉnh theo kết quả xác thực ở
trên)**:

- `HF_BATCH_SIZE = 25` (hằng số nội bộ module, không thuộc `config.py` —
  xem mục 7) — 2.228 chunk / 25 ≈ **90 request** cho 1 lần chạy full corpus,
  dư rất nhiều so với ngân sách 1.000/ngày kể cả khi phải retry.
- Không cần rate limiter theo phút (TPM/RPM) như Groq — quy mô ~90
  request/lần chạy khó chạm giới hạn theo phút dù có tồn tại. Chỉ cần đếm
  **tổng request đã gọi trong ngày** (in-process, không cần persist qua các
  lần chạy khác nhau vì mỗi lần chạy đã nằm rất xa ngưỡng): dừng lại, raise
  lỗi rõ ràng nếu vượt `HF_RPD_SAFE_LIMIT` (90% × 1.000 = 900) thay vì cố
  gọi tiếp rồi nhận lỗi 429 giữa chừng. Bộ đếm này **dùng chung xuyên suốt
  toàn bộ pha 1** (mục 8), không reset theo từng file nguồn — giống sliding
  window của `formatting/` (mục 1.2 `formatting_spec.md`, "dùng chung cho
  toàn bộ `convert_directory()`, không reset theo từng file").
- Retry theo `max_retries`/`timeout_seconds` (cùng tinh thần `LLMSettings`,
  mục 7) khi request lỗi.
- **Lỗi 1 batch (hết `max_retries`) → bỏ qua toàn bộ chunk thuộc batch đó**
  (không upsert), log QC warning liệt kê `chunk_id` bị bỏ, không chặn các
  batch còn lại — nhất quán pattern lỗi của `formatting/` (1 phần lỗi không
  làm hỏng toàn bộ lần chạy).

## 5. Pinecone — tạo index & upsert (pha 2)

Toàn bộ mục này chỉ chạy **sau khi pha 1 (mục 4) đã xử lý xong hết mọi file
nguồn** và ghi đủ `data/embeddings/*.json` (mục 2) — không upsert xen kẽ
theo từng file/batch trong lúc pha 1 đang chạy.

- **Dimension không cần gọi Inference API để biết**: đọc
  `AutoConfig.from_pretrained(model_name).hidden_size` (tải config.json từ
  HF Hub, không tính vào quota Inference API — khác hẳn việc gọi embedding
  thật ở mục 4). Dùng giá trị này khi tạo index, không hardcode số dimension
  trong `config.py` (tránh sai lệch nếu đổi model sau).
- **Tạo index nếu chưa tồn tại** (kiểm tra qua `list_indexes()`): serverless
  spec, `cloud="aws"`, `region="us-east-1"` (mặc định free tier serverless),
  `metric="cosine"` (chuẩn cho sentence embedding).
- **Trước khi upsert: xoá sạch toàn bộ vector hiện có trong index**
  (`index.delete(delete_all=True)`) — tránh orphan vector khi `chunk_id` cũ
  không còn tồn tại ở lần chunking sau (đã xác nhận với người dùng
  2026-09-17, nhất quán quyết định "full re-embed mỗi lần" ở mục 1).
- **Upsert theo batch** (khuyến nghị Pinecone: ~100 vector/lần gọi, hằng số
  `PINECONE_UPSERT_BATCH_SIZE` — độc lập với `HF_BATCH_SIZE` ở mục 4, 2 bước
  batch riêng không cần đồng bộ kích thước).

## 6. Tools & Integrations

- `huggingface_hub` (`InferenceClient`) — gọi Inference API.
- `pinecone` (SDK chính thức) — tạo/xoá/upsert index, serverless.
- `pyvi` — word-segment (đã là dependency có sẵn từ `chunking/`).
- `transformers` (`AutoConfig`) — chỉ đọc `hidden_size`, không load full
  model (đã có `transformers` dependency từ `chunking/tokenizer.py`).

## 7. Config tập trung (`src/production_legal_qa_rag/config.py`)

- `EmbeddingSettings` thêm field mới:
  `hf_token: str = Field(validation_alias="HF_TOKEN")` — **bắt buộc**
  (không có default), theo đề xuất ban đầu của người dùng để đạt quota
  1.000 request/ngày.
- `VectorDBSettings` thêm 2 field mới cho việc tạo index serverless (mục
  5): `cloud: str = "aws"`, `region: str = "us-east-1"`.
- **Không** đưa `HF_BATCH_SIZE`, `HF_RPD_SAFE_LIMIT`,
  `PINECONE_UPSERT_BATCH_SIZE` vào `config.py` — đây là hằng số nội bộ cơ
  chế batch/rate-limit của riêng `embedding/`, giống cách
  `formatting/llm_client.py` giữ `TPM_SAFE_LIMIT`/`RPM_SAFE_LIMIT` nội bộ
  thay vì đặt trong `config.py` (tinh thần chunking_spec.md mục 7: chỉ khai
  vào `config.py` field dùng chung nhiều package).

## 8. Workflow & quản lý trạng thái

**Pha 1 — embed (mục 4)**, lặp tuần tự theo từng file trong
`data/chunks/*.json` (giống vòng lặp theo file của `chunking/`/
`formatting/`), bộ đếm rate limiter (`HF_RPD_SAFE_LIMIT`) dùng chung xuyên
suốt cả pha, không reset giữa các file:

1. Đọc 1 file `data/chunks/X.json`, parse thành `list[Chunk]` — tái dùng
   `Chunk` từ `production_legal_qa_rag.chunking.models` (không định nghĩa
   lại schema).
2. Segment `content` từng chunk bằng `pyvi` (mục 3), chia thành batch theo
   `HF_BATCH_SIZE` (mục 4).
3. Gọi HF Inference API tuần tự từng batch, thu vector tương ứng theo đúng
   thứ tự input. Batch lỗi (hết `max_retries`) → bỏ qua các chunk thuộc
   batch đó, log QC warning.
4. Build `EmbeddedChunk` (mục 2) cho các chunk embed thành công của file
   này.
5. Ghi `data/embeddings/X.json` atomic (file tạm + rename) — ghi đè hoàn
   toàn nếu file đã tồn tại từ lần chạy trước (không resume, mục 1).

Sau khi **toàn bộ file nguồn đã qua pha 1** mới chuyển sang pha 2 — không
upsert xen kẽ trong lúc pha 1 đang chạy (mục 1, lý do tách 2 pha).

**Pha 2 — upsert (mục 5)**:

6. Đọc toàn bộ `data/embeddings/*.json`, gộp thành 1 danh sách
   `EmbeddedChunk` duy nhất.
7. Build Pinecone record (`id`, `values`, `metadata` — mục 2) từ mỗi
   `EmbeddedChunk`.
8. Xoá sạch index hiện có (mục 5), sau đó upsert toàn bộ record theo batch
   (`PINECONE_UPSERT_BATCH_SIZE`).
9. In tổng kết: tổng số chunk (từ `data/chunks/`), số chunk embed thành
   công (từ `data/embeddings/`), số chunk bị bỏ qua kèm danh sách QC
   warning (batch/chunk_id lỗi ở pha 1).

Không có DB/trạng thái nào được lưu giữa các lần chạy khác nhau — mỗi lần
chạy xử lý lại toàn bộ input hiện có, cả pha 1 lẫn pha 2 (mục 1).
`data/embeddings/*.json` chỉ là checkpoint nội bộ giữa 2 pha của **cùng 1
lần chạy**. Lỗi 1 batch không chặn các batch/file còn lại (partial
success), nhất quán pattern hiện có ở `formatting/`/`chunking/`.

## 9. Cấu trúc module trong `src/production_legal_qa_rag/embedding/`

- `models.py` — `EmbeddedChunk` (`Chunk` + field `embedding: list[float]`,
  mục 2) và Pydantic model cho Pinecone record (`id`, `values`, `metadata`)
  dùng để validate trước khi upsert (coding-convention: cấu trúc dữ liệu
  trao đổi giữa các bước dùng Pydantic, không dict thô).
- `hf_client.py` — wrapper gọi HF Inference API: segment, chia batch, rate
  limiter theo request/ngày, retry.
- `pinecone_client.py` — tạo index nếu chưa có, xoá toàn bộ vector, upsert
  theo batch.
- `pipeline.py` — điều phối toàn bộ workflow (mục 8, 2 pha): `embed()` (pha
  1: đọc `data/chunks/` → ghi `data/embeddings/`) và `upsert()` (pha 2: đọc
  `data/embeddings/` → Pinecone), gọi tuần tự từ entrypoint CLI (`tools/`,
  Typer theo coding-convention).
- `__init__.py`

## 10. Tiêu chí hoàn thành

- Chạy pipeline trên toàn bộ `data/chunks/*.json` hiện có (2.228 chunk)
  không crash; log rõ số chunk thành công/bị bỏ qua.
- Sau pha 1: `data/embeddings/*.json` tồn tại đủ 1-1 theo tên file nguồn
  `data/chunks/*.json`, mỗi `EmbeddedChunk` có `embedding` đúng dimension
  model thật; `data/chunks/*.json` không bị thay đổi.
- Pinecone index sau khi chạy chứa đúng số vector = số chunk embed thành
  công, mỗi vector có đủ metadata theo mục 2 (trừ `raw_table` khi
  `has_table=False`).
- Vector dimension khớp đúng với model thật (đọc từ `AutoConfig`, không
  hardcode sai).
- 1 lần chạy full corpus không vượt quota HF Inference API 1.000
  request/ngày (ước tính ~90 request với `HF_BATCH_SIZE=25`, mục 4).
