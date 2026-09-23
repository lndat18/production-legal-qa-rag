# Embedding — Chunk → Vector Store: Reference Spec

## 1. Mục đích

Chuyển các `Chunk` đã được chuẩn hoá thành vector có thể tìm kiếm, nhưng vẫn
giữ được citation và dữ liệu cần để sinh câu trả lời sau này.

Nguyên tắc cốt lõi:

> **Embed là một snapshot bất biến; vector store chỉ công bố snapshot hoàn
> chỉnh.**

Embedding API là tài nguyên tốn tiền/quota và vector store là trạng thái có
thể thay đổi. Vì vậy phải checkpoint kết quả embedding trước, rồi mới publish
lên vector store. Không trộn hai trách nhiệm vào một vòng lặp mạng duy nhất.

### Trong phạm vi

- Đọc `Chunk` JSON, tiền xử lý đúng theo embedding model, gọi model theo batch.
- Ghi checkpoint `EmbeddedChunk` atomic, validate và publish lên vector store.
- Tạo index, metadata retrieval và kiểm soát quota/retry.

### Ngoài phạm vi

- Chunking, retrieval/reranking, generation, quyền truy cập người dùng.
- Database trạng thái, delta embedding, multi-tenant hay song song nhiều key,
  trừ khi quy mô thực tế chứng minh cần thiết.

Với corpus nhỏ hoặc vừa, full rebuild là mặc định đơn giản nhất. Chỉ làm
incremental khi chi phí đo được của full rebuild đáng kể.

## 2. Bất biến hệ thống

1. **Cùng model, cùng tiền xử lý.** Text index và text query phải qua chính
   xác cùng segmentation/normalization trước khi embed. Lệch preprocessing là
   lỗi semantic: hệ thống vẫn chạy nhưng vector không còn cùng không gian.
2. **Không sửa output của bước trước.** `data/chunks/` chỉ đọc; checkpoint
   embedding nằm ở vùng output riêng.
3. **Snapshot phải tự nhất quán.** Một lần publish chỉ đọc danh sách checkpoint
   thuộc đúng lần build đó; không quét mù các file cũ trong output directory.
4. **Không publish index partial mặc định.** Batch/file embed lỗi được log và
   checkpoint phần thành công được giữ để điều tra, nhưng pha publish dừng nếu
   snapshot chưa complete. Partial publish chỉ là lựa chọn vận hành rõ ràng,
   không phải hành vi ngầm định.
5. **ID và metadata ổn định.** Vector dùng `chunk_id` deterministic của
   chunking; metadata phải đủ để truy hồi, lọc và trích dẫn mà không cần mở lại
   file corpus.
6. **Validate ở biên.** JSON checkpoint, response model và record vector DB
   đều được validate trước khi chuyển sang bước tiếp theo.

## 3. Contract dữ liệu

### Input

`data/chunks/**/*.json`, mỗi file là mảng Pydantic `Chunk`. Các field đầu vào
cần giữ nguyên gồm `chunk_id`, `content`, `breadcrumb`, `source_document`,
`has_table` và `raw_table`.

### Checkpoint

`EmbeddedChunk` kế thừa đầy đủ `Chunk` và thêm:

```python
embedding: list[float]
```

Checkpoint được ghi atomic (temporary sibling rồi rename), một file nguồn ứng
với một file checkpoint. Mỗi build phải có một **manifest snapshot** chứa:

- danh sách file nguồn và checkpoint tương ứng;
- tổng chunk, tổng embed thành công, danh sách chunk/batch lỗi;
- model name, dimension và version/timestamp build.

Pha publish chỉ nhận manifest `complete`. Manifest giải quyết hai lỗi vận hành
hay gặp: dùng lại checkpoint của source đã bị xoá, và vô tình publish một lần
embed đang dở dang.

### Record vector store

| Field | Giá trị |
| --- | --- |
| `id` | `chunk_id` |
| `values` | `embedding` |
| `metadata.content` | `content` gốc, không phải bản đã word-segment |
| `metadata.breadcrumb` | Citation đầy đủ |
| `metadata.source_document` | Định danh văn bản |
| `metadata.has_table` | Cờ bảng |
| `metadata.raw_table` | Chỉ có khi `has_table=True` |

Không lưu metadata chỉ hữu ích lúc ingest, như token count hoặc quan hệ split,
trừ khi một use case retrieval chứng minh cần chúng. Với Pinecone, bỏ hẳn key
`raw_table` khi không có bảng thay vì gửi `null`.

`EmbeddedChunk`, `PineconeMetadata` và `PineconeRecord` là Pydantic v2 models;
không trao đổi `dict` thô giữa các module.

## 4. Tiền xử lý và model

Model multilingual/PhoBERT thường yêu cầu word segmentation tiếng Việt. Với
profile hiện tại, dùng `pyvi.ViTokenizer.tokenize()` cho `Chunk.content` ngay
trước khi gọi embedding API. Checkpoint và metadata vẫn giữ content nguyên văn.

Code embedding index và query có thể là hai module độc lập, nhưng phải cùng
quy tắc tiền xử lý. Không tái sử dụng module tokenizer của chunking chỉ vì có
một hàm giống nhau nếu điều đó làm hai package phụ thuộc sai chiều.

`EmbeddingSettings` là nguồn chung của:

- `model_name`;
- secret/token của provider;
- token budget nếu chunking dùng cùng model.

Dimension index lấy từ model config hoặc response đã xác thực; không hard-code.
Đổi model là đổi không gian vector: phải xác nhận dimension và rebuild index,
không được trộn vector hai model trong cùng index.

## 5. Gọi embedding API

Batch để giảm số request, nhưng giữ thứ tự input để ghép vector trở lại đúng
`chunk_id`.

Trước khi chốt hằng số production, đo với provider thật:

1. API có nhận một list text và trả vector cùng thứ tự không.
2. Giới hạn batch/payload, timeout và quota theo ngày/phút.
3. Kiểu response, dimension và lỗi rate-limit thực tế.

Sau đó áp dụng:

- batch size bảo thủ, là hằng số nội bộ `embedding/`;
- timeout hữu hạn và retry giới hạn cho lỗi tạm thời;
- request counter dùng chung cho toàn bộ build, dừng trước ngưỡng quota an toàn;
- validate số vector, từng vector số thực, và dimension nhất quán trong batch.

Khi một batch hết retry, log `chunk_id` cụ thể. Không tự biến lỗi đó thành vector
rỗng, không đổi thứ tự các chunk còn lại, và không âm thầm publish snapshot.

## 6. Publish vector store

Pha này chỉ bắt đầu khi manifest complete.

1. Đọc checkpoint được manifest liệt kê và validate lại.
2. Tạo index nếu chưa có: metric phù hợp model (thường cosine), cloud/region từ
   settings; chờ index ready trước data-plane request.
3. Chuyển từng `EmbeddedChunk` thành record đã validate, rồi upsert theo batch
   riêng với batch embedding.
4. Xác nhận số vector đã publish khớp snapshot trước khi đánh dấu build thành
   công.

### Chính sách thay thế index

Với ingestion offline và một corpus nhỏ, có thể `delete_all` rồi upsert full
snapshot: cách này loại orphan vector khi chunking đổi. Trade-off là index có
khoảng trống/partial nếu lỗi xảy ra sau delete.

Nếu index đang phục vụ traffic, build snapshot vào namespace/index phiên bản
mới, validate count, rồi chuyển consumer sang version mới. Không bổ sung cơ chế
này sớm khi chưa có yêu cầu availability; nhưng cũng không gọi quy trình
`delete_all → upsert` là atomic.

## 7. Workflow

```text
Chunk JSON (read-only)
  → preprocess + embed theo batch
  → validate vector
  → atomic EmbeddedChunk checkpoints
  → complete snapshot manifest
  → validate toàn snapshot
  → full refresh hoặc publish version mới
  → vector store
```

Tách CLI thành hai thao tác rõ ràng:

- `embed`: tạo checkpoint và manifest.
- `upsert`: chỉ publish một manifest complete được chỉ định.

CLI ở `tools/` dùng Typer, còn business logic nằm trong package. Batch chạy
tuần tự mặc định; tăng concurrency chỉ sau khi quota và giới hạn provider đã
được đo.

## 8. Module boundaries

| Module | Trách nhiệm duy nhất |
| --- | --- |
| `models.py` | Pydantic `EmbeddedChunk`, metadata, vector record và manifest. |
| `hf_client.py` | Preprocess, batching, retry, quota, validate response API. |
| `pinecone_client.py` | Lifecycle index và publish record theo batch. |
| `pipeline.py` | Điều phối snapshot: input → checkpoint → manifest → publish. |
| `tools/embed_documents.py` | Typer entrypoint mỏng. |

Config môi trường tập trung ở `config.py`; không đọc `.env` trực tiếp từ client
hay CLI. Constants chỉ thuộc một cơ chế (batch size, retry) ở module đó, không
biến mọi chi tiết thành environment variable.

## 9. Tiêu chí hoàn thành

- Input chunk không bị sửa; checkpoint atomic và truy vết được về đúng source.
- Mọi vector trong snapshot có cùng dimension, đúng thứ tự với `chunk_id` và
  được tạo bằng preprocessing giống query-time.
- Manifest complete có mapping 1-1 source/checkpoint, count chính xác, không
  chứa checkpoint cũ ngoài snapshot.
- Record có ID deterministic và metadata citation đầy đủ; không gửi metadata
  `null` provider không hỗ trợ.
- Quota, retry và batch lỗi có log rõ; snapshot lỗi không tự publish.
- Index mới tồn tại đủ số vector của snapshot và không chứa orphan vector sau
  full rebuild.

## 10. Áp dụng cho hệ thống mới

Chốt trước khi code:

1. Model, dimension, preprocessing bắt buộc và metric vector.
2. Nguồn dữ liệu bất biến cùng schema checkpoint/metadata tối thiểu.
3. Quota, batch/payload, retry và mức lỗi nào cho phép publish.
4. Chiến lược snapshot: full refresh offline hay versioned publish không
   downtime.
5. Ngưỡng thực tế để cần delta embedding, resume hoặc concurrency.

Nếu chưa có dữ liệu đo, chọn sequential full rebuild + checkpoint + manifest.
Đó là baseline nhỏ nhất vẫn bảo vệ được chi phí API, tính đúng semantic và sự
toàn vẹn của index.
