# Embedding — Chunk → Vector Store: Reference Spec

- Giữ nguyên số mục để không làm hỏng tham chiếu từ code/spec khác.
- Spec liên quan: [chunking_spec.md](../chunking/chunking_spec.md),
  [retrieval_spec.md](../retrieval/retrieval_spec.md).

## 1. Mục đích

- Chuyển `Chunk` thành vector tìm kiếm được, giữ citation và dữ liệu generation.
- **Checkpoint snapshot trước, publish sau:** không trộn embed và publish trong một vòng lặp mạng.
- Làm: đọc JSON, preprocess, embed batch, checkpoint atomic, validate, publish/tạo index, quota/retry.
- Không làm: chunk/retrieve/rerank/generate, phân quyền, DB trạng thái, delta embedding, multi-tenant, nhiều
  key song song.
- Mặc định: full rebuild; vector store chỉ nhận snapshot hoàn chỉnh.

## 2. Bất biến hệ thống

- Index/query cùng model và preprocessing; đổi model phải rebuild.
- `data/chunks/` chỉ đọc; checkpoint ở vùng output riêng.
- Publish chỉ đọc checkpoint thuộc đúng build; không trộn checkpoint cũ.
- Batch lỗi: log, giữ checkpoint thành công; không tự publish snapshot partial.
- ID deterministic từ chunking; metadata đủ citation, không cần mở corpus lại.
- Validate tại biên: checkpoint JSON, response embedding, record vector DB.

## 3. Contract dữ liệu

- Input: `data/chunks/**/*.json`, mảng `Chunk` gồm `chunk_id`, `content`, `breadcrumb`, `source_document`,
  `has_table`, `raw_table`.
- `EmbeddedChunk`: `Chunk` + `embedding: list[float]`; ghi file tạm rồi rename; một file nguồn ↔ một
  checkpoint.
- Manifest mỗi build: nguồn↔checkpoint, tổng chunk/thành công/lỗi, model, dimension, version; chỉ trạng thái
  `complete` được publish.
- Record: `id=chunk_id`, `values=embedding`; metadata `content` nguyên văn, `breadcrumb`, `source_document`,
  `has_table`.
- Chỉ thêm `raw_table` khi có bảng; Pinecone không nhận metadata `null`.
- Contract Pydantic v2: `EmbeddedChunk`, `PineconeMetadata`, `PineconeRecord`; không trao đổi `dict` thô.

## 4. Tiền xử lý và model

- Model PhoBERT hiện tại dùng `pyvi.ViTokenizer.tokenize(Chunk.content)` ngay trước API; index/query cùng quy
  tắc.
- Checkpoint/metadata giữ nguyên văn, không lưu bản word-segment thay `content`.
- `EmbeddingSettings`: nguồn chung `model_name`, provider token và token budget.
- Dimension lấy từ config/response đã validate; không hard-code hoặc trộn vector khác model.

## 5. Gọi embedding API

- Batch tuần tự, kích thước bảo thủ là hằng nội bộ; giữ thứ tự để ghép đúng `chunk_id`.
- Timeout hữu hạn; retry giới hạn cho lỗi tạm thời; bộ đếm request dùng chung dừng trước ngưỡng quota.
- Validate số vector, giá trị số thực và dimension.
- Hết retry: log ID lỗi; không tạo vector rỗng, đổi thứ tự hoặc âm thầm publish.

## 6. Publish vector store

- Manifest complete → đọc/validate checkpoint → tạo index cosine nếu thiếu → chờ ready → upsert batch → kiểm
  số vector khớp snapshot.
- Full replace offline: `delete_all` rồi upsert toàn snapshot.
- Replace không atomic: lỗi sau delete có thể để index trống/partial; không coi là publish thành công.

## 7. Workflow

- Chunk JSON chỉ đọc → preprocess/embed → validate → checkpoint atomic → manifest complete → publish.
- CLI tách `embed` (checkpoint + manifest) và `upsert` (manifest complete được chỉ định).
- CLI Typer ở `tools/`; logic ở package; batch tuần tự.

## 8. Module boundaries

- `models.py`: contract.
- `hf_client.py`: preprocess, batch, retry, quota, validate response.
- `pinecone_client.py`: index lifecycle, publish batch.
- `pipeline.py`: điều phối; `tools/embed_documents.py`: Typer mỏng; config tập trung `config.py`.

## 9. Tiêu chí hoàn thành

- Input nguyên vẹn; vector đúng thứ tự/dimension/preprocessing query-time.
- Manifest một-một nguồn/checkpoint, không lẫn checkpoint cũ; snapshot lỗi không tự publish.
- Record có ID deterministic, metadata không `null`; full rebuild đủ vector, không orphan.

## 10. Áp dụng cho hệ thống mới

- Baseline: sequential full rebuild + checkpoint + manifest.
- Chỉ thêm delta/resume/concurrency khi có số đo chứng minh nhu cầu.
