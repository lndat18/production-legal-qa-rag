# Embedding — Chunk → Vector Store: Reference Spec

> Giữ nguyên số mục vì code/spec khác tham chiếu.

## 1. Mục đích

Chuyển `Chunk` đã chuẩn hoá thành vector tìm kiếm được, vẫn giữ citation và dữ liệu cần để sinh câu trả lời.

> **Embed là một snapshot bất biến; vector store chỉ công bố snapshot hoàn chỉnh.** Checkpoint kết quả embed TRƯỚC, rồi mới publish; không trộn hai trách nhiệm trong một vòng lặp mạng.

**Trong phạm vi:** đọc `Chunk` JSON, tiền xử lý đúng theo model, embed theo batch, ghi checkpoint atomic, validate, publish, tạo index, kiểm soát quota/retry. **Ngoài phạm vi:** chunking, retrieval/rerank, generation, quyền truy cập, DB trạng thái, delta embedding, multi-tenant, song song nhiều key. **Full rebuild là mặc định.**

## 2. Bất biến hệ thống

1. **Cùng model, cùng tiền xử lý** giữa index và query (lệch preprocessing làm vector khác không gian mà không báo lỗi).
2. `data/chunks/` chỉ đọc; checkpoint ở vùng output riêng.
3. **Snapshot tự nhất quán:** publish chỉ đọc checkpoint thuộc đúng build đó.
4. **Không publish index partial mặc định:** batch lỗi được log, checkpoint thành công được giữ, publish dừng nếu snapshot chưa complete.
5. ID (`chunk_id` deterministic của chunking) và metadata đủ để trích dẫn mà không mở lại corpus.
6. Validate ở biên: JSON checkpoint, response model, record vector DB.

## 3. Contract dữ liệu

- **Input:** `data/chunks/**/*.json`, mảng `Chunk` (`chunk_id`, `content`, `breadcrumb`, `source_document`, `has_table`, `raw_table`).
- **Checkpoint:** `EmbeddedChunk` = `Chunk` + `embedding: list[float]`; ghi atomic (file tạm rồi rename), 1 file nguồn ↔ 1 file checkpoint. Mỗi build có **manifest snapshot** (nguồn↔checkpoint, tổng chunk/thành công/lỗi, model, dimension, version); publish chỉ nhận manifest `complete`.
- **Record vector store:** `id=chunk_id`, `values=embedding`, metadata `content` (nguyên văn, không phải bản word-segment), `breadcrumb`, `source_document`, `has_table`, `raw_table` (chỉ khi có bảng; Pinecone bỏ hẳn key khi không có bảng, **không gửi `null`**).
- `EmbeddedChunk`, `PineconeMetadata`, `PineconeRecord` là Pydantic v2; không trao đổi `dict` thô giữa module.

## 4. Tiền xử lý và model

PhoBERT/multilingual cần word segmentation: `pyvi.ViTokenizer.tokenize()` cho `Chunk.content` ngay trước khi gọi API; checkpoint/metadata giữ nguyên văn. Index và query **cùng quy tắc**. `EmbeddingSettings` là nguồn chung của `model_name`, token provider, token budget; dimension lấy từ config/response đã xác thực, không hard-code. **Đổi model = rebuild index**, không trộn vector hai model.

## 5. Gọi embedding API

Batch giữ thứ tự input để ghép đúng `chunk_id`. Batch bảo thủ (hằng số nội bộ), timeout hữu hạn, retry giới hạn cho lỗi tạm thời, bộ đếm request chung dừng trước ngưỡng quota, validate số vector/số thực/dimension. Batch hết retry: log `chunk_id` cụ thể; **không** biến lỗi thành vector rỗng, không đổi thứ tự, không âm thầm publish.

## 6. Publish vector store

Chỉ khi manifest complete: đọc và validate lại checkpoint → tạo index nếu chưa có (metric cosine, chờ ready) → upsert theo batch → xác nhận số vector khớp snapshot. Chính sách thay thế (ingestion offline, corpus nhỏ): `delete_all` rồi upsert full snapshot; đây **không atomic** (index trống/partial nếu lỗi sau delete).

## 7. Workflow

Chunk JSON (read-only) → preprocess + embed batch → validate → checkpoint atomic → manifest complete → publish. CLI tách `embed` (checkpoint + manifest) và `upsert` (chỉ publish một manifest complete được chỉ định). Typer ở `tools/`, logic ở package; batch tuần tự.

## 8. Module boundaries

`models.py`, `hf_client.py` (preprocess, batching, retry, quota, validate response), `pinecone_client.py` (index lifecycle, publish theo batch), `pipeline.py` (điều phối), `tools/embed_documents.py` (Typer mỏng). Config tập trung ở `config.py`.

## 9. Tiêu chí hoàn thành

Input không bị sửa; mọi vector cùng dimension, đúng thứ tự, cùng preprocessing với query-time; manifest 1-1 source/checkpoint, không chứa checkpoint cũ; record có ID deterministic, không metadata `null`; snapshot lỗi không tự publish; index đủ số vector, không orphan sau full rebuild.

## 10. Áp dụng cho hệ thống mới

Baseline: **sequential full rebuild + checkpoint + manifest**; chỉ thêm delta/resume/concurrency khi có số đo chứng minh.
