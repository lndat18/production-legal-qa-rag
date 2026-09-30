# Embedding — Chunk → Vector Store: Reference Spec

> Cô đọng 2026-09-30 (giữ số mục vì code/spec khác tham chiếu). Bản đầy đủ: git history.

## 1. Mục đích

Chuyển `Chunk` đã chuẩn hoá thành vector tìm kiếm được, vẫn giữ citation và dữ liệu cần để sinh câu trả lời.

> **Embed là một snapshot bất biến; vector store chỉ công bố snapshot hoàn chỉnh.**

Embedding API tốn tiền/quota, vector store là trạng thái đổi được → checkpoint kết quả embed TRƯỚC, rồi mới
publish; không trộn hai trách nhiệm trong một vòng lặp mạng.

**Trong phạm vi:** đọc `Chunk` JSON, tiền xử lý đúng theo model, embed theo batch, ghi checkpoint atomic,
validate, publish, tạo index, kiểm soát quota/retry. **Ngoài phạm vi:** chunking, retrieval/rerank, generation,
quyền truy cập, DB trạng thái, delta embedding, multi-tenant, song song nhiều key (trừ khi quy mô chứng minh).
Corpus nhỏ/vừa: **full rebuild là mặc định**; incremental chỉ khi chi phí đo được đáng kể.

## 2. Bất biến hệ thống

1. **Cùng model, cùng tiền xử lý** giữa index và query. Lệch preprocessing là lỗi semantic âm thầm: hệ thống
   vẫn chạy nhưng vector không còn cùng không gian.
2. `data/chunks/` chỉ đọc; checkpoint ở vùng output riêng.
3. **Snapshot tự nhất quán:** publish chỉ đọc checkpoint thuộc đúng build đó (không quét mù file cũ).
4. **Không publish index partial mặc định:** batch lỗi được log, checkpoint thành công được giữ, nhưng publish
   dừng nếu snapshot chưa complete.
5. ID (`chunk_id` deterministic của chunking) và metadata ổn định, đủ để truy hồi/trích dẫn mà không mở lại corpus.
6. Validate ở biên: JSON checkpoint, response model, record vector DB.

## 3. Contract dữ liệu

- **Input:** `data/chunks/**/*.json`, mảng `Chunk` (giữ `chunk_id`, `content`, `breadcrumb`, `source_document`,
  `has_table`, `raw_table`).
- **Checkpoint:** `EmbeddedChunk` = `Chunk` + `embedding: list[float]`; ghi atomic (file tạm rồi rename), 1 file
  nguồn ↔ 1 file checkpoint. Mỗi build có **manifest snapshot**: danh sách nguồn↔checkpoint, tổng chunk/thành
  công/lỗi, model, dimension, version. Publish chỉ nhận manifest `complete` — chặn hai lỗi hay gặp: dùng lại
  checkpoint của source đã xoá, và publish lần embed đang dở.
- **Record vector store:** `id=chunk_id`, `values=embedding`, metadata `content` (nguyên văn, không phải bản đã
  word-segment), `breadcrumb`, `source_document`, `has_table`, `raw_table` (chỉ khi có bảng). Không lưu metadata chỉ
  hữu ích lúc ingest. Pinecone: bỏ hẳn key `raw_table` khi không có bảng, **không gửi `null`**.
- `EmbeddedChunk`, `PineconeMetadata`, `PineconeRecord` là Pydantic v2; không trao đổi `dict` thô giữa module.

## 4. Tiền xử lý và model

Model PhoBERT/multilingual cần word segmentation: `pyvi.ViTokenizer.tokenize()` cho `Chunk.content` ngay trước
khi gọi API; checkpoint/metadata giữ nguyên văn. Index và query có thể là hai module, nhưng **cùng quy tắc**;
không tái dùng tokenizer của chunking nếu làm hai package phụ thuộc sai chiều. `EmbeddingSettings` là nguồn chung
của `model_name`, token provider, token budget. Dimension lấy từ config/response đã xác thực, không hard-code.
**Đổi model = đổi không gian vector:** xác nhận dimension và rebuild index, không trộn vector hai model.

## 5. Gọi embedding API

Batch giữ thứ tự input để ghép đúng `chunk_id`. Trước khi chốt hằng số, đo với provider thật: nhận list text và
trả cùng thứ tự không, giới hạn batch/payload/timeout/quota, kiểu response/dimension/lỗi rate-limit. Áp dụng:
batch bảo thủ (hằng số nội bộ), timeout hữu hạn, retry giới hạn cho lỗi tạm thời, bộ đếm request chung dừng trước
ngưỡng quota, validate số vector/số thực/dimension nhất quán. Batch hết retry: log `chunk_id` cụ thể; **không** biến
lỗi thành vector rỗng, không đổi thứ tự, không âm thầm publish.

## 6. Publish vector store

Chỉ khi manifest complete: đọc và validate lại checkpoint → tạo index nếu chưa có (metric cosine theo model,
chờ ready) → chuyển thành record đã validate, upsert theo batch riêng → xác nhận số vector khớp snapshot.

**Chính sách thay thế:** ingestion offline + corpus nhỏ: `delete_all` rồi upsert full snapshot (loại orphan khi
chunking đổi) — đánh đổi là index trống/partial nếu lỗi sau delete. Nếu index đang phục vụ traffic: build vào
namespace/index phiên bản mới, validate count rồi chuyển. Đừng gọi `delete_all → upsert` là atomic.

## 7. Workflow

Chunk JSON (read-only) → preprocess + embed batch → validate vector → checkpoint atomic → manifest complete →
validate snapshot → publish. CLI tách hai thao tác: `embed` (checkpoint + manifest) và `upsert` (chỉ publish một
manifest complete được chỉ định). Typer ở `tools/`, logic ở package. Batch tuần tự mặc định; chỉ tăng concurrency
sau khi đo quota.

## 8. Module boundaries

`models.py` (EmbeddedChunk, metadata, record, manifest), `hf_client.py` (preprocess, batching, retry, quota,
validate response), `pinecone_client.py` (index lifecycle, publish theo batch), `pipeline.py` (điều phối),
`tools/embed_documents.py` (Typer mỏng). Config tập trung ở `config.py`; hằng số chỉ thuộc một cơ chế ở module đó.

## 9. Tiêu chí hoàn thành

Input không bị sửa; mọi vector cùng dimension, đúng thứ tự và cùng preprocessing với query-time; manifest 1-1
source/checkpoint, count chính xác, không chứa checkpoint cũ; record có ID deterministic, không metadata `null`;
quota/retry/batch lỗi có log, snapshot lỗi không tự publish; index có đủ số vector và không có orphan sau full rebuild.

## 10. Áp dụng cho hệ thống mới

Chốt trước khi code: model/dimension/preprocessing/metric; nguồn bất biến + schema tối thiểu; quota/batch/retry và
mức lỗi cho phép publish; chiến lược snapshot (full refresh vs versioned); ngưỡng cần delta/resume/concurrency.
Chưa có số đo thì chọn **sequential full rebuild + checkpoint + manifest** — baseline nhỏ nhất vẫn bảo vệ chi phí
API, tính đúng semantic và toàn vẹn index.
