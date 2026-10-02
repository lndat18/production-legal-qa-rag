# Chunking — Markdown → Legal Retrieval Chunks: Reference Spec

- Giữ nguyên số mục để không làm hỏng tham chiếu từ code/spec khác.
- Spec liên quan: [formatting_spec.md](../formatting/formatting_spec.md), [embedding_spec.md](../embedding/embedding_spec.md).

## 1. Mục đích

- Chuyển Markdown đã chuẩn hóa thành chunk tự đủ nghĩa khi retrieval riêng lẻ.
- **Đơn vị mặc định là Khoản:** chỉ tách vì token budget; không gộp hai Khoản.
- Làm: parse contract formatting; tạo chunk/breadcrumb/ID cho Khoản, front/back matter và bảng; đếm token thật; ghi JSON.
- Không làm: embed/upsert/retrieval/generation, metadata pháp lý, sửa nguồn, LLM suy hierarchy, OCR, DB trạng thái.

## 2. Contract đầu vào và đầu ra

- Input: `data/markdown/**/*.md`; output: `data/chunks/**/*.json`, giữ cấu trúc thư mục, ghi atomic, mỗi file là mảng `Chunk` không vector.
- `Chunk` dùng Pydantic v2:
  - `chunk_id`: SHA-256 từ `source_document` + breadcrumb đầy đủ.
  - `source_document`: định danh đầy đủ văn bản.
  - `breadcrumb`: vị trí pháp lý để hiển thị/lọc; không embed.
  - `content`: tiếng Việt nguyên văn, riêng bảng theo mục 7.
  - `token_count`: số token theo tokenizer embedding thật.
  - `has_table`, `raw_table`, `standardization_table`: cờ bảng, bảng gốc cho generation, text bảng để embed.
  - `is_split`, `split_index`, `split_total`: quan hệ giữa các phần khi tách.
- Contract nội bộ: `ChunkingResult`, `DocumentTree`, `KhoanNode`; không truyền `dict` thô.

## 3. Bất biến không được phá vỡ

- Không gộp Khoản; front/back matter là đơn vị riêng cấp văn bản (mục 5).
- Tách Điểm phải lặp câu dẫn để giữ điều kiện áp dụng.
- Breadcrumb đầy đủ, không embed; ID ổn định và duy nhất; ID trùng làm fail file.
- Không cắt bảng, kể cả vượt token budget.
- Tách back matter khỏi Khoản cuối trước khi parse heading trong vùng đó.
- Lỗi từng file không chặn batch; output thành công ghi atomic.

## 4. Parse hierarchy và định danh văn bản

- Formatting là nguồn sự thật về hierarchy; không đọc style DOCX hoặc hỏi LLM.
- Nhận heading bằng cả cấp Markdown và regex text:
  - H1: Phần/Phụ lục nếu khớp regex; nếu không, có thể là tên tài liệu.
  - H2/H3/H4/H5: Chương/Mục/Điều/Khoản.
  - `a)`, `b)`…: Điểm trong nội dung, không phải heading.
- `source_document`: paragraph không rỗng ngay trước H1 tên tài liệu + text H1 bỏ emphasis; không có H1 hợp lệ thì dùng tên file.
- Breadcrumb: `{source_document} - Phần {x} - Chương {y} - Mục {z} - Điều {n}. {tên} - Khoản {m}`; bỏ cấp không có.
- Chunk con thêm `- Điểm a, b` khi có và `(phần i/n)` cho mọi phần.
- Nội dung dưới Điều không có heading Khoản hoặc trước Khoản đầu: Khoản ngầm định, breadcrumb dừng ở Điều.
- H5 gộp trong Phụ lục (`##### 1. Tên thực thể`) hợp lệ.
- H5 trong trích dẫn pháp lý `“...”` đang mở là text của Khoản hiện tại; không đổi tree.

## 5. Hai document region ngoài hierarchy

- Front matter: mọi block trước heading pháp lý đầu tiên, gồm H1 tên tài liệu; breadcrumb `{source_document}`.
- Back matter: sau dòng đứng riêng đúng `---`; breadcrumb `{source_document} - Chú thích sửa đổi (cuối văn bản)`.
- Gặp separator: flush Khoản đang mở, đưa phần còn lại vào `backmatter_content`, dừng parse hierarchy.
- Mỗi vùng là một đơn vị logic, có thể tạo nhiều chunk qua `split_implicit_khoan()`.
- Không vượt `max_tokens`: một chunk. Vượt: `RecursiveCharacterTextSplitter` từ `langchain-text-splitters`.
- Splitter: `length_function=count_tokens`, `chunk_overlap=0`; separator `\n\n`, `\n`, `. `, `; `, space, ký tự; gắn `(phần i/n)`.

## 6. Thuật toán cắt Khoản

- Có bảng → một chunk (mục 7); không vượt `max_tokens` → một chunk.
- Vượt budget và có Điểm: trừ token câu dẫn, đóng gói Điểm greedy; mỗi Điểm chỉ thuộc một chunk; lặp nguyên văn câu dẫn ở mọi chunk con.
- Câu dẫn là overlap duy nhất ở tầng đóng gói Điểm.
- Điểm tự vượt budget hoặc Khoản không có Điểm: tách theo câu (`. `, `; `).
- Khi buộc cắt: lặp câu cuối của chunk trước ở chunk sau; câu vẫn quá dài thì tách mệnh đề theo dấu phẩy.
- Hết ranh giới hợp lý: chấp nhận vượt budget, không làm hỏng câu pháp lý.

## 7. Bảng: giữ cấu trúc, tạo hai biểu diễn

- Bảng pipe Markdown/HTML công thức → `has_table=True`; giữ nguyên một chunk dù `token_count > max_tokens`.
- `raw_table`: bảng gốc cho generation.
- `standardization_table`: mỗi hàng thành `nhãn hàng - cột: giá trị - ...`; chuẩn hóa `<br>` trong header.
- `content`: narrative của Khoản + `standardization_table`; không đưa pipe-table thô vào text embed.

## 8. Token budget và cấu hình

- `count_tokens(text)`: word-segment bằng `pyvi.ViTokenizer`, đếm bằng `AutoTokenizer` PhoBERT, gồm special tokens.
- Chỉ segment để đếm; `Chunk.content` giữ nguyên tiếng Việt; embedding áp cùng preprocessing khi tạo vector.
- `EmbeddingSettings` trong `config.py` là nguồn duy nhất của `model_name`/`max_tokens`; không hard-code splitter.
- Hiện tại: `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2`, `max_tokens=236`.

## 9. Thiết kế module và workflow

- `models.py`: contract; `patterns.py`: heading/Điểm/bảng/separator/fallback boundary.
- `parser.py`: tree, region, breadcrumb prefix; `tokenizer.py`: segment/count/cache; `tables.py`: biểu diễn bảng.
- `splitter.py`: cắt Khoản/region, breadcrumb phần, ID; `pipeline.py`: điều phối, kiểm ID, atomic write, summary.
- CLI Typer `tools/chunk_documents.py` gọi `pipeline.convert_directory()`.
- Luồng: parse → split front matter → split từng Khoản → split back matter → kiểm ID unique → ghi atomic.
- Batch tuần tự, stateless; bắt lỗi theo file; mỗi lần xử lý lại toàn bộ.

## 10. Tiêu chí hoàn thành

- Chunk có source/breadcrumb/ID unique/token count đúng; cùng input sinh cùng JSON/breadcrumb/ID.
- Khoản không bảng nằm trong budget, trừ đơn vị không thể cắt hợp lý.
- Chunk con có Điểm bắt đầu bằng cùng câu dẫn; bảng nguyên vẹn.
- Front/back không mất hoặc nhập vào Khoản gần nhất; file lỗi không để output dở hay dừng batch.

## 11. Áp dụng cho corpus mới

- Giữ workflow/bất biến; chốt mapping hierarchy, marker phần cuối, tokenizer/budget thật và chính sách bảng.
- Gộp đơn vị pháp lý khác nhau, cắt bảng hoặc dùng LLM quyết định hierarchy là đổi kiến trúc; chốt lại trước khi code.
