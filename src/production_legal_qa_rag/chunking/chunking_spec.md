# Chunking — Markdown → Legal Retrieval Chunks: Reference Spec

> Giữ nguyên số mục vì code/spec khác tham chiếu.

## 1. Mục đích

Chuyển Markdown pháp luật đã chuẩn hoá thành chunk sẵn sàng embedding, **tự đủ nghĩa khi truy hồi một mình**.

> **Chunk theo đơn vị pháp lý trước, ngân sách token sau.** Đơn vị mặc định là **Khoản**; token budget là lý do duy nhất để cắt nhỏ một Khoản, không bao giờ để gộp hai Khoản.

**Trong phạm vi:** parse Markdown theo contract của `formatting/`; sinh chunk + breadcrumb + ID deterministic cho Khoản, front matter, back matter, bảng; đếm token bằng tokenizer thật của embedding model; ghi JSON trung gian. **Ngoài phạm vi:** embedding/upsert/retrieval/generation; trích metadata pháp lý; sửa Markdown nguồn, dùng LLM suy hierarchy, OCR; DB trạng thái (xử lý lại toàn bộ mỗi lần).

## 2. Contract đầu vào và đầu ra

Input `data/markdown/**/*.md`; output `data/chunks/**/*.json` (giữ cấu trúc thư mục, ghi atomic), mỗi file là mảng `Chunk` (không vector). `Chunk` (Pydantic v2):

| Field | Ý nghĩa |
| --- | --- |
| `chunk_id` | SHA-256 deterministic từ `source_document` + breadcrumb đầy đủ |
| `source_document` | Tên định danh đầy đủ của văn bản |
| `breadcrumb` | Viện dẫn đầy đủ để hiển thị/lọc; **không đưa vào chuỗi embedding** |
| `content` | Tiếng Việt nguyên văn để embed (trừ bảng, mục 7) |
| `token_count` | Số token `content` theo tokenizer embedding thật |
| `has_table` / `raw_table` / `standardization_table` | Cờ bảng / bảng gốc cho generation (không embed) / bảng dạng text gắn vào `content` |
| `is_split`, `split_index`, `split_total` | Quan hệ giữa các phần khi một đơn vị bị cắt |

`ChunkingResult`, `DocumentTree`, `KhoanNode` là contract nội bộ; không truyền `dict` thô.

## 3. Bất biến không được phá vỡ

1. **Không gộp Khoản** (trừ hai vùng document-level, mục 5).
2. **Không mất context áp dụng:** tách danh sách Điểm thì câu dẫn phải có ở đầu mọi chunk con.
3. Breadcrumb không cắt ngắn và không được embed.
4. `chunk_id` ổn định và duy nhất; trùng ID trong một file là lỗi file đó.
5. **Không làm hỏng bảng:** giữ nguyên một đơn vị kể cả vượt budget.
6. **Back matter tách khỏi Khoản cuối TRƯỚC khi parser đọc heading trong nó.**
7. Một file lỗi không chặn file khác; output thành công ghi atomic.

## 4. Parse hierarchy và định danh văn bản

`formatting/` là nguồn sự thật cho hierarchy; chunking không đọc style DOCX, không dùng LLM suy cấu trúc.

| Markdown | Cấp |
| --- | --- |
| `#` | Phần/Phụ lục (nếu text khớp regex cấu trúc) hoặc H1 tên tài liệu |
| `##` / `###` / `####` / `#####` | Chương / Mục / Điều / Khoản |
| `a)`, `b)`… | Điểm (trong nội dung, không phải heading) |

`parser.py` nhận diện heading bằng **cấp Markdown VÀ regex text cùng lúc** (phân biệt H1 tên tài liệu với H1 Phần/Phụ lục thật). `source_document` lấy từ paragraph không rỗng liền trước H1 tên tài liệu và text H1 (bỏ emphasis); không có H1 hợp lệ thì fallback tên file.

Breadcrumb Khoản: `{source_document} - Phần {x} - Chương {y} - Mục {z} - Điều {n}. {tên} - Khoản {m}` (bỏ cấp không có; Khoản bị cắt thêm `- Điểm a, b` khi có và `(phần i/n)` cho mọi chunk con).

**Nội dung không có heading Khoản** (dưới Điều hoặc trước Khoản đầu tiên) = **Khoản ngầm định cấp Điều** (breadcrumb dừng ở Điều). Khoản gộp trong Phụ lục (`##### 1. Tên thực thể`) cũng hợp lệ. **Heading H5 trong đoạn trích dẫn đang mở** (ngoặc kép pháp lý `“...”`) không đổi tree, là text của Khoản đang mở.

## 5. Hai document region ngoài hierarchy

Front matter và back matter là **hai đơn vị logic cấp văn bản** (mỗi cái có thể sinh nhiều chunk).

| Vùng | Boundary | Breadcrumb prefix |
| --- | --- | --- |
| Front matter | Mọi block trước heading pháp lý đầu tiên (H1 tên tài liệu vẫn thuộc vùng này) | `{source_document}` |
| Back matter | Mọi block sau dòng `---` đứng riêng (đúng 3 ký tự) | `{source_document} - Chú thích sửa đổi (cuối văn bản)` |

Gặp separator: flush Khoản chính đang mở, lấy mọi block còn lại làm `backmatter_content`, dừng parse hierarchy (Điều/Khoản trích trong chú thích không bị nhận thành luật). Hai vùng dùng `split_implicit_khoan()`: không vượt `max_tokens` thì một chunk; vượt thì `RecursiveCharacterTextSplitter` (`langchain-text-splitters`, `length_function=count_tokens`, `chunk_overlap=0`, separator `\n\n`, `\n`, `. `, `; `, space, ký tự), gắn `(phần i/n)`.

## 6. Thuật toán cắt Khoản

Khoản có bảng → một chunk (mục 7); `<= max_tokens` → một chunk; vượt budget → có Điểm thì đóng gói Điểm, lặp câu dẫn; không Điểm thì tách theo câu.

- **Khoản có Điểm:** câu dẫn (text trước Điểm đầu) lặp nguyên văn ở đầu mọi chunk con. Trừ token câu dẫn khỏi `max_tokens`, duyệt Điểm greedy; mỗi Điểm chỉ ở một chunk; câu dẫn là overlap duy nhất ở tầng này.
- **Fallback:** Điểm tự vượt budget → tách theo câu (`. `, `; `); buộc cắt thì lặp một câu cuối của chunk trước vào đầu chunk sau. Câu vẫn quá lớn → hạ xuống mệnh đề theo dấu phẩy; hết ranh giới hợp lý thì **chấp nhận chunk vượt budget thay vì làm hỏng câu pháp lý**. Khoản không Điểm đi thẳng vào fallback theo câu.

## 7. Bảng: giữ cấu trúc, tạo hai biểu diễn

Khoản chứa bảng pipe Markdown hoặc bảng HTML công thức = `has_table=True`, luôn **một chunk duy nhất, không cắt**, kể cả `token_count > max_tokens`. `raw_table` giữ nguyên khối bảng cho generation; `standardization_table` biến mỗi data row thành `nhãn hàng - cột: giá trị - ...` (header có `<br>` được chuẩn hoá); `content` = narrative Khoản + `standardization_table`, **không nhúng pipe-table thô vào text embed**.

## 8. Token budget và cấu hình

`count_tokens(text)` dùng **đúng preprocessing và tokenizer mà embedding model thấy**: `pyvi.ViTokenizer` word-segment rồi `AutoTokenizer` (PhoBERT) đếm token trên text đã segment, gồm special tokens. Word-segment chỉ để đếm; `Chunk.content` giữ tiếng Việt gốc, `embedding/` áp cùng preprocessing khi tạo vector. `EmbeddingSettings` (`config.py`) là nguồn duy nhất của `model_name` và `max_tokens` (hiện `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2`, `max_tokens=236`); không hard-code trong splitter.

## 9. Thiết kế module và workflow

`models.py`, `patterns.py` (regex heading/Điểm/bảng/separator/ranh giới fallback), `parser.py` (DocumentTree, tách region, breadcrumb prefix), `tokenizer.py` (segment + `count_tokens`, cache), `tables.py`, `splitter.py` (split Khoản và region, breadcrumb phần, ID), `pipeline.py` (điều phối, kiểm ID unique, atomic write, summary); CLI Typer mỏng `tools/chunk_documents.py` → `pipeline.convert_directory()`. Luồng: parse → split front matter → split từng Khoản → split back matter → từ chối `chunk_id` trùng → atomic write. Batch tuần tự, stateless, bắt lỗi theo file.

## 10. Tiêu chí hoàn thành

Mỗi chunk có `source_document`, breadcrumb, ID unique, token count đúng; Khoản không bảng không vượt `max_tokens` (trừ đơn vị không thể cắt hợp lý); chunk con từ Khoản có Điểm luôn bắt đầu bằng cùng câu dẫn; bảng không bị cắt; front/back matter không mất và không nhập vào Khoản gần nhất; cùng input luôn sinh cùng JSON/breadcrumb/ID; file lỗi không để output dở hay dừng batch.

## 11. Áp dụng cho corpus mới

Giữ bất biến và workflow; chốt trước mapping hierarchy nguồn → Markdown, marker tách phần cuối, tokenizer/budget của model thật, chính sách bảng. Gộp đơn vị pháp lý khác nhau, cắt bảng giữa chừng, hoặc dùng LLM quyết định hierarchy = đổi kiến trúc, phải chốt lại trước khi code.
