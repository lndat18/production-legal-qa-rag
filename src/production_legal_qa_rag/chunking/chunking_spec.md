# Chunking — Markdown → Legal Retrieval Chunks: Reference Spec

## 1. Mục đích

Chuyển Markdown pháp luật đã chuẩn hoá thành các chunk sẵn sàng embedding mà
vẫn tự đủ nghĩa khi được truy hồi một mình.

Nguyên tắc cốt lõi:

> **Chunk theo đơn vị pháp lý trước, ngân sách token sau.**

Với văn bản pháp luật Việt Nam, đơn vị mặc định là **Khoản**. Token budget là
lý do duy nhất được phép cắt nhỏ một Khoản; không bao giờ là lý do để gộp hai
Khoản khác nhau.

Mỗi chunk phải trả lời được ba câu hỏi mà không cần đọc file nguồn: thuộc văn
bản nào, thuộc vị trí pháp lý nào, và nội dung quy định là gì.

### Trong phạm vi

- Parse Markdown theo contract của `formatting/` thành cấu trúc pháp lý.
- Sinh chunk, breadcrumb và ID deterministic cho Khoản, front matter,
  back matter và bảng.
- Đếm token theo tokenizer thật của embedding model để quyết định ranh giới.
- Ghi JSON trung gian cho bước embedding sau này.

### Ngoài phạm vi

- Sinh embedding, upsert vector DB, retrieval hoặc generation.
- Trích metadata pháp lý thành field riêng từ front matter.
- Sửa Markdown nguồn, đoán hierarchy bằng LLM, hoặc OCR.
- Theo dõi trạng thái bằng database; corpus nhỏ được xử lý lại toàn bộ mỗi
  lần chạy.

## 2. Contract đầu vào và đầu ra

- Input: `data/markdown/**/*.md`, là output Markdown thuần của `formatting/`.
- Output: `data/chunks/**/*.json`, giữ cấu trúc thư mục và đổi phần mở rộng
  `.md` thành `.json`. File được ghi atomic bằng temporary sibling rồi rename.
- Mỗi file JSON là một mảng `Chunk`, không có vector.

`Chunk` là Pydantic v2 contract công khai:

| Field | Ý nghĩa |
| --- | --- |
| `chunk_id` | SHA-256 deterministic từ `source_document` và breadcrumb đầy đủ. |
| `source_document` | Tên định danh đầy đủ của văn bản. |
| `breadcrumb` | Viện dẫn đầy đủ để hiển thị/lọc; không đưa vào chuỗi embedding. |
| `content` | Nội dung tiếng Việt nguyên văn để embed, trừ bảng được chuẩn hoá ở mục 6. |
| `token_count` | Số token `content` theo tokenizer embedding thực. |
| `has_table` | Khoản gốc có bảng hay không. |
| `raw_table` | Markdown/HTML bảng gốc cho generation; không embed. |
| `standardization_table` | Bảng chuyển sang text để gắn vào `content`. |
| `is_split`, `split_index`, `split_total` | Quan hệ giữa các phần khi một đơn vị bị cắt. |

`ChunkingResult`, `DocumentTree` và `KhoanNode` cũng là Pydantic contracts
nội bộ giữa parser, splitter và pipeline. Không truyền `dict` thô giữa các
module.

## 3. Bất biến không được phá vỡ

1. **Không gộp Khoản**: một chunk chỉ thuộc một Khoản pháp lý, trừ hai vùng
   document-level ở mục 5.
2. **Không mất context áp dụng**: khi tách danh sách Điểm, câu dẫn phải có ở
   đầu mọi chunk con.
3. **Citation đầy đủ, content tối giản**: breadcrumb không bị cắt ngắn và
   không được embed; `content` mới là chuỗi semantic gửi model.
4. **ID ổn định và duy nhất**: cùng input phải sinh cùng `chunk_id`; trùng ID
   trong một file là lỗi file đó, không được âm thầm ghi đè khi upsert sau này.
5. **Không làm hỏng bảng**: bảng được giữ nguyên một đơn vị, kể cả khi vượt
   budget token.
6. **Không nhầm chú thích cuối với luật chính**: back matter bị tách khỏi
   Khoản cuối trước khi parser đọc các heading bên trong nó.
7. **Batch cô lập lỗi**: một file không parse/chunk được không chặn file khác;
   file output thành công luôn được ghi atomic.

## 4. Parse hierarchy và định danh văn bản

`formatting/` là nguồn sự thật cho Markdown hierarchy. Chunking không đọc
style DOCX và không suy luận cấu trúc bằng LLM.

| Markdown | Cấp pháp lý |
| --- | --- |
| `#` | Phần/Phụ lục nếu text khớp regex cấu trúc; hoặc H1 tên tài liệu. |
| `##` | Chương |
| `###` | Mục |
| `####` | Điều |
| `#####` | Khoản |
| `a)`, `b)`, ... | Điểm trong nội dung, không phải heading |

`parser.py` phải nhận diện heading bằng **cấp Markdown và regex text cùng lúc**.
Điều này phân biệt H1 tên tài liệu ở front matter với H1 Phần/Phụ lục thật.

`source_document` được tạo deterministic từ paragraph không rỗng liền ngay
trước H1 tên tài liệu và text H1, đã bỏ Markdown emphasis. Nếu không có H1
hợp lệ, fallback là tên file; parser vẫn giữ toàn bộ front matter, không
đoán tên bằng LLM.

Breadcrumb của Khoản:

```text
{source_document} - Phần {x} - Chương {y} - Mục {z} - Điều {n}. {tên điều} - Khoản {m}
```

Bỏ cấp không tồn tại. Nếu một Khoản được cắt, thêm `- Điểm a, b` khi có và
`(phần i/n)` cho mọi chunk con.

### Nội dung không có heading Khoản

Nội dung nằm trực tiếp dưới Điều, hoặc nằm trước Khoản đầu tiên của Điều, là
**Khoản ngầm định cấp Điều**: giữ content và breadcrumb dừng ở cấp Điều.
Khoản gộp trong Phụ lục (`##### 1. Tên thực thể`) cũng là Khoản hợp lệ, dù
không có Điều bao ngoài.

Heading H5 nằm trong đoạn trích dẫn đang mở (ngoặc kép pháp lý `“...”`) không
được làm thay đổi tree của văn bản hiện tại; nó là text của Khoản đang mở.
Với corpus mới, quote rule này phải được xác nhận hoặc thay bằng rule parser
tương đương trước khi áp dụng.

## 5. Hai document region ngoài hierarchy

Front matter và back matter là nội dung thật nhưng không thuộc Điều/Khoản.
Chúng là **hai đơn vị logic cấp văn bản**, không phải Khoản pháp lý; mỗi đơn
vị có thể sinh một hoặc nhiều `Chunk`.

| Vùng | Boundary deterministic | Breadcrumb prefix |
| --- | --- | --- |
| Front matter | Mọi block trước heading pháp lý đầu tiên. H1 tên tài liệu vẫn thuộc vùng này. | `{source_document}` |
| Back matter | Mọi block sau dòng `---` đứng riêng, chính xác ba ký tự. | `{source_document} - Chú thích sửa đổi (cuối văn bản)` |

Khi gặp back-matter separator, parser phải flush Khoản chính đang mở, lấy tất
cả block còn lại làm `backmatter_content`, rồi dừng parse hierarchy. Nhờ vậy
Điều/Khoản trích dẫn trong chú thích sửa đổi không bị nhận thành luật của tài
liệu hiện tại.

Hai vùng không dùng thuật toán Điểm/câu của Khoản. Gọi
`split_implicit_khoan()`:

1. Đếm toàn vùng bằng tokenizer embedding.
2. Nếu không vượt `max_tokens`, giữ nguyên một chunk.
3. Nếu vượt, dùng `langchain_text_splitters.RecursiveCharacterTextSplitter`
   với `length_function=count_tokens`, `chunk_overlap=0`, và separator theo
   thứ tự `\n\n`, `\n`, `. `, `; `, space, ký tự đơn.
4. Gắn `(phần i/n)` theo thứ tự khi có nhiều phần.

`RecursiveCharacterTextSplitter` là dependency Python nhỏ
`langchain-text-splitters`; nó chỉ xử lý ranh giới text cho hai vùng phi-cấu
trúc, không sở hữu logic pháp lý hay làm repository bớt "Python".

## 6. Thuật toán cắt Khoản

### Luồng quyết định

```text
Khoản
  ├─ có bảng?          → giữ nguyên một chunk (mục 7)
  ├─ <= max_tokens?    → giữ nguyên một chunk
  └─ vượt budget
       ├─ có Điểm?     → đóng gói Điểm, lặp câu dẫn
       └─ không có Điểm → tách theo câu
```

### Khoản có Điểm

**Câu dẫn** là text đứng trước Điểm đầu tiên. Nó mang chủ thể, điều kiện,
ngoại lệ hoặc phủ định áp dụng cho mọi Điểm; vì vậy phải được lặp nguyên văn
ở đầu mọi chunk con.

Trừ token câu dẫn khỏi `max_tokens`, rồi duyệt Điểm theo thứ tự với thuật
toán greedy cận dưới: chỉ thêm Điểm tiếp theo khi tổng không vượt phần budget
còn lại. Mỗi Điểm chỉ thuộc một chunk; câu dẫn là overlap có chủ đích duy nhất
ở tầng này.

### Fallback cho đơn vị quá lớn

Nếu một Điểm tự nó vượt budget, tách theo câu (`. `, `; `). Khi buộc phải cắt
ở tầng này, lặp một câu cuối của chunk trước vào đầu chunk sau để giữ mạch
ngữ cảnh. Nếu một câu vẫn quá lớn, có thể hạ tiếp xuống mệnh đề theo dấu phẩy;
nếu không còn ranh giới hợp lý, chấp nhận chunk vượt budget thay vì làm hỏng
câu pháp lý.

Khoản không có Điểm đi thẳng vào cùng fallback theo câu; không có khái niệm
câu dẫn riêng để lặp.

## 7. Bảng: giữ cấu trúc, tạo hai biểu diễn

Khoản chứa bảng pipe Markdown hoặc bảng HTML công thức là `has_table=True`.
Nó luôn là một chunk duy nhất, không cắt, kể cả khi `token_count > max_tokens`.
Tính toàn vẹn hàng/cột quan trọng hơn việc tuân token budget tuyệt đối.

- `raw_table`: giữ nguyên khối bảng cho generation/hiển thị sau này.
- `standardization_table`: mỗi data row thành text theo dạng
  `nhãn hàng - cột: giá trị - ...`; header có `<br>` được chuẩn hoá.
- `content`: narrative của Khoản, sau đó là `standardization_table`; không
  nhúng syntax pipe table thô vào embedding text.

## 8. Token budget và cấu hình

`count_tokens(text)` phải dùng đúng preprocessing và tokenizer mà embedding
model thấy. Với profile hiện tại:

1. `pyvi.ViTokenizer.tokenize()` word-segment tiếng Việt.
2. `transformers.AutoTokenizer` đếm token PhoBERT trên text đã segment, gồm
   special tokens.

Word-segment chỉ dùng để đếm. `Chunk.content` luôn giữ tiếng Việt gốc; package
`embedding/` chịu trách nhiệm áp cùng preprocessing khi tạo vector.

`EmbeddingSettings` trong `src/production_legal_qa_rag/config.py` là nguồn
duy nhất của `model_name` và `max_tokens`. Default hiện tại là
`CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` và `max_tokens=236`; corpus
hoặc model mới phải xác nhận lại budget này, không hard-code trong splitter.

## 9. Thiết kế module và workflow

| Module | Trách nhiệm duy nhất |
| --- | --- |
| `models.py` | Pydantic models: `Chunk`, `ChunkingResult`, `DocumentTree`, `KhoanNode`. |
| `patterns.py` | Regex headings, Điểm, bảng, back-matter separator và ranh giới fallback. |
| `parser.py` | Parse Markdown thành `DocumentTree`; tách document regions; dựng breadcrumb prefix. |
| `tokenizer.py` | Word-segment và `count_tokens()`, cache tokenizer. |
| `tables.py` | Chuyển `raw_table` thành `standardization_table`. |
| `splitter.py` | Split Khoản và document region; tạo breadcrumb phần và deterministic ID. |
| `pipeline.py` | Điều phối một file/batch, kiểm tra ID unique, atomic write, summary. |

CLI là Typer command mỏng tại `tools/chunk_documents.py`, chỉ gọi
`pipeline.convert_directory()`; không chứa business logic.

```text
Markdown
  → parse thành DocumentTree
  → split front matter (nếu có)
  → split từng Khoản
  → split back matter (nếu có)
  → từ chối chunk_id trùng
  → atomic write JSON
```

Batch xử lý tuần tự, stateless. `convert_directory()` bắt lỗi theo từng file,
in summary, rồi tiếp tục file sau.

## 10. Tiêu chí hoàn thành

- Mỗi chunk có `source_document`, breadcrumb, ID unique và token count đúng.
- Khoản không bảng không vượt `max_tokens`, trừ câu/đơn vị không thể cắt hợp
  lý hơn; Khoản có bảng là ngoại lệ được chấp nhận.
- Chunk con từ Khoản có Điểm luôn bắt đầu bằng cùng câu dẫn gốc.
- Bảng không bị cắt; `raw_table` giữ nguyên và `content` dùng dạng text
  chuẩn hoá thay vì pipe syntax.
- Front matter và back matter không mất, không nhập vào Khoản gần nhất, và
  có breadcrumb khác nhau để ID không thể va chạm.
- Heading trích dẫn trong back matter, hoặc H5 nằm trong quote pháp lý, không
  làm sai tree.
- Cùng Markdown input luôn sinh JSON, content, breadcrumb và chunk ID giống
  hệt; một file lỗi không để lại output dở dang hoặc dừng batch.

## 11. Áp dụng cho corpus mới

Giữ các bất biến và workflow. Trước khi code, cần chốt:

1. Mapping hierarchy nguồn → Markdown và vocabulary cho regex.
2. Đơn vị semantic tối thiểu (Khoản, Điều, khoản hợp đồng, mục handbook...).
3. Quy tắc title, front matter và back matter; đặc biệt marker tách phần cuối.
4. Tokenizer, preprocessing và token budget của embedding model thật.
5. Chính sách bảng: giữ nguyên, chuẩn hoá, hay tách theo hàng.
6. Các cấu trúc lồng như quote, appendix hoặc heading không chuẩn cần giữ.

Nếu yêu cầu mới buộc phải gộp đơn vị pháp lý khác nhau, cắt bảng giữa chừng,
hoặc dùng LLM để quyết định hierarchy, đó là thay đổi kiến trúc và phải được
chốt lại trước khi implementation.
