# Chunking — Markdown → Chunk theo Khoản

## 1. Mục tiêu & phạm vi

Tách văn bản pháp luật đã chuẩn hoá (`data/markdown/*.md`, output của
`formatting/`) thành các **chunk cấp Khoản**, sẵn sàng cho bước `embedding/`.

Nguyên tắc lõi: **1 chunk = 1 Khoản**. Chỉ cắt nhỏ hơn khi nội dung Khoản vượt
ngân sách token của embedding model; không bao giờ gộp nhiều Khoản lại với
nhau (kể cả Khoản rất ngắn — giữ nguyên, không merge sang Khoản kế bên).

Mỗi chunk luôn mang theo **viện dẫn** (breadcrumb) xác định vị trí pháp lý
chính xác của nó trong văn bản gốc.

**Trong phạm vi:**

- Đọc `data/markdown/*.md`, dựng lại cây cấu trúc Phần/Chương/Mục/Điều/Khoản
  từ heading markdown.
- Với mỗi Khoản: tính token count, quyết định giữ nguyên hay cắt nhỏ theo
  thuật toán ở mục 4.
- Sinh breadcrumb cho từng chunk, bao gồm xử lý đặc biệt cho Khoản có mệnh đề
  phủ định/loại trừ (mục 4.4).
- Phát hiện Khoản chứa bảng dữ liệu (markdown table do `formatting/tables.py`
  sinh ra), giữ nguyên không cắt, sinh `raw_table`/`standardization_table`
  (mục 5).
- Tạo `src/production_legal_qa_rag/config.py` — nơi tập trung config dùng
  chung cho embedding model, vector DB (Pinecone), LLM (mục 7).

**Ngoài phạm vi (chủ động không làm):**

- Gọi embedding model để sinh vector — thuộc package `embedding/`. Chunking
  chỉ **đếm** token để quyết định ranh giới cắt, không sinh vector.
- Đẩy chunk lên Pinecone — thuộc bước sau (`embedding/` hoặc bước upsert
  riêng), chunking chỉ ghi ra file trung gian ở định dạng JSON sẵn sàng cho
  bước đó dùng (mục 2).
- Xử lý front matter phức tạp — kế thừa nguyên trạng từ `formatting/`,
  chunking chỉ đọc lại các field đã có sẵn.

## 2. Input & Output

- **Input**: `data/markdown/*.md` (front matter YAML + heading `#`–`#####`
  theo mapping đã định trong `formatting_spec.md` mục 3).
- **Output**: `data/chunks/*.jsonl`, ánh xạ 1-1 theo tên file nguồn (đổi đuôi
  `.md` → `.jsonl`), mỗi dòng là 1 `Chunk` (JSON). Ghi atomic (file tạm +
  rename), giống `formatting/`. Đây là định dạng trung gian — bước
  `embedding/` sẽ đọc lại, sinh vector rồi upsert từng `Chunk` lên **Pinecone**
  (id, vector, metadata), chunking không tự đẩy lên Pinecone.
- Mỗi `Chunk` gồm các field (`chunking/models.py`):| Field                             | Kiểu          | Ý nghĩa                                                                                                                                                                      |
  | --------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
  | `chunk_id`                      | `str`        | Deterministic, sinh từ`so_hieu` + đường dẫn breadcrumb + chỉ số phần (nếu bị cắt).                                                                                |
  | `source_document`               | `str`        | `so_hieu` (fallback: tên file) — để phân biệt chunk giữa các văn bản khác nhau.                                                                                   |
  | `breadcrumb`                    | `str`        | Viện dẫn đầy đủ, metadata hiển thị/lọc —**không đưa vào chuỗi embed**.                                                                                    |
  | `content`                       | `str`        | Nội dung thuần (nguyên văn tiếng Việt, chưa word-segment) — chuỗi thực sự đem đi embed. Nếu Khoản có bảng, đã nối thêm`standardization_table` (mục 5). |
  | `token_count`                   | `int`        | Số token của`content` theo tokenizer của embedding model (mục 6).                                                                                                        |
  | `has_table`                     | `bool`       | Khoản gốc có chứa bảng dữ liệu hay không (mục 5).                                                                                                                     |
  | `raw_table`                     | `str \| None` | Nguyên văn bảng markdown gốc, chỉ có khi`has_table=True` — gửi cho LLM ở bước generation, không đưa vào chuỗi embed (mục 5).                                |
  | `standardization_table`         | `str \| None` | Bảng đã chuyển thành text bằng cách nối theo hàng, chỉ có khi`has_table=True` (mục 5).                                                                           |
  | `is_split`                      | `bool`       | Khoản gốc có bị cắt thành nhiều chunk hay không. Luôn`False` nếu `has_table=True` (mục 5).                                                                      |
  | `split_index` / `split_total` | `int \| None` | Vị trí/tổng số phần nếu`is_split=True`.                                                                                                                                |
  | `negation_note`                 | `str \| None` | Câu chứa từ khoá phủ định/loại trừ trích từ đoạn mở đầu Khoản, nếu có (mục 4.4).                                                                           |

## 3. Breadcrumb — cấu trúc viện dẫn

Vì breadcrumb **không bị giới hạn bởi ngân sách token** (chỉ là metadata,
không embed), breadcrumb luôn ghi đầy đủ, không rút gọn.

Format (bỏ qua cấp không tồn tại trong văn bản — không phải văn bản nào cũng
có Phần/Mục):

```
{ten_van_ban} ({so_hieu}) - Phần {x} - Chương {y} - Mục {z} - Điều {n}. {tên điều} - Khoản {m}
```

Khi Khoản bị cắt nhỏ (mục 4), breadcrumb của từng chunk con nối thêm:

- Nhãn Điểm mà chunk con bao phủ, nếu cắt theo Điểm: `- Điểm a` (1 điểm)
  hoặc `- Điểm a, b` (nhiều điểm được gộp trong cùng 1 chunk con).
- `(phần i/n)` — luôn thêm khi 1 Khoản sinh ra nhiều hơn 1 chunk, kể cả khi
  cắt theo câu (không có nhãn Điểm rõ ràng).
- Câu phủ định (nếu phát hiện, mục 4.4).

Ví dụ breadcrumb 1 chunk con:

```
Luật Bảo hiểm xã hội (41/2024/QH15) - Chương II - Điều 3 - Khoản 4 - Điểm a, b (phần 1/2)
```

`ten_van_ban`/`so_hieu` lấy trực tiếp từ front matter YAML của file markdown
(đã có sẵn từ `formatting/frontmatter.py`, chunking chỉ đọc lại, không phân
tích thêm).

## 4. Quy tắc cắt Khoản

### 4.1. Trường hợp cơ bản

Trước tiên kiểm tra Khoản có chứa bảng dữ liệu hay không — nếu có, áp dụng
thẳng mục 5 (giữ nguyên, không cắt, bỏ qua toàn bộ mục 4.2–4.5). Nếu không có
bảng: tính `token_count` của toàn bộ nội dung Khoản (mục 6). Nếu
`token_count <= MAX_TOKENS` → **giữ nguyên, 1 Khoản = 1 chunk**, bất kể Khoản
dài hay ngắn (kể cả chỉ 1 câu).

### 4.2. Xác định danh sách "đơn vị" khi vượt ngân sách

Khi `token_count > MAX_TOKENS`, dựng danh sách đơn vị theo thứ tự xuất hiện:

- Nếu Khoản có các Điểm gắn nhãn (`a)`, `b)`, `c)`...):
  - **Đơn vị #0** (nếu có) = đoạn văn bản mở đầu, đứng trước nhãn Điểm đầu
    tiên (ví dụ câu dẫn "Khoản này không áp dụng đối với các trường hợp
    sau:").
  - Đơn vị tiếp theo = nội dung từng Điểm, theo đúng thứ tự a, b, c...
- Nếu Khoản **không có** Điểm gắn nhãn: toàn bộ nội dung là 1 khối, tách
  thẳng bằng câu (coi mỗi câu là 1 đơn vị) — bỏ qua tầng Điểm.

### 4.3. Thuật toán ghép "cận dưới" (greedy, không overlap ở tầng này)

Duyệt tuần tự các đơn vị, gom vào chunk hiện tại miễn tổng token không vượt
`MAX_TOKENS`; hễ thêm đơn vị kế tiếp làm vượt thì chốt chunk hiện tại và mở
chunk mới. Mã giả (không phải code thật, chỉ mô tả logic):

```
KHỞI TẠO chunk_đang_gom = rỗng, tổng_token = 0

LẶP QUA từng đơn_vị theo đúng thứ tự trong danh sách đơn vị (mục 4.2):
    số_token_đơn_vị = ĐẾM_TOKEN(đơn_vị)                (mục 6)

    NẾU số_token_đơn_vị > MAX_TOKENS:
        # đơn vị này tự nó đã vượt ngân sách, không thể gộp được nữa
        NẾU chunk_đang_gom KHÔNG rỗng: CHỐT chunk_đang_gom thành 1 chunk con
        THỰC HIỆN fallback tách theo câu cho riêng đơn_vị này (mục 4.4)
        RESET chunk_đang_gom = rỗng, tổng_token = 0
        CHUYỂN sang đơn_vị kế tiếp

    NGƯỢC LẠI, NẾU tổng_token + số_token_đơn_vị > MAX_TOKENS:
        # thêm đơn vị này vào sẽ vượt ngưỡng → "cận dưới": chốt trước, không vượt
        CHỐT chunk_đang_gom thành 1 chunk con
        chunk_đang_gom = [đơn_vị], tổng_token = số_token_đơn_vị

    NGƯỢC LẠI:
        # còn đủ chỗ, gộp tiếp
        THÊM đơn_vị vào chunk_đang_gom
        tổng_token = tổng_token + số_token_đơn_vị

SAU KHI hết đơn vị: NẾU chunk_đang_gom KHÔNG rỗng, CHỐT nốt thành chunk con
cuối cùng
```

Không overlap giữa các chunk con ở tầng Điểm — mỗi câu/Điểm chỉ thuộc đúng 1
chunk con.

### 4.4. Đơn vị vượt ngân sách ngay cả một mình (fallback theo câu)

Nếu 1 đơn vị (1 Điểm, hoặc đoạn không có Điểm) tự nó đã > `MAX_TOKENS`: tách
tiếp theo câu, áp dụng lại đúng thuật toán "cận dưới" ở mục 4.3 nhưng ở cấp
câu, và **cho phép overlap 1 câu cuối** giữa 2 chunk câu liên tiếp (câu cuối
của chunk trước lặp lại ở đầu chunk sau) để giữ mạch ngữ cảnh khi phải cắt
sâu tới mức này.

Tách câu bằng regex đơn giản trên dấu kết câu (`.`, `;`) theo sau bởi
khoảng trắng — chấp nhận sai số nhỏ ở tầng fallback hiếm gặp này, không dùng
thư viện NLP nặng cho việc này.

### 4.5. Câu phủ định/loại trừ — lan truyền vào breadcrumb

Đây là điểm **quan trọng nhất** của module: khi 1 Khoản có đoạn mở đầu (đơn
vị #0, mục 4.2) chứa từ khoá phủ định/loại trừ, ý nghĩa đó áp dụng cho **toàn
bộ** các Điểm phía sau — nhưng sau khi cắt, không phải chunk con nào cũng còn
giữ nguyên văn câu đó trong `content`. Vì vậy câu chứa từ khoá phải được lặp
lại vào `breadcrumb`/`negation_note` của **mọi** chunk con thuộc Khoản đó,
kể cả chunk không chứa đơn vị #0.

Danh sách từ khoá phủ định (đặt tại `chunking/patterns.py`, dễ mở rộng):

```
"trừ", "ngoại trừ", "loại trừ", "không áp dụng", "không thuộc", "không bao gồm"
```

Quy trình: nếu đơn vị #0 tồn tại và chứa 1 trong các từ khoá trên → trích
nguyên câu chứa từ khoá đó làm `negation_note`, gán cho **tất cả** chunk con
sinh ra từ Khoản này (kể cả khi Khoản chỉ bị cắt thành 1 chunk gồm toàn bộ
đơn vị #0 gộp Điểm a — vẫn gán, để nhất quán và dễ kiểm tra).

Nếu từ khoá phủ định nằm trong nội dung 1 Điểm cụ thể (không phải đơn vị #0)
thì không cần xử lý gì thêm — câu đó tự nhiên nằm trong `content` của đúng
chunk chứa Điểm đó.

## 5. Xử lý Khoản chứa bảng

Ví dụ thật trong corpus: Điều 3 - Khoản 1 của `Quy định mức lương tối thiểu.md` (bảng 4 vùng lương). `formatting/tables.py` render bảng dữ liệu
dạng markdown GFM chuẩn (`| ... |`, dòng phân cách `| --- | ... |`), tiêu đề
cột có thể chứa `<br>` để xuống dòng.

### 5.1. Không cắt khi có bảng

Nếu nội dung Khoản chứa 1 bảng markdown (có dòng bắt đầu bằng `|`) →
`has_table = True`, **giữ nguyên toàn bộ Khoản làm 1 chunk duy nhất, không áp
dụng thuật toán cắt ở mục 4.2–4.5, kể cả khi `token_count` vượt
`MAX_TOKENS`**. Đây là ngoại lệ có chủ đích: bảng bị cắt ngang sẽ mất hàng/mất
cột, vô nghĩa hơn nhiều so với việc 1 chunk vượt ngân sách token (embedding
model tự truncate phần dư khi đó — chấp nhận được, đổi lại giữ toàn vẹn dữ
liệu bảng).

### 5.2. `raw_table` — nguyên văn cho LLM

`raw_table` = đúng nguyên văn khối markdown table trích từ Khoản (header +
dòng phân cách + toàn bộ data row), không chỉnh sửa. Field này **không đưa
vào chuỗi embed** — chỉ mang theo để bước generation sau này đưa cho LLM,
giúp LLM trả lời/trình bày lại đúng số liệu gốc.

### 5.3. `standardization_table` — chuyển bảng thành text để embed

Chuyển mỗi dòng dữ liệu của bảng thành 1 câu text bằng cách nối cột đầu tiên
làm nhãn, các cột còn lại nối theo dạng `{tên cột}: {giá trị}`, các phần nối
bằng `-`; các dòng câu cách nhau bằng xuống dòng. Tiêu đề cột nào có `<br>`
thì bỏ `<br>` (nối liền, không thêm khoảng trắng) trước khi dùng làm tên cột.

Mã giả:

```
LẤY header = dòng tiêu đề bảng (bỏ dòng phân cách "---")
VỚI MỖI ô trong header: XOÁ "<br>" khỏi ô đó (nối liền, không chèn khoảng trắng)

KẾT QUẢ = danh sách rỗng
VỚI MỖI dòng_dữ_liệu trong bảng (theo đúng thứ tự):
    câu = dòng_dữ_liệu[0]                     # cột đầu tiên làm nhãn dòng
    VỚI MỖI cột i TỪ 1 ĐẾN hết (bỏ cột 0):
        câu = câu + " - " + header[i] + ": " + dòng_dữ_liệu[i]
    THÊM câu vào KẾT QUẢ

standardization_table = nối các phần tử của KẾT QUẢ bằng ký tự xuống dòng
```

Ví dụ áp dụng lên bảng Điều 3 - Khoản 1 (`Quy định mức lương tối thiểu.md`):

```
Vùng I - Mức lương tối thiểu tháng(Đơn vị: đồng/tháng): 5.310.000 - Mức lương tối thiểu giờ(Đơn vị: đồng/giờ): 25.500
Vùng II - Mức lương tối thiểu tháng(Đơn vị: đồng/tháng): 4.730.000 - Mức lương tối thiểu giờ(Đơn vị: đồng/giờ): 22.700
...
```

### 5.4. `content` khi có bảng

`content` (chuỗi thực sự đem đi embed) = phần văn bản tường thuật của Khoản
(câu dẫn trước bảng, nếu có) nối thêm `standardization_table` — **không**
nhúng nguyên văn markdown table (`|...|`) vào `content`, vì cú pháp pipe-table
thô không phù hợp để embedding model hiểu ngữ nghĩa.

## 6. Đếm token

Model `CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2` là PhoBERT-based
(`tokenizer_config.json`: `model_max_length = 192`; `config.json`:
`max_position_embeddings = 258`). PhoBERT được huấn luyện trên văn bản đã
**word-segment** (từ ghép nối bằng `_`) — đếm token trên văn bản tiếng Việt
thô sẽ sai lệch so với thực tế model xử lý.

Quy trình đếm token (`chunking/tokenizer.py`):

1. Word-segment văn bản bằng `pyvi.ViTokenizer.tokenize()`.
2. Đếm token bằng `transformers.AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)`
   trên văn bản đã segment.

Kết quả segment **chỉ dùng nội bộ để đếm**, không lưu vào `Chunk.content` —
`content` giữ nguyên văn tiếng Việt gốc (chưa segment), việc segment lại
trước khi embed thật sự là trách nhiệm của package `embedding/`.

`MAX_TOKENS = 236` (giá trị chốt cùng người dùng — nằm giữa `model_max_length`
192 và giới hạn kiến trúc tuyệt đối 258, chừa biên nhưng không quá bảo thủ).
Đặt tại `config.py` (mục 7), không hardcode trong `chunking/`.

## 7. Config tập trung (`src/production_legal_qa_rag/config.py`)

Module dùng chung, nằm **ngoài** package `chunking/` (không thuộc riêng 1
business logic nào — `embedding/` và các bước sau cũng sẽ import từ đây).
Dùng `pydantic-settings` (`BaseSettings`), đọc từ `.env` hiện có.

```python
class EmbeddingSettings(BaseSettings):
    model_name: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
    max_tokens: int = 236


class VectorDBSettings(BaseSettings):
    pinecone_api_key: str  # từ PINECONE_API_KEY, cần thêm vào .env
    index_name: str  # từ PINECONE_INDEX_NAME, cần thêm vào .env
    # VectorDB = Pinecone (đổi từ Postgres/pgvector ban đầu). docker-compose.yml
    # hiện vẫn chạy Postgres — spec này không quyết định có bỏ Postgres hay
    # không (Postgres có thể vẫn cần cho việc khác ngoài lưu vector), để ngỏ
    # cho quyết định sau.


class LLMSettings(BaseSettings):
    groq_api_key: str  # từ GROQ_API_KEY trong .env, đã có sẵn
    # model name: chưa chốt — để trống/TODO, thuộc phạm vi bước retrieval/
    # generation sau này, không đoán trước trong spec này
```

`VectorDBSettings`/`LLMSettings` chỉ khai báo phần đã có giá trị thật hoặc đã
chốt tên biến — không thêm field cho tính năng chưa được thiết kế (tránh
over-engineering).

Cần thêm dependency mới vào `pyproject.toml`: `pydantic-settings` (hiện chỉ
có transitive qua `langchain`, cần khai trực tiếp vì `config.py` import thẳng)
và `transformers` (hiện transitive qua `sentence-transformers`, `chunking/tokenizer.py`
import thẳng `AutoTokenizer`). Việc gọi Pinecone SDK thật sự (`pinecone` package)
thuộc phạm vi `embedding/`, chunking chỉ cần `config.py` khai đúng field.

## 8. Tools & Integrations

| Việc                               | Công cụ                                                                         |
| ----------------------------------- | --------------------------------------------------------------------------------- |
| Word segmentation tiếng Việt      | `pyvi` (đã có sẵn trong dependencies)                                       |
| Tokenizer đếm token               | `transformers.AutoTokenizer` (model PhoBERT-based ở mục 6)                    |
| Data models trao đổi giữa module | `pydantic` v2 `BaseModel`                                                     |
| Config tập trung                   | `pydantic-settings` `BaseSettings`                                            |
| CLI                                 | `typer`, đặt tại `tools/chunk_documents.py` (ngoài package `chunking/`) |
| Format/lint                         | `ruff`                                                                          |

## 9. Workflow & quản lý trạng thái

Stateless, không DB — cùng triết lý với `formatting/`:

```
tools/chunk_documents.py (Typer CLI)
  → chunking.pipeline.convert_directory(markdown_dir, out_dir)
      for each *.md in markdown_dir (tuần tự):
        → parse_markdown(path) -> DocumentTree (front matter + cây Phần/.../Khoản)
        → for each Khoản trong tree:
            split_khoan(khoan, max_tokens) -> list[Chunk]   # mục 4, 5
        → write_atomic(out_path, chunks dưới dạng JSONL)
        → gom thống kê vào summary (số chunk, số Khoản bị cắt...)
      → in summary cuối: số file thành công/lỗi, tổng số chunk, số Khoản bị cắt
```

- Không bảng theo dõi trạng thái, xử lý lại toàn bộ mỗi lần chạy (giống
  `formatting/`, corpus hiện tại nhỏ).
- Tuần tự, không multiprocessing.
- Lỗi 1 file không chặn các file khác.
- Deterministic: chạy lại trên cùng input phải ra `chunk_id` và nội dung
  giống hệt.

## 10. Cấu trúc module trong `src/production_legal_qa_rag/chunking/`

| Module           | Trách nhiệm                                                                                                                                                                                 |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `patterns.py`  | Regex nhận diện heading markdown (Phần/Chương/Mục/Điều/Khoản), nhãn Điểm (`a)`, `b)`...), dòng bảng markdown (`\|...\|`), danh sách từ khoá phủ định.                 |
| `parser.py`    | Đọc file markdown đã có front matter, dựng`DocumentTree` — cây breadcrumb tới từng Khoản kèm nội dung thô; phát hiện`has_table`/trích `raw_table` cho từng Khoản.    |
| `tables.py`    | Chuyển`raw_table` (markdown) thành `standardization_table` (mục 5.3).                                                                                                                  |
| `tokenizer.py` | `count_tokens(text) -> int` — word-segment (pyvi) + đếm bằng `AutoTokenizer` (mục 6).                                                                                                |
| `splitter.py`  | Thuật toán cắt Khoản (mục 4): xác định đơn vị, ghép "cận dưới", fallback theo câu có overlap, gắn`negation_note`; áp dụng ngoại lệ bảng (mục 5.1) trước khi cắt. |
| `models.py`    | Pydantic models:`Chunk`, `ChunkingResult`, `DocumentTree` (input/output giữa các module trên).                                                                                       |
| `pipeline.py`  | `convert_markdown_to_chunks(path) -> ChunkingResult` và `convert_directory(markdown_dir, out_dir)` — orchestration, atomic write, gom summary.                                          |

`tools/chunk_documents.py` chỉ là Typer CLI mỏng gọi
`pipeline.convert_directory()`.

## 11. Tiêu chí hoàn thành

- Chạy CLI trên toàn bộ `data/markdown/` sinh ra `data/chunks/*.jsonl`, mỗi
  chunk có breadcrumb đầy đủ và `token_count <= MAX_TOKENS` (trừ chunk có
  `has_table=True`, xem mục 5.1).
- Với các Khoản có đoạn mở đầu chứa từ khoá phủ định (mục 4.5) — xác nhận thủ
  công ít nhất 1 ví dụ thật trong corpus (`data/markdown/Luật bảo hiểm y tế.md`
  dòng 58, hoặc điểm b/c Khoản 4/7 Điều 2 trong `Luật bảo hiểm xã hội.md`) —
  mọi chunk con sinh ra từ Khoản đó đều có `negation_note` khớp câu gốc.
- Với Điều 3 - Khoản 1 của `data/markdown/Quy định mức lương tối thiểu.md`
  (bảng 4 vùng lương, mục 5) — sinh đúng 1 chunk duy nhất
  (`is_split=False`, `has_table=True`), `raw_table` khớp nguyên văn bảng
  markdown gốc, `standardization_table` có đúng 4 dòng theo mẫu mục 5.3.
- Chạy lại nhiều lần trên cùng input ra `chunk_id` và nội dung giống hệt
  (deterministic).
- 1 file lỗi không làm dừng toàn bộ batch; CLI kết thúc với summary rõ ràng.
- `config.py` được các package khác (`chunking/`, và sau này `embedding/`)
  import `EmbeddingSettings`/`VectorDBSettings` thay vì đọc `.env` trực tiếp.
