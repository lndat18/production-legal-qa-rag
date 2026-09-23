# Formatting — DOCX → Markdown: Reference Spec

## 1. Mục đích

Chuyển tài liệu pháp luật `.docx` thành Markdown có cấu trúc để các bước
sau (đặc biệt là chunking và retrieval) có thể đọc **vị trí pháp lý** của
mỗi đoạn một cách tin cậy.

Nguyên tắc thiết kế cốt lõi:

> **Code deterministic sở hữu cấu trúc; LLM chỉ chuyển đổi cách trình bày
> của văn bản tự do.**

LLM không được quyết định ranh giới pháp lý, cấp heading, tên tài liệu hay
thứ tự nội dung. Những quyết định đó phải tái lập được từ DOCX gốc bằng code.

Đây là reference spec cho các corpus pháp luật có cấu trúc tương tự. Khi áp
dụng cho corpus mới, chỉ thay đổi profile nhận diện (regex, loại văn bản,
quy tắc chữ ký), không thay đổi các bất biến ở mục 3.

### Trong phạm vi

- Đọc DOCX theo đúng thứ tự paragraph/table xuất hiện.
- Nhận diện và render hierarchy pháp lý của phần thân.
- Giữ front matter và back matter dưới dạng Markdown thuần, không làm mất
  nội dung.
- Dùng LLM cho phần văn bản tự do cần giữ định dạng, có quota và lỗi API.
- Ghi output an toàn, xử lý batch độc lập theo từng file, phát QC warnings.

### Ngoài phạm vi

- OCR, xử lý ảnh scan, hay suy luận cấu trúc từ ảnh.
- Trích xuất metadata thành schema/YAML (`số hiệu`, `ngày ban hành`, ...).
- Embedding, chunk retrieval, hoặc upsert vector DB.
- Database theo dõi tiến trình; corpus nhỏ được xử lý lại toàn bộ mỗi lần.
- Khôi phục nội dung chú thích vào đúng vị trí inline trong phần thân.

## 2. Contract đầu vào và đầu ra

- Input: `data/raw/**/*.docx`.
- Output: `data/markdown/**/*.md`, ánh xạ một-một với file nguồn và giữ tên
  Unicode. File được ghi atomic bằng temporary sibling rồi rename.
- Output là Markdown thuần; không có YAML front matter hay object metadata.
- Một tài liệu được ghép theo thứ tự:

  ```text
  [front matter, nếu có]

  [middle / legal body]

  ---

  [back matter, nếu có]
  ```

`---` chỉ xuất hiện khi back matter thực sự có nội dung. Front/back matter
không được gán heading pháp lý; ngoại lệ duy nhất là H1 tên đầy đủ tài liệu
ở front matter (mục 4).

`Block` là immutable internal value object. Các contract công khai giữa
package/tầng orchestration dùng Pydantic v2 (`FormattingResult`, `QcWarning`);
không truyền `dict` thô giữa các tầng.

## 3. Bất biến không được phá vỡ

1. **Thứ tự và nội dung**: mỗi block trong DOCX giữ đúng thứ tự tương đối.
   Formatter không tóm tắt, bịa thêm, hay đổi thứ tự đoạn/bảng.
2. **Cấu trúc deterministic**: regex và heuristic code xác định mọi ranh giới
   front/middle/back cùng heading pháp lý. Không gọi LLM để phân loại chúng.
3. **Heading là API downstream**: heading chỉ mang nghĩa được quy định tại
   mục 5. Các bước sau không được suy luận hierarchy từ style DOCX.
4. **Tính nguyên tử theo vùng LLM**: nếu một chunk LLM của front hoặc back
   matter thất bại, bỏ toàn bộ vùng đó; tuyệt đối không ghép output dở dang.
5. **QC không làm ngừng batch**: warning biểu thị chất lượng cần xem xét,
   không làm fail một file đã chuyển đổi được phần an toàn.
6. **Một lỗi file không chặn file khác**: `convert_directory()` bắt lỗi theo
   file, tổng hợp kết quả và tiếp tục.

## 4. Phân vùng tài liệu và tên tài liệu

`docx_reader.py` trả về danh sách `Block` theo document order. `Block` giữ
`kind` (`paragraph` hoặc `table`), text, và các tín hiệu trình bày cần thiết
(bold/italic/table cells). Style DOCX chỉ là tín hiệu phụ; không phải nguồn
sự thật cho hierarchy.

Pipeline phân vùng bằng profile deterministic:

- **Front matter**: mọi block trước block đầu tiên khớp heading cấu trúc
  (`Phần`, `Phụ lục`, `Chương`, `Mục`, `Điều`, `Khoản` theo corpus profile).
- **Middle**: phần có hierarchy pháp lý; đây là vùng duy nhất được emitter
  sinh heading pháp lý.
- **Back matter**: block sau chữ ký cuối cùng được nhận diện bằng rule bảng/
  khối chữ ký của corpus. Không còn block sau chữ ký là trạng thái hợp lệ,
  không cần gọi LLM và không có warning.

### H1 tên tài liệu

Front matter cần đúng một điểm neo ổn định để downstream tạo
`source_document`. Tên đầy đủ tài liệu vì vậy phải được chèn bằng code thành:

```markdown
# TÊN ĐẦY ĐỦ CỦA TÀI LIỆU
```

Nó được tìm deterministic, không qua LLM. Với văn bản Việt Nam, profile mặc
định là: tìm paragraph chỉ chứa loại văn bản trong `RE_DOC_TYPE_ONLY`
(`LUẬT`, `BỘ LUẬT`, `NGHỊ ĐỊNH`, `THÔNG TƯ`, ...), rồi chọn paragraph không
rỗng ngay sau nó làm tên đầy đủ. Không phụ thuộc bold hay ALL CAPS của tên,
vì đây không phải tín hiệu ổn định.

Dòng loại văn bản vẫn là văn bản thường; không gộp vào H1. Nếu không tìm được
tên, render toàn bộ front matter như văn bản thường và phát warning
`frontmatter_title_not_found`. Không được để LLM tự đoán hoặc tự thêm H1.

H1 tên tài liệu không phải heading `Phần/Phụ lục`. Downstream phải phân biệt
hai loại bằng regex cấu trúc, không dựa vào vị trí của H1 đầu tiên trong file.

## 5. Mapping hierarchy của phần thân

Mapping dưới đây là contract Markdown cho văn bản pháp luật Việt Nam. Profile
khác có thể thay regex nhận diện, nhưng phải công bố mapping rõ ràng tương tự.

| Cấp pháp lý | Markdown | Quy tắc |
| --- | --- | --- |
| Phần / Phụ lục | `#` | Giữ tiêu đề gốc; có thể nối tên kéo dài sang block kế tiếp. |
| Chương | `##` | Hỗ trợ số La Mã hoặc số thường. |
| Mục | `###` | Hỗ trợ số và hậu tố chữ, ví dụ `3a`. |
| Điều | `####` | Giữ nguyên `Điều N. Tên điều`; hỗ trợ `41a`. |
| Khoản | `#####` | Chỉ heading `Khoản N`; nội dung nằm ở block sau. |
| Điểm | Không phải heading | Giữ `a)`, `b)` hoặc bullet như paragraph/list. |

Ngoại lệ có chủ đích: Khoản trong Phụ lục có nội dung định danh ngắn có thể
gộp nhãn và nội dung vào H5 để downstream không mất tên thực thể.

`patterns.py` phải:

- Normalize whitespace trước khi match, nhưng không làm mất nội dung.
- Hỗ trợ số hiệu có hậu tố chữ bằng `sort_key(number) -> tuple[int, str]`;
  không ép toàn bộ chuỗi qua `int()`.
- Xoá marker chú thích tham chiếu như `[12]` trong **middle** trước khi nhận
  diện heading và render. Marker trong back matter là nội dung chú thích thật
  nên phải giữ nguyên.

`validator.validate()` chạy sau emitter để phát warning khi skip cấp heading,
Điều rỗng, hay bất thường cấu trúc khác. Validator không sửa nội dung và
không làm fail file.

## 6. Chuyển đổi front/back matter bằng LLM

Hai vùng này là văn bản tự do: quốc hiệu, số hiệu, căn cứ, lời ban hành, nơi
nhận, chú thích sửa đổi và khối xác thực. Chúng được chuyển sang Markdown để
giữ bold, italic, xuống dòng và bảng đơn giản, nhưng không bị ép vào schema.

### Prompt contract

Prompt phải yêu cầu:

- Giữ nguyên nội dung, thứ tự đoạn và mọi thông tin.
- Giữ bold/italic khi input cung cấp tín hiệu tương ứng.
- Không tóm tắt, diễn giải, sửa luật, thêm/bớt nội dung hay thêm heading.
- Nếu gặp dòng chỉ gồm `-` hoặc `=`, phải tách nó khỏi text phía trên bằng một
  dòng trống.

Input prompt là serialization rõ ràng của `Block`, gồm ranh giới paragraph/
table và tín hiệu định dạng. LLM chỉ trả text Markdown, không dùng structured
output và không trích field.

### Guardrail sau LLM

Không tin prompt tuyệt đối. `patterns.escape_setext_underline(markdown)` phải
quét output đã ghép: nếu dòng `^[-=]{3,}\s*$` đứng ngay dưới một dòng không
rỗng, chèn một dòng trống ở giữa. Điều này ngăn CommonMark hiểu text phía
trên là setext heading vô tình, trong khi vẫn giữ nguyên đường kẻ trang trí.

Chỉ áp dụng guardrail này cho front/back matter từ LLM. Phần thân do emitter
sinh ATX headings nên không cần thay đổi.

## 7. Quota, chunking request và concurrency

Một back matter có thể lớn hơn TPM của provider, do đó request LLM được chunk
theo `Block`, không cắt giữa paragraph/table/câu.

- `chunk_blocks_for_llm(blocks, token_limit)` dồn tuần tự các block.
- Token estimate là heuristic bảo thủ `len(text) / 2.5`; chỉ dùng trước call
  để đặt ranh giới và quyết định chờ.
- Sau call thành công, rate limiter dùng `response.usage.total_tokens` thay
  estimate để cập nhật budget.
- Settings mặc định hiện tại: `chunk_token_limit=1500`, `tpm_limit=8000`,
  `rpm_limit=30`; limiter chỉ dùng 90% quota. Các giá trị nằm trong
  `LLMSettings`, không hard-code trong logic.
- Rate limiter là sliding window 60 giây, tồn tại suốt một lần
  `convert_directory()`; quota thuộc API key, không thuộc file.

`llm_client._convert_one(...) -> str | None` retry tối đa theo
`LLMSettings.max_retries` và timeout theo `LLMSettings.timeout_seconds`.
Sau khi hết retry, hàm trả `None`, không ném exception lên orchestration.

Khi có `GROQ_API_KEY_2`, `convert_chunks_concurrently(prompts)` dùng đúng hai
worker thread cố định: mỗi worker gắn với một client và một limiter của key
đó, cùng lấy job từ một `queue.Queue`. Kết quả được đặt theo index gốc nên
output vẫn đúng thứ tự prompt. Khi không có key thứ hai, chạy tuần tự, không
khởi tạo thread thừa. Concurrency chỉ nằm trong một tài liệu; batch vẫn tuần
tự theo file để giữ failure isolation và đơn giản vận hành.

## 8. Thiết kế module

| Module | Trách nhiệm duy nhất |
| --- | --- |
| `models.py` | Pydantic contracts: `QcWarning`, `FormattingResult`. |
| `docx_reader.py` | Đọc DOCX thành `Block`, serialize block, chunk theo block. |
| `patterns.py` | Regex cấu trúc, normalize/strip marker, sort key, guardrail setext. |
| `tables.py` | Render bảng phần thân và nhận diện bảng chữ ký. |
| `frontmatter.py` | Tìm boundary/title, `build_prompts()`, `assemble()`. Không gọi API. |
| `backmatter.py` | Tìm boundary, `build_prompts()`, `assemble()`. Không gọi API. |
| `emitter.py` | Render middle theo mapping ở mục 5. |
| `llm_client.py` | Provider adapter, retry, limiter, dispatch một/hai key. |
| `validator.py` | Phát QC warnings sau render. |
| `pipeline.py` | Điều phối, ghép vùng, atomic write và summary batch. |

Ranh giới quan trọng:

```text
frontmatter/backmatter: build_prompts (pure)
             │
             ├── pipeline: gộp mọi prompt của một file, gọi LLM đúng một lần
             │
             └── frontmatter/backmatter: assemble (pure)
```

`pipeline.py` nhớ độ dài từng nhóm prompt để phân chia kết quả trở lại đúng
vùng. Một `None` trong bất kỳ kết quả chunk của một vùng làm `assemble()` bỏ
toàn bộ vùng đó và trả một `QcWarning`:

- `llm_frontmatter_conversion_failed`
- `llm_backmatter_conversion_failed`
- `frontmatter_title_not_found`

CLI duy nhất là Typer command mỏng tại `tools/format_documents.py`, gọi
`pipeline.convert_directory()`; CLI không chứa business logic.

## 9. Workflow vận hành

```text
DOCX
  → read ordered blocks
  → split front / middle / back deterministically
  → build LLM prompts for front + back
  → convert prompts with quota-aware client
  → assemble front/back or emit region-level QC warning
  → emit + validate middle deterministically
  → compose Markdown
  → atomic write one output file
```

`convert_directory()` xử lý từng file độc lập, in lỗi có tên file, tiếp tục
batch, rồi in summary gồm số file thành công/lỗi và warning theo code. Không
có persistent state hay idempotency database.

## 10. Tiêu chí hoàn thành

- Output giữ đúng hierarchy, thứ tự block, nội dung phần thân và bảng của
  DOCX; heading tuân thủ đúng mapping mục 5.
- H1 tên tài liệu, nếu tìm thấy, giống byte-for-byte text DOCX và không chứa
  tiền tố loại văn bản.
- Phần thân deterministic giữa các lần chạy cùng input. Front/back có thể
  khác cách diễn đạt nhỏ do LLM, nhưng không đổi H1 deterministic hay thứ tự
  nội dung.
- Back matter trống không tạo call LLM, warning hoặc separator `---`.
- Chunk LLM không vượt budget an toàn trong bất cứ window 60 giây nào; prompt
  lớn vẫn được giữ nguyên ranh giới block.
- Lỗi một chunk LLM không tạo vùng half-converted; nó chỉ sinh warning vùng
  tương ứng và không chặn phần thân hay tài liệu khác.
- Không còn output có text ngay trên dòng `---`/`===` trang trí mà bị parser
  CommonMark hiểu là setext heading.
- Batch không để lại file output dở dang khi bị gián đoạn giữa lần ghi.

## 11. Cách áp dụng cho corpus mới

Giữ nguyên pipeline và bất biến. Chỉ xác nhận trước khi implement:

1. Vocabulary hierarchy và mapping heading.
2. Rule nhận diện đoạn bắt đầu phần thân, khối chữ ký và back matter.
3. Danh sách `RE_DOC_TYPE_ONLY` hoặc heuristic xác định title tương đương.
4. Quota provider, model, token limit và số key khả dụng.
5. Một tập tài liệu đại diện, gồm ít nhất một file có bảng, một file không có
   back matter, và một file có back matter lớn.

Nếu một yêu cầu mới buộc LLM quyết định hierarchy hoặc biến kết quả LLM thành
nguồn sự thật cho identifier/breadcrumb, đó là thay đổi kiến trúc và phải được
chốt lại trước khi code.
