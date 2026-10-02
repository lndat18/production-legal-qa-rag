# Formatting — DOCX → Markdown: Reference Spec

- Giữ nguyên số mục để không làm hỏng tham chiếu từ code/spec khác.
- Spec liên quan: [chunking_spec.md](../chunking/chunking_spec.md).

## 1. Mục đích

- Chuyển DOCX pháp luật thành Markdown giữ đúng vị trí pháp lý cho chunking/retrieval.
- **Code quyết định cấu trúc; LLM chỉ đổi cách trình bày văn bản tự do.**
- Làm: đọc paragraph/table đúng thứ tự; phân vùng; render hierarchy; chuyển front/back matter; ghi atomic; QC
  theo file.
- Không làm: OCR, schema/YAML metadata, chunk/embed/upsert, DB tiến trình, đưa chú thích inline trở lại phần
  thân.

## 2. Contract đầu vào và đầu ra

- Input: `data/raw/**/*.docx`; output: `data/markdown/**/*.md`, một-một, giữ tên Unicode, ghi file tạm rồi
  rename.
- Markdown thuần, không YAML; thứ tự: front matter → thân luật → `---` → back matter.
- Chỉ thêm `---` khi back matter có nội dung.
- Front/back matter không có heading pháp lý; chỉ front matter được có H1 tên tài liệu (mục 4).
- `Block` immutable; contract công khai: Pydantic v2 `FormattingResult`, `QcWarning`; không truyền `dict` thô.

## 3. Bất biến không được phá vỡ

- Giữ nguyên nội dung và thứ tự tương đối của block; không tóm tắt, bịa hoặc đổi thứ tự.
- Regex/heuristic code xác định ranh giới front/middle/back và heading; LLM không tham gia.
- Heading là contract downstream theo mục 5; bước sau không suy hierarchy từ style DOCX.
- Một chunk LLM lỗi → bỏ cả vùng front/back tương ứng; không ghép output dở.
- QC chỉ cảnh báo; không làm fail phần đã chuyển đổi an toàn.
- `convert_directory()` bắt lỗi từng file và tiếp tục batch.

## 4. Phân vùng tài liệu và tên tài liệu

- `docx_reader.py` trả `Block` theo document order: `kind` paragraph/table, text, bold/italic/cells; style chỉ
  là tín hiệu phụ.
- Front matter: trước heading cấu trúc đầu tiên (`Phần`, `Phụ lục`, `Chương`, `Mục`, `Điều`, `Khoản`).
- Middle: vùng duy nhất emitter sinh heading pháp lý.
- Back matter: sau chữ ký cuối cùng, nhận bằng rule bảng/khối chữ ký.
- Không có nội dung sau chữ ký: không gọi LLM, không cảnh báo.
- H1 tên tài liệu do code chèn: tìm paragraph chỉ có loại văn bản bằng `RE_DOC_TYPE_ONLY` (`LUẬT`, `BỘ LUẬT`,
  `NGHỊ ĐỊNH`, `THÔNG TƯ`…), lấy paragraph không rỗng ngay sau làm `# TÊN ĐẦY ĐỦ`.
- Không nhận tên bằng bold/ALL CAPS; dòng loại văn bản vẫn là text thường.
- Không tìm được tên: giữ front matter dạng thường, cảnh báo `frontmatter_title_not_found`; LLM không
  đoán/thêm H1.
- Downstream phân biệt H1 tên tài liệu và H1 Phần/Phụ lục bằng regex cấu trúc.

## 5. Mapping hierarchy của phần thân

- Phần/Phụ lục → H1 (`#`); giữ tiêu đề gốc, có thể nối tên kéo dài sang block kế.
- Chương → H2 (`##`); nhận số La Mã và số thường.
- Mục → H3 (`###`); nhận hậu tố chữ như `3a`.
- Điều → H4 (`#### Điều N. Tên điều`); nhận hậu tố như `41a`.
- Khoản → H5 (`##### Khoản N`); nội dung ở block sau.
- Điểm → paragraph/list, giữ `a)`, `b)` hoặc bullet; không tạo heading.
- Ngoại lệ: Khoản trong Phụ lục có định danh ngắn được gộp nhãn + nội dung vào H5.
- `patterns.py`: chuẩn hóa whitespace trước match; `sort_key(number) -> tuple[int, str]` cho số có hậu tố,
  không ép `int()`.
- Xóa marker `[12]` ở middle trước nhận heading; giữ nguyên ở back matter.
- `validator.validate()` chạy sau emitter: cảnh báo skip cấp/Điều rỗng; không sửa nội dung hoặc làm fail file.

## 6. Chuyển đổi front/back matter bằng LLM

- Input: serialization rõ ràng của `Block`; output: text Markdown, không schema.
- Áp dụng cho quốc hiệu, số hiệu, căn cứ, nơi nhận, chú thích sửa đổi, khối xác thực.
- Prompt: giữ toàn bộ nội dung/thứ tự/bold/italic; không diễn giải, sửa luật, thêm/bớt hoặc thêm heading.
- Dòng chỉ có `-`/`=` phải cách text phía trên một dòng trống.
- Sau khi ghép output, `patterns.escape_setext_underline(markdown)` chèn dòng trống trước dòng `^[-=]{3,}\s*$`
  nếu phía trên không rỗng.
- Guardrail setext chỉ áp dụng front/back từ LLM; tránh biến text thường thành heading.

## 7. Quota, chunking request và concurrency

- `chunk_blocks_for_llm(blocks, token_limit)` dồn block tuần tự; không cắt giữa paragraph/table/câu.
- Ước lượng `len(text)/2.5` chỉ để chia request/chờ quota; sau call dùng `response.usage.total_tokens`.
- `LLMSettings`: `chunk_token_limit=1500`, `tpm_limit=8000`, `rpm_limit=30`; dùng 90% quota.
- Limiter: sliding window 60 giây, sống suốt `convert_directory()`; quota theo key, không theo file.
- `_convert_one(...) -> str | None`: timeout/retry theo `LLMSettings.timeout_seconds`/`max_retries`; hết retry
  trả `None`, không ném exception.
- `convert_chunks_concurrently(prompts)` có `GROQ_API_KEY_2`: hai worker thread, mỗi worker một client +
  limiter, lấy job từ `queue.Queue`, trả kết quả theo index gốc.
- Không có key hai: chạy tuần tự. Chỉ song song trong một tài liệu; batch file vẫn tuần tự.

## 8. Thiết kế module

- `models.py`: `QcWarning`, `FormattingResult`.
- `docx_reader.py`: `Block`, serialize, chia request theo block.
- `patterns.py`: nhận diện cấu trúc; `tables.py`: render bảng thân/nhận bảng chữ ký.
- `frontmatter.py`, `backmatter.py`: boundary, `build_prompts()`, `assemble()`; hàm thuần, không gọi API.
- `emitter.py`: heading/thân luật; `validator.py`: QC.
- `llm_client.py`: adapter, retry, limiter, dispatch 1/2 key.
- `pipeline.py`: gộp prompt front+back của một file, dispatch một lần, assemble từng vùng, ghi atomic,
  summary.
- `assemble()` gặp `None`: bỏ cả vùng; mã cảnh báo
  `llm_frontmatter_conversion_failed`/`llm_backmatter_conversion_failed`; tên tài liệu thiếu dùng
  `frontmatter_title_not_found`.
- `tools/format_documents.py`: CLI Typer mỏng.

## 9. Workflow vận hành

- DOCX → đọc block → phân vùng bằng code → dựng prompt front+back → dispatch LLM có quota.
- Assemble front/back hoặc cảnh báo vùng → emit/validate middle → ghép Markdown → ghi atomic.
- `convert_directory()`: xử lý file độc lập, in lỗi kèm tên file, tiếp tục, in summary; không DB/state bền.

## 10. Tiêu chí hoàn thành

- Giữ hierarchy, thứ tự, nội dung và bảng thân luật; heading đúng mục 5.
- H1 tìm được khớp byte-for-byte text DOCX, không thêm tiền tố loại văn bản.
- Middle deterministic; back matter trống không tạo call/warning/`---`.
- Lỗi chunk LLM chỉ bỏ vùng tương ứng; không có vùng half-converted.
- Text trên `---`/`===` không thành setext heading; gián đoạn không để output dở.

## 11. Cách áp dụng cho corpus mới

- Giữ pipeline/bất biến; chốt vocabulary hierarchy, mapping heading, rule thân/chữ ký/back matter,
  `RE_DOC_TYPE_ONLY` và quota provider.
- Dùng LLM quyết định hierarchy/identifier/breadcrumb là đổi kiến trúc; phải chốt lại trước khi code.
