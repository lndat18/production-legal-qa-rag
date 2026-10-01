# Formatting — DOCX → Markdown: Reference Spec

> Giữ nguyên số mục vì code/spec khác tham chiếu (đặc biệt mục 1.1).

## 1. Mục đích

Chuyển `.docx` pháp luật thành Markdown có cấu trúc để chunking/retrieval đọc tin cậy **vị trí pháp lý** của mỗi đoạn.

> **Code deterministic sở hữu cấu trúc; LLM chỉ chuyển đổi cách trình bày của văn bản tự do.** LLM không quyết định ranh giới pháp lý, cấp heading, tên tài liệu hay thứ tự nội dung.

**Trong phạm vi:** đọc DOCX theo đúng thứ tự paragraph/table; nhận diện và render hierarchy phần thân; giữ front/back matter dạng Markdown thuần; dùng LLM cho văn bản tự do; ghi atomic, batch độc lập theo file, phát QC warning. **Ngoài phạm vi:** OCR/ảnh scan; trích metadata thành schema/YAML; embedding/chunk/upsert; DB theo dõi tiến trình; khôi phục chú thích inline vào phần thân.

## 2. Contract đầu vào và đầu ra

Input `data/raw/**/*.docx`; output `data/markdown/**/*.md` (một-một, giữ tên Unicode, ghi atomic), Markdown thuần, không YAML. Thứ tự: `[front matter]` → `[middle / legal body]` → `---` → `[back matter]`; `---` chỉ có khi back matter có nội dung. Front/back không gán heading pháp lý; ngoại lệ duy nhất là H1 tên tài liệu ở front matter (mục 4). `Block` là value object immutable; contract công khai dùng Pydantic v2 (`FormattingResult`, `QcWarning`), không truyền `dict` thô.

## 3. Bất biến không được phá vỡ

1. **Thứ tự và nội dung:** mỗi block giữ thứ tự tương đối; không tóm tắt, bịa, đổi thứ tự.
2. **Cấu trúc deterministic:** regex/heuristic code xác định mọi ranh giới front/middle/back và heading; không gọi LLM.
3. **Heading là API downstream:** heading chỉ mang nghĩa ở mục 5; bước sau không suy hierarchy từ style DOCX.
4. **Nguyên tử theo vùng LLM:** một chunk LLM của front/back matter thất bại → bỏ toàn bộ vùng đó, không ghép output dở.
5. **QC không dừng batch:** warning không làm fail file đã chuyển đổi được phần an toàn.
6. **Lỗi một file không chặn file khác:** `convert_directory()` bắt lỗi theo file rồi tiếp tục.

## 4. Phân vùng tài liệu và tên tài liệu

`docx_reader.py` trả `Block` theo document order (`kind` paragraph/table, text, tín hiệu bold/italic/cells); style DOCX chỉ là tín hiệu phụ. Phân vùng deterministic: **front matter** = mọi block trước heading cấu trúc đầu tiên (`Phần`, `Phụ lục`, `Chương`, `Mục`, `Điều`, `Khoản`); **middle** = vùng duy nhất emitter sinh heading pháp lý; **back matter** = block sau chữ ký cuối cùng (nhận bằng rule bảng/khối chữ ký). Không có block sau chữ ký là hợp lệ: không gọi LLM, không warning.

**H1 tên tài liệu:** chèn bằng code thành `# TÊN ĐẦY ĐỦ`: tìm paragraph chỉ chứa loại văn bản (`RE_DOC_TYPE_ONLY`: `LUẬT`, `BỘ LUẬT`, `NGHỊ ĐỊNH`, `THÔNG TƯ`…), lấy paragraph không rỗng ngay sau làm tên. **Không dựa vào bold/ALL CAPS.** Dòng loại văn bản vẫn là văn bản thường. Không tìm được tên: render front matter như văn bản thường + warning `frontmatter_title_not_found`; LLM không được đoán hay thêm H1. H1 tên tài liệu **không phải** heading Phần/Phụ lục — downstream phân biệt bằng regex cấu trúc.

## 5. Mapping hierarchy của phần thân

| Cấp | Markdown | Quy tắc |
| --- | --- | --- |
| Phần / Phụ lục | `#` | Giữ tiêu đề gốc; có thể nối tên kéo dài sang block kế |
| Chương | `##` | Số La Mã hoặc số thường |
| Mục | `###` | Số và hậu tố chữ (`3a`) |
| Điều | `####` | Giữ `Điều N. Tên điều`; hỗ trợ `41a` |
| Khoản | `#####` | Chỉ heading `Khoản N`; nội dung ở block sau |
| Điểm | không phải heading | Giữ `a)`, `b)` hoặc bullet như paragraph/list |

Ngoại lệ: Khoản trong Phụ lục có nội dung định danh ngắn được gộp nhãn + nội dung vào H5. `patterns.py`: normalize whitespace trước khi match; số hiệu có hậu tố chữ dùng `sort_key(number) -> tuple[int, str]` (không ép `int()`); **xoá marker tham chiếu `[12]` trong middle** trước khi nhận diện heading, **giữ nguyên trong back matter**. `validator.validate()` chạy sau emitter, phát warning (skip cấp heading, Điều rỗng…), không sửa nội dung, không làm fail file.

## 6. Chuyển đổi front/back matter bằng LLM

Hai vùng là văn bản tự do (quốc hiệu, số hiệu, căn cứ, nơi nhận, chú thích sửa đổi, khối xác thực) → chuyển sang Markdown, không ép schema. **Prompt phải yêu cầu:** giữ nguyên nội dung/thứ tự/mọi thông tin; giữ bold/italic; không tóm tắt/diễn giải/sửa luật/thêm bớt/thêm heading; dòng chỉ gồm `-` hoặc `=` phải tách khỏi text phía trên bằng một dòng trống. Input là serialization rõ ràng của `Block`; LLM chỉ trả text Markdown.

**Guardrail sau LLM:** `patterns.escape_setext_underline(markdown)` quét output đã ghép: dòng `^[-=]{3,}\s*$` đứng ngay dưới dòng không rỗng thì chèn dòng trống ở giữa (tránh text phía trên thành setext heading). Chỉ áp cho front/back từ LLM.

## 7. Quota, chunking request và concurrency

Request chunk theo `Block`, không cắt giữa paragraph/table/câu. `chunk_blocks_for_llm(blocks, token_limit)` dồn tuần tự; ước lượng token `len(text)/2.5` (chỉ để đặt ranh giới/quyết định chờ), sau call dùng `response.usage.total_tokens`. Mặc định `chunk_token_limit=1500`, `tpm_limit=8000`, `rpm_limit=30`, limiter chỉ dùng 90% quota (trong `LLMSettings`). Rate limiter là sliding window 60 giây sống suốt một `convert_directory()`; **quota thuộc API key, không thuộc file**.

`llm_client._convert_one(...) -> str | None` retry theo `LLMSettings.max_retries`/`timeout_seconds`; hết retry trả `None`, không ném exception. Khi có `GROQ_API_KEY_2`, `convert_chunks_concurrently(prompts)` dùng đúng hai worker thread (mỗi worker một client + một limiter) lấy job từ `queue.Queue`, kết quả đặt theo index gốc; không có key hai thì chạy tuần tự. Concurrency chỉ trong một tài liệu; batch vẫn tuần tự theo file.

## 8. Thiết kế module

`models.py` (`QcWarning`, `FormattingResult`), `docx_reader.py` (Block, serialize, chunk theo block), `patterns.py`, `tables.py` (render bảng thân, nhận bảng chữ ký), `frontmatter.py`/`backmatter.py` (boundary, `build_prompts()` và `assemble()` — **thuần, không gọi API**), `emitter.py`, `llm_client.py` (adapter, retry, limiter, dispatch 1/2 key), `validator.py`, `pipeline.py` (điều phối, ghép vùng, atomic write, summary); CLI Typer mỏng `tools/format_documents.py`. Luồng: `build_prompts` (thuần) → `pipeline` gộp mọi prompt của một file và gọi LLM đúng một lần → `assemble` (thuần); một `None` trong bất kỳ chunk của vùng làm `assemble()` bỏ cả vùng và trả `QcWarning`: `llm_frontmatter_conversion_failed`, `llm_backmatter_conversion_failed`, `frontmatter_title_not_found`.

## 9. Workflow vận hành

DOCX → đọc block có thứ tự → tách front/middle/back deterministic → dựng prompt front+back → gọi LLM có quota → ghép front/back hoặc phát warning cấp vùng → emit + validate middle → ghép Markdown → ghi atomic. `convert_directory()` xử lý từng file độc lập, in lỗi kèm tên file, tiếp tục, rồi in summary. Không state bền, không DB.

## 10. Tiêu chí hoàn thành

Output giữ đúng hierarchy/thứ tự/nội dung/bảng phần thân, heading đúng mapping mục 5; H1 tên tài liệu (nếu tìm thấy) giống byte-for-byte text DOCX và không chứa tiền tố loại văn bản; phần thân deterministic giữa các lần chạy; back matter trống không tạo call/warning/`---`; lỗi một chunk LLM chỉ sinh warning vùng, không tạo vùng half-converted; không text trên dòng `---`/`===` bị hiểu là setext heading; không file output dở khi bị gián đoạn.

## 11. Cách áp dụng cho corpus mới

Giữ pipeline và bất biến; chốt trước vocabulary hierarchy + mapping heading, rule nhận diện phần thân/khối chữ ký/back matter, `RE_DOC_TYPE_ONLY` hoặc heuristic tương đương, quota provider. LLM quyết định hierarchy hoặc làm nguồn sự thật cho identifier/breadcrumb = đổi kiến trúc, phải chốt lại trước khi code.
