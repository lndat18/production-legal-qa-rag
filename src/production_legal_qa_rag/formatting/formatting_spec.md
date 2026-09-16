# Formatting — DOCX → Markdown

## 1. Mục tiêu & phạm vi

Chuyển văn bản pháp luật Việt Nam từ `.docx` (`data/raw/`) sang Markdown có cấu
trúc (`data/markdown/`), giữ trọn vẹn cấu trúc heading phân cấp:

```
Phần / Phụ Lục → Chương → Mục → Điều → Khoản → Điểm
```

Đây là bước đầu pipeline ingestion, làm nền cho `chunking/` đọc heading để xác
định breadcrumb của từng đoạn.

Thiết kế này **kế thừa gần như nguyên vẹn** logic đã được validate trước đây ở
`src/rag/ingestion/loader.py` (git history, commit `31b89a0` / `cb0a239` —
file đã bị xoá khỏi working tree khi tách package `rag` →
`production_legal_qa_rag`, nhưng heading mapping và QC rules đã chạy đúng trên
toàn bộ corpus thật và không cần thiết kế lại) **cho phần thân văn bản**
(Chương/Mục/Điều/Khoản/Điểm, bảng dữ liệu). Phần **front matter** (nội dung
đứng trước Chương/Mục/Điều/Khoản/Điểm đầu tiên) và **back matter** (nội dung
đứng sau chữ ký, nếu có) được **thiết kế lại hoàn toàn** ở bản này — xem mục
1.1. Output không còn khối YAML metadata — file `.md` là markdown thuần, ghép
trực tiếp từ 3 phần.

**Trong phạm vi:**

- Nhận diện heading Phần/Phụ Lục/Chương/Mục/Điều/Khoản, render đúng cấp `#`
  tương ứng cho **phần nội dung ở giữa** — logic này pipeline hiện có đã làm
  đúng, không sửa.
- Bảng dữ liệu và bảng đính kèm trong phần nội dung ở giữa → render sang
  markdown/HTML theo logic `tables.py` hiện có — không sửa.
- QC warnings (không bao giờ làm fail file, chỉ cảnh báo).
- **[Mới — Gemini conversion]** Front matter và back matter (mục 1.1):
  chuyển đổi trực tiếp từ DOCX sang markdown bằng Gemini API (model
  `gemini-3.6-flash`, free tier) — **không trích field**, không sinh YAML,
  chỉ giữ nguyên nội dung/định dạng (bold, nghiêng, xuống dòng) dưới dạng
  markdown.

### 1.1. Front matter & back matter — chuyển đổi bằng Gemini, không trích field

**Định nghĩa** (theo xác nhận người dùng, 2026-09-16):

- **Front matter**: toàn bộ nội dung đứng **trước** Chương/Mục/Điều/Khoản/
  Điểm đầu tiên trong văn bản — quốc hiệu, số hiệu, ngày ban hành, tên văn
  bản, các dòng "Căn cứ...", câu ban hành mở đầu.
- **Back matter**: toàn bộ nội dung đứng **sau** khối chữ ký cuối cùng, **nếu
  có** — kể cả chú thích sửa đổi dạng `[n]` (nếu văn bản có), "Nơi nhận", hay
  khối xác thực văn bản hợp nhất (vd. "XÁC THỰC VĂN BẢN HỢP NHẤT" kèm danh
  sách căn cứ sửa đổi từng điều). Nhiều văn bản **không có** back matter — đó
  là trạng thái hợp lệ, không phải lỗi.

**Cách xác định biên (boundary) — 100% deterministic, dùng lại logic hiện
có, không đổi:**

- Biên front matter: vị trí block đầu tiên khớp regex heading
  Phần/Chương/Mục/Điều/Khoản (`patterns.py`). Mọi block trước đó thuộc front
  matter.
- Biên back matter: vị trí sau bảng/khối được `tables.py` phân loại là **chữ
  ký**. Nếu còn block nào sau đó → là back matter; nếu không → không có back
  matter, **không gọi Gemini**, không phát QC warning.

**Cách chuyển đổi nội dung — đơn giản hoá so với thiết kế trước:**

- Các block (paragraph/table) thuộc vùng front matter (hoặc back matter)
  được serialize thành text kèm tín hiệu định dạng sẵn có từ
  `docx_reader.py` (bold, nghiêng, block là bảng hay đoạn văn — không cần
  phân loại "quốc hiệu"/"chữ ký" cụ thể ở bước này, phân loại chữ ký chỉ
  dùng để **tìm biên**, xem trên), gửi nguyên khối cho Gemini
  (`gemini-3.6-flash`) qua `llm_client.convert_to_markdown()`.
- Gemini trả về **text markdown thuần** — không có schema/structured output,
  không tách field. Yêu cầu trong prompt: giữ nguyên nội dung, giữ in
  đậm/nghiêng nếu bản gốc có, giữ đúng thứ tự đoạn, **không tóm tắt, không
  thêm/bớt nội dung**. Bảng 2 cột dạng quốc hiệu (CHÍNH PHỦ / CỘNG HÒA XÃ HỘI
  CHỦ NGHĨA VIỆT NAM...) không bắt buộc giữ đúng bố cục song song (Markdown
  thuần không hỗ trợ cột) — Gemini tự quyết định trình bày tuần tự hợp lý,
  miễn giữ đúng nội dung.
- **Không dùng heading markdown (`#`) trong front matter/back matter** —
  kể cả tên loại văn bản ("NGHỊ ĐỊNH") hay tên đầy đủ văn bản. Prompt yêu cầu
  Gemini chỉ dùng đoạn văn thường + in đậm/nghiêng để nhấn mạnh (giống văn
  bản gốc), không tạo heading level nào. Lý do: heading level 1-5 trong
  output chỉ dành riêng cho mapping Phần/Chương/Mục/Điều/Khoản ở mục 3 (do
  `chunking/` dựa vào để dựng breadcrumb) — để Gemini tự gán heading cho tên
  văn bản sẽ lẫn vào hệ heading đó và làm sai breadcrumb.
- **Không có schema, không có "field bắt buộc"** — khác thiết kế trước đó
  (không còn khái niệm `so_hieu`/`loai_van_ban`/... là field riêng). Toàn bộ
  thông tin này vẫn tồn tại **dưới dạng text** trong phần front matter
  markdown, không bị mất, chỉ không được tách thành field có cấu trúc.
- **Không có regex baseline/fallback** (đã xác nhận 2026-09-14/16). Khi
  Gemini lỗi/timeout: bỏ qua phần front matter hoặc back matter tương ứng
  (không render), phát QC warning, **không làm fail file**. Với front
  matter, đây là mất mát nội dung thật sự (không phải chỉ thiếu field) —
  chấp nhận theo yêu cầu đơn giản hoá của người dùng, không thiết kế cơ chế
  bù đắp nào khác.

**Ghép output cuối cùng (`pipeline.py`/`emitter.py`):**

```
[front matter markdown từ Gemini, nếu có]

[nội dung ở giữa — pipeline hiện có sinh ra, không đổi]

[back matter markdown từ Gemini, nếu có, ngăn cách bằng dòng `---`]
```

Không có YAML, không có heading level gán riêng cho front/back matter —
đây là **văn bản thường** nối trực tiếp vào file `.md`.

**Ranh giới rõ ràng:** việc **xác định biên cấu trúc** (heading
Phần/Chương/.../Khoản, phân loại bảng chữ ký) vẫn giữ nguyên 100%
deterministic bằng regex/heuristic hiện có — Gemini **chỉ** chuyển đổi nội
dung hai vùng đã xác định biên sẵn sang markdown, không tham gia phân tích
cấu trúc thân văn bản.

**Ngoài phạm vi (chủ động không làm):**

- Idempotent tracking qua Postgres — mỗi lần chạy xử lý lại toàn bộ input,
  không có DB theo dõi trạng thái.
- Quy trình test/golden file — đã có agent CI/CD riêng đảm nhiệm theo
  coding-convention, spec này không mô tả.
- Chunking, embedding — thuộc package `chunking/`, `embedding/` riêng. Cách
  `chunking/` xử lý khối front/back matter (văn bản thường, không có heading
  level) là quyết định của `chunking_spec.md`, không mô tả ở đây.
- Xử lý ảnh/scan nhúng trong `.docx` (OCR...) — xác nhận với người dùng
  (2026-09-14): toàn bộ 6 file `data/raw/*.docx` hiện tại là text gõ trực
  tiếp, không có file scan/ảnh nhúng nào.
- QC thủ công bắt buộc trước khi ghi output — output đi thẳng vào
  `data/markdown/`, chỉ dựa vào QC warning tự động (mục 7) để phát hiện bất
  thường.
- Tự host mô hình LLM (vd. vLLM) — dùng Gemini API free tier (Google,
  cloud-hosted).
- Trích field có cấu trúc (`so_hieu`, `loai_van_ban`, `ten_van_ban`,
  `ngay_ban_hanh`, `ngay_hieu_luc`...) từ front matter — đã xác nhận với
  người dùng (2026-09-16): bỏ hoàn toàn, không sinh YAML metadata. Nếu
  package khác (`chunking/`, `embedding/`) cần các field này, đó là việc của
  spec khác, không phải formatting.
- Regex baseline/fallback cho front matter, back matter — bỏ hoàn toàn.
- Khôi phục chú thích `[n]` về vị trí gốc trong thân văn bản (inline
  reinsertion) — bỏ hoàn toàn, back matter luôn là 1 khối ở cuối văn bản.

## 2. Input & Output

- **Input**: `data/raw/*.docx` — tên file tiếng Việt có dấu và khoảng trắng.
- **Output**: `data/markdown/*.md`, ánh xạ 1-1 theo tên file (đổi đuôi
  `.docx` → `.md`). Ghi atomic (file tạm + rename) để không để lại file dở
  dang nếu tiến trình bị ngắt giữa chừng.
- Mỗi file output là markdown thuần, ghép: front matter (Gemini) + nội dung
  giữa (pipeline hiện có) + back matter (Gemini, nếu có). Ví dụ (rút gọn từ
  `data/markdown/Quy định mức lương tối thiểu.md`):

  ```markdown
  **CHÍNH PHỦ**
  -------

  **CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM**
  *Độc lập - Tự do - Hạnh phúc*
  ---------------

  Số: 293/2025/NĐ-CP

  *Hà Nội, ngày 10 tháng 11 năm 2025*

  **NGHỊ ĐỊNH**

  **QUY ĐỊNH MỨC LƯƠNG TỐI THIỂU ĐỐI VỚI NGƯỜI LAO ĐỘNG LÀM VIỆC THEO HỢP ĐỒNG LAO ĐỘNG**

  *Căn cứ Luật Tổ chức Chính phủ số 63/2025/QH15;*

  *Căn cứ Bộ luật Lao động số 45/2019/QH14;*

  *Theo đề nghị của Bộ trưởng Bộ Nội vụ;*

  Chính phủ ban hành Nghị định quy định mức lương tối thiểu đối với người lao
  động làm việc theo hợp đồng lao động.

  #### Điều 2. Đối tượng áp dụng

  ##### Khoản 1

  Người lao động làm việc theo hợp đồng lao động...

  ##### Khoản 2

  Người sử dụng lao động theo quy định của Bộ luật Lao động, bao gồm:

  a) Doanh nghiệp theo quy định của Luật Doanh nghiệp.

  ---

  [1] Luật Công nghiệp công nghệ số có căn cứ ban hành như sau: "Căn cứ Hiến
  pháp nước Cộng hoà xã hội chủ nghĩa Việt Nam; Quốc hội ban hành Luật Công
  nghiệp công nghệ số."...
  ```

  "NGHỊ ĐỊNH"/tên văn bản trong ví dụ trên **không** dùng heading markdown —
  chỉ in đậm, như văn bản gốc trình bày. Heading level 1-5 (mục 3) chỉ dành
  cho phần nội dung ở giữa, do pipeline sinh. Khối cuối (sau `---`) chỉ xuất
  hiện khi văn bản có back matter.

## 3. Heading mapping (phần lõi — áp dụng cho nội dung ở giữa)

| Cấp pháp lý    | Heading markdown           | Ghi chú                                                                                                                                                                                                                                                                                      |
| ----------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phần / Phụ Lục | `#`                      | Phụ lục dùng tiêu đề gốc, có thể ghép thêm`— tên` nếu tiêu đề trải nhiều dòng.                                                                                                                                                                                          |
| Chương          | `##`                     | Số La Mã hoặc số thường (`Chương IV`, `Chương 4`).                                                                                                                                                                                                                              |
| Mục              | `###`                    | Số + chữ phụ (`Mục 3a`).                                                                                                                                                                                                                                                                |
| Điều            | `####`                   | Giữ nguyên tiêu đề gốc "Điều N. Tên điều"; hỗ trợ số hiệu chữ (`Điều 41a`) cho văn bản hợp nhất.                                                                                                                                                                      |
| Khoản            | `#####`                  | **Chỉ chứa nhãn** — `##### Khoản N`, nội dung xuống dòng riêng. Ngoại lệ: trong Phụ Lục, nếu nội dung ngắn (≤ 120 ký tự) thì gộp vào heading (`##### 28. Thành phố Hồ Chí Minh`) vì tên là thông tin định danh cho breadcrumb của các đoạn con. |
| Điểm            | *(không phải heading)* | `a) ...` hoặc `- ...` xuất nguyên văn dưới dạng đoạn văn thường, chỉ dùng để QC/ngữ cảnh, không tạo heading level 6.                                                                                                                                                  |

Quy tắc bổ trợ:

- Đánh số Điều/Mục/Khoản có thể có hậu tố chữ (`41a`, `48b`) khi văn bản hợp
  nhất chèn điều khoản mới — so sánh thứ tự phải tách phần số và phần chữ,
  không parse nguyên chuỗi bằng `int()`.
- Nhận diện heading chỉ dựa trên regex đầu dòng của **level đó**, không dựa
  vào style DOCX gốc (style trong corpus thực tế không nhất quán).
- `validate()` chạy sau cùng, kiểm tra heading-level-skip (vd. gặp `#####`
  ngay sau `###` mà không qua `####`) và log QC warning, **không fail file**.

## 4. Tools & Integrations

| Việc                                                                                       | Công cụ                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Đọc`.docx`                                                                              | `python-docx`                                                                                                                                                                                         |
| Data models trao đổi giữa module (kết quả cuối, warnings)               | `pydantic` v2 `BaseModel`                                                                                                                                                                           |
| **[Mới]** Chuyển front matter/back matter từ DOCX sang markdown bằng Gemini (mục 1.1) | `google-genai` SDK (Gemini API, free tier) — model `gemini-3.6-flash`, gọi text generation thuần (`client.models.generate_content`, đọc `.text`) — **không dùng `response_schema`**, không cần structured output vì đây là tác vụ convert text, không trích field |
| CLI                                                                                         | `typer`, đặt tại `tools/format_documents.py` theo coding-convention (không đặt CLI trong package `formatting/`)                                                                             |
| Format/lint                                                                                 | `ruff`                                                                                                                                                                                                |

**[Mới]** Cấu hình gọi Gemini (`model_name="gemini-3.6-flash"`,
`max_retries=2`, `timeout_seconds=30`) thay thế hoàn toàn cấu hình Groq trong
`LLMSettings` (`src/production_legal_qa_rag/config.py`). Đọc key từ biến môi
trường `GEMINI_API_KEY` trong `.env`. Xoá `groq` và `instructor` khỏi
`pyproject.toml`, thêm `google-genai`. Free tier có rate limit theo phút —
với corpus 6 file (tối đa 2 lệnh gọi Gemini/file, xử lý tuần tự — mục 5),
không cần cơ chế rate-limit riêng; nếu corpus tăng quy mô lớn, đây là chỗ
đầu tiên cần xét lại.

## 5. Workflow & quản lý trạng thái

Khác biệt lớn nhất so với bản cũ: **stateless, không có DB**.

```
tools/format_documents.py (Typer CLI)
  → formatting.pipeline.convert_directory(raw_dir, out_dir)
      for each *.docx in raw_dir (tuần tự):
        → convert_docx_to_markdown(path) -> FormattingResult
        → write_atomic(out_path, result.markdown)
        → gom warnings vào summary
      → in summary cuối: số file thành công/lỗi, đếm QC warning theo code
```

- **Không có bảng theo dõi trạng thái** — mỗi lần chạy xử lý lại toàn bộ
  `data/raw/`, ghi đè `data/markdown/`. Corpus hiện tại chỉ 6 file, chi phí
  chạy lại toàn bộ là chấp nhận được; nếu corpus tăng lên hàng nghìn file sau
  này, đây là chỗ đầu tiên cần xét lại (không thiết kế trước cho quy mô chưa
  có).
- **Tuần tự, không multiprocessing** — với 6 file, xử lý tuần tự đủ nhanh và
  đơn giản hơn hẳn multiprocessing (không cần xử lý `BrokenProcessPool`,
  timeout, hay serialize kết quả giữa process).
- **Lỗi 1 file không chặn các file khác** — bắt exception theo từng file, in
  ra lỗi kèm tên file, tiếp tục file kế tiếp; tổng kết cuối cùng liệt kê file
  nào lỗi.
- **QC warnings** không bao giờ làm fail file — in summary đếm theo code ở
  cuối lần chạy.
- **[Mới]** **Gemini lỗi/timeout không làm fail file, và không có fallback**
  — `llm_client.convert_to_markdown` bắt lỗi, trả `None` sau
  `max_retries=2` lần thử hoặc `timeout_seconds=30`. Khi đó: bỏ qua phần
  front matter hoặc back matter tương ứng (không render vào output), phát QC
  warning tương ứng (mục 6, mục 7) tại chỗ — không để exception thoát lên
  tầng `convert_directory` per-file catch, không thử regex nào khác.

## 6. Cấu trúc module trong `src/production_legal_qa_rag/formatting/`

| Module             | Trách nhiệm                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `patterns.py`    | Regex nhận diện cấu trúc thân văn bản (Phần/Chương/Mục/Điều/Khoản/Điểm) + `sort_key()` cho số hiệu có hậu tố chữ. Dùng để xác định **biên** front matter (mục 1.1).                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `docx_reader.py` | Đọc`.docx` bằng `python-docx`, trích xuất `Block` (paragraph/table) theo đúng thứ tự xuất hiện, giữ style/bold/nghiêng làm tín hiệu phụ — dùng cho cả heading mapping (mục 3) lẫn serialize block cho Gemini (mục 1.1).                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `tables.py`      | Nhận diện loại bảng theo nội dung cho **phần nội dung ở giữa** (đính kèm, bảng dữ liệu) và render bảng dữ liệu sang markdown/HTML — không đổi. Riêng phân loại bảng **chữ ký** vẫn giữ, dùng để xác định biên back matter (mục 1.1); **bỏ** phân loại "quốc hiệu" — bảng đó giờ nằm trong vùng front matter, Gemini xử lý nguyên khối cùng các block khác, không cần tables.py can thiệp riêng.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `frontmatter.py` | Xác định biên front matter (block trước heading đầu tiên, dùng `patterns.py`), serialize block thành prompt (giữ tín hiệu bold/nghiêng từ `docx_reader.py`), gọi `llm_client.convert_to_markdown()`, trả về text markdown để `pipeline.py` ghép vào đầu output. Gemini lỗi → bỏ qua (không render front matter), phát `QcWarning` (`llm_frontmatter_conversion_failed`). Không còn schema/field nào (đã bỏ `FrontMatter`, `extract_quoc_hieu`, `_find_ten_van_ban`, `_find_loai_van_ban`, `_find_ngay_hieu_luc`). |
| `backmatter.py`  | **[Đổi tên từ `footnotes.py`, thiết kế lại]** Xác định biên back matter (block sau bảng chữ ký cuối cùng, dùng `tables.py`); nếu không còn block nào → không có back matter, dừng, không gọi Gemini. Nếu có → serialize block thành prompt, gọi `llm_client.convert_to_markdown()`, trả về text markdown để `pipeline.py` append vào cuối output (ngăn cách `---`, mục 2). Không còn khớp marker `[n]` hay khôi phục inline vào Khoản/Điều — các hàm cũ (`find_region_start`, `parse_region`, `strip_markers`, `strip_all`, `render_blockquote`) đã xoá. Gemini lỗi → bỏ qua back matter, phát `QcWarning` (`llm_backmatter_conversion_failed`). |
| `llm_client.py`  | Client gọi Gemini API bằng SDK `google-genai`. Hàm `convert_to_markdown(prompt: str, *, max_retries) -> str \| None` — text generation thuần, không structured output; trả `None` khi hết `max_retries` lần thử hoặc quá `timeout_seconds` (không raise ra ngoài, caller tự quyết bỏ qua phần tương ứng). Đọc `model_name`/`max_retries`/`timeout_seconds`/`gemini_api_key` từ `LLMSettings` (`config.py`, mục 4). |
| `emitter.py`     | Duyệt`Block` theo thứ tự **trong vùng nội dung ở giữa**, áp heading mapping ở mục 3, sinh markdown thân văn bản — không đổi. Không còn bước chèn inline chú thích vào giữa thân văn bản.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `validator.py`   | `validate()` kiểm tra heading-level-skip và các bất thường khác trong nội dung ở giữa, trả về danh sách `QcWarning`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `models.py`      | Pydantic models dùng làm input/output giữa các module trên: `FormattingResult` (markdown: str, warnings: list[QcWarning]), `QcWarning`. **Bỏ** `FrontMatter`, `FrontMatterExtraction`, `BackMatterExtraction` (không còn field có cấu trúc). `QcWarningCode` gồm `llm_frontmatter_conversion_failed`, `llm_backmatter_conversion_failed`. |
| `pipeline.py`    | `convert_docx_to_markdown(path) -> FormattingResult`: gọi `frontmatter.py` lấy front matter markdown (có thể rỗng), lấy nội dung giữa từ `emitter.py` (không đổi), gọi `backmatter.py` lấy back matter markdown (có thể không có), ghép 3 phần theo mục 1.1/2. `convert_directory(raw_dir, out_dir)` — orchestration mức trên, atomic write, gom summary. |

`tools/format_documents.py` (ngoài package `formatting/`, theo
coding-convention) chỉ là Typer CLI mỏng gọi `pipeline.convert_directory()`.

## 7. Tiêu chí hoàn thành

- Chạy CLI trên toàn bộ `data/raw/` sinh ra `data/markdown/*.md` đúng heading
  mapping ở mục 3 cho phần nội dung ở giữa, khớp với 6 file hiện có trong
  `data/markdown/` (dùng làm tham chiếu, không phải golden-file chính thức).
- Chạy lại nhiều lần trên cùng input phải ra output giống hệt (deterministic,
  byte-for-byte) cho **phần nội dung ở giữa** (do `emitter.py`/`patterns.py`/
  `tables.py` quyết định — không đổi). **Không áp dụng** byte-for-byte cho
  front matter hay back matter — cả hai do Gemini sinh, có thể diễn đạt khác
  nhau nhẹ giữa các lần chạy dù cùng input.
- Khi Gemini lỗi (hết `max_retries=2` lần thử, timeout `timeout_seconds=30s`):
  bỏ qua phần front matter hoặc back matter tương ứng (không render vào
  output), **không fail file**, và phát `QcWarning` tương ứng
  (`llm_frontmatter_conversion_failed` hoặc `llm_backmatter_conversion_failed`).
  Không có fallback regex nào được thử.
- Văn bản không có back matter (không có block nào sau bảng chữ ký) → không
  gọi Gemini cho phần này, không có khối back matter trong output, không phát
  QC warning — đây là trạng thái hợp lệ.
- Front matter/back matter markdown do Gemini sinh giữ đúng nội dung gốc
  (không tóm tắt/bịa thêm) — kiểm tra thủ công trên 6 file khi implement,
  không có cơ chế tự động so khớp (đã bỏ regex baseline để so sánh).
- Không phát sinh QC warning code mới ngoài các warning đã biết là hợp lệ
  (theo ghi nhận trước đây khi rà soát bản `loader.py` cũ, cộng với 2 warning
  code Gemini mới ở mục 6).
- 1 file lỗi không làm dừng toàn bộ batch; CLI kết thúc với summary rõ ràng
  file nào thành công/lỗi.
