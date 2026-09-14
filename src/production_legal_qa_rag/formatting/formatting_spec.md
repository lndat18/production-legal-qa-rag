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
`production_legal_qa_rag`, nhưng heading mapping, front matter, QC rules đã
chạy đúng trên toàn bộ corpus thật và không cần thiết kế lại). Điểm khác biệt
so với bản cũ nằm ở mục 5 và mục 7.

**Trong phạm vi:**

- Nhận diện heading Phần/Phụ Lục/Chương/Mục/Điều/Khoản, render đúng cấp `#`
  tương ứng.
- Front matter YAML (số hiệu, loại văn bản, ngày ban hành/hiệu lực...).
- Bảng (chữ ký, quốc hiệu, đính kèm → nhận diện theo nội dung; bảng dữ liệu →
  markdown table).
- Chú thích sửa đổi dạng `[n]` ở cuối văn bản (footnote khôi phục về đúng vị
  trí Khoản/Điều liên quan).
- QC warnings (không bao giờ làm fail file, chỉ cảnh báo).
- **[Mới — LLM extraction]** **Trích xuất nội dung** front matter (giá trị các field: số hiệu, loại văn
  bản, tên văn bản, ngày ban hành/hiệu lực...) và **nội dung** chú thích sửa
  đổi (`[n]`) bằng LLM (Groq-hosted `openai/gpt-oss-120b`) làm **đường chính**
  — không phải fallback. Regex hiện có (`frontmatter.py`, `footnotes.py`)
  giữ nguyên, đổi vai trò thành **baseline**: dùng để fallback khi LLM lỗi
  hoặc thiếu field bắt buộc, và để so sánh phát QC warning khi hai nguồn
  lệch nhau (mục 6, mục 7). Ranh giới rõ ràng: LLM chỉ đọc **giá trị nội
  dung**; việc **xác định biên cấu trúc** (heading Phần/Chương/.../Khoản,
  phân loại bảng nào là quốc hiệu/chữ ký/đính kèm, vị trí bắt đầu vùng chú
  thích `[1]`) vẫn giữ nguyên 100% deterministic bằng regex/heuristic, LLM
  không tham gia bước này.

**Ngoài phạm vi (chủ động không làm):**

- Idempotent tracking qua Postgres — bản cũ có bảng `documents` theo dõi
  trạng thái processing/completed/failed theo sha256; bản mới **bỏ hẳn DB**,
  mỗi lần chạy xử lý lại toàn bộ input.
- Quy trình test/golden file — đã có agent CI/CD riêng đảm nhiệm theo
  coding-convention, spec này không mô tả.
- Chunking, embedding — thuộc package `chunking/`, `embedding/` riêng.
- **[Mới — LLM extraction]** Xử lý ảnh/scan nhúng trong `.docx` (OCR, ảnh
  chụp trang văn bản...) — xác nhận với người dùng (2026-09-14): toàn bộ 6
  file `data/raw/*.docx` hiện tại là text gõ trực tiếp, không có file
  scan/ảnh nhúng nào. Không thiết kế trước cho trường hợp chưa xuất hiện
  trong corpus thật.
- **[Mới — LLM extraction]** QC thủ công bắt buộc trước khi ghi output —
  output của LLM đi thẳng vào `data/markdown/`, không có bước duyệt người
  dùng chặn giữa chừng; chỉ dựa vào QC warning tự động (mục 7) để phát hiện
  bất thường.
- **[Mới — LLM extraction]** Tự host mô hình LLM (vd. vLLM) — dùng Groq
  Cloud API (mô hình do Groq hosted), tái dùng `GROQ_API_KEY` đã có sẵn
  trong `.env`.

## 2. Input & Output

- **Input**: `data/raw/*.docx` — tên file tiếng Việt có dấu và khoảng trắng.
- **Output**: `data/markdown/*.md`, ánh xạ 1-1 theo tên file (đổi đuôi
  `.docx` → `.md`). Ghi atomic (file tạm + rename) để không để lại file dở
  dang nếu tiến trình bị ngắt giữa chừng.
- Mỗi file output gồm YAML front matter + nội dung markdown structured
  heading, ví dụ (rút gọn từ `data/markdown/Quy định mức lương tối thiểu.md`):

  ```markdown
  ---
  so_hieu: "293/2025/NĐ-CP"
  loai_van_ban: "Nghị định"
  ten_van_ban: "Nghị định quy định mức lương tối thiểu..."
  ngay_ban_hanh: "2025-11-10"
  ngay_hieu_luc: "2026-01-01"
  is_phu_luc: true
  is_van_ban_hop_nhat: false
  source_path: "..."
  parser_version: "1.0.0"
  ---
  #### Điều 2. Đối tượng áp dụng

  ##### Khoản 1

  Người lao động làm việc theo hợp đồng lao động...

  ##### Khoản 2

  Người sử dụng lao động theo quy định của Bộ luật Lao động, bao gồm:

  a) Doanh nghiệp theo quy định của Luật Doanh nghiệp.
  ```

## 3. Heading mapping (phần lõi)

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

| Việc                                                                         | Công cụ                                                                                                                                                                                               |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Đọc`.docx`                                                                | `python-docx`                                                                                                                                                                                         |
| Data models trao đổi giữa module (kết quả cuối, front matter, warnings) | `pydantic` v2 `BaseModel`                                                                                                                                                                           |
| **[Mới]** Trích nội dung front matter/footnote bằng LLM (đường chính, mục 1)    | `groq` SDK (Groq Cloud API) + `instructor` (ép structured output theo Pydantic schema) — model `openai/gpt-oss-120b`, tái dùng `GROQ_API_KEY` có sẵn trong `.env`; không tự host vLLM |
| CLI                                                                           | `typer`, đặt tại `tools/format_documents.py` theo coding-convention (không đặt CLI trong package `formatting/`)                                                                             |
| Format/lint                                                                   | `ruff`                                                                                                                                                                                                |

**[Mới — LLM extraction]** Cấu hình gọi LLM (`model_name="openai/gpt-oss-120b"`, `max_retries=2`,
`timeout_seconds=30`) bổ sung vào `LLMSettings` đã có sẵn trong
`src/production_legal_qa_rag/config.py` (dùng chung với `chunking/`, xem
`chunking_spec.md` mục 7) — không tạo config riêng trong `formatting/`.
Cần thêm `groq`, `instructor` vào `pyproject.toml`.

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
- **Tuần tự, không multiprocessing** — bản cũ dùng `ProcessPoolExecutor` với
  timeout riêng vì phải phối hợp với DB (mỗi worker tự cập nhật trạng thái).
  Bỏ DB thì lý do đó không còn; với 6 file, xử lý tuần tự đủ nhanh và đơn giản
  hơn hẳn (không cần xử lý `BrokenProcessPool`, timeout, hay serialize kết quả
  giữa process).
- **Lỗi 1 file không chặn các file khác** — bắt exception theo từng file, in
  ra lỗi kèm tên file, tiếp tục file kế tiếp; tổng kết cuối cùng liệt kê file
  nào lỗi.
- **QC warnings** không bao giờ làm fail file — in summary đếm theo code ở
  cuối lần chạy.
- **[Mới — LLM extraction]** **LLM lỗi/timeout không làm fail file** — `frontmatter.py`/`footnotes.py`
  tự bắt lỗi từ `llm_client.extract_structured` (trả `None` sau
  `max_retries=2` lần thử hoặc `timeout_seconds=30`), fallback về giá trị
  regex baseline và phát QC warning ngay tại chỗ (mục 6, mục 7) — không để
  exception thoát lên tầng `convert_directory` per-file catch.

## 6. Cấu trúc module trong `src/production_legal_qa_rag/formatting/`

| Module             | Trách nhiệm                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `patterns.py`    | Toàn bộ regex nhận diện cấu trúc (Phần/Chương/Mục/Điều/Khoản/Điểm, front matter, bảng theo nội dung) +`sort_key()` cho số hiệu có hậu tố chữ.                                                                                                                                                                                                                                                                                                                                                                                                           |
| `docx_reader.py` | Đọc`.docx` bằng `python-docx`, trích xuất `Block` (paragraph/table) theo đúng thứ tự xuất hiện, giữ style/bold làm tín hiệu phụ.                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `tables.py`      | Nhận diện loại bảng theo nội dung (chữ ký, quốc hiệu, đính kèm, bảng dữ liệu) và render bảng dữ liệu sang markdown/HTML.                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `frontmatter.py` | Trích xuất metadata cấp văn bản (số hiệu, loại văn bản, ngày...) và render YAML. **[Mới — LLM extraction]** Đọc **giá trị** field bằng LLM (`llm_client.extract_structured` với schema `FrontMatterExtraction`) làm đường chính; hàm regex hiện có (`extract_quoc_hieu`, `_find_ten_van_ban`, `_find_loai_van_ban`, `_find_ngay_hieu_luc`) **không đổi logic**, giữ nguyên làm baseline — fallback khi LLM trả `None`/thiếu field bắt buộc (`so_hieu`, `ngay_hieu_luc`), và để so sánh phát `QcWarning` khi hai nguồn lệch nhau. |
| `footnotes.py`   | Tìm vùng chú thích sửa đổi (`find_region_start`, xác định **vị trí** — không đổi, vẫn 100% deterministic), chèn lại inline hoặc dời xuống cuối nếu quá dài (`strip_markers`/`strip_all`/`render_blockquote` — không đổi). **[Mới — LLM extraction]** Đọc **nội dung** từng chú thích bằng LLM (schema `FootnoteExtraction`) làm đường chính; `parse_region` (regex) giữ nguyên làm baseline — fallback khi LLM lỗi, và để so sánh số lượng chú thích phát `QcWarning`.                                                               |
| `llm_client.py`  | **[Mới — LLM extraction, module hoàn toàn mới]** Client gọi Groq API bằng`groq` SDK + `instructor`. Hàm generic `extract_structured(prompt, schema, *, max_retries) -> BaseModel \| None` — trả `None` khi hết `max_retries` lần thử hoặc quá `timeout_seconds` (không raise ra ngoài, caller tự quyết fallback). Đọc `model_name`/`max_retries`/`timeout_seconds`/`groq_api_key` từ `LLMSettings` (`config.py`, mục 4).                                                                                                                                                                 |
| `emitter.py`     | Duyệt`Block` theo thứ tự, áp heading mapping ở mục 3, sinh markdown cuối cùng.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `validator.py`   | `validate()` kiểm tra heading-level-skip và các bất thường khác, trả về danh sách `QcWarning`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `models.py`      | Pydantic models dùng làm input/output giữa các module trên và cho package khác gọi vào:`FormattingResult`, `FrontMatter`, `QcWarning`. **[Mới — LLM extraction]** Thêm hai schema structured-output cho LLM: `FrontMatterExtraction`, `FootnoteExtraction`; thêm `QcWarningCode` mới: `llm_frontmatter_extraction_failed`, `llm_footnote_extraction_failed`, `llm_frontmatter_mismatch`, `llm_footnote_count_mismatch`.                                                                                                                                                       |
| `pipeline.py`    | `convert_docx_to_markdown(path) -> FormattingResult` và `convert_directory(raw_dir, out_dir)` — orchestration mức trên, atomic write, gom summary.                                                                                                                                                                                                                                                                                                                                                                                                                       |

`tools/format_documents.py` (ngoài package `formatting/`, theo
coding-convention) chỉ là Typer CLI mỏng gọi `pipeline.convert_directory()`.

## 7. Tiêu chí hoàn thành

- Chạy CLI trên toàn bộ `data/raw/` sinh ra `data/markdown/*.md` đúng heading
  mapping ở mục 3, khớp với 6 file hiện có trong `data/markdown/` (dùng làm
  tham chiếu, không phải golden-file chính thức).
- **[Sửa — LLM extraction]** Chạy lại nhiều lần trên cùng input phải ra
  output giống hệt (deterministic, byte-for-byte) cho **heading markdown,
  nội dung Khoản/Điều gốc, và vị trí chèn blockquote chú thích** (do
  `find_region_start`/`strip_all`/`emitter` quyết định — không đổi).
  **Không áp dụng** byte-for-byte cho nội dung YAML front matter hay nội
  dung text bên trong blockquote chú thích — hai phần này do LLM sinh, có
  thể diễn đạt khác nhau nhẹ giữa các lần chạy dù cùng input.
- **[Mới — LLM extraction]** Khi LLM lỗi (hết `max_retries=2` lần thử,
  timeout `timeout_seconds=30s`, hoặc structured output thiếu field bắt
  buộc), pipeline fallback về giá trị regex baseline, không fail file, và
  phát `QcWarning` (`llm_frontmatter_extraction_failed` hoặc
  `llm_footnote_extraction_failed`).
- **[Mới — LLM extraction]** Khi LLM trả kết quả hợp lệ nhưng lệch với
  baseline regex (khác nội dung field frontmatter, hoặc số lượng chú thích
  trích được không khớp), pipeline vẫn dùng giá trị LLM (đường chính) trong
  output, đồng thời phát `QcWarning` cảnh báo (`llm_frontmatter_mismatch`
  hoặc `llm_footnote_count_mismatch`) — không tự động chọn baseline khi có
  lệch.
- Không phát sinh QC warning code mới ngoài các warning đã biết là hợp lệ
  (theo ghi nhận trước đây khi rà soát bản `loader.py` cũ, cộng với 4 warning
  code LLM mới ở mục 6).
- 1 file lỗi không làm dừng toàn bộ batch; CLI kết thúc với summary rõ ràng
  file nào thành công/lỗi.
