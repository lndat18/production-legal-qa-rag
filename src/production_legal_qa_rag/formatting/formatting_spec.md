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

**Ngoài phạm vi (chủ động không làm):**

- Idempotent tracking qua Postgres — bản cũ có bảng `documents` theo dõi
  trạng thái processing/completed/failed theo sha256; bản mới **bỏ hẳn DB**,
  mỗi lần chạy xử lý lại toàn bộ input.
- Quy trình test/golden file — đã có agent CI/CD riêng đảm nhiệm theo
  coding-convention, spec này không mô tả.
- Chunking, embedding — thuộc package `chunking/`, `embedding/` riêng.

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

| Việc                                                                         | Công cụ                                                                                                                   |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Đọc`.docx`                                                                | `python-docx`                                                                                                             |
| Data models trao đổi giữa module (kết quả cuối, front matter, warnings) | `pydantic` v2 `BaseModel`                                                                                               |
| CLI                                                                           | `typer`, đặt tại `tools/format_documents.py` theo coding-convention (không đặt CLI trong package `formatting/`) |
| Format/lint                                                                   | `ruff`                                                                                                                    |

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

## 6. Cấu trúc module trong `src/production_legal_qa_rag/formatting/`

| Module             | Trách nhiệm                                                                                                                                                          |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `patterns.py`    | Toàn bộ regex nhận diện cấu trúc (Phần/Chương/Mục/Điều/Khoản/Điểm, front matter, bảng theo nội dung) +`sort_key()` cho số hiệu có hậu tố chữ. |
| `docx_reader.py` | Đọc`.docx` bằng `python-docx`, trích xuất `Block` (paragraph/table) theo đúng thứ tự xuất hiện, giữ style/bold làm tín hiệu phụ.                 |
| `tables.py`      | Nhận diện loại bảng theo nội dung (chữ ký, quốc hiệu, đính kèm, bảng dữ liệu) và render bảng dữ liệu sang markdown/HTML.                            |
| `frontmatter.py` | Trích xuất metadata cấp văn bản (số hiệu, loại văn bản, ngày...) và render YAML.                                                                           |
| `footnotes.py`   | Tìm vùng chú thích sửa đổi, parse map số hiệu → nội dung, chèn lại inline hoặc dời xuống cuối nếu quá dài.                                         |
| `emitter.py`     | Duyệt`Block` theo thứ tự, áp heading mapping ở mục 3, sinh markdown cuối cùng.                                                                               |
| `validator.py`   | `validate()` kiểm tra heading-level-skip và các bất thường khác, trả về danh sách `QcWarning`.                                                           |
| `models.py`      | Pydantic models dùng làm input/output giữa các module trên và cho package khác gọi vào:`FormattingResult`, `FrontMatter`, `QcWarning`.                  |
| `pipeline.py`    | `convert_docx_to_markdown(path) -> FormattingResult` và `convert_directory(raw_dir, out_dir)` — orchestration mức trên, atomic write, gom summary.             |

`tools/format_documents.py` (ngoài package `formatting/`, theo
coding-convention) chỉ là Typer CLI mỏng gọi `pipeline.convert_directory()`.

## 7. Tiêu chí hoàn thành

- Chạy CLI trên toàn bộ `data/raw/` sinh ra `data/markdown/*.md` đúng heading
  mapping ở mục 3, khớp với 6 file hiện có trong `data/markdown/` (dùng làm
  tham chiếu, không phải golden-file chính thức).
- Chạy lại nhiều lần trên cùng input phải ra output giống hệt (deterministic,
  byte-for-byte) dù không có DB tracking.
- Không phát sinh QC warning code mới ngoài các warning đã biết là hợp lệ
  (theo ghi nhận trước đây khi rà soát bản `loader.py` cũ).
- 1 file lỗi không làm dừng toàn bộ batch; CLI kết thúc với summary rõ ràng
  file nào thành công/lỗi.
