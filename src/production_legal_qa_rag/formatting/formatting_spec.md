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
1.1, 1.2. Output không còn khối YAML metadata — file `.md` là markdown thuần,
ghép trực tiếp từ 3 phần.

**Trong phạm vi:**

- Nhận diện heading Phần/Phụ Lục/Chương/Mục/Điều/Khoản, render đúng cấp `#`
  tương ứng cho **phần nội dung ở giữa** — logic này pipeline hiện có đã làm
  đúng, không sửa.
- Bảng dữ liệu và bảng đính kèm trong phần nội dung ở giữa → render sang
  markdown/HTML theo logic `tables.py` hiện có — không sửa.
- QC warnings (không bao giờ làm fail file, chỉ cảnh báo).
- **[Mới — Groq conversion]** Front matter và back matter (mục 1.1): chuyển
  đổi trực tiếp từ DOCX sang markdown bằng Groq API (model
  `openai/gpt-oss-120b`, free tier) — **không trích field**, không sinh
  YAML, chỉ giữ nguyên nội dung/định dạng (bold, nghiêng, xuống dòng) dưới
  dạng markdown.
- **[Mới — chunking & rate limit]** Chia nhỏ front/back matter thành từng
  chunk theo ranh giới đoạn văn và giới hạn tốc độ gọi Groq free tier (mục
  1.2) — bắt buộc vì back matter một số văn bản vượt xa giới hạn token/phút
  của free tier nếu gửi nguyên khối.

### 1.1. Front matter & back matter — chuyển đổi bằng Groq, không trích field

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
  matter, **không gọi Groq**, không phát QC warning.

**Cách chuyển đổi nội dung — đơn giản hoá so với thiết kế trước:**

- Các block (paragraph/table) thuộc vùng front matter (hoặc back matter)
  được serialize thành text kèm tín hiệu định dạng sẵn có từ
  `docx_reader.py` (bold, nghiêng, block là bảng hay đoạn văn — không cần
  phân loại "quốc hiệu"/"chữ ký" cụ thể ở bước này, phân loại chữ ký chỉ
  dùng để **tìm biên**, xem trên), **chia thành chunk** (mục 1.2) rồi gửi
  từng chunk cho Groq (`openai/gpt-oss-120b`) qua
  `llm_client.convert_to_markdown()`.
- Groq trả về **text markdown thuần** — không có schema/structured output,
  không tách field. Yêu cầu trong prompt: giữ nguyên nội dung, giữ in
  đậm/nghiêng nếu bản gốc có, giữ đúng thứ tự đoạn, **không tóm tắt, không
  thêm/bớt nội dung**. Bảng 2 cột dạng quốc hiệu (CHÍNH PHỦ / CỘNG HÒA XÃ HỘI
  CHỦ NGHĨA VIỆT NAM...) không bắt buộc giữ đúng bố cục song song (Markdown
  thuần không hỗ trợ cột) — Groq tự quyết định trình bày tuần tự hợp lý,
  miễn giữ đúng nội dung.
- **[CẬP NHẬT 2026-09-17]** Đúng **1 dòng** trong front matter — tên đầy đủ
  của văn bản — dùng heading markdown (`#`), vd. `# QUY ĐỊNH MỨC LƯƠNG TỐI
  THIỂU ĐỐI VỚI NGƯỜI LAO ĐỘNG LÀM VIỆC THEO HỢP ĐỒNG LAO ĐỘNG`. Trước đây
  cấm hoàn toàn heading trong front matter/back matter; nay đổi vì
  `chunking/` cần 1 điểm neo xác định để lấy `source_document` (xem
  `chunking_spec.md` mục 2, mục 4.6). **Mọi nội dung front matter/back
  matter còn lại** (quốc hiệu, tên loại văn bản như "NGHỊ ĐỊNH", số hiệu,
  ngày ban hành, các dòng "Căn cứ...", câu ban hành mở đầu, toàn bộ back
  matter) **vẫn giữ nguyên quy tắc cũ** — đoạn văn thường + in đậm/nghiêng,
  không tạo thêm heading level nào khác. Lý do giới hạn chỉ đúng 1 dòng:
  heading level 1-5 trong output vẫn chủ yếu dành riêng cho mapping Phần/
  Chương/Mục/Điều/Khoản ở mục 3 (`chunking/` dựa vào để dựng breadcrumb) —
  `chunking/patterns.py` phân biệt heading tên văn bản này với heading
  Phần/Phụ Lục thật bằng regex cấu trúc chuyên biệt (không coi "gặp `#`
  đầu tiên trong file" là Phần/Phụ Lục), xem `chunking_spec.md` mục 4.6.
- **[CẬP NHẬT 2026-09-17, sửa lại — không còn để Groq tự quyết định]** Bản
  đầu của mục này giao cho Groq tự thêm `#` qua prompt — kiểm thử thực tế
  cho thấy Groq không làm việc này nhất quán (người dùng phải tự sửa tay
  front matter sau khi chạy CLI trên `Bộ luật Lao động.docx`). Nay đổi
  sang **xác định dòng tên văn bản bằng heuristic deterministic trong
  code**, tách nó khỏi phần gửi Groq, và tự chèn `# <nguyên văn dòng đó>`
  vào markdown — Groq không bao giờ thấy hay quyết định về dòng này, chỉ
  convert phần front matter đứng trước và đứng sau nó. Cách này cũng đảm
  bảo tên văn bản trong output giống hệt nguyên văn DOCX (không bị Groq
  diễn đạt lại) — quan trọng vì đây chính là `source_document`, khóa để
  `chunking/` phân biệt breadcrumb giữa các văn bản.

  **Heuristic xác định block tên văn bản** (hàm mới `frontmatter.find_title`,
  chạy trên các block front matter đã cắt biên bởi `find_boundary`, mục
  trên) — **dựa trên cấu trúc thật đã kiểm tra trên toàn bộ 6 file
  `data/raw/*.docx`**, không phải suy đoán: ngay sau bảng quốc hiệu (block
  đầu tiên), mọi văn bản trong corpus đều có đúng 1 block riêng là **dòng
  loại văn bản** (`"LUẬT"`, `"BỘ LUẬT"`, `"NGHỊ ĐỊNH"`...), và **ngay sau
  đó** là 1 block riêng khác là **tên/nội dung chính**. Hai block này
  **luôn tách biệt** (không bao giờ gộp sẵn 1 dòng trong DOCX gốc), và
  block tên/nội dung chính **không phải lúc nào cũng in đậm** — 2/6 file
  (dạng Nghị định) có block này hoàn toàn không bold. Vì vậy heuristic
  **không dựa vào bold/viết hoa của chính dòng tên văn bản** (điều kiện đó
  loại sai 2/6 file nếu áp dụng) — chỉ dựa vào việc định vị được dòng loại
  văn bản, dùng regex mới `patterns.RE_DOC_TYPE_ONLY` (khớp **trọn dòng**,
  danh sách hữu hạn: "LUẬT", "BỘ LUẬT", "NGHỊ ĐỊNH", "NGHỊ QUYẾT", "THÔNG
  TƯ", "THÔNG TƯ LIÊN TỊCH", "QUYẾT ĐỊNH", "PHÁP LỆNH", "CHỈ THỊ"):

  1. Tìm block **đầu tiên** (`kind == "paragraph"`) trong vùng front matter
     khớp `RE_DOC_TYPE_ONLY` → `type_index`.
  2. Nếu tìm thấy và `blocks[type_index + 1]` tồn tại, `kind == "paragraph"`,
     text khác rỗng → đó là block tên văn bản, `find_title` trả về chỉ số
     `type_index + 1`.
  3. Nếu không tìm thấy `type_index`, hoặc không có block hợp lệ ngay sau
     nó → trả `None`.

  **Xác nhận theo yêu cầu người dùng (2026-09-17)**: heading `#` chỉ chứa
  **tên/nội dung chính**, **không kèm tiền tố loại văn bản** — block ở
  `type_index` (`"LUẬT"`/`"BỘ LUẬT"`/`"NGHỊ ĐỊNH"`...) vẫn ở lại trong
  `before` (render qua Groq như văn bản thường, giữ bold), không gộp vào
  heading. Áp dụng **đồng nhất** cho mọi loại văn bản — không có nhánh xử
  lý riêng giữa Luật/Bộ luật và Nghị định/Thông tư. Đã verify trên toàn bộ
  6 file thật:
  - `Luật bảo hiểm xã hội.docx`, `Luật bảo hiểm y tế.docx`, `Luật thuế thu
    nhập cá nhân.docx`: `type_index` là block `"LUẬT"` → heading tương ứng
    `# BẢO HIỂM XÃ HỘI`, `# BẢO HIỂM Y TẾ`, `# THUẾ THU NHẬP CÁ NHÂN`.
  - `Văn bản hợp nhất bộ luật lao động.docx`: `type_index` là block
    `"BỘ LUẬT"` → heading `# LAO ĐỘNG`.
  - `Quy định mức lương tối thiểu.docx`, `Điều kiện lao động và quan hệ lao
    động.docx`: `type_index` là block `"NGHỊ ĐỊNH"` → heading tương ứng
    `# QUY ĐỊNH MỨC LƯƠNG TỐI THIỂU ĐỐI VỚI NGƯỜI LAO ĐỘNG LÀM VIỆC THEO HỢP
    ĐỒNG LAO ĐỘNG` (khớp nguyên ví dụ mục 2), `# QUY ĐỊNH CHI TIẾT VÀ HƯỚNG
    DẪN THI HÀNH MỘT SỐ ĐIỀU CỦA BỘ LUẬT LAO ĐỘNG...`.

  **Không tìm thấy** (không có block nào khớp `RE_DOC_TYPE_ONLY` trong
  front matter, hoặc không có block hợp lệ ngay sau nó) → không tạo
  heading, toàn bộ front matter render nguyên khối qua Groq như văn bản
  thường (hành vi như bản trước 2026-09-17), phát `QcWarning` mới
  `frontmatter_title_not_found` (không fail file).

  **Ghép output front matter khi tìm thấy tên văn bản**: `frontmatter.py`
  cắt `front_blocks` tại chỉ số block tên văn bản thành 3 phần — `before`
  (có thể rỗng), block tên văn bản, `after` (có thể rỗng). `before` và
  `after` mỗi phần đi qua `chunk_blocks_for_llm` + Groq **độc lập** như
  trước (mục 1.2 — chỉ khác là block tên văn bản không còn nằm trong text
  gửi Groq); prompt Groq (mục dưới) vẫn giữ chỉ dẫn "không tự thêm heading"
  cho 2 phần này để phòng Groq tự ý thêm `#` khác. Kết quả cuối:
  `[markdown(before) nếu có] + "\n\n# " + text(block tên văn bản) + "\n\n" +
  [markdown(after) nếu có]`. Lỗi convert `before` hoặc `after` vẫn theo quy
  tắc cũ mục dưới (1 chunk lỗi → bỏ qua toàn bộ phần đó) — dòng heading vẫn
  luôn được chèn nếu tìm thấy, kể cả khi `before`/`after` lỗi và bị bỏ qua.
- **Không có schema, không có "field bắt buộc"** — khác thiết kế trước đó
  (không còn khái niệm `so_hieu`/`loai_van_ban`/... là field riêng). Toàn bộ
  thông tin này vẫn tồn tại **dưới dạng text** trong phần front matter
  markdown, không bị mất, chỉ không được tách thành field có cấu trúc.
  **Ngoại lệ duy nhất**: dòng tên văn bản (heading `#`, xem trên) được xác
  định deterministic để làm **điểm neo cấu trúc** cho `chunking/` — đây
  không phải field trích xuất theo nghĩa schema cũ (không có kiểu dữ liệu
  riêng, không validate nội dung, chỉ là 1 dòng text được chọn ra và giữ
  nguyên văn); không có field `so_hieu`/`ngay_ban_hanh`/... nào khác được
  tách theo cách này.
- **Không có regex baseline/fallback** (đã xác nhận 2026-09-14/16). Khi Groq
  lỗi/timeout (áp dụng cho từng chunk, xem mục 1.2): bỏ qua **toàn bộ** phần
  front matter hoặc back matter tương ứng (không render, không ghép chunk dở
  dang), phát QC warning, **không làm fail file**. Với front matter, đây là
  mất mát nội dung thật sự (không phải chỉ thiếu field) — chấp nhận theo yêu
  cầu đơn giản hoá của người dùng, không thiết kế cơ chế bù đắp nào khác.

**Ghép output cuối cùng (`pipeline.py`/`emitter.py`):**

```
[front matter markdown từ Groq (nối các chunk), nếu có]

[nội dung ở giữa — pipeline hiện có sinh ra, không đổi]

[back matter markdown từ Groq (nối các chunk), nếu có, ngăn cách bằng dòng `---`]
```

Không có YAML, không có heading level gán riêng cho front/back matter —
đây là **văn bản thường** nối trực tiếp vào file `.md`.

**Ranh giới rõ ràng:** việc **xác định biên cấu trúc** (heading
Phần/Chương/.../Khoản, phân loại bảng chữ ký) vẫn giữ nguyên 100%
deterministic bằng regex/heuristic hiện có — Groq **chỉ** chuyển đổi nội
dung hai vùng đã xác định biên sẵn sang markdown, không tham gia phân tích
cấu trúc thân văn bản.

**Lưu ý quan trọng — marker `[n]` trong vùng nội dung ở giữa:** bản DOCX gốc
thường có marker chú thích dạng `[n]` dính liền ngay trong text của vùng nội
dung ở giữa (vd. `"1.[2] Bảo hiểm y tế là..."`, `"Điều 7a. ...Xã hội[16]"`).
Việc **bỏ khôi phục inline chú thích** (mục trên) chỉ có nghĩa là không còn
chèn *nội dung* chú thích trở lại vị trí đó — **không** có nghĩa là được để
nguyên marker `[n]` trong text. Marker này vẫn phải được loại bỏ khỏi text
**trước khi** áp regex heading (mục 3) và trước khi render, như hành vi cũ
(`strip_markers`/`strip_all` trong `footnotes.py` trước đây) — nếu không,
marker dính liền sau số Khoản/Điều sẽ phá regex heading (`RE_KHOAN` yêu cầu
khoảng trắng ngay sau dấu chấm) và làm mất heading thật sự. Đây là bước làm
sạch text **độc lập** với thiết kế back matter mới, thuộc phạm vi "nội dung
ở giữa — pipeline cũ, không đổi" — vẫn phải giữ lại khi implement, không bị
xoá theo `footnotes.py`. Không áp dụng bước này cho text gửi vào Groq ở
back matter: marker `[n]` ở đó (vd. `"[1] Luật Công nghiệp..."`) là số thứ
tự chú thích thật, phải giữ nguyên.

### 1.2. Chunking & rate limiting cho Groq free tier

**Vấn đề đo được thực tế trên corpus** (2026-09-16, ước lượng token bằng
`len(text) // 4`, đo trên chính hàm xác định biên hiện có):

| File                                | Front matter (ước lượng) | Back matter (ước lượng) |
| ----------------------------------- | ---------------------------- | --------------------------- |
| Luật bảo hiểm xã hội           | ~239 token                   | ~1,050 token                |
| Luật bảo hiểm y tế              | ~434 token                   | **~11,671 token**     |
| Luật thuế thu nhập cá nhân     | ~207 token                   | ~509 token                  |
| Quy định mức lương tối thiểu | ~149 token                   | **~5,664 token**      |
| Văn bản hợp nhất BLLĐ          | ~212 token                   | ~787 token                  |
| Điều kiện lao động             | ~218 token                   | ~59 token                   |

Free tier Groq cho `openai/gpt-oss-120b`: **RPM 30, RPD 1.000, TPM 8.000, TPD
200.000**. Front matter luôn nhỏ, không đáng lo. Back matter có thể **tự nó
vượt TPM=8.000** (vd. ~11.671 token) — không thể giải quyết chỉ bằng cách chờ
giữa các lần gọi, bắt buộc phải **chia nhỏ (chunk) trước khi gửi**.

**Chiến lược:**

1. **Chunk theo ranh giới `Block`, không cắt giữa đoạn văn/câu**: front
   matter/back matter (list `Block`) được dồn tuần tự thành từng chunk, dừng
   lại khi ước lượng token của chunk hiện tại vượt `CHUNK_TOKEN_LIMIT = 1500` (đã hạ từ đề xuất ban đầu 2500 xuống mức an toàn hơn theo yêu cầu
   người dùng — tổng ước tính mỗi request ~3.000 token kể cả output, có
   margin rộng dưới TPM=8.000). Không bao giờ cắt giữa 1 block.
2. **Ước lượng token trước khi gửi bằng heuristic ký tự**, không cần thêm
   dependency: `estimated_tokens = len(text) // 2.5` — hệ số bảo thủ hơn mức
   phổ biến 4 ký tự/token của tiếng Anh, vì tiếng Việt có dấu thường tách
   nhiều subword token hơn. Heuristic này **chỉ dùng để quyết định ranh giới
   chunk trước khi gửi**, không dùng để track budget (xem bước 3).
3. **Đọc token thật từ response để cập nhật rate limiter**: sau mỗi lần gọi
   Groq thành công, đọc `response.usage.total_tokens` (Groq trả về, tương
   thích OpenAI API) — số **thật** (input + output) của request đó — dùng số
   này để cập nhật sliding-window tracker cho các request tiếp theo (chính
   xác hơn heuristic).
4. **Sliding-window rate limiter, in-process, đơn luồng** (khớp thiết kế
   tuần tự sẵn có mục 5): giữ 1 deque các `(timestamp, tokens)` trong 60 giây
   gần nhất, dùng chung cho toàn bộ `convert_directory()` (không reset theo
   từng file). Trước mỗi request: dọn entry cũ hơn 60s; nếu tổng token trong
   cửa sổ + ước lượng token chunk sắp gửi > `TPM_SAFE_LIMIT` (90% × 8.000 =
   7.200) → `time.sleep()` tới khi entry cũ nhất hết hạn khỏi cửa sổ; tương
   tự đếm số request trong cửa sổ so với `RPM_SAFE_LIMIT` (90% × 30 = 27).
   RPD/TPD (1.000 request / 200.000 token mỗi ngày) không cần tracker riêng
   — với corpus hiện tại (6 file, chunk hoá tối đa ~15-20 request) không
   chạm ngưỡng ngày; nếu corpus tăng quy mô lớn, đây là chỗ đầu tiên cần xét
   lại (giữ tinh thần "không over-engineering" đã có ở mục 5).
5. **Ghép kết quả theo thứ tự chunk**: mỗi chunk gọi
   `llm_client.convert_to_markdown()` độc lập (mô hình chỉ cần convert
   nguyên văn, không cần ngữ cảnh toàn văn bản để làm đúng việc "giữ nguyên
   nội dung/định dạng"), nối các đoạn markdown trả về theo đúng thứ tự
   chunk, cách nhau 1 dòng trống.
6. **Lỗi 1 chunk → toàn bộ front/back matter đó coi như lỗi**: nếu bất kỳ
   chunk nào hết `max_retries` mà vẫn lỗi, toàn bộ phần front matter hoặc
   back matter tương ứng bị bỏ qua (không ghép phần dở dang), phát QC
   warning như mục 1.1 — tránh render 1 văn bản half-converted gây hiểu lầm.

**Ngoài phạm vi (chủ động không làm):**

- Idempotent tracking qua Postgres — mỗi lần chạy xử lý lại toàn bộ input,
  không có DB theo dõi trạng thái.
- Quy trình test/golden file — đã có agent CI/CD riêng đảm nhiệm theo
  coding-convention, spec này không mô tả.
- Chunking, embedding — thuộc package `chunking/`, `embedding/` riêng
  (không liên quan tới "chunk" ở mục 1.2, vốn chỉ là chia nhỏ request gọi
  LLM, không phải chunk cho retrieval). Cách `chunking/` xử lý khối
  front/back matter (văn bản thường, không có heading level) là quyết định
  của `chunking_spec.md`, không mô tả ở đây.
- Xử lý ảnh/scan nhúng trong `.docx` (OCR...) — xác nhận với người dùng
  (2026-09-14): toàn bộ 6 file `data/raw/*.docx` hiện tại là text gõ trực
  tiếp, không có file scan/ảnh nhúng nào.
- QC thủ công bắt buộc trước khi ghi output — output đi thẳng vào
  `data/markdown/`, chỉ dựa vào QC warning tự động (mục 7) để phát hiện bất
  thường.
- Tự host mô hình LLM (vd. vLLM) — dùng Groq Cloud API free tier
  (cloud-hosted).
- Trích field có cấu trúc (`so_hieu`, `loai_van_ban`, `ten_van_ban`,
  `ngay_ban_hanh`, `ngay_hieu_luc`...) từ front matter — đã xác nhận với
  người dùng (2026-09-16): bỏ hoàn toàn, không sinh YAML metadata. Nếu
  package khác (`chunking/`, `embedding/`) cần các field này, đó là việc của
  spec khác, không phải formatting.
- Regex baseline/fallback cho front matter, back matter — bỏ hoàn toàn.
- Khôi phục chú thích `[n]` về vị trí gốc trong thân văn bản (inline
  reinsertion) — bỏ hoàn toàn, back matter luôn là 1 khối ở cuối văn bản.
- Retry/backoff dựa trên header rate-limit Groq trả về (vd.
  `x-ratelimit-remaining-tokens`) — dùng sliding-window tự tính ở mục 1.2 là
  đủ cho quy mô hiện tại, không cần đọc thêm response header để tối ưu.

## 2. Input & Output

- **Input**: `data/raw/*.docx` — tên file tiếng Việt có dấu và khoảng trắng.
- **Output**: `data/markdown/*.md`, ánh xạ 1-1 theo tên file (đổi đuôi
  `.docx` → `.md`). Ghi atomic (file tạm + rename) để không để lại file dở
  dang nếu tiến trình bị ngắt giữa chừng.
- Mỗi file output là markdown thuần, ghép: front matter (Groq) + nội dung
  giữa (pipeline hiện có) + back matter (Groq, nếu có). Ví dụ (rút gọn từ
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

  # QUY ĐỊNH MỨC LƯƠNG TỐI THIỂU ĐỐI VỚI NGƯỜI LAO ĐỘNG LÀM VIỆC THEO HỢP ĐỒNG LAO ĐỘNG

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

  "NGHỊ ĐỊNH" (tên loại văn bản) trong ví dụ trên **không** dùng heading
  markdown — chỉ in đậm, như văn bản gốc trình bày. Riêng tên đầy đủ văn bản
  ("QUY ĐỊNH MỨC LƯƠNG TỐI THIỂU...") dùng heading `#` thật, do `frontmatter.py`
  **tự xác định và chèn deterministic** (mục 1.1) — không phải Groq quyết
  định. Đây là **ngoại lệ duy nhất**, mọi heading level 1-5 khác (mục 3)
  chỉ dành cho phần nội dung ở giữa, do pipeline sinh. Khối cuối (sau
  `---`) chỉ xuất hiện khi văn bản có back matter.

## 3. Heading mapping (phần lõi — áp dụng cho nội dung ở giữa)

Lưu ý: heading `#` của **tên văn bản** trong front matter (mục 1.1) không
thuộc bảng mapping dưới đây — `chunking/patterns.py` phân biệt 2 loại heading
`#` này bằng regex cấu trúc chuyên biệt cho Phần/Phụ Lục, không dựa vào việc
"gặp `#` đầu tiên trong file" (xem `chunking_spec.md` mục 4.6).

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

| Việc                                                                                               | Công cụ                                                                                                                                                                                                                                                                                                                                 |
| --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Đọc`.docx`                                                                                      | `python-docx`                                                                                                                                                                                                                                                                                                                           |
| Data models trao đổi giữa module (kết quả cuối, warnings)                                     | `pydantic` v2 `BaseModel`                                                                                                                                                                                                                                                                                                             |
| **[Mới]** Chuyển front matter/back matter từ DOCX sang markdown bằng Groq (mục 1.1, 1.2) | SDK chính thức`groq` (OpenAI-compatible chat completions) — model `openai/gpt-oss-120b`, free tier — text generation thuần (`client.chat.completions.create`, đọc `.choices[0].message.content` và `.usage.total_tokens`) — **không cần `instructor`**, đây là tác vụ convert text, không trích field |
| CLI                                                                                                 | `typer`, đặt tại `tools/format_documents.py` theo coding-convention (không đặt CLI trong package `formatting/`)                                                                                                                                                                                                               |
| Format/lint                                                                                         | `ruff`                                                                                                                                                                                                                                                                                                                                  |

**[Mới]** Cấu hình gọi Groq (`model_name="openai/gpt-oss-120b"`,
`max_retries=2`, `timeout_seconds=30`, cộng các hằng số rate-limit mục 1.2:
`chunk_token_limit=1500`, `tpm_limit=8000`, `rpm_limit=30`) trong
`LLMSettings` (`src/production_legal_qa_rag/config.py`). Đọc key từ biến môi
trường `GROQ_API_KEY` trong `.env`. Xoá `google-genai` khỏi
`pyproject.toml` (không dùng nữa), thêm lại `groq`. Không cần `instructor`
vì không còn structured output ở bất kỳ bước nào trong `formatting/`.

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
  đơn giản hơn hẳn multiprocessing; cũng là điều kiện cần để sliding-window
  rate limiter (mục 1.2) hoạt động đúng bằng 1 state đơn giản trong tiến
  trình, không cần khoá/đồng bộ giữa nhiều worker.
- **Lỗi 1 file không chặn các file khác** — bắt exception theo từng file, in
  ra lỗi kèm tên file, tiếp tục file kế tiếp; tổng kết cuối cùng liệt kê file
  nào lỗi.
- **QC warnings** không bao giờ làm fail file — in summary đếm theo code ở
  cuối lần chạy.
- **[Mới]** **Groq lỗi/timeout không làm fail file, và không có fallback**
  — `llm_client.convert_to_markdown` bắt lỗi cho từng chunk, trả `None` sau
  `max_retries=2` lần thử hoặc `timeout_seconds=30`. Khi 1 chunk lỗi: toàn
  bộ front matter hoặc back matter tương ứng bị bỏ qua (không render vào
  output, không ghép phần dở dang — mục 1.2), phát QC warning tương ứng
  (mục 6, mục 7) tại chỗ — không để exception thoát lên tầng
  `convert_directory` per-file catch, không thử regex nào khác.
- **[Mới]** **Rate limiter dùng chung xuyên suốt cả lần chạy** — sliding
  window (mục 1.2) không reset theo từng file, vì free tier tính theo toàn
  tài khoản Groq, không theo từng file/lệnh gọi riêng lẻ.

## 6. Cấu trúc module trong `src/production_legal_qa_rag/formatting/`

| Module             | Trách nhiệm                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `patterns.py`    | Regex nhận diện cấu trúc thân văn bản (Phần/Chương/Mục/Điều/Khoản/Điểm) +`sort_key()` cho số hiệu có hậu tố chữ. Dùng để xác định **biên** front matter (mục 1.1). **Giữ nguyên** regex/hàm strip marker `[n]` khỏi text — vẫn cần chạy trên mọi block thuộc vùng nội dung ở giữa **trước khi** áp regex heading, xem "Lưu ý quan trọng" ở mục 1.1. **[Mới]** `RE_DOC_TYPE_ONLY` (regex khớp trọn dòng cho danh sách tên loại văn bản: LUẬT, BỘ LUẬT, NGHỊ ĐỊNH, NGHỊ QUYẾT, THÔNG TƯ, THÔNG TƯ LIÊN TỊCH, QUYẾT ĐỊNH, PHÁP LỆNH, CHỈ THỊ) — dùng bởi `frontmatter.find_title` (mục 1.1) để định vị dòng loại văn bản; block tên văn bản là block paragraph ngay sau đó, không cần điều kiện bold/viết hoa riêng (đã verify trên corpus thật — 2/6 file có dòng tên không bold).                                                                                                                                                                                                                                                                                                                                                                                               |
| `docx_reader.py` | Đọc`.docx` bằng `python-docx`, trích xuất `Block` (paragraph/table) theo đúng thứ tự xuất hiện, giữ style/bold/nghiêng làm tín hiệu phụ — dùng cho cả heading mapping (mục 3) lẫn serialize block cho Groq (mục 1.1). **[Mới]** Thêm `chunk_blocks_for_llm(blocks, token_limit) -> list[list[Block]]` — dồn block tuần tự thành chunk theo heuristic ước lượng token mục 1.2, ranh giới luôn trùng ranh giới block.                                                                                                                                                                                                                                                                                                                                                     |
| `tables.py`      | Nhận diện loại bảng theo nội dung cho**phần nội dung ở giữa** (đính kèm, bảng dữ liệu) và render bảng dữ liệu sang markdown/HTML — không đổi. Riêng phân loại bảng **chữ ký** vẫn giữ, dùng để xác định biên back matter (mục 1.1); **bỏ** phân loại "quốc hiệu" — bảng đó giờ nằm trong vùng front matter, Groq xử lý nguyên khối cùng các block khác, không cần tables.py can thiệp riêng.                                                                                                                                                                                                                                                                                                                                                 |
| `frontmatter.py` | Xác định biên front matter (block trước heading đầu tiên, dùng`patterns.py`). **[Mới]** `find_title(blocks)` — xác định deterministic block tên văn bản: tìm block khớp `RE_DOC_TYPE_ONLY` (dòng loại văn bản) rồi lấy block paragraph ngay sau đó, không cần bold/viết hoa (mục 1.1); nếu tìm thấy, cắt `blocks` thành `before` (gồm cả dòng loại văn bản)/tên văn bản/`after`, convert `before` và `after` **độc lập** qua chunk (`docx_reader.chunk_blocks_for_llm`) + Groq (`llm_client.convert_to_markdown()`, giữ tín hiệu bold/nghiêng khi serialize) rồi chèn `# <nguyên văn>` xen giữa; nếu không tìm thấy → convert nguyên khối `blocks` như cũ, phát `QcWarning` (`frontmatter_title_not_found`). Bất kỳ chunk Groq nào lỗi → bỏ qua toàn bộ phần front matter tương ứng (`before` hoặc `after`), phát `QcWarning` (`llm_frontmatter_conversion_failed`) — không ảnh hưởng dòng heading đã chèn. Không còn schema/field nào (đã bỏ `FrontMatter`, `extract_quoc_hieu`, `_find_ten_van_ban`, `_find_loai_van_ban`, `_find_ngay_hieu_luc`) — `find_title` không phải phục hồi các hàm này, chỉ tìm 1 dòng heading bằng vị trí tương đối so với dòng loại văn bản.                                                                                                                                          |
| `backmatter.py`  | **[Đổi tên từ `footnotes.py`, thiết kế lại]** Xác định biên back matter (block sau bảng chữ ký cuối cùng, dùng `tables.py`); nếu không còn block nào → không có back matter, dừng, không gọi Groq. Nếu có → chia chunk, gọi `llm_client.convert_to_markdown()` cho từng chunk, nối kết quả theo thứ tự (mục 1.2), trả về text markdown để `pipeline.py` append vào cuối output (ngăn cách `---`, mục 2). Không còn khớp marker `[n]` hay khôi phục inline vào Khoản/Điều — các hàm cũ (`find_region_start`, `parse_region`, `strip_markers`, `strip_all`, `render_blockquote`) đã xoá. Bất kỳ chunk nào lỗi → bỏ qua toàn bộ back matter, phát `QcWarning` (`llm_backmatter_conversion_failed`).                     |
| `llm_client.py`  | Client gọi Groq API bằng SDK`groq`. Hàm `convert_to_markdown(prompt: str, *, max_retries) -> str \| None` — text generation thuần, không structured output; trả `None` khi hết `max_retries` lần thử hoặc quá `timeout_seconds` (không raise ra ngoài, caller tự quyết bỏ qua phần tương ứng). Đọc `model_name`/`max_retries`/`timeout_seconds`/`groq_api_key` từ `LLMSettings` (`config.py`, mục 4). **[Mới]** Sở hữu **sliding-window rate limiter** (mục 1.2): trước mỗi lệnh gọi, ước lượng token của `prompt` (`len(prompt) // 2.5`), chờ (`time.sleep`) nếu vượt `TPM_SAFE_LIMIT`/`RPM_SAFE_LIMIT`; sau khi gọi thành công, đọc `response.usage.total_tokens` để cập nhật state tracker (thay cho số ước lượng). |
| `emitter.py`     | Duyệt`Block` theo thứ tự **trong vùng nội dung ở giữa**, strip marker `[n]` khỏi text (dùng hàm ở `patterns.py`, giữ nguyên hành vi cũ), áp heading mapping ở mục 3, sinh markdown thân văn bản. Không còn bước chèn inline **nội dung** chú thích vào giữa thân văn bản (khác với việc strip marker, vẫn giữ).                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `validator.py`   | `validate()` kiểm tra heading-level-skip và các bất thường khác trong nội dung ở giữa, trả về danh sách `QcWarning`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `models.py`      | Pydantic models dùng làm input/output giữa các module trên:`FormattingResult` (markdown: str, warnings: list[QcWarning]), `QcWarning`. **Bỏ** `FrontMatter`, `FrontMatterExtraction`, `BackMatterExtraction` (không còn field có cấu trúc). `QcWarningCode` gồm `llm_frontmatter_conversion_failed`, `llm_backmatter_conversion_failed`, **[Mới]** `frontmatter_title_not_found` (không tìm được block tên văn bản theo heuristic mục 1.1 — front matter vẫn render, không có heading).                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `pipeline.py`    | `convert_docx_to_markdown(path) -> FormattingResult`: gọi `frontmatter.py` lấy front matter markdown (có thể rỗng), lấy nội dung giữa từ `emitter.py` (không đổi), gọi `backmatter.py` lấy back matter markdown (có thể không có), ghép 3 phần theo mục 1.1/2. `convert_directory(raw_dir, out_dir)` — orchestration mức trên, atomic write, gom summary, khởi tạo rate limiter dùng chung (mục 1.2, 5) cho cả lần chạy.                                                                                                                                                                                                                                                                                                                                                            |

`tools/format_documents.py` (ngoài package `formatting/`, theo
coding-convention) chỉ là Typer CLI mỏng gọi `pipeline.convert_directory()`.

## 7. Tiêu chí hoàn thành

- Chạy CLI trên toàn bộ `data/raw/` sinh ra `data/markdown/*.md` đúng heading
  mapping ở mục 3 cho phần nội dung ở giữa, khớp với 6 file hiện có trong
  `data/markdown/` (dùng làm tham chiếu, không phải golden-file chính thức).
- Chạy lại nhiều lần trên cùng input phải ra output giống hệt (deterministic,
  byte-for-byte) cho **phần nội dung ở giữa** (do `emitter.py`/`patterns.py`/
  `tables.py` quyết định — không đổi). **Không áp dụng** byte-for-byte cho
  phần front matter/back matter render qua Groq — có thể diễn đạt khác nhau
  nhẹ giữa các lần chạy dù cùng input. **Ngoại lệ**: dòng heading `#` tên
  văn bản trong front matter (mục 1.1) — do `frontmatter.find_title` chèn
  deterministic từ text gốc, không qua Groq — phải byte-for-byte giống hệt
  giữa các lần chạy.
- Dòng heading `#` tên văn bản phải khớp **nguyên văn** đoạn text gốc trong
  DOCX (không bị Groq diễn đạt lại, vì không qua Groq — mục 1.1), và
  **không kèm tiền tố loại văn bản** (dòng "LUẬT"/"BỘ LUẬT"/"NGHỊ ĐỊNH"...
  vẫn ở lại phần văn bản thường phía trước, không lẫn vào heading) — kiểm
  tra thủ công trên toàn bộ 6 file khi implement, đối chiếu đúng kết quả đã
  verify trong mục 1.1 (`# BẢO HIỂM XÃ HỘI`, `# LAO ĐỘNG`,
  `# QUY ĐỊNH MỨC LƯƠNG TỐI THIỂU...`, v.v.) — không phân biệt nhánh xử lý
  giữa Luật/Bộ luật và Nghị định/Thông tư, cùng 1 heuristic áp dụng đồng
  nhất.
- Khi 1 chunk gọi Groq lỗi (hết `max_retries=2` lần thử, timeout
  `timeout_seconds=30s`): bỏ qua **toàn bộ** front matter hoặc back matter
  tương ứng (không render vào output, không ghép phần dở dang), **không
  fail file**, và phát `QcWarning` tương ứng
  (`llm_frontmatter_conversion_failed` hoặc `llm_backmatter_conversion_failed`).
  Không có fallback regex nào được thử.
- Văn bản không có back matter (không có block nào sau bảng chữ ký) → không
  gọi Groq cho phần này, không có khối back matter trong output, không phát
  QC warning — đây là trạng thái hợp lệ.
- Chạy CLI trên toàn bộ corpus **không** bị Groq trả lỗi 429 (rate limit) —
  sliding-window rate limiter (mục 1.2) phải giữ tổng token/request trong
  mọi cửa sổ 60 giây dưới `TPM_SAFE_LIMIT`/`RPM_SAFE_LIMIT`. Riêng
  `Luật bảo hiểm y tế.docx` (back matter ~11.671 token ước lượng) là ca kiểm
  thử chính cho việc chunk hoạt động đúng (phải tách thành nhiều chunk, mỗi
  chunk dưới `CHUNK_TOKEN_LIMIT=1500`).
- Front matter/back matter markdown do Groq sinh giữ đúng nội dung gốc
  (không tóm tắt/bịa thêm) — kiểm tra thủ công trên 6 file khi implement,
  không có cơ chế tự động so khớp (đã bỏ regex baseline để so sánh).
- Không phát sinh QC warning code mới ngoài các warning đã biết là hợp lệ
  (theo ghi nhận trước đây khi rà soát bản `loader.py` cũ, cộng với 3
  warning code mới ở mục 6: `llm_frontmatter_conversion_failed`,
  `llm_backmatter_conversion_failed`, `frontmatter_title_not_found`).
- 1 file lỗi không làm dừng toàn bộ batch; CLI kết thúc với summary rõ ràng
  file nào thành công/lỗi.
