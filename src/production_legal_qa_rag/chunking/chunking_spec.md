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
- Sinh breadcrumb cho từng chunk; với Khoản có Điểm bị cắt nhỏ, lặp lại câu
  dẫn vào `content` của mọi chunk con để giữ ngữ cảnh (mục 4.5), và giữ lại
  frontmatter (trước heading đầu tiên) lẫn backmatter (chú thích sửa đổi dời
  cuối file) thành chunk riêng thay vì để lẫn/mất (mục 4.6). **[CẬP NHẬT
  2026-09-14]** — xem mục 12 "Lịch sử cập nhật" để biết chi tiết những gì
  thay đổi so với bản spec gốc.
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
- **Output**: `data/chunks/*.json`, ánh xạ 1-1 theo tên file nguồn (đổi đuôi
  `.md` → `.json`), mỗi file là 1 mảng JSON các `Chunk`. Ghi atomic (file tạm +
  rename), giống `formatting/`. Đây là định dạng trung gian — bước
  `embedding/` sẽ đọc lại, sinh vector rồi upsert từng `Chunk` lên **Pinecone**
  (id, vector, metadata), chunking không tự đẩy lên Pinecone.
- Mỗi `Chunk` gồm các field (`chunking/models.py`):

  | Field                          | Kiểu          | Ý nghĩa                                                                                                                                                 |
  | ------------------------------ | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `chunk_id`                     | `str`          | Deterministic, sinh từ `so_hieu` + đường dẫn breadcrumb + chỉ số phần (nếu bị cắt).                                                                     |
  | `source_document`              | `str`          | `so_hieu` (fallback: tên file) — để phân biệt chunk giữa các văn bản khác nhau.                                                                         |
  | `breadcrumb`                   | `str`          | Viện dẫn đầy đủ, metadata hiển thị/lọc — **không đưa vào chuỗi embed**.                                                                                |
  | `content`                      | `str`          | Nội dung thuần (nguyên văn tiếng Việt, chưa word-segment) — chuỗi thực sự đem đi embed. Nếu Khoản có bảng, đã nối thêm `standardization_table` (mục 5). |
  | `token_count`                  | `int`          | Số token của `content` theo tokenizer của embedding model (mục 6).                                                                                       |
  | `has_table`                    | `bool`         | Khoản gốc có chứa bảng dữ liệu hay không (mục 5).                                                                                                        |
  | `raw_table`                    | `str \| None`  | Nguyên văn bảng markdown gốc, chỉ có khi `has_table=True` — gửi cho LLM ở bước generation, không đưa vào chuỗi embed (mục 5).                            |
  | `standardization_table`        | `str \| None`  | Bảng đã chuyển thành text bằng cách nối theo hàng, chỉ có khi `has_table=True` (mục 5).                                                                  |
  | `is_split`                     | `bool`         | Khoản gốc có bị cắt thành nhiều chunk hay không. Luôn `False` nếu `has_table=True` (mục 5).                                                              |
  | `split_index` / `split_total`  | `int \| None`  | Vị trí/tổng số phần nếu `is_split=True`.                                                                                                                 |
  | `loai_van_ban` **[MỚI]**       | `str \| None`  | Loại văn bản (vd. "Luật", "Nghị định") — đọc trực tiếp từ front matter YAML (`formatting/frontmatter.py` đã trích sẵn), không phân tích thêm.            |
  | `co_quan_ban_hanh` **[MỚI]**   | `str \| None`  | Cơ quan ban hành — đọc từ front matter YAML.                                                                                                             |
  | `ngay_ban_hanh` **[MỚI]**      | `str \| None`  | Ngày ban hành (ISO `YYYY-MM-DD`) — đọc từ front matter YAML.                                                                                             |
  | `ngay_hieu_luc` **[MỚI]**      | `str \| None`  | Ngày hiệu lực (ISO `YYYY-MM-DD`) — đọc từ front matter YAML.                                                                                             |

  **[MỚI 2026-09-14]** 4 field cuối lấy nguyên giá trị từ front matter (giống
  cách `source_document` đã đọc `so_hieu`/`ten_van_ban`) — chunking chỉ đọc
  lại, không tự dò/parse thêm. Đây đều là metadata phục vụ hiển thị/lọc ở
  bước sau (không đưa vào chuỗi embed), nên giữ `str | None` kể cả với
  `ngay_hieu_luc` (dù `frontmatter.py` coi là bắt buộc, vẫn có thể `None` khi
  phát sinh `QcWarning`, xem `formatting/frontmatter.py::build_frontmatter`)
  — chunking không chặn pipeline vì thiếu field optional.

  **[ĐÃ XOÁ 2026-09-14]** Field `negation_note` (`str | None`) đã bị xoá khỏi
  `Chunk` — câu dẫn/câu phủ định giờ luôn nằm sẵn trong `content` (mục 4.5),
  không cần field riêng để lặp lại.

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

**[CẬP NHẬT 2026-09-14]** Breadcrumb **không** lặp lại câu dẫn/câu phủ định
của Khoản — câu đó đã luôn có mặt nguyên văn trong `content` của mọi chunk
con rồi (mục 4.5), nên nhắc lại ở breadcrumb là dư thừa. (Trước đây breadcrumb
có nối thêm câu phủ định khi phát hiện; nay đã bỏ, xem mục 12.)

Ví dụ breadcrumb 1 chunk con:

```
Luật Bảo hiểm xã hội (41/2024/QH15) - Chương II - Điều 3 - Khoản 4 - Điểm a, b (phần 1/2)
```

`ten_van_ban`/`so_hieu` lấy trực tiếp từ front matter YAML của file markdown
(đã có sẵn từ `formatting/frontmatter.py`, chunking chỉ đọc lại, không phân
tích thêm).

**[MỚI 2026-09-14]** Trường hợp đặc biệt: chunk ứng với frontmatter/backmatter
(mục 4.6) không có Phần/Chương/Mục/Điều/Khoản nào bao ngoài — xem bảng
`breadcrumb_prefix` ở mục 4.6 (2 vùng dùng breadcrumb khác nhau để tránh
trùng `chunk_id`).

## 4. Quy tắc cắt Khoản

### 4.1. Trường hợp cơ bản

Trước tiên kiểm tra Khoản có chứa bảng dữ liệu hay không — nếu có, áp dụng
thẳng mục 5 (giữ nguyên, không cắt, bỏ qua toàn bộ mục 4.2–4.5). Nếu không có
bảng: tính `token_count` của toàn bộ nội dung Khoản (mục 6). Nếu
`token_count <= MAX_TOKENS` → **giữ nguyên, 1 Khoản = 1 chunk**, bất kể Khoản
dài hay ngắn (kể cả chỉ 1 câu).

### 4.2. Tách câu dẫn (đơn vị #0) và danh sách đơn vị cần đóng gói

**[CẬP NHẬT 2026-09-14]** — trước đây mục này chỉ liệt kê "đơn vị #0" như 1
đơn vị bình thường trong pool ghép chung với các Điểm; nay đơn vị #0 (đổi tên
thành "câu dẫn") được tách riêng hẳn khỏi pool ngay từ bước này, xem mục 12.

Khi `token_count > MAX_TOKENS`:

- Nếu Khoản có các Điểm gắn nhãn (`a)`, `b)`, `c)`...):
  - **Câu dẫn** (nếu có) = đoạn văn bản mở đầu, đứng trước nhãn Điểm đầu
    tiên (ví dụ "Người sau đây khi chết thì tổ chức, cá nhân lo mai táng
    được nhận một lần trợ cấp mai táng:"). Câu dẫn được tách riêng ngay từ
    bước này — **không** đưa vào danh sách đơn vị cần đóng gói ở mục 4.3.
    Câu dẫn luôn được lặp lại nguyên văn vào `content` của **mọi** chunk con
    sinh ra từ Khoản này (mục 4.5), bất kể có chứa từ khoá phủ định/loại trừ
    hay không.
  - Danh sách đơn vị cần đóng gói (mục 4.3) = nội dung từng Điểm, theo đúng
    thứ tự a, b, c... (không gồm câu dẫn).
- Nếu Khoản **không có** Điểm gắn nhãn: không có khái niệm câu dẫn — toàn bộ
  nội dung là 1 khối, tách thẳng bằng câu (coi mỗi câu là 1 đơn vị) — bỏ qua
  tầng Điểm, giữ nguyên như trước.

### 4.3. Thuật toán ghép "cận dưới" (greedy, không overlap ở tầng này)

**[CẬP NHẬT 2026-09-14]** — thêm bước trừ ngân sách cho câu dẫn
(`budget_hiệu_dụng`) và bước ghép câu dẫn vào mọi chunk con ở cuối thuật
toán; trước đây thuật toán chạy thẳng trên `MAX_TOKENS` không trừ hao gì.

Nếu Khoản có câu dẫn (mục 4.2), trừ trước số token của câu dẫn ra khỏi ngân
sách — phần ngân sách còn lại (`budget_hiệu_dụng`) mới dùng để đóng gói các
Điểm, vì câu dẫn sẽ được cộng thêm vào mọi chunk con ở bước cuối (mục 4.5).
Nếu Khoản không có câu dẫn, `budget_hiệu_dụng = MAX_TOKENS`.

Duyệt tuần tự các đơn vị (mục 4.2), gom vào chunk hiện tại miễn tổng token
không vượt `budget_hiệu_dụng`; hễ thêm đơn vị kế tiếp làm vượt thì chốt chunk
hiện tại và mở chunk mới. Mã giả (không phải code thật, chỉ mô tả logic):

```
số_token_câu_dẫn = ĐẾM_TOKEN(câu_dẫn) NẾU có câu dẫn, NGƯỢC LẠI 0
budget_hiệu_dụng = MAX_TOKENS - số_token_câu_dẫn

KHỞI TẠO chunk_đang_gom = rỗng, tổng_token = 0

LẶP QUA từng đơn_vị theo đúng thứ tự trong danh sách đơn vị (mục 4.2):
    số_token_đơn_vị = ĐẾM_TOKEN(đơn_vị)                (mục 6)

    NẾU số_token_đơn_vị > budget_hiệu_dụng:
        # đơn vị này tự nó đã vượt ngân sách, không thể gộp được nữa
        NẾU chunk_đang_gom KHÔNG rỗng: CHỐT chunk_đang_gom thành 1 chunk con
        THỰC HIỆN fallback tách theo câu cho riêng đơn_vị này (mục 4.4)
        RESET chunk_đang_gom = rỗng, tổng_token = 0
        CHUYỂN sang đơn_vị kế tiếp

    NGƯỢC LẠI, NẾU tổng_token + số_token_đơn_vị > budget_hiệu_dụng:
        # thêm đơn vị này vào sẽ vượt ngưỡng → "cận dưới": chốt trước, không vượt
        CHỐT chunk_đang_gom thành 1 chunk con
        chunk_đang_gom = [đơn_vị], tổng_token = số_token_đơn_vị

    NGƯỢC LẠI:
        # còn đủ chỗ, gộp tiếp
        THÊM đơn_vị vào chunk_đang_gom
        tổng_token = tổng_token + số_token_đơn_vị

SAU KHI hết đơn vị: NẾU chunk_đang_gom KHÔNG rỗng, CHỐT nốt thành chunk con
cuối cùng

VỚI MỖI chunk con vừa chốt: NẾU có câu dẫn, GHÉP câu_dẫn vào ĐẦU content của
chunk con đó (mục 4.5) — kể cả chunk con chỉ có đúng 1 chunk duy nhất (Khoản
bị cắt nhưng chỉ sinh ra 1 chunk, hiếm gặp).
```

Không overlap giữa các chunk con ở tầng Điểm — mỗi câu/Điểm chỉ thuộc đúng 1
chunk con (câu dẫn là ngoại lệ có chủ đích, được lặp lại ở mọi chunk, không
tính là overlap giữa các Điểm với nhau).

### 4.4. Đơn vị vượt ngân sách ngay cả một mình (fallback theo câu)

**[CẬP NHẬT 2026-09-14]** — dùng `budget_hiệu_dụng` (mục 4.3) thay vì
`MAX_TOKENS` thẳng, và ghép câu dẫn vào chunk câu sinh ra ở tầng fallback này
(trước đây fallback theo câu không có khái niệm câu dẫn).

Nếu 1 đơn vị (1 Điểm, hoặc đoạn không có Điểm) tự nó đã > `budget_hiệu_dụng`
(mục 4.3): tách tiếp theo câu, áp dụng lại đúng thuật toán "cận dưới" ở mục
4.3 nhưng ở cấp câu và trên cùng `budget_hiệu_dụng` đó, và **cho phép overlap
1 câu cuối** giữa 2 chunk câu liên tiếp (câu cuối của chunk trước lặp lại ở
đầu chunk sau) để giữ mạch ngữ cảnh khi phải cắt sâu tới mức này. Câu dẫn
(nếu có) vẫn được ghép vào đầu content của từng chunk câu sinh ra ở tầng
fallback này, giống mọi chunk con khác (mục 4.3, 4.5).

Tách câu bằng regex đơn giản trên dấu kết câu (`.`, `;`) theo sau bởi
khoảng trắng — chấp nhận sai số nhỏ ở tầng fallback hiếm gặp này, không dùng
thư viện NLP nặng cho việc này.

Trường hợp hiếm: nếu bản thân câu dẫn đã > `MAX_TOKENS` (không còn ngân sách
nào cho bất kỳ Điểm nào) — chấp nhận vượt ngân sách ở chunk ghép câu dẫn +
Điểm đầu tiên, cùng tinh thần "chấp nhận sai số nhỏ" đã nêu ở trên; không cần
thêm cơ chế cắt câu dẫn riêng (chưa gặp trong corpus thực tế).

### 4.5. Câu dẫn — lặp lại vào content của mọi chunk con

**[CẬP NHẬT 2026-09-14]** — mục này trước đây tên "Câu phủ định/loại trừ —
lan truyền vào breadcrumb": chỉ trích câu phủ định (nếu Khoản chứa 1 trong
các từ khoá loại trừ) và chỉ lặp vào `breadcrumb`/field `negation_note`
(nay đã xoá, xem mục 2). Nay tổng quát hoá: **mọi** câu dẫn (có phủ định hay
không) đều lặp vào **`content`** của mọi chunk con — cơ chế `NEGATION_KEYWORDS`
trong `patterns.py` không còn cần thiết và đã bị xoá. Xem mục 12.

Đây là điểm **quan trọng nhất** của module: khi 1 Khoản có Điểm và bị cắt
thành nhiều chunk, mỗi chunk con chỉ chứa nội dung (các) Điểm nó bao phủ —
nhưng câu dẫn đứng trước danh sách Điểm (đơn vị #0, mục 4.2) mang ngữ cảnh
áp dụng cho toàn bộ Khoản (chủ ngữ, điều kiện chung, kể cả phủ định/loại trừ
nếu có). Nếu không lặp lại, chunk con đứng một mình sẽ mất ngữ cảnh đó khi
đem đi embed/hiển thị riêng lẻ.

Vì vậy: câu dẫn (nếu Khoản có) được ghép vào **đầu** `content` của **mọi**
chunk con sinh ra từ Khoản đó (mục 4.3, 4.4) — kể cả chunk chứa chính đơn vị
#0 ở vị trí gốc. Không phân biệt câu dẫn có chứa từ khoá phủ định/loại trừ
hay không — hành vi áp dụng thống nhất cho mọi câu dẫn.

Ví dụ Điều 85 Khoản 1 `Luật bảo hiểm xã hội.md` (câu dẫn "Người sau đây khi
chết thì tổ chức, cá nhân lo mai táng được nhận một lần trợ cấp mai táng:"),
nếu bị cắt theo Điểm, mỗi chunk con có `content` dạng:

```
Người sau đây khi chết thì tổ chức, cá nhân lo mai táng được nhận một lần
trợ cấp mai táng:

a) Đối tượng quy định tại khoản 1 và khoản 2 Điều 2 của Luật này có thời
gian đóng bảo hiểm xã hội bắt buộc từ đủ 12 tháng trở lên;
```

Nếu Khoản **không có** Điểm gắn nhãn (tách thẳng theo câu, mục 4.2), không có
khái niệm câu dẫn riêng biệt — cơ chế này không áp dụng, giữ nguyên hành vi
overlap 1 câu cuối của mục 4.4.

### 4.6. Phần mở đầu và phần cuối văn bản (frontmatter/backmatter)

**[MỚI 2026-09-14]** — toàn bộ mục 4.6 (frontmatter + backmatter +
`RecursiveCharacterTextSplitter`) không tồn tại trong bản spec gốc; văn bản
trước heading đầu tiên bị bỏ qua hoàn toàn, xem mục 12.

Hai vùng nội dung nằm **ngoài** cấu trúc Phần/Chương/Mục/Điều/Khoản, ở hai
đầu file, đều không được phép mất khi chunk hoá:

- **Frontmatter** (phần mở đầu): đoạn văn bản đứng trước heading cấp 1 đầu
  tiên của toàn bộ file (Phần/Chương, hoặc thẳng Điều nếu văn bản không chia
  Phần/Chương) — ví dụ tên loại văn bản ("LUẬT"), tên văn bản
  ("BẢO HIỂM XÃ HỘI"), số hiệu và ngày ban hành, các đoạn "Căn cứ Hiến
  pháp...", và câu "Quốc hội ban hành Luật ...". Đây là nội dung thật của
  văn bản (không phải front matter YAML — front matter đã tách riêng các
  field cấu trúc từ đoạn này, xem `formatting/frontmatter.py`).
- **Backmatter** (phần cuối): vùng chú thích sửa đổi dài bị
  `formatting/footnotes.py::render_blockquote` dời xuống cuối file thay vì
  inline tại chỗ (khi chú thích vượt `FOOTNOTE_INLINE_MAX_CHARS`), đánh dấu
  bằng 1 dòng in đậm không phải heading: `**[n]**` (regex
  `^\*\*\[\d+\]\*\*$`). Từ dòng `**[n]**` đầu tiên xuất hiện sau heading cuối
  cùng của file cho tới hết file là vùng backmatter — toàn bộ phần này trích
  dẫn nguyên văn Điều/Khoản của **luật khác** (vd. Điều 41 Luật Nhà giáo,
  Điều 63 Luật Thanh tra trong `Luật bảo hiểm xã hội.md` dòng 3456), không
  phải cấu trúc Điều/Khoản thật của văn bản đang parse — **không** được coi
  là heading/Khoản mới. Hiện `parser.py` không nhận diện vùng này, khiến nó
  bị nuốt nhầm vào nội dung Khoản cuối cùng đang mở (vd. lẫn vào Khoản 15
  Điều 141 `Luật bảo hiểm xã hội.md`) — đây là lỗi cần sửa.

Coi mỗi vùng là **1 Khoản ngầm định cấp văn bản** riêng biệt, không có
Phần/Chương/Mục/Điều/Khoản bao ngoài:

| Vùng | `breadcrumb_prefix` |
| ---- | -------------------- |
| Frontmatter | `{ten_van_ban} ({so_hieu})` |
| Backmatter | `{ten_van_ban} ({so_hieu}) - Chú thích sửa đổi (cuối văn bản)` |

Breadcrumb khác nhau giữa 2 vùng để tránh trùng `chunk_id` khi 1 file có cả
frontmatter và backmatter (mục 2: `chunk_id` băm từ `source_document` +
breadcrumb).

**Thuật toán cắt riêng — không dùng mục 4.1–4.5**: frontmatter/backmatter
không có cấu trúc Điểm và có thể trích dẫn xen kẽ nhiều luật khác nhau, nên
áp dụng thuật toán Điểm/câu (vốn thiết kế cho Khoản pháp luật thật) là không
phù hợp. Thay vào đó:

1. Tính `token_count` toàn vùng (mục 6). Nếu `<= MAX_TOKENS` → giữ nguyên 1
   chunk, giống mục 4.1.
2. Nếu vượt `MAX_TOKENS`, dùng
   `langchain_text_splitters.RecursiveCharacterTextSplitter` (mục 8) để cắt:

   ```python
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=MAX_TOKENS,
       chunk_overlap=0,
       length_function=count_tokens,  # chunking/tokenizer.py (mục 6) — đếm
       # token PhoBERT thật, không phải ký tự
       separators=["\n\n", "\n", ". ", "; ", " ", ""],
   )
   pieces = splitter.split_text(vùng_content)
   ```

   `separators` ưu tiên ranh giới đoạn/câu tiếng Việt trước khi rơi xuống
   khoảng trắng/ký tự đơn lẻ, nhất quán với ranh giới câu ở mục 4.4
   (`.`, `;`).
3. Mỗi phần tử `pieces` → 1 chunk con, breadcrumb nối thêm `(phần i/n)` như
   mọi Khoản bị cắt khác (mục 3) — không có nhãn Điểm. `chunk_overlap=0`:
   không lặp lại nội dung giữa các chunk con (khác với overlap 1 câu ở mục
   4.4 — chấp nhận vì đây là vùng phụ, không phải nội dung Khoản pháp luật
   chính).

Cơ chế câu dẫn (mục 4.5) **không áp dụng** cho frontmatter/backmatter — hai
vùng này không có Điểm gắn nhãn nên không có khái niệm câu dẫn.

Phân biệt với nội dung đứng giữa 2 heading Phần/Chương/Mục mà không có
Điều/Khoản nào theo sau (vd. dòng chú thích ngay dưới tiêu đề 1 Phụ Lục) —
trường hợp đó **vẫn bị bỏ qua** như thiết kế hiện tại của `parser.py`, vì nó
không phải nội dung pháp lý độc lập và không khớp định nghĩa frontmatter
(không đứng trước heading đầu tiên của cả file) hay backmatter (không đứng
sau marker `**[n]**` ở cuối file).

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
có transitive qua `langchain`, cần khai trực tiếp vì `config.py` import thẳng),
`transformers` (hiện transitive qua `sentence-transformers`, `chunking/tokenizer.py`
import thẳng `AutoTokenizer`), và **[MỚI 2026-09-14]** `langchain-text-splitters`
(gói con nhẹ, không kéo theo toàn bộ `langchain` — dùng
`RecursiveCharacterTextSplitter` cho frontmatter/backmatter, mục 4.6/8). Việc
gọi Pinecone SDK thật sự
(`pinecone` package) thuộc phạm vi `embedding/`, chunking chỉ cần `config.py`
khai đúng field.

## 8. Tools & Integrations

| Việc                               | Công cụ                                                                         |
| ----------------------------------- | --------------------------------------------------------------------------------- |
| Word segmentation tiếng Việt      | `pyvi` (đã có sẵn trong dependencies)                                       |
| Tokenizer đếm token               | `transformers.AutoTokenizer` (model PhoBERT-based ở mục 6)                    |
| Cắt frontmatter/backmatter (mục 4.6) **[MỚI]** | `langchain-text-splitters` — `RecursiveCharacterTextSplitter`, dùng `length_function=count_tokens` |
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
        → write_atomic(out_path, chunks dưới dạng mảng JSON)
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
| `patterns.py` **[CẬP NHẬT]** | Regex nhận diện heading markdown (Phần/Chương/Mục/Điều/Khoản), nhãn Điểm (`a)`, `b)`...), dòng bảng markdown (`\|...\|`), dòng marker chú thích dời cuối `**[n]**` (mục 4.6, mới). Đã xoá `NEGATION_KEYWORDS`/`RE_NEGATION` (mục 12).                 |
| `parser.py` **[CẬP NHẬT]**  | Đọc file markdown đã có front matter (gồm cả `loai_van_ban`/`co_quan_ban_hanh`/`ngay_ban_hanh`/`ngay_hieu_luc`, mục 2, mới), dựng`DocumentTree` — cây breadcrumb tới từng Khoản kèm nội dung thô, gồm cả Khoản ngầm định cấp văn bản cho frontmatter/backmatter (mục 4.6, mới); phát hiện`has_table`/trích `raw_table` cho từng Khoản.    |
| `tables.py`    | Chuyển`raw_table` (markdown) thành `standardization_table` (mục 5.3).                                                                                                                  |
| `tokenizer.py` | `count_tokens(text) -> int` — word-segment (pyvi) + đếm bằng `AutoTokenizer` (mục 6).                                                                                                |
| `splitter.py` **[CẬP NHẬT]** | Thuật toán cắt Khoản (mục 4): tách câu dẫn, ghép "cận dưới" trên ngân sách hiệu dụng, fallback theo câu có overlap, ghép câu dẫn vào mọi chunk con (mục 4.5, cập nhật — trước đây gán `negation_note`); áp dụng ngoại lệ bảng (mục 5.1) trước khi cắt; với Khoản ngầm định frontmatter/backmatter, cắt bằng `RecursiveCharacterTextSplitter` thay vì thuật toán Điểm/câu (mục 4.6, mới). |
| `models.py`    | Pydantic models:`Chunk`, `ChunkingResult`, `DocumentTree` (input/output giữa các module trên).                                                                                       |
| `pipeline.py`  | `convert_markdown_to_chunks(path) -> ChunkingResult` và `convert_directory(markdown_dir, out_dir)` — orchestration, atomic write, gom summary.                                          |

`tools/chunk_documents.py` chỉ là Typer CLI mỏng gọi
`pipeline.convert_directory()`.

## 11. Tiêu chí hoàn thành

- Chạy CLI trên toàn bộ `data/markdown/` sinh ra `data/chunks/*.json`, mỗi
  chunk có breadcrumb đầy đủ và `token_count <= MAX_TOKENS` (trừ chunk có
  `has_table=True`, xem mục 5.1).
- **[CẬP NHẬT 2026-09-14]** Với Khoản có Điểm và có câu dẫn, bị cắt thành
  nhiều chunk (mục 4.5) — xác nhận thủ công ít nhất 1 ví dụ thật trong corpus
  (Điều 85 Khoản 1 `Luật bảo hiểm xã hội.md`, hoặc điểm b/c Khoản 4/7 Điều 2
  cùng file) — mọi chunk con sinh ra từ Khoản đó đều có `content` bắt đầu
  bằng đúng câu dẫn gốc. (Trước đây tiêu chí này kiểm tra field `negation_note`
  khớp câu phủ định — nay đã thay thế, xem mục 12.)
- Với Điều 3 - Khoản 1 của `data/markdown/Quy định mức lương tối thiểu.md`
  (bảng 4 vùng lương, mục 5) — sinh đúng 1 chunk duy nhất
  (`is_split=False`, `has_table=True`), `raw_table` khớp nguyên văn bảng
  markdown gốc, `standardization_table` có đúng 4 dòng theo mẫu mục 5.3.
- **[MỚI 2026-09-14]** Với mỗi file `data/markdown/*.md` có đoạn mở đầu
  trước heading đầu tiên (frontmatter, mục 4.6) — sinh đúng 1 (hoặc nhiều,
  nếu vượt `MAX_TOKENS`) chunk cho đoạn đó, breadcrumb chỉ còn
  `{ten_van_ban} ({so_hieu})`, nội dung khớp nguyên văn đoạn mở đầu gốc (vd.
  "LUẬT", "BẢO HIỂM XÃ HỘI", "Căn cứ Hiến pháp...", "Quốc hội ban hành..."
  trong `Luật bảo hiểm xã hội.md`). Tiêu chí này không tồn tại trong bản spec
  gốc (đoạn mở đầu trước đây bị bỏ qua hoàn toàn).
- **[MỚI 2026-09-14]** Với `Luật bảo hiểm xã hội.md` (dòng 3456, marker
  `**[10]**`) — vùng chú thích sửa đổi dời cuối (backmatter, mục 4.6) tách
  thành chunk riêng, breadcrumb `Luật Bảo hiểm xã hội (41/2024/QH15) - Chú
  thích sửa đổi (cuối văn bản)`, **không** còn lẫn vào `content` của Khoản 15
  Điều 141 (Khoản ngầm định gần nhất trước đó).
- **[MỚI 2026-09-14]** Mọi chunk của 1 file có
  `loai_van_ban`/`co_quan_ban_hanh`/`ngay_ban_hanh`/`ngay_hieu_luc` khớp đúng
  giá trị front matter YAML của file đó (mục 2).
- Chạy lại nhiều lần trên cùng input ra `chunk_id` và nội dung giống hệt
  (deterministic).
- 1 file lỗi không làm dừng toàn bộ batch; CLI kết thúc với summary rõ ràng.
- `config.py` được các package khác (`chunking/`, và sau này `embedding/`)
  import `EmbeddingSettings`/`VectorDBSettings` thay vì đọc `.env` trực tiếp.

## 12. Lịch sử cập nhật

Các thay đổi dưới đây được đánh dấu **[MỚI]** (nội dung hoàn toàn không có ở
bản spec gốc) hoặc **[CẬP NHẬT]** (sửa hành vi đã có) ngay tại vị trí liên
quan trong các mục ở trên. Mục này tóm tắt lại để xem nhanh, không thay thế
phần đánh dấu inline.

**2026-09-14 — brainstorm cùng người dùng, 4 điểm phát hiện khi tự test:**

1. **Câu dẫn lặp vào `content`** (mục 4.2–4.5, **CẬP NHẬT**): trước đây
   "đơn vị #0" chỉ là 1 đơn vị bình thường trong pool ghép "cận dưới" — chunk
   con nào không chứa nó thì mất câu dẫn trong `content`; câu phủ định (nếu
   có) chỉ được lặp vào `breadcrumb`/field `negation_note`. Nay: câu dẫn tách
   riêng khỏi pool, luôn lặp lại nguyên văn vào `content` của **mọi** chunk
   con (không chỉ khi có từ khoá phủ định), ngân sách token tính bù trừ qua
   `budget_hiệu_dụng`.
2. **Frontmatter/backmatter** (mục 4.6, **MỚI**): đoạn mở đầu văn bản (trước
   heading đầu tiên) và vùng chú thích sửa đổi dời cuối file (marker
   `**[n]**`) trước đây bị bỏ qua/nuốt nhầm vào Khoản khác — nay tách thành
   Khoản ngầm định cấp văn bản riêng, cắt bằng
   `RecursiveCharacterTextSplitter` (LangChain) thay vì thuật toán Điểm/câu.
3. **Bỏ `negation_note`** (mục 2, **ĐÃ XOÁ**): field và cơ chế phát hiện từ
   khoá phủ định (`NEGATION_KEYWORDS`/`RE_NEGATION` trong `patterns.py`) bị
   xoá hoàn toàn — thay thế bởi cơ chế câu dẫn tổng quát ở điểm 1.
4. **4 field front matter mới** (mục 2, **MỚI**): `loai_van_ban`,
   `co_quan_ban_hanh`, `ngay_ban_hanh`, `ngay_hieu_luc` thêm vào `Chunk`, đọc
   trực tiếp từ front matter YAML (đã có sẵn từ `formatting/frontmatter.py`).

Code (`parser.py`, `splitter.py`, `models.py`, `patterns.py`) **chưa** được
cập nhật theo các thay đổi này — đây mới chỉ là cập nhật spec, cần chạy qua
develop-cycle để implement.
