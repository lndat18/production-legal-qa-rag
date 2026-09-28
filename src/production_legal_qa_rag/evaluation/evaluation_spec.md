# Evaluation — Synthetic Testset Generation (Phase 1): Reference Spec

> **Phạm vi spec này chỉ là Phase 1 của roadmap RAGAS** (xem `CLAUDE.md` mục Tiến độ):
> sinh bộ câu hỏi + đáp án chuẩn ("golden testset") từ corpus pháp luật thật, không tự
> gán nhãn thủ công. **Phase 2 — chạy pipeline retrieval/generation thật trên testset này,
> tính metric RAGAS (`faithfulness`, `answer_relevancy`, `context_precision`,
> `context_recall`), chọn judge LLM, ngưỡng theo dõi — CHƯA được thiết kế và sẽ mở rộng
> thêm module vào package `evaluation/` này ở một vòng brainstorm riêng sau.** Người đọc
> spec này không nên hiểu đây là toàn bộ scope đánh giá chất lượng của dự án.
>
> Quyết định phạm vi (chốt cùng ngày viết spec): **không dùng dữ liệu `chatlog`/
> `chat_turns` thật ở phase này** — lý do hệ thống mới nghiệm thu production, dữ liệu
> thật tích luỹ chưa đủ, và người dùng ưu tiên có ngay một bộ dữ liệu đánh giá chạy được
> mà không cần chờ hay tự gán nhãn tay. `chat_turns` vẫn tích luỹ song song, để dành cho
> một hướng đánh giá khác (theo phân bố câu hỏi người dùng thật) nếu cần ở phase sau.

## 1. Mục tiêu & phạm vi

Sinh tự động một bộ câu hỏi mẫu ("golden testset") gồm câu hỏi + câu trả lời chuẩn +
context chuẩn, trực tiếp từ corpus pháp luật thật (`data/markdown/`), bằng
`ragas.testset.TestsetGenerator` (LLM đọc văn bản, tự đặt câu hỏi và tự viết đáp án đúng
theo văn bản đó) — thay cho việc người vận hành tự đọc luật và viết tay từng cặp
câu hỏi/đáp án chuẩn.

**Trong phạm vi:**

- Đọc 6 file `.md` trong `data/markdown/`, cắt mỗi văn bản thành các **Chương** (theo tiêu
  đề `## Chương ...`) và **xử lý TỪNG Chương một** (mỗi Chương một `KnowledgeGraph`
  riêng — quyết định 2026-09-28, xem "Mỗi Chương một đồ thị" bên dưới).
- Với mỗi Chương, build `KnowledgeGraph` (đồ thị tri thức nội bộ của ragas: tóm tắt,
  trích entity, dựng quan hệ giữa các đoạn văn bản) và sinh câu hỏi. Tổng
  **`GENERATE_SIZE = 240`** câu cho cả corpus theo tỷ lệ **80/10/10** (192 single-hop / 24
  multi-hop abstract / 24 multi-hop specific — người dùng chốt 2026-09-28, mục 10.10),
  chia cho các Chương tỷ lệ theo kích thước (mục 4) — **sinh dư so với đích 180 để trừ
  hao** khi người dùng đọc lướt và xoá câu xấu (xem "Về việc chọn 240/180" bên dưới).
- Luân phiên (round-robin) **6** tài khoản Groq độc lập (`GROQ_API_KEY`…`_6`) cho
  `generator_llm` để rải tải gọi LLM khi build KG + sinh 240 câu (mục 3.1) — **quyết định
  có chủ đích của người dùng**. (Lịch sử: chốt ban đầu 3 tài khoản/360 câu; 2026-09-28
  người dùng có thêm `GROQ_API_KEY_4` rồi `_5`, `_6` (tổng **6 tài khoản**, tên biến
  `GROQ_API_KEY_5`/`_6` đã được người dùng xác nhận), chốt đích cuối cùng **180 câu** để còn thời gian
  làm deploy + observability, và chốt cách sinh dư rồi trừ hao.)
- Lưu testset thô ra `golden_testset_raw.json` (để người dùng đọc lướt/xoá) **ngay sau
  khi xong từng Chương** (checkpoint theo Chương — chạy dở, hết quota Groq ngày hôm đó
  thì hôm sau chạy tiếp, mục 4), rồi lệnh `finalize` chốt còn đúng `TARGET_SIZE = 180`
  câu ra `golden_testset.json` (mục 4.2). Lưu `KnowledgeGraph` của từng Chương ra file
  riêng để tái dùng (tránh build lại — tốn LLM call — khi cần sinh bù).
- CLI Typer mỏng: lệnh `generate` (có `--only`, `--testset-size`,
  `--reuse-knowledge-graph`, `--append`) và lệnh `finalize` (mục 7).

**Mục tiêu của bộ eval (chốt 2026-09-28, người dùng): kiểm tra TOÀN HỆ THỐNG — chạy
câu hỏi qua pipeline thật từ đầu đến cuối (HyDE → hybrid retrieval → RRF/MMR → rerank →
generation → verify; phạm vi chính xác của Phase 2 chưa chốt) — với bộ câu hỏi/đáp án chuẩn
ĐỘC LẬP với thiết kế của hệ thống.** "Độc lập" ở đây nói về **nguồn của testset**: câu hỏi
và đáp án chuẩn được sinh từ văn bản luật gốc, không đi qua `chunking/` nên không mang
theo quyết định cắt chunk của hệ thống. Vì vậy input của `TestsetGenerator` là **văn bản
nguyên bản** (`generate_with_langchain_docs`), KHÔNG phải chunk của `chunking/`
(`generate_with_chunks`). Xem mục 2.1 để biết lý do và số đo.

**Mỗi Chương một đồ thị (chốt 2026-09-28, người dùng).** Hệ thống RAG hiện tại KHÔNG có
khả năng xử lý câu hỏi cần ghép thông tin **chéo giữa các luật**, và cũng xử lý kém câu
cần ghép **chéo giữa các Điều** (chunk cắt theo Khoản). Vì vậy testset chỉ cần quan hệ
**trong phạm vi hẹp**: build `KnowledgeGraph` riêng cho từng **Chương**, không nối quan hệ
giữa các Chương hay các văn bản (mất quan hệ chéo Chương — chấp nhận được vì hệ thống vốn
không xử lý chéo Điều). Đây cũng là điều kiện để chạy được trên quota Groq free: token/ngày
có hạn (mục 3.1), mỗi Chương chỉ vài chục KB nên vừa quota một ngày, lỗi chỉ mất một Chương
và mỗi Chương xong là lưu ngay nên chia nhiều ngày được (mục 4).
Tỷ lệ loại câu **80/10/10** (người dùng chốt): multi-hop vẫn có thể cần ghép nhiều Điều
**trong cùng một Chương** — loại câu hệ thống đã biết là yếu — nên chỉ chiếm 20% để điểm gộp
không bị kéo xuống bởi điểm yếu đã biết; giữ `synthesizer_name` để Phase 2 vẫn báo điểm
riêng theo loại (mục 10.10).

**Không làm (ở phase này):**

- Không chạy `retrieve()`/`generate()` thật của hệ thống, không tính bất kỳ metric RAGAS
  nào (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) — các
  metric này cần `response`/`retrieved_contexts` thật, chỉ có ở Phase 2.
- Không dùng `chat_turns`/chatlog.
- Không tự động hoá việc lọc câu hỏi vô nghĩa bằng code (LLM-as-judge lọc, heuristic...).
  Review là thao tác đọc bằng mắt của người vận hành, ngoài phạm vi code.
- Không CI/cron; đây là script chạy tay, sinh dữ liệu tĩnh dùng lại nhiều lần.
- Không lưu Postgres; corpus và output đều là file tĩnh, không cần một CSDL cho việc này.

**Tiêu chí hoàn thành Phase 1:** có file `data/eval/golden_testset.json` chứa **đúng 180**
câu hỏi hợp lệ (đủ `user_input`/`reference`/`reference_contexts`, không rỗng), đã được
người vận hành đọc lướt và loại bỏ câu vô nghĩa (ngôn ngữ đúng theo kết quả pilot, mục 9.0),
sẵn sàng dùng thẳng làm input cho Phase 2 mà không cần sinh lại từ đầu.

**Về việc chọn 240 sinh / 180 đích (chốt 2026-09-28, người dùng):** build
`KnowledgeGraph` là chi phí cố định theo Chương (không đổi theo số câu, và đã cache ra
`knowledge_graph/<văn bản>__<chương>.json`); chỉ bước sinh câu hỏi mới tỉ lệ theo số câu. **Nút
thắt là token/ngày của Groq free** (TPD 200K/tài khoản, mục 3.1), không phải TPM; nút thắt
thứ hai là **công review tay** (mục 9). Người dùng chốt đích **180** (đủ lớn để có ý nghĩa
thống kê thô, đồng thời nhường thời gian cho deploy + observability) và chọn cách **sinh
dư rồi trừ hao**: ước tính đọc lướt loại ~25–40% câu (pilot: ~6/8 dùng được, còn trùng ý,
mục 4.1) nên sinh 240 (dư ~33%) để còn ~145–180 sau khi xoá; thiếu thì sinh bù. Nếu vẫn thiếu, sinh bù bằng
`--only <văn bản> --reuse-knowledge-graph --append` (mục 4.2) — không phải build lại đồ thị
(đường dẫn KG của từng Chương suy ra được từ tên văn bản và số Chương).

## 2. Input & Output

**Input:** `data/markdown/*.md` — đúng 6 văn bản pháp luật hoàn chỉnh (không phải chunk
đã cắt cho embedding; `TestsetGenerator` tự chunk/build graph nội bộ từ tài liệu gốc).

**Output** (thư mục mới `data/eval/`, không có trước):

| File | Nội dung |
| --- | --- |
| `golden_testset_raw.json` | ~240 `GoldenTestCase` do ragas sinh, **chưa review**, được ghi thêm sau khi xong từng Chương. Người dùng đọc lướt và xoá câu xấu ngay trong file này. |
| `golden_testset.json` | **Đúng 180** `GoldenTestCase` sau review, do lệnh `finalize` (mục 4.2) tạo ra từ file raw. **Đây là file duy nhất Phase 2 đọc.** Mỗi item (mục 5): `user_input`, `reference`, `reference_contexts`, `synthesizer_name`, `source_document`, `source_section`. |
| `generation_progress.json` | **File theo dõi tiến độ** (mục 4.5): đơn vị nào đã xong, xong lúc nào, sinh được bao nhiêu câu, tốn bao nhiêu lượt gọi/token. Do code ghi, người dùng không sửa tay (trừ khi muốn chạy lại một đơn vị). Nguồn sự thật duy nhất cho câu hỏi "đã làm tới đâu" — `generate` chạy lại hôm sau đọc file này để tiếp tục, không chạy lại đơn vị đã xong. |
| `knowledge_graph/<văn bản>__<chương>.json` | `KnowledgeGraph` của từng Chương, serialize bằng API save/load sẵn có của ragas — dùng lại khi cần sinh thêm câu hỏi mà không build lại từ đầu. |

`golden_testset.json` phải giữ đúng 3 cột bắt buộc theo schema `EvaluationDataset` của
ragas (`user_input`, `reference`, `reference_contexts`) để Phase 2 dùng thẳng không cần
convert; `synthesizer_name` là cột phụ giữ lại để biết loại câu hỏi (single-hop/multi-hop)
lúc review bằng mắt.

**Ai tạo cột nào (để không nhầm phạm vi Phase 1 với Phase 2).** RAGAS chấm điểm cần một
"hàng dữ liệu" gồm `user_input`, `reference`, `response`, `retrieved_contexts`. Phase 1
(spec này) chỉ tạo được nửa đầu; nửa sau chỉ có khi chạy chatbot thật ở Phase 2:

| Cột | Ai tạo | Phase |
| --- | --- | --- |
| `user_input` (câu hỏi) | `TestsetGenerator` sinh từ văn bản luật | 1 |
| `reference` (đáp án đúng, bám luật) | `TestsetGenerator` sinh | 1 |
| `reference_contexts` (đoạn luật ragas đã đọc để đặt câu hỏi) | `TestsetGenerator` sinh; cột phụ, chỉ để truy vết | 1 |
| `synthesizer_name` (loại câu hỏi) | `TestsetGenerator` sinh; cột phụ để review | 1 |
| `response` (câu trả lời thật của chatbot) | Chạy pipeline hệ thống lấy `user_input` làm đầu vào | 2 |
| `retrieved_contexts` (chunk thật hệ thống tìm về) | Cùng lần chạy pipeline trên | 2 |

Ví dụ minh hoạ 1 `GoldenTestCase` single-hop (nội dung viết tay dựa trên Điều 2 luật thuế
TNCN, không phải output thật):

| Cột | Ví dụ |
| --- | --- |
| `user_input` | "Người nước ngoài ở Việt Nam 200 ngày trong năm thì có được coi là cá nhân cư trú không?" |
| `reference` | "Có. Cá nhân cư trú là người có mặt tại Việt Nam từ 183 ngày trở lên tính trong một năm dương lịch hoặc trong 12 tháng liên tục kể từ ngày đầu tiên có mặt, nên 200 ngày là đáp ứng điều kiện." |
| `reference_contexts` | `["Điều 2. Người nộp thuế … 2. Cá nhân cư trú là người đáp ứng một trong các điều kiện sau đây: a) Có mặt tại Việt Nam từ 183 ngày trở lên … b) Có nơi ở thường xuyên tại Việt Nam … 3. Cá nhân không cư trú là …"]` (1 đoạn dài do ragas tự cắt; câu multi-hop có ≥2 đoạn) |
| `synthesizer_name` | `single_hop_specific_query_synthesizer` |

Ở Phase 2, chỉ `user_input` được đưa vào chatbot; `reference` là thứ đem so với kết quả.

### 2.1 Vì sao đưa văn bản nguyên bản, không dùng chunk có sẵn (chốt 2026-09-28)

RAGAS có `generate_with_chunks` (nhận chunk dựng sẵn, bỏ bước tự cắt) — đã cân nhắc và
**không dùng**:

- **Mục tiêu là kiểm tra hệ thống độc lập với thiết kế của nó.** Nếu sinh câu hỏi từ chính
  chunk của `chunking/`, ground truth do chính chunker định nghĩa: chunker cắt sai (một ý
  pháp lý bị chia đôi) thì testset không bao giờ hỏi trúng chỗ lỗi; câu hỏi sinh từ đúng
  văn bản chunk cũng khiến BM25/dense dễ bắt trúng hơn thực tế (thiên lệch lạc quan). Văn
  bản nguyên bản không phụ thuộc chunker nên không có vòng tròn này.
- **Corpus chunk hiện tại không hợp làm input cho RAGAS** (đo 2026-09-28 trên
  `data/chunks/`): 2.228 chunk, trung vị 41 token, 58% dưới 50 token (thường là một
  khoản/điểm đơn lẻ), `breadcrumb` chỉ nằm trong metadata nên `content` không nói thuộc
  luật/Điều nào. Summary/themes/NER trên đoạn cỡ này cho tín hiệu mỏng, câu hỏi dễ mơ hồ
  giữa 6 luật, và chi phí LLM cao hơn (summary+themes+NER+filter chạy trên 2.228 node).
- **Đánh đổi chấp nhận (đã kiểm chứng trên `ragas==0.4.3` cài thật, 2026-09-28):**
  `reference_contexts` là đoạn do ragas tự cắt (`HeadlineSplitter(min_tokens=500)`, đo
  bằng tokenizer của ragas), còn chunk của hệ thống tối đa 236 token (trung vị 41) — một
  đoạn ragas bao trùm nhiều chunk của hệ thống, quan hệ nhiều-1 chứ không 1-1. Hệ quả:
  - **Không ảnh hưởng** hai metric chính của Phase 2: `context_recall` và
    `context_precision` (bản LLM) chỉ cần `user_input`, `retrieved_contexts`, `reference`
    — LLM so nội dung chunk hệ thống tìm về với **đáp án đúng `reference`**, không so với
    `reference_contexts`, nên ranh giới/độ dài chunk không quan trọng.
  - **Có ảnh hưởng** nếu muốn metric xác định không tốn LLM judge: bản `NonLLM...` (so
    chuỗi với `reference_contexts`) hoặc hit@k/MRR theo `chunk_id` sẽ không khớp 1-1.
    Map nhiều-1 (chunk hệ thống nằm trong đoạn ragas) khả thi vì 2.096/2.228 chunk
    (94%) là trích nguyên văn từ markdown, nhưng tập chuẩn thô (thừa chunk không thực sự
    cần cho câu trả lời) nên recall@k thấp giả; chỉ hit@k là dùng được. Để dành cho Phase
    2 nếu cần (mục 10.8), không làm ở Phase 1.

## 3. Công cụ

| Việc | Công cụ |
| --- | --- |
| Sinh testset | `ragas.testset.TestsetGenerator` (`ragas==0.4.3`) |
| LLM sinh câu hỏi (`generator_llm`) | `ChatOpenAI` trỏ Groq (cùng pattern `generation_spec.md` mục 8), bọc `ragas.llms.LangchainLLMWrapper` |
| Embedding cho build KnowledgeGraph (`generator_embeddings`) | Adapter mới quanh `HuggingFaceEmbedder`/`InferenceClient` hiện có ở `embedding/hf_client.py`, bọc `ragas.embeddings.LangchainEmbeddingsWrapper` |
| Document loader | `langchain_core.documents.Document` (đã có sẵn qua `langchain-openai`, không cần thêm dependency loader) |
| CLI | `typer` (`tools/generate_testset.py`) |

**Dependency mới (`pyproject.toml`):** `ragas>=0.4.3`, `langchain-community<0.4` và
**`rapidfuzz`**, nằm trong `[dependency-groups] eval` (không phải `[project] dependencies`
gốc). `rapidfuzz` được phát hiện ở pilot 2026-09-28: `ragas` import nó lúc dựng
`OverlapScoreBuilder` (`relationship_builders/traditional.py`) nhưng KHÔNG khai báo là
dependency, nên `uv sync --group eval` không cài → `ImportError` ngay khi gọi
`generate_with_langchain_docs`. Developer phải thêm `rapidfuzz` vào group `eval` (và
cập nhật `uv.lock`). `uv sync`
mặc định (venv production, `deploy/Dockerfile`) KHÔNG cài các gói này lẫn
`scikit-network`/`instructor` mà `ragas` kéo theo. Chạy script/test của package này cần
group `eval`:

```bash
uv sync --group eval --no-group production   # cài eval, bỏ nhóm pin `production`
uv run --group eval --no-group production tools/generate_testset.py ...
uv run --group eval --no-group production pytest tests/test_evaluation.py
```

`uv run` KHÔNG kèm cờ sẽ tự re-sync venv theo `default-groups = ["dev", "production"]`:
kéo `openai` về bản production trong khi `ragas`/`instructor` (cần `jiter<0.15`) vẫn nằm
trong venv, không có cảnh báo. Vì vậy MỌI lệnh `uv run` liên quan tới `eval` phải mang đủ
`--group eval --no-group production`, không chỉ lệnh `uv sync` đầu tiên. Xong việc, chạy
`uv sync` (không cờ) để trả venv về profile mặc định.

`[dependency-groups] production = ["openai>=3.19.0"]` (không chứa package thật, mặc định
bật cùng `dev`) chỉ tồn tại để buộc `uv` tách resolve `openai` riêng cho nhóm `eval` —
`ragas` phụ thuộc `instructor`, ép `jiter<0.15`, nếu không tách sẽ kéo `openai` xuống bản
cũ mà `langchain-openai` dùng cho Groq thật trong production (`tool.uv.conflicts` giữa
`production` và `eval` trong `pyproject.toml`). Vì `production` và `eval` xung đột, không
thể `uv sync` cả hai cùng lúc — luôn dùng `--no-group production` khi cần `eval`.

**Rủi ro kỹ thuật đã biết, không chặn spec:** `HuggingFaceEmbedder` (`embedding/hf_client.py`)
hiện chỉ có `embed_chunks(chunks: list[Chunk])`, không implement interface
`langchain_core.embeddings.Embeddings` (`embed_documents(texts: list[str]) ->
list[list[float]]`, `embed_query(text: str) -> list[float]`) mà `LangchainEmbeddingsWrapper`
cần. Cần viết một adapter nhỏ (`embeddings_adapter.py`, mục 6) gọi thẳng
`InferenceClient.feature_extraction` trên text thô — xử lý lúc implement, không phải
quyết định kiến trúc mới.

**Không áp dụng `pyvi.ViTokenizer.tokenize()` (word-segmentation) trong adapter này** —
khác mục đích với embedding production (`embedding_spec.md` mục 4, dùng để khớp
query/index cùng không gian vector tìm kiếm): ở đây embedding chỉ phục vụ nội bộ
`TestsetGenerator` để so sánh độ tương đồng giữa các đoạn tóm tắt do LLM sinh khi build
`KnowledgeGraph`, không phải để tìm kiếm chéo với index sản xuất. Áp segmentation sai chỗ
sẽ chỉ thêm phức tạp không có lợi ích đo được.

**API ragas đã xác nhận trên `ragas==0.4.3` cài thật** (đọc source + chạy pilot 2026-09-28):
`TestsetGenerator(llm=..., embedding_model=...)`, `generate_with_langchain_docs(docs,
testset_size=..., transforms=..., query_distribution=..., run_config=...)`,
`default_transforms(documents, llm, embedding_model)`, `adapt_prompts(language, llm)` +
`set_prompts(**...)` trên synthesizer. Theo nguyên tắc "đo trước khi tin" của project, vẫn
đọc source của package cài thật (không suy diễn từ doc web khác version) khi dùng thêm API
mới, đặc biệt cho 2 synthesizer multi-hop (mục 4.3).

### 3.1 Round-robin 6 tài khoản Groq (`groq_round_robin.py`) — quyết định mới

> **Cập nhật 2026-09-28: 6 tài khoản** (`GROQ_API_KEY`, `_2`…`_6`). Key 5, 6 là bucket
> `gpt-oss-120b` **hoàn toàn rảnh** (production chỉ dùng key 1-4). Quota ngày = 6 × 200K =
> **~1,2M token/ngày** cho `gpt-oss-120b`; các số đo "4 tài khoản/~800K/ngày" trong mục
> 4.1 là lịch sử pilot — lập kế hoạch theo 6 tài khoản (mục 4.4, 4.5).

Với `testset_size = 240` (`GENERATE_SIZE`), số lượt gọi `generator_llm` (build `KnowledgeGraph` + sinh câu
hỏi) đủ lớn để 1 tài khoản Groq duy nhất dễ chạm rate limit theo phút/ngày (`CLAUDE.md`:
Groq giới hạn theo **tài khoản**, không theo key). Người dùng có 6 tài khoản Groq riêng
biệt (`GROQ_API_KEY`, `_2`…`_6` — 6 email khác nhau, không phải 6 key cùng 1 tài
khoản) và muốn luân phiên cả 6 cho luồng gọi này.

**Quyết định (người dùng, 2026-09-28): dùng cả 6 key cho việc sinh testset lẫn đánh giá,
và sẽ không chạy việc gì khác trên 6 tài khoản này trong lúc đó.** Cần biết bối cảnh: Groq
tính rate limit theo `(tài khoản, model)` (`conversation_spec.md` mục 12.1). Job này dùng
`gpt-oss-120b`; bucket 120b của key 3, 4 vốn là bucket generation của production
(xoay vòng key 3 ⇄ 4), còn bucket 120b của key 1, 2 đang rảnh. Vì người dùng cam kết
không chạy song song việc khác (kể cả traffic thật), **không làm cơ chế nhường-chỗ** nào
trong code; ràng buộc "chạy khi hệ thống rảnh" là kỷ luật vận hành, ghi ở mục 9.

**Ràng buộc đo được từ pilot (2026-09-28) — đã giải quyết ở "Hướng giải quyết đã chốt" bên dưới.**
Groq tier `on_demand` giới hạn `gpt-oss-120b` ở **8.000 token/phút (TPM) cho MỖI tài
khoản**, và một request lớn hơn mức đó bị từ chối ngay (HTTP 413 `Request too large`,
không phải 429 nên không retry được). Pilot trên file nhỏ nhất (`Quy định mức lương tối
thiểu.md`, 37 KB) xin **12.325 token** trong MỘT lượt gọi (`HeadlinesExtractor` gửi cả
văn bản, cắt ở `max_token_limit=32000` mặc định của ragas) → thất bại, lãng phí ~171s
retry vô ích của ragas (`RunConfig`: `max_retries=10`). Hệ quả:
- Round-robin nhiều tài khoản **không giải quyết được** lỗi này: nó rải tổng tải qua các tài
  khoản, nhưng không làm một request đơn lẻ nhỏ đi. Mỗi lượt gọi phải nhỏ hơn 8.000 token
  (cả input lẫn output/reasoning) — thực tế nên ≲ 4.000–5.000 token input.
- Tổng thông lượng tối đa ≈ 6 × 8.000 = 48.000 token/phút; corpus ~1 MB tiếng Việt ước
  ~345.000 token/lượt đọc toàn corpus (12.325 token / 37 KB), mà build KG đọc corpus vài
  lần (headline, summary, themes, NER...) → job chạy giờ, không phút.
- **Giới hạn Free Plan của Groq (người dùng chụp từ console, 2026-09-28), theo
  (tài khoản, model):** `gpt-oss-120b` và `gpt-oss-20b` đều RPM 30, RPD 1K, TPM 8K,
  **TPD 200K**. Tổng 6 tài khoản = ~1,2M token/ngày cho mỗi model (lúc đo mới có 4: ~800K). Giới hạn ràng buộc là
  **TPD**, không phải TPM. **Số đo thật từ pilot Groq (mục 4.1, lần 3-4):** dựng KG ≈ 5
  token/ký tự văn bản, sinh câu ≈ 1,5K token/câu → cả corpus (781.007 ký tự — bản trước tính nhầm theo byte 1,04M) + 240 câu ≈ **4,1-4,3M token**
  (khoảng 3-5,5M) ≈ 5 ngày với 4 tài khoản (~800K/ngày, lúc đo), **~3-4 ngày với 6 tài khoản** (~1,2M/ngày). Sau khi chia đơn vị thật và bỏ khối chú thích cuối file (mục 4.4): **50 đơn vị, 720.573 ký tự, ~4,16M token**,
  không chạy nổi trong 1 ngày trên free tier. Các model khác (llama/qwen/kimi...) có bucket
  riêng — chưa xem hết bảng; tab Developer Plan Limits chưa xem.
- `GroqRoundRobinChatModel` hiện chỉ xử lý `RateLimitError`, không xử lý 413.

**Hướng giải quyết đã chốt (2026-09-28):** hạ `max_token_limit` của các extractor ragas xuống
~4.000 qua transforms mặc định đã chỉnh (mã mẫu ở mục 4.3; không phải người dùng cắt tay) và
`max_workers` xuống ~4; xử lý theo từng **Chương** (mục 1, 4) để mỗi đơn vị vừa quota ngày và
chạy trải nhiều ngày. Nâng 1 tài khoản Groq lên Dev Tier vẫn là phương án nếu 7 ngày là quá
lâu — chưa chọn.

**Đây là pattern MỚI trong repo, không phải áp dụng lại cái đã có.** Các chỗ dùng nhiều
key Groq hiện tại (`GenerationSettings`/`JudgeSettings` — `generation_spec.md` mục 8,
`formatting/llm_client.py` — `formatting_spec.md` mục 1.3) đều dùng nhiều key để **tách
ngân sách theo bước cố định** (mỗi bước luôn dùng đúng 1 key, bước khác dùng key khác,
không đổi trong lúc chạy). Ở đây là **luân phiên nhiều key cho CÙNG một luồng gọi** để rải
tải — mục đích khác hẳn, nên không tái dùng được `_SlidingWindowRateLimiter`/
`convert_chunks_concurrently` của `formatting/llm_client.py` (thiết kế cho 2 worker thread
xử lý song song một hàng đợi job, không phải round-robin tuần tự).

Thiết kế (đơn giản nhất đủ dùng cho 1 script chạy 1 lần, không over-engineer):

```python
class GroqRoundRobinChatModel(BaseChatModel):
    """Proxy luân phiên round-robin qua N ChatOpenAI (Groq) độc lập tài khoản.

    Không phải rate-limiter: chỉ đổi client theo vòng lặp cố định trước mỗi
    lượt gọi thật, để rải tải đều qua các tài khoản độc lập.
    """

    clients: list[ChatOpenAI]  # đúng 6, mỗi client gắn 1 key cố định

    def _generate(self, messages, ...):
        client = self._next_client()          # itertools.cycle, state riêng instance
        try:
            return client._generate(messages, ...)
        except RateLimitError:                 # bounded fallback, không vô hạn
            for _ in range(len(self.clients) - 1):
                client = self._next_client()
                try:
                    return client._generate(messages, ...)
                except RateLimitError:
                    continue
            raise                               # hết vòng vẫn lỗi -> raise nguyên lỗi cuối
```

- Chỉ cần implement `_generate` (sync) — `BaseChatModel._agenerate` mặc định fallback gọi
  `_generate` qua executor khi không override, nên round-robin vẫn đúng dù ragas gọi qua
  đường async.
- Không thêm rate-limiter mới: mỗi `ChatOpenAI` con giữ nguyên timeout/retry riêng từ
  `TestsetGeneratorSettings` (mục 6); round-robin chỉ chọn client, không kiểm soát tốc độ.
- Bounded fallback khi 429: thử tối đa `len(clients)` client cho MỘT lượt gọi rồi mới raise
  — cùng tinh thần retry giới hạn đã dùng ở `conversation/condenser.py` (retry đúng 1 lần).
- **Rủi ro chấp nhận được, không xử lý thêm:** nếu ragas gọi `generator_llm` đồng thời
  (concurrent, ví dụ qua `asyncio.gather` nội bộ khi build KG), `itertools.cycle` không
  thread/async-safe tuyệt đối — có thể 2 lượt gọi cùng lúc nhận cùng 1 client thay vì luân
  phiên hoàn hảo. Đây chỉ ảnh hưởng đến độ *đều* của việc rải tải (vẫn đúng chức năng,
  không phải lỗi đúng/sai), chấp nhận được cho một script chạy 1 lần — không cần khoá
  `asyncio.Lock`/`threading.Lock` chỉ để tối ưu độ đều tuyệt đối.
- Không dùng `LoopBoundClient` (pattern ở `generation/`/`retrieval/` cho client sống suốt
  vòng đời server qua nhiều event loop) — không áp dụng ở đây vì đây là script chạy tuần
  tự trong đúng 1 process, không phải server long-running.

## 4. Workflow (`testset_generator.py`)

```text
clients = [ChatOpenAI(key=GROQ_API_KEY), ..._2, ... , ..._6]            # 6 client, config mục 6
generator_llm = LangchainLLMWrapper(GroqRoundRobinChatModel(clients=clients))  # mục 3.1
generator_embeddings = LangchainEmbeddingsWrapper(adapter)              # embeddings_adapter.py
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
[ngôn ngữ: adapt_prompts("vietnamese") cho MỖI synthesizer, gọi 1 lần rồi dùng lại — bắt buộc,
 mục 4.1/4.3; synthesizer ép PERFECT_GRAMMAR]

tất cả đơn vị của mọi văn bản (chia bằng `unit_splitter.split_directory`, đã implement), sắp xếp
theo THỨ TỰ CHẠY = số ký tự tăng dần (nhỏ trước, mục 4.5), lọc theo --only nếu có:
  for mỗi đơn vị <chương>:
    → nếu "<tên>#<số thứ tự>" đã có trong generation_progress.json (mục 4.5) và không có
      --append: BỎ QUA (đã xong ở lần chạy trước)
    → Document(page_content = nội dung Chương, metadata = {"source": tên file})
    → transforms = default_transforms(...) rồi hạ max_token_limit của mọi
      LLMBasedExtractor xuống ~4.000 và giảm RunConfig.max_workers xuống ~4 (mục 3.1:
      Groq free TPM 8K/tài khoản — mặc định ragas 32.000 token/lượt bị 413)
    → generate_with_langchain_docs(doc, testset_size=..., transforms=...,
                                   query_distribution=[(single_hop, w1), (abstract, w2),
                                                       (specific, w3)])
        (build KnowledgeGraph CỦA RIÊNG Chương này: tóm tắt, trích entity, dựng quan hệ
         trong nội bộ Chương — tốn nhiều lượt gọi, rải qua 6 tài khoản round-robin; số câu
         và trọng số từng loại lấy từ `allocate_questions` bên dưới)
    → lưu KG ra data/eval/knowledge_graph/<tên>__<chương>.json
    → convert → list[GoldenTestCase] (gắn source_document=<tên>, source_section=<chương>)
      → NỐI vào data/eval/golden_testset_raw.json NGAY, rồi ghi đơn vị vào
        generation_progress.json (checkpoint theo đơn vị, mục 4.5)
    → nếu lỗi/hết quota giữa chừng: DỪNG cả chương trình (mục 4.5), không thử đơn vị kế

[thao tác tay, ngoài code] người vận hành mở golden_testset_raw.json, đọc lướt, xoá câu
  vô nghĩa/lặp, ghi đè lại file
lệnh finalize (mục 4.2): raw đã review → đúng 180 câu → golden_testset.json
```

**Cắt đơn vị (`unit_splitter.split_document`, hàm thuần, ĐÃ implement — mục 4.4):** cắt tại
`## ` (Chương), tách theo `### ` (Mục) khi quá lớn, gộp phần nhỏ (`MIN_UNIT_CHARS`), bỏ phần
mở đầu và khối chú thích cuối file. `source_section` của đơn vị gộp là các nhãn nối bằng
" + " (vd. "Chương III + Chương IV"). Quy tắc chi tiết và số đo ở mục 4.4.

**Quota câu theo đơn vị** (`allocate_questions`, hàm thuần): tổng `GENERATE_SIZE = 240` chia
thành 3 loại đúng **192 / 24 / 24** (single-hop / multi-hop abstract / multi-hop specific,
80/10/10); mỗi loại phân bổ cho các đơn vị **tỷ lệ theo số ký tự**, làm tròn bằng phương
pháp phần dư lớn nhất để tổng từng loại đúng. Đơn vị nào nhận 0 câu multi-hop thì chỉ chạy
single-hop; tổng số câu của đơn vị = tổng 3 loại, `query_distribution` truyền trọng số =
số câu từng loại chia tổng (developer xác nhận ragas làm tròn ra đúng số câu mong muốn —
số câu ragas trả có thể lệch vài câu, `finalize` chịu được). Multi-hop cần ≥2 đoạn liên
quan trong đơn vị nên có thể sinh ít hơn quota; không bù. `--testset-size` ghi đè tổng.

`GENERATE_SIZE`, `TARGET_SIZE` (ở `testset_generator.py`), `MIN_UNIT_CHARS`, `MAX_UNIT_CHARS` (ở `unit_splitter.py`) là hằng số nội bộ module, không phải biến
môi trường (theo coding-convention: constants chỉ thuộc một cơ chế, không biến mọi chi
tiết thành env var). Chạy **sau khi pilot đạt (mục 9.0)**. Đây là job offline không gấp,
có thể trải nhiều ngày theo quota Groq (mục 3.1).

**Chạy nhiều ngày / resume theo đơn vị:** xem mục 4.5 (file theo dõi tiến độ, thứ tự chạy
nhỏ → lớn, hành vi khi hết quota).

### 4.1 Ngôn ngữ và chất lượng câu hỏi/đáp án — kết luận từ 4 lần pilot

**Kết luận (đã chốt, 2026-09-28):** prompt nội bộ ragas mặc định bằng tiếng Anh nên câu
hỏi/đáp án ra lẫn ngữ. **Bắt buộc** `adapt_prompts("vietnamese", llm=...)` cho mọi
synthesizer và **ép `QueryStyle.PERFECT_GRAMMAR`** (mã mẫu ở mục 4.3). Với cấu hình này,
pilot Groq `gpt-oss-120b` cho câu hỏi tiếng Việt sạch 12/12 (single-hop). Chi tiết 4 lần
pilot bên dưới (giữ làm bằng chứng đo).

**Kết quả pilot lần 1 (2026-09-28, đoạn 3.067 ký tự luật thuế TNCN, 4 câu single-hop,
KHÔNG adapt): ngôn ngữ LẪN, cần `adapt_prompts`.** Câu hỏi: 1 tiếng Việt (nhưng nhiễu:
"cá nhâncư trû", "1 8 3 ngày"), 1 lẫn Việt-Anh ("What is the pham vi dieu chinh of Điều
1?"), 2 tiếng Anh; đáp án cũng có câu tiếng Anh. Còn thấy lỗi chất lượng văn bản: dính
chữ ("itemsexpressly", "thương mạiđiện tử"), sai dấu, và rò rỉ persona vào câu hỏi ("Tôi
là một việc viênthuế…"). Chưa rõ nhiễu này do prompt tiếng Anh hay do `gpt-oss-120b` với
tiếng Việt — kiểm ở lần pilot sau khi adapt. `reference_contexts` **nguyên văn** (3/3 đoạn
khớp đoạn nguồn). Sinh 4 câu khi yêu cầu 3: số câu ragas trả có thể lệch, `finalize` (mục
4.2) đã xử lý bằng cách đếm/cắt.

**Kết quả pilot lần 2 (2026-09-28, cùng đoạn 3.067 ký tự, CÓ `adapt_prompts("vietnamese")`
cho synthesizer, model `qwen/qwen3.8-27b:free` qua OpenRouter — KHÁC model dự kiến dùng
thật):**
- **`adapt_prompts` hiệu quả về ngôn ngữ:** 4/4 đáp án và 3/4 câu hỏi tiếng Việt, không còn
  tiếng Anh (lần 1 không adapt: 2/4 câu hỏi tiếng Anh). → **Quyết định: bước `adapt_prompts`
  sang tiếng Việt là bắt buộc** trong `build_testset_generator` (áp cho synthesizer; transforms
  chưa cần — pilot không thấy lỗi ngôn ngữ ở đó, nhưng chưa kiểm).
- **Nhiễu "không dấu/viết sai" là do thiết kế của ragas, không phải do model:** ragas chọn
  `QueryStyle` ngẫu nhiên trong cả 4 loại (`sample["styles"] = list(QueryStyle)` ở
  `single_hop/base.py`): MISSPELLED, POOR_GRAMMAR, WEB_SEARCH_LIKE chiếm 3/4. Câu #1-#2 của
  pilot ("...chiet tieu dieu nao de Nguyen Minh Anh...") là kiểu này, kèm rò rỉ **tên persona**
  vào câu hỏi. → **Quyết định: ép `PERFECT_GRAMMAR`** bằng subclass synthesizer ghi đè
  `prepare_combinations` (mẫu code trong `tools/pilot_testset.py`); áp cho cả 2 synthesizer
  multi-hop khi implement. Nhiễu có thể được thêm lại như một lát đánh giá độ bền riêng ở
  phase sau — không trộn vào testset chính.
- **Chất lượng nội dung:** 3/4 câu xoay quanh cùng một mệnh đề "Chính phủ quy định chi tiết"
  (Điều 2 khoản 4) — NER trích "Chính phủ" làm chủ đề nên câu hỏi nông và lặp; đáp án đôi
  chỗ dính chữ ("ngườinộp thuế"). Mẫu 4 câu quá nhỏ để kết luận về tỷ lệ bỏ (ước lượng thô:
  ~2/4 dùng được), nhưng nếu tỷ lệ bỏ thật ≈50% thì sinh 240 để còn 180 là thiếu — đánh
  giá lại sau pilot Groq (≥10 câu, phong cách sạch); hệ số dư (`GENERATE_SIZE`) có thể phải
  tăng (mục 1).
- **`reference_contexts` nguyên văn 4/4** (1.040 và 2.030 ký tự khớp đoạn nguồn).
- **Chi phí đo thật (Qwen, không suy ra được cho `gpt-oss-120b`):** 34.989 token cho 4 câu
  (input 14.299, output 20.690 — output chiếm 59% vì model suy luận) ≈ 8,7K token/câu ≈
  11,4 token/ký tự văn bản. Ước lượng tách: dựng KG ≈ 6-7 token/ký tự, sinh câu ≈ 3-4K
  token/câu → cả corpus + 240 câu cỡ **7-8M token** (cao hơn ước lượng trước 5-6M). Chưa
  có số cho `gpt-oss-120b`: cần pilot Groq. Lưu ý số "109 lượt gọi" trong log gồm cả các
  lượt bị 429 rồi fallback sang key khác, không phải số lượt thành công.
- **Round-robin chia đều 4 key** (28/27/27/27 lượt thử).
- Thời gian 739s (12 phút) cho 4 câu do OpenRouter free bị 429 liên tục — không đại diện
  cho Groq.

**Kết quả pilot lần 3 (2026-09-28, đoạn 3.067 ký tự, Groq `gpt-oss-120b`, CÓ
`adapt_prompts` + ép `PERFECT_GRAMMAR`, 8 câu single-hop):**
- **Ngôn ngữ 8/8 tiếng Việt, không còn nhiễu không dấu/viết sai/tên persona** → cấu hình
  `adapt_prompts("vietnamese")` + `PERFECT_GRAMMAR` đạt cho single-hop. Multi-hop CHƯA kiểm.
- `reference_contexts` nguyên văn 8/8; đủ trường; 8/8 câu như yêu cầu.
- **Token thật (`gpt-oss-120b`): 29.702 cho 8 câu** (input 19.442, output 10.260) ≈ 3,7K/câu
  ≈ 9,7 token/ký tự văn bản; 399s, không bị 429. Ước lượng tách (chưa đo tách được): dựng KG
  ≈ 4 token/ký tự, sinh câu ≈ 1,5K token/câu → cả corpus + 240 câu ≈ **5M token** (khoảng
  4-7M) ≈ 6 ngày ở ~800K token/ngày (mục 3.1). Số này thay cho các ước lượng trước.
- **Chất lượng: ~6/8 dùng được** (#2 hời hợt, #3 đáp án chỉ "Chính phủ quy định chi tiết Điều
  này"); #5-#7 trùng ý (đoạn mẫu chỉ có 2 đoạn ragas). Ước lượng bỏ ~25-40% khi đọc lướt →
  240 sinh có thể chỉ còn ~145-180: giữ 240 và dựa vào sinh bù (`--append`, mục 4.2); nâng
  `GENERATE_SIZE` lên ~270 nếu muốn chắc hơn.
- **Đáp án hay bị dính chữ** (5/8 đáp án: "cưtrú", "thườngxuyên", "kinhdoanh", "hoạtđộng",
  "bằngtiền", "khoản 1Điều 2"; câu hỏi hiếm hơn: "nàovà"); cũng thấy ở pilot lần 1 (Groq) và
  lần 2 (Qwen), tức không riêng một model.
- Round-robin chỉ dùng key 1-3 (7/7/7): code Phase 1 hiện vẫn tạo 3 client (việc của
  developer, mục 3.1/6).

**Kết quả pilot lần 4 (2026-09-28, cùng cấu hình lần 3, 4 câu; chẩn đoán ký tự lạ):**
- **Giả thuyết "ký tự khoảng trắng lạ" BỊ BÁC BỎ:** 4/4 câu báo "không có" ký tự lạ
  (Zs/Cf/Cc khác dấu cách thường) trong câu hỏi và đáp án. Vậy chữ dính (lần này 1/4 đáp
  án: "dịchvụ") là **thiếu dấu cách thật do model sinh ra** (`gpt-oss-120b` với tiếng Việt),
  không sửa được bằng chuẩn hoá Unicode → **KHÔNG thêm bước NFKC**. Tần suất thay đổi giữa
  các lần chạy (5/8 rồi 1/4 đáp án). **Quyết định: chấp nhận là khuyết điểm nhỏ đã biết** —
  `reference` chỉ được LLM chấm (mục 2.1), LLM đọc được chữ dính nên ít ảnh hưởng điểm; người
  dùng thấy khi đọc lướt (mục 9.4); ghi nhận ở mục 10.12. Không sửa bằng code.
- **Token thật, điểm đo thứ hai:** 23.572 cho 4 câu (input 15.014, output 8.558), 17 lượt gọi,
  302s. Hai điểm đo Groq (N=4: 23.572; N=8: 29.702) cho **ước lượng tách**: sinh câu ≈ 1,5K
  token/câu, phần cố định (dựng KG cho 3.067 ký tự + persona + kịch bản + adapt) ≈ 17,5K →
  dựng KG ≈ **5 token/ký tự** (sai số lớn vì chỉ 2 điểm). Cả corpus (781.007 ký tự; bản đầu tính nhầm 1,04M là số byte) + 240 câu ≈
  3,9M + 0,4M ≈ **~4,2M token** (khoảng 3-5,5M) ≈ **5 ngày** ở ~800K token/ngày (mục 3.1).
  Số này thay cho ước lượng ở lần 3.
- **Trùng ý trong cùng một đoạn ragas:** #1-#2 cùng hỏi định nghĩa "cá nhân cư trú", #3-#4
  cùng hỏi "loại thu nhập chịu thuế" (4 câu → 2 chủ đề). Đoạn mẫu chỉ có 2 đoạn ragas nên
  trùng nhiều; trên Chương thật sẽ đa dạng hơn, nhưng vẫn phải đọc lướt bỏ trùng (mục 9.4).
- 4/4 tiếng Việt, `reference_contexts` nguyên văn 4/4; yêu cầu 3 câu, ragas trả 4.

### 4.3 Mã mẫu đã kiểm chứng bằng pilot (để developer dùng; script pilot tạm đã xoá)

```python
# 1) Hạ max_token_limit: Groq free TPM 8K/tài khoản, mặc định ragas 32.000 token/lượt -> 413.
def cap_token_limit(transforms, limit: int) -> int:
    if isinstance(transforms, Parallel):  # ragas.testset.transforms.engine
        transforms = transforms.transformations
    if isinstance(transforms, LLMBasedExtractor):  # ragas.testset.transforms.base
        transforms.max_token_limit = limit
        return 1
    if isinstance(transforms, list):
        return sum(cap_token_limit(item, limit) for item in transforms)
    return 0


transforms = default_transforms(  # ragas.testset.transforms.default
    documents=documents, llm=generator.llm, embedding_model=generator.embedding_model
)
cap_token_limit(transforms, 4000)  # số extractor đổi được = 4 (pilot)


# 2) Ép phong cách câu hỏi sạch (ragas mặc định trộn 4 QueryStyle, 3/4 là nhiễu).
@dataclass
class CleanSingleHopSynthesizer(SingleHopSpecificQuerySynthesizer):
    def prepare_combinations(self, *args, **kwargs):
        combos = super().prepare_combinations(*args, **kwargs)
        for combo in combos:
            combo["styles"] = [
                QueryStyle.PERFECT_GRAMMAR
            ]  # ragas.testset.synthesizers.base
        return combos


# 3) adapt_prompts sang tiếng Việt (2 prompt của synthesizer) + giảm đồng thời.
synthesizer = CleanSingleHopSynthesizer(llm=generator.llm)
adapted = asyncio.run(synthesizer.adapt_prompts("vietnamese", llm=generator.llm))
synthesizer.set_prompts(**adapted)
testset = generator.generate_with_langchain_docs(
    documents,
    testset_size=n,
    transforms=transforms,
    query_distribution=[(synthesizer, 1.0)],
    run_config=RunConfig(max_workers=4),  # ragas.run_config; mặc định 16
)
```

Lưu ý khi implement (chưa kiểm chứng ở pilot): (a) đã kiểm cho **single-hop**; hai synthesizer
multi-hop (`MultiHopAbstractQuerySynthesizer`, `MultiHopSpecificQuerySynthesizer`) cần override
tương tự và xác nhận chữ ký `prepare_combinations` cũng như việc `styles` có tác dụng, vì
20% testset là multi-hop; (b) `adapt_prompts` gọi LLM (~2 lượt) — gọi một lần cho mỗi
synthesizer rồi dùng lại cho mọi Chương, không gọi lại mỗi Chương; (c) `LangchainLLMWrapper`
và `LangchainEmbeddingsWrapper` đang deprecated ở `ragas==0.4.3` (vẫn chạy) — giữ nguyên
version đã ghim.

### 4.2 Chốt đúng 180 câu (`finalize`) và sinh bù (`--append`)

**`finalize`** (hàm thuần trong `testset_generator.py` + lệnh CLI): đọc
`golden_testset_raw.json` đã review, chọn ra đúng `TARGET_SIZE = 180` câu → ghi
`golden_testset.json`.

- **Cắt phân tầng theo (`source_document`, `synthesizer_name`)**, tất định: chia các dòng
  vào nhóm theo cặp đó; mỗi dòng nhận khoá `(thứ hạng trong nhóm theo thứ tự file + 0,5) /
  kích thước nhóm`; lấy 180 dòng có khoá nhỏ nhất (hoà thì theo thứ tự file). Cách này giữ
  tỷ lệ theo văn bản và theo loại câu sau khi người dùng đã xoá, mà không cần bảng quota
  cứng — và không dồn hết các câu bị giữ vào vài văn bản đầu file (lỗi của cách "lấy N
  dòng đầu").
- Tổng số câu còn lại **< 180** → dừng với lỗi nêu rõ thiếu bao nhiêu và gợi ý lệnh sinh bù;
  không ghi `golden_testset.json` thiếu.
- Không kiểm tra/chấm chất lượng nội dung: chỉ đếm và cắt (review là việc người dùng).

**Sinh bù (`generate --only <tên văn bản> --reuse-knowledge-graph --append --testset-size
N`):** nạp lại các `knowledge_graph/<văn bản>__<chương>.json` (không build lại đồ thị), sinh `N` câu mới (chia cho các Chương của văn bản đó) rồi
**nối** vào `golden_testset_raw.json` hiện có, bỏ qua câu có `user_input` trùng y hệt câu
đã có. Câu mới có thể trùng ý (không trùng chữ) với câu cũ — người dùng đọc lướt phần mới
thêm như lúc đầu. Sau đó chạy `finalize` lại.

### 4.4 Chia đơn vị theo Chương/Mục và lập kế hoạch token (đo 2026-09-28)

**Số đo cấu trúc** (số **ký tự**, không phải byte; corpus 781.007 ký tự): 6 văn bản có 4-17
Chương; `Quy định mức lương tối thiểu.md` **không có** tiêu đề `## ` (29.075 ký tự, coi cả
file là 1 đơn vị ≈ 164K token). `Điều kiện lao động và quan hệ lao động.md` có **hai** Chương
đánh số "XI" → khoá đơn vị phải dùng **số thứ tự** chứ không dùng số La Mã. Chương lớn
nhất: BHXH Chương V 75K ký tự (52 Điều, 4 Mục), Điều kiện lao động Chương XI 44,5K
(4 Mục) và IV 34,8K (4 Mục), Bộ luật lao động Chương III 42K (5 Mục) và XIV 31K (5 Mục);
BHYT Chương X 48,5K ký tự nhưng chỉ 3 Điều — **46,8K ký tự trong đó là khối chú thích**
`[1]…[114]` ("Khoản này được sửa đổi theo Luật số…") nằm sau dòng `---` cuối file, không phải điều
luật (xem quy tắc "bỏ chú thích" bên dưới).

**Quy tắc chia (chốt cùng người dùng):**
- **Đơn vị chuẩn = Chương.** Chương lớn hơn `MAX_UNIT_CHARS = 30.000` ký tự thì **tách
  theo `### Mục`**; Mục nhỏ hơn `MIN_UNIT_CHARS = 6.000` gộp với Mục/Chương liền kề (mọi
  Chương lớn đo được đều có Mục, Mục lớn nhất 27K ký tự nên đủ dưới ngưỡng). Đơn vị sau
  chia ≤ ~30K ký tự ≈ ≤ 165K token, đủ nhỏ để một ngày quota chứa được nhiều đơn vị.
- **KHÔNG bỏ Chương "ĐIỀU KHOẢN THI HÀNH"** (người dùng chốt 2026-09-28). Hệ quả: các
  Chương này vẫn sinh câu; nhiều câu có thể vô nghĩa → bị xoá khi đọc lướt (mục 9.4), có thể
  cần sinh bù. Chương nhỏ được gộp với Chương kề (vd. BHYT "Chương IX + Chương X").
- **Bỏ khối chú thích cuối file** (`_strip_footnotes`; quyết định 2026-09-28, đo được ở 4/6
  văn bản: BHXH 10, BHYT 114, TNCN 3, BLLĐ 1 chú thích): cắt từ dòng `---` đứng ngay trước
  dòng `[1] …` cuối file. Đây là chỉ dẫn văn bản sửa đổi, không phải điều luật; sinh câu hỏi
  từ đó chỉ ra rác và tốn ~300K token (4,47M → 4,16M). **Ngoài quy tắc gốc "không bỏ Điều
  khoản thi hành"** — Chương Điều khoản thi hành vẫn giữ nguyên nội dung điều luật của nó.
  Phần mở đầu trước Chương đầu (quốc hiệu, căn cứ ban hành) cũng bỏ.
- **Fallback khi một phần vẫn > `MAX_UNIT_CHARS` sau khi tách Mục** (chưa gặp ở corpus hiện tại
  sau khi bỏ chú thích): tách tiếp theo `#### Điều`, rồi theo đoạn văn (dòng trống), gói
  cân bằng ≤ `MAX_UNIT_CHARS`.
- **Ước lượng chi phí một đơn vị** (dùng cho `--dry-run`): `≈ 5,5 token/ký tự + 4K cố
  định` (5 token/ký tự dựng KG + ~0,5 token/ký tự cho câu hỏi khi 240 câu chia đều theo ký
  tự; 4K = persona/kịch bản/headline cố định). Sai số ±30% — chỉ 2 điểm đo; số thật ghi lại
  sau mỗi đơn vị xong để cập nhật hệ số.
- **Kết quả áp dụng** (đã chạy `tools/split_eval_units.py`, 2026-09-28): **50 đơn vị**
  (BHXH 12, BHYT 6, TNCN 3, mức lương tối thiểu 1, BLLĐ hợp nhất 16, Điều kiện lao động 12),
  **720.573 ký tự, ~4,16M token**; đơn vị nhỏ nhất 6.000 ký tự (~37K token), lớn nhất 29.075
  ký tự (mức lương tối thiểu, cả văn bản, ~163K token), không đơn vị nào vượt 30.000. Kế
  hoạch và nguyên văn từng đơn vị lưu ở `data/eval/units_plan.md` (bảng sắp xếp nhỏ → lớn,
  cột token cộng dồn) và `data/eval/units/<văn bản>__<số thứ tự>.md`.

**Lập kế hoạch theo ngày:** không cần chọn tay — thứ tự chạy nhỏ → lớn và cơ chế dừng/tiếp
tục ở mục 4.5 tự dùng hết quota mỗi ngày (~1,2M token/6 tài khoản, mục 3.1). Đơn vị lỗi giữa
chừng mất toàn bộ token đã tiêu của nó — đơn vị càng nhỏ thì rủi ro lãng phí càng thấp, nên
chạy nhỏ trước cũng là chiến lược giảm lãng phí. Hệ quả cần biết: nếu dừng giữa chừng (chưa
chạy hết 50 đơn vị) thì testset lệch về các đơn vị nhỏ; `finalize` chỉ cân bằng đúng khi đã
chạy đủ. `--only` vẫn dùng được để chọn đơn vị cụ thể.

### 4.5 Theo dõi tiến độ và chạy tiếp nhiều ngày (chốt 2026-09-28)

**Yêu cầu:** hôm nay chạy tới đơn vị thứ 36 thì hết quota, chương trình dừng; mai chạy lại
CÙNG một lệnh thì làm tiếp từ đơn vị 37, không chạy lại 36 đơn vị đã xong.

**Thứ tự chạy** = số ký tự tăng dần (nhỏ trước, khớp `units_plan.md`); hai đơn vị bằng nhau
thì theo (tên văn bản, số thứ tự). Thứ tự xác định từ văn bản nguồn nên mỗi lần chạy giống
nhau. **Khoá đơn vị** = `<tên file .md>#<số thứ tự đơn vị trong văn bản>` (số thứ tự 1-based
của `split_document`, không đổi khi sắp xếp lại; dùng số thứ tự vì ĐKLĐ có hai "Chương XI").

**File `data/eval/generation_progress.json`** (Pydantic `GenerationProgress`, mục 5), ghi
**nguyên tử** (ghi file tạm rồi đổi tên) SAU KHI đơn vị xong và raw đã được nối:

```json
{
  "units": {
    "Luật bảo hiểm y tế.md#6": {
      "title": "Chương IX + Chương X", "chars": 6361, "estimated_tokens": 38000,
      "questions": {"single_hop": 4, "abstract": 1, "specific": 0},
      "llm_calls": 61, "seconds": 412.5, "completed_at": "2026-09-29T09:41:07+07:00"
    }
  },
  "last_failure": {
    "unit": "Văn bản hợp nhất bộ luật lao động.md#5", "error": "RateLimitError: ...",
    "at": "2026-09-29T10:12:55+07:00"
  }
}
```

- **File này (không phải `golden_testset_raw.json`) quyết định đơn vị nào đã xong.** Lý do:
  người dùng xoá dòng xấu trong raw khi đọc lướt — nếu suy trạng thái từ raw thì đơn vị bị
  xoá hết dòng, hoặc đơn vị sinh 0 câu, sẽ bị chạy lại và tốn token oan.
- **Thứ tự ghi:** nối raw trước, ghi progress sau. Nếu chương trình chết giữa hai bước, lần
  chạy sau thấy đơn vị có dòng trong raw (theo `source_document` + `source_section`) mà
  chưa có trong progress → **coi là đã xong** (ghi bổ sung vào progress, log rõ), không sinh
  lại (tránh trùng câu).
- **Khi hết quota/lỗi giữa đơn vị** (429 sau khi mọi tài khoản đã thử, timeout, 413...):
  đơn vị đang dở KHÔNG được ghi vào raw/KG/`units`; ghi `last_failure` (tên đơn vị, loại lỗi,
  thời điểm, không có nội dung câu hỏi/context); in tóm tắt tiến độ; **dừng luôn, không thử
  đơn vị kế** (quota các tài khoản chung nhau nên đơn vị kế cũng sẽ lỗi, chỉ đốt thêm token
  vào lượt dở); thoát với mã khác 0. Lần chạy sau thành công thì xoá `last_failure`.
- **Kiểm tra khớp nguồn:** mỗi lần chạy, đơn vị đã có trong `units` mà `chars` khác với kết
  quả chia hiện tại (văn bản `.md` hoặc quy tắc chia đã đổi) → dừng với lỗi rõ chỉ tên đơn vị,
  không tự bỏ qua hay ghi đè (số thứ tự có thể đã trỏ sang đơn vị khác).
- **Chạy lại một đơn vị** (muốn sinh thêm, hoặc sinh lại vì hỏng): `generate --only
  "<tên>#<số>" --append` — bỏ qua kiểm tra "đã xong", nối thêm dòng vào raw (không xoá dòng
  cũ) và cộng dồn `questions`/`llm_calls` trong `units`. Muốn sinh lại từ đầu thì người dùng
  tự xoá dòng của đơn vị đó trong raw trước.
- **Xem tiến độ không tốn token:** `generate --dry-run` in bảng theo thứ tự chạy gồm: thứ tự,
  khoá đơn vị, ký tự, ước lượng token, **trạng thái** (`xong dd/mm hh:mm` / `chưa` / `lỗi lần
  cuối`), rồi dòng tóm tắt "đã xong N/50 đơn vị, còn M đơn vị ước lượng ~X token (≈ Y ngày
  ở 1,2M token/ngày)". Đơn vị chạy tiếp theo được đánh dấu.
- **Số đo thật:** `llm_calls`, `seconds` ghi lại theo đơn vị để chỉnh hệ số ước lượng
  5,5 token/ký tự (mục 4.4). Nếu đếm được token thật từ response Groq (`usage`) thì ghi
  thêm `tokens`; không bắt buộc ở Phase 1.

Hôm sau chỉ cần chạy đúng một lệnh, không cần nhớ đã dừng ở đâu:
`tools/generate_testset.py generate` (không `--only`) — tự chạy tiếp theo thứ tự nhỏ → lớn.

## 5. Model dữ liệu (`models.py`)

```python
class GoldenTestCase(BaseModel):
    """Một câu hỏi mẫu trong golden testset, sinh tự động bởi ragas."""

    user_input: str
    reference: str
    reference_contexts: list[str]
    synthesizer_name: str | None = None
    source_document: str | None = (
        None  # tên file .md nguồn, do code gắn (ragas không trả)
    )
    source_section: str | None = (
        None  # tiêu đề Chương (hoặc các Chương gộp), do code gắn
    )
```

```python
class UnitProgress(BaseModel):
    """Kết quả của một đơn vị đã sinh xong (mục 4.5)."""

    title: str
    chars: int  # để phát hiện văn bản/quy tắc chia đã đổi so với lúc sinh
    estimated_tokens: int
    questions: dict[str, int]  # single_hop / abstract / specific, cộng dồn khi --append
    llm_calls: int
    seconds: float
    completed_at: datetime


class UnitFailure(BaseModel):
    """Lần dừng gần nhất (không chứa nội dung câu hỏi/context)."""

    unit: str  # khoá "<tên file>#<số thứ tự>"
    error: str
    at: datetime


class GenerationProgress(BaseModel):
    """Nội dung `generation_progress.json`: khoá đơn vị -> `UnitProgress`."""

    units: dict[str, UnitProgress] = {}
    last_failure: UnitFailure | None = None
```

`source_document`/`source_section` không do ragas sinh: `generate` biết đang xử lý đơn vị
nào nên gắn vào mỗi câu; dùng cho resume theo Chương (mục 4), cắt phân tầng ở `finalize`
(mục 4.2), và để Phase 2 báo điểm theo từng luật/Chương.

Không cần model phức tạp hơn ở Phase 1; không lẫn `GoldenTestCase` với bất kỳ contract
nào của `retrieval/`/`generation/` (đây là dữ liệu tham chiếu tĩnh, không phải
`RetrievedChunk`).

## 6. Config (`config.py`)

Thêm vào `config.py` gốc (cạnh `GenerationSettings`, `JudgeSettings`... theo đúng pattern
`pydantic-settings` hiện có):

```python
class TestsetGeneratorSettings(BaseSettings):
    """Cấu hình 6 tài khoản Groq round-robin cho generator_llm (Phase 1 RAGAS, mục 3.1).

    Cả 6 key BẮT BUỘC (không optional/fallback như GenerationSettings/JudgeSettings) —
    round-robin chỉ có ý nghĩa khi đủ 6 tài khoản độc lập; thiếu key nào, pydantic báo lỗi
    rõ ràng ngay lúc khởi tạo thay vì âm thầm chạy round-robin với ít tài khoản hơn.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: str = Field(validation_alias="GROQ_API_KEY")
    api_key_2: str = Field(validation_alias="GROQ_API_KEY_2")
    api_key_3: str = Field(validation_alias="GROQ_API_KEY_3")
    api_key_4: str = Field(validation_alias="GROQ_API_KEY_4")
    api_key_5: str = Field(validation_alias="GROQ_API_KEY_5")
    api_key_6: str = Field(validation_alias="GROQ_API_KEY_6")
    model_name: str = "openai/gpt-oss-120b"
    max_retries: int = 2
    timeout_seconds: int = 60
```

- Dùng cả 6 khoá `GROQ_API_KEY`/`_2`…`_6` (6 tài khoản Groq độc lập, xác nhận bởi
  người dùng) qua round-robin (mục 3.1) — khác mục đích tách ngân sách theo bước của
  `GenerationSettings`/`JudgeSettings` (`GenerationSettings` cũng đọc `GROQ_API_KEY_3`/
  `_4`, nhưng để xoay vòng cho generation production), nên **không dùng chung class** với
  chúng dù tên biến trùng.
- `model_name` dùng lại `gpt-oss-120b` (không phải `20b` như condense) vì sinh câu hỏi
  multi-hop cần khả năng tổng hợp/suy luận qua nhiều đoạn văn bản.
- Không thêm setting riêng cho embeddings: `embeddings_adapter.py` tái dùng thẳng
  `EmbeddingSettings` đã có (`config.py`).
- Module không đọc `.env` trực tiếp. `.env.example` mới có `GROQ_API_KEY`…`_4` — **thêm
  `GROQ_API_KEY_5`, `GROQ_API_KEY_6`** và xoá 4 dòng `OPENROUTER_API_KEY*` (chỉ phục vụ pilot
  đã xong); bổ sung ghi chú: 6 key này cũng được dùng round-robin bởi `tools/generate_testset.py`
  (chạy khi không có traffic thật, mục 3.1).

## 7. Module (`src/production_legal_qa_rag/evaluation/`)

| Module | Trách nhiệm duy nhất |
| --- | --- |
| `models.py` | `GoldenTestCase` (Pydantic v2). |
| `corpus_loader.py` | Đọc `data/markdown/*.md` → `list[langchain_core.documents.Document]`. |
| `embeddings_adapter.py` | Adapter implement interface `Embeddings` của LangChain quanh `InferenceClient.feature_extraction` (`embedding/hf_client.py`), không word-segment (mục 3). |
| `groq_round_robin.py` | `GroqRoundRobinChatModel` — proxy `BaseChatModel` luân phiên 6 `ChatOpenAI` (mục 3.1). |
| `unit_splitter.py` | **Đã implement:** `EvalUnit` (Pydantic: `source_document`, `index`, `title`, `text`, `char_count`, `estimated_tokens`), `split_document`, `split_directory` (chia Chương/Mục, gộp, bỏ chú thích, thuần Python, không import `ragas`); hằng số `MAX_UNIT_CHARS=30.000`, `MIN_UNIT_CHARS=6.000`, `TOKENS_PER_CHAR=5,5`, `FIXED_TOKENS_PER_UNIT=4.000`. Test: `tests/test_evaluation_unit_splitter.py`. |
| `tools/split_eval_units.py` | **Đã implement:** CLI chia + ghi `data/eval/units/` và `data/eval/units_plan.md` để duyệt, không gọi LLM. Khi lệnh `generate --dry-run` xong (mục 4.5) thì CLI này chỉ còn dùng để xem nguyên văn đơn vị; giữ lại, không xoá. |
| `testset_generator.py` | Điều phối: khởi tạo `TestsetGenerator` (+ transforms hạ `max_token_limit`, `adapt_prompts` tiếng Việt, 3 synthesizer ép `PERFECT_GRAMMAR` — mục 4.3); **vòng lặp từng văn bản rồi từng Chương** (bỏ qua đơn vị đã xong theo `generation_progress.json`, mục 4.5): `generate_with_langchain_docs` → lưu `KnowledgeGraph` riêng + nối `golden_testset_raw.json` ngay; `allocate_questions` (quota 192/24/24 theo đơn vị); `finalize_testset(raw_cases, target=TARGET_SIZE)` cắt phân tầng đúng 180 câu (mục 4.2). Hằng số `GENERATE_SIZE=240`, `TARGET_SIZE=180`, `MIN_UNIT_CHARS`. |
| `tools/generate_testset.py` | Typer app mỏng với **2 lệnh**, chỉ gọi `testset_generator.py`, không chứa business logic. `generate`: `--markdown-dir`, `--output-dir`, `--only <tên>[#<số thứ tự đơn vị>]` (lặp lại được; chọn cả văn bản hoặc từng đơn vị), `--dry-run` (in kế hoạch theo thứ tự chạy nhỏ → lớn: đơn vị, ước lượng token, trạng thái xong/chưa/lỗi lần cuối, tóm tắt tiến độ; KHÔNG gọi LLM — mục 4.4, 4.5), `--reuse-knowledge-graph`, `--append`, `--testset-size` (tổng, mặc định `GENERATE_SIZE=240`). `finalize`: `--output-dir` → đọc raw, ghi `golden_testset.json` (đúng 180). Hiện `generate_golden_testset` chỉ xử lý cả thư mục một lần và ghi `golden_testset.json` — cần viết lại theo vòng lặp từng văn bản rồi từng Chương. |

Ghi chú vị trí: đặt ngay ở package `evaluation/` (không phải chỉ 1 script rời trong
`tools/`) vì Phase 2 sẽ thêm module chạy eval (`run_eval.py`/tương đương) vào **cùng**
package này — tránh phải dời code khi mở rộng.

## 8. Xử lý lỗi

| Sự cố | Xử lý |
| --- | --- |
| Groq lỗi/quota (429 hết token/ngày)/timeout giữa lúc build `KnowledgeGraph` hoặc sinh câu hỏi của MỘT Chương | Dừng chương trình (không thử đơn vị kế), log lỗi rõ ràng (không log nội dung câu hỏi/context — theo chính sách log chung của project) kèm **khoá đơn vị đang dở**, ghi `last_failure` vào `generation_progress.json`. Đơn vị đang dở KHÔNG được lưu (không ghi KG/raw dở dang); các đơn vị đã xong trước đó giữ nguyên. Chạy lại `generate` ngày hôm sau sẽ làm tiếp từ đơn vị đó (mục 4.5). Không resume trong lòng một đơn vị (mục 10.11). |
| `--only` nêu tên văn bản không tồn tại trong `data/markdown/` | Raise lỗi rõ liệt kê tên hợp lệ, trước khi gọi Groq. |
| `embeddings_adapter.py` trả response sai định dạng (khác kỳ vọng của HF API) | Raise lỗi rõ, không âm thầm trả vector rỗng — cùng nguyên tắc validate ở biên như `embedding/hf_client.py`. |
| File `data/markdown/*.md` trống hoặc thiếu | Raise lỗi rõ trước khi gọi `TestsetGenerator` (fail fast, không lãng phí LLM call). |
| Một tài khoản Groq bị 429 giữa vòng round-robin (mục 3.1) | Thử ngay tài khoản kế tiếp trong vòng lặp, tối đa `len(clients)` lần cho MỘT lượt gọi; hết vòng vẫn lỗi thì raise nguyên lỗi cuối — không giữ vòng lặp vô hạn, không tự ý bỏ qua câu hỏi đó. |
| Thiếu `GROQ_API_KEY_2`…`_6` trong `.env` | `TestsetGeneratorSettings` raise lỗi validate ngay lúc khởi tạo (fail fast) — không âm thầm chạy round-robin với ít hơn 6 tài khoản. |
| `generate` gặp đơn vị đã có trong `generation_progress.json` mà không có `--append` | Bỏ qua đơn vị đó, log rõ "đã xong" (không gọi Groq, không ghi đè) — vừa là resume vừa bảo vệ công review tay (mục 4.5). Không bao giờ ghi đè/sắp xếp lại các dòng có sẵn trong file raw (người dùng đã sửa tay). |
| Đơn vị đã xong nhưng `chars` trong progress khác kết quả chia hiện tại | Dừng với lỗi nêu tên đơn vị (nguồn hoặc quy tắc chia đã đổi); không tự bỏ qua/ghi đè (mục 4.5). |
| Đơn vị có dòng trong raw nhưng chưa có trong progress (chết giữa hai bước ghi) | Coi là đã xong, ghi bổ sung vào progress và log; không sinh lại. |
| `generation_progress.json` hỏng/không parse được | Raise lỗi rõ; không âm thầm coi như "chưa làm gì" (sẽ đốt lại toàn bộ quota). |
| `finalize`: sau review còn ít hơn 180 câu | Dừng với lỗi nêu rõ thiếu bao nhiêu + gợi ý `generate --only <tên> --reuse-knowledge-graph --append` (mục 4.2); không ghi `golden_testset.json` thiếu. |
| `finalize`: `golden_testset_raw.json` không tồn tại hoặc một dòng thiếu/rỗng `user_input`/`reference`/`reference_contexts` | Raise lỗi rõ chỉ ra vị trí dòng lỗi (validate bằng `GoldenTestCase`), không âm thầm bỏ qua. |

## 9. Nghiệm thu thủ công

0. **Pilot (người dùng + architect chạy chung, ngoài vòng developer, TRƯỚC khi chạy full).**
   **Pilot thăm dò ĐÃ XONG (2026-09-28, 4 lần, mục 4.1)** bằng script tạm (đã xoá) trên 1
   đoạn 3.067 ký tự: đã trả lời ngôn ngữ, phong cách câu hỏi, `reference_contexts` nguyên
   văn, giới hạn Groq, chi phí token. **Còn lại — chạy sau khi developer xong code, bằng CLI
   thật:** `generate --only "Quy định mức lương tối thiểu" --testset-size ~10
   --output-dir data/eval_pilot` (văn bản nhỏ nhất, không đè `data/eval/`), để kiểm những
   thứ pilot thăm dò chưa chạm tới:
   - **Multi-hop:** 2 synthesizer multi-hop có sinh được, có tiếng Việt sạch không, câu
     multi-hop có ≥2 đoạn `reference_contexts` không; `query_distribution` với trọng số lẻ
     có ra đúng số câu từng loại không.
   - **Đơn vị thật đủ lớn cho ragas:** đơn vị nhỏ nhất (~6.000 ký tự) có dựng được KG và sinh
     được câu không, không bị lỗi "tài liệu quá ngắn".
   - **Resume và tái dùng KG:** chạy lại lệnh bỏ qua đơn vị đã xong theo
     `generation_progress.json` (kể cả sau khi xoá hết dòng của một đơn vị trong raw);
     `--reuse-knowledge-graph` không build lại đồ thị. Thử dừng giữa chừng (Ctrl+C hoặc gỡ một
     key để ép lỗi) rồi chạy lại: đơn vị dở không nằm trong progress, `last_failure` được ghi,
     lần sau tiếp tục đúng chỗ.
   - **Round-robin 6 key thật:** cả 6 key nhận tải (pilot thăm dò chỉ thấy 3 key vì code cũ).
   Chỉ chạy full khi pilot này đạt. (Đã thử dò lỗi bằng OpenRouter free: chỉ 50 request/ngày/
   tài khoản và kho chung upstream hay 429 — không đáng dùng cho việc này.)
0b. **Trước khi chạy full:** người dùng đã cam kết dùng cả 6 key cho job này và không chạy
   việc khác song song (mục 3.1) — xác nhận không có traffic thật/stack production đang
   dùng bucket `gpt-oss-120b` của key 3, 4.
1. Chạy `uv run --group eval --no-group production tools/generate_testset.py generate
   [--only <tên>]...` → mỗi Chương xong thì có `data/eval/knowledge_graph/<văn
   bản>__<chương>.json` và các dòng của nó nối vào `data/eval/golden_testset_raw.json`. Chia
   nhiều ngày theo quota Groq (mục 3.1): mỗi ngày chạy tới khi hết quota (dừng ở đơn vị
   đang dở, hôm sau chạy lại đúng lệnh đó sẽ tự bỏ qua đơn vị đã xong và chạy tiếp, mục 4.5;
   xem tiến độ bằng `generate --dry-run`). Không cần tối ưu tốc độ ở
   Phase 1 (mục 1, mục 4).
2. Mở `golden_testset_raw.json`: khi đủ 6 văn bản có khoảng 240 dòng (192/24/24 theo loại
   câu, có thể lệch vài câu), mỗi dòng có đủ `user_input`, `reference`,
   `reference_contexts` không rỗng, `source_document`, `source_section`.
3. Kiểm tra round-robin hoạt động thật (mục 3.1): log số lượt gọi theo từng tài khoản (hoặc
   nhìn dashboard usage của cả 6 tài khoản Groq sau khi chạy) — mỗi tài khoản nhận tải xấp xỉ
   nhau (~1/6 tổng số lượt gọi), không dồn hết vào 1 tài khoản; không có request nào thất bại
   hẳn vì rate limit (nếu có, bounded fallback ở mục 3.1/8 phải xử lý được).
4. **Người dùng đọc LƯỚT ~240 câu trong `golden_testset_raw.json`** (chốt 2026-09-28: chỉ
   đọc lướt, không đối chiếu đáp án với luật — hệ quả ở mục 10.9): xoá câu vô nghĩa, câu
   quá máy móc kiểu "so sánh Điều X và Điều Y" không giống câu hỏi tình huống thật, hoặc
   câu trùng lặp ý; ghi đè lại file raw. Không có ngưỡng số cứng, chấp nhận đánh giá định
   tính; có thể chia nhỏ việc đọc thành nhiều lần.
5. Chạy `tools/generate_testset.py finalize` → tạo `golden_testset.json` với **đúng 180
   câu** (cắt phân tầng theo văn bản và loại câu, mục 4.2). Nếu báo thiếu (còn <180 sau
   review): chạy `generate --only <tên> --reuse-knowledge-graph --append --testset-size
   <số thiếu + dư ~30%>` → xác
   nhận không build lại đồ thị (số lượt gọi Groq giảm rõ rệt so với lần chạy đầu) → đọc
   lướt phần mới thêm → `finalize` lại.

## 10. Rủi ro / điểm mở

1. `embeddings_adapter.py` là code mới, chưa có tiền lệ trong repo — cần test tay kỹ trước
   khi tin dùng cho việc build `KnowledgeGraph`.
2. Câu hỏi ragas sinh ra có thể lệch phân bố so với câu hỏi người dùng thật sẽ hỏi (thiên
   về cấu trúc bề mặt tài liệu — "Điều X quy định gì" — hơn là tình huống pháp lý cụ thể).
   Đây là hạn chế đã biết của synthetic data nói chung, xử lý bằng review thủ công ở Phase
   1, không có gate tự động.
3. Resume chỉ ở mức **Chương**, không ở mức lượt gọi LLM — xem mục 10.11 cho điều kiện
   (Chương lớn nhất phải vừa quota ngày).
4. Signature `TestsetGenerator(llm=..., embedding_model=...)`, `generate_with_langchain_docs`
   và `generate()` (tái dùng KG) **đã xác nhận** trên `ragas==0.4.3` cài thật (docstring
   `testset_generator.py`). Pilot (mục 4.1, 4 lần chạy 2026-09-28) đã kiểm chứng: ngôn ngữ
   (cần `adapt_prompts`), phong cách câu hỏi (`PERFECT_GRAMMAR`), `reference_contexts`
   nguyên văn, giới hạn Groq (hạ `max_token_limit`), chi phí token. **Chưa kiểm chứng:**
   multi-hop, `query_distribution` với trọng số lẻ, cắt/gộp Chương thật, tái dùng KG
   (`--reuse-knowledge-graph`) — kiểm khi implement/chạy thật.
5. **Round-robin 6 tài khoản Groq (mục 3.1) là pattern mới, chưa có tiền lệ trong repo** —
   khác hẳn cách dùng nhiều key hiện có (tách ngân sách theo bước cố định). Không nên coi
   đây là tiền lệ để áp dụng lại ở nơi khác trừ khi có nhu cầu tương tự (khối lượng LLM
   call lớn, job offline không nhạy latency); `generation/`/`retrieval/`/`conversation/`
   vẫn giữ nguyên pattern 1 key cố định/bước như đã chốt trước đó.
6. `itertools.cycle` không thread/async-safe tuyệt đối (mục 3.1) — chấp nhận rải tải không
   hoàn toàn đều nếu ragas gọi `generator_llm` đồng thời; không ảnh hưởng tính đúng đắn của
   testset sinh ra, chỉ ảnh hưởng độ cân bằng tải giữa 6 tài khoản.
7. Phase 2 (chạy pipeline thật, tính metric, chọn judge LLM, ngưỡng theo dõi, tần suất
   chạy) chưa được thiết kế — brainstorm riêng sau khi có `golden_testset.json` dùng được.
   Các điểm mở đã biết cho brainstorm đó: pipeline chạy eval đi từ đâu (gọi thẳng
   `retrieve()` + `generate()` hay qua cả `conversation/` với condense/guardrail; đề xuất
   bỏ cache khi eval để không đo nhầm kết quả cache); judge LLM dùng key/model nào (người
   dùng chốt dùng cả 6 key Groq, không chạy việc khác song song).
8. (Phase 2) Metric retrieval xác định (hit@k theo `chunk_id`) cần map đoạn `reference_contexts`
   của ragas sang tập chunk hệ thống (mục 2.1). Chưa xác nhận `reference_contexts` do
   ragas trả về có phải trích nguyên văn hay đã bị biến đổi: **pilot thăm dò xác nhận nguyên
   văn 12/12** đoạn (trên 1 văn bản nhỏ, single-hop); còn phải đo tỷ lệ khớp trên testset
   thật nhiều Chương và câu multi-hop trước khi tin vào hit@k. Không chặn Phase 1.
9. **Đáp án chuẩn (`reference`) không được đối chiếu với luật** (người dùng chốt chỉ đọc
   lướt, 2026-09-28). `reference` do LLM viết; nếu sai thì điểm Phase 2 lệch mà không ai
   biết. **Cách đọc kết quả Phase 2 phải tính đến điều này:** điểm là "mức khớp với đáp án
   do LLM sinh", không phải "đúng luật tuyệt đối". Chấp nhận có chủ đích cho Phase 1; nếu
   điểm Phase 2 bất thường, việc đầu tiên nên làm là kiểm mẫu vài câu `reference` với
   luật gốc.
10. **Câu multi-hop đòi ghép nhiều Điều — loại câu hệ thống đã biết là yếu** (chunk cắt theo
    Khoản, không xử lý chéo Điều; người dùng, 2026-09-28). **Đã chốt: tỷ lệ 80/10/10**
    (192/24/24) để điểm gộp không bị kéo thấp bởi điểm yếu đã biết; Phase 2 vẫn **báo điểm
    tách theo `synthesizer_name`** (single-hop là chỉ số chính). Hệ quả cần kiểm ở pilot:
    24+24 câu multi-hop chia cho nhiều Chương nhỏ (mỗi Chương chỉ 0-2 câu multi-hop) — ragas
    có sinh được đủ không, và `query_distribution` với trọng số lẻ có làm tròn đúng không.
11. **Đơn vị xử lý là Chương/Mục (đã chốt và ĐÃ đo, 2026-09-28)** để mỗi đơn vị vừa quota
    ngày: 50 đơn vị, lớn nhất 29.075 ký tự (~163K token) so với quota ~1,2M token/ngày —
    vừa thoải mái (mục 4.4). Ước lượng token là sai số ±30% (mới 2 điểm đo) nên vẫn cần ghi số
    thật từng đơn vị (mục 4.5) để chỉnh hệ số. Hai điểm còn mở: (a) đơn vị gộp nhiều Chương
    (vd. BLLĐ #16 = Chương XV+XVI+XVII) có thể cho câu hỏi kém đồng nhất; (b) nếu dừng giữa
    chừng, testset lệch về đơn vị nhỏ (mục 4.4).
12. **Đáp án chuẩn đôi khi bị dính chữ** ("cưtrú", "dịchvụ"; 1/4-5/8 đáp án tuỳ lần chạy):
    thiếu dấu cách thật do `gpt-oss-120b` sinh ra, không phải ký tự lạ (mục 4.1, pilot lần 4)
    nên không sửa được bằng NFKC. Chấp nhận có chủ đích; người dùng thấy khi đọc lướt. Nếu
    sau này thấy ảnh hưởng điểm Phase 2, cân nhắc đổi model sinh testset hoặc lọc bằng
    từ điển âm tiết tiếng Việt — không làm ở Phase 1.
13. **Trùng ý giữa các câu cùng một đoạn ragas** (pilot: 4 câu → 2 chủ đề) và **chủ đề nông
    kiểu "Chính phủ quy định chi tiết"** (NER trích thực thể chung chung). Chỉ xử lý bằng đọc
    lướt (mục 9.4); nếu tỷ lệ bỏ thực tế >40% thì tăng `GENERATE_SIZE` hoặc dùng sinh bù.
