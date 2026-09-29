# Evaluation — Synthetic Testset Generation (Phase 1): Reference Spec

> **Mục 1–10 là Phase 1 của roadmap RAGAS** (xem `CLAUDE.md` mục Tiến độ): sinh bộ câu
> hỏi + đáp án chuẩn ("golden testset") từ corpus pháp luật thật, không tự gán nhãn thủ
> công. **Phase 2 — chạy retrieval/generation thật trên testset này và tính metric RAGAS
> (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) — được
> thiết kế ở mục 11 (chốt 2026-09-29, CHƯA implement).** Các mục 1–10 không mô tả
> Phase 2, trừ chỗ có ghi chú trỏ sang mục 11.
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
- Luân phiên (round-robin) **6** tài khoản Groq độc lập (`GROQ_API_KEY_1`…`_6`) cho
  `generator_llm` để rải tải gọi LLM khi build KG + sinh 240 câu (mục 3.1) — **quyết định
  có chủ đích của người dùng**. (Lịch sử: chốt ban đầu 3 tài khoản/360 câu; 2026-09-28
  người dùng có thêm `GROQ_API_KEY_4` rồi `_5`, `_6` (tổng **6 tài khoản**, tên biến
  `GROQ_API_KEY_5`/`_6` đã được người dùng xác nhận), chốt đích cuối cùng **180 câu** để còn thời gian
  làm deploy + observability, và chốt cách sinh dư rồi trừ hao. 2026-09-29 người dùng thêm
  3 tài khoản nữa `GROQ_API_KEY_7`/`_8`/`_9` → **tổng 9 tài khoản**, mục 3.1.)
- Lưu testset thô ra `golden_testset_raw.json` (để người dùng đọc lướt/xoá) **ngay sau
  khi xong từng Chương** (checkpoint theo Chương — chạy dở, hết quota Groq ngày hôm đó
  thì hôm sau chạy tiếp, mục 4), rồi lệnh `finalize` chốt còn đúng `TARGET_SIZE = 180`
  câu ra `golden_testset.json` (mục 4.2). Lưu `KnowledgeGraph` của từng Chương ra file
  riêng để tái dùng (tránh build lại — tốn LLM call — khi cần sinh bù).
- CLI Typer mỏng: lệnh `generate` (có `--only`, `--testset-size`,
  `--reuse-knowledge-graph`, `--append`) và lệnh `finalize` (mục 7).

**Mục tiêu của bộ eval (chốt 2026-09-28, người dùng): kiểm tra TOÀN HỆ THỐNG — chạy
câu hỏi qua pipeline thật (HyDE → hybrid retrieval → RRF/MMR → rerank → generation →
Evidence Judge; **phạm vi Phase 2 đã chốt ở mục 11: bỏ guardrail/condense/cache/api**, vì đó
là tầng phục vụ end-user, không phải chất lượng hỏi-đáp) — với bộ câu hỏi/đáp án chuẩn
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

> **Cập nhật 2026-09-28: 6 tài khoản** (`GROQ_API_KEY_1`, `_2`…`_6`). Key 5, 6 là bucket
> `gpt-oss-120b` **hoàn toàn rảnh** (production chỉ dùng key 1-4). Quota ngày = 6 × 200K =
> **~1,2M token/ngày** cho `gpt-oss-120b`; các số đo "4 tài khoản/~800K/ngày" trong mục
> 4.1 là lịch sử pilot — lập kế hoạch theo 6 tài khoản (mục 4.4, 4.5).
>
> **Cập nhật 2026-09-29: 9 tài khoản** (thêm `GROQ_API_KEY_7`, `_8`, `_9` — 3 tài khoản Groq
> mới, độc lập, người dùng xác nhận). Quota ngày = 9 × 200K = **~1,8M token/ngày**
> (`TOKENS_PER_DAY = 1_800_000`), thông lượng tối đa ≈ 9 × 8.000 = 72.000 token/phút. Mọi con
> số "6 tài khoản / ~1,2M / 48K token/phút" ở các mục khác của spec là lịch sử tại thời điểm
> viết — từ nay **N = 9** (cả Phase 1 lẫn Phase 2, mục 11). Ba tài khoản mới chưa từng chạy
> pipeline này nên **kiểm dashboard usage sau lần chạy đầu** xem chúng có nhận tải thật không
> (nếu Groq báo cùng tổ chức với key cũ thì quota không tăng — mục 4.6).
>
> **Đổi tên biến 2026-09-29: `GROQ_API_KEY` → `GROQ_API_KEY_1`** (đánh số đủ `_1`…`_9` cho nhất
> quán). Đổi cứng, KHÔNG giữ alias tên cũ; đã áp dụng trong code và `.env.example`. Chính sách
> key/model chi tiết ở `conversation_spec.md` mục 12.1.

Với `testset_size = 240` (`GENERATE_SIZE`), số lượt gọi `generator_llm` (build `KnowledgeGraph` + sinh câu
hỏi) đủ lớn để 1 tài khoản Groq duy nhất dễ chạm rate limit theo phút/ngày (`CLAUDE.md`:
Groq giới hạn theo **tài khoản**, không theo key). Người dùng có 6 tài khoản Groq riêng
biệt (`GROQ_API_KEY_1`, `_2`…`_6` — 6 email khác nhau, không phải 6 key cùng 1 tài
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

Thiết kế (đơn giản nhất đủ dùng cho 1 script chạy 1 lần, nhưng ĐÚNG khi ragas gọi đồng thời):

```python
class GroqRoundRobinChatModel(BaseChatModel):
    """Proxy luân phiên round-robin qua N ChatOpenAI (Groq) độc lập tài khoản."""

    clients: list[ChatOpenAI]  # đúng 9 (từ 2026-09-29), mỗi client gắn 1 key cố định

    def _generate(self, messages, ...):
        order = self._plan_attempts()            # DƯỚI LOCK: chốt điểm bắt đầu 1 lần/lượt gọi
        errors = []
        for index in order:                      # n client KHÁC NHAU, bounded, không vô hạn
            self._record_attempt(index)          # dưới lock
            try:
                return self.clients[index]._generate(messages, ...)   # gọi mạng NGOÀI lock
            except RateLimitError as error:      # lỗi khác 429 (400/401/403/413...) bay thẳng ra
                errors.append(error)
                self._mark_result(index, daily_limited=_is_daily_limit(error))
        if all(_is_daily_limit(e) for e in errors):   # cả n client khác nhau, cùng lượt gọi
            raise self._trip_breaker(errors) from errors[-1]
        raise errors[-1]                         # hết vòng vẫn lỗi -> raise nguyên lỗi cuối
```

- **Thứ tự thử client.** `_plan_attempts` lấy `start = next(cycle)` đúng MỘT lần mỗi lượt gọi rồi
  duyệt `(start + offset) % n` cho `offset in range(n)` — mỗi lượt gọi luôn thử đúng `n` client
  KHÁC NHAU, không phụ thuộc các luồng khác đang gọi xen kẽ (lấy `next()` mỗi lần thử như bản
  đầu làm các luồng xen kẽ nhau nên một lượt gọi có thể thấy lặp/thiếu client). Client vừa báo
  hết quota ngày còn trong cooldown (bên dưới) được xếp CUỐI danh sách (sắp xếp ổn định) chứ
  không bị loại: vẫn nằm trong `n` client của lượt gọi.
- **Điều kiện bật circuit breaker** (mục 4.5, 8): chỉ khi CẢ `n` client khác nhau đều trả 429
  hết quota THEO NGÀY ("per day"/"(TPD)"/"(RPD)") trong CÙNG một lượt gọi (bằng chứng mới của
  cả `n`, cooldown không thay thế bằng chứng). Một 429 theo phút, hay bất kỳ client nào trong
  vòng trả kết quả, đều không bật breaker. Khi bật: ghi nhớ thông điệp và từ chối mọi lượt gọi
  sau đó ngay (`DailyQuotaExhaustedError`, không request nào); mỗi lần raise là một instance
  MỚI (không tái dùng một exception qua nhiều luồng), chỉ lần bật đầu giữ `__cause__`.
- **Cooldown 5 phút** (`_DAILY_COOLDOWN_SECONDS = 300`): client vừa báo TPD được đánh dấu để các
  lượt gọi sau ưu tiên tài khoản khác, tránh tốn `1 + max_retries` request 429 mỗi lượt cho
  tài khoản đã cạn (TPD Groq là cửa sổ trượt nên hồi dần; hết cooldown, hoặc khi mọi client
  còn lại đều lỗi, client đó được thử lại; thành công thì xoá cooldown).
- **Ràng buộc thread-safe (bắt buộc, không còn là "rủi ro chấp nhận được"):** ragas chạy nhiều
  worker (`max_workers=4`) và `_agenerate` mặc định chạy `_generate` trong executor, nên con
  trỏ vòng, `call_counts`, cooldown và cờ breaker CHỈ được đọc/ghi dưới một `threading.Lock`
  của instance; lock không giữ trong lúc gọi mạng. Cờ breaker khoá cả process, nên một lượt
  bật oan (do luồng xen kẽ) sẽ dừng cả job nhiều giờ.
- Chỉ cần implement `_generate` (sync) — `BaseChatModel._agenerate` mặc định fallback gọi
  `_generate` qua executor khi không override, nên round-robin vẫn đúng dù ragas gọi qua
  đường async.
- Không thêm rate-limiter mới: mỗi `ChatOpenAI` con giữ nguyên timeout/retry riêng từ
  `TestsetGeneratorSettings` (mục 6); round-robin chỉ chọn client, không kiểm soát tốc độ.
- Bounded fallback khi 429: thử tối đa `len(clients)` client cho MỘT lượt gọi rồi mới raise
  — cùng tinh thần retry giới hạn đã dùng ở `conversation/condenser.py` (retry đúng 1 lần).
- Không dùng `LoopBoundClient` (pattern ở `generation/`/`retrieval/` cho client sống suốt
  vòng đời server qua nhiều event loop) — không áp dụng ở đây vì đây là script chạy tuần
  tự trong đúng 1 process, không phải server long-running.

### 3.2 Điều phối key mượt và tiết kiệm token (chốt 2026-09-29, chưa implement)

**Mục tiêu (người dùng):** các key phối hợp mượt, tiêu token ít nhất có thể. Chỉ làm 3 thay
đổi nhỏ trong `groq_round_robin.py`/`ragas_runner.py`; KHÔNG đổi thuật toán chọn key.

**Sự thật đã kiểm (docs Groq, 2026-09-29):** free `gpt-oss-120b` mỗi tài khoản RPM 30, RPD 1K,
TPM 8K, TPD 200K. Header trả về `x-ratelimit-*` chỉ có RPD và TPM (KHÔNG có TPD còn lại), và
`retry-after` chỉ xuất hiện trên 429. Docs không nói 429 bị từ chối có tính vào RPD hay không.
Hệ quả: không dựng được sổ TPD chính xác từ header; và mỗi lượt gọi bị 429 rồi cascade qua cả
9 tài khoản có thể ăn tới 9 request RPD nếu chúng được tính — nên tránh bắn 429 vô ích.

**Không làm (và lý do):** chọn key theo "còn nhiều quota nhất" (không đọc được TPD, các lượt
gọi cùng cỡ nên xoay vòng đều là đủ); lưu sổ TPD qua nhiều ngày (không có nguồn sự thật);
bỏ qua đơn vị lỗi tất định, ghi chẩn đoán lỗi parse, lịch sử `last_failure` (không phục vụ
mục tiêu trên; để dịp khác nếu cần).

**A. Cooldown ngắn khi 429 theo phút (TPM/RPM).** Hiện chỉ 429 THEO NGÀY mới đặt cooldown
(300 giây, `_DAILY_COOLDOWN_SECONDS`); 429 theo phút để tài khoản đó vẫn nằm đầu vòng nên lượt
gọi kế lại đập vào đúng tài khoản đang bị giới hạn.
- Khi bắt `RateLimitError` KHÔNG phải giới hạn ngày: đọc header `retry-after` (giây) từ
  `error.response.headers`; thiếu/không parse được thì dùng `_MINUTE_COOLDOWN_DEFAULT = 15`
  giây; kẹp vào `[1, _MINUTE_COOLDOWN_MAX = 60]`. Đặt `_cooldown_until[index]` = `max(giá trị
  hiện có, now + số giây đó)` (không được rút ngắn cooldown ngày đang có).
- Tài khoản đang cooldown vẫn xếp CUỐI như cũ (`_plan_attempts`, sắp xếp ổn định).
- **Nếu CẢ n tài khoản đều đang cooldown** (tất cả do 429 theo phút, không có bằng chứng ngày):
  `_generate` chờ (`time.sleep`) tới lúc tài khoản sớm nhất hết cooldown (tối đa
  `_MINUTE_COOLDOWN_MAX` giây) TRƯỚC khi thử, thay vì bắn một loạt 429 qua cả n tài khoản. Không
  giữ lock khi ngủ. Không áp dụng khi có cooldown ngày (để breaker vẫn tính đúng bằng bằng chứng
  mới, mục 3.1).
- Điều kiện bật breaker KHÔNG đổi: cả n tài khoản báo giới hạn NGÀY trong cùng lượt gọi.
- Đồng hồ và `sleep` tiêm được (tham số/thuộc tính) để test không chờ thật.

**B. Đếm token thật theo tài khoản và theo đơn vị.** Hiện chỉ đếm số lượt gọi (`call_counts`,
gồm cả lượt 429); ước lượng "5,5 token/ký tự" sai số ±30% (mục 4.4) và không biết token suy
luận chiếm bao nhiêu.
- Router cộng dồn từ kết quả thành công: `prompt_tokens`, `completion_tokens`, và
  `reasoning_tokens` (`completion_tokens_details.reasoning_tokens` nếu Groq trả, thiếu thì 0)
  theo từng client, dưới cùng lock; thuộc tính `token_totals` trả bản sao. Lấy từ
  `ChatResult.llm_output["token_usage"]` của `ChatOpenAI`; nếu không có thì coi là 0 và log một
  lần cảnh báo (developer xác nhận đường lấy số trong venv `eval` bằng pilot).
- `UnitResult` thêm `tokens: int` và `reasoning_tokens: int` của riêng đơn vị (hiệu số trước/sau,
  như `llm_calls`); `UnitProgress` ghi thêm hai trường này (cộng dồn khi `--append`), mặc định
  `None` để đọc được `generation_progress.json` đã có (33 đơn vị cũ không có số đo).
- Log INFO cuối mỗi đơn vị: tổng token và token theo từng tài khoản (nhìn được có rải đều không;
  chỉ số, không có key).
- `generate --dry-run`: khi ≥ 3 đơn vị đã xong có `tokens`, ước lượng số token còn lại bằng hệ
  số token/ký tự ĐO ĐƯỢC (thay 5,5) và ghi rõ trong dòng tóm tắt "(hệ số đo được X token/ký tự)";
  ít hơn 3 thì giữ 5,5.

**C. Hạ `reasoning_effort` xuống `low` CHỈ khi dựng KG** (người dùng chốt 2026-09-29: "chỉ
hạ khi dựng KG, sinh câu giữ nguyên"). Token suy luận của `gpt-oss-120b` tính vào TPM/TPD; KG
(summary/themes/NER/headlines) là trích xuất đơn giản, còn câu hỏi/đáp án cuối vẫn dùng mức mặc
định để chất lượng testset không đổi so với 33 đơn vị đã sinh.
- Router có thuộc tính `reasoning_effort: str | None` (mặc định `None` = không gửi) và context
  manager `with router.reasoning_effort("low"):`; trong `_generate`, nếu đặt thì truyền
  `reasoning_effort=...` xuống `client._generate(...)` (cách truyền — kwarg hay `extra_body` —
  developer chốt theo phiên bản `openai`/`langchain-openai` của venv `eval`; pilot phải xác nhận
  Groq nhận và số `reasoning_tokens` giảm). Đặt/đọc dưới lock.
- `RagasUnitRunner._build_knowledge_graph` (kể cả `apply_transforms`) chạy trong context manager
  `low`; `_get_synthesizers` (`adapt_prompts`), `_query_distribution` và `_generate_cases` KHÔNG.
  Hai giai đoạn chạy tuần tự trong một đơn vị nên không xen kẽ; thoát context dù có lỗi.
- KG đã dựng trước đó (lưu ở `knowledge_graph/`) không bị dựng lại nên không hưởng lợi; chỉ các
  đơn vị chưa có KG.
- Không thêm biến `.env`; hằng số `_KG_REASONING_EFFORT: Final = "low"` trong `ragas_runner.py`.

**Test bắt buộc:** cooldown theo `retry-after` (có/thiếu/kẹp), không rút ngắn cooldown ngày;
chờ khi cả n đang cooldown phút và KHÔNG chờ khi có cooldown ngày; breaker không đổi; cộng dồn
token theo client và thread-safe; `reasoning_effort` chỉ được truyền trong context và được gỡ
sau (kể cả khi lỗi); `UnitProgress` đọc được file cũ không có `tokens`; dry-run dùng hệ số đo
khi đủ 3 đơn vị.

**Tiêu chí hoàn thành:** chạy tiếp 17 đơn vị còn lại; sau đó `generation_progress.json` có
`tokens`/`reasoning_tokens` cho các đơn vị mới, log cho thấy token rải xấp xỉ đều 9 tài khoản, và
so token/ký tự giữa đơn vị dựng KG mới (đã hạ effort) với hệ số 5,5 để biết mức tiết kiệm thật.

### 3.3 Giữ phần đã sinh khi lỗi giữa đơn vị (chốt 2026-09-29, chưa implement)

**Vấn đề:** hôm nay lỗi giữa đơn vị làm mất TOÀN BỘ câu đã sinh dở (chỉ KG được giữ). Nguyên
nhân: `TestsetGenerator.generate` của ragas chạy mọi câu rồi trả một lần, và mặc định
`raise_exceptions=True` nên một câu lỗi (429 ngày hay parse output) huỷ cả lô. Đặt
`raise_exceptions=False` **không dùng được**: ragas 0.4.3 trả `NaN` cho job lỗi rồi lặp/dựng
`TestsetSample` trên `NaN` và tự crash.

**Quy mô lãng phí (nói thẳng để người dùng cân nhắc):** lượt quota tối đa mất mỗi lần dừng là
phần SINH CÂU của một đơn vị (KG đã lưu), ước tính vài chục K token ≈ 1–2% quota ngày; trong 33
đơn vị đã chạy chỉ ghi nhận đúng 1 lần dừng (hết quota ngày). Lợi ích thật nằm ở chỗ khác: **một
câu lỗi tất định (parse) không còn làm hỏng cả đơn vị và chặn cả chương trình.** Người dùng chọn
làm (2026-09-29).

**Thiết kế (thay thế cách gọi `generate` trong `RagasUnitRunner._generate_cases`):**
- Không gọi `TestsetGenerator.generate`. Tự làm đúng các bước của nó với API công khai của
  synthesizer: (1) `generate_personas_from_kg(llm, kg, num_personas=3)` một lần cho đơn vị;
  (2) với từng loại có quota > 0: `await synthesizer.generate_scenarios(n, knowledge_graph,
  persona_list)`; (3) với từng scenario: `await synthesizer.generate_sample(scenario)` bọc
  `try/except`, đồng thời tối đa `MAX_WORKERS` (`asyncio.Semaphore`). Ragas ghim `==0.4.3`; có
  test khoá hành vi này để nâng phiên bản không âm thầm phá.
- Kết quả một sample lỗi:
  - `DailyQuotaExhaustedError` (breaker đã bật): ngừng đưa sample mới, giữ mọi sample đã xong,
    đơn vị kết thúc ở trạng thái **dở** (`partial`) rồi chương trình dừng như cũ (mã thoát 1).
  - Lỗi khác (parse, timeout đã hết retry...): BỎ sample đó, ghi log CHỈ tên loại lỗi (chính sách
    mục 8, không log nội dung), đếm `skipped_samples`; đơn vị vẫn `done` với ít câu hơn quota.
  - Nếu đơn vị không sinh được câu nào mà có sample lỗi: raise `UnitGenerationError` như cũ (lỗi
    có tính hệ thống; không ghi `done` rỗng). Lỗi ở bước (2) sinh scenario của một loại: cũng
    coi là lỗi đơn vị, nhưng các loại đã xong trước đó được giữ theo quy tắc `partial`.
- **Ghi ngay khi kết thúc bước sinh (kể cả `partial`):** nối câu vào raw TRƯỚC, ghi progress SAU
  (đúng thứ tự mục 4.5, `_recover_unfinished` không đổi).
- **`UnitProgress` thêm** `status: Literal["done", "partial"] = "done"` và `skipped_samples: int
  = 0` (mặc định để đọc được file 33 đơn vị cũ). `partial` cộng dồn `questions`/`llm_calls`/
  `seconds` như `--append`; KHÔNG ghi `completed_at` mới cho tới khi `done` (giữ giá trị cũ hoặc
  thời điểm dừng — developer chốt, phải là datetime hợp lệ).
- **Chạy tiếp một đơn vị `partial`:** nằm trong danh sách chờ như đơn vị chưa xong, dùng lại KG,
  quota còn lại theo loại = `quota − questions` đã ghi (kẹp ≥ 0); chỉ sinh phần còn thiếu rồi
  chuyển `done`. Loại đã bị `_has_clusters` bỏ (mục 4.5) được kiểm lại và bỏ lại như cũ, không tính
  là thiếu vĩnh viễn.
- `--dry-run` hiển thị trạng thái `dở (đã có N/M câu)` cho đơn vị `partial`; `--append` vẫn chạy
  lại đơn vị `done` theo cách cũ.
- Mã thoát và `last_failure`: không đổi (đơn vị `partial` do quota vẫn ghi `last_failure`).

**Thay thế các quy tắc cũ:** mục 4.5 ("đơn vị đang dở KHÔNG được ghi vào raw/`units`") và bảng
mục 8 dòng Groq lỗi/quota, mục 10.11 ("Không resume trong lòng một đơn vị") — chỉ còn đúng khi
lỗi xảy ra TRƯỚC khi có sample nào xong (dựng KG, sinh scenario); từ lúc có sample xong thì áp
dụng mục này.

**Test bắt buộc:** breaker bật giữa lúc sinh ⇒ sample đã xong vào raw, đơn vị `partial`, lần chạy
sau chỉ sinh phần còn lại (đếm lượt gọi synthesizer giả) rồi `done`; sample lỗi parse bị bỏ, đơn vị
`done`, `skipped_samples` đúng; mọi sample lỗi ⇒ `UnitGenerationError`, không ghi raw/progress;
progress cũ (không có `status`) đọc thành `done`; thứ tự raw trước progress khi chết giữa hai bước;
log lỗi sample không chứa nội dung.

## 4. Workflow (`testset_generator.py`)

```text
clients = [ChatOpenAI(key=GROQ_API_KEY_1), ..._2, ... , ..._9]            # 9 client, config mục 6
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
    → lưu KG ra data/eval/knowledge_graph/<tên>__<chương>.json NGAY SAU KHI dựng xong (nguyên
      tử, TRƯỚC bước sinh câu; lần sau đơn vị dở dùng lại KG đó nếu còn khớp văn bản — mục 4.5)
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
    # mã pilot; code thật dùng build_run_config() (mục 4.5)
    run_config=RunConfig(max_workers=4),
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
- > **Từ 2026-09-29 (mục 3.3, chưa implement):** khi đã có sample xong, phần đã xong được ghi
  > vào raw và đơn vị ở trạng thái `partial` thay vì bỏ hết; bullet dưới đây chỉ còn đúng cho
  > lỗi xảy ra trước khi có sample nào xong (dựng KG, sinh scenario).
- **Khi hết quota/lỗi giữa đơn vị** (429 sau khi mọi tài khoản đã thử, timeout, 413...):
  đơn vị đang dở KHÔNG được ghi vào raw/`units`; ghi `last_failure` (tên đơn vị, loại lỗi,
  thời điểm, không có nội dung câu hỏi/context); in tóm tắt tiến độ; **dừng luôn, không thử
  đơn vị kế** (quota các tài khoản chung nhau nên đơn vị kế cũng sẽ lỗi, chỉ đốt thêm token
  vào lượt dở); thoát với mã khác 0. Lần chạy sau thành công thì xoá `last_failure`.
- **KG và đơn vị dở (chốt 2026-09-28, sau review PR #60):** KG dựng xong là một đồ thị hoàn
  chỉnh (~30-160K token, tới ~10% TPD của 6 tài khoản), nên được lưu ngay sau `apply_transforms`,
  trước bước sinh câu, bằng ghi nguyên tử (file tạm rồi đổi tên; chết giữa chừng không để lại
  KG dở). Vì vậy KG có thể tồn tại cho đơn vị CHƯA xong; trạng thái "xong" vẫn chỉ do
  `generation_progress.json` quyết định. Đơn vị chưa có trong `units` (dở do lỗi, hoặc mới
  chạy) **tự dùng lại** KG đã lưu nếu file còn và `page_content` của node DOCUMENT vẫn khớp
  văn bản đơn vị hiện tại (lệch hoặc file hỏng thì dựng lại, có log); đơn vị đã xong (`--append`)
  chỉ dùng lại khi có `--reuse-knowledge-graph`. Muốn ép dựng lại KG của đơn vị dở thì xoá file
  `knowledge_graph/<văn bản>__<số>.json`.
- **`--append` bắt buộc đi kèm `--only`:** `--append` chạy lại cả đơn vị đã xong, thiếu `--only`
  sẽ chạy lại mọi đơn vị (đốt quota nhiều ngày) nên báo lỗi đầu vào (mã thoát 2, không gọi LLM).
- **Mã thoát của CLI:** 0 xong; 1 một đơn vị lỗi giữa chừng (chạy lại để làm tiếp); 2 đầu vào/cấu
  hình sai (`--only` sai, `--append` thiếu `--only`, thiếu thư mục/`.md` nguồn, progress hoặc raw
  hỏng/không phải UTF-8, thiếu `GROQ_API_KEY_*`, `finalize` thiếu câu). Cả `generate` và
  `finalize` dùng cùng quy ước.
- **Chính sách retry của ragas (chốt 2026-09-28, sau review PR #60):** ragas đọc retry từ
  `llm.run_config` (mặc định 10 lần với mọi `Exception`; `LangchainLLMWrapper` chỉ thu hẹp sang
  `RateLimitError` khi llm là `ChatOpenAI`, ở đây là `GroqRoundRobinChatModel` nên KHÔNG thu hẹp),
  còn `run_config` truyền vào `apply_transforms` chỉ được dùng cho `max_workers`. `RagasUnitRunner`
  tạo MỘT `RunConfig` (`max_retries=3`, `max_wait=30`, `max_workers=4`, `exception_types` = 429
  `RateLimitError`, `APIConnectionError` gồm timeout, `InternalServerError` 5xx — KHÔNG có
  400/401/403/413) và gán cho `LangchainLLMWrapper` ngay lúc khởi tạo để dựng KG,
  `adapt_prompts` và sinh câu cùng dùng. Ngoài ra, khi MỌI tài khoản (đúng `n` client khác nhau, trong CÙNG một lượt gọi — mục 3.1)
  đều báo hết quota THEO NGÀY (thông điệp 429 chứa "per day"/"(TPD)"/"(RPD)"), router ném
  `DailyQuotaExhaustedError` (không kế thừa `RateLimitError` nên ragas không retry) và từ chối
  ngay mọi lượt gọi sau đó trong process: ragas không huỷ các task còn lại khi một task lỗi,
  không chặn thì mỗi task còn lại vẫn đốt hàng chục request vô ích. 429 theo phút vẫn được retry
  (chờ là hết). Kết quả mong muốn khi hết quota: dừng sau một vòng vài chục request, ghi
  `last_failure`, mã thoát 1.
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

### 4.6 Tiến độ thực tế và việc còn lại (chốt 2026-09-29)

**Số liệu từ `generation_progress.json` / `golden_testset_raw.json`:**

| Hạng mục | Giá trị |
| --- | --- |
| Đơn vị đã xong | **33 / 50** (47,9% số ký tự; ước lượng ~2,03M / 4,16M token) |
| Câu trong raw | **97**: 90 single-hop, 7 multi-hop specific, **0 multi-hop abstract** |
| Lượt gọi LLM / thời gian | 672 lượt / ~4,3 giờ chạy thực (các đơn vị nhỏ hoàn thành hết trong ngày 2026-09-29) |
| Dừng lần cuối | 2026-09-29 16:14: `DailyQuotaExhaustedError` (6/6 tài khoản hết TPD 200K của `gpt-oss-120b`) ở đơn vị `Điều kiện lao động và quan hệ lao động.md#8` (chưa ghi vào raw/progress đúng thiết kế mục 4.5) |
| Còn lại | **17 đơn vị lớn nhất** (17.093 → 29.075 ký tự; 374.010 ký tự; ~2,12M token ước lượng) |
| Thời gian ước tính | ~2,12M / 1,8M token/ngày (9 tài khoản) ≈ **1,2 ngày** → thực tế 2 lần chạy (TPD là cửa sổ trượt; trước đó 6 tài khoản ≈ 1,8 ngày) |

**Chạy tiếp** (không đổi so với mục 4.5): `tools/generate_testset.py generate` — tự bắt đầu ở đơn
vị `Điều kiện lao động và quan hệ lao động.md#8` (thứ 34), dùng lại KG đã lưu nếu có. Trước khi
chạy bắt buộc xong việc sửa code cho 9 key bên dưới, vì `TestsetGeneratorSettings` hiện chỉ đọc
6 key và code sẽ không dùng `GROQ_API_KEY_7`–`_9`.

**Việc code cho 9 key (developer, Phase 1 — thay đổi nhỏ, không đổi thiết kế):**
- `config.py`: thêm `api_key_7`/`_8`/`_9` (`Field(min_length=1, validation_alias=...)`, bắt buộc như 6 key trước).
- `ragas_runner.py`: `_build_groq_clients` thêm 3 key vào tuple; docstring/comment "6" → "9".
- `testset_generator.py`: `TOKENS_PER_DAY = 1_800_000`; thông báo lỗi thiếu key nêu `GROQ_API_KEY_2 ... GROQ_API_KEY_9`; comment "6 tài khoản".
- `.env.example`: thêm `GROQ_API_KEY_7`, `_8`, `_9`.
- Test: cập nhật các test dựng `TestsetGeneratorSettings`/đếm 6 client sang 9.
- `groq_round_robin.py` KHÔNG cần đổi (router dùng `len(clients)`, không hard-code 6).

**Việc code kế tiếp (đã chốt 2026-09-29, chưa implement, PR riêng sau PR #63):** mục 3.2 (cooldown
429 theo phút, đếm token thật theo key/đơn vị, `reasoning_effort=low` chỉ khi dựng KG) và mục 3.3
(giữ phần đã sinh khi lỗi giữa đơn vị, đơn vị `partial`). Chạy tiếp 17 đơn vị còn lại có thể làm
ngay bằng code hiện tại; làm xong 3.2/3.3 trước thì tiết kiệm hơn và ít rủi ro mất công hơn.

**Rủi ro 3 tài khoản mới (chưa kiểm chứng):** Groq tính quota theo tổ chức; nếu `_7`–`_9` cùng
tổ chức với key cũ thì bucket dùng chung, quota không tăng và round-robin chỉ thêm request thừa.
Cách kiểm nhanh: sau lần chạy đầu, dashboard usage của `gpt-oss-120b` ở 3 tài khoản mới phải có
token tiêu thụ (mục 3.1). Key `_7`–`_9` là bucket `gpt-oss-120b` **rảnh hoàn toàn** như key 5, 6
(production chỉ dùng 1-4, không dùng thêm), nên không cần né traffic thật.

**Vấn đề mở — `multi-hop abstract` = 0 (phát hiện 2026-09-29, người dùng chốt: chạy tiếp cho xong đã, xét sau):**
33 đơn vị (~48% ký tự) lẽ ra đã cho ~11 câu abstract (quota 24/240, mục 4 phân bổ theo ký tự) và
~11 câu specific, thực tế 0 và 7. Single-hop đúng kế hoạch (90 so với ~92). Nghi ngờ: KG theo
từng Chương nhỏ không có cụm đoạn thoả `MultiHopAbstractQuerySynthesizer` (theme chung giữa các
node) nên `_has_clusters` trả `False` và synthesizer bị bỏ qua — nhưng **chưa xác minh** là do
`_has_clusters` hay do lỗi khác (ví dụ synthesizer sinh ra rồi bị filter loại hết). Hệ quả nếu
giữ nguyên: raw kết thúc ~207 dòng (~192 single + ~15 specific + 0 abstract), đủ trên `TARGET_SIZE
= 180` nhưng **không có câu multi-hop abstract**; phân bố 80/10/10 (mục 10.10) thành ~92/0/8.
Quyết định sau khi xong 50 đơn vị, một trong ba: (a) sinh bù abstract bằng `generate --only
"<tên>#<số>" --append` trên các đơn vị lớn có KG dày; (b) chấp nhận testset không có abstract và ghi
vào chú thích báo cáo Phase 2 (mục 11.6: multi-hop chỉ có specific); (c) chỉnh tỷ lệ 90/0/10 và sửa
mục 10.10. Nên xác minh nguyên nhân (đọc `_has_clusters` trên KG đã lưu của vài đơn vị lớn) trước khi chọn (a).

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
    """Cấu hình 9 tài khoản Groq round-robin cho generator_llm (Phase 1 RAGAS, mục 3.1).

    Cả 9 key BẮT BUỘC (không optional/fallback như GenerationSettings/JudgeSettings) —
    round-robin chỉ có ý nghĩa khi đủ 9 tài khoản độc lập; thiếu key nào, pydantic báo lỗi
    rõ ràng ngay lúc khởi tạo thay vì âm thầm chạy round-robin với ít tài khoản hơn.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: str = Field(
        validation_alias="GROQ_API_KEY_1"
    )  # đổi tên từ GROQ_API_KEY, 2026-09-29
    api_key_2: str = Field(validation_alias="GROQ_API_KEY_2")
    api_key_3: str = Field(validation_alias="GROQ_API_KEY_3")
    api_key_4: str = Field(validation_alias="GROQ_API_KEY_4")
    api_key_5: str = Field(validation_alias="GROQ_API_KEY_5")
    api_key_6: str = Field(validation_alias="GROQ_API_KEY_6")
    api_key_7: str = Field(validation_alias="GROQ_API_KEY_7")
    api_key_8: str = Field(validation_alias="GROQ_API_KEY_8")
    api_key_9: str = Field(validation_alias="GROQ_API_KEY_9")
    model_name: str = "openai/gpt-oss-120b"
    max_retries: int = 2
    timeout_seconds: int = 60
```

- Dùng cả 9 khoá `GROQ_API_KEY_1`…`_9` (9 tài khoản Groq độc lập, xác nhận bởi
  người dùng; `_7`–`_9` thêm 2026-09-29) qua round-robin (mục 3.1) — khác mục đích tách ngân sách theo bước của
  `GenerationSettings`/`JudgeSettings` (`GenerationSettings` cũng đọc `GROQ_API_KEY_3`/
  `_4`, nhưng để xoay vòng cho generation production), nên **không dùng chung class** với
  chúng dù tên biến trùng.
- `model_name` dùng lại `gpt-oss-120b` (không phải `20b` như condense) vì sinh câu hỏi
  multi-hop cần khả năng tổng hợp/suy luận qua nhiều đoạn văn bản.
- Không thêm setting riêng cho embeddings: `embeddings_adapter.py` tái dùng thẳng
  `EmbeddingSettings` đã có (`config.py`).
- Module không đọc `.env` trực tiếp. `.env.example` mới có `GROQ_API_KEY_1`…`_4` — **thêm
  `GROQ_API_KEY_5`, `GROQ_API_KEY_6`** và xoá 4 dòng `OPENROUTER_API_KEY*` (chỉ phục vụ pilot
  đã xong; 2026-09-29 thêm `GROQ_API_KEY_7`, `_8`, `_9` vào `.env.example`); bổ sung ghi chú: 9 key này cũng được dùng round-robin bởi `tools/generate_testset.py`
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
| `tools/generate_testset.py` | Typer app mỏng với **2 lệnh**, chỉ gọi `testset_generator.py`, không chứa business logic. `generate`: `--markdown-dir`, `--output-dir`, `--only <tên>[#<số thứ tự đơn vị>]` (lặp lại được; chọn cả văn bản hoặc từng đơn vị), `--dry-run` (in kế hoạch theo thứ tự chạy nhỏ → lớn: đơn vị, ước lượng token, trạng thái xong/chưa/lỗi lần cuối, tóm tắt tiến độ; KHÔNG gọi LLM — mục 4.4, 4.5), `--reuse-knowledge-graph` (chỉ cần cho đơn vị đã xong; đơn vị dở tự dùng lại KG, mục 4.5), `--append` (bắt buộc kèm `--only`), `--testset-size` (tổng, mặc định `GENERATE_SIZE=240`); mã thoát 0/1/2 (mục 4.5). `finalize`: `--output-dir` → đọc raw, ghi `golden_testset.json` (đúng 180). Hiện `generate_golden_testset` chỉ xử lý cả thư mục một lần và ghi `golden_testset.json` — cần viết lại theo vòng lặp từng văn bản rồi từng Chương. |

Ghi chú vị trí: đặt ngay ở package `evaluation/` (không phải chỉ 1 script rời trong
`tools/`) vì Phase 2 sẽ thêm module chạy eval (`run_eval.py`/tương đương) vào **cùng**
package này — tránh phải dời code khi mở rộng.

## 8. Xử lý lỗi

| Sự cố | Xử lý |
| --- | --- |
| Groq lỗi/quota (429 hết token/ngày)/timeout giữa lúc build `KnowledgeGraph` hoặc sinh câu hỏi của MỘT Chương | Dừng chương trình (không thử đơn vị kế), log lỗi rõ ràng (không log nội dung câu hỏi/context — theo chính sách log chung của project) kèm **khoá đơn vị đang dở** và một dòng `traceback: <Loại lỗi> @ file:dòng:hàm -> ...` (`_describe_traceback`, chốt 2026-09-29: chỉ tên loại lỗi + basename file + số dòng + tên hàm của từng frame để debug được; KHÔNG dùng `exc_info=True`/`logger.exception` vì chúng in cả thông điệp exception, có thể chứa nội dung câu hỏi/context), ghi `last_failure` vào `generation_progress.json` (file được commit nên thông điệp HTTP bỏ tiền tố SDK, che mã tổ chức `org_...` và mọi chuỗi giống key `gsk_...` RỒI mới cắt 200 ký tự, để không mất phần "per day (TPD): Limit/Used"). Đơn vị đang dở KHÔNG được ghi vào raw/progress (raw không bao giờ ghi dở) — **trừ phần sample đã xong theo mục 3.3 (ghi `partial`, chưa implement)**; KG chỉ được giữ nếu đã dựng xong hoàn chỉnh (mục 4.5), còn lỗi giữa lúc dựng KG thì không có file KG; các đơn vị đã xong trước đó giữ nguyên. Retry ragas giới hạn ở lỗi tạm thời và hết quota ngày dừng ngay sau một vòng (mục 4.5). Chạy lại `generate` ngày hôm sau sẽ làm tiếp từ đơn vị đó (mục 4.5). Không resume trong lòng một đơn vị (mục 10.11) — **được nới bởi mục 3.3: đơn vị `partial` chạy tiếp phần còn thiếu**. |
| `--only` nêu tên văn bản không tồn tại trong `data/markdown/` | Raise lỗi rõ liệt kê tên hợp lệ, trước khi gọi Groq. |
| `embeddings_adapter.py` trả response sai định dạng (khác kỳ vọng của HF API) | Raise lỗi rõ, không âm thầm trả vector rỗng — cùng nguyên tắc validate ở biên như `embedding/hf_client.py`. |
| File `data/markdown/*.md` trống hoặc thiếu | Raise lỗi rõ trước khi gọi `TestsetGenerator` (fail fast, không lãng phí LLM call). |
| Một tài khoản Groq bị 429 giữa vòng round-robin (mục 3.1) | Thử ngay tài khoản kế tiếp, tối đa `len(clients)` client KHÁC NHAU cho MỘT lượt gọi (điểm bắt đầu chốt một lần dưới lock); hết vòng vẫn lỗi thì raise nguyên lỗi cuối — không giữ vòng lặp vô hạn, không tự ý bỏ qua câu hỏi đó. Chỉ khi cả `n` client khác nhau đều báo hết quota ngày trong cùng lượt gọi mới bật breaker (`DailyQuotaExhaustedError`). Tài khoản vừa báo TPD được xếp cuối trong 5 phút. |
| Thiếu hoặc để trống (`GROQ_API_KEY_5=`) bất kỳ key nào trong `GROQ_API_KEY_1`…`_9` trong `.env` | `TestsetGeneratorSettings` (`Field(min_length=1)`) raise lỗi validate ngay lúc khởi tạo (fail fast), thông báo chỉ nêu TÊN biến — không âm thầm chạy round-robin với ít hơn 9 tài khoản. |
| `generate` gặp đơn vị đã có trong `generation_progress.json` mà không có `--append` | Bỏ qua đơn vị đó, log rõ "đã xong" (không gọi Groq, không ghi đè) — vừa là resume vừa bảo vệ công review tay (mục 4.5). Không bao giờ ghi đè/sắp xếp lại các dòng có sẵn trong file raw (người dùng đã sửa tay). |
| Đơn vị đã xong nhưng `chars` trong progress khác kết quả chia hiện tại | Dừng với lỗi nêu tên đơn vị (nguồn hoặc quy tắc chia đã đổi); không tự bỏ qua/ghi đè (mục 4.5). |
| Đơn vị có dòng trong raw nhưng chưa có trong progress (chết giữa hai bước ghi) | Coi là đã xong, ghi bổ sung vào progress và log; không sinh lại. |
| `--append` không kèm `--only` | Lỗi đầu vào (mã thoát 2) trước khi gọi LLM; không chạy lại cả corpus (mục 4.5). |
| Thiếu/sai `GROQ_API_KEY_1`…`_9` khi chạy `generate` thật | `EvalInputError` (mã thoát 2) chỉ nêu TÊN biến còn thiếu, KHÔNG in giá trị (lỗi gốc của pydantic in đầu/đuôi key). |
| `data/markdown` thiếu/không có `.md`, hoặc file `.md`/progress/raw không phải UTF-8 | `EvalInputError` (mã thoát 2), không traceback. |
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
     `--reuse-knowledge-graph` (với `--append`) không build lại đồ thị. Thử dừng giữa chừng
     (Ctrl+C hoặc gỡ một key để ép lỗi) rồi chạy lại: đơn vị dở không nằm trong progress,
     `last_failure` được ghi, lần sau tiếp tục đúng chỗ và, nếu KG đã dựng xong trước khi lỗi,
     KHÔNG dựng lại KG (log "Tái dùng KG đã lưu"). Kiểm thêm: khi ép hết quota ngày, chương
     trình dừng sau một vòng (không treo hàng chục phút) và thoát mã 1.
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
6. `GroqRoundRobinChatModel` được gọi đồng thời (ragas `max_workers=4`, `_agenerate` chạy
   `_generate` trong executor) nên trạng thái dùng chung phải thread-safe (mục 3.1): chốt điểm
   bắt đầu một lần mỗi lượt gọi dưới `threading.Lock` để breaker không bật oan. Bản đầu
   ghi đây là "rủi ro chấp nhận được" — sai, vì cờ breaker khoá cả process.
7. Phase 2 đã được thiết kế ở **mục 11** (2026-09-29): chạy `retrieve` + `generate` thẳng
   (không qua `conversation/`), bỏ cache; judge RAGAS `gpt-oss-120b` trên cả 9 key Groq.
   Ngưỡng theo dõi và tần suất chạy vẫn chưa chốt (mục 11.9).
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

## 11. Phase 2 — Chạy pipeline thật và chấm điểm (chốt 2026-09-29, chưa implement)

### 11.1 Mục tiêu, phạm vi

Chạy từng câu của `golden_testset.json` qua retrieval + generation thật của hệ thống, rồi
chấm bằng RAGAS. Trả lời hai câu hỏi: (1) **bật hay tắt MMR** thì retrieval tốt hơn
(retrieval_spec mục 6 coi MMR là cờ evaluation, không phải tối ưu tin sẵn); (2) chất lượng
câu trả lời cuối (đã qua Evidence Judge) đang ở mức nào, theo loại câu và theo văn bản luật.

**Trong phạm vi:** HyDE → embed → retrieve (2 cấu hình `mmr_on`/`mmr_off`) → chấm retrieval →
generation của cấu hình thắng (dùng `GenerationPipeline.generate`, gồm draft + hard gate +
Evidence Judge + repair, `generation_spec.md` mục 3) → chấm câu trả lời → báo cáo.

**Không làm:**
- Guardrail, condense, cache, `api/`, `conversation/`: là tầng phục vụ end-user, testset là câu
  hỏi đơn lượt độc lập, và cache làm sai số đo. Hệ quả: tỷ lệ guardrail từ chối nhầm câu hợp lệ
  KHÔNG được đo ở Phase 2.
- **Groq Batch API:** tài khoản free không dùng được (console `dashboard/batch` báo "Upgrade
  your plan", 2026-09-29; người dùng không có Developer plan). Cách "gom" duy nhất dùng là gom
  nhiều text vào một request embed HF (25 text/request, như `embedding/hf_client.py`).
- Metric xác định `hit@k` theo `chunk_id` (mục 10.8), lát đánh giá nhiễu (mục 4.1), lấy mẫu
  `chat_turns`, CI/cron.

**Tiêu chí hoàn thành:** chạy đủ các stage trên `golden_testset.json` (180 câu), có
`data/eval/phase2/report.json` (mục 11.6) và người vận hành đọc được bảng so sánh MMR bật/tắt
lẫn điểm câu trả lời theo `synthesizer_name`/`source_document`.

### 11.2 Stage, file trung gian, resume

Mỗi stage đọc file của stage trước, ghi một file JSONL ở `data/eval/phase2/`, và chạy lại
được (resume) — đổi prompt generation chỉ chạy lại S5–S6, không đụng retrieval. Khoá bản ghi
= `case_id` = 12 ký tự hex đầu của `sha256(user_input)` (testset không có id; hash ổn định dù
người dùng xoá/đổi thứ tự dòng khi đọc lướt).

| Stage | Việc | Tài nguyên | File ra |
| --- | --- | --- | --- |
| S1 `hyde` | `HydeGenerator.generate(user_input)` cho mọi câu | LLM `gpt-oss-20b`, cả 9 key (mục 11.3), throttle theo bucket `(model, key)` sẵn có | `hyde.jsonl`: `{case_id, hypothetical_document \| null, error \| null}` |
| S2 `embed` | Embed `[hypo, query]` bằng `QueryEmbedder` (pyvi + cùng model như index), gom 25 text/request | HF Inference API | `embeddings.jsonl`: `{case_id, hypothetical_embedding \| null, query_embedding}` (không commit git — ~5 MB, tái sinh được) |
| S3 `retrieve` | `RetrievalPipeline.retrieve(query, use_mmr=…, precomputed=…)`, **tuần tự** từng cấu hình (`mmr_on` xong hết rồi `mmr_off`) | Pinecone, BM25, rerank GPU local | `retrieved_mmr_on.jsonl`, `retrieved_mmr_off.jsonl`: `{case_id, chunks: [RetrievedChunk], error \| null}` |
| S4 `score-retrieval` | RAGAS `context_precision` + `context_recall` cho từng cấu hình | LLM `gpt-oss-120b`, 9 key round-robin | `retrieval_scores.jsonl`: `{case_id, config, context_precision, context_recall}` |
| S5 `generate --config <mmr_on\|mmr_off>` | `GenerationPipeline.generate(query, chunks)` cho cấu hình do người dùng chọn sau S4 | 120b draft/repair + 20b Judge, **cả 9 key** (mỗi câu một pipeline theo key, mục 11.3) | `answers.jsonl` (mục 11.4) |
| S6 `score-answers` | RAGAS `faithfulness` + `answer_relevancy` trên các câu `answered` | 120b 9 key + embedding HF | `answer_scores.jsonl`: `{case_id, faithfulness, answer_relevancy}` |
| `report` | Tổng hợp | không LLM | `report.json` + bảng in ra terminal |

**Quy tắc chung:**
- **Resume:** stage bỏ qua `case_id` đã có trong file ra; thiếu bản ghi ở stage trước thì báo số
  lượng còn thiếu và chỉ xử lý phần đã có. Bản ghi lỗi tạm thời (`error` khác null) được chạy lại
  bằng `--retry-failed`, mặc định KHÔNG chạy lại (tránh đốt token vào lỗi tất định).
- Ghi nối từng dòng ngay sau mỗi bản ghi (S4, S6: sau mỗi lô `SCORING_BATCH_SIZE = 10`, để hết
  quota chỉ mất tối đa một lô); dòng cuối hỏng do chết giữa chừng thì bỏ và log, không raise.
- Mã thoát như Phase 1: 0 xong; 1 dừng giữa chừng (hết quota/lỗi, chạy lại để làm tiếp); 2 đầu
  vào/cấu hình sai (thiếu key, file hỏng, `--config` sai).
- `--testset` (mặc định `data/eval/golden_testset.json`) cho phép trỏ `golden_testset_raw.json`
  để chạy thử khi chưa `finalize` (raw đang lệch về đơn vị nhỏ, mục 4.4 — chỉ để kiểm pipeline,
  không đọc điểm như kết quả chính thức). `--limit N` lấy N câu đầu để pilot đo token thật.
- Người dùng cam kết không chạy traffic thật/việc khác trên các bucket Groq trong lúc chạy
  (như mục 3.1); S1 và S5 dùng cả 9 key nên càng cần: bucket `gpt-oss-120b` của key 3, 4 vốn
  là bucket generation của production.

### 11.3 Chi tiết từng stage

**Rải 9 key (S1 và S5).** `HydeGenerator`, `AnswerGenerator` và `EvidenceJudge` mỗi cái chỉ
đọc key cố định từ settings (HyDE 1 key, generation 2 key, Judge 1 key) nên không tự xoay
9 key. Eval dựng 9 bộ (`HydeSettings`/`GenerationSettings`/`JudgeSettings` truyền key tường
minh — developer xác nhận cách truyền theo `validation_alias`, và đặt `GROQ_API_KEY_4` của
`GenerationSettings` là `None` để mỗi bộ đúng một key, không đọc lẫn từ `.env`), rồi gán các
bản ghi chờ xử lý cho 9 bộ theo vòng tròn **tính trên danh sách còn lại lúc chạy** (nên
`--retry-failed` tự rơi sang key khác). Đồng thời mỗi stage = 9 (một lượt gọi mỗi tài khoản);
tốc độ do throttle theo bucket `(model, key)` sẵn có điều tiết. Một tài khoản hết TPD chỉ làm
các câu được gán cho nó ra `error` (429), các tài khoản khác vẫn chạy; chạy lại bằng
`--retry-failed`. Helper dựng 9 bộ nằm ở `key_pool.py` (mục 11.7), không sửa code production.

**S1 HyDE.** Tái dùng `HydeGenerator` nguyên trạng để prompt/model giống production. HyDE là
best-effort (retrieval_spec mục 8): trả `None` thì lưu `hypothetical_document: null` và stage
sau bỏ nhánh A, đúng như production. Phân biệt `error` (429/timeout, chạy lại được) với
`hypothetical_document: null` không lỗi (model trả rỗng). Đồng thời và rải key như trên.

**S2 Embed.** Gom `[hypothetical_document, user_input]` của nhiều câu vào request 25 text,
không đổi thứ tự (`QueryEmbedder.embed` đã word-segment bằng `ViTokenizer`). Vector của một
text không phụ thuộc text cùng batch nên tách/gom không đổi kết quả. Không embed song song
với S1: vector `query` độc lập HyDE, nhưng tách riêng thêm phức tạp mà lợi ích chỉ vài request.

**S3 Retrieve.** Cần đường vào `precomputed` của retrieval (`retrieval_spec.md` mục 2) — thay
đổi nhỏ duy nhất ở code production. Chỉ tạo **một** `RetrievalPipeline` (model rerank nạp một
lần), chạy **tuần tự**: hết 180 câu với `use_mmr=True` rồi mới sang `use_mmr=False` (chốt
2026-09-29: máy người dùng chạy rerank sát giới hạn GPU 2GB, không song song hoá gì cả),
`RETRIEVE_CONCURRENCY = 1`. Nếu vẫn CUDA OOM: giảm `batch_size` của `LocalReranker` (chỉ đổi
lượng bộ nhớ mỗi forward pass, không đổi điểm về nguyên tắc) rồi chạy lại các câu lỗi bằng
`--retry-failed`; không đổi model hay `max_length` vì sẽ đo sai hệ thống thật.
**Fallback rerank là lỗi, không phải kết quả:** nếu bất kỳ chunk nào có `rerank_score is
None` (CUDA OOM → fallback xen kẽ nhánh, retrieval_spec mục 8) thì ghi `error` và không chấm —
nếu để lọt, S4 sẽ so hai cấu hình bằng thứ tự không qua rerank mà không ai biết. Kết quả rỗng
cũng ghi `error = "no_context"`. `RetrievalError` (HF/Pinecone) ghi `error`.

**S4 Chấm retrieval.** `LLMContextPrecisionWithReference` và `LLMContextRecall` (ragas; chỉ cần
`user_input`, `retrieved_contexts`, `reference`, KHÔNG cần `response` — nên chọn được cấu hình
MMR trước khi tốn token generation). Judge = `LangchainLLMWrapper(GroqRoundRobinChatModel)`
của Phase 1 (`gpt-oss-120b`, 9 key, cùng `RunConfig` mục 4.5: retry giới hạn, hết quota ngày
dừng ngay; dùng lại `TestsetGeneratorSettings`, không đổi tên). Chuỗi mỗi chunk trong
`retrieved_contexts` = đúng phần chunk đó trong `build_context` của generation (breadcrumb +
content/raw_table), để judge thấy cái LLM sinh câu thấy; developer tách hàm dùng chung nếu
`build_context` chưa cho phép, không tự chế format khác. Chi phí: `context_precision` 1 lượt/
chunk (5), `context_recall` 1 lượt → 6 lượt/record/cấu hình (`ragas==0.4.3`, đếm từ source).

**Chọn cấu hình sau S4 (người dùng quyết, không tự động).** `report` in bảng so sánh và số câu
`mmr_on` hơn / thua / hoà `mmr_off` theo từng câu (180 câu: chênh trung bình nhỏ dễ là nhiễu
judge, số câu thắng/thua nói rõ hơn). Gợi ý quy tắc, người dùng có thể đổi: ưu tiên
`context_recall` (thiếu chunk là lỗi nặng hơn với pháp luật), `context_precision` để phá hoà.
Lý do (chốt cùng người dùng 2026-09-29): MMR chỉ đổi **tập candidate vào union trước rerank**
(top 10 mỗi nhánh), còn thứ tự cuối và cắt top 5 do reranker quyết — nên `context_precision`
chủ yếu do rerank + `FINAL_TOP_K` cố định quyết định, ít khác giữa hai cấu hình; `context_recall`
mới là chỗ MMR có thể giúp (candidate đa dạng hơn) hoặc hại (phạt oan các Khoản liền kề cùng
cần, retrieval_spec mục 6). Nếu `context_precision` lệch nhiều giữa hai cấu hình thì ghi nhận
như tín hiệu bất thường cần xem lại, không dùng để quyết;
nếu không có khác biệt rõ thì chọn tắt MMR (đơn giản hơn, bớt một lượt Pinecone lấy vector;
retrieval_spec mục 6 coi RRF là baseline). Kết luận được ghi lại vào `retrieval_spec.md` mục 6.

### 11.4 S5 — Generation (chốt phương án B, 2026-09-29)

Gọi `GenerationPipeline.generate(query, chunks)` (`generation/pipeline.py`): hàm này sẵn không
gọi guardrail và không retrieve, nhận context đã chốt — đúng "bỏ tầng end-user" mà không viết
lại logic. Kết quả là câu người dùng thật sẽ thấy (đã qua hard gate + Evidence Judge + tối đa
1 repair), không phải draft thô. **Dùng cả 9 key (6 chốt 2026-09-29, nâng lên 9 cùng ngày khi có thêm `_7`–`_9`, người dùng).**
Cấu hình key production chỉ có 2 tài khoản cho generation (3⇄4) và 1 cho Judge (key 2), nên
eval dựng **9 `GenerationPipeline` độc lập, mỗi cái một key** (mục 11.3, "Rải 9 key"): pipeline
thứ i có `AnswerGenerator` (120b) và `EvidenceJudge` (20b) cùng dùng key i. Judge và generator
cùng tài khoản không tranh nhau vì Groq tính rate limit theo `(tài khoản, model)` và hai bước
khác model. Không sửa code production; chỉ khác cách dựng settings. Hết quota thì dừng, mai
chạy tiếp (resume theo `case_id`).

Gom event thành một bản ghi `answers.jsonl`:

```python
class AnswerRecord(BaseModel):
    case_id: str
    config: Literal["mmr_on", "mmr_off"]
    outcome: Literal["answered", "insufficient_evidence", "unable_to_verify", "error"]
    response: str | None  # ghép các TokenEvent; None nếu không answered
    citations: list[int]  # chỉ số [n] hợp lệ trong câu trả lời
    repair_used: bool  # có StatusEvent("repairing")
    warning_codes: list[str]
    error_code: str | None  # ErrorEvent.code khi outcome == "error"
    usage: Usage | None
```

- `insufficient_evidence` và `unable_to_verify` là **kết quả hợp lệ của hệ thống**, không chạy
  lại; chỉ `error` (429, lỗi LLM) mới chạy lại được bằng `--retry-failed`.
- **Hệ quả cho cách đọc điểm:** câu bị từ chối không có `response` nên RAGAS không chấm được;
  báo riêng **tỷ lệ từ chối** (theo loại và theo `synthesizer_name`). `faithfulness` đo trên
  câu ĐÃ qua Evidence Judge nên cao hơn faithfulness của draft — điểm này phản ánh "hệ thống
  cả Judge", không tách được chất lượng generator riêng.

### 11.5 S6 — Chấm câu trả lời

`Faithfulness` (2 lượt/câu: tách statement + NLI với context) và `ResponseRelevancy`
(`answer_relevancy`, `strictness = 3` → 3 lượt; wrapper round-robin không có thuộc tính `n`
nên ragas gọi 3 lượt riêng, tránh lỗi Groq không hỗ trợ `n>1`) — 5 lượt/câu, chỉ trên bản ghi
`answered`. `context_precision`/`context_recall` của cấu hình thắng đã có từ S4, không tính lại.
`answer_relevancy` cần embedding: dùng `RagasEmbeddingsAdapter` với **`segment=True`** (thêm
tham số, mặc định `False` để KHÔNG đổi hành vi Phase 1) để áp `ViTokenizer` — khác quyết định
mục 3 (không segment cho KG) vì ở đây là cosine giữa câu hỏi gốc và các câu hỏi ragas sinh lại,
cần cùng không gian với model đã huấn luyện trên văn bản đã segment (`embedding_spec.md` mục 4);
kiểm ở pilot rằng điểm không lệch bất thường.

### 11.6 Báo cáo (`report.json`)

- **So sánh retrieval:** trung bình `context_precision`/`context_recall` của `mmr_on` và
  `mmr_off` (tổng, theo `synthesizer_name`, theo `source_document`) + số câu thắng/thua/hoà.
  Chỉ so trên các `case_id` có kết quả hợp lệ ở CẢ HAI cấu hình; báo số câu bị loại (lỗi/OOM).
- **Điểm câu trả lời** (cấu hình đã chọn): trung bình bốn metric theo cùng các lát, kèm `n` mỗi
  lát và số giá trị `null` (ragas trả NaN khi lỗi format; bỏ khỏi trung bình, không coi là 0).
- **Vận hành:** tỷ lệ `answered`/`insufficient_evidence`/`unable_to_verify`/`error`, tỷ lệ dùng
  repair, số câu HyDE `null`, số câu retrieval lỗi/fallback.
- **Ghi chú diễn giải cố định trong báo cáo:** điểm = mức khớp với `reference` do LLM sinh (mục
  10.9); judge cùng họ model với generator; single-hop là chỉ số chính, multi-hop báo riêng
  (mục 10.10).

### 11.7 Module

Thêm vào `src/production_legal_qa_rag/evaluation/` (Pydantic v2, docstring theo
`coding-convention`; chỉ `scoring.py` import `ragas`, các module còn lại không cần):

| Module | Trách nhiệm duy nhất |
| --- | --- |
| `run_models.py` | `case_id()`, các record Pydantic (`HydeRecord`, `EmbeddingRecord`, `RetrievalRecord`, `AnswerRecord`, `RetrievalScore`, `AnswerScore`), `EvalConfig`. |
| `jsonl_store.py` | Đọc/ghi nối JSONL có validate theo model, bỏ dòng cuối hỏng, tập `case_id` đã xong (resume). |
| `key_pool.py` | Dựng 9 bộ settings/instance (HyDE, generation + Judge) mỗi bộ một key `GROQ_API_KEY_1`…`_9`, và gán bản ghi cho các bộ theo vòng tròn (mục 11.3). Đọc key qua `TestsetGeneratorSettings` (đã bắt buộc đủ 9 key). |
| `hyde_stage.py`, `embed_stage.py`, `retrieve_stage.py`, `generate_stage.py` | Mỗi file một stage S1, S2, S3, S5; chỉ điều phối (gọi code production, ghi file). |
| `scoring.py` | S4 và S6: dựng `EvaluationDataset`, chạy metric ragas theo lô, chuyển kết quả thành record. Tái dùng `build_unit_runner`-style wrapper LLM/`RunConfig` của `ragas_runner.py`. |
| `report.py` | Hàm thuần tổng hợp `report.json` từ các file JSONL (không I/O mạng, dễ test). |
| `tools/run_eval.py` | Typer mỏng: lệnh `hyde`, `embed`, `retrieve`, `score-retrieval`, `generate`, `score-answers`, `report`, `status` (in tiến độ từng stage, không tốn token); option chung `--testset`, `--output-dir`, `--limit`, `--retry-failed`. |

Thay đổi ngoài `evaluation/`: `retrieval/` thêm `PrecomputedQuery` + tham số `precomputed`
(mục 2 `retrieval_spec.md`); `RagasEmbeddingsAdapter` thêm `segment`; `.gitignore` thêm
`data/eval/phase2/embeddings.jsonl`. Chạy trong venv `eval` (`--group eval --no-group production`,
mục 3).

### 11.8 Ước lượng chi phí (thô, chưa đo — pilot `--limit 10–20` để đo token thật)

180 câu: S1 180 lượt 20b; S2 ~15 request HF; S3 không LLM; S4 2 cấu hình × 180 × 6 = **2.160
lượt** 120b; S5 ~180–360 lượt 120b + 180–360 lượt Judge 20b; S6 180 × 5 = **900 lượt** 120b.
Tổng judge RAGAS ~3.060 lượt (chạy đủ bộ metric cho cả hai cấu hình sẽ là ~3.960). Token mỗi
lượt chưa đo (giả định thô 1–2K kể cả reasoning → ~3–6M token, cỡ Phase 1); nút thắt vẫn là
TPD Groq free 200K/(tài khoản, model), nên chia nhiều ngày và dựa vào resume.

### 11.9 Rủi ro / điểm mở

1. **Venv `eval` dùng `openai` cũ hơn production** (mục 3): S1/S5 chạy code production
   (`AsyncGroq`, `ChatOpenAI.with_structured_output(method="json_mode")`) trong venv này. Phase 1
   đã chạy `ChatOpenAI` + Groq ổn nhưng chưa kiểm luồng generation/Judge — kiểm ở pilot; nếu
   lệch hành vi thì phải tách venv chạy S1–S3, S5.
2. **Điểm bị lạc quan hoá:** Evidence Judge lọc trước (faithfulness), judge RAGAS cùng họ với
   generator, `reference` chưa đối chiếu với luật (mục 10.9) — đọc điểm như xu hướng/so sánh giữa
   các lần chạy, không phải chuẩn tuyệt đối.
3. **`n = 180` nhỏ:** chênh lệch điểm giữa hai cấu hình dưới nhiễu judge không kết luận được;
   vì vậy báo số câu thắng/thua thay vì chỉ trung bình.
4. **Testset lệch nếu chạy trên raw dở dang** (mục 4.4): chỉ dùng để kiểm pipeline.
5. **Chưa chốt:** ngưỡng theo dõi (điểm nào là "đạt"), tần suất chạy lại (mỗi lần đổi prompt/
   retrieval?), có chạy lại toàn bộ hay chỉ S5–S6. Để sau khi có số đo thật đầu tiên.
6. **Rerank trên GPU 2GB của người dùng sát giới hạn:** S3 có thể gặp CUDA OOM lẻ tẻ; cách xử
   lý ở mục 11.3 (giảm `batch_size`, `--retry-failed`). Câu OOM bị bỏ khỏi so sánh MMR nếu chưa
   chạy lại được — báo số câu bị loại ở `report.json` để không so hai cấu hình trên tập khác nhau.
7. **Không đo guardrail** (mục 11.1) — tỷ lệ chặn nhầm câu hợp lệ cần một lát đánh giá riêng
   nếu sau này muốn biết.

### 11.10 Pilot trước khi chạy full (chốt 2026-09-29, người dùng đồng ý)

Sau khi developer xong, chạy toàn bộ S1→S6 với `--limit 10–20` (có thể trỏ `--testset` vào
`golden_testset_raw.json` khi chưa `finalize`), ghi ra `--output-dir data/eval/phase2_pilot`
để không lẫn kết quả thật. Pilot phải trả lời:

- **Token và số lượt gọi thật** của từng stage (thay ước lượng mục 11.8) → tính số ngày cho 180 câu.
- **Venv `eval` chạy được S1 và S5 không** (`openai` cũ hơn production, mục 11.9.1): Judge
  trả JSON đúng schema, generation ra câu bình thường.
- **S3 trên GPU 2GB có OOM không**, và `batch_size` nào đủ (mục 11.3).
- **`answer_relevancy` với embedding có segment** cho điểm hợp lý (không toàn ~0 hoặc ~1).
- **Rải 9 key hoạt động thật:** cả 9 tài khoản nhận tải xấp xỉ nhau (nhìn usage Groq), câu gán
  cho tài khoản hết quota ra `error` rồi `--retry-failed` chạy được sang key khác.
- **Resume:** dừng giữa chừng (Ctrl+C) rồi chạy lại từng stage không làm lại bản ghi đã xong.
- `report.json` đọc được, có đủ các lát (tổng, `synthesizer_name`, `source_document`).

Chỉ chạy full khi pilot đạt.
