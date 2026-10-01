# Evaluation — RAGAS: sinh golden testset (Phase 1) và chấm hệ thống (Phase 2): Reference Spec

> Giữ nguyên số mục vì code/spec khác tham chiếu (3.1, 3.2, 3.3, 4.2, 4.4, 4.5, 8, 9.3 nhất là).
> **Trạng thái (2026-10-01):** Phase 1 đã implement (PR #55, #60, #63, #64, #66); sinh testset **xong**: 49/50 đơn vị, raw 203 câu (180 single-hop + 23 multi-hop specific, 0 abstract; đơn vị `Văn bản hợp nhất bộ luật lao động.md#5` bỏ hẳn); luna (agent Codex) review ra `golden_testset_review.json`; **testset cuối `golden_testset.json` = 157 mẫu `keep`** (mục 4.2). Phase 2 (mục 11) **đã implement code trên `feat/gen-testset-ans` (2026-10-01); chưa chạy đánh giá thật 157 câu**.

## 0. Bài học xương máu và phương pháp đúng

1. **Chạy full mà chưa pilot** → 4 pilot nhỏ mới lộ: prompt ragas mặc định tiếng Anh; 3/4 `QueryStyle` là nhiễu và rò tên persona; request 12K token bị **413** vì Groq TPM 8K. → bắt buộc `adapt_prompts("vietnamese")` + ép `PERFECT_GRAMMAR`, hạ `max_token_limit` extractor ~4.000; `ragas` import `rapidfuzz` mà không khai báo dependency.
2. **Retry mặc định của ragas (10 lần với mọi `Exception`) đốt hàng trăm request** → mỗi lượt gọi chỉ **một** chủ sở hữu retry (router, mục 3.4); `ChatOpenAI(max_retries=0)`, `RunConfig(max_retries=0)`; ragas chỉ đọc retry từ `llm.run_config`.
3. **Trạng thái dùng chung phải thread-safe từ đầu:** mọi trạng thái router dưới `threading.Lock`, chốt điểm bắt đầu một lần mỗi lượt gọi, không giữ lock khi gọi mạng/ngủ.
4. **Đo bằng ký tự, ghi token thật từng đơn vị** thay vì tin hệ số ước lượng (corpus 781.007 **ký tự**, không phải byte).
5. **`generation_progress.json` là nguồn sự thật "đã xong"**, không suy từ raw; ghi raw TRƯỚC, progress SAU.
6. **`raise_exceptions=False` của ragas không dùng được cho luồng sinh testset** (0.4.3 trả `NaN` rồi crash) → tự viết vòng sinh bằng API công khai (mục 3.3).
7. **Test phụ thuộc ragas bị CI bỏ qua** → chạy cục bộ trong venv eval (`~/.cache/eval-venv-run/bin/python -m pytest tests/test_evaluation*.py`) trước khi tin.
8. **Không dùng `exc_info=True`/`logger.exception`** (in cả nội dung câu hỏi/context) → `_describe_traceback`: chỉ tên loại lỗi + `file:dòng:hàm`.
9. **Kiểm phân bố loại câu theo từng đơn vị ngay từ pilot** (multi-hop abstract = 0 đã bị bỏ sót tới lúc tổng kết).

**Phương pháp đúng:** sinh từ **văn bản nguyên bản** (độc lập thiết kế hệ thống, mục 2.1); đơn vị = Chương/Mục, mỗi đơn vị một KG riêng, checkpoint theo đơn vị, chạy nhỏ → lớn, resume nhiều ngày; dừng hẳn khi hết quota ngày; chấm bằng LLM so với `reference`; báo điểm theo loại câu và theo văn bản; đọc điểm như xu hướng, không phải chuẩn tuyệt đối.

## 1. Mục tiêu & phạm vi

Sinh **golden testset** (câu hỏi + đáp án chuẩn + context chuẩn) từ corpus pháp luật thật (`data/markdown/`) bằng `ragas.testset.TestsetGenerator` (Phase 1); chạy hệ thống thật trên testset và chấm điểm (Phase 2, mục 11). **Không dùng dữ liệu Langfuse/`chat_turns` ở Phase 1.**

**Làm (Phase 1):** cắt 6 văn bản `.md` thành **Chương** và xử lý **từng Chương một** (mỗi Chương một `KnowledgeGraph` riêng — hệ thống không xử lý được câu ghép chéo luật nên chỉ cần quan hệ trong phạm vi hẹp; mỗi Chương vừa quota Groq một ngày, lỗi chỉ mất một Chương); sinh `GENERATE_SIZE = 240` câu theo kế hoạch 192/24/24 (single / multi-hop abstract / multi-hop specific), chia theo kích thước Chương; luân phiên **9 tài khoản Groq** cho `generator_llm` (mục 3.1); lưu `golden_testset_raw.json` ngay sau mỗi đơn vị; lưu KG từng Chương; CLI Typer mỏng (`generate` với `--only`, `--testset-size`, `--reuse-knowledge-graph`, `--append`, `--retry-skipped`, `--dry-run`; `finalize`). Giữ `synthesizer_name` để Phase 2 báo điểm riêng theo loại.

**Kết quả thực tế (2026-10-01):** sinh ra 0 multi-hop abstract (mục 4.6); luna review chỉ giữ 157/203 mẫu; testset cuối = **157 mẫu `keep` (142 single-hop + 15 multi-hop specific, ≈ 9,5:1)**, không cắt 180, không random (mục 4.2).

**Không làm (Phase 1):** chạy `retrieve()`/`generate()` thật hay tính metric RAGAS; lọc câu vô nghĩa tự động bằng code; CI/cron; Postgres.

**Tiêu chí hoàn thành Phase 1:** `data/eval/golden_testset.json` chứa 157 câu hợp lệ (đủ `user_input`/`reference`/`reference_contexts`, không rỗng), đã qua review của luna, dùng thẳng làm input Phase 2.

## 2. Input & Output

**Input:** `data/markdown/*.md` — 6 văn bản pháp luật. **Output** (`data/eval/`): `golden_testset_raw.json` (203 `GoldenTestCase` chưa review, nối thêm sau mỗi đơn vị); `golden_testset_review.json` + `golden_testset_review_summary.md` (kết quả review của luna, mục 4.2); `golden_testset.json` (**157** mẫu `keep`, file duy nhất Phase 2 đọc; mỗi item: `user_input`, `reference`, `reference_contexts`, `synthesizer_name`, `source_document`, `source_section`); `generation_progress.json` (nguồn sự thật "đã làm tới đâu", mục 4.5); `knowledge_graph/<văn bản>__<số>.json` (KG từng đơn vị); `units/`, `units_plan.md` (kế hoạch chia đơn vị, mục 4.4). `golden_testset.json` giữ đúng 3 cột bắt buộc của `EvaluationDataset` ragas (`user_input`, `reference`, `reference_contexts`); `synthesizer_name` là cột phụ.

**Ai tạo cột nào:** Phase 1 tạo `user_input`, `reference`, `reference_contexts` (đoạn luật ragas đã đọc), `synthesizer_name`; Phase 2 tạo `response` (câu trả lời thật của chatbot, đầu vào chỉ `user_input`) và `retrieved_contexts` (chunk thật hệ thống tìm về); `reference` là thứ đem so.

### 2.1 Vì sao đưa văn bản nguyên bản, không dùng chunk

`generate_with_chunks` **không dùng**: (1) sinh câu hỏi từ chính chunk của `chunking/` thì ground truth do chunker định nghĩa (chunker cắt sai thì testset không hỏi trúng chỗ lỗi; câu hỏi sinh từ đúng văn bản chunk khiến BM25/dense bắt trúng dễ hơn thực tế); (2) corpus chunk không hợp làm input RAGAS (2.228 chunk, trung vị 41 token; `breadcrumb` chỉ ở metadata nên `content` không nói thuộc luật/Điều nào). **Đánh đổi chấp nhận:** `reference_contexts` là đoạn ragas tự cắt (`HeadlineSplitter(min_tokens=500)`), một đoạn bao trùm nhiều chunk hệ thống; không ảnh hưởng `context_recall`/`context_precision` bản LLM (so với `reference`, không với `reference_contexts`).

## 3. Công cụ

`ragas==0.4.3` (ghim; API xác nhận bằng đọc source: `TestsetGenerator(llm, embedding_model)`, `generate_with_langchain_docs`, `default_transforms`, `adapt_prompts(language, llm)` + `set_prompts(**...)`); `generator_llm` = `ChatOpenAI` trỏ Groq bọc `LangchainLLMWrapper`; embedding cho KG = adapter (`embeddings_adapter.py`) quanh `InferenceClient.feature_extraction` của `embedding/hf_client.py` (interface LangChain `Embeddings`), **không word-segment** (chỉ để so tương đồng trong KG); `langchain_core.documents.Document`; Typer (`tools/generate_testset.py`). `LangchainLLMWrapper`/`LangchainEmbeddingsWrapper` deprecated ở 0.4.3 (vẫn chạy) — giữ version ghim.

**Dependency (`[dependency-groups] eval`, không phải `[project] dependencies`):** `ragas>=0.4.3`, `langchain-community<0.4`, `rapidfuzz`, `langfuse>=4.15.6` (Phase 2 import tracing qua generation/HyDE; giữ group production riêng để không đổi version OpenAI). `production = ["openai>=3.19.0"]` tồn tại chỉ để `uv` tách resolve `openai`; `[tool.uv.conflicts]` giữa `production` và `eval` nên **không sync cả hai cùng lúc**. Mọi lệnh `uv run` liên quan `eval` phải mang `--group eval --no-group production`; xong việc chạy `uv sync` không cờ để trả venv mặc định.

### 3.1 Round-robin 9 tài khoản Groq (`groq_round_robin.py`)

**Credentials:** `GROQ_API_KEY_1`…`_9` (`_5`–`_9` không thuộc production). Router chỉ dùng capacity mà Groq cho phép; nếu các key chia sẻ một tổ chức thì quota dùng chung và rotation không tăng capacity (9 key = 9 tổ chức khác nhau, người dùng xác nhận 2026-10-01). **Giới hạn Groq (free, theo `(tài khoản, model)`):** `gpt-oss-120b` và `20b` đều RPM 30, RPD 1K, **TPM 8K, TPD 200K**; ràng buộc là **TPD**. Request > 8K token bị **413** (không retry được) — mỗi lượt gọi phải ≲ 4–5K token input. Header `x-ratelimit-*` không có TPD còn lại, `retry-after` chỉ có trên 429 → không dựng được sổ TPD từ header.

**Pattern luân phiên nhiều key cho CÙNG một luồng gọi** (khác `GenerationSettings`/`JudgeSettings`, mỗi bước một key cố định); không dùng `LoopBoundClient` (script chạy đúng 1 process). **`GroqRoundRobinChatModel(BaseChatModel)`** (proxy N `ChatOpenAI`; chỉ implement `_generate` sync, `_agenerate` mặc định chạy trong executor; Phase 2 thêm throttle chủ động theo `(model, key)`, mục 11.11):
- **Thứ tự thử:** `_plan_attempts` (dưới lock) lấy `start = next(cycle)` đúng **một lần**/lượt gọi rồi duyệt `(start + offset) % n`; client đang cooldown xếp CUỐI; lỗi khác 429 (400/401/403/413) bay thẳng ra; hết vòng vẫn lỗi → raise lỗi cuối.
- **Circuit breaker:** chỉ bật khi **cả n client** đều 429 **THEO NGÀY** trong CÙNG một lượt gọi; khi bật từ chối mọi lượt sau ngay (`DailyQuotaExhaustedError`, **không kế thừa `RateLimitError`** nên ragas không retry).
- **Cooldown ngày 5 phút** (`_DAILY_COOLDOWN_SECONDS = 300`): client vừa báo TPD được ưu tiên tài khoản khác; thành công thì xoá cooldown.
- **Thread-safe (bắt buộc):** con trỏ vòng, `call_counts`, cooldown, cờ breaker, token, `reasoning_effort` chỉ đọc/ghi dưới một `threading.Lock`.

### 3.2 Điều phối key mượt và tiết kiệm token (**đã implement**, PR #64)

Không đổi thuật toán chọn key; không chọn key theo "còn nhiều quota nhất" (không đọc được TPD).
- **A. Cooldown ngắn khi 429 theo phút:** đọc `retry-after` từ `error.response.headers`; thiếu/hỏng → `_MINUTE_COOLDOWN_DEFAULT = 15`; kẹp `[1, _MINUTE_COOLDOWN_MAX = 60]`. Cả n tài khoản đều cooldown phút (không có bằng chứng ngày): chờ tới lúc sớm nhất hết (≤60s, không giữ lock) TRƯỚC khi thử. Clock và `sleep` tiêm được để test không chờ thật.
- **B. Đếm token thật:** router cộng dồn `prompt_tokens`, `completion_tokens`, `reasoning_tokens` theo client dưới lock (`token_totals`); `UnitResult`/`UnitProgress` ghi `tokens`/`reasoning_tokens`; `--dry-run` ước lượng phần còn lại bằng hệ số đo được `(Σtokens − 4000·n)/Σchars` khi ≥ 3 đơn vị `done` có `tokens > 0` (đo được 3,94 token/ký tự, thấp hơn ước tính 5,5).
- **C. `reasoning_effort="low"` CHỈ khi dựng KG:** `with router.reasoning_effort("low"):` truyền kwarg `reasoning_effort` xuống `client._generate(...)`; `_build_knowledge_graph` chạy trong context; `adapt_prompts`, persona, scenario, sample thì KHÔNG; hằng `_KG_REASONING_EFFORT = "low"` trong `ragas_runner.py`.

### 3.3 Giữ phần đã sinh khi lỗi giữa đơn vị (**đã implement**, PR #64)

`TestsetGenerator.generate` huỷ cả lô khi một câu lỗi → thay bằng vòng sinh riêng trong `RagasUnitRunner._generate_cases`: `generate_personas_from_kg(kg, llm, num_personas=3)` một lần/đơn vị → mỗi loại có quota > 0 (tuần tự): `generate_scenarios(n, knowledge_graph, persona_list)` → `generate_sample(scenario)` từng cái bọc `try/except`, đồng thời tối đa `MAX_WORKERS` (`asyncio.Semaphore`). Test khoá chữ ký các hàm ragas (`importorskip("ragas")`).
- **Sample lỗi:** `DailyQuotaExhaustedError` → ngừng sample mới, giữ mọi sample đã xong, đơn vị **`partial`**, chương trình dừng (mã thoát 1); lỗi khác (parse, timeout hết retry) → BỎ sample đó, log CHỈ tên synthesizer + tên loại lỗi, đếm `skipped_samples`; không sinh được câu nào mà có lỗi → `UnitGenerationError` (không ghi `done` rỗng).
- **Ghi:** raw TRƯỚC, progress SAU (kể cả `partial`). `UnitProgress.status ∈ {done, partial, skipped}`. **Chạy tiếp `partial`:** dùng lại KG, quota còn lại = `quota − questions` (kẹp ≥ 0), rồi chuyển `done`; **`--append` trên đơn vị `done` bị ngắt giữa chừng GIỮ `done`**.

### 3.4 Retry/pass, fail-fast và báo suy giảm (**đã implement**, PR #66)

Một lỗi RAGAS hoặc mạng không được làm job dừng và lãng phí phần đã chạy; không dùng retry mù làm nhân số request/token. `DailyQuotaExhaustedError` dừng ngay toàn process, giữ checkpoint. Lỗi lập trình/I/O không rõ nguồn fail-fast.

**Một chủ sở hữu retry:** `GroqRoundRobinChatModel` là tầng DUY NHẤT retry request HTTP; trạng thái từng credential: `ready`, `minute_cooldown`, `daily_cooldown`, `disabled`.
- **429 phút:** đọc `retry-after`, đưa vào `minute_cooldown`, thử credential khác trước; cả vòng chỉ gặp cooldown phút → `RateLimitError`, lượt sau chờ credential sớm nhất hết cooldown.
- **429 ngày:** `daily_cooldown` 5 phút; chỉ khi toàn bộ credential còn hoạt động báo quota ngày trong cùng lượt thì raise `DailyQuotaExhaustedError`.
- **Timeout/connection/5xx/498:** retry **tối đa một lần** (backoff + jitter), mỗi lần chỉ một credential `ready`. **400/401/403:** không retry; `401`/`403` vô hiệu hoá credential. **413:** không retry hay đổi key.
- **Lỗi RAGAS sau HTTP 200** (`KeyError`/parse/pydantic): vá tất định ở wrapper (3 `Clean*Synthesizer.prepare_combinations` lọc `persona_item_mapping`/`persona_concepts` còn persona thực có); lỗi semantic chưa biết → `generate_personas`/`generate_scenarios`/`generate_sample` chạy lại đúng một lần, lần hai lỗi thì bỏ loại/sample. Nếu chuỗi `__cause__`/`__context__` có `DailyQuotaExhaustedError`, `APIStatusError`, `APIConnectionError`, `APITimeoutError`, `RateLimitError` thì **không** gửi thêm request ở tầng RAGAS. Dựng KG không retry nguyên unit.
- **Không được `done` rỗng:** unit có quota dương mà không đổi được sample nào và có `skipped_*` → `UnitGenerationError(stage="generation")`; unit ghi `skipped` (không ghi raw/progress `done`), KG giữ để `--retry-skipped --only`.
- **Systemic breaker:** **2 unit liên tiếp** `skipped` cùng chữ ký `(stage, root exception type, HTTP status hoặc null)` → dừng trước unit kế, ghi `last_failure` đã che, mã thoát 1; một unit sinh được ≥1 case reset bộ đếm.
- **Mã thoát (`generate`):** 0 phạm vi sạch; 1 quota ngày, systemic breaker hoặc lỗi không phân loại; 2 đầu vào/cấu hình sai; 3 chạy hết nhưng có unit/type/sample bị bỏ. `GenerationReport` tách unit bị bỏ trong invocation khỏi unit vốn `done`.

## 4. Workflow (`testset_generator.py`)

Dựng 9 `ChatOpenAI` → `generator_llm = LangchainLLMWrapper(GroqRoundRobinChatModel(...))` (gán `RunConfig` ngay lúc khởi tạo) → `TestsetGenerator` với `adapt_prompts("vietnamese")` cho MỖI synthesizer (gọi 1 lần rồi dùng lại; ép `PERFECT_GRAMMAR`). Cho từng đơn vị (sắp xếp số ký tự tăng dần, lọc `--only`): bỏ qua nếu `done` trong progress (trừ `--append`) → `Document` → `default_transforms` với `max_token_limit` extractor ~4.000 và `max_workers` ~4 → dựng KG (trong `reasoning_effort=low`) và **lưu KG ngay** (nguyên tử) → sinh câu (mục 3.3/3.4) → gắn `source_document`, `source_section` → nối raw, ghi progress. Không còn thao tác tay ngoài code: sinh xong → luna review → `finalize` lấy mẫu `keep`.

**Cắt đơn vị (`unit_splitter.split_document`, thuần):** cắt tại `## ` (Chương), tách theo `### ` (Mục) khi quá lớn, gộp phần nhỏ, bỏ mở đầu và chú thích cuối; `source_section` của đơn vị gộp là các nhãn nối bằng " + ". **Quota câu (`allocate_questions`, thuần):** `GENERATE_SIZE = 240` thành 192/24/24; mỗi loại phân bổ cho đơn vị tỷ lệ theo ký tự (phần dư lớn nhất); đơn vị nhận 0 câu multi-hop chỉ chạy single-hop; multi-hop cần ≥2 đoạn nên có thể sinh ít hơn quota, không bù. `GENERATE_SIZE`, `MIN_UNIT_CHARS`, `MAX_UNIT_CHARS` là hằng số nội bộ.

### 4.1 Ngôn ngữ và chất lượng câu hỏi/đáp án

**Bắt buộc:** `adapt_prompts("vietnamese", llm=...)` cho mọi synthesizer và **ép `QueryStyle.PERFECT_GRAMMAR`** (không adapt → câu hỏi lẫn tiếng Anh; mặc định ragas trộn 4 `QueryStyle` nhiễu kèm rò **tên persona**). Nhiễu có thể thêm lại như một lát đo độ bền riêng ở phase sau, không trộn vào testset chính.

**Khuyết điểm chấp nhận có chủ đích:** đáp án đôi khi **dính chữ** ("cưtrú", "dịchvụ") — thiếu dấu cách thật do `gpt-oss-120b` sinh, **không thêm NFKC**; `reference` chỉ LLM chấm nên ít ảnh hưởng; không dùng làm lý do loại khi review. Câu trùng ý trong cùng đoạn ragas và chủ đề nông ("Chính phủ quy định chi tiết") do luna review loại (mục 4.2).

### 4.2 Chốt testset cuối (`finalize`) và sinh bù

**Chốt 2026-10-01 (người dùng chọn hướng B sau khi luna review):** testset cuối = các mẫu raw có `verdict = "keep"` trong `data/eval/golden_testset_review.json` — **157 mẫu (142 single-hop + 15 multi-hop specific, ≈ 9,5:1)**, giữ thứ tự raw; **không cắt xuống 180, không random, không bù mẫu bị loại**. Mốc 180 và tỷ lệ 162/18 bị bỏ vì luna chỉ giữ 157/203 mẫu; bù 23 mẫu bị loại (`forced_fill`, `quality` 1–2) sẽ đưa nhiễu vào đúng chỗ cần đo.
**Quy trình đã làm:** luna (agent Codex) đọc cả 203 mẫu, chấm `keep/drop` + `quality` 1–5 + `reason_code`; 46 mẫu bị loại: `answer_unsupported` 13, `shallow_topic` 12, `mechanical` 5, `bad_multihop` 5, `duplicate` 4, `not_self_contained` 3, `language` 3, `transitional_clause` 1. `golden_testset.json` sinh một lần từ review (đã kiểm: 157 mẫu, nguyên văn raw, `case_id` duy nhất, trường bắt buộc không rỗng). `golden_testset_candidate.json` (180 mẫu có bù) **không dùng**.
**Code `finalize` (đã sửa):** bỏ cơ chế cắt phân tầng đúng 180; hàm thuần: đọc raw + review, ghi các mẫu `keep` theo thứ tự raw, kiểm số dòng review khớp raw, `case_id` (12 hex đầu của sha256 `user_input`) khớp và duy nhất, trường bắt buộc không rỗng; không còn `TARGET_SIZE`.
**Sinh bù:** `generate --only <văn bản> --reuse-knowledge-graph --append --testset-size N` vẫn dùng được (bỏ câu `user_input` trùng y hệt), nhưng câu mới phải qua review lại; hiện không dùng.

### 4.3 Mã mẫu đã kiểm chứng bằng pilot

Ba việc (mã ở `ragas_runner.py`): (1) `cap_token_limit(transforms, limit)` duyệt `Parallel`/list, gán `max_token_limit` cho mọi `LLMBasedExtractor`; (2) subclass 3 synthesizer (`CleanSingleHop/MultiHopAbstract/MultiHopSpecificSynthesizer`) ghi đè `prepare_combinations` để ép `styles = [QueryStyle.PERFECT_GRAMMAR]`; hai multi-hop còn lọc `persona_item_mapping` về đúng tập persona đã sinh (RAGAS 0.4.3 ném `KeyError` với key lạ); (3) `asyncio.run(synthesizer.adapt_prompts("vietnamese", llm=...))` + `set_prompts(**adapted)`, gọi một lần cho mỗi synthesizer.

### 4.4 Chia đơn vị và lập kế hoạch token

**Quy tắc:** đơn vị chuẩn = Chương; lớn hơn `MAX_UNIT_CHARS = 30.000` thì tách theo `### Mục`; nhỏ hơn `MIN_UNIT_CHARS = 6.000` gộp với kề; **KHÔNG bỏ Chương "Điều khoản thi hành"**; **bỏ khối chú thích cuối file** (`_strip_footnotes`: cắt từ `---` đứng ngay trước `[1] …` cuối file) và phần mở đầu trước Chương đầu. Khoá đơn vị dùng **số thứ tự**, không dùng số La Mã (`Điều kiện lao động và quan hệ lao động.md` có hai Chương "XI"); `Quy định mức lương tối thiểu.md` không có `## ` (cả file là 1 đơn vị). Ước lượng chi phí một đơn vị: `≈ 5,5 token/ký tự + 4K cố định`, thay bằng hệ số đo được (mục 3.2 B).
**Kết quả:** **50 đơn vị** (BHXH 12, BHYT 6, TNCN 3, mức lương tối thiểu 1, BLLĐ hợp nhất 16, Điều kiện lao động 12), 720.573 ký tự; nhỏ nhất 6.000, lớn nhất 29.075 ký tự; kế hoạch ở `data/eval/units_plan.md`, nguyên văn ở `data/eval/units/`. Chạy nhỏ → lớn.

### 4.5 Theo dõi tiến độ và chạy tiếp nhiều ngày

Hôm nay dừng ở đơn vị 36, mai chạy lại CÙNG lệnh thì tiếp từ 37. **Thứ tự chạy** = số ký tự tăng dần. **Khoá đơn vị** = `<tên file .md>#<số thứ tự 1-based của split_document>`. `generation_progress.json` (Pydantic `GenerationProgress`, ghi **nguyên tử**) gồm `units` (khoá → `UnitProgress`) và `last_failure` (`unit`, `error` đã che, `at`).
- **File này (không phải raw) quyết định "đã xong".** Chết giữa hai bước ghi → lần sau thấy đơn vị có dòng trong raw mà chưa có trong progress → coi đã xong, ghi bổ sung + log.
- **Khi hết quota ngày, systemic breaker hoặc exception không phân loại:** ghi `last_failure`, in tóm tắt, **dừng luôn không thử đơn vị kế**, mã thoát 1; lần chạy sạch sau đó xoá `last_failure`.
- **KG và đơn vị dở:** KG lưu ngay sau `apply_transforms`; đơn vị chưa `done` **tự dùng lại** KG nếu `page_content` node DOCUMENT vẫn khớp văn bản; đơn vị `done` (`--append`) chỉ dùng lại khi có `--reuse-knowledge-graph`. Ép dựng lại: xoá `knowledge_graph/<văn bản>__<số>.json`.
- **`--append` và `--retry-skipped` bắt buộc đi kèm `--only`** (thiếu → mã thoát 2, không gọi LLM). Chạy lại unit `skipped`: `generate --only "<tên>#<số>" --retry-skipped`.
- **Kiểm tra khớp nguồn:** unit `done`/`skipped` mà `chars` khác kết quả chia hiện tại → dừng với lỗi nêu tên unit. **`--dry-run`** (không tốn token): bảng theo thứ tự chạy gồm khoá, ký tự, ước lượng token, trạng thái (`xong dd/mm hh:mm` / `dở (đã có N/M câu)` / `bỏ qua <stage>` / `chưa`).
Chạy nền: `setsid nohup uv run --group eval --no-group production tools/generate_testset.py generate > data/eval/generate.log 2>&1 &`.

### 4.6 Tiến độ thực tế và việc còn lại

Sinh xong 49/50 đơn vị, raw 203 câu (180 single-hop + 23 multi-hop specific, **abstract = 0**: nghi KG theo Chương nhỏ không có cụm đoạn cho `MultiHopAbstractQuerySynthesizer`, chưa xác minh). **Chốt 2026-10-01 (người dùng):** chấp nhận không có multi-hop abstract (ghi vào báo cáo Phase 2, mục 11.6); không sinh bù abstract; testset cuối là 157 mẫu `keep` của review luna (mục 4.2). Đơn vị `skipped` `Văn bản hợp nhất bộ luật lao động.md#5` **bỏ hẳn** (203 > 180, BLLĐ còn nhiều đơn vị khác). Việc còn lại: sửa code `finalize` theo mục 4.2 (nhánh Phase 2); `golden_testset.json` đã sinh. Mẫu tiếng Anh trong raw: đã dịch ở PR #71.

## 5. Model dữ liệu (`models.py`)

`GoldenTestCase(user_input, reference, reference_contexts: list[str], synthesizer_name: str | None, source_document: str | None, source_section: str | None)` — `source_document`/`source_section` do code gắn, dùng cho resume và báo điểm theo luật/Chương ở Phase 2; không lẫn với `RetrievedChunk`.
`UnitProgress(title, chars, estimated_tokens, questions: dict[str,int], llm_calls, seconds, completed_at, status: Literal["done","partial","skipped"]="done", skipped_samples: int=0, skipped_question_types: set[str]=set(), skipped_stage: Literal["knowledge_graph","generation"]|None=None, error_type: str|None=None, attempts: int=0, tokens: int|None=None, reasoning_tokens: int|None=None)`; `UnitFailure(unit, error, at)`; `GenerationProgress(units: dict[str, UnitProgress], last_failure: UnitFailure | None)`. Trường `skipped_*`/`error_type`/`attempts` là metadata, không chứa nội dung prompt hay exception message.

## 6. Config (`config.py`)

`TestsetGeneratorSettings` (`SettingsConfigDict(env_file=".env", extra="ignore")`): `api_key`…`api_key_9` (alias `GROQ_API_KEY_1`…`_9`, mỗi cái `Field(min_length=1)`), `model_name = "openai/gpt-oss-120b"`, `timeout_seconds = 60`. **Cả 9 key BẮT BUỘC và không rỗng**; thiếu thì `EvalInputError` bọc lỗi pydantic (chỉ nêu TÊN biến, mã thoát 2). Không dùng chung class với `GenerationSettings`; không setting riêng cho embeddings (`embeddings_adapter.py` tái dùng `EmbeddingSettings`). Module không đọc `.env` trực tiếp; `.env.example` có `GROQ_API_KEY_1`…`_9`.

## 7. Module (`src/production_legal_qa_rag/evaluation/`)

`models.py`; `corpus_loader.py`; `embeddings_adapter.py`; `groq_round_robin.py` (router 9 client, mục 3.1–3.2); `unit_splitter.py` (`EvalUnit`, `split_document`, `split_directory`; thuần Python, không import `ragas`); `ragas_runner.py` (`RagasUnitRunner`, `cap_token_limit`, 3 `Clean*Synthesizer`, `build_run_config`, vòng sinh mục 3.3–3.4; **toàn bộ code chạm ragas nằm ở đây**); `testset_generator.py` (điều phối, `allocate_questions`, systemic breaker, `finalize_testset`, `_describe_traceback`, dry-run); `tools/split_eval_units.py` (chia + ghi `units/`, `units_plan.md`); `tools/generate_testset.py` (Typer mỏng: `generate`, `finalize`). Phase 2 thêm module vào cùng package (mục 11.7).

## 8. Xử lý lỗi

| Sự cố | Xử lý |
| --- | --- |
| Hết quota ngày trên mọi credential còn hoạt động | Dừng process ngay, log khoá unit + loại lỗi đã che, ghi `last_failure`, giữ `partial` nếu đã có sample; không thử credential/unit mới |
| 429 theo phút; timeout/connection/5xx/498 | Chỉ router retry theo mục 3.4; hết retry thì pass scope RAGAS nhỏ nhất, không dừng job |
| 400/401/403/413 hoặc lỗi RAGAS sau HTTP 200 | Không retry mù (mục 3.4); dựng KG lỗi đã phân loại → unit `skipped` |
| Hai unit liên tiếp `skipped` cùng chữ ký; hoặc exception không phân loại | Dừng process trước unit kế, ghi `last_failure`, mã thoát 1; không nuốt `OSError`/`TypeError`/bug thành `skipped` |
| Job chạy hết nhưng có unit/type/sample bị bỏ | Giữ checkpoint/raw, in tóm tắt suy giảm, mã thoát 3 |
| `--only` sai tên; `data/markdown` thiếu/không UTF-8; `--append` thiếu `--only`; thiếu/rỗng `GROQ_API_KEY_*` | `EvalInputError` (mã thoát 2) trước khi gọi LLM, chỉ nêu TÊN biến, không traceback |
| `embeddings_adapter.py` nhận response sai định dạng | Raise lỗi rõ, không trả vector rỗng |
| Unit `done` không có `--append`; unit `done` mà `chars` khác kết quả chia; unit có dòng raw chưa có trong progress; `generation_progress.json` hỏng | Bỏ qua + log / dừng nêu tên unit / coi đã xong + log / raise lỗi rõ (không coi như "chưa làm gì") |
| `finalize`: review thiếu/lệch raw (số dòng, `case_id`); một mẫu `keep` thiếu/rỗng trường bắt buộc | Dừng với lỗi nêu vị trí dòng lỗi/lệch; không ghi file thiếu |

## 9. Nghiệm thu thủ công

1. Chạy `uv run --group eval --no-group production tools/generate_testset.py generate [--only ...]`, chia nhiều ngày theo quota (xem tiến độ bằng `--dry-run`). 2. Raw đủ 6 văn bản, đủ trường không rỗng. 3. Kiểm round-robin: log token/lượt theo tài khoản hoặc dashboard Groq — tải xấp xỉ đều (~1/9), không request nào thất bại hẳn vì rate limit. 4. Không duyệt tay: luna review 203 mẫu → `golden_testset_review.json`; rủi ro ở 10.14. 5. Sửa `finalize` theo mục 4.2; `golden_testset.json` = 157 mẫu `keep` (đã sinh).

## 10. Rủi ro / điểm mở

1. `embeddings_adapter.py` là code mới, chưa có tiền lệ — test tay kỹ trước khi tin cho việc build KG. 2. Câu ragas sinh lệch phân bố so với câu hỏi người dùng thật (thiên về "Điều X quy định gì" hơn tình huống); hạn chế chung của synthetic data. 3. Resume chỉ ở mức đơn vị (và sample đã xong, mục 3.3), không ở mức lượt gọi LLM. 6. Router gọi đồng thời (ragas `max_workers=4`) nên trạng thái dùng chung phải thread-safe.
8. (Phase 2) hit@k theo `chunk_id` cần map `reference_contexts` sang chunk hệ thống (nhiều-1; còn phải đo trên testset thật) — không làm ở Phase 2. 9. **`reference` không được đối chiếu với luật bởi người** (luna chỉ đối chiếu với `reference_contexts` và loại 13 mẫu `answer_unsupported`): điểm Phase 2 là "mức khớp với đáp án do LLM sinh", không phải "đúng luật tuyệt đối"; nếu điểm bất thường, việc đầu tiên là kiểm mẫu vài `reference` với luật gốc. 10. **Multi-hop ghép nhiều Điều là loại hệ thống yếu** → Phase 2 báo điểm tách theo `synthesizer_name` (single-hop là chỉ số chính); multi-hop abstract = 0, chỉ còn specific.
11. Đơn vị gộp nhiều Chương (BLLĐ #16 = XV+XVI+XVII) có thể cho câu kém đồng nhất. 12. Đáp án dính chữ (mục 4.1): chấp nhận.
14. **Không duyệt tay, luna review thay (2026-10-01):** luna đọc cả 203 mẫu và loại 46 (mục 4.2); testset cuối 157 mẫu `keep` (12 mẫu `quality` 2, 49 mẫu `quality` 3). Rủi ro còn lại: nhận xét của luna chưa có người đối chiếu với luật; câu "nhạt" (`quality` 2–3) vẫn nằm lại; `reference_contexts` của 16/23 multi-hop chỉ có 1 đoạn (mất `<2-hop>`) nên không kiểm được đáp án với đoạn thứ hai bằng dữ liệu lưu sẵn. Điểm Phase 2 đọc như xu hướng; nếu bất thường, đọc câu hỏi và `reference` của các case điểm thấp nhất (lọc theo `quality` trong review).

## 11. Phase 2 — Chạy pipeline thật và chấm điểm (code implement 2026-10-01, chưa chạy đánh giá thật)

### 11.1 Mục tiêu, phạm vi

Chạy từng câu của `golden_testset.json` (157 câu: 142 single-hop + 15 multi-hop specific) qua retrieval + generation thật, chấm bằng RAGAS; trả lời (1) **bật hay tắt MMR** thì retrieval tốt hơn (`retrieval_spec.md` mục 6) và (2) chất lượng câu trả lời cuối (đã qua Evidence Judge) theo loại câu và theo văn bản luật.
**Bộ metric = 4 metric chuẩn RAGAS:** `context_recall` (S4, cả hai cấu hình — MMR là bài toán bao phủ), `context_precision` (S4b, chỉ cấu hình đã chọn — đo thứ hạng do reranker quyết, **không dùng để chọn MMR**), `faithfulness` + `answer_relevancy` (S6, cấu hình đã chọn).
**Làm:** HyDE → embed → retrieve (`mmr_on` rồi `mmr_off`, tuần tự cả hai) → chấm recall → *(người dùng chọn cấu hình)* → generation (`GenerationPipeline.generate`) → chấm câu trả lời → chấm precision → báo cáo.
**Không làm:** guardrail/condense/cache/`api/`/`conversation/` (testset là câu đơn lượt độc lập, cache làm sai số đo; tỷ lệ guardrail chặn nhầm KHÔNG được đo); Groq Batch API; hit@k; lát đánh giá nhiễu; lấy mẫu Langfuse; CI/cron; `factual_correctness`/`answer_correctness` (người dùng bỏ; chấp nhận chỗ hở "bám chunk nhưng thiếu ý"); pilot bắt buộc (mục 11.10).
**Tiêu chí hoàn thành:** chạy đủ stage trên 157 câu, có `data/eval/phase2/report.json` và bảng so sánh MMR bật/tắt + điểm câu trả lời theo `synthesizer_name`/`source_document`.

### 11.2 Stage, file trung gian, resume

Mỗi stage đọc file stage trước, ghi một JSONL ở `data/eval/phase2/`, chạy lại được — đổi prompt generation chỉ chạy lại S5, S6, S4b. Khoá bản ghi = `case_id` = 12 ký tự hex đầu `sha256(user_input)`.

| Stage | Việc | Tài nguyên | File ra |
| --- | --- | --- | --- |
| S1 `hyde` | `HydeGenerator.generate(user_input)` cho mọi câu | `gpt-oss-20b`, 9 key, hàng đợi chung (11.3) | `hyde.jsonl` |
| S2 `embed` | Embed `[hypo, query]` bằng `QueryEmbedder` (pyvi, cùng model index), gom 25 text/request | HF Inference | `embeddings.jsonl` (không commit) |
| S3 `retrieve` | `RetrievalPipeline.retrieve(query, use_mmr=…, precomputed=…)`, **tuần tự** từng câu, **tuần tự cả hai cấu hình** | Pinecone, BM25, rerank GPU local | `retrieved_mmr_on.jsonl`, `retrieved_mmr_off.jsonl` |
| S4 `score-recall` | RAGAS `context_recall`, **cả hai cấu hình** (1 lượt/mẫu → **≤314 lượt**; câu mà hai cấu hình ra **cùng danh sách `chunk_id` theo cùng thứ tự** chỉ chấm một lần, bản ghi cấu hình kia chép điểm + `reused_from`, không gọi LLM) | `gpt-oss-120b`, 9 key, router + throttle | `recall_scores.jsonl` |
| S5 `generate --config` | `GenerationPipeline.generate(query, chunks)` cho cấu hình đã chọn | 120b + Judge 20b, 9 key, hàng đợi chung + throttle | `answers.jsonl` |
| S6 `score-answers` | `faithfulness` + `answer_relevancy` trên câu `answered` (5 lượt/câu, ≤785) | 120b 9 key + embedding HF | `answer_scores.jsonl` |
| S4b `score-precision` | `LLMContextPrecisionWithReference` (1 lượt/chunk → 5 lượt/câu, 785 lượt), **chỉ cấu hình đã chọn** | 120b, 9 key | `precision_scores.jsonl` |
| `report` | Tổng hợp, không LLM | — | `report.json` + bảng terminal |

**Thứ tự chạy: S1 → S2 → S3 → S4 → *chọn MMR* → S5 → S6 → S4b** (S4b cuối vì chủ yếu đo rerank; hết quota thì phần thiếu là S4b). S4, S5, S6, S4b cùng bucket 120b → chạy lần lượt (mục 11.11).

**Quy tắc chung:** stage bỏ qua `case_id` đã có; thiếu bản ghi stage trước thì báo số còn thiếu và chỉ xử lý phần đã có; ghi nối từng dòng bởi **một nơi ghi duy nhất** (một lock hoặc một task ghi; S4/S4b/S6 ghi theo lô `SCORING_BATCH_SIZE = 10`); dòng cuối hỏng thì bỏ + log; mã thoát 0/1/2 như Phase 1; `--testset` (mặc định `golden_testset.json`; có thể trỏ raw để kiểm pipeline), `--limit N` (tuỳ chọn); `status` in số xong/lỗi/chưa xử lý từng stage. **Phân loại lỗi:** lỗi không đoán trước (parse, NaN, timeout, 5xx, 429 lẻ tẻ) → ghi `error` + đánh dấu rồi đi tiếp; xong hết người dùng xem `status` và chạy lại bằng `--retry-failed` (mặc định KHÔNG); 429 theo ngày → dừng, phần chưa làm ở lại "chưa xử lý", không ghi `error`.

### 11.3 Chi tiết từng stage

**Hàng đợi chung 9 worker (S1, S5).** `HydeGenerator`, `AnswerGenerator`, `EvidenceJudge` mỗi cái đọc key cố định nên không tự xoay. Eval dựng 9 bộ (`HydeSettings`/`GenerationSettings`/`JudgeSettings` truyền key tường minh theo `validation_alias`; đặt `GROQ_API_KEY_4` của `GenerationSettings` là `None`), mỗi bộ một **worker** (asyncio task) lấy `case_id` còn lại từ **một hàng đợi dùng chung**, xử lý tuần tự từng câu. Key báo 429 theo ngày → worker đó dừng, các worker khác chạy tiếp, câu chưa làm ở lại "chưa xử lý". Helper ở `key_pool.py`, không sửa code production. **S1:** tái dùng `HydeGenerator` nguyên trạng (đã có throttle theo bucket); `None` → `hypothetical_document: null` và bỏ nhánh A đúng như production; phân biệt `error` với `null` không lỗi.
**S2:** gom `[hypo, user_input]` nhiều câu vào request 25 text, không đổi thứ tự; cả hai cấu hình S3 dùng chung embedding. **S3:** cần `precomputed` (`retrieval_spec.md` mục 2, thay đổi duy nhất ở code production); **một** `RetrievalPipeline`, chạy **tuần tự** (hết 157 câu `use_mmr=True` rồi `False`; GPU 2GB, `RETRIEVE_CONCURRENCY = 1`); CUDA OOM → giảm `batch_size` của `LocalReranker` rồi `--retry-failed`, **không đổi model hay `max_length`**. **Fallback rerank là lỗi, không phải kết quả:** chunk nào có `rerank_score is None` → ghi `error` và không chấm; kết quả rỗng → `error = "no_context"`; `RetrievalError` → `error`.
**S4:** `LLMContextRecall` (cần `user_input`, `retrieved_contexts`, `reference`; **1 lượt gọi/mẫu**: câu hỏi + 5 chunk nối + `reference` → LLM tách từng câu của `reference` và gán 1/0 "suy ra được từ chunk"; điểm = tỉ lệ câu được 1) trên **cả hai cấu hình**, 157 × 2 = 314 mẫu, **≤314 lượt** (câu hai cấu hình ra cùng danh sách `chunk_id` theo cùng thứ tự chỉ chấm một lần ở cấu hình đầu, cấu hình sau ghi `reused_from` + chép điểm — cùng prompt thì cùng điểm); judge = `LangchainLLMWrapper(GroqRoundRobinChatModel)` (9 key, cùng `RunConfig`, thêm throttle chủ động — mục 11.11); chuỗi mỗi chunk = đúng phần chunk trong `build_context` của generation (breadcrumb + content/raw_table), tách hàm dùng chung nếu cần.
**Chọn cấu hình sau S4 (người dùng quyết):** `report` in bảng so sánh `context_recall` và số câu `mmr_on` hơn/thua/hoà `mmr_off` theo từng câu (157 câu: chênh trung bình nhỏ dễ là nhiễu judge). Lý do dùng recall: MMR chỉ đổi **tập candidate vào union trước rerank**, thứ tự cuối và cắt top 5 do reranker quyết, nên MMR ảnh hưởng chủ yếu tới độ **bao phủ**. Không khác biệt rõ thì **tắt MMR** (đơn giản hơn, bớt một lượt Pinecone). Kết luận ghi vào `retrieval_spec.md` mục 6.
**S4b:** `LLMContextPrecisionWithReference` trên cấu hình đã chọn. RAGAS gốc gọi LLM **một lượt cho mỗi chunk** (mỗi lượt 1 phán quyết 0/1, rồi tính average precision@k theo thứ hạng) → 5 lượt/câu, 785 lượt; mỗi lượt chỉ chứa 1 chunk + câu hỏi + `reference` nên nhẹ. Đo **chất lượng thứ hạng của reranker**, KHÔNG dùng để chọn MMR. Không tự viết bản gộp 5 chunk/1 lượt.

### 11.4 S5 — Generation (phương án B)

`GenerationPipeline.generate(query, chunks)` không gọi guardrail và không retrieve; kết quả là câu người dùng thật thấy (đã qua hard gate + Judge + tối đa 1 repair). Eval dựng **9 `GenerationPipeline` độc lập, mỗi cái một key** (truyền `generator` và `judge` vào `GenerationPipeline.__init__`, không sửa code production): `AnswerGenerator` (120b) và `EvidenceJudge` (20b) cùng dùng key i (hai bucket khác model).
**Luồng một câu:** draft (120b, 1 lượt) → hard gate (code) → Judge (20b, 1 lượt) → `pass` / `insufficient_evidence` (từ chối) / `repair` (1 draft + 1 Judge nữa; ngân sách repair **1 lần/câu**, hết mà vẫn lỗi → `unable_to_verify`). Mỗi câu 2–4 lượt; 157 câu = 314–628 lượt; **tỷ lệ repair thật chưa đo**.
**Throttle (mục 11.11):** wrapper quanh `AnswerGenerator` gọi `acquire(est)` trên `get_throttle(model, key)` trước `draft`/`repair`, `settle` bằng `usage` thật (ước lượng = độ dài prompt/`CHARS_PER_TOKEN` + `EXPECTED_COMPLETION_TOKENS`, **không** dùng `max_completion_tokens`). `EvidenceJudge` đã throttle theo bucket sẵn có. **Bẫy:** `ThrottleTimeout` ở Judge bị pipeline coi là lỗi không phải 429 → ra `unable_to_verify` (kết quả hợp lệ, không chạy lại) làm sai số đo; wrapper quanh `EvidenceJudge.judge` phải bắt `ThrottleTimeout` và ném lại lỗi mà `_is_rate_limited` nhận ra (status 429) để record thành `error` chạy lại được.
`AnswerRecord`: `case_id`, `config`, `outcome ∈ {answered, insufficient_evidence, unable_to_verify, error}`, `response` (thô, còn marker `[n]`), `citations`, `repair_used`, `warning_codes`, `error_code`, `usage`, **`prompt_version`** (`PROMPT_VERSION` generation, hiện v11). `insufficient_evidence` và `unable_to_verify` là **kết quả hợp lệ của hệ thống**, không chạy lại; chỉ `error` chạy lại được.
**Hệ quả đọc điểm:** câu bị từ chối không có `response` nên RAGAS không chấm — báo riêng **tỷ lệ từ chối** (theo loại và `synthesizer_name`) và **điểm end-to-end** (mục 11.6); `faithfulness` đo trên câu ĐÃ qua Judge nên cao hơn faithfulness của draft.

### 11.5 S6 — Chấm câu trả lời

`Faithfulness` (2 lượt cố định của RAGAS: tách khẳng định, rồi kiểm tất cả khẳng định so với chunk trong 1 lượt) + `ResponseRelevancy` (`answer_relevancy`, **`strictness = 3`** → 3 lượt riêng vì wrapper round-robin không có `n` và Groq không hỗ trợ `n>1`; mỗi lượt đoán lại câu hỏi từ câu trả lời + cờ `noncommittal`, điểm = trung bình cosine với câu hỏi gốc) — 5 lượt/câu, chỉ trên `answered` (tối đa 785 lượt).
**Strip citation:** trước khi chấm, bỏ marker `[n]` khỏi `response` — chỉ xoá `[n]` với n thuộc tập `citations` hợp lệ của record, không đụng `[n]` khác (vd. trích dẫn trong văn bản luật); `answers.jsonl` giữ bản thô.
`answer_relevancy` cần embedding: dùng `RagasEmbeddingsAdapter` với **`segment=True`** (tham số mới, mặc định `False` để không đổi Phase 1) áp `ViTokenizer` — cosine giữa câu hỏi gốc và các câu hỏi sinh lại cần cùng không gian với model đã huấn luyện trên văn bản đã segment (`embedding_spec.md` mục 4).

### 11.6 Báo cáo (`report.json`)

**So sánh retrieval:** trung bình `context_recall` của `mmr_on` và `mmr_off` (tổng, theo `synthesizer_name`, theo `source_document`) + số câu thắng/thua/hoà (câu `reused_from` tính hoà; báo riêng số câu trùng tập chunk), chỉ trên `case_id` hợp lệ ở CẢ HAI cấu hình (báo số câu bị loại). **Chất lượng rerank:** `context_precision` của cấu hình đã chọn theo cùng các lát. **Điểm câu trả lời:** trung bình `faithfulness`/`answer_relevancy` theo cùng các lát, kèm `n` mỗi lát và số `null` (NaN bỏ khỏi trung bình, không coi là 0).
**Điểm end-to-end:** bên cạnh điểm trên câu `answered`, báo thêm điểm tính trên MỌI câu hợp lệ, trong đó câu `insufficient_evidence`/`unable_to_verify` tính **0** cho mỗi metric câu trả lời; câu `error` loại khỏi cả hai (báo số).
**Vận hành:** tỷ lệ `answered`/`insufficient_evidence`/`unable_to_verify`/`error`, tỷ lệ dùng repair, số câu HyDE `null`, số câu retrieval lỗi/fallback, `prompt_version`. **Lát theo `synthesizer_name` chỉ có 2 loại** (single-hop, multi-hop specific; không có abstract). **Ghi chú diễn giải cố định:** điểm = mức khớp với `reference` do LLM sinh (10.9); judge cùng họ model với generator; single-hop là chỉ số chính, multi-hop (n = 15) báo riêng và chỉ đọc như xu hướng (10.10).

### 11.7 Module

Thêm vào `evaluation/` (chỉ `scoring.py` import `ragas`): `run_models.py` (`case_id()`, các record Pydantic, `EvalConfig`), `jsonl_store.py` (đọc/ghi nối JSONL có validate, bỏ dòng cuối hỏng, tập `case_id` đã xong, một nơi ghi), `key_pool.py` (9 bộ settings/instance, hàng đợi chung + worker, wrapper throttle cho `AnswerGenerator`/`EvidenceJudge`), `hyde_stage.py`/`embed_stage.py`/`retrieve_stage.py`/`generate_stage.py` (chỉ điều phối), `scoring.py` (S4/S4b/S6, tái dùng wrapper LLM/`RunConfig` của `ragas_runner.py`), `report.py` (hàm thuần), `tools/run_eval.py` (Typer: `hyde`, `embed`, `retrieve`, `score-recall`, `generate`, `score-answers`, `score-precision`, `report`, `status`; option chung `--testset`, `--output-dir`, `--limit`, `--retry-failed`, `--workers` mặc định 9, `--config mmr_on|mmr_off` bắt buộc ở S5/S6/S4b).
Thay đổi ngoài module mới: `retrieval/` thêm `PrecomputedQuery` + `precomputed`; `RagasEmbeddingsAdapter` thêm `segment`; **`groq_round_robin.py` thêm throttle chủ động theo `(model, key)`** (`TokenWindowThrottle` từ `retrieval/llm_throttle.py`: `acquire` trước request, `settle` bằng token thật; đổi mô tả "không phải rate-limiter" ở mục 3.1 — sửa code eval Phase 1, không phải code production); `testset_generator.finalize_testset` đổi sang đọc review (mục 4.2); `.gitignore` thêm `data/eval/phase2/embeddings.jsonl`. Chạy trong venv `eval`; group này có thêm Langfuse hiện có để import được production generation/HyDE dù loại group `production`.

### 11.8 Ước lượng chi phí (thô, **chưa đo**)

157 câu: S1 157 lượt 20b (~0,1M token); S4 ≤314 lượt 120b × ~1,5–2K ≈ ≤0,5M; S5 draft 157–314 × ~3–4K ≈ 0,5–1,1M (120b) + Judge 157–314 lượt 20b; S6 ≤785 lượt ≈ 0,8M; S4b 785 lượt × ~0,7–1K ≈ 0,55–0,8M. **Tổng 120b ≈ 2,4–3,2M token** so với ~1,8M/ngày (9 key × 200K TPD) → **tối thiểu 2 ngày lịch**; thời gian chạy thật cỡ 1,5–2 giờ, nút thắt là TPD không phải tốc độ. Lịch dự kiến: ngày 1 S1–S4, chọn MMR, bắt đầu S5; ngày 2 xong S5, S6, S4b (TPD là cửa sổ trượt nên quota hồi dần; resume theo `case_id`).

### 11.9 Rủi ro / điểm mở

1. **Venv `eval` dùng `openai` cũ hơn production:** S1/S5 chạy code production (`AsyncGroq`, `with_structured_output(method="json_mode")`) trong venv này; chưa kiểm luồng generation/Judge — **lần chạy S5 đầu dùng `--limit 2`**; lệch thì phải tách venv chạy S1–S3, S5. 2. **Điểm lạc quan hoá:** Judge lọc trước (faithfulness), judge RAGAS cùng họ với generator, `reference` chưa đối chiếu với luật — đọc điểm như xu hướng. 3. **`n = 157` nhỏ** (multi-hop chỉ 15): chênh dưới nhiễu judge không kết luận được → báo số câu thắng/thua.
5. **Ngưỡng và tần suất:** chưa đặt ngưỡng "đạt"; lần chạy đầu là mốc cơ sở; sau đó mới quyết ngưỡng, tần suất chạy lại. 6. **Rerank GPU 2GB sát giới hạn:** OOM lẻ tẻ ở S3 (11.3); câu OOM bị bỏ khỏi so sánh MMR nếu chưa chạy lại — báo số bị loại. 7. Không đo guardrail.
8. **Giả định bucket:** 9 key = 9 tổ chức khác nhau; Groq tính rate limit theo tổ chức và liệt kê bảng giới hạn riêng từng model, tài liệu **không nói thẳng** 20b/120b độc lập — "độc lập" là quan sát của dự án. Throttle theo `(model, key)` từng bucket nên không phụ thuộc giả định này ngoài chuyện hai model có chia sẻ hạn mức hay không.
9. **Sửa router Phase 1 để gắn throttle** là thay đổi code eval đã chạy ổn: cần test lại round-robin/cooldown/breaker, chạy cục bộ trong venv `eval`. 10. **Tối ưu S4 (chốt):** làm — câu hai cấu hình cùng danh sách `chunk_id` thì chấm một lần (11.3); **không** chuyển S4 sang 20b.
11. Tỷ lệ `error` cao (> ~10–15%) nghĩa là retry đốt thêm TPD — kiểm lại throttle trước khi chạy lại hàng loạt. 12. Kiểm regex strip `[n]` ở S6 trên dữ liệu thật.

### 11.10 Không pilot bắt buộc (người dùng chốt 2026-10-01)

Các số trong 11.8 (token/lượt, tỷ lệ repair, số ngày) và các điểm cần kiểm ở 11.9 (venv `eval` chạy S1/S5, `answer_relevancy` với embedding segment cho điểm hợp lý, OOM `batch_size` rerank) chỉ biết khi chạy thật; `--limit` vẫn có trong CLI. Khuyến nghị tối thiểu (không bắt buộc): chạy S5 với `--limit 2` trước vì chạy code production trong venv `eval` là chỗ rủi ro nhất (11.9.1).

### 11.11 Song song và rate limit

**Nguyên tắc (người dùng):** limit và song song đều **bắt buộc giữ**; chỉ **bỏ qua và đánh dấu** các lỗi không đoán trước được. Cổng cứng "1 request/phút/key" từng được chốt cho S5 rồi **thay bằng throttle theo token** (lượt gọi 0,5K–4K token nên luật cứng vừa chặt quá vừa không bảo vệ khi một lượt nặng).
**Throttle chủ động:** `TokenWindowThrottle` (`retrieval/llm_throttle.py`, `conversation_spec.md` mục 12.1): cửa sổ trượt 60s theo TPM (8K × 0,9) và RPM (30 × 0,9) cho mỗi `(model, key)` (`get_throttle(model, api_key)`); `acquire(estimated_tokens, max_wait_seconds)` chờ **trước khi gửi** nếu cửa sổ sắp đầy, `settle(reservation, actual_tokens)` cập nhật bằng usage thật; "chủ động" = không gửi khi hết chỗ, khác router chỉ phản ứng sau 429. Eval đặt `max_wait` đủ rộng (vài phút).

| Stage | Cơ chế song song | Throttle |
| --- | --- | --- |
| S1 | 9 worker + hàng đợi chung | `HydeGenerator` đã tự throttle theo bucket 20b |
| S2 | gom lô 25 text/request | — (HF) |
| S3 | tuần tự (GPU 2GB), cả hai cấu hình | — |
| S4, S4b, S6 | RAGAS `max_workers` + `GroqRoundRobinChatModel` 9 key | throttle theo `(model, key)` gắn vào router |
| S5 | 9 worker + hàng đợi chung, mỗi key tuần tự từng câu | wrapper `AnswerGenerator` (draft/repair) + throttle sẵn có của `EvidenceJudge` |

**Phân loại lỗi:** (a) 429 theo ngày → circuit breaker, dừng, phần chưa làm "chưa xử lý", resume hôm sau; (b) lỗi **không đoán trước** (parse/NaN, timeout, connection, 5xx, 429 phút lẻ tẻ dù đã throttle) → ghi `error` + đánh dấu, đi tiếp, `status` rồi `--retry-failed` sau; (c) 400/413 tất định không retry mù.
**Chồng stage:** chỉ **S3 (GPU) chồng S4 (Groq)** được. S4, S4b, S5, S6 cùng bucket 120b → không chạy chồng. Throttle chỉ dùng chung được trong **một process** (`get_throttle` là `functools.cache` in-process), nên hai lệnh CLI chạy cùng lúc không thấy ngân sách của nhau và sẽ tranh TPM thật → 429.

### 11.12 Kế hoạch implement và mặc định đã chốt

**Nhánh `/develop-cycle` theo yêu cầu người dùng: `feat/gen-testset-ans`** — toàn bộ mục 11.7, kèm sửa nhỏ `finalize_testset` theo mục 4.2 (đọc review, giữ mẫu `keep`); `golden_testset.json` đã sinh sẵn nên không chặn Phase 2.
**Mặc định:** `--workers` mặc định 9 cho stage RAGAS (throttle mới là thứ chặn tốc độ); `--config` bắt buộc ở S5, S6, S4b; `max_wait` của throttle đặt rộng; hằng ước lượng token draft/repair chốt khi implement.
**Kiểm chứng cục bộ (2026-10-01, không gọi dịch vụ thật):** `evaluate(raise_exceptions=False)` trong ragas 0.4.3 trả NaN khi một metric lỗi và không crash; lớp checkpoint chuyển NaN thành `error`. Import và chạy được `GenerationPipeline`/`HydeGenerator` trong venv `eval` với fake; smoke kiểm resume/JSONL, reuse recall, quota ngày, phân biệt refusal/error, strip citation và throttle acquire/settle; `finalize` trên raw/review thật trả đúng 157 mẫu. Chạy thật S5 lần đầu vẫn theo mục 11.9.1.

**Giao developer kiểm chứng khi làm:** `raise_exceptions=False` của `evaluate()` trong ragas 0.4.3 có trả NaN dùng được không (NaN ghi `error`); venv `eval` chạy được `GenerationPipeline`/`HydeGenerator`; regex strip `[n]` trên dữ liệu thật. `CLAUDE.md`/`AGENTS.md` cập nhật cùng commit với spec.
