# Evaluation — RAGAS: sinh golden testset (Phase 1) và chấm hệ thống (Phase 2): Reference Spec

> Cô đọng 2026-09-30 từ bản 1.396 dòng (lịch sử pilot và số đo cũ: git history). Giữ số mục vì code/spec khác tham chiếu (3.1, 3.2, 3.3, 4.4, 4.5, 8 nhất là).
> **Trạng thái:** Phase 1 đã implement (PR #55, #60, #63, #64: điều phối 9 key, đếm token, giữ phần đã sinh); sinh testset **chạy dở: 33/50 đơn vị, 97 câu** (mục 4.6). Chính sách retry/pass tiết kiệm token (mục 3.4) đã implement trên branch `feature/eval-ragas-retry-pass`.
> Phase 2 (mục 11) **đã thiết kế, chưa implement**.

## 0. Tinh hoa: bài học xương máu và phương pháp đúng

**Sai lầm đã trả giá → cách làm đúng:**
1. **Chạy full mà chưa pilot.** 4 pilot nhỏ (2026-09-28) mới lộ: prompt ragas mặc định tiếng Anh nên câu hỏi lẫn ngữ; 3/4 `QueryStyle` là nhiễu (sai chính tả, rò rỉ tên persona); request 12K token bị **413** vì Groq TPM 8K/tài khoản
   (mặc định ragas 32K token/lượt); `ragas` import `rapidfuzz` mà không khai báo dependency. → **Luôn pilot bằng CLI thật trên văn bản nhỏ nhất trước khi tốn quota nhiều ngày**; bắt buộc `adapt_prompts("vietnamese")` + ép `PERFECT_GRAMMAR`,
   hạ `max_token_limit` extractor xuống ~4.000.
2. **Retry mặc định của ragas (10 lần với mọi `Exception`) đốt hàng trăm request** khi hết quota và thử lại lỗi tất định (400/401/413) vô ích. → `RunConfig` riêng: `max_retries=3`, `max_wait=30`, chỉ `RateLimitError`/`APIConnectionError`/`InternalServerError`.
   ragas chỉ đọc retry từ `llm.run_config` (không từ `run_config` của `apply_transforms`) nên phải gán vào wrapper ngay lúc khởi tạo.
3. **Router round-robin coi "trạng thái dùng chung thread-safe" là "rủi ro chấp nhận được" — sai.** Cờ circuit breaker khoá cả process, một lượt bật oan (do luồng xen kẽ) dừng cả job nhiều giờ. → mọi trạng thái dưới `threading.Lock`,
   chốt điểm bắt đầu một lần mỗi lượt gọi, không giữ lock khi gọi mạng/ngủ.
4. **Ước lượng sai đơn vị:** corpus "1,04M" là số **byte**, thật là 781.007 **ký tự**. → đo bằng ký tự; ghi token thật từng đơn vị (mục 3.2 B) thay vì tin hệ số ước lượng (±30%).
5. **Trạng thái "đã xong" suy từ file raw sai** (người dùng xoá dòng xấu khi đọc lướt → đơn vị bị chạy lại, tốn token). → `generation_progress.json` là nguồn sự thật; ghi raw TRƯỚC, progress SAU.
6. **Đặt `raise_exceptions=False` của ragas không dùng được** (0.4.3 trả `NaN` rồi crash khi dựng sample) → tự viết vòng sinh với API công khai của synthesizer (mục 3.3).
7. **Test phụ thuộc ragas bị CI bỏ qua** (CI không có nhóm `eval`) nên xanh giả. → chạy cục bộ trong venv eval (`~/.cache/eval-venv-run/bin/python -m pytest tests/test_evaluation*.py`) trước khi tin.
8. **Đừng dùng `exc_info=True`/`logger.exception`** trong log lỗi: chúng in cả thông điệp exception, có thể chứa nội dung câu hỏi/context (vi phạm chính sách log). → `_describe_traceback`: chỉ tên loại lỗi + `file:dòng:hàm`.
9. **Tự nhận là xong khi chưa có số đo:** multi-hop abstract = 0 ở 33 đơn vị mà không ai để ý cho tới khi tổng kết (mục 4.6). → kiểm phân bố loại câu theo từng đơn vị ngay từ pilot, không chỉ tổng số.

**Phương pháp đúng (giữ):** sinh từ **văn bản nguyên bản** chứ không từ chunk của hệ thống (giữ testset độc lập với thiết kế cần đo, mục 2.1); đơn vị = Chương/Mục, mỗi đơn vị một KG riêng, checkpoint theo đơn vị, chạy nhỏ → lớn, resume nhiều ngày;
dừng hẳn khi hết quota ngày thay vì thử đơn vị kế; chấm bằng LLM so với `reference` (không phụ thuộc ranh giới chunk); báo điểm tách theo loại câu và theo văn bản; đọc điểm như xu hướng, không phải chuẩn tuyệt đối (mục 10.9).

## 1. Mục tiêu & phạm vi

Sinh tự động bộ **golden testset** (câu hỏi + đáp án chuẩn + context chuẩn) từ corpus pháp luật thật (`data/markdown/`) bằng `ragas.testset.TestsetGenerator`, thay cho việc tự đọc luật và viết tay từng cặp. Phase 1 = sinh testset; Phase 2 = chạy
hệ thống thật trên testset và chấm điểm (mục 11). **Không dùng dữ liệu `chatlog`/`chat_turns` ở phase này** (hệ thống mới nghiệm thu, dữ liệu thật chưa đủ; nay nhật ký nằm ở Langfuse, có thể lấy mẫu Q&A thật ở phase sau).

**Làm:** cắt 6 văn bản `.md` thành **Chương** và xử lý **từng Chương một** (mỗi Chương một `KnowledgeGraph` riêng); build KG và sinh câu; tổng **`GENERATE_SIZE = 240`** câu theo tỷ lệ **80/10/10** (192 single-hop / 24 multi-hop abstract / 24 multi-hop specific),
chia theo kích thước Chương (mục 4) — **sinh dư so với đích 180 để trừ hao** khi người dùng đọc lướt và xoá câu xấu; luân phiên **9 tài khoản Groq** cho `generator_llm` (mục 3.1); lưu `golden_testset_raw.json` ngay sau mỗi đơn vị, `finalize` chốt đúng `TARGET_SIZE = 180`
ra `golden_testset.json` (mục 4.2); lưu KG từng Chương để sinh bù không build lại; CLI Typer mỏng (`generate` với `--only`, `--testset-size`, `--reuse-knowledge-graph`, `--append`, `--dry-run`; `finalize`).

**Mục tiêu bộ eval (2026-09-28):** kiểm tra TOÀN HỆ THỐNG (HyDE → hybrid retrieval → RRF/MMR → rerank → generation → Evidence Judge; Phase 2 bỏ guardrail/condense/cache/api, mục 11) với testset **ĐỘC LẬP thiết kế hệ thống**: câu hỏi/đáp án sinh từ văn bản luật gốc, không qua `chunking/`,
nên input là văn bản nguyên bản (`generate_with_langchain_docs`), KHÔNG phải chunk (`generate_with_chunks`) — mục 2.1.

**Mỗi Chương một đồ thị (người dùng, 2026-09-28):** hệ thống RAG hiện **không xử lý được câu ghép chéo giữa các luật** và xử lý kém câu ghép chéo Điều (chunk cắt theo Khoản), nên chỉ cần quan hệ trong phạm vi hẹp; mất quan hệ chéo Chương là chấp nhận được. Đồng thời là điều kiện để chạy trên quota Groq free:
mỗi Chương vài chục KB vừa quota một ngày, lỗi chỉ mất một Chương. Tỷ lệ **80/10/10**: multi-hop (ghép nhiều Điều trong cùng Chương) là loại hệ thống đã biết yếu nên chỉ 20%, để điểm gộp không bị kéo xuống; giữ `synthesizer_name` để Phase 2 báo điểm riêng theo loại.

**Không làm (Phase 1):** chạy `retrieve()`/`generate()` thật hay tính metric RAGAS (cần `response`/`retrieved_contexts`, chỉ có ở Phase 2); dùng `chat_turns`; tự động lọc câu vô nghĩa bằng code (review là việc đọc bằng mắt của người vận hành); CI/cron; Postgres.

**Tiêu chí hoàn thành Phase 1:** `data/eval/golden_testset.json` chứa **đúng 180** câu hợp lệ (đủ `user_input`/`reference`/`reference_contexts`, không rỗng), đã đọc lướt bỏ câu vô nghĩa, dùng thẳng làm input Phase 2.

**Về 240 sinh / 180 đích (2026-09-28):** dựng KG là chi phí cố định theo Chương (đã cache); chỉ sinh câu tỉ lệ theo số câu. Nút thắt là **TPD Groq free 200K/tài khoản** (không phải TPM) và **công review tay**. Đích 180 (đủ ý nghĩa thống kê thô, nhường thời gian cho deploy + observability); đọc lướt dự kiến loại ~25–40% nên sinh 240 (dư ~33%);
thiếu thì sinh bù bằng `--only <văn bản> --reuse-knowledge-graph --append` (mục 4.2).

## 2. Input & Output

**Input:** `data/markdown/*.md` — đúng 6 văn bản pháp luật hoàn chỉnh. **Output** (`data/eval/`): `golden_testset_raw.json` (~240 `GoldenTestCase` chưa review, nối thêm sau mỗi đơn vị; người dùng đọc lướt, xoá câu xấu ngay trong file); `golden_testset.json` (**đúng 180** sau `finalize`, file duy nhất Phase 2 đọc;
mỗi item: `user_input`, `reference`, `reference_contexts`, `synthesizer_name`, `source_document`, `source_section`); `generation_progress.json` (nguồn sự thật "đã làm tới đâu", mục 4.5); `knowledge_graph/<văn bản>__<số>.json` (KG từng đơn vị, dùng lại khi sinh bù); `units/`, `units_plan.md` (kế hoạch chia đơn vị, mục 4.4).
`golden_testset.json` giữ đúng 3 cột bắt buộc của `EvaluationDataset` ragas (`user_input`, `reference`, `reference_contexts`) để Phase 2 dùng thẳng; `synthesizer_name` là cột phụ để biết loại câu.

**Ai tạo cột nào:** Phase 1 tạo `user_input`, `reference` (đáp án bám luật), `reference_contexts` (đoạn luật ragas đã đọc, cột phụ truy vết), `synthesizer_name`; Phase 2 tạo `response` (câu trả lời thật của chatbot, đầu vào là `user_input`) và `retrieved_contexts` (chunk thật hệ thống tìm về). Ở Phase 2 chỉ `user_input` vào chatbot, `reference` là thứ đem so.

### 2.1 Vì sao đưa văn bản nguyên bản, không dùng chunk (2026-09-28)

`generate_with_chunks` đã cân nhắc và **không dùng**: (1) sinh câu hỏi từ chính chunk của `chunking/` thì ground truth do chunker định nghĩa — chunker cắt sai (một ý bị chia đôi) thì testset không bao giờ hỏi trúng chỗ lỗi, và câu hỏi sinh từ đúng văn bản chunk khiến BM25/dense bắt trúng dễ hơn thực tế (lạc quan giả);
(2) **corpus chunk không hợp làm input RAGAS** (đo trên `data/chunks/`: 2.228 chunk, trung vị 41 token, 58% dưới 50 token; `breadcrumb` chỉ ở metadata nên `content` không nói thuộc luật/Điều nào → summary/themes/NER mỏng, câu hỏi mơ hồ giữa 6 luật, chi phí LLM cao). **Đánh đổi chấp nhận (kiểm trên `ragas==0.4.3`):** `reference_contexts` là đoạn ragas tự cắt
(`HeadlineSplitter(min_tokens=500)`), một đoạn bao trùm nhiều chunk hệ thống (nhiều-1). Không ảnh hưởng `context_recall`/`context_precision` (bản LLM chỉ cần `user_input`, `retrieved_contexts`, `reference` — so với `reference`, không với `reference_contexts`); có ảnh hưởng nếu muốn metric xác định (`NonLLM...`, hit@k theo `chunk_id`):
map nhiều-1 khả thi (94% chunk là trích nguyên văn) nhưng recall@k thấp giả, chỉ hit@k dùng được — để dành mục 10.8.

## 3. Công cụ

`ragas.testset.TestsetGenerator` (`ragas==0.4.3`, ghim; API xác nhận bằng đọc source cài thật: `TestsetGenerator(llm, embedding_model)`, `generate_with_langchain_docs`, `default_transforms`, `adapt_prompts(language, llm)` + `set_prompts(**...)`); `generator_llm` = `ChatOpenAI` trỏ Groq (`generation_spec.md` mục 8) bọc `LangchainLLMWrapper`;
embedding cho KG = adapter mới (`embeddings_adapter.py`) quanh `InferenceClient.feature_extraction` của `embedding/hf_client.py` (interface LangChain `Embeddings`), **không word-segment** (embedding chỉ để so tương đồng tóm tắt trong KG, không tìm kiếm chéo với index production; Phase 2 khác, mục 11.5); `langchain_core.documents.Document`; Typer (`tools/generate_testset.py`).

**Dependency (`[dependency-groups] eval`, không phải `[project] dependencies`):** `ragas>=0.4.3`, `langchain-community<0.4`, **`rapidfuzz`** (ragas import mà không khai báo). `production = ["openai>=3.19.0"]` tồn tại chỉ để `uv` tách resolve `openai` (ragas kéo `instructor` ép `jiter<0.15` nên hạ `openai`), có `[tool.uv.conflicts]` giữa `production` và `eval` nên **không sync cả hai cùng lúc**.
Mọi lệnh `uv run` liên quan `eval` phải mang `--group eval --no-group production` (`uv run` không cờ tự re-sync về `default-groups` và kéo `openai` về bản production mà không cảnh báo): `uv sync --group eval --no-group production`; xong việc chạy `uv sync` không cờ để trả venv mặc định.
`LangchainLLMWrapper`/`LangchainEmbeddingsWrapper` deprecated ở 0.4.3 (vẫn chạy) — giữ version ghim. Rủi ro: `HuggingFaceEmbedder` chỉ có `embed_chunks`, không có `embed_documents`/`embed_query` nên cần adapter gọi thẳng `feature_extraction` trên text thô.

### 3.1 Round-robin 9 tài khoản Groq (`groq_round_robin.py`)

**Credentials:** 9 (`GROQ_API_KEY_1`…`_9`; `_5`–`_9` không thuộc production). Router coi chúng là credential được người vận hành cấp quyền cho workload này, **không suy ra hay cố vượt quota bằng số key**. Chỉ dùng capacity/rate limit mà Groq cho phép; nếu các key chia sẻ một tổ chức/project thì quota có thể dùng chung và rotation không làm tăng capacity. `TOKENS_PER_DAY = 1_800_000` chỉ là ước lượng vận hành khi dashboard xác nhận 9 bucket độc lập; không phải điều kiện đúng của thuật toán. Lịch sử: 3 → 4 → 6 (2026-09-28) → 9 (2026-09-29); đổi cứng `GROQ_API_KEY` → `GROQ_API_KEY_1` (không alias). **Cam kết vận hành:** không chạy workload khác trên credential này lúc sinh testset (key 3, 4 vốn phục vụ generation production) → không làm cơ chế nhường-chỗ trong code.

**Giới hạn Groq đo được (free, theo `(tài khoản, model)`):** `gpt-oss-120b` và `20b` đều RPM 30, RPD 1K, **TPM 8K, TPD 200K**. Ràng buộc là **TPD**. Request > 8K token bị **413** (không phải 429 nên không retry; round-robin không làm request đơn lẻ nhỏ đi) — mỗi lượt gọi phải ≲ 4–5K token input. Header `x-ratelimit-*` chỉ có RPD và TPM (**không có TPD còn lại**); `retry-after`
chỉ có trên 429; docs không nói 429 bị từ chối có tính vào RPD hay không → không dựng được sổ TPD từ header, và tránh bắn 429 vô ích. Khối lượng: dựng KG ≈ 5 token/ký tự, sinh câu ≈ 1,5K token/câu → cả corpus + 240 câu ≈ **4,2M token** (3–5,5M); sau chia đơn vị thật: **50 đơn vị, 720.573 ký tự, ~4,16M token**, không chạy nổi trong 1 ngày.

**Pattern MỚI trong repo:** các chỗ dùng nhiều key khác (`GenerationSettings`/`JudgeSettings`, `formatting/llm_client.py`) tách ngân sách theo bước cố định (mỗi bước luôn một key); ở đây **luân phiên nhiều key cho CÙNG một luồng gọi** để rải tải, nên không tái dùng `_SlidingWindowRateLimiter`/`convert_chunks_concurrently` (thiết kế cho 2 worker thread). Không dùng `LoopBoundClient` (script chạy tuần tự
đúng 1 process). Không phải tiền lệ để áp lại ở nơi khác.

**Thiết kế `GroqRoundRobinChatModel(BaseChatModel)`** (proxy N `ChatOpenAI`; chỉ implement `_generate` sync — `_agenerate` mặc định chạy nó trong executor nên vẫn đúng khi ragas gọi async; **không phải rate-limiter**, mỗi client giữ timeout/retry riêng từ `TestsetGeneratorSettings`):
- **Thứ tự thử:** `_plan_attempts` (dưới lock) lấy `start = next(cycle)` đúng **một lần**/lượt gọi rồi duyệt `(start + offset) % n` — luôn n client KHÁC NHAU, không phụ thuộc luồng khác. Client đang cooldown xếp CUỐI (sắp xếp ổn định) chứ không bị loại. Lỗi khác 429 (400/401/403/413) bay thẳng ra, không thử tài khoản khác. Hết vòng vẫn lỗi → raise nguyên lỗi cuối, không vòng lặp vô hạn.
- **Circuit breaker:** chỉ bật khi **cả n client khác nhau** đều 429 hết quota **THEO NGÀY** ("per day"/"(TPD)"/"(RPD)") trong CÙNG một lượt gọi (cooldown không thay thế bằng chứng mới; 429 theo phút hay bất kỳ client nào thành công đều không bật). Khi bật: ghi nhớ thông điệp, từ chối mọi lượt gọi sau ngay (`DailyQuotaExhaustedError`, không request nào; mỗi lần raise là instance
  MỚI, chỉ lần đầu giữ `__cause__`). Lỗi này **không kế thừa `RateLimitError`** nên ragas không retry. Lý do: ragas không huỷ task còn lại khi một task lỗi, không chặn thì mỗi task đốt hàng chục request vô ích.
- **Cooldown ngày 5 phút** (`_DAILY_COOLDOWN_SECONDS = 300`): client vừa báo TPD được ưu tiên tài khoản khác (TPD là cửa sổ trượt nên hồi dần); thành công thì xoá cooldown.
- **Thread-safe (bắt buộc):** con trỏ vòng, `call_counts`, cooldown, cờ breaker, token, `reasoning_effort` chỉ đọc/ghi dưới một `threading.Lock`; lock KHÔNG giữ khi gọi mạng hay ngủ.

### 3.2 Điều phối key mượt và tiết kiệm token (**đã implement**, PR #64, 2026-09-29)

Mục tiêu (người dùng): key phối hợp mượt, tiêu token ít nhất. **Không đổi thuật toán chọn key.** Không làm: chọn key theo "còn nhiều quota nhất" (không đọc được TPD); sổ TPD nhiều ngày (không có nguồn sự thật).
- **A. Cooldown ngắn khi 429 theo phút.** Trước đây 429 theo phút để tài khoản đó vẫn đầu vòng nên lượt kế lại đập vào đúng nó. Nay: đọc `retry-after` (giây) từ `error.response.headers`; thiếu/hỏng (kể cả `inf`/`nan`) → `_MINUTE_COOLDOWN_DEFAULT = 15`; kẹp `[1, _MINUTE_COOLDOWN_MAX = 60]`; đặt `_cooldown_until` = `max(hiện có, now + giây)` (không rút ngắn cooldown ngày).
  **Nếu cả n tài khoản đều cooldown phút** (không có bằng chứng ngày): chờ tới lúc sớm nhất hết (≤60s, không giữ lock) TRƯỚC khi thử, thay vì bắn loạt 429; không chờ khi có cooldown ngày hay breaker đã bật. Điều kiện breaker không đổi. Clock và `sleep` tiêm được (`GroqRoundRobinChatModel(clients=..., clock=..., sleep=...)`) để test không chờ thật.
- **B. Đếm token thật.** Router cộng dồn từ kết quả thành công `prompt_tokens`, `completion_tokens`, `reasoning_tokens` (`completion_tokens_details`, thiếu → 0) theo client, dưới lock (`token_totals`); nguồn `ChatResult.llm_output["token_usage"]`; thiếu `usage` coi là 0 và log cảnh báo một lần. `UnitResult` có `tokens`/`reasoning_tokens` (hiệu số như `llm_calls`); `UnitProgress` ghi thêm (cộng dồn khi `--append`,
  mặc định `None` nên đọc được 33 đơn vị cũ; bản ghi cũ `tokens=None` thì tổng sau cộng dồn là `None` để không lệch hệ số). Log INFO cuối đơn vị (cả khi lỗi, trong `finally`): tổng và theo tài khoản (chỉ số, không key). **`--dry-run`:** khi ≥ 3 đơn vị `done` có `tokens > 0`, ước lượng còn lại bằng hệ số đo được
  `(Σtokens − 4000·n)/Σchars` (giữ dạng "hệ số·chars + 4K"), dòng tóm tắt kết thúc ` (hệ số đo được X.XX token/ký tự)`; ít hơn thì giữ 5,5. **Hạn chế biết trước:** đơn vị dùng lại KG chỉ có token phần sinh câu (hệ số thấp hơn thật); `--append` cộng token mới nhưng `chars` giữ nguyên (hệ số bị thổi lên).
- **C. `reasoning_effort="low"` CHỈ khi dựng KG** (người dùng: sinh câu giữ nguyên để chất lượng testset không đổi so với 33 đơn vị đã sinh). Token suy luận của `gpt-oss-120b` tính vào TPM/TPD; KG (summary/themes/NER/headlines) là trích xuất đơn giản. `with router.reasoning_effort("low"):` (context manager; đọc qua `router.current_reasoning_effort`, mặc định `None` = không gửi) truyền kwarg
  `reasoning_effort=...` xuống `client._generate(...)` (không dùng `extra_body`: `openai` hỗ trợ tham số chính thức, `ChatOpenAI` giữ ở body chat-completions; kwarg của người gọi thắng giá trị của router; xác nhận bằng `httpx.MockTransport`). `_build_knowledge_graph` (kể cả `apply_transforms`) chạy trong context; `adapt_prompts`, persona, scenario, sample thì KHÔNG; gỡ trong
  `finally`. Hằng `_KG_REASONING_EFFORT = "low"` trong `ragas_runner.py`. KG đã dựng trước đó không hưởng lợi. **Chưa xác nhận với Groq thật** — cần pilot nhỏ: Groq nhận tham số và `reasoning_tokens` giảm khi dựng KG.

### 3.3 Giữ phần đã sinh khi lỗi giữa đơn vị (**đã implement**, PR #64)

**Vấn đề:** `TestsetGenerator.generate` chạy mọi câu rồi trả một lần, `raise_exceptions=True` mặc định nên một câu lỗi (429 ngày hay parse) huỷ cả lô; `raise_exceptions=False` **không dùng được** (0.4.3 trả `NaN` cho job lỗi rồi crash khi dựng `TestsetSample`). Quy mô lãng phí mỗi lần dừng ≈ phần sinh câu của một đơn vị (~1–2% quota ngày, ước tính chưa đo); lợi ích thật: **một câu lỗi tất định không còn hỏng cả đơn vị và chặn cả chương trình.**

**Thiết kế (thay `generate` trong `RagasUnitRunner._generate_cases`):** không gọi `generate`; tự làm đúng các bước bằng API công khai: `generate_personas_from_kg(kg, llm, num_personas=3)` một lần/đơn vị → mỗi loại có quota > 0 (tuần tự từng loại): `generate_scenarios(n, knowledge_graph, persona_list)` (số scenario = `calculate_split_values` của ragas, đúng quota nhờ trọng số `(n−0,5)/tổng` trong `_query_distribution`)
→ `generate_sample(scenario)` từng cái bọc `try/except`, đồng thời tối đa `MAX_WORKERS` (`asyncio.Semaphore`). Test khoá chữ ký các hàm ragas (`importorskip("ragas")`) để nâng phiên bản không âm thầm phá.
- **Sample lỗi:** `DailyQuotaExhaustedError` → ngừng sample mới, giữ mọi sample đã xong, đơn vị **`partial`**, chương trình dừng như cũ (mã thoát 1); lỗi khác (parse, timeout hết retry, sample thiếu cột bắt buộc) → BỎ sample đó, log CHỈ tên synthesizer + tên loại lỗi, đếm `skipped_samples`, đơn vị vẫn `done` với ít câu hơn quota; không sinh được câu nào mà có lỗi → `UnitGenerationError`
  (chain lỗi gốc, ưu tiên lỗi quota; không ghi `done` rỗng); lỗi sinh scenario của một loại khi chưa có câu → lỗi đơn vị, nhưng các loại đã xong trước đó được giữ.
- **Ghi:** raw TRƯỚC, progress SAU (kể cả `partial`); runner trả `UnitResult.interruption` (không serialize) khi đã có câu; orchestrator nối raw, ghi progress `partial` + `last_failure` trong một lần lưu rồi raise `UnitGenerationError` (mã thoát 1). `UnitProgress` thêm `status: Literal["done","partial"] = "done"`, `skipped_samples: int = 0`; `completed_at` của `partial` là **thời điểm dừng**
  (đổi thành lúc xong khi chuyển `done`); `partial` cộng dồn `questions`/`llm_calls`/`seconds` như `--append`.
- **Chạy tiếp `partial`:** nằm trong danh sách chờ, dùng lại KG, quota còn lại = `quota − questions` (kẹp ≥ 0), rồi chuyển `done`; loại bị `_has_clusters` bỏ được kiểm lại và bỏ lại. `--dry-run` hiển thị `dở (đã có N/M câu)`. **`--append` trên đơn vị `done` bị ngắt giữa chừng GIỮ `done`** (chuyển `partial` sẽ làm `quota − questions` về 0 và đơn vị kẹt); vẫn ghi raw + `last_failure`, mã thoát 1.
- **Phạm vi còn đúng của quy tắc cũ ("đơn vị dở không ghi raw", "không resume trong lòng đơn vị"):** chỉ khi lỗi xảy ra TRƯỚC khi có sample nào xong (dựng KG, sinh scenario). Rủi ro biết trước (thấp): chết giữa "nối raw" và "ghi progress" của một đơn vị `partial` → `_recover_unfinished` coi `done` ít câu (bù bằng `--append`); `partial` đang chạy tiếp mà chết giữa hai bước có thể sinh thừa câu (không hỏng dữ liệu);
  `partial` có `quota − questions = 0` ở mọi loại (chỉ khi dùng `--testset-size` nhỏ) bị loại khỏi `to_run` nhưng vẫn hiển thị "dở"; SIGTERM/`pkill` giữa lúc sinh mất phần sinh câu của đơn vị đó (mẫu chỉ nằm trong bộ nhớ tới hết bước sinh).

### 3.4 Retry/pass tiết kiệm token (**đã chốt, chưa implement**)

**Mục tiêu:** một lỗi RAGAS hoặc mạng không được làm job dừng và lãng phí toàn bộ phần đã chạy; đồng thời không dùng retry mù làm nhân số request/token. `DailyQuotaExhaustedError` là ngoại lệ duy nhất: dừng ngay toàn process, giữ checkpoint, không thử unit/key mới. Không dùng cơ chế này để vượt quota/rate-limit của Groq.

**Một chủ sở hữu retry:** `GroqRoundRobinChatModel` là tầng DUY NHẤT retry request HTTP. `ChatOpenAI(max_retries=0)` và `RunConfig(max_retries=0)` để không nhân retry SDK × RAGAS × router. Router quản lý trạng thái từng credential dưới lock: `ready`, `minute_cooldown`, `daily_cooldown`, `disabled`. Mỗi request round-robin credential sẵn sàng trước; thành công xoá cooldown và cộng token thật. Nếu mọi credential đang cooldown phút, router chờ credential sớm nhất theo `retry-after` (kẹp 60 giây) trước vòng thử tiếp theo.

- **429 phút:** đọc `retry-after`, đưa credential vào `minute_cooldown`; thử credential khác trước. Nếu cả vòng chỉ gặp cooldown phút, request trả `RateLimitError`; lượt sau chờ credential sớm nhất hết cooldown thay vì tạo vòng retry SDK/RAGAS.
- **429 ngày:** đánh dấu `daily_cooldown` 5 phút để ưu tiên credential khác nhưng vẫn thu thập bằng chứng mới cho circuit breaker. Chỉ khi toàn bộ credential còn hoạt động báo quota ngày trong cùng lượt thì raise `DailyQuotaExhaustedError`; sau đó process từ chối mọi request mới.
- **Timeout/connection/5xx/498:** router thử lại **tối đa một lần** cho request (hai lần gửi tối đa), exponential backoff có jitter, rồi trả lỗi cho tầng nghiệp vụ. Lỗi sau retry không được thử sang tất cả credential một cách mù; mỗi lần thử có giới hạn chỉ chọn một credential `ready`.
- **400/401/403:** không retry; `401`/`403` vô hiệu hoá credential cho các request kế tiếp trong job, còn `400` là lỗi request nên không đổi key. **413:** không retry hay đổi key vì request quá lớn là lỗi tất định.

**Retry theo giá trị của bước RAGAS:** lỗi cấu trúc sau HTTP 200 (ví dụ `KeyError`/parse/pydantic) có thể đã tiêu token; phải vá tính tất định ở wrapper (như lọc persona mapping), không retry chung cả unit. Nếu vẫn là lỗi chưa biết: `generate_personas` và `generate_scenarios` được chạy lại **một lần**; lần hai lỗi thì bỏ loại câu hiện tại và tiếp tục loại sau. `generate_sample` chạy lại **một lần**; lần hai lỗi thì bỏ sample. Dựng KG không retry nguyên unit sau khi transforms đã chạy: vì có thể đốt lại hàng chục nghìn token, đánh dấu cả unit `skipped` và chạy unit kế. `KeyboardInterrupt`/SIGTERM không bị nuốt.

**Checkpoint và minh bạch:** `UnitProgress.status` mở rộng thành `done | partial | skipped`; `skipped` lưu stage `knowledge_graph`, loại exception đã che, số lần gửi request và thời điểm. `skipped` không phải `done`, không được chạy lại trong `generate` thường; CLI thêm `--retry-skipped` (bắt buộc đi cùng `--only`) để người vận hành chủ động thử lại sau khi sửa code/cấu hình. Với unit `done`, thêm `skipped_question_types` và tiếp tục dùng `skipped_samples`; raw vẫn ghi trước progress. `--dry-run`/summary phải tách `done`, `partial`, `skipped unit`, `skipped type`, `skipped sample`; `finalize` giữ luật 180 câu hiện tại và báo thiếu để người dùng sinh bù, không che sự thiếu hụt.

## 4. Workflow (`testset_generator.py`)

Dựng `clients` (9 `ChatOpenAI`) → `generator_llm = LangchainLLMWrapper(GroqRoundRobinChatModel(...))` (gán `RunConfig` ngay lúc khởi tạo, mục 4.5) → `TestsetGenerator` với `adapt_prompts("vietnamese")` cho MỖI synthesizer (gọi 1 lần rồi dùng lại mọi đơn vị; ép `PERFECT_GRAMMAR`, mục 4.3). Cho từng đơn vị (sắp xếp số ký tự tăng dần, lọc `--only`):
bỏ qua nếu đã `done` trong progress (trừ `--append`) → `Document` → `default_transforms` với `max_token_limit` extractor ~4.000 và `max_workers` ~4 (mặc định ragas 32K token/lượt bị 413) → dựng KG (trong `reasoning_effort=low`, mục 3.2 C) và **lưu KG ngay** (nguyên tử) → sinh câu (mục 3.3/3.4) → gắn `source_document`, `source_section` → nối raw, ghi progress. Lỗi xử lý theo mục 3.4: chỉ hết quota ngày dừng cả chương trình; lỗi không quota bị retry có giới hạn rồi bỏ scope nhỏ nhất có thể và tiếp tục.
Thao tác tay ngoài code: người vận hành đọc lướt/xoá câu xấu trong raw; rồi `finalize`.

**Cắt đơn vị (`unit_splitter.split_document`, thuần):** cắt tại `## ` (Chương), tách theo `### ` (Mục) khi quá lớn, gộp phần nhỏ, bỏ mở đầu và chú thích cuối; `source_section` của đơn vị gộp là các nhãn nối bằng " + ". **Quota câu (`allocate_questions`, thuần):** `GENERATE_SIZE = 240` thành đúng 192/24/24; mỗi loại phân bổ cho đơn vị tỷ lệ theo ký tự bằng phương pháp phần dư lớn nhất;
đơn vị nhận 0 câu multi-hop chỉ chạy single-hop; `query_distribution` truyền trọng số = số câu từng loại / tổng; multi-hop cần ≥2 đoạn liên quan nên có thể sinh ít hơn quota, không bù. `--testset-size` ghi đè tổng. `GENERATE_SIZE`, `TARGET_SIZE`, `MIN_UNIT_CHARS`, `MAX_UNIT_CHARS` là hằng số nội bộ, không phải env var.

### 4.1 Ngôn ngữ và chất lượng câu hỏi/đáp án — kết luận từ 4 pilot (2026-09-28)

**Bắt buộc:** `adapt_prompts("vietnamese", llm=...)` cho mọi synthesizer và **ép `QueryStyle.PERFECT_GRAMMAR`**. Với cấu hình này, Groq `gpt-oss-120b` cho câu hỏi tiếng Việt sạch 12/12 (single-hop; multi-hop chưa kiểm ở pilot). Bằng chứng: không adapt → 2/4 câu hỏi tiếng Anh, lẫn Việt-Anh; ragas mặc định trộn 4 `QueryStyle` (MISSPELLED, POOR_GRAMMAR,
WEB_SEARCH_LIKE chiếm 3/4) kèm rò rỉ **tên persona** vào câu hỏi. Nhiễu có thể thêm lại như một lát đo độ bền riêng ở phase sau, không trộn vào testset chính. `reference_contexts` **nguyên văn** 12/12.

**Khuyết điểm chấp nhận có chủ đích:** đáp án đôi khi **dính chữ** ("cưtrú", "dịchvụ"; 1/4–5/8 tuỳ lần) — đã bác bỏ giả thuyết ký tự Unicode lạ (4/4 báo không có), là thiếu dấu cách thật do `gpt-oss-120b` sinh ra → **không thêm NFKC**; `reference` chỉ LLM chấm nên ít ảnh hưởng. **Chất lượng nội dung:** ~6/8 dùng được; trùng ý trong cùng đoạn ragas (4 câu → 2 chủ đề), chủ đề nông
kiểu "Chính phủ quy định chi tiết" (NER trích thực thể chung chung) — chỉ xử lý bằng đọc lướt; nếu tỷ lệ bỏ thực tế >40% thì tăng `GENERATE_SIZE` hoặc sinh bù.

### 4.3 Mã mẫu đã kiểm chứng bằng pilot

Ba việc (mã ở `ragas_runner.py`): (1) `cap_token_limit(transforms, limit)` duyệt `Parallel`/list, gán `max_token_limit` cho mọi `LLMBasedExtractor` (đổi được 4 extractor); (2) subclass 3 synthesizer (`CleanSingleHop/MultiHopAbstract/MultiHopSpecificSynthesizer`) ghi đè `prepare_combinations` để ép `styles = [QueryStyle.PERFECT_GRAMMAR]`; hai multi-hop còn lọc `persona_item_mapping` về đúng tập persona đã sinh — prompt RAGAS đôi khi trả key lạ, mà RAGAS 0.4.3 ném `KeyError` thay vì bỏ qua;
(3) `asyncio.run(synthesizer.adapt_prompts("vietnamese", llm=...))` + `set_prompts(**adapted)`, gọi một lần cho mỗi synthesizer rồi dùng lại mọi đơn vị. Đã kiểm cho single-hop; multi-hop cần xác nhận `styles` có tác dụng.

### 4.2 Chốt đúng 180 câu (`finalize`) và sinh bù

**`finalize`** (hàm thuần + CLI): đọc raw đã review, chọn đúng 180 câu → `golden_testset.json`. **Cắt phân tầng theo (`source_document`, `synthesizer_name`)**, tất định: mỗi dòng nhận khoá `(thứ hạng trong nhóm theo thứ tự file + 0,5) / kích thước nhóm`; lấy 180 dòng khoá nhỏ nhất (hoà theo thứ tự file) — giữ tỷ lệ theo văn bản và loại câu sau khi người dùng đã xoá, không dồn vào vài văn bản đầu file.
Còn < 180 → dừng với lỗi nêu thiếu bao nhiêu + gợi ý lệnh sinh bù, không ghi file thiếu; không chấm chất lượng, chỉ đếm và cắt. **Sinh bù:** `generate --only <văn bản> --reuse-knowledge-graph --append --testset-size N` nạp lại KG (không build lại), sinh N câu mới, nối vào raw, bỏ câu có `user_input` trùng y hệt; rồi `finalize` lại.

### 4.4 Chia đơn vị và lập kế hoạch token (đo 2026-09-28)

**Số đo:** corpus 781.007 **ký tự** (không phải byte); 6 văn bản 4–17 Chương; `Quy định mức lương tối thiểu.md` không có `## ` (coi cả file là 1 đơn vị); `Điều kiện lao động và quan hệ lao động.md` có **hai** Chương "XI" → khoá đơn vị dùng **số thứ tự**, không dùng số La Mã; BHYT Chương X 48,5K ký tự nhưng 46,8K là khối chú thích `[1]…[114]`.
**Quy tắc (chốt cùng người dùng):** đơn vị chuẩn = Chương; lớn hơn `MAX_UNIT_CHARS = 30.000` thì tách theo `### Mục`; nhỏ hơn `MIN_UNIT_CHARS = 6.000` gộp với kề (fallback nếu vẫn lớn: tách theo `#### Điều` rồi theo đoạn; chưa gặp); **KHÔNG bỏ Chương "Điều khoản thi hành"** (nhiều câu có thể vô nghĩa, bỏ khi đọc lướt); **bỏ khối chú thích cuối file**
(`_strip_footnotes`: cắt từ `---` đứng ngay trước `[1] …` cuối file — chỉ dẫn văn bản sửa đổi, sinh câu từ đó ra rác, tiết kiệm ~300K token: 4,47M → 4,16M) và phần mở đầu trước Chương đầu. Ước lượng chi phí một đơn vị: `≈ 5,5 token/ký tự + 4K cố định` (sai số ±30% từ 2 điểm đo; thay bằng hệ số đo được khi đủ dữ liệu, mục 3.2 B).
**Kết quả:** **50 đơn vị** (BHXH 12, BHYT 6, TNCN 3, mức lương tối thiểu 1, BLLĐ hợp nhất 16, Điều kiện lao động 12), 720.573 ký tự, ~4,16M token; nhỏ nhất 6.000 ký tự (~37K token), lớn nhất 29.075 (~163K token); kế hoạch ở `data/eval/units_plan.md`, nguyên văn ở `data/eval/units/`. **Chạy nhỏ → lớn** cũng là chiến lược giảm lãng phí; hệ quả: dừng giữa chừng thì testset lệch về đơn vị nhỏ,
`finalize` chỉ cân bằng đúng khi đã chạy đủ.

### 4.5 Theo dõi tiến độ và chạy tiếp nhiều ngày

Yêu cầu: hôm nay dừng ở đơn vị 36, mai chạy lại CÙNG lệnh thì tiếp từ 37. **Thứ tự chạy** = số ký tự tăng dần (hoà: tên văn bản, số thứ tự). **Khoá đơn vị** = `<tên file .md>#<số thứ tự 1-based của split_document>`. `generation_progress.json` (Pydantic `GenerationProgress`, ghi **nguyên tử**) gồm `units` (khoá → `UnitProgress`: `title`, `chars`, `estimated_tokens`, `questions` theo loại,
`llm_calls`, `seconds`, `completed_at`, + `status`, `skipped_samples`, `tokens`, `reasoning_tokens`; khi implement 3.4 thêm metadata `skipped`) và `last_failure` (`unit`, `error` đã che, `at`).
- **File này (không phải raw) quyết định "đã xong"** — người dùng xoá dòng xấu trong raw nên suy từ raw sẽ chạy lại đơn vị bị xoá hết dòng hoặc sinh 0 câu, tốn token oan. **Thứ tự ghi: raw trước, progress sau**; chết giữa hai bước → lần sau thấy đơn vị có dòng trong raw (theo `source_document` + `source_section`) mà chưa có trong progress → coi đã xong, ghi bổ sung + log.
- **Khi hết quota ngày:** ghi `last_failure` (không có nội dung câu hỏi/context), in tóm tắt, **dừng luôn không thử đơn vị kế**, mã thoát ≠ 0; lần sau thành công thì xoá `last_failure`. Lỗi không quota xử lý retry/pass theo 3.4, không dùng `last_failure` để báo một job đã thất bại.
- **KG và đơn vị dở:** KG dựng xong là đồ thị hoàn chỉnh (~30–160K token, tới ~10% TPD 6 tài khoản) nên lưu ngay sau `apply_transforms`, trước sinh câu; đơn vị chưa `done` **tự dùng lại** KG nếu file còn và `page_content` node DOCUMENT vẫn khớp văn bản (lệch/hỏng thì dựng lại + log); đơn vị đã `done` (`--append`) chỉ dùng lại khi có `--reuse-knowledge-graph`. Muốn ép dựng lại: xoá `knowledge_graph/<văn bản>__<số>.json`.
- **`--append` bắt buộc đi kèm `--only`** (thiếu → mã thoát 2, không gọi LLM — tránh chạy lại cả corpus). `--retry-skipped` cũng bắt buộc `--only`, không ngầm chạy lại mọi unit đã bỏ. Chạy lại một đơn vị: `generate --only "<tên>#<số>" --append` (nối dòng, cộng dồn số liệu). **Mã thoát:** 0 khi job chạy hết cả unit `done`/`skipped`; 1 chỉ khi hết quota ngày; 2 đầu vào/cấu hình sai (`--only` sai, thiếu thư mục/`.md`, progress/raw hỏng hoặc không UTF-8, thiếu `GROQ_API_KEY_*`, `finalize` thiếu câu).
- **Retry:** theo đúng một tầng/router và retry/pass phân tầng tại mục 3.4; `max_workers=4` giữ nguyên. Không có `RunConfig` retry bổ sung.
- **Kiểm tra khớp nguồn:** unit `done` hoặc `skipped` mà `chars` khác kết quả chia hiện tại → dừng với lỗi nêu tên unit (không tự bỏ qua/ghi đè; số thứ tự có thể trỏ sang unit khác). **`--dry-run`** (không tốn token): bảng theo thứ tự chạy gồm khoá, ký tự, ước lượng token, **trạng thái** (`xong dd/mm hh:mm` / `dở (đã có N/M câu)` / `bỏ qua <stage>` / `chưa`), unit chạy tiếp theo được đánh dấu, dòng tóm tắt tách số done/skipped/còn lại và ước lượng token của phần chưa chạy.
Hôm sau chỉ chạy `tools/generate_testset.py generate` (không `--only`). Chạy nền: `setsid nohup uv run --group eval --no-group production tools/generate_testset.py generate > data/eval/generate.log 2>&1 &`.

### 4.6 Tiến độ thực tế và việc còn lại (2026-09-29)

| Hạng mục | Giá trị |
| --- | --- |
| Đơn vị đã xong | **33 / 50** (47,9% ký tự; ước lượng ~2,03M / 4,16M token) |
| Câu trong raw | **97**: 90 single-hop, 7 multi-hop specific, **0 multi-hop abstract** |
| Lượt gọi / thời gian | 672 lượt / ~4,3 giờ chạy thực |
| Dừng lần cuối | 2026-09-29 16:14: `DailyQuotaExhaustedError` (6/6 tài khoản hết TPD 200K) ở `Điều kiện lao động và quan hệ lao động.md#8` |
| Còn lại | **17 đơn vị lớn nhất** (17.093 → 29.075 ký tự; ~2,12M token) ≈ 1,2 ngày ở 1,8M/ngày → thực tế 2 lần chạy |

**Chạy tiếp:** `tools/generate_testset.py generate` (tự bắt đầu ở đơn vị thứ 34). **Trước khi chạy full:** `.env` phải có đủ `GROQ_API_KEY_1`…`_9`; **pilot nhỏ** để xác nhận Groq nhận `reasoning_effort` và `reasoning_tokens` giảm khi dựng KG, và 3 key mới có tiêu thụ token trên dashboard.

**Vấn đề mở — `multi-hop abstract` = 0 (người dùng chốt: chạy cho xong đã, xét sau):** 33 đơn vị lẽ ra cho ~11 câu abstract và ~11 specific, thực tế 0 và 7 (single-hop đúng kế hoạch: 90 so ~92). Nghi ngờ: KG theo Chương nhỏ không có cụm đoạn cho `MultiHopAbstractQuerySynthesizer` nên `_has_clusters` trả `False` và synthesizer bị bỏ với log cảnh báo
`KG không có cụm cho loại abstract: bỏ N câu (không bù)` — **chưa xác minh**. Hệ quả nếu giữ: raw ~207 dòng (~192 single + ~15 specific + 0 abstract), đủ trên 180 nhưng không có multi-hop abstract (phân bố ~92/0/8). Sau khi xong 50 đơn vị chọn một: (a) sinh bù abstract bằng `generate --only "<tên>#<số>" --append` trên đơn vị lớn có KG dày;
(b) chấp nhận và ghi vào báo cáo Phase 2 (multi-hop chỉ có specific); (c) đổi tỷ lệ 90/0/10 và sửa mục 10.10. Nên xác minh nguyên nhân (đọc `_has_clusters` trên KG đã lưu của vài đơn vị lớn) trước khi chọn (a).

## 5. Model dữ liệu (`models.py`)

`GoldenTestCase(user_input, reference, reference_contexts: list[str], synthesizer_name: str | None, source_document: str | None, source_section: str | None)` — `source_document`/`source_section` do code gắn (ragas không trả), dùng cho resume, cắt phân tầng `finalize`, và báo điểm theo luật/Chương ở Phase 2; không lẫn với `RetrievedChunk`.
`UnitProgress(title, chars, estimated_tokens, questions: dict[str,int], llm_calls, seconds, completed_at, status: Literal["done","partial","skipped"]="done", skipped_samples: int=0, skipped_question_types: set[str]=set(), skipped_stage: Literal["knowledge_graph","generation"]|None=None, error_type: str|None=None, attempts: int=0, tokens: int|None=None, reasoning_tokens: int|None=None)`; `UnitFailure(unit, error, at)`; `GenerationProgress(units: dict[str, UnitProgress], last_failure: UnitFailure | None)`. Trường `skipped_*`/`error_type`/`attempts` là metadata không chứa nội dung prompt hay exception message.

## 6. Config (`config.py`)

`TestsetGeneratorSettings` (`SettingsConfigDict(env_file=".env", extra="ignore")`): `api_key`…`api_key_9` (alias `GROQ_API_KEY_1`…`_9`, mỗi cái `Field(min_length=1)`), `model_name = "openai/gpt-oss-120b"` (sinh multi-hop cần khả năng tổng hợp), `timeout_seconds = 60`. Retry SDK không lấy từ settings: implementation 3.4 luôn truyền `ChatOpenAI(max_retries=0)`. **Cả 9 key BẮT BUỘC và không rỗng** (khác `GenerationSettings`/`JudgeSettings` có fallback) — round-robin chỉ có ý nghĩa khi đủ credential được người vận hành cấp quyền; thiếu thì pydantic báo lỗi ngay lúc khởi tạo,
thông báo chỉ nêu TÊN biến (lỗi gốc của pydantic in đầu/đuôi key nên `EvalInputError` bọc lại, mã thoát 2). Không dùng chung class với `GenerationSettings` dù trùng tên biến (mục đích khác). Không setting riêng cho embeddings (`embeddings_adapter.py` tái dùng `EmbeddingSettings`). Module không đọc `.env` trực tiếp; `.env.example` có `GROQ_API_KEY_1`…`_9`.

## 7. Module (`src/production_legal_qa_rag/evaluation/`)

`models.py`; `corpus_loader.py` (đọc `.md` → `Document`); `embeddings_adapter.py`; `groq_round_robin.py` (router 9 client, mục 3.1–3.2); `unit_splitter.py` (`EvalUnit`, `split_document`, `split_directory`; hằng `MAX_UNIT_CHARS=30.000`, `MIN_UNIT_CHARS=6.000`, `TOKENS_PER_CHAR=5,5`, `FIXED_TOKENS_PER_UNIT=4.000`; thuần Python, không import `ragas`);
`ragas_runner.py` (`RagasUnitRunner`, `cap_token_limit`, 3 `Clean*Synthesizer`, `build_run_config`, vòng sinh mục 3.3; **toàn bộ code chạm ragas nằm ở đây** để `testset_generator.py` không cần nhóm `eval`); `testset_generator.py` (điều phối, `allocate_questions`, `finalize_testset`, `_describe_traceback`, dry-run); `tools/split_eval_units.py`
(chia + ghi `units/`, `units_plan.md`, không gọi LLM; giữ để xem nguyên văn đơn vị); `tools/generate_testset.py` (Typer mỏng: `generate`, `finalize`; mã thoát 0/1/2). Đặt ở package `evaluation/` (không chỉ script rời) vì Phase 2 thêm module vào cùng package.

## 8. Xử lý lỗi

| Sự cố | Xử lý |
| --- | --- |
| Hết quota ngày trên mọi credential còn hoạt động | Dừng process ngay, log **khoá unit** + loại lỗi/traceback đã che, ghi `last_failure`, giữ `partial` nếu đã có sample; không thử credential/unit mới. Chạy lại process sau khi quota hồi phục. |
| 429 theo phút; timeout/connection/5xx/498 | Chỉ router retry theo giới hạn mục 3.4; 429 dùng `retry-after` và cooldown, lỗi tạm thời khác chỉ có một retry với backoff+jitter. Hết retry thì pass scope RAGAS nhỏ nhất có thể, không dừng job. |
| 400/401/403/413 hoặc lỗi RAGAS sau HTTP 200 | Không retry mù: 401/403 disable credential, 400/413 bỏ request; lỗi RAGAS được vá ở wrapper nếu xác định được, nếu không thì retry đúng một lần ở scenario/sample rồi pass. Dựng KG lỗi → unit `skipped`, ghi metadata đã che và chạy unit sau. |
| `--only` sai tên văn bản; `data/markdown` thiếu/không `.md`/không UTF-8; `--append` thiếu `--only` | `EvalInputError` (mã thoát 2) trước khi gọi LLM, không traceback |
| Thiếu/rỗng/sai `GROQ_API_KEY_1`…`_9` | `EvalInputError` (mã thoát 2) chỉ nêu TÊN biến, không in giá trị |
| `embeddings_adapter.py` nhận response sai định dạng | Raise lỗi rõ, không trả vector rỗng (validate ở biên như `embedding/hf_client.py`) |
| `generate` gặp đơn vị đã `done` không có `--append` | Bỏ qua + log "đã xong" (không gọi Groq, không ghi đè; không bao giờ sắp xếp lại/ghi đè dòng raw người dùng đã sửa tay) |
| Đơn vị `done` mà `chars` khác kết quả chia hiện tại | Dừng, nêu tên đơn vị (mục 4.5) |
| Đơn vị có dòng raw nhưng chưa có trong progress | Coi đã xong, ghi bổ sung + log, không sinh lại |
| `generation_progress.json` hỏng/không parse được | Raise lỗi rõ; không coi như "chưa làm gì" (đốt lại toàn bộ quota) |
| `finalize`: còn < 180 câu; raw thiếu hoặc một dòng thiếu/rỗng trường bắt buộc | Dừng với lỗi nêu thiếu bao nhiêu + gợi ý sinh bù / vị trí dòng lỗi; không ghi file thiếu, không âm thầm bỏ qua |

## 9. Nghiệm thu thủ công

0. **Pilot (người dùng + architect, TRƯỚC khi chạy full).** Pilot thăm dò đã xong (4 lần, mục 4.1). Còn lại, bằng CLI thật: `generate --only "Quy định mức lương tối thiểu" --testset-size ~10 --output-dir data/eval_pilot` để kiểm: **multi-hop** (2 synthesizer có sinh được, tiếng Việt sạch, ≥2 `reference_contexts`, trọng số lẻ ra đúng số câu); **đơn vị nhỏ nhất (~6.000 ký tự)** có dựng KG và sinh câu được;
   **resume và tái dùng KG** (chạy lại bỏ qua đơn vị đã xong kể cả sau khi xoá dòng trong raw; Ctrl+C hoặc gỡ một key ép lỗi → `last_failure` ghi, lần sau tiếp tục đúng chỗ, KG không dựng lại; ép hết quota ngày → dừng sau một vòng và thoát mã 1); **round-robin thật** (cả 9 key nhận tải); **`reasoning_effort` khi dựng KG** (Groq nhận, `reasoning_tokens` giảm).
1. Chạy `uv run --group eval --no-group production tools/generate_testset.py generate [--only ...]`, chia nhiều ngày theo quota (xem tiến độ bằng `--dry-run`). 2. Raw khi đủ 6 văn bản ~240 dòng (192/24/24 lệch vài câu, đủ trường không rỗng). 3. Kiểm round-robin: log token/lượt theo tài khoản hoặc dashboard Groq — tải xấp xỉ đều (~1/9), không request nào thất bại hẳn vì rate limit.
4. **Người dùng đọc LƯỚT ~240 câu** (chỉ đọc lướt, không đối chiếu đáp án với luật — hệ quả mục 10.9): xoá câu vô nghĩa, máy móc ("so sánh Điều X và Y"), trùng ý; không có ngưỡng cứng. 5. `finalize` → đúng 180 câu; báo thiếu thì sinh bù (`--reuse-knowledge-graph --append`, dư ~30%) rồi `finalize` lại.

## 10. Rủi ro / điểm mở

1. `embeddings_adapter.py` là code mới, chưa có tiền lệ — test tay kỹ trước khi tin cho việc build KG. 2. Câu ragas sinh lệch phân bố so với câu hỏi người dùng thật (thiên về "Điều X quy định gì" hơn tình huống); hạn chế chung của synthetic data, xử lý bằng review tay, không có gate tự động. 3. Resume chỉ ở mức đơn vị (và phần sample đã xong, mục 3.3), không ở mức lượt gọi LLM.
4. Chữ ký ragas 0.4.3 đã xác nhận cho single-hop và khoá bằng test; **chưa kiểm chứng ở quy mô thật:** multi-hop, `query_distribution` trọng số lẻ, tái dùng KG. 5. Round-robin 9 tài khoản là pattern mới, chưa có tiền lệ — không áp lại ở nơi khác trừ khi có nhu cầu tương tự (khối lượng LLM lớn, job offline); `generation/`/`retrieval/`/`conversation/` giữ 1 key cố định/bước.
6. Router gọi đồng thời (ragas `max_workers=4`) nên trạng thái dùng chung phải thread-safe (bài học 0.3). 7. Phase 2 thiết kế ở mục 11: `retrieve` + `generate` thẳng (không qua `conversation/`), bỏ cache; judge RAGAS `gpt-oss-120b` trên 9 key; ngưỡng theo dõi và tần suất chạy chưa chốt (11.9).
8. (Phase 2) hit@k theo `chunk_id` cần map `reference_contexts` sang chunk hệ thống; pilot thăm dò xác nhận nguyên văn 12/12 trên 1 văn bản nhỏ single-hop, còn phải đo trên testset thật nhiều Chương và multi-hop trước khi tin. 9. **`reference` không được đối chiếu với luật** (người dùng chốt chỉ đọc lướt, 2026-09-28): `reference` do LLM viết, nếu sai thì điểm Phase 2 lệch mà không ai biết → điểm là
   "mức khớp với đáp án do LLM sinh", không phải "đúng luật tuyệt đối"; nếu điểm Phase 2 bất thường, việc đầu tiên là kiểm mẫu vài `reference` với luật gốc.
10. **Multi-hop ghép nhiều Điều là loại hệ thống yếu** → tỷ lệ 80/10/10, Phase 2 báo điểm tách theo `synthesizer_name` (single-hop là chỉ số chính); 24+24 multi-hop chia cho nhiều Chương nhỏ (mỗi Chương 0–2 câu) — ragas có thể sinh không đủ (thực tế đã thấy abstract = 0, mục 4.6). 11. Đơn vị = Chương/Mục (đã đo): 50 đơn vị, lớn nhất ~163K token vừa quota ngày; hai điểm mở: đơn vị gộp
    nhiều Chương (BLLĐ #16 = XV+XVI+XVII) có thể cho câu kém đồng nhất; dừng giữa chừng thì testset lệch về đơn vị nhỏ. 12. Đáp án dính chữ (mục 4.1): chấp nhận; nếu sau này ảnh hưởng điểm Phase 2, cân nhắc đổi model sinh hoặc lọc bằng từ điển âm tiết. 13. Trùng ý giữa câu cùng đoạn ragas và chủ đề nông: chỉ đọc lướt xử lý; >40% bỏ thì tăng `GENERATE_SIZE` hoặc sinh bù.

## 11. Phase 2 — Chạy pipeline thật và chấm điểm (chốt 2026-09-29, chưa implement)

### 11.1 Mục tiêu, phạm vi

Chạy từng câu của `golden_testset.json` qua retrieval + generation thật, chấm bằng RAGAS; trả lời (1) **bật hay tắt MMR** thì retrieval tốt hơn (`retrieval_spec.md` mục 6 coi MMR là cờ evaluation) và (2) chất lượng câu trả lời cuối (đã qua Evidence Judge) theo loại câu và theo văn bản luật. **Làm:** HyDE → embed → retrieve (2 cấu hình `mmr_on`/`mmr_off`) → chấm retrieval → generation của cấu hình thắng
(`GenerationPipeline.generate`: draft + hard gate + Judge + repair) → chấm câu trả lời → báo cáo. **Không làm:** guardrail/condense/cache/`api/`/`conversation/` (tầng end-user; testset là câu đơn lượt độc lập, cache làm sai số đo — hệ quả: tỷ lệ guardrail chặn nhầm KHÔNG được đo); **Groq Batch API** (tài khoản free không dùng được; chỉ gom 25 text/request embed HF);
hit@k (mục 10.8); lát đánh giá nhiễu; lấy mẫu Langfuse; CI/cron. **Tiêu chí hoàn thành:** chạy đủ stage trên 180 câu, có `data/eval/phase2/report.json` và bảng so sánh MMR bật/tắt + điểm câu trả lời theo `synthesizer_name`/`source_document`.

### 11.2 Stage, file trung gian, resume

Mỗi stage đọc file stage trước, ghi một JSONL ở `data/eval/phase2/`, chạy lại được — đổi prompt generation chỉ chạy lại S5–S6. Khoá bản ghi = `case_id` = 12 ký tự hex đầu `sha256(user_input)` (ổn định dù người dùng xoá/đổi thứ tự dòng).

| Stage | Việc | Tài nguyên | File ra |
| --- | --- | --- | --- |
| S1 `hyde` | `HydeGenerator.generate(user_input)` cho mọi câu | `gpt-oss-20b`, cả 9 key, throttle bucket `(model, key)` sẵn có | `hyde.jsonl` |
| S2 `embed` | Embed `[hypo, query]` bằng `QueryEmbedder` (pyvi, cùng model index), gom 25 text/request | HF Inference | `embeddings.jsonl` (không commit, ~5 MB) |
| S3 `retrieve` | `RetrievalPipeline.retrieve(query, use_mmr=…, precomputed=…)`, **tuần tự** từng cấu hình | Pinecone, BM25, rerank GPU local | `retrieved_mmr_on.jsonl`, `retrieved_mmr_off.jsonl` |
| S4 `score-retrieval` | RAGAS `context_precision` + `context_recall` mỗi cấu hình | `gpt-oss-120b`, 9 key | `retrieval_scores.jsonl` |
| S5 `generate --config` | `GenerationPipeline.generate(query, chunks)` cho cấu hình người dùng chọn sau S4 | 120b + Judge 20b, cả 9 key | `answers.jsonl` |
| S6 `score-answers` | `faithfulness` + `answer_relevancy` trên câu `answered` | 120b 9 key + embedding HF | `answer_scores.jsonl` |
| `report` | Tổng hợp, không LLM | — | `report.json` + bảng terminal |

**Quy tắc chung:** stage bỏ qua `case_id` đã có; thiếu bản ghi stage trước thì báo số còn thiếu và chỉ xử lý phần đã có; lỗi tạm thời (`error` ≠ null) chạy lại bằng `--retry-failed`, mặc định KHÔNG (tránh đốt token vào lỗi tất định); ghi nối từng dòng (S4/S6 theo lô `SCORING_BATCH_SIZE = 10`); dòng cuối hỏng thì bỏ + log, không raise; mã thoát 0/1/2 như Phase 1;
`--testset` (mặc định `golden_testset.json`; có thể trỏ raw để kiểm pipeline, không đọc điểm như chính thức), `--limit N` pilot. Người dùng cam kết không chạy việc khác trên các bucket Groq lúc chạy.

### 11.3 Chi tiết từng stage

**Rải 9 key (S1, S5).** `HydeGenerator`, `AnswerGenerator`, `EvidenceJudge` mỗi cái đọc key cố định nên không tự xoay. Eval dựng 9 bộ (`HydeSettings`/`GenerationSettings`/`JudgeSettings` truyền key tường minh theo `validation_alias`; đặt `GROQ_API_KEY_4` của `GenerationSettings` là `None` để mỗi bộ đúng một key), gán bản ghi chờ xử lý cho 9 bộ theo vòng tròn **tính trên danh sách còn lại lúc chạy**
(nên `--retry-failed` tự rơi sang key khác). Một tài khoản hết TPD chỉ làm các câu gán cho nó ra `error` 429. Helper ở `key_pool.py`, không sửa code production. **S1:** tái dùng `HydeGenerator` nguyên trạng; `None` → `hypothetical_document: null` và bỏ nhánh A đúng như production; phân biệt `error` (chạy lại được) với `null` không lỗi.
**S2:** gom `[hypo, user_input]` nhiều câu vào request 25 text, không đổi thứ tự (`QueryEmbedder.embed` đã word-segment). **S3:** cần `precomputed` (`retrieval_spec.md` mục 2, thay đổi duy nhất ở code production); **một** `RetrievalPipeline`, chạy **tuần tự** (hết 180 câu `use_mmr=True` rồi `False`; máy chạy rerank sát giới hạn GPU 2GB, không song song hoá; `RETRIEVE_CONCURRENCY = 1`);
CUDA OOM → giảm `batch_size` của `LocalReranker` rồi `--retry-failed`, **không đổi model hay `max_length`** (sẽ đo sai hệ thống thật). **Fallback rerank là lỗi, không phải kết quả:** chunk nào có `rerank_score is None` → ghi `error` và không chấm (để lọt thì S4 so hai cấu hình bằng thứ tự không qua rerank mà không ai biết); kết quả rỗng → `error = "no_context"`; `RetrievalError` → `error`.
**S4:** `LLMContextPrecisionWithReference` + `LLMContextRecall` (chỉ cần `user_input`, `retrieved_contexts`, `reference` — chọn cấu hình MMR trước khi tốn token generation); judge = `LangchainLLMWrapper(GroqRoundRobinChatModel)` của Phase 1 (9 key, cùng `RunConfig`); chuỗi mỗi chunk = đúng phần chunk trong `build_context` của generation (breadcrumb + content/raw_table), tách hàm dùng chung nếu cần, không tự chế format; 6 lượt/record/cấu hình.
**Chọn cấu hình sau S4 (người dùng quyết):** `report` in bảng so sánh và số câu `mmr_on` hơn/thua/hoà `mmr_off` theo từng câu (180 câu: chênh trung bình nhỏ dễ là nhiễu judge). Gợi ý: ưu tiên `context_recall` (thiếu chunk nặng hơn với luật), `context_precision` phá hoà. Lý do: MMR chỉ đổi **tập candidate vào union trước rerank**, thứ tự cuối và cắt top 5 do reranker quyết, nên `context_precision` ít khác giữa hai cấu hình;
`context_recall` mới là chỗ MMR giúp (candidate đa dạng) hoặc hại (phạt oan Khoản liền kề). `context_precision` lệch nhiều là tín hiệu bất thường, không dùng để quyết; không khác biệt rõ thì **tắt MMR** (đơn giản hơn, bớt một lượt Pinecone). Kết luận ghi vào `retrieval_spec.md` mục 6.

### 11.4 S5 — Generation (phương án B, 2026-09-29)

`GenerationPipeline.generate(query, chunks)` không gọi guardrail và không retrieve — đúng "bỏ tầng end-user" mà không viết lại logic; kết quả là câu người dùng thật thấy (đã qua hard gate + Judge + tối đa 1 repair). Production chỉ có 2 tài khoản cho generation (3⇄4) và 1 cho Judge (key 2), nên eval dựng **9 `GenerationPipeline` độc lập, mỗi cái một key**: `AnswerGenerator` (120b) và `EvidenceJudge` (20b) cùng dùng key i
(không tranh nhau vì rate limit tính theo `(tài khoản, model)` và hai bước khác model). `AnswerRecord`: `case_id`, `config`, `outcome ∈ {answered, insufficient_evidence, unable_to_verify, error}`, `response`, `citations`, `repair_used`, `warning_codes`, `error_code`, `usage`. `insufficient_evidence` và `unable_to_verify` là **kết quả hợp lệ của hệ thống**, không chạy lại; chỉ `error` chạy lại được.
**Hệ quả đọc điểm:** câu bị từ chối không có `response` nên RAGAS không chấm — báo riêng **tỷ lệ từ chối** (theo loại và theo `synthesizer_name`); `faithfulness` đo trên câu ĐÃ qua Judge nên cao hơn faithfulness của draft — phản ánh "hệ thống cả Judge", không tách được generator riêng.

### 11.5 S6 — Chấm câu trả lời

`Faithfulness` (2 lượt/câu) + `ResponseRelevancy` (`answer_relevancy`, `strictness = 3` → 3 lượt riêng vì wrapper round-robin không có `n`, tránh lỗi Groq không hỗ trợ `n>1`) — 5 lượt/câu, chỉ trên `answered`; `context_precision`/`recall` của cấu hình thắng đã có từ S4. `answer_relevancy` cần embedding: dùng `RagasEmbeddingsAdapter` với **`segment=True`** (tham số mới, mặc định `False` để không đổi Phase 1) áp `ViTokenizer` — khác mục 3 vì ở đây là cosine
giữa câu hỏi gốc và các câu hỏi ragas sinh lại, cần cùng không gian với model đã huấn luyện trên văn bản đã segment (`embedding_spec.md` mục 4); kiểm ở pilot rằng điểm không toàn ~0 hay ~1.

### 11.6 Báo cáo (`report.json`)

**So sánh retrieval:** trung bình `context_precision`/`context_recall` của `mmr_on` và `mmr_off` (tổng, theo `synthesizer_name`, theo `source_document`) + số câu thắng/thua/hoà, chỉ trên `case_id` hợp lệ ở CẢ HAI cấu hình (báo số câu bị loại). **Điểm câu trả lời** (cấu hình đã chọn): trung bình bốn metric theo cùng các lát, kèm `n` mỗi lát và số `null` (NaN bỏ khỏi trung bình, không coi là 0).
**Vận hành:** tỷ lệ `answered`/`insufficient_evidence`/`unable_to_verify`/`error`, tỷ lệ dùng repair, số câu HyDE `null`, số câu retrieval lỗi/fallback. **Ghi chú diễn giải cố định:** điểm = mức khớp với `reference` do LLM sinh (10.9); judge cùng họ model với generator; single-hop là chỉ số chính, multi-hop báo riêng (10.10).

### 11.7 Module

Thêm vào `evaluation/` (chỉ `scoring.py` import `ragas`): `run_models.py` (`case_id()`, các record Pydantic, `EvalConfig`), `jsonl_store.py` (đọc/ghi nối JSONL có validate, bỏ dòng cuối hỏng, tập `case_id` đã xong), `key_pool.py` (9 bộ settings/instance + gán vòng tròn), `hyde_stage.py`/`embed_stage.py`/`retrieve_stage.py`/`generate_stage.py` (chỉ điều phối),
`scoring.py` (S4/S6, tái dùng wrapper LLM/`RunConfig` của `ragas_runner.py`), `report.py` (hàm thuần), `tools/run_eval.py` (Typer: `hyde`, `embed`, `retrieve`, `score-retrieval`, `generate`, `score-answers`, `report`, `status`; option chung `--testset`, `--output-dir`, `--limit`, `--retry-failed`). Thay đổi ngoài `evaluation/`: `retrieval/` thêm `PrecomputedQuery` + `precomputed`; `RagasEmbeddingsAdapter` thêm `segment`;
`.gitignore` thêm `data/eval/phase2/embeddings.jsonl`. Chạy trong venv `eval`.

### 11.8 Ước lượng chi phí (thô, chưa đo — pilot `--limit 10–20` để đo token thật)

180 câu: S1 180 lượt 20b; S2 ~15 request HF; S4 2 cấu hình × 180 × 6 = 2.160 lượt 120b; S5 ~180–360 lượt 120b + 180–360 Judge 20b; S6 180 × 5 = 900 lượt 120b. Tổng judge RAGAS ~3.060 lượt (~3.960 nếu chạy đủ metric cả hai cấu hình); giả định thô 1–2K token/lượt → ~3–6M token, cỡ Phase 1; nút thắt vẫn là TPD 200K/(tài khoản, model) → chia nhiều ngày, dựa vào resume.

### 11.9 Rủi ro / điểm mở

1. **Venv `eval` dùng `openai` cũ hơn production:** S1/S5 chạy code production (`AsyncGroq`, `with_structured_output(method="json_mode")`) trong venv này; Phase 1 đã chạy `ChatOpenAI` + Groq ổn nhưng chưa kiểm luồng generation/Judge — kiểm ở pilot; lệch thì phải tách venv chạy S1–S3, S5. 2. **Điểm lạc quan hoá:** Judge lọc trước (faithfulness), judge RAGAS cùng họ với generator, `reference` chưa đối chiếu với luật —
đọc điểm như xu hướng/so sánh giữa các lần chạy. 3. **`n = 180` nhỏ:** chênh dưới nhiễu judge không kết luận được → báo số câu thắng/thua. 4. Testset raw dở dang lệch (mục 4.4): chỉ để kiểm pipeline. 5. **Chưa chốt:** ngưỡng theo dõi ("đạt" là bao nhiêu), tần suất chạy lại, chạy lại toàn bộ hay chỉ S5–S6 — sau khi có số đo đầu tiên.
6. **Rerank GPU 2GB sát giới hạn:** OOM lẻ tẻ ở S3; xử lý ở 11.3; câu OOM bị bỏ khỏi so sánh MMR nếu chưa chạy lại — báo số bị loại để không so hai cấu hình trên tập khác nhau. 7. Không đo guardrail — cần lát đánh giá riêng nếu muốn biết tỷ lệ chặn nhầm.

### 11.10 Pilot trước khi chạy full (2026-09-29, người dùng đồng ý)

Sau khi implement, chạy S1→S6 với `--limit 10–20` (có thể trỏ raw khi chưa `finalize`) ra `--output-dir data/eval/phase2_pilot`. Phải trả lời: token và số lượt gọi thật từng stage (thay 11.8 → số ngày cho 180 câu); venv `eval` chạy được S1/S5 không (11.9.1); S3 trên GPU 2GB có OOM không và `batch_size` nào đủ; `answer_relevancy` với embedding có segment cho điểm hợp lý; **rải 9 key hoạt động thật**
(tải xấp xỉ đều; câu gán cho tài khoản hết quota ra `error` rồi `--retry-failed` sang key khác); resume (Ctrl+C rồi chạy lại từng stage không làm lại bản ghi xong); `report.json` đọc được, đủ các lát. Chỉ chạy full khi pilot đạt.
