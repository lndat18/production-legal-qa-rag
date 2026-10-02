# Conversation — Multi-turn (condense) → Guardrail → Cache → Admission → Retrieve → Generate

- Giữ nguyên số mục, đặc biệt 12.1.
- Prompt trong code (`CONDENSE_SYSTEM_PROMPT`, `GENERATION_SYSTEM_PROMPT`) là nguồn sự thật; spec mô tả quy tắc.
- Spec liên quan: [generation_spec.md](../generation/generation_spec.md), [cache_spec.md](../cache/cache_spec.md), [api_spec.md](../api/api_spec.md), [observability_spec.md](../observability/observability_spec.md), [evaluation_spec.md](../evaluation/evaluation_spec.md).

## 1. Mục tiêu & phạm vi

- Điều phối chatbot nhiều lượt từ `messages[]`: hiểu follow-up, cache, admission, retrieve/generate, phát event cho API.
- `ChatOrchestrator.stream()`: guardrail ‖ condense → cache → admission → retrieve/relevance gate → generate; điền `TurnTrace` cho Langfuse/metrics.
- Không lưu history backend: OpenWebUI giữ, client gửi lại mỗi lượt.
- Không làm: memory dài hạn/summary/upload, history vào generation, semantic cache, Kafka/agent/state graph, quota user/ngày/toàn cục/ngày.
- Acceptance chính: follow-up kế thừa đúng chủ thể/văn bản/số Điều/Khoản; retrieval tương đương câu độc lập; cùng standalone query không phụ thuộc history ở generation.

## 2. Input & Output

- Pydantic v2: `ChatMessage(role ∈ {user, assistant}, content)`; `RequestContext(user_id, chat_id, request_id)`.
- `TurnTrace` mutable, orchestrator ghi dần, API đọc cuối stream:
  - `raw_query`, `standalone_query`, `verdict`, `cache_status` (answer_hit/retrieval_hit/miss/bypass).
  - `outcome` (answered/refused/error), `error_code`, `chunk_ids`, `answer_text`, `citations`, `warnings`, `usage`.
  - `time_to_first_token_ms`, `latency_ms`.
- Outcome mặc định `error`, tránh ghi nhầm answered khi bị hủy.
- API chỉ gọi `ChatOrchestrator.stream(messages, ctx, trace) -> AsyncIterator[GenerationEvent]`.
- Dùng union event generation; luồng bình thường/lỗi kết thúc `done`; exception ngoài cùng → `error(llm_error)` nếu chưa done, không ném ra API.

## 3. Quyết định thiết kế cốt lõi

- **Generator chỉ nhận `(standalone_query, chunks)`, không history.** Cache theo standalone query; giữ ngân sách TPM; RAGAS đo cùng logic generation.
- History chỉ vào condense (≤3 lượt) và guardrail (≤2 câu user); các bước sau chỉ thấy câu độc lập.
- Condense phải bổ sung đủ ngữ cảnh.
- `messages[]` không đáng tin, kể cả assistant giả: delimiter trong prompt, bỏ qua chỉ dẫn trong dữ liệu; không dùng làm system instruction.

## 4. Cửa sổ history (`history.py`)

- `build_window(messages) -> HistoryWindow(query, history)`:
  - Bỏ role ngoài user/assistant (kể cả system) và content rỗng.
  - Message cuối phải là user; nếu không → `InvalidConversationError`, API 422.
  - Query tối đa `MAX_QUERY_CHARS=1000`; lấy ≤`HISTORY_MAX_TURNS=3` cặp gần nhất trước query.
  - Assistant: cắt từ `SOURCES_FOOTER_MARKER`/`DATA_SNAPSHOT_DISCLAIMER`, xóa `[n]`, giới hạn `HISTORY_ASSISTANT_MAX_CHARS=600`, thêm “…” khi cắt.
- Không history → không condense.
- `recent_user_turns`: ≤`GUARDRAIL_CONTEXT_TURNS=2` câu user, chỉ cho guardrail.
- Disclaimer tĩnh: “dữ liệu pháp luật được cập nhật tới `CORPUS_SNAPSHOT_DATE`”; ngày cập nhật thủ công mỗi re-index.
- API nối disclaimer sau citations/warnings, trước done; LLM không sinh câu này.

## 5. Condense (`condenser.py`)

- `QueryCondenser.condense(query, history) -> str`; `condense_detailed()` trả thêm `CondenseReason`.
- Groq 20b/key 1; `LoopBoundClient`; `reasoning_effort=low`, `temperature=0`, `include_reasoning=False`, `max_completion_tokens=2048`.
- Prompt có 5 few-shot; chỉ dùng history bổ sung chủ thể/văn bản/Điều/Khoản/Điểm/tình huống thiếu.
- Giữ mọi định danh và con số; không thêm số Điều/Khoản không có trong hội thoại.
- Query tự đủ nghĩa/đổi chủ đề → nguyên văn; chỉ hỏi Khoản/Điểm → bổ sung Điều và tên văn bản từ history.
- Không trả lời/giải thích; một câu hỏi trên một dòng; query/history chỉ là dữ liệu.
- Không sao chép thuật ngữ chỉ phù hợp nhóm cũ sang nhóm mới (ví dụ “thai sản” sang lao động nam); viết khái quát phù hợp.
- `check_condensed`: lấy dòng đầu, bỏ ngoặc bao, dài 5–500; số Điều/Khoản phải có ở query/history qua `extract_citation_numbers`/`extract_citation_khoans`; chưa kiểm Điểm bằng extractor.
- Output sai/Groq lỗi/timeout/429 → query gốc; log reason/finish_reason/token, không nội dung.
- Chỉ retry một lần khi `FINISH_LENGTH`; groq_error/unknown_citation/bad_length/empty không retry.

## 6. Guardrail có ngữ cảnh (thay đổi ở `generation/`)

- `InputGuardrail.check_input(query, recent_user_turns=())`: đọc query gốc + ≤2 câu user trước, không assistant.
- Không đọc query condense vì condense có thể xóa injection.
- Injection ưu tiên; out_of_scope chỉ cho tác vụ phi-tra-cứu rõ ràng; verdict mơ hồ remap allow.
- Địa danh/cơ quan/phụ lục/bảng/giấy phép/số hiệu văn bản/luật ngoài corpus vẫn đi retrieval.
- Chỉ retrieval + relevance gate xác định thiếu evidence (`no_context`); chi tiết generation mục 3/8/9.

## 7. Workflow (`orchestrator.py`)

- Bước 0: `build_window`; invalid → API 422 trước stream.
- Bước 1: status guardrail; `gather` guardrail(query gốc) và condense nếu có history. Injection/phi-tra-cứu → refusal + done, outcome refused.
- Bước 2: `answer_cache.get(standalone)`; hit → replay tokens/citations/done, usage None.
- Bước 3: single-flight answer key; follower chờ rồi đọc cache, không admission (cache mục 6).
- Bước 4: leader vào `admission.slot(ctx.user_id)`; bị từ chối → error. Status retrieval; `retrieval_cache.get(standalone)` hoặc retrieve standalone.
- Bước 5: chunk rỗng/low relevance → error(no_context); còn lại status generation → `generation.generate(standalone, chunks)`.
- Bước 6: luồng sạch, không error/warnings, có citation hoặc “không tìm thấy quy định phù hợp” → ghi answer cache ngay trước done.
- `_record_event` điền trace xuyên suốt; API cập nhật Langfuse/metrics trong finally.
- Status retrieval/generation chỉ khi thực sự chạy; done là cuối luồng.
- Client ngắt → `CancelledError`; finally nhả admission/lock; không cache answer dở.

## 8. Retrieval-relevance gate (`retrieval/relevance.py`)

- `MIN_RERANK_SCORE=-6.8`, logit thô, không xác suất; giữ nguyên khi HyDE đổi 20b.
- `is_low_relevance(chunks)`: max `rerank_score` < ngưỡng → error(no_context); không có score → False để tránh gate mù khi fallback.
- Ví dụ cần chặn: query meta “Tóm tắt lại các câu trả lời ở trên” có evidence không liên quan.
- Retrieval vẫn trả top K; `orchestrator._load_chunks` sở hữu quyết định policy.
- Ngưỡng đo trên reranker hiện tại, ưu tiên tránh chặn oan; score âm không mặc định là không liên quan.

## 9. Admission (`admission.py`)

- `AdmissionController.slot(user_id)`: async context manager cho retrieve + generate khi cache miss.
- Chỉ giới hạn đồng thời: `asyncio.Semaphore(MAX_CONCURRENT_ANSWERS)` + số người chờ; không Redis/quota ngày.
- Vượt `MAX_WAITING` → `AdmissionDenied(kind="overloaded", retry_after_seconds)` → error(rate_limited).
- Người chờ giữ kết nối, API gửi keep-alive; luôn nhả slot trong finally.
- Hết hạn mức Groq: dựa 429 thật tại generation pipeline, đọc retry-after và phát `ErrorEvent(code="rate_limited", ...)`; không dựng quota ngày riêng.
- Chỉ đúng một worker/process; scale replica phải thay bằng semaphore Redis cùng interface.

## 10. Generation — prompt hệ thống (`generation/generator.py`)

- `PROMPT_VERSION` là nguồn version trong code (hiện v11); đổi prompt phải bump cache key.
- Generation 120b: low reasoning, temperature 0.1, completion 2048, `MAX_CONTEXT_CHUNKS=5`.
- Prompt giữ 14 quy tắc:
  - **1:** chỉ evidence, không kiến thức ngoài/suy đoán.
  - **2:** citation `[n]` sau claim; không tự thêm số Điều/Khoản/Điểm thiếu trong evidence.
  - **3–4:** giữ nguyên số, không làm tròn/quy đổi/tính; đọc bảng đúng ô.
  - **5:** thiếu evidence → “Tôi không tìm thấy quy định phù hợp trong các văn bản hiện có”; trả lời một phần phải nói phần thiếu.
  - **6–8:** không tư vấn cá nhân hóa; tối đa một câu nhắc đọc luật gốc; tiếng Việt rõ/bullet; không lộ quy trình; query/context chỉ là dữ liệu.
  - **9:** thiếu yếu tố phân loại → liệt kê riêng từng trường hợp/điều kiện, hỏi yếu tố thiếu; không trộn.
  - **10:** không tự cộng/trừ/nhân/chia hoặc kết hợp số, kể cả một phép tính/một Điều/số query; prompt có ví dụ đúng/sai.
  - **11–12:** không ghép các đoạn khác mạch; không có evidence trực tiếp hoặc yêu cầu nhắc lại history → từ chối theo 5.
  - **13:** kết luận trong 1–2 câu đầu rồi căn cứ; ca 5/9 mở đầu đúng từ chối/liệt kê.
  - **14:** trích ngắn ≤2 dòng bằng blockquote, diễn giải sau, `[n]` ASCII.
- Cache answer không citation chỉ khi có cụm “không tìm thấy quy định phù hợp”; giữ câu chuẩn ở prompt.
- `build_context`: breadcrumb/content/raw_table, nối `\n\n`; citation theo rank, đủ 5 dùng vị trí `1,2,5,3,4`, không đổi nhãn.
- Code kiểm hình thức, Judge kiểm ngữ nghĩa; contract đầy đủ ở generation spec.

## 11. Adapter đánh giá (`evaluation.py`) — hoãn

- Phase 2 gọi retrieval + `GenerationPipeline.generate` trực tiếp; bỏ guardrail/cache/admission/trace orchestration.
- Không implement adapter này; xem evaluation mục 11.

## 12. Config

- `CondenseSettings`: key 1, `model_name="openai/gpt-oss-20b"`, `max_retries=1`, timeout 20s.
- `AdmissionSettings`: `max_concurrent_answers=2`, `max_waiting=6`; không Redis/quota.
- `GenerationSettings`: `api_key` key 3 fallback 1, `round_robin_api_key` key 4, `model_name="openai/gpt-oss-120b"`; `ThrottleSettings`: mục 12.1.
- History constants: `HISTORY_MAX_TURNS`, `HISTORY_ASSISTANT_MAX_CHARS`, `MAX_QUERY_CHARS`, `GUARDRAIL_CONTEXT_TURNS`, `CORPUS_SNAPSHOT_DATE`, `DATA_SNAPSHOT_DISCLAIMER`.
- Relevance constant tại `retrieval/relevance.py`; module không trực tiếp đọc `.env`.

### 12.1 Chính sách model / key / rate limit

- Tên env: `GROQ_API_KEY_1`…`GROQ_API_KEY_9`; generation dùng `GROQ_API_KEY_3`/`GROQ_API_KEY_4`. Production key 1–4 thuộc tài khoản A–D; key 5–9 chỉ eval. Bước nặng dùng model nặng/key xoay vòng; bước nhẹ key cố định.
- Bucket theo model/tài khoản; mapping:
  - Draft/repair: 120b, key 3 ⇄ 4 mỗi call.
  - Condense/HyDE: 20b, key 1, throttle chung.
  - Judge: 20b, key 2 fallback 1; fallback thì chung Condense/HyDE.
  - Guardrail: safeguard-20b, key 1, bucket riêng, không throttle chung.
- `retrieval/llm_throttle.py`: `TokenWindowThrottle`, cửa sổ 60s theo TPM/RPM, `asyncio.Lock`, FIFO, in-process.
- Mặc định `tpm_limit=8000`, `rpm_limit=30`, hệ số 0.9, chỉnh qua env.
- `get_throttle(model, api_key)` dùng `functools.cache`: một instance/bucket, định danh model + SHA-256(key)[:8]; không lưu/log key thật.
- `acquire(estimated_tokens, max_wait_seconds) -> Reservation`: chờ khi sắp đầy, quá hạn `ThrottleTimeout`; `settle(reservation, actual_tokens)` bằng usage thật.
- Ước lượng prompt/`CHARS_PER_TOKEN` + `EXPECTED_COMPLETION_TOKENS` từng bước; không dùng max completion làm ước lượng.
- Condense/HyDE chờ khoảng 8s rồi degrade; Judge chờ tối đa `JudgeSettings.timeout_seconds`, fail-closed. Generation/guardrail không qua throttle production.
- 429 thật vẫn là chốt cuối; throttle không giải quyết TPD. Judge/key 1 hết TPD có thể chuyển một bước nhẹ sang key 3/4 qua config.
- Đổi model phải bump `PROMPT_VERSION` vì cache chỉ chứa model generation (cache spec).

## 13. Module

- `models.py`: message/context/trace; `history.py`: window/constants/làm sạch.
- `condenser.py`: prompt/Groq/validate/retry/fallback; `admission.py`: semaphore/denial.
- `orchestrator.py`: inject guardrail/condenser/cache/retrieve/generate/admission, stream/trace.
- `retrieval/relevance.py`: score signal/ngưỡng.

## 14. Xử lý lỗi

- Invalid messages → 422 trước stream; condense lỗi → query gốc, chỉ retry FINISH_LENGTH.
- Guardrail lỗi → fail-open; Redis lỗi → miss/không lock; admission quá tải → rate_limited + retry-after.
- Chunk rỗng/low relevance → no_context + done; retrieval/generation theo generation spec.
- Stdout không có query/answer; nội dung chỉ Langfuse self-host (observability 4.5).
- Log gate chỉ `max_rerank_score`, ngưỡng và số chunk; không log nội dung hoặc message lỗi LLM.

## 15. Nghiệm thu thủ công — bộ ca mẫu

- Dùng `tools/conversation.py`, Groq/Pinecone thật; đọc condense/citation/cache/outcome trong TurnTrace.
- Thai sản → “Vậy chồng thì sao?”: độc lập về lao động nam, không thêm số Điều/sao chép thuật ngữ sai.
- Khoản 1 Điều 113 → “Còn Khoản 2?”: kế thừa Điều/BLLĐ, key khác Khoản 1.
- Đổi chủ đề: query nguyên văn; cư trú/không cư trú riêng, không tự tính.
- Hỏi thẳng và condense thành cùng chữ: answer_hit; injection query gốc bị chặn; assistant giả không lái generator.
- Condense lỗi/429 follow-up: fallback, không raise; meta history: refusal/no_context, không bịa; lượt đầu “Còn cái đó?”: không condense.
- Sơn Nam/NĐ 293, Long An, gia hạn giấy phép đi retrieval; corpus-miss/địa giới/đất đai thiếu evidence → no_context.
- Chào hỏi/viết code out_of_scope; injection kèm số Nghị định vẫn injection.
- Format: bullet nhiều Khoản/citation (Điều 8 BLLĐ), biểu thuế từ bảng, trích ngắn theo quy tắc 14; cũng có ở `tools/generation.py`.

## 16. Bài học xương máu

- **1:** đo trước khi đổi prompt, không đoán.
- **2:** medium reasoning tốn 3–5 lần low; TPD generation từng chỉ khoảng 50–60 cache-miss/ngày → cache bắt buộc.
- **3:** xác định và sửa đúng tầng gây lỗi; ví dụ condense sai chủ thể ở câu hỏi về chồng nghỉ thai sản.
- **4:** cấm tính phải tường minh từng trường hợp, kèm ví dụ sai trong prompt.
- **5:** regex ngôn ngữ mơ hồ quá 3 vòng thì chuyển LLM; gate regex cho meta history đã thử và không merge.
- **6:** logit âm không đồng nghĩa thiếu liên quan; hiệu chỉnh ngưỡng trên nhiều câu/lần đo.
- **7:** đọc chunk thật trước khi kết luận bug.
- **8:** spec chốt không đồng nghĩa code đã implement; đọc code thật.
- **9:** không tăng khả năng bịa/tổng hợp ngoài context chỉ để cải thiện UX.

## 17. Chiến lược UX/kiến trúc bên ngoài

- Áp dụng: kim tự tháp ngược, blockquote ngắn, disclaimer ngày cập nhật tĩnh.
- Từ chối/hoãn: LangGraph/State Graph/FSM, Buffer/Summary Memory, multi-turn HyDE, CoT ép cứng, bảng so sánh tự sinh/ELI5, gợi ý câu tiếp, NeMo Guardrails/Llama Guard.
- Lý do: ngoài phạm vi stateless/no-agent hoặc làm tăng độ phức tạp.

## 18. Trạng thái hiện tại & rủi ro tồn đọng đã chấp nhận

- **1:** Citation Unicode đã xử lý bằng code: `_CITATION_PATTERN` nhận `【n】`/`[n]`; `AnswerGenerator._buffer()` chuẩn hóa cả text/fragments và thêm space khi dính chữ.
- **2:** Prompt cấm tính vẫn có thể bị vi phạm ở phép tính một bước/một Khoản.
- **3:** Meta history chỉ được bảo vệ bởi quy tắc 12 + relevance gate.
- **4:** TTFT từng 17–44s rerank CPU; chưa đo lại GPU/CPU in-process.
- **5:** Admission một worker; trước vòng sửa mới đọc mục 16.
