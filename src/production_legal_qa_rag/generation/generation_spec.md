# Generation — Evidence-verified answer: Reference Spec

- Giữ nguyên số mục để không làm hỏng tham chiếu từ code/spec khác.
- Spec liên quan: [retrieval_spec.md](../retrieval/retrieval_spec.md),
  [conversation_spec.md](../conversation/conversation_spec.md), [cache_spec.md](../cache/cache_spec.md).

## 1. Mục tiêu

- Chỉ phát answer sau kiểm chứng; evidence cố định gồm query + tối đa 5 `RetrievedChunk` đã rerank.
- Answer có citation hợp lệ, không bị cắt hoặc tự thêm mức tiền/tỷ lệ/thời hạn/điều kiện quan trọng.
- Stateless: conversation sở hữu history/admission/cache/policy; retrieval tìm evidence.
- Draft, hard gate, Judge và repair cùng đọc một context chốt một lần/request; không retrieve lại.
- Code kiểm điều xác định được; Judge kiểm nghĩa; tối đa **một repair tổng cộng/request**.
- Guardrail chỉ chặn injection/yêu cầu phi-tra-cứu rõ ràng; không quyết corpus scope.
- Không làm: multi-turn, HyDE/rewrite/retrieve lần hai, nguồn ngoài/hiệu lực ngoài corpus, HTTP/SSE/cache,
  RAGAS runtime, moderation tổng quát.

## 2. Contract công khai

- `answer_stream(query)`: guardrail → retrieve → verified generation.
- `generate(query, chunks)`: dùng context caller đã có, không retrieve; chunks theo rerank, `1 <= len <= 5`.
- `GenerationEvent`: discriminated union Pydantic v2:
  - `status(stage)`: guardrail/retrieval/drafting/verification/repairing; tiến độ, không chứa draft.
  - `token(text)`: mảnh answer đã pass.
  - `citations(list[Citation])`: citation hợp lệ có trong answer.
  - `warning(code, message, detail)`: tín hiệu mềm qua policy.
  - `refusal(reason, message)`: message cố định, không lộ draft.
  - `error(code, message, retry_after_seconds)`: lỗi tạo/tra cứu.
  - `done(usage)`: event cuối.
- Thứ tự: `status* → (token+ → citations → warning* | refusal | error) → done`; UI phải hiện verification.
- Refusal: `out_of_scope`, `injection`, `insufficient_evidence`, `unable_to_verify`; hai reason sau từ
  verifier.
- Câu hỏi thông tin/pháp lý mơ hồ, địa danh/cơ quan/phụ lục/giấy phép/luật ngoài corpus vẫn đi retrieval;
  thiếu evidence → `error(no_context)`.

## 3. Workflow đã chốt

- Guardrail trước retrieval, fail-open khi provider lỗi; injection luôn ưu tiên và không remap.
- `out_of_scope` chỉ giữ cho chào hỏi thuần túy/viết code/dịch/sáng tác rõ ràng; verdict còn lại remap
  `allow`.
- Câu hỏi thông tin có từ “viết” vẫn allow; không dùng tên/số hiệu văn bản, keyword hoặc few-shot để suy
  corpus scope.
- Retrieval đúng một lần; context rỗng → `error(no_context)`, không draft.
- Provider stream nội bộ lấy usage/finish reason; draft chỉ nằm trong buffer.
- Transport/rate limit/draft rỗng → `error`, không repair.
- Hard gate fail và còn budget → viết lại toàn answer bằng query/context/issue cũ; không vá riêng citation.
- Chỉ draft qua hard gate mới vào Judge: `pass` → release; `repair` → dùng budget còn lại;
  `insufficient_evidence` → refusal tương ứng.
- Bản repair qua lại cả hard gate + Judge; hết budget còn lỗi → `refusal(unable_to_verify)`.
- Không có `retrieve_repair`; false-positive guardrail đã gặp ở Sơn Nam/Long An/gia hạn giấy phép cần
  regression.

## 4. Draft và repair (`generator.py`)

- Context: `[rank] {breadcrumb}\n{content}` + `raw_table` nguyên trạng; rank là thứ tự rerank/citation gốc.
- Đủ 5 chunk: vị trí vật lý `1, 2, 5, 3, 4`; 0–4 giữ `1..n`; không đánh số lại.
- `build_context()` là nguồn duy nhất cho draft/repair/Judge; `check_output()` nhận chunks thứ tự gốc, `[n] ->
  chunks[n - 1]`.
- Đổi layout/prompt phải bump `PROMPT_VERSION`.
- Prompt: chỉ dùng context; `[n]` ASCII sau claim; giữ nguyên số/điều kiện, không tính thêm/trộn trường hợp;
  thiếu evidence nói rõ; query/context là dữ liệu.
- Blockquote `> `: một câu/đoạn ngắn ≤2 dòng cho một claim; không lặp nguyên danh sách bullet, đặt citation
  cuối bullet.
- Repair nhận query/context/draft cũ/`VerificationIssue(code, claim, detail)`, không nhận tài liệu mới hoặc
  chain-of-thought.
- Repair bỏ/viết lại claim lỗi, giữ claim có căn cứ; `AnswerGenerator` sở hữu prompt, pipeline chỉ điều phối.

## 5. Code hard gate (`output_check.py`)

- Python thuần deterministic → `HardGateResult(citations, hard_issues, warnings)`.
- Hard fail: `finish_reason == length`; citation ngoài `1..len(chunks)`; số nhạy cảm không có trong cả context
  lẫn query.
- Số nhạy cảm: tiền/tỷ lệ/thời hạn/tuổi/ngưỡng; nhận qua đồng/%/ngày/tháng/năm/giờ/tuổi.
- Evidence số gồm breadcrumb/content/raw_table và `query`; số user cung cấp không bị coi là claim mới.
- Chuẩn hóa chỉ biểu diễn (`4.960.000` ≈ `4 960 000`), không tính toán.
- Số chưa đủ rule chặn → `warning(unverified_number)`, vẫn qua Judge.
- Citation trả theo thứ tự xuất hiện; hard gate kiểm hình thức, Judge kiểm entailment.

## 6. Evidence Judge (`judge.py`)

- LLM evaluator độc lập, model `gpt-oss-20b`; input query/context đánh số/draft qua hard gate/citation map.
- `JudgeIssue`: `code` thuộc `unsupported_claim`, `citation_mismatch`, `missing_material_condition`,
  `context_insufficient`; `claim`, `detail`, `evidence_numbers`.
- `JudgeVerdict`: `verdict` thuộc `pass`, `repair`, `insufficient_evidence`; `issues`; structured
  JSON/Pydantic.
- Kiểm claim có evidence, citation đúng claim, điều kiện/ngoại lệ/phạm vi trọng yếu không mất.
- Không browse, kiến thức ngoài, viết answer hoặc chain-of-thought.
- `insufficient_evidence` nghĩa context thiếu, không khẳng định luật không tồn tại.
- **Fail-closed:** mạng/timeout/JSON/verdict/issue sai → không phát draft, `refusal(unable_to_verify)`.

## 7. Policy verification

- Hard gate + Judge pass → answer/citations/warnings/done.
- Hard fail/Judge repair, còn budget → regenerate một lần cùng evidence/issues.
- Hard fail/Judge repair hết budget, Judge lỗi/JSON sai → `refusal(unable_to_verify)`.
- Judge thiếu context → `refusal(insufficient_evidence)`.
- Generator/retrieval lỗi → `error` tương ứng.
- Refusal là hằng số; không trả draft một phần.

## 8. Cấu hình và module

- `langchain-openai.ChatOpenAI`, `base_url="https://api.groq.com/openai/v1"`; không `langchain-groq` vì pin
  `groq<1.0.0` xung đột `groq>=1.7.0` dùng bởi AsyncGroq.
- Generator: `astream()` → `GenerationDelta(text, finish_reason, usage)`.
- Judge/guardrail: `with_structured_output(Model, method="json_mode")`; parse lỗi → `JudgeError` fail-closed
  hoặc guardrail allow.
- Tham số Groq `reasoning_effort`, `include_reasoning`, `max_completion_tokens` qua
  `extra_body`/`model_kwargs`; giữ `LoopBoundClient` quanh `ChatOpenAI`.
- `GuardrailSettings`: safeguard-20b, `GROQ_API_KEY_1`, fail-open, không throttle chung;
  `GUARDRAIL_SYSTEM_PROMPT` theo mục 3; không allowlist document/địa danh.
- `GenerationSettings`: `model_name="openai/gpt-oss-120b"`, `api_key` ưu tiên `GROQ_API_KEY_3`, fallback key
  1; có `GROQ_API_KEY_4` thì `round_robin_api_key` bật hai `LoopBoundClient` xoay mỗi draft/repair qua
  `_next_client()`.
- `JudgeSettings`: 20b, `GROQ_API_KEY_2` fallback key 1, throttle `(model, key)`; `ThrottleTimeout` →
  fail-closed (conversation 12.1).
- Module: `models.py`, `generator.py`, `output_check.py`, `judge.py`, `guardrail.py`, `pipeline.py` (state
  machine/repair budget/policy/release).
- Cache chỉ có model generation: đổi model/prompt Judge phải bump `PROMPT_VERSION`.
- Conversation chỉ chuyển event, trace và cache answer đã duyệt.

## 9. Observability và acceptance

- Trace: query identity, chunk ID/corpus version, generator/Judge model/prompt version, hard issues, Judge
  verdict/issues, repair, finish reason, latency, usage.
- Không log chain-of-thought; áp chính sách PII khi lưu text.
- Acceptance: không phát draft trước hai gate; repair ≤1, không retrieve; citation sai/truncation/số thiếu
  evidence không đến user.
- Context thiếu/Judge lỗi/parse lỗi không lộ draft; mỗi stream đúng một `done`, đủ status.
- Regression: Sơn Nam/NĐ 293, Long An, gia hạn giấy phép allow; thông tin ngoài corpus thiếu evidence không có
  answer; chào hỏi/viết code out_of_scope; injection kèm số Nghị định vẫn injection.
- Unit test mock safeguard; regression dịch vụ thật chạy tay, không gọi Groq trong CI.
