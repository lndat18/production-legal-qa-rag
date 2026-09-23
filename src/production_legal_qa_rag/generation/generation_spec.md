# Generation — Evidence-verified answer: Reference Spec

## 1. Mục tiêu

Tạo câu trả lời pháp luật chỉ được phát sau khi đã kiểm chứng. Nó chỉ dựa trên
query và tối đa năm RetrievedChunk đã rerank; câu trả lời phải có citation hợp lệ,
không bị cắt, và không tự thêm mức tiền, tỷ lệ, thời hạn hay điều kiện áp dụng
quan trọng.

generation/ là package stateless. conversation/ sở hữu history, admission, cache
và policy cấp sản phẩm; retrieval/ tìm evidence. Generation không tự gọi retrieve
lại khi kiểm tra hay repair.

Nguyên tắc:

1. Context được chốt đúng một lần/request. Draft, hard gate, Judge và repair cùng
   đọc query + context đó.
2. Không phát draft trước khi qua tất cả gate. Event token chỉ mang answer đã duyệt.
3. Code kiểm tra điều xác định được; Judge kiểm tra claim theo nghĩa. Judge không
   xác nhận luật ngoài context và không thay luật sư.
4. Có tối đa **một repair tổng cộng/request**, không phải một repair cho mỗi gate.

Không làm: multi-turn, HyDE/query rewrite/retrieve lần hai, nguồn bên ngoài, xác
nhận hiệu lực pháp lý ngoài corpus, HTTP/SSE, cache, RAGAS runtime, moderation
tổng quát hay human review queue.

## 2. Contract công khai

- answer_stream(query): guardrail input → retrieve → generation verified.
- generate(query, chunks): dùng khi caller đã có context cố định.
- chunks: list[RetrievedChunk], theo rerank, 1 <= len(chunks) <= 5.

answer_stream() inject được dependency retrieve; generate() không gọi retrieve.
Nhờ vậy context và số citation [n] của repair luôn audit được.

GenerationEvent là discriminated union Pydantic v2. Mọi nhánh kết thúc bằng done.

| Event | Dữ liệu | Nghĩa |
|---|---|---|
| status | stage: guardrail, retrieval, drafting, verification, repairing | Tiến độ, không chứa draft. |
| token | text | Mảnh của **answer đã pass**; ghép các mảnh phải ra đúng answer đã duyệt. |
| citations | list[Citation] | Citation hợp lệ có trong answer đã phát. |
| warning | code, message, detail | Tín hiệu mềm đã pass policy. |
| refusal | reason, message | Message cố định, không lộ draft. |
| error | code, message, retry_after_seconds | Không thể tạo/tra cứu câu trả lời. |
| done | usage | Event cuối cùng. |

Thứ tự: status* → (token+ → citations → warning* | refusal | error) → done.

RefusalEvent.reason gồm out_of_scope, injection, insufficient_evidence,
unable_to_verify. Hai reason verification do pipeline map từ kết quả verifier;
Judge không tự viết phản hồi cho user.

Buffer làm time-to-first-token tăng thêm thời gian tạo draft và verification. UI
phải hiện status(verification). Sau khi pass, pipeline có thể chia answer thành
nhiều token event, nhưng không được thêm delay nhân tạo.

## 3. Workflow đã chốt

~~~mermaid
flowchart TD
    A[Query + retrieved context cố định] --> B[LLM tạo draft vào buffer]
    B --> C[Code hard gate]
    C -->|Hard fail, chưa repair| D[LLM viết lại: cùng query + context + lỗi]
    D --> C
    C -->|Hard fail, đã repair| X[Từ chối an toàn]
    C -->|Pass| E[LLM Judge: claim so với context]
    E -->|pass| F[Phát answer đã duyệt]
    E -->|repair, chưa repair| D
    E -->|repair, đã repair| X
    E -->|insufficient evidence| X
    F --> G[citations + done]
    X --> G
~~~

Quy tắc state machine:

1. Input guardrail chạy trước retrieval và giữ fail-open khi provider lỗi. Đây là
   bảo vệ availability, không phải evidence verification.
2. Retrieval chạy đúng một lần. Context rỗng phát error(no_context), không tạo
   draft.
3. Provider có thể stream nội bộ để lấy usage và finish reason, nhưng draft chỉ ở
   buffer. Lỗi transport, rate limit hoặc draft rỗng phát error, không repair.
4. Hard gate fail thì LLM viết lại **toàn bộ** answer với cùng query + context +
   issue. Không vá citation riêng, không gọi HyDE/retrieve.
5. Chỉ draft qua hard gate mới vào Judge. Judge trả JSON pass, repair, hoặc
   insufficient_evidence.
6. repair từ bất kỳ gate nào dùng cùng một repair budget. Bản repair phải quay lại
   hard gate và Judge. Hết budget thì refusal(unable_to_verify).
7. Pipeline map insufficient_evidence thành refusal(insufficient_evidence).
   Judge đánh giá evidence; pipeline sở hữu quyết định sản phẩm và message user.

Không có retrieve_repair. Nếu cần re-retrieval vì thiếu evidence, đó là workflow
agentic riêng với giới hạn và audit riêng.

## 4. Draft và repair (generator.py)

Context đánh số một-based:

~~~text
[1] {breadcrumb}
{content}
~~~

Khi có bảng, thêm raw_table nguyên trạng. Prompt generator phải:

- chỉ dùng context đánh số;
- đặt citation [n] ASCII ngay sau khẳng định pháp lý;
- giữ nguyên số, mức tiền, tỷ lệ, thời hạn; không tự tính hay suy diễn số mới;
- giữ điều kiện áp dụng quan trọng, không trộn các trường hợp;
- nói rõ evidence thiếu thay vì dùng kiến thức ngoài context;
- coi query/context là dữ liệu, không phải chỉ dẫn hệ thống.

Repair không nhận tài liệu mới. Nó nhận query, context đánh số, draft cũ và
VerificationIssue không chứa chain-of-thought, ví dụ:

~~~json
{
  "code": "citation_mismatch",
  "claim": "Người lao động luôn được nghỉ 12 ngày.",
  "detail": "Nguồn [1] chỉ áp dụng khi đủ điều kiện X."
}
~~~

Prompt repair yêu cầu bỏ hoặc viết lại claim lỗi, giữ claim có căn cứ, không tranh
luận với issue. AnswerGenerator sở hữu prompt draft/repair; pipeline.py không tự
dựng prompt Groq.

## 5. Code hard gate (output_check.py)

Hard gate là Python thuần, deterministic, trả
HardGateResult(citations, hard_issues, warnings). Nó không thay Judge.

Hard fail khi:

- finish_reason == length (truncated);
- citation [n] nằm ngoài 1..len(chunks);
- số nhạy cảm không tìm thấy sau chuẩn hoá trong breadcrumb, content hoặc raw_table
  của context. Số nhạy cảm là mức tiền, tỷ lệ, thời hạn, tuổi và ngưỡng định lượng
  pháp lý; nhận diện qua đơn vị như đồng, %, ngày, tháng, năm, giờ, tuổi.

Chuẩn hoá chỉ so khớp biểu diễn như 4.960.000 và 4 960 000; không chứng minh số
được dùng đúng điều kiện. Số không nhạy cảm chưa đủ rule để block là
warning(unverified_number) và vẫn qua Judge.

Citation hợp lệ về chỉ số chưa chứng minh nó hỗ trợ claim. Hard gate chỉ lấy
Citation theo thứ tự xuất hiện; Judge kiểm tra entailment.

## 6. Evidence Judge (judge.py)

Judge là LLM evaluator độc lập với generator, dùng structured JSON/Pydantic:

~~~python
class JudgeIssue(BaseModel):
    code: Literal[
        "unsupported_claim",
        "citation_mismatch",
        "missing_material_condition",
        "context_insufficient",
    ]
    claim: str
    detail: str
    evidence_numbers: list[int] = []


class JudgeVerdict(BaseModel):
    verdict: Literal["pass", "repair", "insufficient_evidence"]
    issues: list[JudgeIssue] = []
~~~

Input duy nhất: query, context đánh số, draft qua hard gate, citation map. Judge
kiểm tra:

1. Claim pháp lý trọng yếu có được context hỗ trợ.
2. Citation thực sự hỗ trợ claim đứng cạnh nó.
3. Draft có làm mất điều kiện, ngoại lệ hay phạm vi làm đổi kết luận.
4. Context có đủ để trả lời hay draft chỉ có vẻ thuyết phục.

Judge không browse, không dùng knowledge ngoài context, không viết câu trả lời,
không xuất chain-of-thought, và không khẳng định luật đúng/sai ngoài corpus.
insufficient_evidence nghĩa là context retrieved không đủ, không phải luật không
tồn tại.

Trước enforce, Judge phải được hiệu chỉnh trên case có nhãn người duyệt. Khi
enforce, lỗi mạng, timeout, JSON sai hoặc verdict/issue không hợp lệ là fail-closed:
không phát draft và refusal(unable_to_verify). Shadow mode chỉ phục vụ hiệu chỉnh,
không phải chế độ production an toàn cho workflow này.

## 7. Policy verification

| Kết quả | Hành động |
|---|---|
| Hard gate pass + Judge pass | Phát answer, citations, warning mềm (nếu có), done. |
| Hard fail/Judge repair, còn budget | Regenerate một lần với context và issue cũ. |
| Hard fail/Judge repair, hết budget | refusal(unable_to_verify), done. |
| Judge insufficient_evidence | refusal(insufficient_evidence), done. |
| Judge lỗi/JSON sai | refusal(unable_to_verify), done. |
| Generator/retrieval lỗi | error tương ứng, done. |

Message refusal là hằng số. Không trả draft một phần và không để generation tự quyết
định từ chối sau verdict thiếu evidence.

## 8. Cấu hình và module

Dùng langchain-openai (`ChatOpenAI`) trỏ vào endpoint OpenAI-compatible của Groq
(`base_url="https://api.groq.com/openai/v1"`, cùng `api_key` Groq hiện có), thay
AsyncGroq thô, để rút boilerplate client/parse JSON:

KHÔNG dùng langchain-groq: mọi version của nó pin `groq<1.0.0`, xung đột trực tiếp
với `groq>=1.7.0` mà `conversation/condenser.py`, `formatting/llm_client.py` và
`retrieval/hyde.py` (ngoài phạm vi generation/) đang dùng qua AsyncGroq. Groq công
bố chính thức endpoint OpenAI-compatible này nên `ChatOpenAI` trỏ vào đó vẫn là
tích hợp thật, không phải workaround; đồng thời tránh đổi version `groq` toàn repo.

- generator.py: `ChatOpenAI.astream()` thay `chat.completions.create(stream=True)`.
  GenerationDelta vẫn map từ chunk trung gian như hiện tại (text, finish_reason,
  usage) — không đổi hợp đồng generator/pipeline.
- judge.py, guardrail.py: `ChatOpenAI.with_structured_output(PydanticModel,
  method="json_mode")` thay `response_format={"type": "json_object"}` +
  `model_validate_json` tay. Parse lỗi/JSON sai vẫn phải map về JudgeError
  (judge.py) hoặc verdict allow fail-open (guardrail.py) như cũ — hành vi
  fail-closed/fail-open ở mục 6 và guardrail không đổi.
- Tham số riêng của Groq (`reasoning_effort`, `include_reasoning`,
  `max_completion_tokens`) truyền qua `extra_body`/`model_kwargs` của ChatOpenAI;
  không đổi giá trị các tham số này.
- Vẫn giữ pattern LoopBoundClient bọc quanh instance `ChatOpenAI` (không đổi qua
  event loop), do ChatOpenAI cũng giữ AsyncOpenAI client nội bộ với cùng giới hạn
  nêu ở loop_bound.py.
- Đây là đổi thư viện gọi model, không đổi workflow, event, hard gate hay policy ở
  mục 1-7. Pydantic v2 vẫn là nguồn sự thật cho mọi schema.

- GuardrailSettings: input safeguard, fail-open.
- GenerationSettings: generator, ưu tiên GROQ_API_KEY_2.
- JudgeSettings: model, timeout, retry, key qua pydantic-settings; model phải
  đổi được bằng config. Judge/generator không share client state qua event loop.

| Module | Trách nhiệm |
|---|---|
| models.py | Pydantic events, hard-gate issue/result, Judge verdict/issue. |
| generator.py | Context/prompt, buffer draft hoặc repair, text + finish reason + usage. |
| output_check.py | Citation, chuẩn hoá số và deterministic hard gate. |
| judge.py | Evaluator prompt, structured parsing, fail-closed error boundary. |
| guardrail.py | Admission input, không kiểm tra grounding. |
| pipeline.py | State machine, repair budget, policy map, phát approved answer. |

Judge model/version và prompt version là một phần trace và cache identity. Đổi model,
prompt hoặc shadow/enforce phải invalidate cache ở cache/; không replay answer chưa
được Judge duyệt. conversation/ chỉ chuyển event, record trace và cache answer
approved; không diễn giải verdict hay verify lần hai.

## 9. Observability và acceptance

Trace mỗi request gồm query identity, chunk_id/corpus version, generator/Judge
model + prompt version, hard issues, Judge verdict/issues, repair used, finish
reason, latency từng stage và usage. Không log chain-of-thought; áp dụng chính sách
PII trước khi lưu text.

Implementation đạt spec khi:

1. Không token draft nào phát trước hard gate và Judge pass.
2. Repair không gọi retrieval và không quá một lần/request.
3. Citation ngoài context, output bị cắt, số nhạy cảm thiếu evidence không đến user
   dưới dạng answer.
4. Context thiếu, Judge lỗi hoặc parse lỗi đều không làm lộ draft.
5. Event stream luôn có đúng một done, đủ status để UI diễn giải thời gian chờ.
