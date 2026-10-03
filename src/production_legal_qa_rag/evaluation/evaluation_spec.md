# Evaluation — RAGAS: sinh golden testset (Phase 1) và chấm hệ thống (Phase 2): Reference Spec

- Giữ nguyên số mục, đặc biệt 3.x/4.x/11.x và các nhãn nghiệm thu/rủi ro được tham chiếu.
- Mốc trạng thái 2026-10-03: Phase 1 đã implement (PR #55/#60/#63/#64/#66), raw 203 → review luna → 157 mẫu keep; Phase 2 đã merge và chạy đủ 157 mẫu (MMR tắt), kết quả ở README và `data/eval/phase2/report.json`.
- Spec là contract triển khai; có code/checks không đồng nghĩa đã nghiệm thu bằng dịch vụ thật.
- Spec liên quan: [retrieval_spec.md](../retrieval/retrieval_spec.md), [generation_spec.md](../generation/generation_spec.md), [embedding_spec.md](../embedding/embedding_spec.md), [conversation_spec.md](../conversation/conversation_spec.md).

## 0. Bài học xương máu và phương pháp đúng

- Pilot Phase 1 trước chạy dài: adapt tiếng Việt, ép PERFECT_GRAMMAR, extractor khoảng 4000 token; mặc định từng sinh tiếng Anh/persona leak/request 12K bị 413.
- Một tầng sở hữu HTTP retry: router; ChatOpenAI/RunConfig max_retries=0, gán RunConfig trên llm.run_config.
- Router state dưới threading.Lock; chốt điểm bắt đầu một lần/call, không giữ lock khi mạng/sleep.
- Đo ký tự/token thật; corpus gốc 781007 ký tự, không byte; không tin hệ số ước lượng tuyệt đối.
- Ghi raw trước, progress sau; generation_progress.json là nguồn trạng thái.
- RAGAS 0.4.3 raise_exceptions=False không dùng cho sinh testset; vòng sinh riêng qua API công khai (3.3).
- Test ragas có thể bị CI skip; chạy eval venv, ví dụ ~/.cache/eval-venv-run/bin/python -m pytest tests/test_evaluation*.py.
- Không exc_info/logger.exception; _describe_traceback chỉ loại lỗi và file:dòng:hàm, không prompt/context.
- Kiểm phân bố từng unit ngay từ pilot; abstract=0 từng chỉ phát hiện lúc tổng kết.
- Phương pháp: văn bản gốc → unit Chương/Mục/KG riêng → checkpoint/resume nhiều ngày → dừng quota ngày → chấm reference, báo theo loại/văn bản; điểm là xu hướng.

## 1. Mục tiêu & phạm vi

- Phase 1: sinh question/reference/reference_contexts từ sáu văn bản data/markdown, không Langfuse/chat_turns hay chunk hệ thống.
- Chia Chương/Mục, mỗi unit `KnowledgeGraph` riêng: scope hẹp phù hợp retrieval, vừa quota, lỗi không mất cả corpus.
- Kế hoạch ban đầu GENERATE_SIZE=240: 192 single/24 abstract/24 specific, phân bổ theo ký tự; giữ synthesizer_name để báo riêng.
- Chín Groq accounts round-robin; checkpoint raw/KG sau mỗi unit; CLI generate/finalize.
- Options generate: --only, --testset-size, --reuse-knowledge-graph, --append, --retry-skipped, --dry-run.
- Kết quả đã chốt: raw 203, abstract=0; final 157 keep (142 single/15 specific), không random/cắt 180/sinh bù.
- Phase 1 không chạy pipeline/metrics thật, lọc chất lượng tự động bằng code, CI/cron/Postgres.
- Acceptance Phase 1: 157 mẫu review keep, đủ trường bắt buộc không rỗng, dùng thẳng Phase 2.

## 2. Input & Output

- Input: data/markdown/*.md, sáu văn bản.
- Các artifact Phase 1 tương đối với --output-dir:
  - golden_testset_raw.json: GoldenTestCase nối sau mỗi unit.
  - golden_testset_review.json/golden_testset_review_summary.md: review luna.
  - golden_testset.json: 157 keep, input Phase 2.
  - generation_progress.json: trạng thái; knowledge_graph/<văn bản>__<số>.json: KG.
  - units/ và units_plan.md: nguyên văn unit/kế hoạch chia.
- CLI hiện mặc định data/eval; dữ liệu đã tổ chức ở data/eval/phase1, units ở data/eval/units; chỉ định --output-dir/--testset đúng vị trí, không suy rằng CLI tự đổi mặc định.
- Phase 1 tạo user_input/reference/reference_contexts/synthesizer_name, code gắn source_document/source_section.
- Ba cột RAGAS EvaluationDataset bắt buộc: user_input/reference/reference_contexts; synthesizer_name là phụ.
- Phase 2 chỉ nhận user_input để tạo response/retrieved_contexts thật; không đưa reference làm input chatbot.

### 2.1 Vì sao đưa văn bản nguyên bản, không dùng chunk

- Không generate_with_chunks: chunker sai có thể bị ground truth che; hỏi từ chính chunk làm dense/BM25 dễ hơn thực tế.
- Corpus chunk: 2228 chunk, trung vị 41 token; content thiếu định danh luật/Điều vì breadcrumb là metadata.
- Chấp nhận reference_contexts RAGAS tự cắt bằng HeadlineSplitter(min_tokens=500), bao nhiều chunk hệ thống.
- Metric recall/precision LLM so với reference, không cần map reference_contexts sang chunk ID.

## 3. Công cụ

- RAGAS 0.4.3 là phiên bản đã xác minh API; giữ version resolve trong lock.
- API: `TestsetGenerator(llm, embedding_model)`, generate_with_langchain_docs, default_transforms, adapt_prompts/set_prompts.
- Generator: ChatOpenAI trỏ Groq → LangchainLLMWrapper; Document từ langchain_core.
- Embeddings KG: RagasEmbeddingsAdapter quanh InferenceClient.feature_extraction của embedding/hf_client.py, interface LangChain Embeddings; không word-segment ở Phase 1.
- `LangchainLLMWrapper`/`LangchainEmbeddingsWrapper` deprecated nhưng còn chạy ở 0.4.3; không tự nâng version.
- Eval dependency group: ragas>=0.4.3, langchain-community<0.4, rapidfuzz, langfuse>=4.15.6; rapidfuzz là dependency ragas dùng nhưng từng thiếu khai báo.
- Production group tách openai>=3.19.0; tool.uv.conflicts cấm sync cùng eval.
- Mọi lệnh eval: uv run --group eval --no-group production; xong uv sync để trả venv mặc định.

### 3.1 Round-robin 9 tài khoản Groq (`groq_round_robin.py`)

- GROQ_API_KEY_1…9; key 5–9 chỉ eval. Người dùng xác nhận chín tổ chức độc lập ngày 2026-10-01; key chung tổ chức không tăng quota.
- Hạn mức đã quan sát 120b/20b: RPM 30/RPD 1000/TPM 8000/TPD 200000; TPD là nút thắt.
- Request >8K có thể 413, không retry; giữ input khoảng ≤4–5K. Header không có TPD còn lại; retry-after trên 429.
- Script một process không dùng `LoopBoundClient`; `GroqRoundRobinChatModel(BaseChatModel)`: proxy N ChatOpenAI; _generate sync, _agenerate mặc định executor; Phase 2 gắn throttle (11.11).
- _plan_attempts dưới lock lấy start=next(cycle) một lần/call; thử (start+offset)%n, cooldown xếp cuối.
- Các lỗi không 429 theo policy 3.4, không xoay/retry mù; hết vòng raise lỗi cuối.
- Circuit breaker chỉ khi mọi credential còn hoạt động báo 429 ngày trong cùng call → DailyQuotaExhaustedError, không kế thừa RateLimitError, call sau fail ngay.
- Daily cooldown 300s; success xóa cooldown.
- Con trỏ/call_counts/cooldown/breaker/token/reasoning_effort đều dưới một threading.Lock.

### 3.2 Điều phối key mượt và tiết kiệm token (**đã implement**, PR #64)

- Giữ round-robin; không chọn key theo TPD còn lại vì không có số đo.
- **A — Cooldown phút:** retry-after, thiếu/hỏng mặc định 15s, kẹp 1–60s; mọi key cooldown phút thì chờ key sớm nhất, không giữ lock; clock/sleep inject cho test.
- **B — Token thật:** token_totals cộng prompt_tokens/completion_tokens/reasoning_tokens theo client dưới lock; UnitResult/UnitProgress lưu tokens/reasoning_tokens.
- Dry-run dùng (Σtokens − 4000·n)/Σchars khi ≥3 unit done có tokens>0; hệ số đo 3.94 token/ký tự, thay ước 5.5.
- **C — Reasoning:** chỉ dựng KG dùng `with router.reasoning_effort("low")`; _KG_REASONING_EFFORT tại ragas_runner.py.
- Adapt/persona/scenario/sample giữ setting riêng, không ép low bằng context dựng KG.

### 3.3 Giữ phần đã sinh khi lỗi giữa đơn vị (**đã implement**, PR #64)

- Không dùng TestsetGenerator.generate hủy cả batch; _generate_cases dùng API public.
- Sinh persona một lần/unit: generate_personas_from_kg(knowledge_graph, generator_llm, num_personas=3).
- Từng loại quota>0 tuần tự: generate_scenarios(n, knowledge_graph, persona_list) → generate_sample cho từng scenario, try/except, Semaphore(MAX_WORKERS).
- Test khóa chữ ký RAGAS bằng importorskip.
- Quota ngày: ngừng sample mới, giữ sample đã xong, unit partial, dừng process exit 1.
- Parse/timeout hết retry: bỏ sample, tăng skipped_samples, log tên synthesizer/loại lỗi; không có sample và có lỗi → UnitGenerationError.
- Ghi raw rồi progress cả partial; status done/partial/skipped.
- Resume partial dùng KG, quota thiếu=max(quota−questions,0), hoàn tất chuyển done.
- Append unit vốn done bị ngắt vẫn giữ done.

### 3.4 Retry/pass, fail-fast và báo suy giảm (**đã implement**, PR #66)

- Router là chủ sở hữu HTTP retry duy nhất; credential ready/minute_cooldown/daily_cooldown/disabled.
- 429 phút: cooldown rồi thử key khác; toàn vòng chỉ cooldown phút → RateLimitError, call sau chờ.
- 429 ngày: cooldown 5 phút; toàn bộ key hoạt động hết ngày cùng call → breaker/dừng/checkpoint.
- Timeout/connection/5xx/498: tối đa một retry với backoff+jitter, mỗi attempt một key ready.
- 400/401/403: không retry; 401/403 disable key. 413: không retry/đổi key.
- Lỗi sau HTTP 200: Clean*Synthesizer.prepare_combinations lọc persona_item_mapping/persona_concepts về persona thật.
- Semantic lỗi chưa biết: generate_personas/generate_scenarios/generate_sample chạy lại một lần, còn lỗi bỏ scope nhỏ nhất; không retry toàn unit dựng KG.
- Cause/context có DailyQuotaExhaustedError/APIStatusError/APIConnectionError/APITimeoutError/RateLimitError → không retry tầng RAGAS.
- Quota dương, không sample, có skipped_* → UnitGenerationError(stage=generation), unit skipped, giữ KG cho retry; không done rỗng.
- Systemic breaker: hai unit skipped liên tiếp cùng (stage, root exception type, HTTP status/null) → dừng trước unit kế, last_failure đã che, exit 1; unit có ≥1 sample reset đếm.
- Exit generate: 0 sạch, 1 quota ngày/breaker/unknown error, 2 input/config sai, 3 hoàn tất nhưng có unit/type/sample bỏ.
- GenerationReport tách skipped trong invocation và unit done từ trước; I/O/bug không phân loại phải fail-fast.

## 4. Workflow (`testset_generator.py`)

- Tạo chín ChatOpenAI → router/wrapper, gán RunConfig ngay → TestsetGenerator.
- Adapt tiếng Việt một lần/synthesizer, set prompts, ép PERFECT_GRAMMAR.
- Unit tăng dần ký tự, lọc --only; skip done trừ --append.
- Document → default_transforms (extractor khoảng 4000 token, max_workers khoảng 4) → KG low → lưu KG atomic ngay.
- Sinh mẫu (3.3/3.4) → gắn source → nối raw → ghi progress; xong luna review → finalize keep.
- split_document thuần: Chương `##`, lớn tách Mục `###`, gộp nhỏ, bỏ mở đầu/chú thích cuối; nhãn gộp nối “ + ”.
- allocate_questions thuần: 240→192/24/24, tỷ lệ ký tự/phần dư lớn nhất; multi-hop quota 0 thì chỉ single.
- Multi-hop cần ≥2 đoạn nên có thể thiếu quota, không bù; GENERATE_SIZE/MIN_UNIT_CHARS/MAX_UNIT_CHARS là hằng nội bộ.

### 4.1 Ngôn ngữ và chất lượng câu hỏi/đáp án

- Bắt buộc adapt_prompts(vietnamese) + QueryStyle.PERFECT_GRAMMAR; không trộn bốn query styles mặc định gây nhiễu/persona leak.
- Nhiễu để lát eval riêng ở phase sau.
- Reference có thể dính chữ do model (“cưtrú”): chấp nhận, không NFKC/không loại chỉ vì lỗi này.
- Review luna loại trùng ý/chủ đề nông như “Chính phủ quy định chi tiết”.

### 4.2 Chốt testset cuối (`finalize`) và sinh bù

- Quyết định 2026-10-01: final là raw có verdict keep trong review; 157=142 single+15 specific, tỷ lệ khoảng 9.5:1, giữ thứ tự raw.
- Không random/cắt 180/forced_fill/sinh bù; golden_testset_candidate.json 180 mẫu không dùng.
- Luna review 203 mẫu: keep/drop, quality 1–5, reason_code; drop 46: answer_unsupported 13, shallow_topic 12, mechanical 5, bad_multihop 5, duplicate 4, not_self_contained 3, language 3, transitional_clause 1.
- finalize_testset: kiểm số dòng review/raw, case_id khớp/unique, trường bắt buộc không rỗng; ghi keep nguyên văn raw, không TARGET_SIZE.
- case_id = 12 hex đầu SHA-256(user_input).
- Generate append/reuse KG/testset-size vẫn hỗ trợ sinh thêm theo --only, dedupe user_input y hệt; mẫu mới phải review lại, hiện không dùng.

### 4.3 Mã mẫu đã kiểm chứng bằng pilot

- cap_token_limit duyệt Parallel/list, đặt max_token_limit cho mọi LLMBasedExtractor.
- CleanSingleHop/MultiHopAbstract/MultiHopSpecificSynthesizer ghi đè prepare_combinations, styles=[PERFECT_GRAMMAR].
- Multi-hop lọc mapping về persona thật, tránh RAGAS 0.4.3 KeyError.
- asyncio.run adapt_prompts(vietnamese) → set_prompts, một lần/synthesizer.

### 4.4 Chia đơn vị và lập kế hoạch token

- Chuẩn Chương; >MAX_UNIT_CHARS=30000 tách Mục; <MIN_UNIT_CHARS=6000 gộp kề.
- Không bỏ “Điều khoản thi hành”; bỏ mở đầu trước Chương, _strip_footnotes từ `---` ngay trước `[1] …` cuối file.
- Key theo thứ tự unit, không La Mã vì có văn bản lặp Chương XI; văn bản lương tối thiểu không H2 dùng cả file.
- Ước lượng 5.5 token/ký tự + 4K cố định, thay hệ số thật theo 3.2.
- Kế hoạch: 50 unit (BHXH 12/BHYT 6/TNCN 3/lương tối thiểu 1/BLLĐ 16/điều kiện lao động 12), 720573 ký tự, min 6000/max 29075.
- Nguyên văn data/eval/units, kế hoạch units_plan.md (mục 2); chạy nhỏ → lớn.

### 4.5 Theo dõi tiến độ và chạy tiếp nhiều ngày

- Chạy lại cùng lệnh resume; thứ tự ký tự tăng dần; key `<tên .md>#<thứ tự 1-based>`.
- Progress atomic: units→UnitProgress, last_failure(unit,error đã che,at); không suy done từ raw trong luồng bình thường.
- Khôi phục crash giữa raw/progress: có unit raw chưa progress → coi xong, bổ sung progress/log theo cơ chế hiện tại.
- Quota ngày/breaker/unknown exception → last_failure, summary, dừng không unit kế, exit 1; lần sạch xóa failure.
- KG lưu sau apply_transforms; chưa done tự reuse nếu node DOCUMENT.page_content khớp nguồn.
- Done append chỉ reuse khi --reuse-knowledge-graph; buộc rebuild bằng xóa file KG tương ứng khi vận hành.
- --append/--retry-skipped phải kèm --only; thiếu exit 2, không LLM; skipped không tự retry.
- Done/skipped chars khác split hiện tại → dừng nêu unit; progress hỏng → lỗi rõ, không coi chưa chạy.
- Dry-run không token: unit/chars/ước token/trạng thái xong/dở N/M/skipped stage/chưa theo thứ tự chạy.
- Chạy nền, điều chỉnh output-dir nếu dùng phase1:

```bash
setsid nohup uv run --group eval --no-group production tools/generate_testset.py generate > data/eval/generate.log 2>&1 &
```

### 4.6 Tiến độ thực tế và việc còn lại

- Mốc 2026-10-01: 49/50 unit, raw 203=180 single+23 specific, abstract 0.
- `MultiHopAbstractQuerySynthesizer` thiếu cụm trong KG nhỏ là nghi vấn chưa xác minh; người dùng chấp nhận abstract=0, không bù, ghi report.
- `Văn bản hợp nhất bộ luật lao động.md#5` skipped bỏ hẳn; corpus BLLĐ còn unit khác; final 157 keep đã sinh.
- Finalize theo review đã sửa trên nhánh Phase 2 (4.2/11.12); raw tiếng Anh đã dịch PR #71.
- Công việc còn lại ở mốc này: nghiệm thu/chạy Phase 2, không sinh lại Phase 1.

## 5. Model dữ liệu (`models.py`)

- GoldenTestCase: user_input/reference/reference_contexts:list[str], synthesizer_name/source_document/source_section tùy chọn; không RetrievedChunk.
- Source do code gắn, dùng resume/báo theo văn bản/Chương.
- UnitProgress: title/chars/estimated_tokens, questions:dict[str,int], llm_calls/seconds/completed_at.
- status done|partial|skipped (default done); skipped_samples=0; skipped_question_types:set[str] mặc định rỗng; skipped_stage knowledge_graph|generation|None.
- error_type:str|None, attempts=0, tokens/reasoning_tokens:int|None; metadata lỗi không chứa prompt/exception message.
- UnitFailure(unit,error,at); GenerationProgress(units:dict[str,UnitProgress],last_failure:UnitFailure|None).

## 6. Config (`config.py`)

- TestsetGeneratorSettings: `SettingsConfigDict(env_file=".env", extra="ignore")`, api_key…api_key_9 aliases GROQ_API_KEY_1…9, min_length=1.
- Cả chín key bắt buộc; thiếu/rỗng → EvalInputError bọc Pydantic, chỉ tên biến, exit 2.
- `model_name="openai/gpt-oss-120b"`, `timeout_seconds=60`; class riêng, không dùng GenerationSettings.
- Embedding adapter reuse EmbeddingSettings; module không trực tiếp đọc .env; example có chín tên key.

## 7. Module (`src/production_legal_qa_rag/evaluation/`)

- models.py/corpus_loader.py/embeddings_adapter.py; groq_round_robin.py router/throttle.
- unit_splitter.py: EvalUnit/split_document/split_directory, thuần Python, không ragas.
- ragas_runner.py: RagasUnitRunner, cap_token_limit, Clean*Synthesizer, build_run_config, vòng sinh; điểm tích hợp ragas Phase 1.
- testset_generator.py: orchestration/allocate_questions/breaker/finalize/dry-run/_describe_traceback.
- tools/split_eval_units.py: unit/plan; tools/generate_testset.py: Typer generate/finalize.
- Phase 2 thêm module cùng package (11.7).

## 8. Xử lý lỗi

- Mọi key hoạt động hết ngày → dừng/checkpoint partial/last_failure, không gọi unit mới.
- 429 phút/timeout/connection/5xx/498 → router retry 3.4; hết retry bỏ scope RAGAS nhỏ nhất.
- 400/401/403/413/semantic sau HTTP 200 → policy 3.4; KG lỗi phân loại → skipped.
- Hai skipped cùng signature hoặc unknown exception → exit 1; OSError/TypeError/bug không được nuốt thành skipped.
- Hoàn tất có skipped unit/type/sample → giữ raw/checkpoint, summary, exit 3.
- Only sai/Markdown thiếu hoặc không UTF-8/append thiếu only/key thiếu → input error exit 2 trước LLM, không traceback secret.
- Embedding response sai → raise, không vector rỗng.
- Done không append → skip; chars lệch → dừng; raw chưa progress → phục hồi 4.5; progress hỏng → raise.
- Finalize review thiếu/lệch ID/line count hoặc keep field rỗng → nêu dòng lỗi, không ghi file thiếu.

## 9. Nghiệm thu thủ công

- **Ca 1:** generate với eval group, --only/dry-run; resume nhiều ngày theo quota.
- **Ca 2:** raw bao sáu văn bản, đủ trường không rỗng.
- **Ca 3:** dashboard/log lượt/tokens phân bổ khoảng 1/9 account; cooldown/breaker đúng, không giả thành công khi hết quota.
- **Ca 4:** không duyệt tay; luna review toàn 203, rủi ro 10.14.
- **Ca 5:** finalize cho đúng 157 keep, unique case_id, nguyên văn raw.

## 10. Rủi ro / điểm mở

- **1:** embedding adapter mới cần nghiệm thu thật cho KG.
- **2:** synthetic lệch câu hỏi user; **3:** resume unit/sample, không call LLM; **6:** shared router state thread-safe khi max_workers=4.
- **8:** hit@k cần map reference_contexts→chunk nhiều-một; không làm Phase 2.
- **9:** reference chưa được người kiểm luật; điểm là khớp đáp án LLM, bất thường thì kiểm reference với nguồn gốc.
- **10:** multi-hop nhiều Điều yếu; báo riêng, single là chính, không abstract.
- **11:** unit gộp nhiều Chương (BLLĐ #16 XV+XVI+XVII) có thể không đồng nhất; **12:** chấp nhận dính chữ reference.
- **14:** luna giữ 157 (12 quality 2/49 quality 3), không người đối chiếu; 16/23 multi-hop raw chỉ lưu một context, không kiểm được evidence hop hai từ dữ liệu lưu sẵn.
- Điểm chỉ xu hướng; kiểm case thấp và quality review trước kết luận.

## 11. Phase 2 — Chạy pipeline thật và chấm điểm (code implement 2026-10-01, đã chạy đủ 157 mẫu với MMR tắt)

### 11.1 Mục tiêu, phạm vi

- Chạy 157 case qua retrieval/generation thật; so MMR recall, đo answer sau Judge theo loại/văn bản.
- Bốn metric chuẩn: context_recall S4 cả hai config; context_precision S4b chỉ config chọn; faithfulness/answer_relevancy S6.
- Workflow: HyDE → embed → retrieve on/off tuần tự → recall → user chọn MMR → generate → answer metrics → precision → report.
- Không guardrail/condense/cache/API/conversation orchestration; không đo tỷ lệ guardrail chặn sai.
- Không Groq Batch/hit@k/nhiễu/Langfuse sampling/CI/cron/factual_correctness/answer_correctness/pilot bắt buộc.
- Chấp nhận chỗ hở answer bám chunk nhưng thiếu ý; acceptance đủ stage/157 case/report.json và các lát báo cáo.

### 11.2 Stage, file trung gian, resume

- Output mặc định data/eval/phase2; record key case_id; stage đọc input trước/ghi JSONL riêng.
- **S1 hyde:** 20b, pool chín key → hyde.jsonl.
- **S2 embed:** QueryEmbedder/PyVi cùng model index, 25 text/request HF → embeddings.jsonl, không commit.
- **S3 retrieve:** một pipeline, mmr_on rồi mmr_off tuần tự → retrieved_mmr_on.jsonl/retrieved_mmr_off.jsonl.
- **S4 score-recall:** 120b/router/throttle, hai config ≤314 lượt → recall_scores.jsonl; cùng danh sách `chunk_id` theo cùng thứ tự thì reuse/reused_from.
- **S5 generate --config:** 120b draft +20b Judge, key pool/throttle → answers.jsonl.
- **S6 score-answers:** answered-only, faithfulness+answer_relevancy ≤785 lượt → answer_scores.jsonl.
- **S4b score-precision:** config chọn, một call/chunk, 5/case tối đa 785 → precision_scores.jsonl.
- **Report:** không LLM → report.json/bảng terminal.
- Thứ tự S1→S2→S3 → S4 → chọn MMR→S5→S6→S4b; metric 120b chạy lần lượt, precision cuối nếu thiếu quota.
- Đổi prompt generation chạy lại S5/S6/S4b; phải invalidation checkpoint phù hợp, không tự coi case cũ đã mới.
- Resume skip case đã có; thiếu upstream thì báo số thiếu và chỉ xử lý phần có.
- Một nơi ghi duy nhất/lock; scoring batch SCORING_BATCH_SIZE=10; bỏ dòng cuối hỏng + log.
- --testset mặc định data/eval/golden_testset.json, dữ liệu phase1 phải truyền path; có thể dùng raw kiểm pipeline; --limit tùy chọn; status đếm xong/lỗi/chưa xử lý.
- Parse/NaN/timeout/5xx/429 phút lẻ → record error, đi tiếp; chỉ retry khi --retry-failed, giữ success.
- Quota ngày → dừng, phần chưa làm không ghi error; exit 0 sạch/1 quota hoặc stage lỗi/2 input/config.

### 11.3 Chi tiết từng stage

- S1/S5: một queue chung, chín async worker; mỗi worker xử lý tuần tự với key cố định, không sửa production rotation.
- Dựng `HydeSettings`/`GenerationSettings`/`JudgeSettings` bằng validation_alias; S5 đặt `GROQ_API_KEY_4=None` để không trộn bucket.
- Worker hết quota ngày dừng; worker còn capacity tiếp tục; case chưa xong giữ pending.
- S1 reuse HydeGenerator/throttle; None không lỗi → hypothetical_document=null, bỏ A đúng production; phân biệt error với null hợp lệ.
- S2 batch hypo/query cùng thứ tự; hai config dùng chung embeddings.
- S3 `retrieve(query, use_mmr=…, precomputed=…)` (retrieval 2), một `RetrievalPipeline`/RETRIEVE_CONCURRENCY=1, hết on mới off vì GPU 2GB.
- S3 OOM: giảm LocalReranker batch_size rồi retry-failed; không đổi model/max_length.
- S3 fallback `rerank_score is None` là error, không chấm; rỗng no_context; RetrievalError thành error.
- S4 LLMContextRecall cần user_input/retrieved_contexts/reference; một call phân claim reference theo evidence, tỷ lệ entailment 1/0.
- Context mỗi chunk cùng breadcrumb/content/raw_table như build_context generation; dùng hàm chung nếu cần.
- S4 wrapper LangchainLLMWrapper(GroqRoundRobinChatModel), cùng RunConfig; ID sequence hai config giống → chấm config đầu, config kia chép score/reused_from.
- User chọn MMR từ trung bình recall và case thắng/thua/hòa; MMR đổi candidate coverage, reranker quyết thứ hạng cuối.
- Chênh chưa rõ → ưu tiên off (đơn giản/bớt Pinecone call), ghi kết luận retrieval mục 6; chưa tự chốt trước số đo.
- S4b LLMContextPrecisionWithReference: một verdict 0/1 cho mỗi chunk rồi average precision@k; không tự gộp năm chunk/call, không dùng chọn MMR.

### 11.4 S5 — Generation (phương án B)

- Dùng nguyên GenerationPipeline.generate(query,chunks); không guardrail/retrieve, giữ hard gate/Judge/repair ≤1.
- Chín pipeline độc lập, inject generator/Judge; cùng key i cho 120b/20b, hai bucket.
- Mỗi case: draft→hard gate→Judge; repair nếu cần qua lại hai gate; khoảng 2–4 lượt/case, 314–628 tổng, repair rate chưa đo.
- Wrapper AnswerGenerator draft/repair acquire throttle trước, settle usage; estimate prompt/CHARS_PER_TOKEN + EXPECTED_COMPLETION_TOKENS, không max_completion_tokens.
- EvidenceJudge vốn throttle; wrapper phải đổi ThrottleTimeout thành lỗi status 429 mà _is_rate_limited nhận để lưu error retry được, tránh biến thành refusal hợp lệ.
- AnswerRecord: case_id/config/outcome/response thô `[n]`/citations/repair_used/warning_codes/error_code/usage/prompt_version.
- Outcome answered/insufficient_evidence/unable_to_verify/error; prompt version lấy `PROMPT_VERSION` generation (hiện v11).
- Refusal là kết quả hợp lệ, không retry; chỉ error retry. Báo refusal/end-to-end riêng vì RAGAS chỉ chấm answered.
- Faithfulness của answer đã qua Judge có thể cao hơn draft.

### 11.5 S6 — Chấm câu trả lời

- Faithfulness: hai calls, tách claim rồi kiểm toàn claim với context.
- ResponseRelevancy, tên báo answer_relevancy: strictness=3, ba lượt sinh câu hỏi/cờ noncommittal, mean cosine với query.
- Không n>1: wrapper không hỗ trợ và Groq không dùng; tổng năm lượt/answered case, ≤785 lượt.
- Chỉ strip marker `[n]` thuộc citations hợp lệ record, giữ marker khác; answers.jsonl giữ response thô.
- Relevancy embedding RagasEmbeddingsAdapter(segment=True) dùng ViTokenizer; default False giữ Phase 1; cùng không gian model (embedding 4).

### 11.6 Báo cáo (`report.json`)

- Retrieval: trung bình recall cả hai, tổng/theo synthesizer_name/source_document; chỉ case hợp lệ ở cả hai, báo excluded.
- Thắng/thua/hòa theo case; reused_from tính hòa, báo số cùng ID sequence; precision config chọn theo cùng lát.
- Answer: trung bình faithfulness/relevancy trên answered; n/null mỗi lát; NaN loại khỏi trung bình, không coi 0.
- End-to-end: refusal insufficient_evidence/unable_to_verify tính 0 mỗi answer metric; error loại cả hai cách, báo số.
- Vận hành: tỷ lệ outcome/repair, HyDE null, retrieval error/fallback, prompt_version.
- Hai loại câu single/specific, không abstract; single chính, specific n=15 chỉ xu hướng.
- Note cố định: reference LLM chưa kiểm luật (10.9), judge cùng họ generator, không điểm đúng luật tuyệt đối.

### 11.7 Module

- run_models.py: case_id/record Pydantic/EvalConfig; jsonl_store.py: validate/read/append/tail lỗi/completed IDs/một writer.
- key_pool.py: settings/key queue/worker/throttle wrappers; hyde_stage.py/embed_stage.py/retrieve_stage.py/generate_stage.py: orchestration.
- scoring.py: điểm tích hợp ragas Phase 2, reuse wrapper/RunConfig; report.py thuần.
- tools/run_eval.py: hyde/embed/retrieve/score-recall/generate/score-answers/score-precision/report/status.
- Options --testset/--output-dir/--limit/--retry-failed/--workers (default 9); --config mmr_on|mmr_off bắt buộc S5/S6/S4b.
- Thay đổi liên quan đã chốt: retrieval PrecomputedQuery/precomputed; embedding adapter segment; router acquire/settle theo model/key; finalize review; gitignore phase2 embeddings.
- Eval group có Langfuse để import code generation/HyDE; không sync production cùng eval.

### 11.8 Ước lượng chi phí (thô, **chưa đo**)

- 157 case: S1 157 lượt 20b khoảng 0.1M token; S4 ≤314 lượt 120b ×1.5–2K, khoảng ≤0.5M.
- S5 draft 157–314 ×3–4K, khoảng 0.5–1.1M token 120b + Judge 157–314 lượt 20b; S6 ≤785 khoảng 0.8M; S4b 785 ×0.7–1K khoảng 0.55–0.8M.
- Tổng 120b khoảng 2.4–3.2M so capacity khoảng 1.8M/ngày (9×200K), ước ít nhất hai ngày lịch; runtime từng ước 1.5–2h, TPD nút thắt.
- Dự kiến ngày 1 S1–S4/chọn MMR/bắt đầu S5, ngày 2 xong S5/S6/S4b; TPD hồi theo cửa sổ, resume case_id; đây chưa là số đo.

### 11.9 Rủi ro / điểm mở

- **1:** eval OpenAI cũ hơn production; S1/S5 dùng `AsyncGroq`/`with_structured_output(method="json_mode")` cần nghiệm thu, khuyến nghị S5 đầu `--limit 2`; lệch thì tách venv cho S1–S3/S5.
- **2:** Judge lọc trước/model cùng họ/reference chưa kiểm → điểm lạc quan; **3:** 157 case, specific 15, báo thắng/thua thay kết luận từ chênh nhỏ.
- **5:** chưa ngưỡng đạt/tần suất; lần đầu baseline; **6:** GPU 2GB/CUDA OOM bỏ case làm lệch so sánh, phải báo excluded/retry.
- **7:** không đo guardrail; **8:** bucket model độc lập là quan sát dự án, tài liệu không nói rõ 20b/120b độc lập; chín tổ chức đã xác nhận.
- **9:** router Phase 1 thêm throttle phải test round-robin/cooldown/breaker trong eval venv.
- **10:** reuse S4 ID sequence đã chốt; không đổi judge S4 sang 20b.
- **11:** error khoảng >10–15% → kiểm throttle trước retry hàng loạt/đốt TPD; **12:** kiểm regex strip citation trên dữ liệu thật.

### 11.10 Không pilot bắt buộc (người dùng chốt 2026-10-01)

- Không bắt pilot Phase 2; --limit vẫn tùy chọn, S5 đầu `--limit 2` là khuyến nghị nghiệm thu venv.
- Token/repair/ngày/quota, relevancy embedding/OOM chỉ xác nhận khi chạy thật; không biến 11.8 thành kết quả.

### 11.11 Song song và rate limit

- Giữ cả parallelism và limiter; lỗi không đoán trước thì đánh dấu/đi tiếp, không bỏ limiter.
- Không gate cứng một request/phút/key; lượt 0.5–4K cần throttle theo token.
- `TokenWindowThrottle/get_throttle` (retrieval/llm_throttle.py): sliding 60s TPM8000/RPM30 ×0.9 cho model/key; `acquire(estimated_tokens, max_wait_seconds)` trước gửi, `settle(reservation, actual_tokens)` theo usage; eval max_wait rộng vài phút.
- S1: chín worker/queue, HyDE throttle 20b; S2 HF 25 text/batch; S3 tuần tự GPU.
- S4/S4b/S6: RAGAS workers/router chín key, router throttle; S5 chín worker, draft/repair wrapper + Judge throttle.
- Quota ngày: worker/breaker theo 3.1/11.3, dừng scope hết capacity, pending để resume; parse/NaN/timeout/connection/5xx/429 phút → error; 400/413 không retry mù.
- Chỉ S3 GPU được chạy chồng S4 Groq; S4/S4b/S5/S6 cùng 120b không chồng.
- Throttle in-process, hai CLI process không thấy ngân sách nhau, có thể tranh TPM gây 429.

### 11.12 Kế hoạch implement và mặc định đã chốt

- Đã merge vào main (không còn nhánh feat/gen-testset-ans): module 11.7 + finalize review 4.2; testset 157 đã có, không chặn Phase 2.
- Default workers 9, config bắt buộc S5/S6/S4b, max_wait throttle rộng; estimate draft/repair là hằng nội bộ.
- Kiểm local 2026-10-01, không dịch vụ thật: evaluate(raise_exceptions=False) trả NaN, checkpoint thành error; GenerationPipeline/HyDE chạy fake trong eval venv.
- Đã smoke resume/JSONL/reuse recall/quota ngày/refusal-error/strip citation/acquire-settle; finalize raw/review thật ra 157.
- Còn kiểm thật: venv S1/S5, strip citation/embedding relevancy/OOM; S5 `--limit 2` khuyến nghị theo 11.9.1.
