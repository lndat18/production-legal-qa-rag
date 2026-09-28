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

- Đọc toàn bộ 6 file `.md` trong `data/markdown/` thành tài liệu LangChain.
- Build `KnowledgeGraph` (đồ thị tri thức nội bộ của ragas: tóm tắt, trích entity, dựng
  quan hệ giữa các đoạn văn bản) và sinh `testset_size = 360` câu hỏi single-hop/multi-hop
  (180 single-hop / 90 multi-hop abstract / 90 multi-hop specific theo đúng tỷ lệ mặc định
  50/25/25 của ragas — số chia đẹp, không phải số ước lượng).
- Luân phiên (round-robin) 3 tài khoản Groq độc lập cho `generator_llm` để rải tải gọi LLM
  khi build KG + sinh 360 câu (mục 3.1) — **quyết định có chủ đích của người dùng**, đánh
  đổi lấy khối lượng lớn hơn hẳn con số mặc định nhỏ (30) ban đầu được đề xuất.
- Lưu testset ra file JSON, lưu `KnowledgeGraph` ra file riêng để tái dùng (tránh build
  lại — tốn LLM call — khi cần sinh thêm câu sau này).
- CLI Typer mỏng để chạy thao tác trên.

**Không làm (ở phase này):**

- Không chạy `retrieve()`/`generate()` thật của hệ thống, không tính bất kỳ metric RAGAS
  nào (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) — các
  metric này cần `response`/`retrieved_contexts` thật, chỉ có ở Phase 2.
- Không dùng `chat_turns`/chatlog.
- Không tự động hoá việc lọc câu hỏi vô nghĩa bằng code (LLM-as-judge lọc, heuristic...).
  Review là thao tác đọc bằng mắt của người vận hành, ngoài phạm vi code.
- Không CI/cron; đây là script chạy tay, sinh dữ liệu tĩnh dùng lại nhiều lần.
- Không lưu Postgres; corpus và output đều là file tĩnh, không cần một CSDL cho việc này.

**Tiêu chí hoàn thành Phase 1:** có file `data/eval/golden_testset.json` chứa khoảng 360
câu hỏi hợp lệ (đủ `user_input`/`reference`/`reference_contexts`, không rỗng), đã được
người vận hành đọc qua và loại bỏ câu vô nghĩa, sẵn sàng dùng thẳng làm input cho Phase 2
mà không cần sinh lại từ đầu.

**Về việc chọn 360 thay vì một số nhỏ hơn:** build `KnowledgeGraph` là chi phí cố định
theo corpus (không đổi theo `testset_size`, và đã cache ra `knowledge_graph.json` để không
build lại); chỉ bước sinh câu hỏi mới tỉ lệ theo `testset_size`. Với 3 tài khoản Groq độc
lập luân phiên (mục 3.1) ngân sách token đủ rộng, và đây là job offline không gấp (có thể
chạy nhiều giờ/nhiều ngày nếu cần) — người dùng chủ động chọn 360 để có golden set đủ lớn
ngay từ đầu, chấp nhận đánh đổi thời gian review thủ công lâu hơn hẳn (mục 9). Đây là lựa
chọn của người dùng, không phải mặc định do kiến trúc đề xuất.

## 2. Input & Output

**Input:** `data/markdown/*.md` — đúng 6 văn bản pháp luật hoàn chỉnh (không phải chunk
đã cắt cho embedding; `TestsetGenerator` tự chunk/build graph nội bộ từ tài liệu gốc).

**Output** (thư mục mới `data/eval/`, không có trước):

| File | Nội dung |
| --- | --- |
| `golden_testset.json` | List các `GoldenTestCase` (mục 5): `user_input`, `reference`, `reference_contexts`, `synthesizer_name`. |
| `knowledge_graph.json` | `KnowledgeGraph` đã build, serialize bằng API save/load sẵn có của ragas — dùng lại khi cần sinh thêm câu hỏi mà không build lại từ đầu. |

`golden_testset.json` phải giữ đúng 3 cột bắt buộc theo schema `EvaluationDataset` của
ragas (`user_input`, `reference`, `reference_contexts`) để Phase 2 dùng thẳng không cần
convert; `synthesizer_name` là cột phụ giữ lại để biết loại câu hỏi (single-hop/multi-hop)
lúc review bằng mắt.

## 3. Công cụ

| Việc | Công cụ |
| --- | --- |
| Sinh testset | `ragas.testset.TestsetGenerator` (`ragas==0.4.3`) |
| LLM sinh câu hỏi (`generator_llm`) | `ChatOpenAI` trỏ Groq (cùng pattern `generation_spec.md` mục 8), bọc `ragas.llms.LangchainLLMWrapper` |
| Embedding cho build KnowledgeGraph (`generator_embeddings`) | Adapter mới quanh `HuggingFaceEmbedder`/`InferenceClient` hiện có ở `embedding/hf_client.py`, bọc `ragas.embeddings.LangchainEmbeddingsWrapper` |
| Document loader | `langchain_core.documents.Document` (đã có sẵn qua `langchain-openai`, không cần thêm dependency loader) |
| CLI | `typer` (`tools/generate_testset.py`) |

**Dependency mới (`pyproject.toml`):** `ragas>=0.4.3` và `langchain-community<0.4`,
nằm trong `[dependency-groups] eval` (không phải `[project] dependencies` gốc) — `uv sync`
mặc định (venv production, `deploy/Dockerfile`) KHÔNG cài 2 gói này lẫn
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

**Xác nhận tên tham số/API chính xác lúc implement:** tài liệu ragas tham khảo được
(mục README dự án, không phải bản cài thật) cho thấy chữ ký đại thể
`TestsetGenerator(llm=..., embedding_model=...)` và
`generator.generate_with_langchain_docs(docs, testset_size=...)`, nhưng chưa chạy thật
trên `ragas==0.4.3` đã cài — theo nguyên tắc "đo trước khi tin" của project, xác nhận lại
đúng signature với package cài thật trước khi chốt code, không suy diễn từ doc web khác
version.

### 3.1 Round-robin 3 tài khoản Groq (`groq_round_robin.py`) — quyết định mới

Với `testset_size = 360`, số lượt gọi `generator_llm` (build `KnowledgeGraph` + sinh câu
hỏi) đủ lớn để 1 tài khoản Groq duy nhất dễ chạm rate limit theo phút/ngày (`CLAUDE.md`:
Groq giới hạn theo **tài khoản**, không theo key). Người dùng đã xác nhận tạo 3 tài khoản
Groq riêng biệt (`GROQ_API_KEY`, `GROQ_API_KEY_2`, `GROQ_API_KEY_3` — 3 email khác nhau,
không phải 3 key cùng 1 tài khoản) và muốn luân phiên cả 3 cho luồng gọi này.

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

    clients: list[ChatOpenAI]  # đúng 3, mỗi client gắn 1 key cố định

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
data/markdown/*.md (6 file, đọc nguyên văn)
  → wrap từng file thành langchain_core.documents.Document
    (page_content = nội dung file, metadata = {"source": tên file})
  → clients = [ChatOpenAI(key=GROQ_API_KEY), ChatOpenAI(key=GROQ_API_KEY_2),
               ChatOpenAI(key=GROQ_API_KEY_3)]                  # config mục 6
  → generator_llm = LangchainLLMWrapper(GroqRoundRobinChatModel(clients=clients))  # mục 3.1
  → generator_embeddings = LangchainEmbeddingsWrapper(adapter) # embeddings_adapter.py
  → TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
  → generate_with_langchain_docs(docs, testset_size=TESTSET_SIZE)  # 360, mục 1
      (build KnowledgeGraph nội bộ: tóm tắt node, trích entity, dựng quan hệ — tốn
       nhiều lượt gọi generator_llm, rải qua 3 tài khoản round-robin; rồi sinh 360 câu hỏi
       theo phân phối mặc định của ragas: single-hop 180 / multi-hop abstract 90 /
       multi-hop specific 90)
  → lưu knowledge_graph ra data/eval/knowledge_graph.json (tái dùng lần sau)
  → convert testset → list[GoldenTestCase] → lưu data/eval/golden_testset.json
  → [thao tác tay, ngoài code] người vận hành mở file, đọc qua, xoá câu vô nghĩa/lặp,
    ghi đè lại file
```

`TESTSET_SIZE = 360` là hằng số nội bộ module, không phải biến môi trường (theo
coding-convention: constants chỉ thuộc một cơ chế, không biến mọi chi tiết thành env var).
Chạy trên **toàn bộ 6 văn bản** trong một lần (tổng corpus ~1 MB, đủ nhỏ để không cần
pilot một văn bản riêng trước). Đây là job offline không gấp — chấp nhận chạy lâu (có thể
nhiều giờ tuỳ tốc độ Groq) để đổi lấy golden set 360 câu ngay từ lần chạy đầu.

Không tách bước "build KnowledgeGraph" và "sinh câu hỏi" thành hai lệnh CLI riêng ở Phase
1 — corpus nhỏ, chấp nhận chạy lại toàn bộ nếu lỗi giữa chừng (mục 8); tách nhỏ hơn chỉ
đáng làm khi đo được lỗi hay xảy ra hoặc corpus lớn hơn hẳn.

## 5. Model dữ liệu (`models.py`)

```python
class GoldenTestCase(BaseModel):
    """Một câu hỏi mẫu trong golden testset, sinh tự động bởi ragas."""

    user_input: str
    reference: str
    reference_contexts: list[str]
    synthesizer_name: str | None = None
```

Không cần model phức tạp hơn ở Phase 1; không lẫn `GoldenTestCase` với bất kỳ contract
nào của `retrieval/`/`generation/` (đây là dữ liệu tham chiếu tĩnh, không phải
`RetrievedChunk`).

## 6. Config (`config.py`)

Thêm vào `config.py` gốc (cạnh `GenerationSettings`, `JudgeSettings`... theo đúng pattern
`pydantic-settings` hiện có):

```python
class TestsetGeneratorSettings(BaseSettings):
    """Cấu hình 3 tài khoản Groq round-robin cho generator_llm (Phase 1 RAGAS, mục 3.1).

    Cả 3 key BẮT BUỘC (không optional/fallback như GenerationSettings/JudgeSettings) —
    round-robin chỉ có ý nghĩa khi đủ 3 tài khoản độc lập; thiếu key nào, pydantic báo lỗi
    rõ ràng ngay lúc khởi tạo thay vì âm thầm chạy round-robin với 1-2 tài khoản.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: str = Field(validation_alias="GROQ_API_KEY")
    api_key_2: str = Field(validation_alias="GROQ_API_KEY_2")
    api_key_3: str = Field(validation_alias="GROQ_API_KEY_3")
    model_name: str = "openai/gpt-oss-120b"
    max_retries: int = 2
    timeout_seconds: int = 60
```

- Dùng cả 3 khoá `GROQ_API_KEY`/`GROQ_API_KEY_2`/`GROQ_API_KEY_3` (3 tài khoản Groq độc
  lập, xác nhận bởi người dùng) qua round-robin (mục 3.1) — khác mục đích tách ngân sách
  theo bước của `GenerationSettings`/`JudgeSettings`, nên **không dùng chung class** với
  chúng dù tên biến trùng `api_key_2`.
- `model_name` dùng lại `gpt-oss-120b` (không phải `20b` như condense) vì sinh câu hỏi
  multi-hop cần khả năng tổng hợp/suy luận qua nhiều đoạn văn bản.
- Không thêm setting riêng cho embeddings: `embeddings_adapter.py` tái dùng thẳng
  `EmbeddingSettings` đã có (`config.py`).
- Module không đọc `.env` trực tiếp. **Cập nhật `.env.example`:** thêm `GROQ_API_KEY_3`
  (mô tả rõ: tài khoản Groq thứ 3, dùng cho round-robin sinh testset Phase 1 RAGAS — khác
  `GROQ_API_KEY_2` vốn dùng để tách ngân sách generation theo bước cố định).

## 7. Module (`src/production_legal_qa_rag/evaluation/`)

| Module | Trách nhiệm duy nhất |
| --- | --- |
| `models.py` | `GoldenTestCase` (Pydantic v2). |
| `corpus_loader.py` | Đọc `data/markdown/*.md` → `list[langchain_core.documents.Document]`. |
| `embeddings_adapter.py` | Adapter implement interface `Embeddings` của LangChain quanh `InferenceClient.feature_extraction` (`embedding/hf_client.py`), không word-segment (mục 3). |
| `groq_round_robin.py` | `GroqRoundRobinChatModel` — proxy `BaseChatModel` luân phiên 3 `ChatOpenAI` (mục 3.1). |
| `testset_generator.py` | Điều phối: khởi tạo `TestsetGenerator`, chạy `generate_with_langchain_docs`, lưu `KnowledgeGraph` + `golden_testset.json`. |
| `tools/generate_testset.py` | Typer entrypoint mỏng, chỉ gọi `testset_generator.py`; không chứa business logic. |

Ghi chú vị trí: đặt ngay ở package `evaluation/` (không phải chỉ 1 script rời trong
`tools/`) vì Phase 2 sẽ thêm module chạy eval (`run_eval.py`/tương đương) vào **cùng**
package này — tránh phải dời code khi mở rộng.

## 8. Xử lý lỗi

| Sự cố | Xử lý |
| --- | --- |
| Groq lỗi/quota/timeout giữa lúc build `KnowledgeGraph` hoặc sinh câu hỏi | Dừng chương trình, log lỗi rõ ràng (không log nội dung câu hỏi/context — theo chính sách log chung của project); không lưu file output dở dang. Corpus nhỏ, chấp nhận chạy lại toàn bộ thay vì xây cơ chế resume/partial-save ở Phase 1. |
| `embeddings_adapter.py` trả response sai định dạng (khác kỳ vọng của HF API) | Raise lỗi rõ, không âm thầm trả vector rỗng — cùng nguyên tắc validate ở biên như `embedding/hf_client.py`. |
| File `data/markdown/*.md` trống hoặc thiếu | Raise lỗi rõ trước khi gọi `TestsetGenerator` (fail fast, không lãng phí LLM call). |
| Một tài khoản Groq bị 429 giữa vòng round-robin (mục 3.1) | Thử ngay tài khoản kế tiếp trong vòng lặp, tối đa `len(clients)` lần cho MỘT lượt gọi; hết vòng vẫn lỗi thì raise nguyên lỗi cuối — không giữ vòng lặp vô hạn, không tự ý bỏ qua câu hỏi đó. |
| Thiếu `GROQ_API_KEY_2`/`GROQ_API_KEY_3` trong `.env` | `TestsetGeneratorSettings` raise lỗi validate ngay lúc khởi tạo (fail fast) — không âm thầm chạy round-robin với ít hơn 3 tài khoản. |

## 9. Nghiệm thu thủ công

1. Chạy `uv run --group eval --no-group production tools/generate_testset.py` → tạo được
   `data/eval/golden_testset.json` và `data/eval/knowledge_graph.json`. Chấp nhận job chạy
   lâu (có thể nhiều giờ) do
   `testset_size=360`; không cần tối ưu tốc độ ở Phase 1 (mục 1, mục 4).
2. Mở `golden_testset.json`: có khoảng 360 dòng, mỗi dòng có đủ `user_input`, `reference`,
   `reference_contexts` không rỗng.
3. Kiểm tra round-robin hoạt động thật (mục 3.1): log số lượt gọi theo từng tài khoản (hoặc
   nhìn dashboard usage của cả 3 tài khoản Groq sau khi chạy) — mỗi tài khoản nhận tải xấp xỉ
   nhau (~1/3 tổng số lượt gọi), không dồn hết vào 1 tài khoản; không có request nào thất bại
   hẳn vì rate limit (nếu có, bounded fallback ở mục 3.1/8 phải xử lý được).
4. **Đọc qua bằng mắt cả 360 câu** (khối lượng lớn hơn hẳn con số 30 mặc định ban đầu —
   đây là lựa chọn có chủ đích của người dùng, chấp nhận tốn nhiều thời gian review hơn để
   đổi lấy golden set lớn ngay từ đầu, mục 1): xoá câu vô nghĩa, câu quá máy móc kiểu "so
   sánh Điều X và Điều Y" không giống câu hỏi tình huống thật, hoặc câu trùng lặp ý; ghi đè
   lại file. Không có ngưỡng số cứng ở Phase 1 — chấp nhận đánh giá định tính, có thể chia
   nhỏ việc review theo nhiều lần đọc thay vì làm 1 lần liên tục.
5. Chạy lại CLI với cờ tái dùng `KnowledgeGraph` đã lưu (nếu implement) → xác nhận không
   phải build lại đồ thị (số lượt gọi Groq giảm rõ rệt so với lần chạy đầu).

## 10. Rủi ro / điểm mở

1. `embeddings_adapter.py` là code mới, chưa có tiền lệ trong repo — cần test tay kỹ trước
   khi tin dùng cho việc build `KnowledgeGraph`.
2. Câu hỏi ragas sinh ra có thể lệch phân bố so với câu hỏi người dùng thật sẽ hỏi (thiên
   về cấu trúc bề mặt tài liệu — "Điều X quy định gì" — hơn là tình huống pháp lý cụ thể).
   Đây là hạn chế đã biết của synthetic data nói chung, xử lý bằng review thủ công ở Phase
   1, không có gate tự động.
3. Build `KnowledgeGraph` không tách nhỏ để resume nếu lỗi giữa chừng — chấp nhận được vì
   corpus nhỏ (~1 MB) và đây là script chạy tay không thường xuyên; xem lại nếu corpus mở
   rộng đáng kể.
4. Chưa xác nhận signature chính xác của `ragas.testset.TestsetGenerator` trên bản cài
   thật `ragas==0.4.3` (mục 3) — việc đầu tiên khi implement là chạy thử với 1 văn bản nhỏ
   để xác nhận API trước khi chạy full 6 văn bản.
5. **Round-robin 3 tài khoản Groq (mục 3.1) là pattern mới, chưa có tiền lệ trong repo** —
   khác hẳn cách dùng nhiều key hiện có (tách ngân sách theo bước cố định). Không nên coi
   đây là tiền lệ để áp dụng lại ở nơi khác trừ khi có nhu cầu tương tự (khối lượng LLM
   call lớn, job offline không nhạy latency); `generation/`/`retrieval/`/`conversation/`
   vẫn giữ nguyên pattern 1 key cố định/bước như đã chốt trước đó.
6. `itertools.cycle` không thread/async-safe tuyệt đối (mục 3.1) — chấp nhận rải tải không
   hoàn toàn đều nếu ragas gọi `generator_llm` đồng thời; không ảnh hưởng tính đúng đắn của
   testset sinh ra, chỉ ảnh hưởng độ cân bằng tải giữa 3 tài khoản.
7. Phase 2 (chạy pipeline thật, tính metric, chọn judge LLM, ngưỡng theo dõi, tần suất
   chạy) chưa được thiết kế — brainstorm riêng sau khi có `golden_testset.json` dùng được.
