# Luồng xử lý một câu hỏi (Activity diagram)

Hình mô tả những gì `ChatOrchestrator` làm cho **một câu hỏi** của người dùng, kể cả các nhánh
rẽ (từ chối, cache trúng, hết hạn mức, lỗi). Chi tiết từng bước: `conversation_spec.md`,
`cache_spec.md`, `api_spec.md`.

Màu: cam = Groq (LLM), đỏ = Redis, xanh lá = retrieval (HF, Pinecone, reranker),
xanh dương = Postgres. Ô bo tròn = điểm bắt đầu/kết thúc; hình thoi = điểm rẽ nhánh.

```mermaid
flowchart TD
    A(["Người dùng gửi câu hỏi<br/>trên Open WebUI"]) --> B["FastAPI: kiểm tra Bearer key<br/>lấy user-id và chat-id"]
    B --> C{"Vượt rate limit<br/>theo phút?"}
    C -- Có --> C1(["HTTP 429"])
    C -- Không --> D{"messages<br/>hợp lệ?"}
    D -- Không --> D1(["HTTP 422"])
    D -- Có --> E["Cắt cửa sổ history<br/>3 lượt gần nhất, bỏ nguồn và số trích dẫn"]

    E --> F1["Guardrail<br/>đọc câu gốc và 2 câu user trước"]
    E --> F2["Condense: viết lại thành câu độc lập<br/>chỉ chạy khi có history"]
    F1 --> G{"Guardrail<br/>cho phép?"}
    F2 --> G

    G -- Không --> G1["Từ chối cố định<br/>ngoài phạm vi hoặc injection"]
    G -- Có --> H{"Cache câu trả lời<br/>đã có?"}
    H -- Có --> H1["Phát lại như stream thật"]
    H -- Không --> I{"Đang có request khác<br/>xử lý cùng câu hỏi?"}
    I -- Có, mình là follower --> I1["Chờ request kia xong<br/>rồi đọc lại cache"]
    I1 --> H1
    I -- Không, mình là leader --> J{"Còn quota ngày<br/>và còn chỗ xếp hàng?"}
    J -- Không --> J1["Báo hết hạn mức<br/>hoặc hệ thống quá tải"]
    J -- Có --> K["Retrieval: HyDE, hybrid, RRF, MMR, rerank<br/>có cache retrieval"]
    K --> L{"Có chunk<br/>phù hợp?"}
    L -- Không --> L1["Báo không tìm thấy văn bản"]
    L -- Có --> M["Generate stream<br/>và kiểm tra đầu ra"]
    M --> N{"Sạch: không lỗi<br/>và không cảnh báo?"}
    N -- Có --> N1["Ghi cache câu trả lời"]
    N -- Không --> O
    N1 --> O["Nối khối Nguồn và cảnh báo"]

    H1 --> P
    O --> P
    G1 --> P
    J1 --> P
    L1 --> P
    P["Stream kết quả về Open WebUI"] --> Q["Ghi chatlog vào Postgres<br/>không chặn người dùng"]
    Q --> R(["Kết thúc"])

    classDef groq fill:#ffe8cc,stroke:#e8590c,color:#000
    classDef redis fill:#ffe3e3,stroke:#e03131,color:#000
    classDef retr fill:#d3f9d8,stroke:#2f9e44,color:#000
    classDef pg fill:#d0ebff,stroke:#1971c2,color:#000
    class F1,F2,M groq
    class C,H,I,I1,J,N1 redis
    class K retr
    class Q pg
```

## Ghi chú đọc hình

- **Song song:** Guardrail và Condense chạy cùng lúc, rồi gặp nhau ở điểm rẽ "Guardrail cho
  phép?". Lượt đầu tiên (chưa có history) thì Condense bỏ qua.
- **Chỉ tốn LLM khi cache trượt:** quota ngày và hàng đợi chỉ áp dụng sau khi cache câu trả lời
  không trúng.
- **Single-flight:** khi 50 người hỏi cùng một câu lúc cache còn trống, chỉ 1 request (leader) gọi
  LLM; các request còn lại (follower) chờ rồi đọc cache, không chiếm slot và không tốn quota.
  Nếu leader lỗi hoặc chờ quá lâu, follower tự thử làm leader (chưa vẽ).
- **Mọi nhánh sau bước kiểm tra đầu vào đều đi về "Stream kết quả"** rồi ghi chatlog, kể cả từ
  chối và lỗi, nên mỗi lượt được nhận đều có đúng 1 dòng log. Request bị chặn từ đầu bằng HTTP
  429/422 chưa thành "lượt" nên không có dòng log.
- Chưa vẽ: guardrail lỗi (fail-open, coi như cho phép) và condense lỗi (dùng câu gốc).
- Chưa vẽ: nhánh lỗi giữa chừng (Groq 429, stream đứt) — chúng rẽ ra từ bước Retrieval/Generate
  về "Stream kết quả" dưới dạng thông báo lỗi.
