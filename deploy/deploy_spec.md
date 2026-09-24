# Deploy — Chạy toàn bộ chatbot trên máy cá nhân, public qua Cloudflare Tunnel

## 1. Mục tiêu & phạm vi

Đóng gói và chạy cả hệ thống chatbot (`api/api_spec.md`) bằng **một lệnh
`docker compose up -d`** trên máy của tác giả (Windows + WSL2), cho người khác truy cập
qua Internet bằng URL HTTPS do **Cloudflare Tunnel** cấp — không thuê VPS, không mở cổng
router. Máy tắt thì dịch vụ tắt (chấp nhận, mục 10).

**Trong phạm vi:**

- `deploy/` (ngoài `src/`, vì không phải code import được): Dockerfile, compose, mẫu biến
  môi trường, script khởi tạo DB, script backup.
- 5 service: `cloudflared`, `open-webui`, `api`, `redis`, `postgres`.
- Chính sách truy cập public (đăng ký mở), cô lập mạng, quản lý bí mật, vận hành cơ bản.

**Không làm:**

- Không VPS/PaaS, không Kubernetes, không Caddy/nginx (Cloudflare lo HTTPS), không nhiều
  replica/worker, không HA.
- Không CI/CD tự deploy (CI hiện có chỉ chạy test/lint); deploy là thao tác tay.
- Không monitoring/alert (Prometheus, Grafana, Langfuse) — thuộc phase cuối sau RAGAS
  (tracing, tracking & CI); tạm thời theo dõi qua bảng `chatlog` và `docker compose logs`.
- Không tự host LLM/Pinecone: vẫn dùng dịch vụ ngoài như hiện nay. Reranker **không**
  còn trong nhóm này — chạy in-process trong `api`, không host tách rời qua
  LightningAI/ngrok nữa (`retrieval_spec.md` mục 6.1, xem mục 4.1 dưới đây).

**Tiêu chí quan trọng nhất:** clone repo + điền `deploy/.env` + `docker compose up -d` →
người dùng bên ngoài mở URL, đăng ký, hỏi đáp nhiều lượt được; ngoài Cloudflare Tunnel
không có cổng nào của hệ thống lộ ra ngoài; bí mật không nằm trong image hay git.

## 2. Kiến trúc

```
Internet ─HTTPS─► Cloudflare ◄─(kết nối ra do cloudflared tự mở)─ cloudflared
                                                                       │ mạng compose `internal`
                                                                       ▼
                                                                  open-webui ──► postgres (DB openwebui)
                                                                       │
                                                                       ▼  Bearer CHATBOT_API_KEY
                                                                      api ─────► postgres (DB chatbot)
                                                                       │  ├────► redis
                                                                       │  └────► reranker (in-process, GPU khuyến nghị/CPU fallback)
                                                                       ▼
                                                              Groq · HF · Pinecone
```

`cloudflared` chỉ **kết nối ra** Cloudflare (outbound), nên không cần mở cổng vào trên
router/firewall. Chỉ `cloudflared` nói chuyện được với `open-webui`; chỉ `open-webui` (và
script vận hành) nói chuyện được với `api`.

## 3. Cloudflare Tunnel

Hai chế độ, cùng một service `cloudflared`:

| Chế độ                | Cần                          | URL                                   | Dùng khi                     |
| --------------------- | ---------------------------- | ------------------------------------- | ---------------------------- |
| **Quick tunnel** (mặc định bản đầu) | Không cần tài khoản/domain | Ngẫu nhiên `*.trycloudflare.com`, **đổi mỗi lần khởi động lại** | Chạy thử, demo |
| **Named tunnel**      | Tài khoản Cloudflare + 1 domain | Cố định, vd. `chat.tenban.com`    | Khi muốn URL ổn định         |

- Quick tunnel: `cloudflared tunnel --no-autoupdate --url http://open-webui:8080`. URL in ra
  log của service (`docker compose logs cloudflared`); Cloudflare không cam kết uptime cho
  chế độ này.
- Named tunnel: `cloudflared tunnel --no-autoupdate run` với `TUNNEL_TOKEN` (trong
  `deploy/.env`); trỏ hostname tới `http://open-webui:8080` trong dashboard Cloudflare.
  Chuyển từ quick sang named **chỉ đổi cấu hình compose/biến**, không đổi code. Chọn qua
  compose profile: `quick` (mặc định) và `named`.
- HTTPS/TLS do Cloudflare đảm nhiệm (kết thúc TLS ở edge); trong mạng compose dùng HTTP.
- Đặt `WEBUI_URL` của OpenWebUI theo URL thật khi dùng named tunnel (quick tunnel: bỏ qua,
  chấp nhận link dạng tương đối).

## 4. Services (`deploy/docker-compose.yml`)

Một network `internal` (bridge). **Không service nào publish cổng ra host** ngoại trừ
tuỳ chọn dev `127.0.0.1:3000 → open-webui:8080` và `127.0.0.1:8000 → api:8000` (chỉ loopback,
trên máy tác giả: mở thử UI, gọi API bằng `curl`/Swagger `/docs` — vẫn không ra Internet).

| Service       | Image                                        | Ghi chú                                                                 |
| ------------- | -------------------------------------------- | ----------------------------------------------------------------------- |
| `cloudflared` | `cloudflare/cloudflared:<tag ghim>`          | Phụ thuộc `open-webui` healthy; xem mục 3                               |
| `open-webui`  | `ghcr.io/open-webui/open-webui:<tag ghim>`   | Cấu hình `api_spec.md` mục 9 + mục 5 dưới đây; volume `openwebui_data` (cache/ảnh; dữ liệu chính ở Postgres) |
| `api`         | build từ `deploy/Dockerfile`                 | Mount `data/bm25/` (read-only) + volume `hf_cache` (cache model reranker); `env_file: .env`; xem mục 4.1, 6 |
| `redis`       | `redis:7-alpine`                             | `--requirepass`, `--appendonly yes`, `--maxmemory 256mb --maxmemory-policy allkeys-lru`; volume `redis_data` |
| `postgres`    | `postgres:17-alpine`                         | Mount `deploy/initdb/` (chạy 1 lần lúc tạo volume); volume `postgres_data` |

Quy tắc chung:

- `restart: unless-stopped` cho mọi service; **ghim tag phiên bản** (không dùng `latest`),
  ghi lại phiên bản OpenWebUI đã nghiệm thu.
- Healthcheck: `postgres` (`pg_isready`), `redis` (`redis-cli ping` có mật khẩu), `api`
  (`GET /readyz` bằng `python -c "urllib…"`, không cần curl), `open-webui` (endpoint
  health của nó). `depends_on: condition: service_healthy` theo chuỗi
  `postgres,redis → api → open-webui → cloudflared`.
- Giới hạn tài nguyên: `mem_limit` cho từng service (gợi ý: `api` 1.5g, `open-webui` 1g,
  `postgres` 512m, `redis` 320m) để không nuốt hết RAM của WSL2; điều chỉnh sau khi đo.
- Log: driver `json-file` với `max-size: 10m`, `max-file: 3`.

### 4.1 GPU passthrough cho reranker (khuyến nghị, không bắt buộc)

`api` chạy reranker in-process (`retrieval_spec.md` mục 6.1), tự phát hiện
`cuda`/`cpu`. Compose gốc **không** yêu cầu GPU — chạy CPU-only ngay không cần
cấu hình thêm, ai không có GPU vẫn dùng được đầy đủ, chỉ rerank chậm hơn.

Khuyến nghị bật GPU nếu máy có card NVIDIA (kể cả VRAM nhỏ, vd 2GB). Hai việc
tách biệt, cả hai đều cần cho GPU thật hoạt động trong container:

1. **Build image đúng biến thể torch**: `deploy/Dockerfile` nhận build arg
   `TORCH_VARIANT` (mặc định `cpu`, dùng
   `--index-url https://download.pytorch.org/whl/cpu`; giá trị `cu121` dùng
   `--index-url https://download.pytorch.org/whl/cu121` để cài wheel CUDA).
   Build bản GPU: `docker compose build --build-arg TORCH_VARIANT=cu121 api`.
2. **Cấp GPU cho container lúc chạy**: cài **NVIDIA Container Toolkit** trên
   Windows (Docker Desktop dùng WSL2 backend đã hỗ trợ sẵn CUDA passthrough,
   chỉ cần bật GPU support trong Docker Desktop settings). Bật qua file
   override riêng, không sửa `docker-compose.yml` gốc — ví dụ
   `deploy/docker-compose.gpu.yml` khai báo cho service `api`:
   ```yaml
   services:
     api:
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]
   ```
   Chạy: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`.

Thiếu 1 trong 2 bước trên: build CPU + override GPU → container có device
nhưng torch không dùng được, coi như CPU; build CUDA + không override → thiếu
device, torch CUDA khởi tạo sẽ tự fallback CPU (`retrieval_spec.md` mục 6.1).
Không làm gì cả (mặc định) → container CPU-only, không cần cấu hình, không lỗi.

VRAM nhỏ (2GB) vẫn có thể CUDA OOM ở batch lớn; hành vi khi đó là fallback
`rerank_score=None` (`retrieval_spec.md` mục 8), retrieval vẫn trả kết quả,
không crash service.

## 5. Chính sách truy cập public

- **Đăng ký mở (đã chốt 2026-09-21):** `ENABLE_SIGNUP=true`, `DEFAULT_USER_ROLE=user`; ai có
  URL đều tự tạo tài khoản dùng ngay. Lý do: máy chạy mới có dịch vụ nên quy mô tự bị
  giới hạn; không muốn duyệt tay.
- **Tài khoản đầu tiên được đăng ký trên OpenWebUI thường thành admin** (docs chưa nêu rõ, xác nhận khi chạy thử) → tác giả phải
  đăng ký tài khoản admin **trước khi công bố URL**. Sau đó tắt các tính năng dành cho
  admin không cần thiết đối với người dùng thường (xem `api_spec.md` mục 9).
- Vì đăng ký mở, hạn mức Groq được bảo vệ bằng **ngân sách toàn cục**
  (`GLOBAL_DAILY_LLM_ANSWERS`), quota theo user/ngày và rate limit theo phút
  (`conversation_spec.md` mục 8, `api_spec.md` mục 5). Lưu ý: một người tạo nhiều tài
  khoản vượt được quota theo user, **chỉ ngân sách toàn cục chặn được** — khi cạn, mọi
  người nhận thông báo `quota_exceeded` (chấp nhận, ưu tiên bảo vệ hạn mức).
- Nếu bị lạm dụng: đóng đăng ký bằng cách đặt `ENABLE_SIGNUP=false` rồi
  `docker compose up -d open-webui`, hoặc tắt `cloudflared`.

## 6. Image API (`deploy/Dockerfile`)

- Base `python:3.14-slim`; cài `uv` (copy từ `ghcr.io/astral-sh/uv`); tầng dependency riêng:
  copy `pyproject.toml` + `uv.lock` → `uv sync --frozen --no-dev --no-install-project`,
  rồi copy `src/`, `alembic/`, `alembic.ini` → cài project (tận dụng cache tầng Docker).
- `torch`: cài theo build arg `TORCH_VARIANT` (mặc định `cpu`) — xem mục 4.1 để build
  bản GPU (`cu121`). Mặc định `cpu` để image build được trên mọi máy không cần driver
  GPU và nhẹ hơn cho người không dùng GPU.
- Model checkpoint (`AITeamVN/Vietnamese_Reranker`) tải từ HF Hub ở lần chạy đầu, không
  bake vào image (tránh build image nặng hơn và cứng phiên bản model). Mount volume
  `hf_cache` vào thư mục cache Hugging Face của user chạy container để không tải lại
  (~1GB) mỗi lần recreate container.
- Chạy bằng user không phải root.
- `data/bm25/` **không** đóng gói vào image (file sinh ra, đã `.gitignore`); mount từ host
  read-only. Thiếu file → `api` lỗi rõ ràng lúc khởi động, không chạy nửa vời.
- Lệnh chạy: `alembic upgrade head` rồi
  `uvicorn production_legal_qa_rag.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 1`.
  Chạy migration ở đây an toàn vì chỉ có 1 worker/1 container.
- **1 worker**: semaphore của admission là in-process (`conversation_spec.md` mục 8).
- `.dockerignore`: `.venv/`, `.git/`, `data/`, `tests/`, `.env`, `deploy/.env`, cache của
  mypy/ruff/pytest.

## 7. Biến môi trường & bí mật (`deploy/.env`, không commit)

`deploy/.env.example` liệt kê đủ tên biến, chia nhóm; giá trị bí mật để trống:

| Nhóm        | Biến                                                                                              |
| ----------- | ------------------------------------------------------------------------------------------------- |
| LLM/dịch vụ | `GROQ_API_KEY`, `GROQ_API_KEY_2`, `HF_TOKEN`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_SPARSE_INDEX_NAME` |
| Backend     | `CHATBOT_API_KEY` (sinh ngẫu nhiên ≥ 32 ký tự), `REDIS_PASSWORD`, `REDIS_URL` (`redis://:${REDIS_PASSWORD}@redis:6379/0`), `CHATLOG_DATABASE_URL` (`postgresql+asyncpg://…@postgres/chatbot`) |
| Postgres    | `POSTGRES_USER`, `POSTGRES_PASSWORD`                                                              |
| OpenWebUI   | `WEBUI_SECRET_KEY` (cố định, để phiên đăng nhập không mất khi khởi động lại), `WEBUI_URL` (tuỳ chọn) |
| Tunnel      | `TUNNEL_TOKEN` (chỉ khi dùng named tunnel)                                                        |

- `.env` đã nằm trong `.gitignore` (khớp mọi thư mục); nếu thêm file khác chứa bí mật, thêm
  vào `.gitignore` trước.
- Không truyền bí mật bằng `build args` hay bake vào image.

## 8. Vận hành cơ bản

- **Khởi động / dừng:** `docker compose up -d` / `docker compose down` (không có `-v`, để
  giữ volume). Xem link quick tunnel: `docker compose logs cloudflared`.
- **Máy Windows:** Docker Desktop (WSL2 backend) bật cùng Windows; tắt chế độ ngủ/hibernate
  khi cắm điện, nếu không tunnel đứt và người dùng không vào được.
- **Backup:** `deploy/backup.sh` chạy `pg_dump` cho cả 2 database (`openwebui`, `chatbot`)
  ra `deploy/backups/<ngày>/` (thư mục này vào `.gitignore`), giữ 7 bản gần nhất; chạy tay
  hoặc bằng cron/Task Scheduler. Redis không cần backup (dữ liệu tính lại được).
- **Cập nhật:** `git pull` → `docker compose build api` → `docker compose up -d`. Đổi
  phiên bản OpenWebUI: sửa tag, nghiệm thu lại mục 9 trước khi dùng.
- **Xem nhật ký:** `docker compose logs -f api`; phân tích lượt hỏi qua bảng `chat_turns`.
- **Dọn dữ liệu quá hạn:** `docker compose exec api python tools/purge_chatlog.py`
  (`chatlog_spec.md` mục 5) — cần `tools/` có trong image hoặc chạy từ host.

## 9. Nghiệm thu thủ công

1. Máy sạch (không có volume): `cp deploy/.env.example deploy/.env`, điền giá trị,
   `docker compose up -d` → mọi service `healthy`, không có cổng nào lắng nghe ngoài
   `127.0.0.1` (kiểm tra `docker compose ps`, `ss -ltn`).
2. Lấy URL từ `cloudflared`, mở bằng điện thoại (mạng 4G, ngoài LAN): thấy trang đăng nhập
   HTTPS; đăng ký tài khoản admin đầu tiên, rồi 1 tài khoản thường; hỏi câu hỏi luật.
3. Từ ngoài mạng, không truy cập được `api`, `redis`, `postgres` (chỉ có URL của
   OpenWebUI).
4. `docker compose down && docker compose up -d` → tài khoản, lịch sử chat, dòng
   `chat_turns`, cache đều còn (volume giữ dữ liệu); phiên đăng nhập không bị đăng xuất
   (`WEBUI_SECRET_KEY` cố định).
5. `deploy/backup.sh` tạo file dump khôi phục được vào Postgres tạm.
6. Tắt Redis rồi Postgres (từng cái) → chat vẫn trả lời, `/readyz` báo 503 (khớp
   `api_spec.md` mục 13).
7. Kiểm tra image: `docker history` / `grep` không thấy bí mật; chạy bằng user không root.
8. (Nếu build/bật GPU theo mục 4.1) `docker compose exec api python -c "import torch;
   print(torch.cuda.is_available())"` trả `True`; hỏi thử nhiều lượt liên tiếp không thấy
   log cảnh báo CUDA OOM. Nếu không bật GPU, bỏ qua bước này (mặc định CPU-only vẫn phải
   nghiệm thu qua các bước 1-7).

## 10. Rủi ro / điểm mở

1. **Máy tắt/ngủ/mất mạng = dịch vụ ngừng**; quick tunnel đổi URL sau mỗi lần khởi động
   lại (người dùng cũ phải lấy link mới). Khi cần URL ổn định: named tunnel (cần domain,
   khoảng 10 USD/năm) — **chưa có domain**, làm sau.
2. Đăng ký mở + quota theo user không chặn được người tạo nhiều tài khoản; chỉ ngân sách
   toàn cục bảo vệ hạn mức (mục 5). Nếu vẫn bị lạm dụng: đóng đăng ký (mục 5) hoặc thêm
   Cloudflare Turnstile/Access phía trước — chưa làm.
3. Máy cá nhân chứa dữ liệu hội thoại của người dùng khác: mã hoá đĩa (BitLocker) và cập
   nhật hệ điều hành là trách nhiệm của tác giả; banner thông báo lưu 90 ngày
   (`chatlog_spec.md` mục 5).
4. Reranker giờ chạy in-process trong `api` (mục 4.1): image mặc định chỉ cài `torch` CPU,
   nên nếu không bật GPU overlay, rerank chậm hơn GPU nhưng không phụ thuộc dịch vụ ngoài
   nào còn ngừng bất kỳ lúc nào như bản LightningAI/ngrok cũ. Bật GPU overlay sai cấu hình
   (thiếu NVIDIA Container Toolkit, chưa bật GPU support trong Docker Desktop) khiến `api`
   không khởi động được — kiểm tra kỹ trước khi thêm `docker-compose.gpu.yml`.
5. Điều khoản dịch vụ Cloudflare cho quick tunnel (không cam kết uptime, dành cho thử
   nghiệm); tác giả tự đối chiếu trước khi chia sẻ rộng.
6. Image nặng hơn do `transformers`/`pyvi`/`torch` + tải checkpoint reranker (~1GB) lần
   chạy đầu; RAM/disk WSL2 mặc định có thể không đủ cho 5 service — chỉnh `.wslconfig` nếu
   cần (gợi ý ≥ 6 GB cho WSL2, cân nhắc thêm cho volume `hf_cache`).
