# Deploy — Chạy toàn bộ chatbot trên máy cá nhân, public qua Cloudflare Tunnel

- Giữ nguyên số mục để không làm hỏng tham chiếu từ code/spec khác; bản đầy đủ ở git history.
- Spec liên quan: [api_spec.md](../src/production_legal_qa_rag/api/api_spec.md),
  [conversation_spec.md](../src/production_legal_qa_rag/conversation/conversation_spec.md),
  [observability_spec.md](../src/production_legal_qa_rag/observability/observability_spec.md).

## 1. Mục tiêu & phạm vi

- Một lệnh chạy toàn chatbot trên máy cá nhân Windows/WSL2, public HTTPS qua Cloudflare Tunnel; không VPS/mở
  cổng router.
- Cùng entrypoint cho laptop tác giả có GPU và người clone CPU-only; không cần chọn biến thể trước.
- Làm: Dockerfile/Compose/scripts trong deploy, năm service cloudflared/open-webui/api/redis/postgres, bảo vệ
  mạng/bí mật, backup/vận hành.
- Postgres chỉ OpenWebUI; reranker in-process trong API (retrieval 6.1).
- Không làm: VPS/PaaS/Kubernetes/Caddy/nginx, replica/HA, CD tự deploy, monitoring/alert trong compose
  production, host LLM/Pinecone.
- Observe ở compose riêng observability (observability spec 7); CI test/lint, deploy tay.
- Acceptance: clone → điền root env → chạy → người ngoài đăng ký/hỏi nhiều lượt; không lộ cổng/bí mật; host
  tắt thì dịch vụ tắt.

## 2. Kiến trúc

- Internet HTTPS → Cloudflare edge → cloudflared → OpenWebUI → Postgres DB openwebui.
- OpenWebUI Bearer chatbot key → API → Redis, local reranker, Groq/HF/Pinecone.
- cloudflared mở kết nối ra; không mở cổng vào; UI/backend trao đổi mạng compose.
- API chỉ cho client nội bộ có key và script vận hành.

## 3. Cloudflare Tunnel

- Profile quick mặc định: không account/domain, URL ngẫu nhiên trycloudflare.com, đổi khi restart, không cam
  kết uptime.
- Quick command: `cloudflared tunnel --no-autoupdate --url http://open-webui:8080`; URL trong log.
- Named profile: account/domain, URL cố định; `cloudflared tunnel --no-autoupdate run`, TUNNEL_TOKEN root env;
  dashboard trỏ UI:8080.
- Quick → named chỉ đổi cấu hình, không code; WEBUI_URL theo URL named thật.
- TLS ở Cloudflare edge; mạng compose HTTP.

## 4. Services (`deploy/docker-compose.yml`)

- Một bridge network internal; production không publish host port.
- Dev tùy chọn chỉ loopback: 127.0.0.1:3000 → UI:8080, 127.0.0.1:8000 → API:8000.
- cloudflared/UI: pin tag, UI volume openwebui_data; ghi phiên bản UI đã nghiệm thu.
- API: build Dockerfile, mount data/bm25 read-only + hf_cache, env_file ../.env.
- Redis 7-alpine: requirepass, appendonly yes, maxmemory 256mb, allkeys-lru.
- Postgres 17-alpine: POSTGRES_DB=openwebui; image tự tạo DB khi init volume; không deploy/initdb.
- Chung: restart unless-stopped, không latest; json-file log max-size 10m/max-file 3.
- Healthcheck: pg_isready, redis-cli ping có mật khẩu, API readyz bằng Python urllib, UI health.
- depends_on service_healthy: Redis → API → UI → tunnel; Postgres → UI; API không phụ thuộc Postgres.
- `mem_limit`: API 3g, UI 1g, Postgres 512m, Redis 320m.
- API 1.5g từng OOM lượt rerank đầu (2026-09-27, model/CUDA/hf-xet khoảng 1.53GB); giữ 3g.

### 4.1 GPU passthrough cho reranker (tự động qua `deploy/up.sh`)

- up.sh là entrypoint duy nhất, chạy từ bất kỳ cwd, tự cd deploy; reranker tự chọn cuda/cpu.
- Build arg TORCH_VARIANT: mặc định cpu, GPU cu126 trở lên; cu121/cu124 dừng torch 2.5.1/2.6.0, không wheel
  cp314.
- GPU override deploy/docker-compose.gpu.yml: reservation driver nvidia/count 1/capabilities gpu; không đặt
  GPU bắt buộc trong compose gốc.
- Host cần NVIDIA Container Toolkit hoặc Docker Desktop WSL2 GPU support.
- Chỉ dùng GPU khi nvidia-smi chạy được và docker info có runtime nvidia; thiếu một điều kiện → CPU, không
  fail.
- CPU: build TORCH_VARIANT=cpu rồi up; GPU: build cu126 và ghép override khi up.
- Script giữ build/override khớp; gọi tay lệch có rủi ro (10.4).
- VRAM 2GB có thể OOM batch lớn → fallback `rerank_score=None` (retrieval 8), không crash service.

## 5. Chính sách truy cập public

- Đăng ký mở: ENABLE_SIGNUP=true, DEFAULT_USER_ROLE=user; không duyệt tay.
- Tài khoản đầu thường là admin: đăng ký trước công bố URL, kiểm khi nghiệm thu, tắt tính năng thừa (API 9).
- Bảo vệ Groq: cache, rate limit phút, admission đồng thời, throttle bước nhẹ và 429 thật (conversation
  9/12.1, API 5).
- Không có quota user/ngày hoặc GLOBAL_DAILY_LLM_ANSWERS; nhiều account có thể vượt rate limit theo user và
  làm cạn provider quota.
- Khi provider hết hạn mức: thông báo rate_limited/retry-after; không tự dựng ngân sách ngày riêng.
- Lạm dụng: tắt signup bằng Admin Panel hoặc cấu hình PersistentConfig phù hợp (API 9), recreate UI; hoặc dừng
  tunnel.

## 6. Image API (`deploy/Dockerfile`)

- Base python:3.14-slim, uv từ ghcr.io/astral-sh/uv; torch theo TORCH_VARIANT, CPU mặc định.
- Tầng deps riêng pyproject.toml/uv.lock → uv sync --frozen --no-dev --no-install-project → copy src/README →
  cài project.
- RUN uv dùng --mount=type=cache: đổi dependency không tải lại torch; không prune builder cache tùy tiện.
- Reranker checkpoint khoảng 1GB tải HF lần đầu, không bake image; hf_cache giữ qua recreate.
- User không root; BM25 không trong image, mount read-only; thiếu file → lỗi startup rõ.
- Uvicorn `production_legal_qa_rag.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 1`; không
  migration; admission in-process.
- dockerignore: .venv/.git/data/tests/.env và mypy/ruff/pytest cache.

## 7. Biến môi trường & bí mật (`.env` ở repo root, không commit)

- Duy nhất root .env/.env.example; API container env_file ../.env, host config đọc cùng file.
- Example có ba block APP/DEPLOY/OBSERVABILITY; prefix DEPLOY_/OBS_ cho biến trùng giữa stack.
- APP: `GROQ_API_KEY_1`…9 (1–4 production, 5–9 eval), `HF_TOKEN`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`,
  `PINECONE_SPARSE_INDEX_NAME`, `CHATBOT_API_KEY`.
- Deploy: COMPOSE_PROFILES, REDIS_PASSWORD (Compose dựng container REDIS_URL),
  `DEPLOY_POSTGRES_USER`/`DEPLOY_POSTGRES_PASSWORD` → container `POSTGRES_USER`/`POSTGRES_PASSWORD`.
- UI: WEBUI_SECRET_KEY cố định qua restart, WEBUI_URL tùy chọn; named tunnel TUNNEL_TOKEN.
- Observe là stack riêng cho traffic production (observability 7), không phục vụ end-user trực tiếp.
- .env phải được gitignore; file bí mật khác cũng ignore trước; không truyền secret qua build args hoặc bake
  image.

## 8. Vận hành cơ bản

- Start/update: ./deploy/up.sh; dò GPU/build/up/in quick URL, tự ghép observe nếu network tồn tại.
- Stop: ./deploy/down.sh, Compose --env-file ../.env --profile '*' down; không -v, giữ volume, không để tunnel
  sót.
- Thiếu .env lần đầu: copy example rồi dừng để điền; named URL theo WEBUI_URL, xem log cloudflared-named nếu
  cần.
- Windows: Docker Desktop WSL2 tự bật; tắt sleep/hibernate khi cắm điện.
- Backup: backup.sh pg_dump openwebui vào deploy/backups/<ngày>/, gitignore, giữ 7 bản; Redis không backup.
- Update: git pull → up.sh; đổi UI tag phải nghiệm thu lại.
- Logs API để vận hành; nhật ký QA tại Langfuse (4.5); DB chatbot cũ người vận hành drop tay.

## 9. Nghiệm thu thủ công

- Máy sạch: up tạo env/dừng; điền/chạy lại, services healthy, đúng CPU/GPU, không cổng ngoài loopback.
- URL trên điện thoại 4G: HTTPS/login, admin đầu rồi user thường, hỏi luật nhiều lượt.
- Ngoài mạng không gọi API/Redis/Postgres; down/up giữ account/history/cache/session key.
- Backup khôi phục được ở Postgres tạm; Redis tắt chat degrade/readyz 503; Postgres tắt chỉ UI ảnh hưởng.
- Image/history không chứa secret, user không root.
- NVIDIA host: container torch.cuda.is_available() True; nhiều lượt không CUDA OOM.

## 10. Rủi ro / điểm mở

- **1:** host tắt/ngủ/mất mạng → dừng; quick URL đổi; named/domain chưa có, chi phí từng ước khoảng 10
  USD/năm.
- **2:** signup mở/nhiều account có thể cạn quota; chưa Turnstile/Access; đóng signup/tunnel khi cần.
- **3:** host giữ hội thoại/trace, chưa tự xóa; tác giả chịu trách nhiệm mã hóa đĩa (BitLocker)/cập nhật OS.
- **4:** CPU-only chậm; up.sh tránh build/override lệch, lệnh Compose tay vẫn có thể lệch.
- **5:** quick tunnel dành thử nghiệm/không uptime; đối chiếu điều khoản trước chia sẻ rộng.
- **6:** image + checkpoint nặng; WSL2 RAM/disk cho năm service, gợi ý ≥6GB, thêm tài nguyên cho
  observe/hf_cache.
