# Deploy — Chạy toàn bộ chatbot trên máy cá nhân, public qua Cloudflare Tunnel

> Cô đọng 2026-09-30 (giữ số mục vì code/spec khác tham chiếu). Bản đầy đủ: git history.

## 1. Mục tiêu & phạm vi

Đóng gói và chạy cả hệ thống (`api/api_spec.md`) bằng **một lệnh** trên máy tác giả (Windows + WSL2), cho người khác truy cập qua URL
HTTPS của **Cloudflare Tunnel** — không VPS, không mở cổng router. Máy tắt thì dịch vụ tắt (chấp nhận, mục 10).

**Làm:** `deploy/` (ngoài `src/`): Dockerfile, compose, mẫu env, script khởi tạo DB, backup, `deploy/up.sh` (tự dò GPU, mục 4.1); 5
service `cloudflared`, `open-webui`, `api`, `redis`, `postgres` (Postgres chỉ cho OpenWebUI); **một entrypoint cho hai use case**: tác giả
chạy trên laptop (thường có GPU) rồi public URL, và người khác `git clone` chạy tự lo (thường CPU-only) — cùng một lệnh, không cần biết
trước có GPU; chính sách truy cập public, cô lập mạng, bí mật, vận hành cơ bản.
**Không làm:** VPS/PaaS, Kubernetes, Caddy/nginx, nhiều replica, HA; CD tự deploy (CI chỉ chạy test/lint; deploy là thao tác tay);
monitoring/alert trong compose production (stack observe là compose riêng `dev/observability/`, `observability_spec.md` mục 7); tự host
LLM/Pinecone. Reranker **không** còn là dịch vụ ngoài — chạy in-process trong `api` (`retrieval_spec.md` mục 6.1).

**Tiêu chí số 1:** clone + điền `.env` (root) + chạy → người bên ngoài mở URL, đăng ký, hỏi đáp nhiều lượt được; ngoài Cloudflare Tunnel
không có cổng nào lộ ra ngoài; bí mật không nằm trong image hay git.

## 2. Kiến trúc

Internet ─HTTPS─► Cloudflare ◄─(kết nối ra do `cloudflared` tự mở)─ `cloudflared` ─(mạng compose `internal`)─► `open-webui` ─► `postgres`
(DB openwebui); `open-webui` ─Bearer `CHATBOT_API_KEY`─► `api` ─► `redis`, reranker in-process (GPU khuyến nghị/CPU fallback), Groq/HF/Pinecone.
`cloudflared` chỉ **kết nối ra** nên không mở cổng vào; chỉ `cloudflared` nói chuyện được với `open-webui`, chỉ `open-webui` (và script vận
hành) nói chuyện được với `api`.

## 3. Cloudflare Tunnel

Hai chế độ, cùng service `cloudflared`, chọn bằng compose profile (`quick` mặc định, `named`):
- **Quick tunnel** (không cần tài khoản/domain): URL ngẫu nhiên `*.trycloudflare.com`, **đổi mỗi lần khởi động lại**, Cloudflare không cam
  kết uptime; `cloudflared tunnel --no-autoupdate --url http://open-webui:8080`, URL in ra `docker compose logs cloudflared`.
- **Named tunnel** (cần tài khoản + domain): URL cố định; `cloudflared tunnel --no-autoupdate run` với `TUNNEL_TOKEN` (`.env` root), trỏ
  hostname tới `http://open-webui:8080` trong dashboard. Chuyển quick → named **chỉ đổi cấu hình/biến**, không đổi code.
TLS kết thúc ở edge Cloudflare, trong mạng compose dùng HTTP. Đặt `WEBUI_URL` theo URL thật khi dùng named tunnel.

## 4. Services (`deploy/docker-compose.yml`)

Một network `internal` (bridge). **Không service nào publish cổng ra host**, trừ tuỳ chọn dev chỉ loopback `127.0.0.1:3000 → open-webui:8080`
và `127.0.0.1:8000 → api:8000`. `cloudflared` (image ghim tag; phụ thuộc `open-webui` healthy), `open-webui` (ghim tag; cấu hình
`api_spec.md` mục 9; volume `openwebui_data`), `api` (build `deploy/Dockerfile`; mount `data/bm25/` read-only + volume `hf_cache`; `env_file:
../.env`), `redis:7-alpine` (`--requirepass`, `--appendonly yes`, `--maxmemory 256mb --maxmemory-policy allkeys-lru`), `postgres:17-alpine`
(mount `deploy/initdb/`, chạy 1 lần lúc tạo volume).

Quy tắc chung: `restart: unless-stopped`; **ghim tag** (không `latest`), ghi phiên bản OpenWebUI đã nghiệm thu; healthcheck (`pg_isready`,
`redis-cli ping` có mật khẩu, `GET /readyz` bằng `python -c "urllib…"` không cần curl, health của open-webui) với `depends_on:
service_healthy` theo chuỗi `redis → api → open-webui → cloudflared` (`postgres → open-webui`; `api` không phụ thuộc `postgres`); log
`json-file` `max-size: 10m`, `max-file: 3`. **`mem_limit`:** `api` **3g** (đo thật 2026-09-27: 1.5g bị OOM-killer giết ngay lượt hỏi
retrieval+rerank đầu tiên — reranker model + CUDA context + tải checkpoint qua hf-xet cộng dồn ~1.53GB), `open-webui` 1g, `postgres` 512m,
`redis` 320m, để không nuốt hết RAM WSL2.

### 4.1 GPU passthrough cho reranker (tự động qua `deploy/up.sh`)

`api` chạy reranker in-process, tự phát hiện `cuda`/`cpu`. `deploy/up.sh` là entrypoint duy nhất, chạy `./deploy/up.sh` từ đâu cũng được (tự
`cd` vào `deploy/`). Cần hai việc tách biệt, script lo cả hai:

1. **Build đúng biến thể torch:** `deploy/Dockerfile` nhận build arg `TORCH_VARIANT` (mặc định `cpu` → `--index-url .../whl/cpu`; `cu126` trở
   lên → CUDA). **PHẢI `cu126` trở lên:** kênh `cu121`/`cu124` đã ngừng ở torch 2.5.1/2.6.0, không có wheel Python 3.14 (`cp314`) — Dockerfile
   dùng `python:3.14-slim` nên build lỗi "no wheels with matching Python ABI tag" nếu nhầm.
2. **Cấp GPU cho container:** `deploy.resources.reservations.devices` (driver nvidia, count 1, capabilities gpu) nằm trong file override riêng
   `deploy/docker-compose.gpu.yml` — **không sửa `docker-compose.yml` gốc** để máy không GPU vẫn `up -d` thẳng được. Cần NVIDIA Container
   Toolkit trên host (Docker Desktop WSL2 đã hỗ trợ sẵn, chỉ bật GPU support).

`up.sh` dò: **cả hai đúng** mới chạy bản GPU — `nvidia-smi` chạy được VÀ `docker info` báo runtime `nvidia`; thiếu 1 trong 2 → bản CPU (không
lỗi). CPU: `docker compose build --build-arg TORCH_VARIANT=cpu api && docker compose up -d`; GPU: `build --build-arg TORCH_VARIANT=cu126 api`
rồi `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`. Script tự build đúng biến thể khớp với việc có ghép override; lệch chỉ
xảy ra khi ai đó gọi tay hai lệnh lệch nhau (tự chịu rủi ro, mục 10.4). VRAM nhỏ (2GB) vẫn có thể CUDA OOM ở batch lớn → fallback
`rerank_score=None` (`retrieval_spec.md` mục 8), service không crash.

## 5. Chính sách truy cập public

- **Đăng ký mở (chốt 2026-09-21):** `ENABLE_SIGNUP=true`, `DEFAULT_USER_ROLE=user`; máy chạy mới có dịch vụ nên quy mô tự giới hạn, không muốn
  duyệt tay. **Tài khoản đầu tiên thường thành admin** (xác nhận khi chạy thử) → đăng ký admin **trước khi công bố URL**, rồi tắt tính năng
  admin không cần thiết (`api_spec.md` mục 9).
- Hạn mức Groq được bảo vệ bằng **ngân sách toàn cục** (`GLOBAL_DAILY_LLM_ANSWERS`), quota theo user/ngày và rate limit phút
  (`conversation_spec.md` mục 8, `api_spec.md` mục 5). **Một người tạo nhiều tài khoản vượt được quota theo user; chỉ ngân sách toàn cục chặn
  được** — khi cạn, mọi người nhận `quota_exceeded` (ưu tiên bảo vệ hạn mức).
- Bị lạm dụng: `ENABLE_SIGNUP=false` rồi `docker compose up -d open-webui`, hoặc tắt `cloudflared`.

## 6. Image API (`deploy/Dockerfile`)

Base `python:3.14-slim`; `uv` copy từ `ghcr.io/astral-sh/uv`; tầng dependency riêng (`pyproject.toml` + `uv.lock` → `uv sync --frozen --no-dev
--no-install-project`, rồi copy `src/`, `README.md` → cài project để tận dụng cache tầng). `torch` theo `TORCH_VARIANT` (mặc định `cpu` để build
được mọi máy, nhẹ). Checkpoint reranker (`AITeamVN/Vietnamese_Reranker`, ~1GB) tải từ HF Hub ở lần chạy đầu, **không bake vào image**, mount volume
`hf_cache` để không tải lại mỗi lần recreate. Chạy user không root. `data/bm25/` **không** vào image (file sinh ra, `.gitignore`), mount
read-only; thiếu file → `api` lỗi rõ lúc khởi động, không chạy nửa vời. Lệnh chạy `uvicorn production_legal_qa_rag.api.app:create_app --factory
--host 0.0.0.0 --port 8000 --workers 1` (**1 worker** vì semaphore admission in-process; không còn migration). `.dockerignore`: `.venv/`, `.git/`, `data/`,
`tests/`, `.env`, cache mypy/ruff/pytest.

## 7. Biến môi trường & bí mật (`.env` ở repo root, không commit)

**Chỉ 1 cặp `.env`/`.env.example` cho toàn project, ở repo root** (chốt 2026-09-29 để tránh rải rác nhiều `.env`). `api` container đọc nguyên file qua
`env_file: ../.env`, cùng file `config.py` đọc trên host. `.env.example` chia 3 block bằng comment: **APP** (dev-trên-host lẫn container `api`),
**DEPLOY** (chỉ container production), **OBSERVABILITY** (chỉ stack dev — `observability_spec.md` mục 7). Prefix `DEPLOY_` chỉ cho
`POSTGRES_USER`/`POSTGRES_PASSWORD` (Postgres riêng cho production, trùng tên với Postgres của OBSERVABILITY nếu không prefix).

Nhóm biến: LLM/dịch vụ (dùng chung APP) `GROQ_API_KEY_1`…`_4` (production; `_5`–`_9` chỉ evaluation), `HF_TOKEN`, `PINECONE_API_KEY`,
`PINECONE_INDEX_NAME`, `PINECONE_SPARSE_INDEX_NAME`, `CHATBOT_API_KEY`; backend `COMPOSE_PROFILES`, `REDIS_PASSWORD` (`REDIS_URL` cho container
do compose tự dựng từ `REDIS_PASSWORD`); Postgres `DEPLOY_POSTGRES_USER`/`DEPLOY_POSTGRES_PASSWORD`; OpenWebUI `WEBUI_SECRET_KEY` (**cố định** để phiên
đăng nhập không mất khi khởi động lại), `WEBUI_URL` (tuỳ chọn); tunnel `TUNNEL_TOKEN` (chỉ named). `.env` nằm trong `.gitignore` (bài học
2026-09-29: dòng `.env` từng bị comment nhầm, không thực sự ignore — đã bật lại); file khác chứa bí mật thì thêm `.gitignore` trước; **không truyền bí mật bằng
build args hay bake vào image**.

## 8. Vận hành cơ bản

- **Khởi động/dừng:** `./deploy/up.sh` (dò GPU, build đúng biến thể, `up -d`, tự in URL quick tunnel) / `docker compose down` (không `-v`, giữ volume;
  chạy trong `deploy/`). Lần đầu chưa có `.env`: script `cp ../.env.example ../.env` rồi dừng, điền giá trị thật rồi chạy lại. Named tunnel: script không dò
  URL (cố định theo `WEBUI_URL`); xem log tay `docker compose logs cloudflared-named`.
- **Máy Windows:** Docker Desktop (WSL2) bật cùng Windows; tắt sleep/hibernate khi cắm điện, nếu không tunnel đứt.
- **Backup:** `deploy/scripts/backup.sh` `pg_dump` DB `openwebui` ra `deploy/backups/<ngày>/` (`.gitignore`), giữ 7 bản; Redis không cần backup.
- **Cập nhật:** `git pull` → `./deploy/up.sh`; đổi phiên bản OpenWebUI: sửa tag, nghiệm thu lại mục 9.
- **Nhật ký:** `docker compose logs -f api`; phân tích từng lượt qua trace Langfuse (`observability_spec.md` mục 4.5). DB `chatbot` cũ (volume từ trước
  2026-09-29) không tự biến mất: `DROP DATABASE chatbot;`.

## 9. Nghiệm thu thủ công

1. Máy sạch: `./deploy/up.sh` → tự tạo `.env` rồi dừng; điền; chạy lại → mọi service `healthy`, không cổng nào lắng nghe ngoài `127.0.0.1`
   (`docker compose ps`, `ss -ltn`); log script in đúng nhánh CPU/GPU. 2. Lấy URL từ `cloudflared`, mở bằng điện thoại 4G: thấy trang đăng nhập HTTPS; đăng
   ký admin rồi 1 tài khoản thường; hỏi câu luật. 3. Từ ngoài mạng không truy cập được `api`, `redis`, `postgres`. 4. `down && up -d` → tài khoản, lịch sử,
   dòng cache còn; phiên đăng nhập không mất (`WEBUI_SECRET_KEY` cố định). 5. `backup.sh` tạo dump khôi phục được vào Postgres tạm. 6. Tắt Redis → chat vẫn
   trả lời, `/readyz` 503; tắt Postgres → OpenWebUI mất DB, `api` không ảnh hưởng. 7. `docker history`/`grep` không thấy bí mật; chạy không root. 8. (Máy có
   GPU NVIDIA) `docker compose exec api python -c "import torch; print(torch.cuda.is_available())"` trả `True`; hỏi nhiều lượt không thấy cảnh báo CUDA OOM.

## 10. Rủi ro / điểm mở

1. **Máy tắt/ngủ/mất mạng = dịch vụ ngừng;** quick tunnel đổi URL sau mỗi lần khởi động (cần named tunnel + domain ~10 USD/năm — chưa có). 2. Đăng ký mở + quota theo user không
   chặn được người tạo nhiều tài khoản; chỉ ngân sách toàn cục bảo vệ (mục 5); nếu vẫn lạm dụng: đóng đăng ký hoặc thêm Cloudflare Turnstile/Access — chưa làm. 3. Máy cá
   nhân chứa hội thoại của người dùng khác: mã hoá đĩa (BitLocker) + cập nhật OS là trách nhiệm tác giả; nội dung còn nằm trong trace Langfuse (chưa có xoá tự động,
   `observability_spec.md` mục 4.5). 4. Reranker in-process: image mặc định chỉ có `torch` CPU nên máy không GPU (hoặc thiếu Container Toolkit) rerank chậm hơn nhưng không phụ thuộc dịch
   vụ ngoài; `up.sh` tự dò nên hết rủi ro lệch tay, trừ khi ai đó gọi tay `docker compose` lệch kết quả dò. 5. Quick tunnel dành cho thử nghiệm (không cam kết uptime): đối
   chiếu điều khoản Cloudflare trước khi chia sẻ rộng. 6. Image nặng (`transformers`/`pyvi`/`torch` + ~1GB checkpoint lần đầu); RAM/disk WSL2 mặc định có thể thiếu cho 5 service:
   chỉnh `.wslconfig` (gợi ý ≥ 6 GB, cân nhắc thêm cho `hf_cache`).
