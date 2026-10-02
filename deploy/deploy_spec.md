# Deploy — Chạy toàn bộ chatbot trên máy cá nhân, public qua Cloudflare Tunnel

- Giữ nguyên số mục để không làm hỏng tham chiếu từ code/spec khác; bản đầy đủ ở git history.
- Spec liên quan: [api_spec.md](../src/production_legal_qa_rag/api/api_spec.md), [conversation_spec.md](../src/production_legal_qa_rag/conversation/conversation_spec.md), [observability_spec.md](../src/production_legal_qa_rag/observability/observability_spec.md).

## 1. Mục tiêu & phạm vi

- Một lệnh chạy toàn chatbot trên máy cá nhân Windows/WSL2, public HTTPS qua Cloudflare Tunnel; không VPS/mở cổng router.
- Cùng entrypoint cho laptop tác giả có GPU và người clone CPU-only; không cần chọn biến thể trước.
- Làm: Dockerfile/Compose/scripts trong deploy, năm service cloudflared/open-webui/api/redis/postgres, bảo vệ mạng/bí mật, backup/vận hành.
- Postgres chỉ OpenWebUI; reranker in-process trong API (retrieval 6.1).
- Không làm: VPS/PaaS/Kubernetes/Caddy/nginx, replica/HA, CD tự deploy (chỉ build+push image lên GHCR, mục 11), monitoring/alert trong compose production, host LLM/Pinecone.
- Observe ở compose riêng observability (observability spec 7); CI test/lint, deploy tay.
- Acceptance: clone → điền root env → chạy → người ngoài đăng ký/hỏi nhiều lượt; không lộ cổng/bí mật; host tắt thì dịch vụ tắt.

## 2. Kiến trúc

- Internet HTTPS → Cloudflare edge → cloudflared → OpenWebUI → Postgres DB openwebui.
- OpenWebUI Bearer chatbot key → API → Redis, local reranker, Groq/HF/Pinecone.
- cloudflared mở kết nối ra; không mở cổng vào; UI/backend trao đổi mạng compose.
- API chỉ cho client nội bộ có key và script vận hành.

## 3. Cloudflare Tunnel

- Profile quick mặc định: không account/domain, URL ngẫu nhiên trycloudflare.com, đổi khi restart, không cam kết uptime.
- Quick command: `cloudflared tunnel --no-autoupdate --url http://open-webui:8080`; URL trong log.
- Named profile: account/domain, URL cố định; `cloudflared tunnel --no-autoupdate run`, TUNNEL_TOKEN root env; dashboard trỏ UI:8080.
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
- Build arg TORCH_VARIANT: mặc định cpu, GPU cu126 trở lên; cu121/cu124 dừng torch 2.5.1/2.6.0, không wheel cp314.
- GPU override deploy/docker-compose.gpu.yml: reservation driver nvidia/count 1/capabilities gpu; không đặt GPU bắt buộc trong compose gốc.
- Host cần NVIDIA Container Toolkit hoặc Docker Desktop WSL2 GPU support.
- Chỉ dùng GPU khi nvidia-smi chạy được và docker info có runtime nvidia; thiếu một điều kiện → CPU, không fail.
- CPU: build TORCH_VARIANT=cpu rồi up; GPU: build cu126 và ghép override khi up.
- Script giữ build/override khớp; gọi tay lệch có rủi ro (10.4).
- VRAM 2GB có thể OOM batch lớn → fallback `rerank_score=None` (retrieval 8), không crash service.

## 5. Chính sách truy cập public

- Đăng ký mở: ENABLE_SIGNUP=true, DEFAULT_USER_ROLE=user; không duyệt tay.
- Tài khoản đầu thường là admin: đăng ký trước công bố URL, kiểm khi nghiệm thu, tắt tính năng thừa (API 9).
- Bảo vệ Groq: cache, rate limit phút, admission đồng thời, throttle bước nhẹ và 429 thật (conversation 9/12.1, API 5).
- Không có quota user/ngày hoặc GLOBAL_DAILY_LLM_ANSWERS; nhiều account có thể vượt rate limit theo user và làm cạn provider quota.
- Khi provider hết hạn mức: thông báo rate_limited/retry-after; không tự dựng ngân sách ngày riêng.
- Lạm dụng: tắt signup bằng Admin Panel hoặc cấu hình PersistentConfig phù hợp (API 9), recreate UI; hoặc dừng tunnel.

## 6. Image API (`deploy/Dockerfile`)

- Base python:3.14-slim, uv từ ghcr.io/astral-sh/uv; torch theo TORCH_VARIANT, CPU mặc định.
- Tầng deps riêng pyproject.toml/uv.lock → uv sync --frozen --no-dev --no-install-project → copy src/README → cài project.
- RUN uv dùng --mount=type=cache: đổi dependency không tải lại torch; không prune builder cache tùy tiện.
- Reranker checkpoint khoảng 1GB tải HF lần đầu, không bake image; hf_cache giữ qua recreate.
- User không root; BM25 không trong image, mount read-only; thiếu file → lỗi startup rõ.
- Uvicorn `production_legal_qa_rag.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 1`; không migration; admission in-process.
- dockerignore: .venv/.git/data/tests/.env và mypy/ruff/pytest cache.

## 7. Biến môi trường & bí mật (`.env` ở repo root, không commit)

- Duy nhất root .env/.env.example; API container env_file ../.env, host config đọc cùng file.
- Example có ba block APP/DEPLOY/OBSERVABILITY; prefix DEPLOY_/OBS_ cho biến trùng giữa stack.
- APP: `GROQ_API_KEY_1`…9 (1–4 production, 5–9 eval), `HF_TOKEN`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_SPARSE_INDEX_NAME`, `CHATBOT_API_KEY`.
- Deploy: COMPOSE_PROFILES, REDIS_PASSWORD (Compose dựng container REDIS_URL), `DEPLOY_POSTGRES_USER`/`DEPLOY_POSTGRES_PASSWORD` → container `POSTGRES_USER`/`POSTGRES_PASSWORD`.
- UI: WEBUI_SECRET_KEY cố định qua restart, WEBUI_URL tùy chọn; named tunnel TUNNEL_TOKEN.
- Observe là stack riêng cho traffic production (observability 7), không phục vụ end-user trực tiếp.
- .env phải được gitignore; file bí mật khác cũng ignore trước; không truyền secret qua build args hoặc bake image.

## 8. Vận hành cơ bản

- Start/update: ./deploy/up.sh; dò GPU/build/up/in quick URL, tự ghép observe nếu network tồn tại.
- Stop: ./deploy/down.sh, Compose --env-file ../.env --profile '*' down; không -v, giữ volume, không để tunnel sót.
- Thiếu .env lần đầu: copy example rồi dừng để điền; named URL theo WEBUI_URL, xem log cloudflared-named nếu cần.
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

- **1:** host tắt/ngủ/mất mạng → dừng; quick URL đổi; named/domain chưa có, chi phí từng ước khoảng 10 USD/năm.
- **2:** signup mở/nhiều account có thể cạn quota; chưa Turnstile/Access; đóng signup/tunnel khi cần.
- **3:** host giữ hội thoại/trace, chưa tự xóa; tác giả chịu trách nhiệm mã hóa đĩa (BitLocker)/cập nhật OS.
- **4:** CPU-only chậm; up.sh tránh build/override lệch, lệnh Compose tay vẫn có thể lệch.
- **5:** quick tunnel dành thử nghiệm/không uptime; đối chiếu điều khoản trước chia sẻ rộng.
- **6:** image + checkpoint nặng; WSL2 RAM/disk cho năm service, gợi ý ≥6GB, thêm tài nguyên cho observe/hf_cache.

## 11. CD: build + push image lên GHCR (chốt 2026-10-02, chưa implement)

Số mục 1–10 giữ nguyên; mục này chỉ thêm. Mục 1 "Không làm: CD tự deploy" vẫn đúng: CD ở đây chỉ **xuất bản image**, máy nhà vẫn cập nhật tay.

### 11.1 Mục tiêu, phạm vi, KHÔNG làm

- Mục tiêu: có image `api` dựng sẵn trên GHCR theo phiên bản, để máy khác (hoặc máy tác giả) `pull` thay vì build ~GB torch.
- Làm: workflow `.github/workflows/release.yml`; thêm `image:` cho service `api` trong compose; chế độ pull trong `up.sh`; cập nhật mục 6/8 khi implement.
- KHÔNG làm: SSH/self-hosted runner/tự deploy vào máy nhà; build mỗi push main hoặc PR; image cho open-webui/postgres/redis/observability; Watchtower/auto-update; ký image (cosign), SBOM/provenance (`provenance: false`); multi-arch (chỉ `linux/amd64`); bake model reranker, `data/bm25` hay secret vào image (mục 6, 7).

### 11.2 Trigger & điều kiện

- Trigger: `push` tag khớp `v[0-9]+.[0-9]+.[0-9]+` (không pre-release). Không có `push: branches`, không `pull_request`.
- Job đầu `gate` (fail thì không build), 2 kiểm tra:
  1. Commit của tag là tổ tiên của `origin/main` (`git merge-base --is-ancestor`), chặn tag trên branch chưa merge.
  2. Check run tên `checks` (CI, `ci.yml`) trên đúng SHA đó có `conclusion=success`, tra bằng `gh api repos/{repo}/commits/{sha}/check-runs`. Không dùng `workflow_run` vì trigger của nó không gắn với tag. Chưa có kết quả/đang chạy → fail với thông báo "chờ CI xanh rồi push lại tag" (xoá tag + tạo lại hoặc `gh run rerun`); không poll chờ.
- CI hiện chạy trên push main nên commit merge squash luôn có check `checks` (kể cả khi skip nặng, job vẫn xanh).
- `concurrency: group=release-${{ github.ref }}`, không cancel dở build.

### 11.3 Image & tag

- Ảnh: `ghcr.io/lndat18/production-legal-qa-rag` (viết thường bắt buộc).
- Matrix 2 biến thể, `fail-fast: false`, build tuần tự hoặc song song đều được (mặc định song song; dọn disk ở từng job):

| Biến thể | `TORCH_VARIANT` | Tag |
| --- | --- | --- |
| cpu | `cpu` | `vX.Y.Z-cpu`, `latest` |
| cu126 | `cu126` | `vX.Y.Z-cu126` |

- `latest` luôn trỏ bản cpu của tag mới nhất (không có `latest-cu126`; GPU phải ghim version). Không có tag `vX.Y.Z` trần, không `vX.Y`/`vX`.
- Tag tạo bằng `docker/metadata-action` hoặc tự ghép trong workflow; label `org.opencontainers.image.source=<repo url>` bắt buộc để GHCR gắn package vào repo.
- Tag phiên bản bất biến theo quy ước (không push đè); sai thì tăng patch.

### 11.4 Workflow (công cụ)

- `permissions` tối thiểu: `contents: read`, `packages: write`; đăng nhập `docker/login-action` với `${{ secrets.GITHUB_TOKEN }}` (không PAT, không secret mới).
- Các bước: `actions/checkout` → dọn disk (chỉ biến thể cu126; cpu bỏ qua) → `docker/setup-buildx-action` → login → `docker/build-push-action` (`context: .`, `file: deploy/Dockerfile`, `build-args: TORCH_VARIANT=...`, `push: true`, `platforms: linux/amd64`).
- Cache: `cache-from/to: type=gha,scope=<biến thể>,mode=min` (scope riêng mỗi biến thể để 2 bản không ghi đè nhau). Chú ý: `RUN --mount=type=cache` (mục 6) **không** được xuất vào cache gha, chỉ cache layer; lần build lẻ vẫn tải lại torch nếu layer deps bị vô hiệu.
- Dọn disk trước build cu126: xoá `/usr/share/dotnet`, `/usr/local/lib/android`, `/opt/ghc`, `/opt/hostedtoolcache/CodeQL`, `docker image prune -af` (script `rm` ngắn trong workflow; không thêm action bên thứ ba chỉ để dọn disk). Ghi `df -h` trước/sau để đo thật.
- Pin action theo major tag (`@v5`... như `ci.yml`); không pin SHA (over-engineering cho dự án cá nhân).

### 11.5 Compose & `up.sh`

- `docker-compose.yml`, service `api`: thêm `image: ${API_IMAGE:-legal-qa-api:local}` cạnh `build:`; không build thì Compose pull `API_IMAGE`, có build thì gắn tên local. Không đổi `args.TORCH_VARIANT: cpu` mặc định.
- `up.sh`: mặc định giữ nguyên hành vi (dò GPU → build → up). Thêm cờ `--pull <vX.Y.Z>`:
  - Vẫn dò GPU như 4.1 để chọn hậu tố `-cu126` (và ghép `docker-compose.gpu.yml`) hay `-cpu`.
  - Đặt `API_IMAGE=ghcr.io/lndat18/production-legal-qa-rag:<version>-<variant>` (export trong script, không ghi vào `.env`), **bỏ bước build**, chạy `docker compose pull api` rồi `up -d --no-build`.
  - Version bắt buộc (không mặc định `latest`, vì `latest` chỉ có cpu và sẽ lệch GPU); thiếu/sai định dạng → in usage, exit 1.
  - Phần còn lại (ghép observe, in quick URL) giữ nguyên.
- Pull thất bại (chưa public/chưa có tag/chưa login) → script dừng với thông báo rõ, KHÔNG tự rơi về build.

### 11.6 Tiêu chí hoàn thành

- Push tag `v0.1.0` trên commit main đã xanh → workflow tạo đúng 3 tag: `v0.1.0-cpu`, `v0.1.0-cu126`, `latest`; push tag trên commit chưa có CI xanh hoặc không thuộc main → `gate` fail, không image nào được push.
- Máy khác (đã đăng xuất GHCR, package public): `./deploy/up.sh --pull v0.1.0` chạy được stack, `readyz` ok, đúng biến thể theo GPU; máy GPU: `torch.cuda.is_available()` True trong container bản cu126.
- `./deploy/up.sh` không cờ vẫn build local như trước (không hồi quy), tên image local `legal-qa-api:local`.
- Image pull về không chứa secret, user không root (mục 9); BM25 vẫn mount lúc chạy.
- Nghiệm thu bằng tag thật (không thử bằng workflow_dispatch/giả lập); lần đầu dùng tag `v0.1.0`, lỗi thì sửa workflow rồi xoá tag + tag lại (chưa ai pull nên chấp nhận được ở phiên bản đầu).

### 11.7 Rủi ro / điểm mở

- **7 — Dung lượng runner:** runner `ubuntu-latest` ~14GB trống; image cu126 + wheel nvidia có thể vượt khi build. Giảm bằng dọn disk (11.4), đo `df -h`; nếu vẫn thiếu thì chỉ làm gọn tầng deps chứ không đổi runner trả phí.
- **8 — Cache gha:** hạn mức 10GB/repo, cache torch có thể làm evict lẫn nhau; chấp nhận build chậm hơn, không thêm registry cache.
- **9 — `data/bm25` & model:** không nằm trong image; máy chạy pull vẫn cần `git clone` repo (compose, `data/bm25`, `tools/`) và `.env`; `up.sh --pull` không thay thế bước đó. Reranker vẫn tải HF lần đầu (mục 6).
- **10 — Package private mặc định:** lần push đầu GHCR tạo package private; **phải tự set Public thủ công** (GitHub → Packages → Package settings → Change visibility), và kiểm tra "Manage Actions access" cho repo có Write nếu push 403. Không tự động hoá được bằng GITHUB_TOKEN. Chưa public thì `--pull` cần `docker login ghcr.io` bằng PAT `read:packages`.
- **11 — Không tự cập nhật:** tag mới không đụng máy đang chạy; cập nhật vẫn là `git pull` + `./deploy/up.sh --pull <version>` (hoặc build). Compose + image phải cùng phiên bản repo, nên pull đúng version của commit đã checkout.
- **12 — Tag sai/lệch CI:** gate chỉ xác thực check `checks`; tag trên commit chỉ đổi `.md` vẫn qua gate (không rủi ro vì CI vẫn xanh).
