#!/usr/bin/env bash
# Entrypoint duy nhất để khởi động toàn bộ stack (deploy_spec.md mục 4.1, mục 8): tự dò
# GPU NVIDIA dùng được cho Docker hay không, rồi build đúng biến thể torch + ghép đúng file
# compose. Dùng chung cho cả tác giả tự chạy (máy có GPU) và người khác clone project chạy
# trên máy của họ (thường CPU-only) — luôn cùng một lệnh: ./deploy/up.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Đúng 1 file .env ở root cho cả app/deploy/observability (deploy_spec.md mục 7) — không
# còn deploy/.env riêng.
env_file="../.env"
if [[ ! -f "${env_file}" ]]; then
    cp ../.env.example "${env_file}"
    echo "Chưa có .env ở repo root — đã tạo từ .env.example. Điền giá trị thật (GROQ_API_KEY_1*, DEPLOY_POSTGRES_USER/PASSWORD, REDIS_PASSWORD, WEBUI_SECRET_KEY, ...) rồi chạy lại ./deploy/up.sh" >&2
    exit 1
fi

# Cả 2 điều kiện đúng mới coi là có GPU dùng được cho Docker; thiếu 1 trong 2 vẫn chạy
# được, chỉ rơi về CPU (deploy_spec.md mục 4.1) — không lỗi, không cần biết trước máy có
# GPU hay không.
compose_files=(-f docker-compose.yml)
torch_variant=cpu

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1 \
    && docker info 2>/dev/null | grep -qi nvidia; then
    echo "Phát hiện GPU NVIDIA + Container Toolkit — build/chạy bản GPU (docker-compose.gpu.yml)."
    # cu121 KHÔNG có wheel cho Python 3.14 (kênh CUDA cũ, đã ngừng cập nhật ở torch 2.5.1,
    # trước khi cp314 tồn tại) — dùng cu126 trở lên. Xem deploy_spec.md mục 4.1.
    torch_variant=cu126
    compose_files+=(-f docker-compose.gpu.yml)
else
    echo "Không phát hiện GPU NVIDIA sẵn dùng cho Docker — build/chạy bản CPU."
fi

# Stack observability dev (dev/observability/) đang chạy thì nối `api` vào để trace/track
# traffic end-user thật; chưa chạy thì bỏ qua, production vẫn dựng bình thường.
if docker network inspect legal-qa-observe >/dev/null 2>&1; then
    echo "Phát hiện stack observability đang chạy — nối api vào để trace/metrics (docker-compose.observe.yml)."
    compose_files+=(-f docker-compose.observe.yml)
else
    echo "Stack observability chưa chạy — api chạy không trace (bật: docker compose -f dev/observability/docker-compose.yml up -d rồi chạy lại ./deploy/up.sh)."
fi

# Chỉ định rõ --env-file thay vì để Compose tự dò .env theo cwd — tránh phụ thuộc hành vi
# auto-detect (có thể khác nhau giữa các phiên bản Compose).
docker compose --env-file "${env_file}" "${compose_files[@]}" build --build-arg "TORCH_VARIANT=${torch_variant}" api
docker compose --env-file "${env_file}" "${compose_files[@]}" up -d

# Chỉ quick tunnel mới cần in URL (URL ngẫu nhiên, đổi mỗi khi container restart — mục 3);
# named tunnel dùng domain cố định đã cấu hình sẵn trong WEBUI_URL, không cần dò. Đọc thẳng
# giá trị (không source/eval, cùng cách né parse .env bằng bash như trên) — không set thì
# coi như mặc định "quick" (khớp docker-compose.yml).
compose_profile="$(grep -E '^COMPOSE_PROFILES=' "${env_file}" | cut -d= -f2- || true)"
compose_profile="${compose_profile:-quick}"

if [[ "${compose_profile}" == "quick" ]]; then
    echo "Đang chờ URL từ Cloudflare quick tunnel..."
    tunnel_url=""
    for _ in $(seq 1 15); do
        # `|| true` bắt buộc: grep không khớp gì (bình thường ở vài vòng đầu, log chưa kịp
        # có URL) trả exit code 1, cộng `pipefail` sẽ làm cả pipeline coi như lỗi và `set -e`
        # giết luôn script ngay vòng đầu tiên nếu không chặn lại.
        tunnel_url="$(docker compose --env-file "${env_file}" "${compose_files[@]}" logs cloudflared-quick 2>/dev/null \
            | grep -oE 'https://[A-Za-z0-9.-]+\.trycloudflare\.com' | tail -1 || true)"
        [[ -n "${tunnel_url}" ]] && break
        sleep 2
    done
    if [[ -n "${tunnel_url}" ]]; then
        echo "URL public: ${tunnel_url}"
    else
        echo "Chưa thấy URL sau ~30s — kiểm tra tay: docker compose logs cloudflared-quick" >&2
    fi
fi
