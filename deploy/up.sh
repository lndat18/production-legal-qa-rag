#!/usr/bin/env bash
# Entrypoint duy nhất để khởi động toàn bộ stack (deploy_spec.md mục 4.1, mục 8): tự dò
# GPU NVIDIA dùng được cho Docker hay không, rồi build đúng biến thể torch + ghép đúng file
# compose. Dùng chung cho cả tác giả tự chạy (máy có GPU) và người khác clone project chạy
# trên máy của họ (thường CPU-only) — luôn cùng một lệnh: ./deploy/up.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -f .env ]]; then
    cp .env.example .env

    # Root `.env` (dev cục bộ, .env.example ở repo root) và `deploy/.env` là 2 file tách
    # riêng có chủ đích (deploy_spec.md mục 7 — cách ly bí mật production khỏi key dev cá
    # nhân), nhưng vài key trùng tên/giá trị (LLM + CHATBOT_API_KEY). Tự điền sẵn từ root
    # .env nếu có, để không phải gõ lại — chỉ copy nguyên dòng chữ (không source/eval), an
    # toàn với ký tự đặc biệt trong giá trị, cùng cách né rủi ro parse .env bằng bash như
    # backup.sh đã áp dụng.
    root_env="../.env"
    if [[ -f "${root_env}" ]]; then
        shared_keys=(GROQ_API_KEY GROQ_API_KEY_2 GROQ_API_KEY_3 GROQ_JUDGE_API_KEY HF_TOKEN PINECONE_API_KEY PINECONE_INDEX_NAME PINECONE_SPARSE_INDEX_NAME CHATBOT_API_KEY)
        for key in "${shared_keys[@]}"; do
            line="$(grep -E "^${key}=.+" "${root_env}" || true)"
            [[ -n "${line}" ]] || continue
            grep -v -E "^${key}=" .env > .env.tmp && mv .env.tmp .env
            printf '%s\n' "${line}" >> .env
        done
        echo "Đã điền sẵn vào deploy/.env các key trùng root .env (GROQ_API_KEY*, HF_TOKEN, PINECONE_*, CHATBOT_API_KEY nếu có giá trị)." >&2
    fi

    echo "Chưa có deploy/.env — đã tạo từ .env.example. Điền các giá trị deploy-only còn thiếu (POSTGRES_USER/PASSWORD, REDIS_PASSWORD, WEBUI_SECRET_KEY, ...) vào deploy/.env rồi chạy lại ./deploy/up.sh" >&2
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

# Chỉ định rõ --env-file thay vì để Compose tự dò .env theo cwd — tránh phụ thuộc hành vi
# auto-detect (có thể khác nhau giữa các phiên bản Compose).
docker compose --env-file .env "${compose_files[@]}" build --build-arg "TORCH_VARIANT=${torch_variant}" api
docker compose --env-file .env "${compose_files[@]}" up -d

# Chỉ quick tunnel mới cần in URL (URL ngẫu nhiên, đổi mỗi khi container restart — mục 3);
# named tunnel dùng domain cố định đã cấu hình sẵn trong WEBUI_URL, không cần dò. Đọc thẳng
# giá trị (không source/eval, cùng cách né parse .env bằng bash như trên) — không set thì
# coi như mặc định "quick" (khớp docker-compose.yml).
compose_profile="$(grep -E '^COMPOSE_PROFILES=' .env | cut -d= -f2- || true)"
compose_profile="${compose_profile:-quick}"

if [[ "${compose_profile}" == "quick" ]]; then
    echo "Đang chờ URL từ Cloudflare quick tunnel..."
    tunnel_url=""
    for _ in $(seq 1 15); do
        # `|| true` bắt buộc: grep không khớp gì (bình thường ở vài vòng đầu, log chưa kịp
        # có URL) trả exit code 1, cộng `pipefail` sẽ làm cả pipeline coi như lỗi và `set -e`
        # giết luôn script ngay vòng đầu tiên nếu không chặn lại.
        tunnel_url="$(docker compose --env-file .env "${compose_files[@]}" logs cloudflared-quick 2>/dev/null \
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
