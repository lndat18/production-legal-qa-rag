#!/usr/bin/env bash
# Entrypoint duy nhất để khởi động toàn bộ stack (deploy_spec.md mục 4.1, mục 8): tự dò
# GPU NVIDIA dùng được cho Docker hay không, rồi build đúng biến thể torch + ghép đúng file
# compose. Dùng chung cho cả tác giả tự chạy (máy có GPU) và người khác clone project chạy
# trên máy của họ (thường CPU-only) — luôn cùng một lệnh: ./deploy/up.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "Chưa có deploy/.env — đã tạo từ .env.example. Điền giá trị thật vào deploy/.env rồi chạy lại ./deploy/up.sh" >&2
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
    torch_variant=cu121
    compose_files+=(-f docker-compose.gpu.yml)
else
    echo "Không phát hiện GPU NVIDIA sẵn dùng cho Docker — build/chạy bản CPU."
fi

docker compose "${compose_files[@]}" build --build-arg "TORCH_VARIANT=${torch_variant}" api
docker compose "${compose_files[@]}" up -d
