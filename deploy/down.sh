#!/usr/bin/env bash
# Dừng toàn bộ stack production (đối xứng với ./deploy/up.sh, deploy_spec.md mục 8). Giữ
# volume (tài khoản OpenWebUI, lịch sử chat, cache Redis) — KHÔNG có `-v`.
#
# Bắt buộc truyền --env-file (compose không tự dò .env ở repo root) và --profile '*' (nếu
# thiếu, cloudflared thuộc profile quick/named bị bỏ qua nên còn chạy và giữ network).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

env_file="../.env"
if [[ ! -f "${env_file}" ]]; then
    echo "Không thấy .env ở repo root — không biết stack nào để dừng." >&2
    exit 1
fi

docker compose --env-file "${env_file}" -f docker-compose.yml --profile '*' down
