#!/usr/bin/env bash
# Bật stack observability (Langfuse + Prometheus + Grafana), thứ tự đúng với production:
#   1) ./observability/up.sh   2) tạo API key trên Langfuse UI (lần đầu) rồi điền vào .env
#   3) ./deploy/up.sh — nối `api` vào network `legal-qa-observe` do stack này tạo.
# Chạy lại nhiều lần an toàn: image/volume/cache đã có thì dùng lại, không tải lại, không
# mất dữ liệu (observability_spec.md mục 7).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

env_file="../.env"
if [[ ! -f "${env_file}" ]]; then
    echo "Không thấy .env ở repo root — copy từ .env.example và điền các biến OBS_* trước." >&2
    exit 1
fi

docker compose --env-file "${env_file}" -f docker-compose.yml up -d --wait --wait-timeout 300

echo
echo "Stack observe đã chạy:"
echo "  Langfuse   http://localhost:3001"
echo "  Prometheus http://localhost:9092"
echo "  Grafana    http://localhost:3002"

# Chỉ kiểm tra dòng có giá trị hay không (không in ra) — key chưa có thì api chạy không trace.
if ! grep -Eq '^LANGFUSE_PUBLIC_KEY=.+' "${env_file}" || ! grep -Eq '^LANGFUSE_SECRET_KEY=.+' "${env_file}"; then
    echo
    echo "Chưa có LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY trong .env: mở Langfuse, tạo project + API key," >&2
    echo "điền vào .env rồi chạy ./deploy/up.sh." >&2
else
    echo
    echo "Đã có key Langfuse trong .env — chạy (hoặc chạy lại) ./deploy/up.sh để nối api vào."
fi
