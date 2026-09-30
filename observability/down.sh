#!/usr/bin/env bash
# Dừng stack observability, GIỮ volume (trace Langfuse, dashboard Grafana, dữ liệu Prometheus) —
# không có `-v`. Xoá sạch dữ liệu observe (không hoàn tác): tự chạy
# `docker compose -f observability/docker-compose.yml --env-file .env down -v`.
#
# Dừng stack khi production còn chạy thì `api` vẫn trả lời bình thường, chỉ mất trace/metrics
# (observability_spec.md mục 1: instrumentation không làm hỏng câu trả lời).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

env_file="../.env"
if [[ ! -f "${env_file}" ]]; then
    echo "Không thấy .env ở repo root — không biết stack nào để dừng." >&2
    exit 1
fi

docker compose --env-file "${env_file}" -f docker-compose.yml down
