#!/usr/bin/env bash
# Backup 2 database Postgres (openwebui, chatbot) ra deploy/backups/<ngày>/, giữ 7 bản gần
# nhất (deploy_spec.md mục 8). Chạy tay hoặc qua cron/Task Scheduler, không cần dừng dịch
# vụ. Redis không backup (dữ liệu tính lại được).
#
# Dùng:
#   deploy/backup.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
ENV_FILE="${SCRIPT_DIR}/.env"
BACKUPS_DIR="${SCRIPT_DIR}/backups"
KEEP=7

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Không tìm thấy ${ENV_FILE} — copy từ .env.example và điền giá trị trước." >&2
    exit 1
fi

compose() {
    docker compose --file "${COMPOSE_FILE}" --env-file "${ENV_FILE}" "$@"
}

today="$(date +%Y-%m-%d)"
out_dir="${BACKUPS_DIR}/${today}"
mkdir -p "${out_dir}"

# $POSTGRES_USER được giãn bên trong container postgres (đã có sẵn trong environment của
# service, mục 4), không đọc/parse deploy/.env bằng bash — tránh rủi ro mật khẩu chứa ký tự
# đặc biệt bị shell trên host diễn giải sai khi source trực tiếp.
for db in openwebui chatbot; do
    echo "Dump database '${db}' -> ${out_dir}/${db}.sql.gz"
    compose exec -T postgres sh -c "pg_dump -U \"\$POSTGRES_USER\" --dbname ${db}" \
        | gzip > "${out_dir}/${db}.sql.gz"
done

# Giữ 7 bản gần nhất (theo tên thư mục ngày, sort giảm dần), xoá phần còn lại.
mapfile -t old_dirs < <(find "${BACKUPS_DIR}" -mindepth 1 -maxdepth 1 -type d | sort -r | tail -n "+$((KEEP + 1))")
for dir in "${old_dirs[@]:-}"; do
    [[ -n "${dir}" ]] || continue
    echo "Xoá backup cũ: ${dir}"
    rm -rf -- "${dir}"
done

echo "Backup xong: ${out_dir}"
