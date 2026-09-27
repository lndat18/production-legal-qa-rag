#!/usr/bin/env bash
# Xoá cache Redis (answer + retrieval) để lượt hỏi tiếp theo chạy lại pipeline từ đầu — dùng
# khi đang thử nghiệm generation/conversation và muốn thấy thay đổi ngay, không đợi TTL (7
# ngày/24 giờ, cache_spec.md mục 5) hay đổi PROMPT_VERSION.
#
# Không tự gọi up.sh (giữ nút này nhanh, không rebuild/restart) — cần container `api` đang
# chạy VÀ đã có mount tools/ (áp dụng từ lần `./deploy/up.sh` gần nhất sau khi thêm mount này
# vào docker-compose.yml). Nếu lệnh dưới báo lỗi "tools/cache.py": No such file, chạy
# ./deploy/up.sh một lần rồi thử lại — sau đó không cần lặp lại bước đó nữa.
#
# Dùng:
#   deploy/reset_cache.sh                          # xoá toàn bộ cache
#   deploy/reset_cache.sh --pattern 'rag:ans:*'    # chỉ xoá answer cache
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
docker compose --env-file .env -f docker-compose.yml exec -T api \
    python tools/cache.py flush --yes "$@"
