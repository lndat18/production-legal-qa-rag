"""Bảo vệ hành vi `alembic/env.py` đọc `CHATLOG_DATABASE_URL` lúc runtime (deploy_spec.md
mục 6): bắt buộc để `alembic upgrade head` trong container `api` trỏ đúng tới host
`postgres` của mạng compose, thay vì giá trị tĩnh `localhost` ghi cứng trong `alembic.ini`.

Không dùng DB thật: chạy `alembic upgrade head --sql` (chế độ offline, chỉ sinh SQL, không
mở connection — xem docstring `run_migrations_offline` trong `alembic/env.py`). Để phân
biệt "override có hiệu lực" với "vẫn dùng giá trị tĩnh trong alembic.ini" (2 giá trị mặc
định trùng nhau nên không thể so sánh bằng SQL output), test đặt `CHATLOG_DATABASE_URL`
với một dialect KHÔNG tồn tại — alembic bắt buộc phải resolve dialect từ URL để render SQL
ngay cả ở chế độ offline, nên chỉ lỗi đúng theo giá trị đã ghi đè mới xảy ra được.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_alembic_env_ghi_de_sqlalchemy_url_bang_chatlog_database_url() -> None:
    """`alembic/env.py` phải ưu tiên `CHATLOG_DATABASE_URL` từ môi trường (không chỉ ini)."""
    env = os.environ.copy()
    env["CHATLOG_DATABASE_URL"] = "notarealdialect+driver://user:pass@example-host/db"

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode != 0, (
        "Kỳ vọng lỗi resolve dialect 'notarealdialect' — nếu lệnh PASS nghĩa là env.py "
        "không còn đọc CHATLOG_DATABASE_URL nữa (đã quay lại dùng giá trị tĩnh trong "
        "alembic.ini), vi phạm deploy_spec.md mục 6."
    )
    assert "notarealdialect" in result.stderr, result.stderr
