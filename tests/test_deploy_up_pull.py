"""Kiểm tra `deploy/up.sh --pull vX.Y.Z` (deploy_spec.md mục 11.5).

Không cần Docker thật: script chạy trong thư mục tạm với `docker` giả (ghi lại lời gọi vào
file log) đặt đầu PATH; `.env` thật ở root không bao giờ bị đọc/ghi.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
UP_SH = REPO_ROOT / "deploy" / "up.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="cần bash")

# Docker giả: ghi mỗi lời gọi (kèm API_IMAGE) vào $FAKE_DOCKER_LOG. `pull` thất bại nếu
# FAKE_DOCKER_PULL_FAIL=1; `network inspect` luôn lỗi (coi như không có stack observe);
# `info` không in gì (không có runtime nvidia -> biến thể cpu).
_FAKE_DOCKER = """#!/usr/bin/env bash
echo "API_IMAGE=${API_IMAGE:-} :: $*" >> "$FAKE_DOCKER_LOG"
case " $* " in
    *" network inspect "*) exit 1 ;;
    *" pull "*) [[ "${FAKE_DOCKER_PULL_FAIL:-0}" == "1" ]] && exit 1 ;;
esac
exit 0
"""


def _run_up(
    tmp_path: Path, args: list[str], *, pull_fail: bool = False
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    shutil.copy(UP_SH, deploy / "up.sh")
    (tmp_path / ".env").write_text("COMPOSE_PROFILES=named\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "docker"
    fake.write_text(_FAKE_DOCKER, encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    # Chặn nhận diện GPU: nvidia-smi giả luôn lỗi.
    smi = bin_dir / "nvidia-smi"
    smi.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    smi.chmod(smi.stat().st_mode | stat.S_IXUSR)

    log = tmp_path / "docker.log"
    log.write_text("", encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_DOCKER_LOG": str(log),
        "FAKE_DOCKER_PULL_FAIL": "1" if pull_fail else "0",
    }
    env.pop("API_IMAGE", None)
    result = subprocess.run(
        ["bash", str(deploy / "up.sh"), *args],
        cwd=deploy,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result, log.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize(
    "args",
    [
        ["--pull"],
        ["--pull", "latest"],
        ["--pull", "1.2.3"],
        ["--pull", "v1.2"],
        ["--pull", "v1.2.3-rc1"],
        ["--pull", "v1.2.3-cpu"],
        ["--pulll", "v1.2.3"],
        ["v1.2.3"],
        ["--pull", "v1.2.3", "extra"],
    ],
)
def test_up_pull_dinh_dang_sai_in_usage_exit_1(tmp_path: Path, args: list[str]) -> None:
    result, calls = _run_up(tmp_path, args)

    assert result.returncode == 1
    assert "Cách dùng" in result.stderr
    assert calls == [], "không được gọi docker khi tham số sai"


def test_up_pull_thanh_cong_khong_build_va_ghim_image(tmp_path: Path) -> None:
    result, calls = _run_up(tmp_path, ["--pull", "v1.2.3"])

    assert result.returncode == 0, result.stderr
    expected = "ghcr.io/lndat18/production-legal-qa-rag:v1.2.3-cpu"
    pull_calls = [c for c in calls if " pull api" in c]
    up_calls = [c for c in calls if " up " in c]
    assert len(pull_calls) == 1 and pull_calls[0].startswith(f"API_IMAGE={expected} ::")
    assert len(up_calls) == 1
    assert "--no-build" in up_calls[0] and up_calls[0].startswith(f"API_IMAGE={expected} ::")
    assert not any(" build " in f" {c} " for c in calls), "chế độ pull không được build"
    # Pull phải đi trước up.
    assert calls.index(pull_calls[0]) < calls.index(up_calls[0])


def test_up_pull_that_bai_dung_lai_khong_roi_ve_build(tmp_path: Path) -> None:
    result, calls = _run_up(tmp_path, ["--pull", "v1.2.3"], pull_fail=True)

    assert result.returncode == 1
    assert "Không tự build" in result.stderr
    assert any(" pull api" in c for c in calls)
    assert not any(" build " in f" {c} " for c in calls)
    assert not any(" up " in c for c in calls)


def test_up_khong_co_co_van_build_local(tmp_path: Path) -> None:
    """Không cờ: giữ hành vi cũ (build api với TORCH_VARIANT rồi up -d), không pull."""
    result, calls = _run_up(tmp_path, [])

    assert result.returncode == 0, result.stderr
    build_calls = [c for c in calls if " build " in f" {c} "]
    assert len(build_calls) == 1
    assert "TORCH_VARIANT=cpu" in build_calls[0] and build_calls[0].endswith(" api")
    assert build_calls[0].startswith("API_IMAGE= ::"), "không ghim API_IMAGE khi build local"
    assert any(" up -d" in c and "--no-build" not in c for c in calls)
    assert not any(" pull " in f" {c} " for c in calls)
