"""Validation tĩnh cho `.github/workflows/release.yml` (deploy_spec.md mục 11.2-11.4).

Workflow chỉ chạy được khi push tag thật nên ở đây chỉ parse YAML và kiểm tra cấu trúc;
không chạy gì và không cần mạng/docker.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"


@pytest.fixture(scope="module")
def workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workflow(workflow_text: str) -> dict[str, Any]:
    data = yaml.safe_load(workflow_text)
    assert isinstance(data, dict)
    return data


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML (YAML 1.1) parse khoá `on` thành True.
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    return triggers


def _build_step(workflow: dict[str, Any]) -> dict[str, Any]:
    for step in workflow["jobs"]["build"]["steps"]:
        if str(step.get("uses", "")).startswith("docker/build-push-action@"):
            return step
    raise AssertionError("build job thiếu docker/build-push-action")


def test_trigger_chi_la_push_tag_semver(workflow: dict[str, Any]) -> None:
    """Chỉ trigger push tag `v[0-9]+.[0-9]+.[0-9]+`; không branches, không pull_request."""
    triggers = _triggers(workflow)
    assert set(triggers) == {"push"}
    push = triggers["push"]
    assert push["tags"] == ["v[0-9]+.[0-9]+.[0-9]+"]
    assert "branches" not in push


def test_gate_co_hai_kiem_tra_va_build_phu_thuoc_gate(
    workflow: dict[str, Any], workflow_text: str
) -> None:
    """Gate kiểm tra tổ tiên của main + check run `checks`; build `needs: gate`."""
    jobs = workflow["jobs"]
    assert set(jobs) == {"gate", "build"}
    assert jobs["build"]["needs"] == "gate"

    gate_runs = "\n".join(s.get("run", "") for s in jobs["gate"]["steps"])
    assert "git merge-base --is-ancestor" in gate_runs
    assert "origin/main" in gate_runs
    assert "check-runs" in gate_runs
    assert '.name == "checks"' in gate_runs
    assert '"success"' in gate_runs
    # Mỗi kiểm tra fail thì job phải thoát lỗi.
    assert gate_runs.count("exit 1") >= 2
    # merge-base cần lịch sử đầy đủ.
    checkout = next(
        s for s in jobs["gate"]["steps"] if "checkout" in str(s.get("uses"))
    )
    assert checkout["with"]["fetch-depth"] == 0


def test_permissions_toi_thieu(workflow: dict[str, Any]) -> None:
    """Mặc định `contents: read`; gate thêm `checks: read`; build thêm `packages: write`."""
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    assert jobs["gate"]["permissions"] == {"contents": "read", "checks": "read"}
    assert jobs["build"]["permissions"] == {"contents": "read", "packages": "write"}


def test_matrix_cpu_cu126_fail_fast_false(workflow: dict[str, Any]) -> None:
    strategy = workflow["jobs"]["build"]["strategy"]
    assert strategy["fail-fast"] is False
    assert strategy["matrix"]["variant"] == ["cpu", "cu126"]


def test_build_push_cau_hinh(workflow: dict[str, Any]) -> None:
    """Context/file/platform/provenance/build-arg/cache/label đúng mục 11.3-11.4."""
    step = _build_step(workflow)
    cfg = step["with"]
    assert cfg["context"] == "."
    assert cfg["file"] == "deploy/Dockerfile"
    assert cfg["platforms"] == "linux/amd64"
    assert cfg["push"] is True
    assert cfg["provenance"] is False
    assert cfg["build-args"] == "TORCH_VARIANT=${{ matrix.variant }}"
    # Cache gha scope riêng theo biến thể, cả đọc lẫn ghi.
    assert cfg["cache-from"] == "type=gha,scope=${{ matrix.variant }}"
    assert cfg["cache-to"].startswith("type=gha,scope=${{ matrix.variant }}")
    assert "mode=min" in cfg["cache-to"]
    assert "org.opencontainers.image.source=" in cfg["labels"]
    assert "github.repository" in cfg["labels"]


def test_tag_cpu_cu126_latest(workflow: dict[str, Any], workflow_text: str) -> None:
    """`vX.Y.Z-<variant>` cho cả hai; `latest` chỉ ghép cho cpu; ảnh viết thường."""
    assert workflow["env"]["IMAGE"] == "ghcr.io/lndat18/production-legal-qa-rag"
    assert workflow["env"]["IMAGE"] == workflow["env"]["IMAGE"].lower()

    tag_step = next(
        s for s in workflow["jobs"]["build"]["steps"] if s.get("id") == "tags"
    )
    script = tag_step["run"]
    assert "${IMAGE}:${GITHUB_REF_NAME}-${{ matrix.variant }}" in script
    # `latest` chỉ xuất hiện trong nhánh cpu, đúng 1 lần, và không có latest-cu126.
    assert script.count(":latest") == 1
    assert 'if [ "${{ matrix.variant }}" = "cpu" ]' in script
    assert script.index(":latest") > script.index('= "cpu"')
    assert "latest-cu126" not in workflow_text
    assert _build_step(workflow)["with"]["tags"] == "${{ steps.tags.outputs.list }}"


def test_chi_dung_github_token_va_khong_secret_khac(workflow_text: str) -> None:
    """Không PAT/secret mới: mọi `secrets.X` đều là GITHUB_TOKEN (mục 11.4)."""
    names = set(re.findall(r"secrets\.([A-Za-z0-9_]+)", workflow_text))
    assert names == {"GITHUB_TOKEN"}


def test_login_ghcr_bang_github_token(workflow: dict[str, Any]) -> None:
    login = next(
        s
        for s in workflow["jobs"]["build"]["steps"]
        if str(s.get("uses", "")).startswith("docker/login-action@")
    )
    assert login["with"]["registry"] == "ghcr.io"
    assert login["with"]["password"] == "${{ secrets.GITHUB_TOKEN }}"


def test_don_disk_chi_cho_cu126(workflow: dict[str, Any]) -> None:
    step = next(
        s for s in workflow["jobs"]["build"]["steps"] if "Dọn disk" in s.get("name", "")
    )
    assert step["if"] == "matrix.variant == 'cu126'"
    assert "docker image prune -af" in step["run"]


def test_concurrency_khong_cancel_build_dang_chay(workflow: dict[str, Any]) -> None:
    concurrency = workflow["concurrency"]
    assert concurrency["group"] == "release-${{ github.ref }}"
    assert concurrency["cancel-in-progress"] is False
