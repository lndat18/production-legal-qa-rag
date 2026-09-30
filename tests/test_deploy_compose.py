"""Validation cho hạ tầng `deploy/` (deploy_spec.md mục 4, 6, 7, 9).

Không có Docker daemon thật trong CI (mục 9 của spec chỉ nghiệm thu thủ công), nên các
test ở đây chỉ dùng `docker compose config` — lệnh này parse/merge/resolve YAML tĩnh,
KHÔNG cần daemon đang chạy (không tạo network/container, không pull image) — an toàn để
chạy trên mọi runner có sẵn Docker CLI. Nếu CI không có `docker` binary, test tự skip.

Mọi test copy `deploy/docker-compose*.yml` + `.env.example` (root) sang thư mục tạm
(`tmp_path`, giữ đúng bố cục `deploy/` + `.env` ở cha) trước khi gọi `docker compose config`,
KHÔNG bao giờ tạo/đọc/xoá `.env` thật ở root — máy dev đã có `.env` chứa bí mật thật, tuyệt
đối không được đụng tới (deploy_spec.md mục 7: chỉ 1 cặp `.env`/`.env.example` ở root).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = REPO_ROOT / "deploy"

_DOCKER_AVAILABLE = shutil.which("docker") is not None


def _docker_compose_available() -> bool:
    """Kiểm tra plugin `docker compose` (v2) có gọi được không (không cần daemon)."""
    if not _DOCKER_AVAILABLE:
        return False
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


requires_docker_compose = pytest.mark.skipif(
    not _docker_compose_available(),
    reason="`docker compose` CLI không sẵn có trong môi trường này",
)


def _resolve_compose_config(
    tmp_path: Path,
    *,
    extra_compose_files: tuple[str, ...] = (),
    profile: str | None = None,
) -> dict[str, Any]:
    """Copy compose stack + `.env.example` (làm `.env` ở cha) vào `tmp_path` rồi resolve config.

    Bố cục tạm: `tmp_path/deploy/docker-compose*.yml` + `tmp_path/.env`, khớp `env_file:
    ../.env` trong compose. Dùng thư mục tạm biệt lập để `docker compose` không bao giờ chạm
    tới `.env` thật của máy dev (mục 7: bí mật không commit, không phải test fixture).

    Args:
        tmp_path: Thư mục tạm do pytest cấp, biệt lập cho mỗi test.
        extra_compose_files: Các file compose override bổ sung (vd. `docker-compose.gpu.yml`).
        profile: Ép `COMPOSE_PROFILES` qua flag `--profile` (bỏ qua thì dùng mặc định
            trong `.env.example`, tức "quick" — mục 3).

    Returns:
        Cấu hình compose đã resolve, dạng dict (parse từ `docker compose config --format json`).
    """
    deploy_tmp = tmp_path / "deploy"
    deploy_tmp.mkdir(exist_ok=True)
    shutil.copy(DEPLOY_DIR / "docker-compose.yml", deploy_tmp / "docker-compose.yml")
    for name in extra_compose_files:
        shutil.copy(DEPLOY_DIR / name, deploy_tmp / name)
    # `env_file: ../.env` để nạp cho container; `--env-file` riêng để giãn ${VAR} trong YAML
    # (2 cơ chế khác nhau, deploy_spec.md mục 7).
    shutil.copy(REPO_ROOT / ".env.example", tmp_path / ".env")

    cmd = ["docker", "compose", "--env-file", "../.env", "-f", "docker-compose.yml"]
    for name in extra_compose_files:
        cmd += ["-f", name]
    if profile is not None:
        cmd += ["--profile", profile]
    cmd += ["config", "--format", "json"]

    result = subprocess.run(
        cmd, cwd=deploy_tmp, capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, (
        f"`docker compose config` thất bại (spec mục 4):\n"
        f"cmd={cmd}\nstderr={result.stderr}"
    )
    return json.loads(result.stdout)


@requires_docker_compose
def test_docker_compose_config_parses_with_quick_profile_mac_dinh(
    tmp_path: Path,
) -> None:
    """Mặc định (không truyền --profile) phải kích hoạt đúng service `cloudflared-quick`."""
    config = _resolve_compose_config(tmp_path)

    assert set(config["services"]) == {
        "postgres",
        "redis",
        "api",
        "open-webui",
        "cloudflared-quick",
    }


@requires_docker_compose
def test_docker_compose_config_named_profile_dung_service_khac(tmp_path: Path) -> None:
    """Chuyển sang named tunnel chỉ đổi profile — `cloudflared-named` thay cho `-quick`."""
    config = _resolve_compose_config(tmp_path, profile="named")

    services = set(config["services"])
    assert "cloudflared-named" in services
    assert "cloudflared-quick" not in services


@requires_docker_compose
def test_docker_compose_gpu_override_them_gpu_reservation_cho_api(
    tmp_path: Path,
) -> None:
    """`docker-compose.gpu.yml` chỉ thêm GPU reservation cho `api`, không đổi service khác."""
    without_gpu = _resolve_compose_config(tmp_path)
    assert not without_gpu["services"]["api"].get("deploy")

    with_gpu = _resolve_compose_config(
        tmp_path, extra_compose_files=("docker-compose.gpu.yml",)
    )
    api_deploy = with_gpu["services"]["api"]["deploy"]
    devices = api_deploy["resources"]["reservations"]["devices"]
    assert devices == [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]

    for name, service in with_gpu["services"].items():
        if name != "api":
            assert not service.get("deploy"), f"{name} không nên có GPU reservation"


@requires_docker_compose
def test_docker_compose_khong_service_nao_publish_port(tmp_path: Path) -> None:
    """Không service nào publish cổng ra host (mục 4, mục 9.1) — kể cả 2 profile + GPU."""
    for profile in ("quick", "named"):
        config = _resolve_compose_config(
            tmp_path, extra_compose_files=("docker-compose.gpu.yml",), profile=profile
        )
        for name, service in config["services"].items():
            assert not service.get("ports"), f"service '{name}' không được publish port"


@requires_docker_compose
def test_docker_compose_image_tag_duoc_ghim_khong_dung_latest(tmp_path: Path) -> None:
    """Mọi image bên ngoài phải ghim tag cụ thể, không dùng `latest` (mục 4)."""
    config = _resolve_compose_config(tmp_path, profile="named")

    pinned_images = {
        name: service["image"]
        for name, service in config["services"].items()
        if service.get("image")
    }
    # `api` build từ Dockerfile, không có `image:` cố định — các service còn lại đều dùng
    # image ngoài và phải xuất hiện ở đây.
    expected = {"postgres", "redis", "open-webui", "cloudflared-named"}
    assert set(pinned_images) == expected
    for name, image in pinned_images.items():
        assert ":" in image, f"image của '{name}' thiếu tag: {image}"
        tag = image.rsplit(":", 1)[1]
        assert tag != "latest", f"image của '{name}' không được ghim bằng tag 'latest'"


@requires_docker_compose
def test_docker_compose_restart_policy_va_logging(tmp_path: Path) -> None:
    """Mọi service `restart: unless-stopped` + log driver json-file giới hạn dung lượng."""
    config = _resolve_compose_config(tmp_path, profile="named")

    for name, service in config["services"].items():
        assert service.get("restart") == "unless-stopped", name
        logging_cfg = service.get("logging") or {}
        assert logging_cfg.get("driver") == "json-file", name
        assert logging_cfg.get("options", {}).get("max-size") == "10m", name
        assert logging_cfg.get("options", {}).get("max-file") == "3", name


@requires_docker_compose
def test_docker_compose_healthcheck_va_depends_on_chain(tmp_path: Path) -> None:
    """Chuỗi phụ thuộc `redis -> api -> open-webui -> cloudflared` (mục 4).

    `api` không còn dùng Postgres (chatlog đã gỡ) nên không phụ thuộc `postgres`.
    """
    config = _resolve_compose_config(tmp_path)
    services = config["services"]

    for name in ("postgres", "redis", "api", "open-webui"):
        assert services[name].get("healthcheck"), f"'{name}' thiếu healthcheck"

    api_depends = services["api"]["depends_on"]
    assert "postgres" not in api_depends
    assert api_depends["redis"]["condition"] == "service_healthy"

    webui_depends = services["open-webui"]["depends_on"]
    assert webui_depends["api"]["condition"] == "service_healthy"

    cloudflared_depends = services["cloudflared-quick"]["depends_on"]
    assert cloudflared_depends["open-webui"]["condition"] == "service_healthy"


@requires_docker_compose
def test_docker_compose_moi_service_co_gioi_han_bo_nho(tmp_path: Path) -> None:
    """`mem_limit` phải khai báo cho các service nặng — tránh nuốt hết RAM WSL2 (mục 4)."""
    config = _resolve_compose_config(tmp_path)
    services = config["services"]

    for name in ("postgres", "redis", "api", "open-webui"):
        assert services[name].get("mem_limit"), f"'{name}' thiếu mem_limit"


def test_env_example_liet_ke_du_bien_theo_spec_muc_7() -> None:
    """`.env.example` phải liệt kê đủ toàn bộ biến ở bảng mục 7 (không thiếu tên nào)."""
    required_vars = {
        "GROQ_API_KEY_1",
        "GROQ_API_KEY_2",
        "HF_TOKEN",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "PINECONE_SPARSE_INDEX_NAME",
        "CHATBOT_API_KEY",
        "REDIS_PASSWORD",
        "REDIS_URL",
        "DEPLOY_POSTGRES_USER",
        "DEPLOY_POSTGRES_PASSWORD",
        "WEBUI_SECRET_KEY",
        "WEBUI_URL",
        "TUNNEL_TOKEN",
    }
    env_vars = _parse_env_file(REPO_ROOT / ".env.example")
    missing = required_vars - set(env_vars)
    assert not missing, f"`.env.example` thiếu biến bắt buộc theo mục 7: {missing}"


def test_env_example_khong_de_lo_gia_tri_bi_mat_mau() -> None:
    """Các biến bí mật phải để trống trong `.env.example` — không commit giá trị mẫu."""
    secret_vars = {
        "GROQ_API_KEY_1",
        "GROQ_API_KEY_2",
        "HF_TOKEN",
        "PINECONE_API_KEY",
        "CHATBOT_API_KEY",
        "REDIS_PASSWORD",
        "DEPLOY_POSTGRES_PASSWORD",
        "WEBUI_SECRET_KEY",
        "TUNNEL_TOKEN",
    }
    env_vars = _parse_env_file(REPO_ROOT / ".env.example")
    for name in secret_vars:
        assert env_vars.get(name, "") == "", (
            f"'{name}' trong .env.example phải để trống, tìm thấy: {env_vars.get(name)!r}"
        )


def test_env_example_co_du_groq_api_key_1_den_9_va_bo_ten_cu() -> None:
    """`.env.example` có `GROQ_API_KEY_1`...`_9` (để trống) và không còn `GROQ_API_KEY` trần."""
    env_vars = _parse_env_file(REPO_ROOT / ".env.example")

    for number in range(1, 10):
        name = f"GROQ_API_KEY_{number}"
        assert name in env_vars, f"`.env.example` thiếu {name}"
        assert env_vars[name] == "", (
            f"{name} phải để trống, tìm thấy: {env_vars[name]!r}"
        )
    assert "GROQ_API_KEY" not in env_vars, "tên cũ GROQ_API_KEY không còn được hỗ trợ"


def test_env_example_redis_url_tro_localhost_cho_dev_tren_host() -> None:
    """`REDIS_URL` trong `.env.example` là giá trị dev trên host.

    Container `api` production KHÔNG dùng giá trị này: `docker-compose.yml` ghi đè bằng URL
    dựng từ `REDIS_PASSWORD` (`environment:` thắng `env_file:`, mục 7).
    """
    env_vars = _parse_env_file(REPO_ROOT / ".env.example")
    assert env_vars["REDIS_URL"] == "redis://localhost:6379/0"
    assert "CHATLOG_DATABASE_URL" not in env_vars
    assert "CHATLOG_RETENTION_DAYS" not in env_vars


@requires_docker_compose
def test_docker_compose_ghi_de_url_redis_cho_api_production(
    tmp_path: Path,
) -> None:
    """`environment:` của `api` phải ghi đè `REDIS_URL` từ `.env`; không còn URL Postgres."""
    config = _resolve_compose_config(tmp_path)
    env = config["services"]["api"]["environment"]
    assert env["REDIS_URL"].endswith("@redis:6379/0")
    assert "CHATLOG_DATABASE_URL" not in env


@requires_docker_compose
def test_docker_compose_observe_override_noi_api_vao_network_observe(
    tmp_path: Path,
) -> None:
    """`docker-compose.observe.yml` nối `api` vào `legal-qa-observe` + trỏ Langfuse nội bộ."""
    config = _resolve_compose_config(
        tmp_path, extra_compose_files=("docker-compose.observe.yml",)
    )
    api = config["services"]["api"]
    assert set(api["networks"]) == {"internal", "observe"}
    assert api["environment"]["LANGFUSE_BASE_URL"] == "http://langfuse-web:3000"
    assert api["environment"]["LANGFUSE_TRACING_ENVIRONMENT"] == "production"
    assert config["networks"]["observe"]["name"] == "legal-qa-observe"
    assert config["networks"]["observe"]["external"] is True

    for name, service in config["services"].items():
        if name != "api":
            assert "observe" not in (service.get("networks") or {}), name


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse file dạng `KEY=value` (bỏ comment/dòng trống), không giãn biến."""
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip()
    return result


def _lines_starting_with(content: str, prefix: str) -> list[str]:
    """Trả về các dòng (đã strip) bắt đầu bằng `prefix` trong `content`."""
    return [
        line.strip() for line in content.splitlines() if line.strip().startswith(prefix)
    ]


def test_dockerfile_chay_bang_user_khong_phai_root() -> None:
    """Image phải chạy bằng user không phải root (mục 6, nghiệm thu mục 9.7)."""
    content = (DEPLOY_DIR / "Dockerfile").read_text(encoding="utf-8")
    user_lines = _lines_starting_with(content, "USER ")

    assert user_lines, "Dockerfile thiếu chỉ thị USER"
    assert user_lines[-1] == "USER appuser", (
        f"Chỉ thị USER cuối cùng phải là 'USER appuser', tìm thấy: {user_lines[-1]!r}"
    )
    assert "USER root" not in content


def test_dockerfile_mac_dinh_torch_cpu() -> None:
    """Build arg `TORCH_VARIANT` mặc định `cpu` — image build được không cần GPU (mục 6)."""
    content = (DEPLOY_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG TORCH_VARIANT=cpu" in content


def test_dockerfile_cmd_chay_uvicorn_factory_khong_con_migration() -> None:
    """CMD chạy `uvicorn --factory`; không còn `alembic upgrade head` (chatlog đã gỡ, mục 6)."""
    content = (DEPLOY_DIR / "Dockerfile").read_text(encoding="utf-8")
    cmd_lines = _lines_starting_with(content, "CMD")
    assert cmd_lines, "Dockerfile thiếu chỉ thị CMD"
    cmd_line = cmd_lines[-1]

    assert "alembic" not in content
    assert "uvicorn" in cmd_line
    assert "--factory" in cmd_line
    assert "production_legal_qa_rag.api.app:create_app" in cmd_line


def test_dockerfile_khong_bake_data_bm25_vao_image() -> None:
    """`data/bm25/` phải mount lúc chạy, không COPY vào image (mục 6)."""
    content = (DEPLOY_DIR / "Dockerfile").read_text(encoding="utf-8")
    copy_lines = _lines_starting_with(content, "COPY")
    assert not any("data/bm25" in line or "data/" in line for line in copy_lines)


def test_dockerignore_loai_tru_bi_mat_va_du_lieu_khong_can() -> None:
    """`.dockerignore` phải loại `.env`, `deploy/.env`, `data/`, `tests/`, `.venv/`, `.git/`."""
    content = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    entries = {
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.startswith("#")
    }

    for required in (".env", "deploy/.env", "data/", "tests/", ".venv/", ".git/"):
        assert required in entries, f".dockerignore thiếu mục '{required}'"


def test_gitignore_loai_tru_thu_muc_backup() -> None:
    """`deploy/backups/` (chứa dump chứa dữ liệu người dùng) không được commit (mục 8)."""
    content = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "deploy/backups/" in content.splitlines()


def test_backup_script_cu_phap_shell_hop_le() -> None:
    """`deploy/backup.sh` phải là bash hợp lệ (kiểm tra cú pháp, không chạy thật)."""
    result = subprocess.run(
        ["bash", "-n", str(DEPLOY_DIR / "backup.sh")],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "script",
    [
        "deploy/down.sh",
        "deploy/reset_cache.sh",
        "observability/up.sh",
        "observability/down.sh",
    ],
)
def test_script_van_hanh_cu_phap_bash_hop_le_va_chay_duoc(script: str) -> None:
    """Các script vận hành phải là bash hợp lệ và có quyền thực thi (không chạy thật)."""
    path = REPO_ROOT / script
    assert path.stat().st_mode & 0o111, f"{script} thiếu quyền thực thi"
    result = subprocess.run(
        ["bash", "-n", str(path)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_script_observe_khong_xoa_volume() -> None:
    """`down.sh` của cả deploy lẫn observe phải giữ volume (không có `down ... -v`)."""
    for script in ("deploy/down.sh", "observability/down.sh"):
        lines = (REPO_ROOT / script).read_text(encoding="utf-8").splitlines()
        commands = [
            ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")
        ]
        assert not any(" down" in ln and " -v" in ln for ln in commands), script


def test_backup_script_giu_dung_7_ban_va_chi_dump_database_openwebui() -> None:
    """Backup chỉ dump `openwebui` (DB `chatbot` đã gỡ), giữ 7 bản gần nhất (mục 8)."""
    content = (DEPLOY_DIR / "backup.sh").read_text(encoding="utf-8")
    assert "KEEP=7" in content
    assert "openwebui" in content
    assert "chatbot" not in content


@requires_docker_compose
def test_docker_compose_postgres_tu_tao_database_openwebui(tmp_path: Path) -> None:
    """Postgres tự tạo DB `openwebui` qua `POSTGRES_DB`, không cần script initdb (mục 4)."""
    postgres = _resolve_compose_config(tmp_path)["services"]["postgres"]
    assert postgres["environment"]["POSTGRES_DB"] == "openwebui"
    mounts = [str(volume.get("target", "")) for volume in postgres.get("volumes", [])]
    assert "/docker-entrypoint-initdb.d" not in mounts
    assert not (DEPLOY_DIR / "initdb").exists()
