# Image cho service `api` (deploy_spec.md mục 6). Build context là root repo:
#   docker compose -f deploy/docker-compose.yml build [--build-arg TORCH_VARIANT=cu126] api
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /uvx /usr/local/bin/

# "cpu" (mặc định) hoặc "cu126" trở lên (cần docker-compose.gpu.yml lúc chạy, mục 4.1).
# PHẢI >= cu126: cu121/cu124 không có wheel Python 3.14. Không pin phiên bản torch — index
# PyTorch không cùng tập phiên bản với PyPI, pin cứng dễ gãy khi uv.lock bump.
ARG TORCH_VARIANT=cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    HF_HOME=/home/appuser/.cache/huggingface

# Tạo user không phải root ngay từ đầu và build bằng user này: chown lại venv sau khi cài
# xong sẽ buộc overlay2 copy-up hàng chục nghìn file, rất chậm.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser \
    && mkdir -p /app "${HF_HOME}" \
    && chown -R appuser:appuser /app /home/appuser

WORKDIR /app
USER appuser

# torch trên PyPI kéo theo ~2GB runtime CUDA dù không dùng: loại khỏi tầng dependency rồi
# cài torch đúng biến thể từ index PyTorch. Cập nhật danh sách nếu uv.lock đổi tên gói CUDA
# (vd. hậu tố cu13 -> cu14 khi torch bump major).
ENV TORCH_CUDA_EXTRAS="torch triton cuda-bindings cuda-pathfinder cuda-toolkit nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx"

# Tầng dependency (được cache khi chỉ code đổi)
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \
    $(printf -- '--no-install-package %s ' $TORCH_CUDA_EXTRAS)

RUN uv pip install --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" torch

# Cài chính project
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser alembic/ ./alembic/
COPY --chown=appuser:appuser alembic.ini README.md ./
# Dùng `uv pip install`, KHÔNG dùng lại `uv sync`: sync sẽ gỡ bản torch vừa cài đúng biến thể.
RUN uv pip install --no-deps .

# `data/bm25/` không copy vào image — mount read-only lúc chạy (mục 6).

EXPOSE 8000

# Chạy migration rồi mới phục vụ; an toàn vì chỉ 1 worker/1 container (admission in-process).
# `exec` để uvicorn nhận SIGTERM từ `docker stop`.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn production_legal_qa_rag.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 1"]
