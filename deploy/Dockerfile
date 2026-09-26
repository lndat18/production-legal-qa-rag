# Image cho service `api` (deploy_spec.md mục 6). Build context là root repo:
#   docker compose -f deploy/docker-compose.yml build api
#   # bản GPU (retrieval_spec.md mục 6.1, deploy_spec.md mục 4.1):
#   docker compose -f deploy/docker-compose.yml build --build-arg TORCH_VARIANT=cu121 api
FROM python:3.14-slim

# Cài `uv` bằng cách copy binary tĩnh từ image chính chủ Astral (không pip install).
COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /uvx /usr/local/bin/

# Biến thể torch: "cpu" (mặc định, chạy được trên mọi máy) hoặc "cu121" (cần NVIDIA
# Container Toolkit + docker-compose.gpu.yml lúc chạy — xem mục 4.1). Không pin phiên bản
# torch cụ thể ở đây: index CPU/CUDA của PyTorch không cam kết cùng tập phiên bản như PyPI,
# pin cứng dễ làm build gãy khi torch trên PyPI (dependency khác trong uv.lock) bump trước.
ARG TORCH_VARIANT=cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    HF_HOME=/home/appuser/.cache/huggingface

# User không phải root, tạo NGAY TỪ ĐẦU (trước khi cài dependency), rồi build mọi thứ trực
# tiếp bằng user này: nếu chown lại cả cây venv/site-packages (hàng chục nghìn file) sau khi
# đã cài xong, overlay2 phải copy-up từng file sang layer mới — cực chậm trên tầng nặng nhất
# của image này. `mkdir -p /app` trước để thư mục làm việc thuộc sở hữu appuser ngay từ đầu.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser \
    && mkdir -p /app "${HF_HOME}" \
    && chown -R appuser:appuser /app /home/appuser

WORKDIR /app
USER appuser

# torch mặc định trên PyPI (Linux) kéo theo toàn bộ runtime CUDA dù không dùng (~2GB:
# nvidia-cublas, nvidia-cudnn, triton, cuda-toolkit...) bất kể TORCH_VARIANT — loại hết khỏi
# tầng dependency bên dưới rồi cài `torch` đúng biến thể (CPU/CUDA) riêng theo index của
# PyTorch (mục 4.1, 6). Cập nhật danh sách nếu `uv.lock` đổi tên gói CUDA runtime của torch
# (vd. hậu tố cu13 -> cu14 khi torch bump major version).
ENV TORCH_CUDA_EXTRAS="torch triton cuda-bindings cuda-pathfinder cuda-toolkit nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx"

# --- Tầng dependency (cache Docker layer khi chỉ code đổi mà pyproject/uv.lock không đổi) ---
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \
    $(printf -- '--no-install-package %s ' $TORCH_CUDA_EXTRAS)

RUN uv pip install --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" torch

# --- Cài chính project ---
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser alembic/ ./alembic/
COPY --chown=appuser:appuser alembic.ini README.md ./
# `uv sync` (kể cả với --no-install-package) đồng bộ hoá venv đúng nghĩa: gỡ bất cứ gói nào
# nó coi là "không mong muốn", nên sẽ GỠ LUÔN bản torch vừa cài đúng biến thể ở trên nếu
# dùng lại `uv sync` ở đây. Dùng `uv pip install` (chỉ cài thêm, không đối chiếu/gỡ toàn bộ
# venv) để cài package `production_legal_qa_rag` mà không đụng tới các gói đã có.
RUN uv pip install --no-deps .

# Không copy `data/bm25/` (sinh ra, .gitignore) — mount read-only từ host lúc chạy (mục 6);
# thiếu file thì `retrieval/` lỗi rõ ràng lúc khởi động thay vì chạy nửa vời.

EXPOSE 8000

# Chạy migration trước rồi mới phục vụ request — an toàn vì chỉ 1 worker/1 container
# (semaphore admission in-process, conversation_spec.md mục 8). `exec` để uvicorn nhận
# đúng SIGTERM từ `docker stop` thay vì chạy dưới shell con.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn production_legal_qa_rag.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 1"]
