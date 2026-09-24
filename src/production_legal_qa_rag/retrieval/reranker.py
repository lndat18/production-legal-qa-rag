"""Reranker chạy in-process: load model 1 lần, batch inference CUDA/CPU (mục 6.1).

Thay thế HTTP client (`reranker_client.py`) bằng inference tại chỗ dùng
`AITeamVN/Vietnamese_Reranker` (fine-tune từ `bge-reranker-v2-m3`). Không
host API riêng — model chạy trong `RetrievalPipeline`, không qua HTTP/microservice.
"""

from __future__ import annotations

import asyncio
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from production_legal_qa_rag.config import RerankerSettings

logger = logging.getLogger(__name__)

MAX_LENGTH = 512
"""Giới hạn tổng token: query + passage + 4 special token.

512 là ước lượng có margin (không phải số đo), tính từ:
- content ≤ 236 PhoBERT-tokens → ×1.5 ≈ 400 SentencePiece-tokens
- breadcrumb ~40 token, query ~64 token, 4 special tokens → tổng ~508
Nếu log cảnh báo truncation hoặc chất lượng rerank giảm bất thường,
đo lại thực tế bằng tokenizer bge-reranker-v2-m3 trên corpus rồi cập nhật.
"""

RERANK_BATCH_SIZE = 16
"""Batch cố định để chặn đỉnh VRAM bất kể số candidate thực tế."""

_MODEL_NAME = "AITeamVN/Vietnamese_Reranker"
_INFERENCE_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="local-reranker-inference",
)
"""Một hàng đợi inference dùng chung process, an toàn giữa các event loop.

Mỗi job là trọn một lần ``rerank()`` và mọi batch của nó. Vì executor chỉ có
một worker, các job không thể forward cùng lúc; job chờ nằm trong queue của
executor, không chiếm thread hay chặn event loop đang await kết quả.
"""


class LocalReranker:
    """In-process CrossEncoder reranker dùng `AITeamVN/Vietnamese_Reranker`.

    Model được load **một lần** khi khởi tạo và giữ suốt vòng đời process.
    Forward pass là blocking call (CPU/GPU-bound) nên `rerank()` dùng
    executor một worker dùng chung process để không chặn event loop. Executor
    này giới hạn toàn bộ request rerank xuống một inference tại một thời điểm,
    kể cả khi các request đến từ event loop khác nhau.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_length: int | None = None,
        batch_size: int | None = None,
        *,
        settings: RerankerSettings | None = None,
    ) -> None:
        """Load model/tokenizer hoặc lưu lỗi để `rerank()` fallback an toàn.

        Args:
            model_name: Ghi đè tên model, chủ yếu cho test/local experiment.
            max_length: Ghi đè tổng giới hạn token của cặp query-passage.
            batch_size: Ghi đè số passage mỗi forward pass.
            settings: Cấu hình tập trung; mặc định đọc môi trường.
        """
        configured = settings or RerankerSettings()
        self._model_name = configured.model_name if model_name is None else model_name
        self._max_length = configured.max_length if max_length is None else max_length
        self._batch_size = configured.batch_size if batch_size is None else batch_size
        if not self._model_name or self._max_length <= 0 or self._batch_size <= 0:
            raise ValueError(
                "model_name phải không rỗng; max_length và batch_size phải dương."
            )
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_str)
        # Log đúng một lần lúc load model (mục 6.1).
        logger.info(
            "LocalReranker: load model '%s' trên device '%s'.",
            self._model_name,
            device_str,
        )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            if device_str == "cuda":
                # fp16 chỉ có lợi tốc độ trên GPU (mục 6.1).
                model = model.half()
            # fp32 trên CPU — không dùng half() vì không có lợi tốc độ trên CPU.
            self._model = model.to(self._device).eval()
        except Exception:
            # Không được làm hỏng retrieval khi cache model hỏng, không tải được
            # model hoặc driver CUDA lỗi: rerank() sẽ trả fallback đúng contract.
            logger.warning(
                "LocalReranker không load được model '%s', sẽ fallback khi retrieve.",
                self._model_name,
                exc_info=True,
            )

    async def rerank(self, query: str, passages: list[str]) -> list[float] | None:
        """Chấm điểm từng passage theo câu hỏi gốc (async, không bao giờ raise).

        Forward pass blocking chạy trong executor riêng để không chặn event
        loop. Nếu runtime error (CUDA OOM, ...) fallback: trả `None`, log đủ
        context để debug (batch size, số passage lúc lỗi).

        Args:
            query: Câu hỏi gốc, không phải hypothetical document.
            passages: Mỗi phần tử là `breadcrumb + "\\n" + content` của một chunk,
                cùng thứ tự với union trước rerank.

        Returns:
            List[float] cùng thứ tự `passages`, hoặc `None` khi fallback.
        """
        if not passages:
            return []
        if self._model is None or self._tokenizer is None:
            logger.warning(
                "LocalReranker chưa sẵn sàng, fallback (batch_size=%d, n_passages=%d).",
                self._batch_size,
                len(passages),
            )
            return None
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                _INFERENCE_EXECUTOR,
                self._sync_rerank,
                query,
                passages,
            )
        except Exception:
            logger.warning(
                "LocalReranker lỗi runtime, fallback (batch_size=%d, n_passages=%d).",
                self._batch_size,
                len(passages),
                exc_info=True,
            )
            return None

    def _sync_rerank(self, query: str, passages: list[str]) -> list[float]:
        """Blocking forward pass; được gọi qua executor một worker dùng chung."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("LocalReranker chưa load model/tokenizer.")
        all_scores: list[float] = []

        for batch_start in range(0, len(passages), self._batch_size):
            batch = passages[batch_start : batch_start + self._batch_size]
            pairs = [[query, passage] for passage in batch]

            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**inputs).logits

            # bge-reranker-v2-m3 ra shape (batch, 1) hoặc scalar per sample.
            if logits.dim() > 1 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            elif logits.dim() > 1 and logits.shape[-1] >= 2:
                # Nếu mô hình ra xác suất 2-class, lấy logit class-1.
                logits = logits[:, 1]

            scores_raw = logits.cpu().to(torch.float32).tolist()
            if not isinstance(scores_raw, list):
                scores_raw = [scores_raw]
            scores = [float(score) for score in scores_raw]
            if len(scores) != len(batch):
                raise ValueError(
                    "LocalReranker: số score không khớp passage "
                    f"({len(scores)} != {len(batch)})."
                )
            for score in scores:
                if not math.isfinite(score):
                    raise ValueError(f"LocalReranker: score không hữu hạn ({score!r}).")

            all_scores.extend(scores)

        if len(all_scores) != len(passages):
            raise ValueError("LocalReranker: tổng số score không khớp passage.")
        return all_scores
