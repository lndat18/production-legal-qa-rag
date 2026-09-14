"""Đếm token theo tokenizer thật của embedding model (mục 6).

PhoBERT được huấn luyện trên văn bản đã word-segment (`pyvi`, từ ghép nối
bằng `_`) — đếm token trên văn bản tiếng Việt thô sẽ sai lệch so với thực tế
model xử lý. Kết quả segment chỉ dùng nội bộ để đếm ở đây, KHÔNG lưu vào
`Chunk.content` — `content` giữ nguyên văn tiếng Việt gốc (chưa segment),
việc segment lại trước khi embed thật sự thuộc trách nhiệm của `embedding/`.
"""

from __future__ import annotations

from functools import lru_cache

from pyvi import ViTokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from production_legal_qa_rag.config import EmbeddingSettings


@lru_cache(maxsize=1)
def _load_tokenizer() -> PreTrainedTokenizerBase:
    """Nạp tokenizer 1 lần duy nhất (nạp từ HuggingFace Hub tốn thời gian)."""
    settings = EmbeddingSettings()
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    return tokenizer


@lru_cache(maxsize=4096)
def count_tokens(text: str) -> int:
    """Đếm số token của `text` theo tokenizer PhoBERT-based của embedding model.

    Quy trình (mục 6): word-segment bằng `pyvi.ViTokenizer.tokenize()`, sau
    đó đếm bằng `AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)` trên
    văn bản đã segment (bao gồm cả special token CLS/SEP, đúng số token thật
    sự model nhìn thấy).
    """
    segmented = ViTokenizer.tokenize(text)
    tokenizer = _load_tokenizer()
    return len(tokenizer.encode(segmented, add_special_tokens=True))
