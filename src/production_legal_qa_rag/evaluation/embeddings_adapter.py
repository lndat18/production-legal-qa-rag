"""Adapter `Embeddings` (LangChain) quanh `InferenceClient` cho `TestsetGenerator`.

`HuggingFaceEmbedder` (`embedding/hf_client.py`) chỉ có `embed_chunks(chunks:
list[Chunk])`, không implement interface `langchain_core.embeddings.Embeddings`
mà `LangchainEmbeddingsWrapper` của ragas cần (evaluation_spec.md mục 3). Adapter
này gọi thẳng `InferenceClient.feature_extraction`; Phase 1 giữ text thô để so
tương đồng trong `KnowledgeGraph` (evaluation_spec.md mục 3). Phase 2 bật
`segment=True` để `answer_relevancy` dùng cùng tiền xử lý pyvi với model index
(mục 11.5).
"""

from __future__ import annotations

from collections.abc import Sequence

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from pyvi import ViTokenizer

from production_legal_qa_rag.config import EmbeddingSettings

# Cùng kích thước batch với embedding/hf_client.py để nhất quán số lượt gọi HF
# mỗi request, không phải một quyết định kiến trúc mới.
_BATCH_SIZE = 25
_TIMEOUT_SECONDS = 30.0


class RagasEmbeddingsAdapter(Embeddings):
    """Bọc `InferenceClient` để dùng làm `generator_embeddings` của ragas.

    Tái dùng thẳng `EmbeddingSettings` đã có (`config.py`) — không thêm setting
    riêng cho embeddings ở Phase 1 (evaluation_spec.md mục 6).
    """

    def __init__(
        self,
        settings: EmbeddingSettings | None = None,
        client: InferenceClient | None = None,
        *,
        segment: bool = False,
    ) -> None:
        self._segment = segment
        self._settings = settings or EmbeddingSettings()
        self._client = client or InferenceClient(
            model=self._settings.model_name,
            token=self._settings.hf_token,
            timeout=_TIMEOUT_SECONDS,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed nhiều đoạn văn bản theo batch, giữ nguyên thứ tự đầu vào."""
        texts = (
            [ViTokenizer.tokenize(text) for text in texts] if self._segment else texts
        )
        embeddings: list[list[float]] = []
        for batch in _batched(texts, _BATCH_SIZE):
            response = self._client.feature_extraction(batch)
            embeddings.extend(_coerce_embeddings(response, expected_count=len(batch)))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed một câu truy vấn/tóm tắt đơn lẻ."""
        text = ViTokenizer.tokenize(text) if self._segment else text
        response = self._client.feature_extraction([text])
        return _coerce_embeddings(response, expected_count=1)[0]


def _batched(items: Sequence[str], size: int) -> list[list[str]]:
    """Chia `items` thành các batch liên tiếp có kích thước tối đa `size`."""
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def _coerce_embeddings(response: object, *, expected_count: int) -> list[list[float]]:
    """Validate response HF thành list vector float theo đúng thứ tự input.

    Raise lỗi rõ khi response sai định dạng, không âm thầm trả vector rỗng
    (evaluation_spec.md mục 8, cùng nguyên tắc validate ở biên như
    `embedding/hf_client.py`).
    """
    raw_embeddings = response.tolist() if hasattr(response, "tolist") else response
    if not isinstance(raw_embeddings, list) or len(raw_embeddings) != expected_count:
        raise ValueError(
            f"HF response không có đúng số vector mong đợi ({expected_count} vector)."
        )

    embeddings: list[list[float]] = []
    for embedding in raw_embeddings:
        if not isinstance(embedding, list):
            raise TypeError("HF response chứa vector không phải list.")
        embeddings.append([float(value) for value in embedding])
    return embeddings
