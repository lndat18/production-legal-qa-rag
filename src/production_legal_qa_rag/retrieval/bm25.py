"""BM25 encoder tự viết cho sparse search (mục 6.2).

Không dùng `pinecone-text` (không cài được trên Python >= 3.14, mặc định
stopword/stem tiếng Anh làm mất từ tiếng Việt). Cùng một hàm `tokenize` dùng
cho fit, encode document và encode query. Dot product giữa vector query
(IDF) và vector document (phần tf đã chuẩn hoá độ dài) chính là điểm BM25.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel
from pyvi import ViTokenizer

from production_legal_qa_rag.retrieval.models import RetrievalError, SparseVector

BM25_K1 = 1.2
BM25_B = 0.75
# 1 = bản cũ (không có field); 2 = có token cấu trúc điều_N/khoản_M (mục 6.2).
BM25_PARAMS_VERSION = 2
REBUILD_COMMAND = "uv run python tools/sparse_index_documents.py"


class BM25ParamsVersionError(RetrievalError):
    """`bm25_params.json` thiếu hoặc khác `params_version` kỳ vọng."""


class BM25Params(BaseModel):
    """Tham số BM25 đã fit — nội dung của `bm25_params.json`."""

    vocab: dict[str, int]
    idf: dict[str, float]
    num_documents: int
    avgdl: float
    k1: float = BM25_K1
    b: float = BM25_B
    params_version: int = 1  # file cũ không có field = bản 1


def tokenize(text: str) -> list[str]:
    """Word-segment bằng pyvi, lowercase, bỏ token chỉ gồm dấu câu.

    Giữ dấu `_` của pyvi (`lao_động`); không stemming, không stopword để
    các từ như "do" không bị loại.

    Args:
        text: Văn bản thô.

    Returns:
        Danh sách term theo thứ tự xuất hiện (có lặp).
    """
    segmented = ViTokenizer.tokenize(text).lower()
    return [token for token in segmented.split() if any(c.isalnum() for c in token)]


class BM25Encoder:
    """Encoder BM25 thuần Python; `fit` một lần trên toàn corpus."""

    def __init__(self, params: BM25Params | None = None) -> None:
        self._params = params

    @property
    def params(self) -> BM25Params:
        """Tham số đã fit/load.

        Raises:
            RuntimeError: Khi encoder chưa `fit` hoặc `load`.
        """
        if self._params is None:
            raise RuntimeError("BM25Encoder chưa fit hoặc load tham số.")
        return self._params

    def fit(
        self,
        texts: Sequence[str],
        *,
        extra_terms: Sequence[Sequence[str]] | None = None,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        """Tính vocabulary, IDF và độ dài trung bình từ toàn bộ corpus.

        Args:
            texts: Mọi văn bản của corpus (`breadcrumb + " " + content`).
            extra_terms: Token cấu trúc của từng văn bản (cùng thứ tự `texts`),
                nối vào danh sách token sau khi tokenize; `None` là không có.
            k1: Hệ số bão hoà tf.
            b: Hệ số chuẩn hoá độ dài.

        Raises:
            ValueError: Khi corpus rỗng hoặc `extra_terms` lệch độ dài `texts`.
        """
        if not texts:
            raise ValueError("Không thể fit BM25 trên corpus rỗng.")
        if extra_terms is not None and len(extra_terms) != len(texts):
            raise ValueError("extra_terms phải cùng độ dài với texts.")

        document_frequency: Counter[str] = Counter()
        total_length = 0
        for position, text in enumerate(texts):
            tokens = tokenize(text) + list(extra_terms[position] if extra_terms else ())
            total_length += len(tokens)
            document_frequency.update(set(tokens))

        num_documents = len(texts)
        terms = sorted(document_frequency)  # thứ tự cố định để tái lập id
        self._params = BM25Params(
            vocab={term: index for index, term in enumerate(terms)},
            idf={
                term: math.log(
                    (num_documents - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                    + 1
                )
                for term in terms
            },
            num_documents=num_documents,
            avgdl=total_length / num_documents,
            k1=k1,
            b=b,
            params_version=BM25_PARAMS_VERSION,
        )

    def encode_document(
        self, text: str, extra_terms: Sequence[str] = ()
    ) -> SparseVector:
        """Sparse vector phía document: `tf·(k1+1) / (tf + k1·(1-b+b·dl/avgdl))`.

        Term ngoài vocabulary bị bỏ. `extra_terms` là token cấu trúc của chunk,
        nối vào danh sách token (cùng cách với `fit`).
        """
        params = self.params
        tokens = tokenize(text) + list(extra_terms)
        length_norm = 1 - params.b + params.b * len(tokens) / params.avgdl
        weights: dict[int, float] = {}
        for term, tf in Counter(tokens).items():
            index = params.vocab.get(term)
            if index is not None:
                weights[index] = tf * (params.k1 + 1) / (tf + params.k1 * length_norm)
        return _to_sparse_vector(weights)

    def encode_query(self, text: str, extra_terms: Sequence[str] = ()) -> SparseVector:
        """Sparse vector phía query: mỗi term trong vocabulary có trọng số IDF.

        `extra_terms` là token cấu trúc sinh từ câu hỏi gốc, nối vào danh sách
        token trước khi tính trọng số; term ngoài vocabulary bị bỏ.
        """
        params = self.params
        weights = {
            params.vocab[term]: params.idf[term]
            for term in set(tokenize(text) + list(extra_terms))
            if term in params.vocab
        }
        return _to_sparse_vector(weights)

    def save(self, path: Path) -> None:
        """Ghi tham số ra JSON (tạo thư mục cha nếu cần)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.params.model_dump(), ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> BM25Encoder:
        """Đọc tham số từ JSON đã `save`.

        Raises:
            BM25ParamsVersionError: Khi file thiếu/khác `params_version` kỳ
                vọng (index và params cũ không có token cấu trúc), kèm lệnh
                rebuild.
        """
        params = BM25Params.model_validate_json(path.read_text(encoding="utf-8"))
        if params.params_version != BM25_PARAMS_VERSION:
            raise BM25ParamsVersionError(
                f"{path} có params_version={params.params_version}, cần "
                f"{BM25_PARAMS_VERSION} (bản có token cấu trúc). Build lại sparse "
                f"index và params bằng: {REBUILD_COMMAND}"
            )
        return cls(params)


def _to_sparse_vector(weights: dict[int, float]) -> SparseVector:
    """Chuyển map index -> weight thành vector có indices tăng dần."""
    indices = sorted(weights)
    return SparseVector(indices=indices, values=[weights[i] for i in indices])
