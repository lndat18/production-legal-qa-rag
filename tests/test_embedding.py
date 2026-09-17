"""Unit và integration test cho checkpoint embedding và Pinecone refresh.

Mọi interaction HuggingFace/Pinecone trong file này dùng fake client. Test
chỉ bảo vệ contract trong ``embedding_spec.md`` và không gọi dịch vụ ngoài.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, cast

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from production_legal_qa_rag.chunking.models import Chunk
from production_legal_qa_rag.config import EmbeddingSettings, VectorDBSettings
from production_legal_qa_rag.embedding import hf_client, pinecone_client, pipeline
from production_legal_qa_rag.embedding.hf_client import (
    HFRequestLimitExceeded,
    HuggingFaceEmbedder,
)
from production_legal_qa_rag.embedding.models import (
    EmbeddedChunk,
    PineconeMetadata,
)
from production_legal_qa_rag.embedding.pinecone_client import PineconeVectorStore


def _chunk(index: int, *, has_table: bool = False) -> Chunk:
    """Tạo Chunk hợp lệ với dữ liệu phân biệt theo ``index``."""
    return Chunk(
        chunk_id=f"chunk-{index}",
        source_document="Văn bản mẫu",
        breadcrumb=f"Văn bản mẫu - Điều {index}",
        content=f"Nội dung {index}",
        token_count=index + 1,
        has_table=has_table,
        raw_table="| Cột |\n| --- |" if has_table else None,
    )


def _embedded_chunk(index: int, *, has_table: bool = False) -> EmbeddedChunk:
    """Tạo checkpoint EmbeddedChunk hợp lệ."""
    return EmbeddedChunk(
        **_chunk(index, has_table=has_table).model_dump(),
        embedding=[float(index), float(index + 1)],
    )


def _embedding_settings(monkeypatch: pytest.MonkeyPatch) -> EmbeddingSettings:
    """Tạo config HF từ environment giả, không dùng secret thật."""
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    return EmbeddingSettings()


def _vector_settings(monkeypatch: pytest.MonkeyPatch) -> VectorDBSettings:
    """Tạo config Pinecone từ environment giả."""
    monkeypatch.setenv("PINECONE_API_KEY", "pinecone-test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "legal-index")
    return VectorDBSettings()  # type: ignore[call-arg]


# ==========================================================================
# models.py -- checkpoint schema và Pinecone metadata (mục 2, 9)
# ==========================================================================


def test_embedded_chunk_ke_thua_day_du_chunk_schema():
    embedded = _embedded_chunk(1, has_table=True)

    assert embedded.chunk_id == "chunk-1"
    assert embedded.raw_table == "| Cột |\n| --- |"
    assert embedded.embedding == [1.0, 2.0]


def test_embedded_chunk_thieu_embedding_bi_tu_choi():
    with pytest.raises(ValidationError):
        EmbeddedChunk(**_chunk(1).model_dump())


def test_pinecone_metadata_bat_buoc_raw_table_khi_co_bang():
    with pytest.raises(ValidationError, match="raw_table"):
        PineconeMetadata(
            content="nội dung",
            breadcrumb="Điều 1",
            source_document="Văn bản",
            has_table=True,
        )


def test_pinecone_record_bo_raw_table_khi_chunk_khong_co_bang():
    record = pinecone_client.to_pinecone_record(_embedded_chunk(1))

    payload = record.model_dump(mode="json", exclude_none=True)
    assert payload["id"] == "chunk-1"
    assert payload["values"] == [1.0, 2.0]
    assert "raw_table" not in payload["metadata"]
    assert set(payload["metadata"]) == {
        "content",
        "breadcrumb",
        "source_document",
        "has_table",
    }


def test_pinecone_record_luu_raw_table_khi_chunk_co_bang():
    record = pinecone_client.to_pinecone_record(_embedded_chunk(1, has_table=True))

    assert record.metadata.raw_table == "| Cột |\n| --- |"


# ==========================================================================
# hf_client.py -- segment, batching, retry và quota dùng chung (mục 3, 4)
# ==========================================================================


class _FakeHFClient:
    """Ghi request HF và trả response/exception đã lập trình sẵn."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls: list[list[str]] = []

    def feature_extraction(self, inputs: list[str]) -> object:
        self.calls.append(inputs)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return response(inputs)
        return response


def test_embedder_segment_batch_va_giu_dung_thu_tu(
    monkeypatch: pytest.MonkeyPatch,
):
    client = _FakeHFClient(
        [lambda inputs: [[float(index)] for index, _ in enumerate(inputs)]] * 2
    )
    monkeypatch.setattr(
        hf_client.ViTokenizer, "tokenize", lambda content: f"seg:{content}"
    )
    embedder = HuggingFaceEmbedder(_embedding_settings(monkeypatch), client)  # type: ignore[arg-type]

    embedded, skipped = embedder.embed_chunks([_chunk(index) for index in range(26)])

    assert [len(call) for call in client.calls] == [25, 1]
    assert client.calls[0][0] == "seg:Nội dung 0"
    assert [chunk.chunk_id for chunk in embedded] == [
        f"chunk-{index}" for index in range(26)
    ]
    assert [chunk.embedding for chunk in embedded] == [
        [float(index)] for index in range(25)
    ] + [[0.0]]
    assert skipped == []
    assert embedder.request_count == 2


def test_embedder_retry_het_lan_bo_ca_batch_va_tiep_tuc_batch_sau(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(hf_client, "HF_BATCH_SIZE", 2)
    monkeypatch.setattr(hf_client.ViTokenizer, "tokenize", lambda content: content)
    client = _FakeHFClient(
        [
            RuntimeError("HF lỗi"),
            RuntimeError("HF lỗi"),
            RuntimeError("HF lỗi"),
            [[9.0]],
        ]
    )
    embedder = HuggingFaceEmbedder(_embedding_settings(monkeypatch), client)  # type: ignore[arg-type]

    embedded, skipped = embedder.embed_chunks([_chunk(index) for index in range(3)])

    assert [chunk.chunk_id for chunk in embedded] == ["chunk-2"]
    assert skipped == ["chunk-0", "chunk-1"]
    assert len(client.calls) == 4
    assert embedder.request_count == 4


def test_embedder_dung_truoc_request_vuot_quota(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(hf_client, "HF_BATCH_SIZE", 1)
    monkeypatch.setattr(hf_client, "HF_RPD_SAFE_LIMIT", 1)
    monkeypatch.setattr(hf_client.ViTokenizer, "tokenize", lambda content: content)
    client = _FakeHFClient([[[1.0]]])
    embedder = HuggingFaceEmbedder(_embedding_settings(monkeypatch), client)  # type: ignore[arg-type]

    with pytest.raises(HFRequestLimitExceeded):
        embedder.embed_chunks([_chunk(0), _chunk(1)])

    assert len(client.calls) == 1
    assert embedder.request_count == 1


def test_coerce_embeddings_tu_response_tolist_va_validate_shape():
    response = SimpleNamespace(tolist=lambda: [[1, 2.5]])

    assert hf_client._coerce_embeddings(response, expected_count=1) == [[1.0, 2.5]]
    with pytest.raises(ValueError, match="đúng số vector"):
        hf_client._coerce_embeddings([], expected_count=1)
    with pytest.raises(TypeError, match="không phải list"):
        hf_client._coerce_embeddings(["không phải vector"], expected_count=1)


# ==========================================================================
# pipeline.py -- checkpoint từng file, lỗi partial và pha upsert (mục 2, 8)
# ==========================================================================


class _FakeEmbedder:
    """Fake embedder giữ số lần khởi tạo để bảo vệ quota counter dùng chung."""

    instances: ClassVar[list[_FakeEmbedder]] = []

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.instances.append(self)

    def embed_chunks(
        self, chunks: list[Chunk]
    ) -> tuple[list[EmbeddedChunk], list[str]]:
        self.calls.append([chunk.chunk_id for chunk in chunks])
        return (
            [
                EmbeddedChunk(
                    **chunk.model_dump(), embedding=[float(chunk.token_count)]
                )
                for chunk in chunks
            ],
            [],
        )


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    """Ghi input JSON cho integration test checkpoint."""
    path.write_text(
        json.dumps([chunk.model_dump(mode="json") for chunk in chunks]),
        encoding="utf-8",
    )


def _write_embedded_chunks(path: Path, chunks: list[EmbeddedChunk]) -> None:
    """Ghi checkpoint JSON hợp lệ."""
    path.write_text(
        json.dumps([chunk.model_dump(mode="json") for chunk in chunks]),
        encoding="utf-8",
    )


def test_embed_ghi_checkpoint_1_1_khong_sua_input_va_dung_chung_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    chunks_dir = tmp_path / "chunks"
    embeddings_dir = tmp_path / "embeddings"
    chunks_dir.mkdir()
    first_source = chunks_dir / "a.json"
    _write_chunks(first_source, [_chunk(0)])
    _write_chunks(chunks_dir / "b.json", [_chunk(1)])
    original_input = first_source.read_text(encoding="utf-8")
    _FakeEmbedder.instances.clear()
    monkeypatch.setattr(pipeline, "HuggingFaceEmbedder", _FakeEmbedder)

    assert pipeline.embed(chunks_dir, embeddings_dir) == 0

    assert first_source.read_text(encoding="utf-8") == original_input
    assert sorted(path.name for path in embeddings_dir.glob("*.json")) == [
        "a.json",
        "b.json",
    ]
    checkpoint = json.loads((embeddings_dir / "a.json").read_text(encoding="utf-8"))
    assert checkpoint[0]["chunk_id"] == "chunk-0"
    assert checkpoint[0]["embedding"] == [1.0]
    assert len(_FakeEmbedder.instances) == 1
    assert _FakeEmbedder.instances[0].calls == [["chunk-0"], ["chunk-1"]]


def test_embed_loi_mot_file_khong_chan_checkpoint_file_khac(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    chunks_dir = tmp_path / "chunks"
    embeddings_dir = tmp_path / "embeddings"
    chunks_dir.mkdir()
    (chunks_dir / "a-loi.json").write_text("{}", encoding="utf-8")
    _write_chunks(chunks_dir / "b-hop-le.json", [_chunk(1)])
    _FakeEmbedder.instances.clear()
    monkeypatch.setattr(pipeline, "HuggingFaceEmbedder", _FakeEmbedder)

    assert pipeline.embed(chunks_dir, embeddings_dir) == 1

    assert not (embeddings_dir / "a-loi.json").exists()
    assert (embeddings_dir / "b-hop-le.json").exists()


def test_write_atomic_xoa_file_tam_khi_fsync_loi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "embeddings" / "checkpoint.json"

    def _raise_fsync(_: int) -> None:
        raise OSError("đĩa lỗi")

    monkeypatch.setattr(pipeline.os, "fsync", _raise_fsync)
    with pytest.raises(OSError, match="đĩa lỗi"):
        pipeline.write_atomic(output, [_embedded_chunk(1)])

    assert not output.exists()
    assert list(output.parent.glob("*.tmp")) == []


def test_upsert_doc_toan_bo_checkpoint_va_goi_full_refresh_mot_lan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    _write_embedded_chunks(embeddings_dir / "b.json", [_embedded_chunk(2)])
    _write_embedded_chunks(embeddings_dir / "a.json", [_embedded_chunk(1)])
    captured: list[list[EmbeddedChunk]] = []
    monkeypatch.setattr(
        pipeline, "upsert_embedded_chunks", lambda chunks: captured.append(chunks)
    )

    assert pipeline.upsert(embeddings_dir) == 0

    assert [[chunk.chunk_id for chunk in chunks] for chunks in captured] == [
        ["chunk-1", "chunk-2"]
    ]


def test_upsert_tra_exit_code_1_khi_checkpoint_khong_hop_le(tmp_path: Path):
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    (embeddings_dir / "hong.json").write_text("{}", encoding="utf-8")

    assert pipeline.upsert(embeddings_dir) == 1


def test_cli_khong_upsert_neu_pha_embed_tra_loi(monkeypatch: pytest.MonkeyPatch):
    from tools import embed_documents

    called = False

    def _upsert(_: Path) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(embed_documents, "embed", lambda _chunks, _embeddings: 1)
    monkeypatch.setattr(embed_documents, "upsert", _upsert)

    result = CliRunner().invoke(embed_documents.app, [])

    assert result.exit_code == 1
    assert called is False


# ==========================================================================
# pinecone_client.py -- create/index/delete/upsert batches (mục 2, 5)
# ==========================================================================


class _FakePineconeIndex:
    """Ghi thứ tự delete/upsert để kiểm tra full refresh."""

    def __init__(self, events: list[tuple[str, object]]) -> None:
        self._events = events

    def delete(self, *, delete_all: bool) -> None:
        self._events.append(("delete", delete_all))

    def upsert(self, *, vectors: list[dict[str, object]]) -> None:
        self._events.append(("upsert", vectors))


class _FakePineconeClient:
    """Fake Pinecone SDK chỉ giữ calls cần cho contract của vector store."""

    def __init__(self, indexes: list[str]) -> None:
        self._indexes = indexes
        self.events: list[tuple[str, object]] = []
        self.index = _FakePineconeIndex(self.events)

    def list_indexes(self) -> list[str]:
        return self._indexes

    def create_index(self, **kwargs: object) -> None:
        self.events.append(("create", kwargs))

    def Index(self, name: str) -> _FakePineconeIndex:
        self.events.append(("Index", name))
        return self.index


def test_vector_store_tao_index_full_refresh_va_upsert_theo_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    client = _FakePineconeClient([])
    monkeypatch.setattr(pinecone_client, "get_embedding_dimension", lambda _: 768)
    monkeypatch.setattr(pinecone_client, "PINECONE_UPSERT_BATCH_SIZE", 2)
    store = PineconeVectorStore(
        _vector_settings(monkeypatch),
        _embedding_settings(monkeypatch),
        client,  # type: ignore[arg-type]
    )

    store.upsert([_embedded_chunk(index) for index in range(3)])

    create_kwargs = next(value for event, value in client.events if event == "create")
    assert isinstance(create_kwargs, dict)
    assert create_kwargs["name"] == "legal-index"
    assert create_kwargs["dimension"] == 768
    assert create_kwargs["metric"] == "cosine"
    spec = cast(pinecone_client.ServerlessSpec, create_kwargs["spec"])
    assert spec.cloud == "aws"
    assert spec.region == "us-east-1"
    assert client.events[1] == ("Index", "legal-index")
    assert client.events[2] == ("delete", True)
    batches = [value for event, value in client.events if event == "upsert"]
    assert [len(batch) for batch in batches] == [2, 1]
    assert [record["id"] for batch in batches for record in batch] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]


def test_vector_store_khong_tao_lai_index_da_ton_tai(
    monkeypatch: pytest.MonkeyPatch,
):
    client = _FakePineconeClient(["legal-index"])
    store = PineconeVectorStore(
        _vector_settings(monkeypatch),
        _embedding_settings(monkeypatch),
        client,  # type: ignore[arg-type]
    )

    store.upsert([])

    assert all(event != "create" for event, _ in client.events)
    assert client.events == [("Index", "legal-index"), ("delete", True)]


@pytest.mark.parametrize("hidden_size", [None, 0, -1, "768"])
def test_get_embedding_dimension_tu_choi_hidden_size_khong_hop_le(
    monkeypatch: pytest.MonkeyPatch, hidden_size: object
):
    monkeypatch.setattr(
        pinecone_client.AutoConfig,
        "from_pretrained",
        lambda _: SimpleNamespace(hidden_size=hidden_size),
    )

    with pytest.raises(ValueError, match="hidden_size"):
        pinecone_client.get_embedding_dimension("model-test")


def test_get_embedding_dimension_doc_hidden_size_tu_model_config(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        pinecone_client.AutoConfig,
        "from_pretrained",
        lambda _: SimpleNamespace(hidden_size=768),
    )

    assert pinecone_client.get_embedding_dimension("model-test") == 768
