"""Data/Schema validation cho `production_legal_qa_rag/config.py` (chunking_spec.md mục 7).

Dùng `monkeypatch.setenv`/`delenv` để không phụ thuộc nội dung thật của
`.env` trong repo (biến môi trường luôn ưu tiên hơn `.env` với
`pydantic-settings`, nên override được dù `.env` có giá trị khác).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from production_legal_qa_rag.config import (
    EmbeddingSettings,
    LLMSettings,
    VectorDBSettings,
)

# ==========================================================================
# EmbeddingSettings
# ==========================================================================


def test_embedding_settings_gia_tri_mac_dinh_dung_spec(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("MAX_TOKENS", raising=False)
    settings = EmbeddingSettings()
    assert settings.model_name == "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
    assert settings.max_tokens == 236


def test_embedding_settings_max_tokens_doc_duoc_tu_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAX_TOKENS", "100")
    settings = EmbeddingSettings()
    assert settings.max_tokens == 100


def test_embedding_settings_max_tokens_phai_la_so_nguyen(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MAX_TOKENS", "không phải số")
    with pytest.raises(ValidationError):
        EmbeddingSettings()


# ==========================================================================
# VectorDBSettings -- bắt buộc PINECONE_API_KEY / PINECONE_INDEX_NAME
# ==========================================================================


def test_vector_db_settings_doc_dung_bien_moi_truong(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    settings = VectorDBSettings()  # type: ignore[call-arg]
    assert settings.pinecone_api_key == "test-key"
    assert settings.index_name == "test-index"


def test_vector_db_settings_bao_loi_khi_thieu_pinecone_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    monkeypatch.setattr(
        VectorDBSettings,
        "model_config",
        {**VectorDBSettings.model_config, "env_file": None},
    )
    with pytest.raises(ValidationError):
        VectorDBSettings()  # type: ignore[call-arg]


def test_vector_db_settings_bao_loi_khi_thieu_index_name(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.delenv("PINECONE_INDEX_NAME", raising=False)
    monkeypatch.setattr(
        VectorDBSettings,
        "model_config",
        {**VectorDBSettings.model_config, "env_file": None},
    )
    with pytest.raises(ValidationError):
        VectorDBSettings()  # type: ignore[call-arg]


# ==========================================================================
# LLMSettings -- bắt buộc GEMINI_API_KEY
# ==========================================================================


def test_llm_settings_doc_dung_bien_moi_truong(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    settings = LLMSettings()  # type: ignore[call-arg]
    assert settings.gemini_api_key == "test-gemini-key"


def test_llm_settings_bao_loi_khi_thieu_gemini_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        LLMSettings, "model_config", {**LLMSettings.model_config, "env_file": None}
    )
    with pytest.raises(ValidationError):
        LLMSettings()  # type: ignore[call-arg]
