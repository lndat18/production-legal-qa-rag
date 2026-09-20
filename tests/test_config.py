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
    GenerationSettings,
    GuardrailSettings,
    LLMSettings,
    RerankerSettings,
    VectorDBSettings,
)

# ==========================================================================
# EmbeddingSettings
# ==========================================================================


def test_embedding_settings_gia_tri_mac_dinh_dung_spec(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("MAX_TOKENS", raising=False)
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
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


def test_embedding_settings_bao_loi_khi_thieu_hf_token(
    monkeypatch: pytest.MonkeyPatch,
):
    """HF_TOKEN là bắt buộc để không rơi xuống anonymous quota (mục 7)."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        EmbeddingSettings,
        "model_config",
        {**EmbeddingSettings.model_config, "env_file": None},
    )
    with pytest.raises(ValidationError):
        EmbeddingSettings()  # type: ignore[call-arg]


def test_embedding_settings_doc_hf_token_tu_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    assert EmbeddingSettings().hf_token == "hf-test-token"


# ==========================================================================
# VectorDBSettings -- bắt buộc PINECONE_API_KEY / PINECONE_INDEX_NAME
# ==========================================================================


def test_vector_db_settings_doc_dung_bien_moi_truong(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    monkeypatch.setenv("PINECONE_SPARSE_INDEX_NAME", "test-sparse-index")
    settings = VectorDBSettings()  # type: ignore[call-arg]
    assert settings.pinecone_api_key == "test-key"
    assert settings.index_name == "test-index"
    assert settings.cloud == "aws"
    assert settings.region == "us-east-1"


def test_vector_db_settings_bao_loi_khi_thieu_pinecone_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    monkeypatch.setenv("PINECONE_SPARSE_INDEX_NAME", "test-sparse-index")
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
# LLMSettings -- bắt buộc GROQ_API_KEY
# ==========================================================================


def test_llm_settings_doc_dung_bien_moi_truong(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    settings = LLMSettings()  # type: ignore[call-arg]
    assert settings.groq_api_key == "test-groq-key"


def test_llm_settings_bao_loi_khi_thieu_groq_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(
        LLMSettings, "model_config", {**LLMSettings.model_config, "env_file": None}
    )
    with pytest.raises(ValidationError):
        LLMSettings()  # type: ignore[call-arg]


def test_llm_settings_gia_tri_mac_dinh_theo_spec(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    settings = LLMSettings()  # type: ignore[call-arg]
    assert settings.model_name == "openai/gpt-oss-120b"
    assert settings.max_retries == 2
    assert settings.timeout_seconds == 30
    assert settings.chunk_token_limit == 1500
    assert settings.tpm_limit == 8000
    assert settings.rpm_limit == 30


# ==========================================================================
# LLMSettings.groq_api_key_2 -- key thứ 2 TÙY CHỌN cho dispatch đồng thời
# (formatting_spec.md mục 1.3).
# ==========================================================================


def test_llm_settings_groq_api_key_2_mac_dinh_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GROQ_API_KEY_2", raising=False)
    monkeypatch.setattr(
        LLMSettings, "model_config", {**LLMSettings.model_config, "env_file": None}
    )
    settings = LLMSettings()  # type: ignore[call-arg]
    assert settings.groq_api_key_2 is None


def test_llm_settings_groq_api_key_2_doc_tu_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("GROQ_API_KEY_2", "test-groq-key-2")
    settings = LLMSettings()  # type: ignore[call-arg]
    assert settings.groq_api_key_2 == "test-groq-key-2"


def test_llm_settings_khong_co_groq_api_key_2_khong_bao_loi(
    monkeypatch: pytest.MonkeyPatch,
):
    # groq_api_key_2 là TÙY CHỌN -- thiếu nó không được raise ValidationError
    # (khác groq_api_key, field bắt buộc).
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("GROQ_API_KEY_2", raising=False)
    monkeypatch.setattr(
        LLMSettings, "model_config", {**LLMSettings.model_config, "env_file": None}
    )
    LLMSettings()  # type: ignore[call-arg] # không raise


# ==========================================================================
# GuardrailSettings / GenerationSettings (generation_spec.md mục 8)
# ===========================================================================


def test_guardrail_settings_doc_key_va_default_theo_spec(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "guardrail-key")
    settings = GuardrailSettings()  # type: ignore[call-arg]

    assert settings.api_key == "guardrail-key"
    assert settings.model_name == "openai/gpt-oss-safeguard-20b"
    assert settings.max_retries == 2
    assert settings.timeout_seconds == 30


def test_guardrail_settings_bao_loi_khi_thieu_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(
        GuardrailSettings,
        "model_config",
        {**GuardrailSettings.model_config, "env_file": None},
    )

    with pytest.raises(ValidationError):
        GuardrailSettings()  # type: ignore[call-arg]


def test_generation_settings_fallback_sang_key_guardrail_khi_key_2_khong_set(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "org-a-key")
    monkeypatch.delenv("GROQ_API_KEY_2", raising=False)
    monkeypatch.setattr(
        GenerationSettings,
        "model_config",
        {**GenerationSettings.model_config, "env_file": None},
    )

    settings = GenerationSettings()  # type: ignore[call-arg]

    assert settings.api_key == "org-a-key"
    assert settings.model_name == "openai/gpt-oss-120b"
    assert settings.max_retries == 2
    assert settings.timeout_seconds == 60


def test_generation_settings_uu_tien_key_2(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "org-a-key")
    monkeypatch.setenv("GROQ_API_KEY_2", "org-b-key")
    monkeypatch.setattr(
        GenerationSettings,
        "model_config",
        {**GenerationSettings.model_config, "env_file": None},
    )

    assert GenerationSettings().api_key == "org-b-key"  # type: ignore[call-arg]


# ==========================================================================
# VectorDBSettings.sparse_index_name / RerankerSettings (retrieval_spec.md mục 12)
# ==========================================================================


def test_vector_db_settings_doc_sparse_index_name(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    monkeypatch.setenv("PINECONE_SPARSE_INDEX_NAME", "sparse-index")
    assert VectorDBSettings().sparse_index_name == "sparse-index"  # type: ignore[call-arg]


def test_reranker_settings_gia_tri_mac_dinh(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANKER_ENDPOINT_URL", "https://example.test/predict")
    monkeypatch.setenv("RERANKER_API_KEY", "secret")
    settings = RerankerSettings()  # type: ignore[call-arg]
    assert settings.max_retries == 2
    assert settings.connect_timeout_seconds == 5
    assert settings.timeout_seconds == 30


def test_reranker_settings_bao_loi_khi_thieu_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RERANKER_ENDPOINT_URL", "https://example.test/predict")
    monkeypatch.delenv("RERANKER_API_KEY", raising=False)
    monkeypatch.setattr(
        RerankerSettings,
        "model_config",
        {**RerankerSettings.model_config, "env_file": None},
    )
    with pytest.raises(ValidationError):
        RerankerSettings()  # type: ignore[call-arg]
