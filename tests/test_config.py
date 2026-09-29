"""Data/Schema validation cho `production_legal_qa_rag/config.py` (chunking_spec.md mục 7).

Dùng `monkeypatch.setenv`/`delenv` để không phụ thuộc nội dung thật của
`.env` trong repo (biến môi trường luôn ưu tiên hơn `.env` với
`pydantic-settings`, nên override được dù `.env` có giá trị khác).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from production_legal_qa_rag.config import (
    EmbeddingSettings,
    GenerationSettings,
    GuardrailSettings,
    HydeSettings,
    JudgeSettings,
    LangfuseSettings,
    LLMSettings,
    RerankerSettings,
    ThrottleSettings,
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
# LLMSettings -- bắt buộc GROQ_API_KEY_1
# ==========================================================================


def test_llm_settings_doc_dung_bien_moi_truong(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY_1", "test-groq-key")
    settings = LLMSettings()  # type: ignore[call-arg]
    assert settings.groq_api_key == "test-groq-key"


def test_llm_settings_bao_loi_khi_thieu_groq_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GROQ_API_KEY_1", raising=False)
    monkeypatch.setattr(
        LLMSettings, "model_config", {**LLMSettings.model_config, "env_file": None}
    )
    with pytest.raises(ValidationError):
        LLMSettings()  # type: ignore[call-arg]


def test_llm_settings_gia_tri_mac_dinh_theo_spec(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY_1", "test-groq-key")
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
    monkeypatch.setenv("GROQ_API_KEY_1", "test-groq-key")
    monkeypatch.delenv("GROQ_API_KEY_2", raising=False)
    monkeypatch.setattr(
        LLMSettings, "model_config", {**LLMSettings.model_config, "env_file": None}
    )
    settings = LLMSettings()  # type: ignore[call-arg]
    assert settings.groq_api_key_2 is None


def test_llm_settings_groq_api_key_2_doc_tu_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY_1", "test-groq-key")
    monkeypatch.setenv("GROQ_API_KEY_2", "test-groq-key-2")
    settings = LLMSettings()  # type: ignore[call-arg]
    assert settings.groq_api_key_2 == "test-groq-key-2"


def test_llm_settings_khong_co_groq_api_key_2_khong_bao_loi(
    monkeypatch: pytest.MonkeyPatch,
):
    # groq_api_key_2 là TÙY CHỌN -- thiếu nó không được raise ValidationError
    # (khác groq_api_key, field bắt buộc).
    monkeypatch.setenv("GROQ_API_KEY_1", "test-groq-key")
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
    monkeypatch.setenv("GROQ_API_KEY_1", "guardrail-key")
    settings = GuardrailSettings()  # type: ignore[call-arg]

    assert settings.api_key == "guardrail-key"
    assert settings.model_name == "openai/gpt-oss-safeguard-20b"
    assert settings.max_retries == 2
    assert settings.timeout_seconds == 30


def test_guardrail_settings_bao_loi_khi_thieu_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GROQ_API_KEY_1", raising=False)
    monkeypatch.setattr(
        GuardrailSettings,
        "model_config",
        {**GuardrailSettings.model_config, "env_file": None},
    )

    with pytest.raises(ValidationError):
        GuardrailSettings()  # type: ignore[call-arg]


def test_generation_settings_fallback_sang_key_guardrail_khi_key_3_khong_set(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GROQ_API_KEY_1", "org-a-key")
    monkeypatch.delenv("GROQ_API_KEY_3", raising=False)
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


def test_generation_settings_uu_tien_key_3(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY_1", "org-a-key")
    monkeypatch.setenv("GROQ_API_KEY_2", "org-b-key")
    monkeypatch.setenv("GROQ_API_KEY_3", "org-c-key")
    monkeypatch.setattr(
        GenerationSettings,
        "model_config",
        {**GenerationSettings.model_config, "env_file": None},
    )

    assert GenerationSettings().api_key == "org-c-key"  # type: ignore[call-arg]


def test_generation_settings_khong_dung_key_2_cua_nhom_nhe(
    monkeypatch: pytest.MonkeyPatch,
):
    """Key 2 thuộc nhóm bước nhẹ (mục 12.1): generation không được lấy nó."""
    monkeypatch.setenv("GROQ_API_KEY_1", "org-a-key")
    monkeypatch.setenv("GROQ_API_KEY_2", "org-b-key")
    monkeypatch.delenv("GROQ_API_KEY_3", raising=False)
    monkeypatch.setattr(
        GenerationSettings,
        "model_config",
        {**GenerationSettings.model_config, "env_file": None},
    )

    assert GenerationSettings().api_key == "org-a-key"  # type: ignore[call-arg]


def test_generation_settings_doc_key_4_cho_round_robin(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY_1", "org-a-key")
    monkeypatch.setenv("GROQ_API_KEY_3", "org-c-key")
    monkeypatch.delenv("GROQ_API_KEY_4", raising=False)
    monkeypatch.setattr(
        GenerationSettings,
        "model_config",
        {**GenerationSettings.model_config, "env_file": None},
    )

    assert GenerationSettings().round_robin_api_key is None  # type: ignore[call-arg]

    monkeypatch.setenv("GROQ_API_KEY_4", "org-d-key")

    assert GenerationSettings().round_robin_api_key == "org-d-key"  # type: ignore[call-arg]


def test_judge_settings_uses_key_2_and_independent_defaults(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GROQ_API_KEY_1", "org-a-key")
    monkeypatch.setenv("GROQ_API_KEY_2", "org-b-key")
    monkeypatch.setenv("GROQ_API_KEY_3", "org-c-key")
    # GROQ_JUDGE_API_KEY đã bỏ (mục 12.1): có đặt cũng không được ưu tiên.
    monkeypatch.setenv("GROQ_JUDGE_API_KEY", "legacy-judge-key")
    monkeypatch.setattr(
        JudgeSettings,
        "model_config",
        {**JudgeSettings.model_config, "env_file": None},
    )

    settings = JudgeSettings()  # type: ignore[call-arg]

    assert settings.api_key == "org-b-key"
    assert settings.model_name == "openai/gpt-oss-20b"
    assert settings.max_retries == 1
    assert settings.timeout_seconds == 45


def test_judge_settings_falls_back_to_groq_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("GROQ_API_KEY_2", raising=False)
    monkeypatch.setenv("GROQ_API_KEY_1", "org-a-key")
    monkeypatch.setattr(
        JudgeSettings,
        "model_config",
        {**JudgeSettings.model_config, "env_file": None},
    )

    assert JudgeSettings().api_key == "org-a-key"  # type: ignore[call-arg]


# ==========================================================================
# HydeSettings / ThrottleSettings (conversation_spec.md mục 12.1)
# ==========================================================================


def test_hyde_settings_dung_key_1_va_model_20b(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY_1", "org-a-key")
    monkeypatch.setenv("GROQ_API_KEY_2", "org-b-key")
    monkeypatch.setattr(
        HydeSettings, "model_config", {**HydeSettings.model_config, "env_file": None}
    )

    settings = HydeSettings()  # type: ignore[call-arg]

    assert settings.api_key == "org-a-key"
    assert settings.model_name == "openai/gpt-oss-20b"
    assert settings.max_retries == 2
    assert settings.timeout_seconds == 30


def test_hyde_settings_bao_loi_khi_thieu_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GROQ_API_KEY_1", raising=False)
    monkeypatch.setattr(
        HydeSettings, "model_config", {**HydeSettings.model_config, "env_file": None}
    )

    with pytest.raises(ValidationError):
        HydeSettings()  # type: ignore[call-arg]


def test_throttle_settings_gia_tri_mac_dinh(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "TPM_LIMIT",
        "RPM_LIMIT",
        "SAFETY_FACTOR",
        "CHARS_PER_TOKEN",
        "CONDENSE_COMPLETION_TOKENS",
        "HYDE_COMPLETION_TOKENS",
        "JUDGE_COMPLETION_TOKENS",
        "OPTIONAL_STEP_MAX_WAIT_SECONDS",
    ):
        monkeypatch.delenv(f"THROTTLE_{name}", raising=False)
    monkeypatch.setattr(
        ThrottleSettings,
        "model_config",
        {**ThrottleSettings.model_config, "env_file": None},
    )

    settings = ThrottleSettings()

    assert settings.tpm_limit == 8000
    assert settings.rpm_limit == 30
    assert settings.safety_factor == 0.9
    assert settings.optional_step_max_wait_seconds == 8.0


def test_throttle_settings_doc_env_prefix_throttle(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("THROTTLE_TPM_LIMIT", "1234")
    monkeypatch.setenv("THROTTLE_RPM_LIMIT", "5")
    monkeypatch.setattr(
        ThrottleSettings,
        "model_config",
        {**ThrottleSettings.model_config, "env_file": None},
    )

    settings = ThrottleSettings()

    assert settings.tpm_limit == 1234
    assert settings.rpm_limit == 5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tpm_limit", 0),
        ("rpm_limit", 0),
        ("safety_factor", 0),
        ("safety_factor", 1.5),
        ("chars_per_token", 0),
        ("optional_step_max_wait_seconds", -1),
    ],
)
def test_throttle_settings_tu_choi_gia_tri_khong_hop_le(field: str, value: float):
    with pytest.raises(ValidationError):
        ThrottleSettings.model_validate({field: value})


# ==========================================================================
# VectorDBSettings.sparse_index_name / RerankerSettings (retrieval_spec.md mục 12)
# ==========================================================================


def test_vector_db_settings_doc_sparse_index_name(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    monkeypatch.setenv("PINECONE_SPARSE_INDEX_NAME", "sparse-index")
    assert VectorDBSettings().sparse_index_name == "sparse-index"  # type: ignore[call-arg]


def test_reranker_settings_gia_tri_mac_dinh(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RERANKER_MODEL_NAME", raising=False)
    monkeypatch.delenv("RERANKER_MAX_LENGTH", raising=False)
    monkeypatch.delenv("RERANKER_BATCH_SIZE", raising=False)
    settings = RerankerSettings()  # type: ignore[call-arg]
    assert settings.model_name == "AITeamVN/Vietnamese_Reranker"
    assert settings.max_length == 512
    assert settings.batch_size == 16


def test_reranker_settings_bo_qua_bien_rong_trong_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "RERANKER_MODEL_NAME=\nRERANKER_MAX_LENGTH=\nRERANKER_BATCH_SIZE=\n",
        encoding="utf-8",
    )
    for name in (
        "RERANKER_MODEL_NAME",
        "RERANKER_MAX_LENGTH",
        "RERANKER_BATCH_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        RerankerSettings,
        "model_config",
        {**RerankerSettings.model_config, "env_file": dotenv},
    )

    settings = RerankerSettings()  # type: ignore[call-arg]

    assert (settings.model_name, settings.max_length, settings.batch_size) == (
        "AITeamVN/Vietnamese_Reranker",
        512,
        16,
    )


def test_reranker_settings_doc_env_va_validate_gia_tri_duong(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("RERANKER_MODEL_NAME", "custom-reranker")
    monkeypatch.setenv("RERANKER_MAX_LENGTH", "256")
    monkeypatch.setenv("RERANKER_BATCH_SIZE", "8")
    settings = RerankerSettings()  # type: ignore[call-arg]
    assert (settings.model_name, settings.max_length, settings.batch_size) == (
        "custom-reranker",
        256,
        8,
    )

    monkeypatch.setenv("RERANKER_MAX_LENGTH", "0")
    with pytest.raises(ValidationError):
        RerankerSettings()  # type: ignore[call-arg]

    monkeypatch.setenv("RERANKER_MAX_LENGTH", "512")
    monkeypatch.setenv("RERANKER_BATCH_SIZE", "-1")
    with pytest.raises(ValidationError):
        RerankerSettings()  # type: ignore[call-arg]


# ==========================================================================
# LangfuseSettings (observability_spec.md mục 6)
# ==========================================================================


def _khong_doc_dotenv_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bỏ qua `.env` thật của máy dev — chỉ dùng biến môi trường test set tường
    minh (đối chiếu docstring đầu file: env var luôn ưu tiên hơn `.env`, nhưng ở
    đây cần cả trường hợp "hoàn toàn không set" nên phải tắt hẳn nguồn dotenv).
    """
    monkeypatch.setattr(
        LangfuseSettings,
        "model_config",
        {**LangfuseSettings.model_config, "env_file": None},
    )


def test_langfuse_settings_mac_dinh_disabled_khi_thieu_ca_hai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mục 4.1: thiếu public_key/secret_key -> SDK tự chuyển sang chế độ
    disabled, không cần cờ bật/tắt riêng."""
    _khong_doc_dotenv_langfuse(monkeypatch)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    settings = LangfuseSettings()

    assert settings.public_key is None
    assert settings.secret_key is None
    assert settings.base_url == "http://localhost:3001"


def test_langfuse_settings_doc_key_tu_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _khong_doc_dotenv_langfuse(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:3001")

    settings = LangfuseSettings()

    assert settings.public_key is not None
    assert settings.public_key.get_secret_value() == "pk-test"
    assert settings.secret_key is not None
    assert settings.secret_key.get_secret_value() == "sk-test"
    assert settings.base_url == "http://localhost:3001"


def test_langfuse_settings_bo_qua_gia_tri_rong(monkeypatch: pytest.MonkeyPatch) -> None:
    """`env_ignore_empty=True`: chuỗi rỗng (vd `.env.example` để trống mặc định)
    không được coi là đã cấu hình key thật."""
    _khong_doc_dotenv_langfuse(monkeypatch)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")

    settings = LangfuseSettings()

    assert settings.public_key is None
    assert settings.secret_key is None
