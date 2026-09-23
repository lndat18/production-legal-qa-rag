"""Config tập trung dùng chung cho embedding model, vector DB (Pinecone), LLM.

Nằm **ngoài** package `chunking/` vì không thuộc riêng 1 business logic nào —
`embedding/` và các bước sau (retrieval/generation) cũng import từ đây thay
vì đọc `.env` trực tiếp (chunking_spec.md mục 7).

Chỉ khai field đã có giá trị thật hoặc đã chốt tên biến — không thêm field
cho tính năng chưa được thiết kế (tránh over-engineering).
"""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    """Config embedding model — dùng để đếm token (chunking/) và sinh vector
    (embedding/, ngoài phạm vi package này)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_name: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
    max_tokens: int = 236
    hf_token: str = Field(validation_alias="HF_TOKEN")


class VectorDBSettings(BaseSettings):
    """Config Pinecone — VectorDB đã chốt (đổi từ Postgres/pgvector ban đầu).

    `docker-compose.yml` hiện vẫn có thể chạy Postgres cho việc khác ngoài
    lưu vector — spec không quyết định có bỏ Postgres hay không, để ngỏ cho
    quyết định sau.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    pinecone_api_key: str = Field(validation_alias="PINECONE_API_KEY")
    index_name: str = Field(validation_alias="PINECONE_INDEX_NAME")
    sparse_index_name: str = Field(validation_alias="PINECONE_SPARSE_INDEX_NAME")
    cloud: str = "aws"
    region: str = "us-east-1"


class LLMSettings(BaseSettings):
    """Config LLM dùng chung cho nhiều bước.

    `model_name`/`max_retries`/`timeout_seconds` đã chốt cho nhu cầu của
    `formatting/` (chuyển đổi front matter/back matter sang markdown bằng
    Groq API free tier, `openai/gpt-oss-120b`, xem `formatting_spec.md` mục
    1.1, 4). Bước retrieval/generation sau này có thể cần model khác — chưa
    đoán trước ở đây, để ngỏ cho quyết định sau (tránh over-engineering).

    `chunk_token_limit`/`tpm_limit`/`rpm_limit` phục vụ riêng cơ chế chunking
    + sliding-window rate limiter của `formatting/llm_client.py` (mục 1.2
    spec) — Groq free tier cho `openai/gpt-oss-120b` giới hạn RPM 30, RPD
    1.000, TPM 8.000, TPD 200.000; `tpm_limit`/`rpm_limit` ở đây là giới hạn
    gốc (chưa nhân hệ số an toàn — hệ số 0.9 áp dụng ngay trong
    `llm_client.py`, không lưu ở đây để tránh hai nơi cùng giữ một hằng số
    dẫn xuất).

    `groq_api_key_2` (mục 1.3 spec) — key Groq thứ 2, TÙY CHỌN. Khi có mặt,
    `llm_client.convert_chunks_concurrently` dispatch job qua 2 thread worker
    chạy đồng thời (mỗi worker gắn chết 1 key, 1 rate-limiter độc lập) thay
    vì lặp tuần tự bằng key 1. Không set → giữ nguyên hành vi tuần tự 1 key
    hiện có, không lỗi, không cảnh báo.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = Field(validation_alias="GROQ_API_KEY")
    groq_api_key_2: str | None = Field(default=None, validation_alias="GROQ_API_KEY_2")
    model_name: str = "openai/gpt-oss-120b"
    max_retries: int = 2
    timeout_seconds: int = 30
    chunk_token_limit: int = 1500
    tpm_limit: int = 8000
    rpm_limit: int = 30


class GuardrailSettings(BaseSettings):
    """Cấu hình Groq riêng cho bước kiểm tra an toàn đầu vào."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: str = Field(validation_alias="GROQ_API_KEY")
    model_name: str = "openai/gpt-oss-safeguard-20b"
    max_retries: int = 2
    timeout_seconds: int = 30


class CondenseSettings(BaseSettings):
    """Cấu hình Groq cho bước condense câu follow-up (conversation_spec.md mục 10).

    Dùng model ``gpt-oss-20b`` để ngân sách rate limit tách khỏi HyDE và
    generation (cùng ``gpt-oss-120b``). Timeout ngắn vì condense lỗi thì
    degrade về câu gốc, không đáng để người dùng chờ.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: str = Field(validation_alias="GROQ_API_KEY")
    model_name: str = "openai/gpt-oss-20b"
    max_retries: int = 1
    timeout_seconds: int = 20


class AdmissionSettings(BaseSettings):
    """Hạn mức và đồng thời của phần tốn LLM (conversation_spec.md mục 8, 10).

    Số liệu vận hành, chỉnh qua env được mà không cần deploy lại.
    ``global_daily_llm_answers`` ≈ TPD 200K của ``gpt-oss-120b`` chia 3-4K
    token/câu, chừa dư địa cho HyDE. ``max_concurrent_answers`` = 2 vì mỗi câu
    ~3-4K token trên TPM 8K.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    redis_url: str = Field(validation_alias="REDIS_URL")
    user_daily_llm_answers: int = 5
    global_daily_llm_answers: int = 50
    max_concurrent_answers: int = 2
    max_waiting: int = 6


class GenerationSettings(BaseSettings):
    """Cấu hình Groq cho bước sinh câu trả lời có stream.

    Ưu tiên ``GROQ_API_KEY_2`` để tách ngân sách rate limit khỏi HyDE và
    guardrail; khi không đặt key thứ hai, dùng lại ``GROQ_API_KEY``.
    """

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_ignore_empty=True
    )

    api_key: str = Field(
        validation_alias=AliasChoices("GROQ_API_KEY_2", "GROQ_API_KEY")
    )
    model_name: str = "openai/gpt-oss-120b"
    max_retries: int = 2
    timeout_seconds: int = 60


class JudgeSettings(BaseSettings):
    """Cấu hình Evidence Judge độc lập với client generation.

    Judge dùng cùng cơ chế fallback key với generator nhưng có model, timeout và
    retry riêng để thay đổi evaluator không ảnh hưởng prompt tạo answer.
    """

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", env_ignore_empty=True
    )

    api_key: str = Field(
        validation_alias=AliasChoices(
            "GROQ_JUDGE_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY"
        )
    )
    model_name: str = "openai/gpt-oss-120b"
    max_retries: int = 1
    timeout_seconds: int = 45


class RerankerSettings(BaseSettings):
    """Config reranker tự host trên LightningAI (retrieval_spec.md mục 9, 12).

    `connect_timeout_seconds` tách khỏi read timeout để Studio đang sleep không
    khiến 1 câu hỏi chờ quá lâu ở bước kết nối. `timeout_seconds` là **sàn** của
    read timeout: read thực tế = `max(timeout_seconds, 2.5 × n_passages)`
    (`RERANK_SECONDS_PER_PASSAGE` trong `retrieval/reranker_client.py`).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    endpoint_url: str = Field(validation_alias="RERANKER_ENDPOINT_URL")
    api_key: str = Field(validation_alias="RERANKER_API_KEY")
    max_retries: int = 2
    connect_timeout_seconds: int = 5
    timeout_seconds: int = 30
