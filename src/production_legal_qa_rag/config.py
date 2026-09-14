"""Config tập trung dùng chung cho embedding model, vector DB (Pinecone), LLM.

Nằm **ngoài** package `chunking/` vì không thuộc riêng 1 business logic nào —
`embedding/` và các bước sau (retrieval/generation) cũng import từ đây thay
vì đọc `.env` trực tiếp (chunking_spec.md mục 7).

Chỉ khai field đã có giá trị thật hoặc đã chốt tên biến — không thêm field
cho tính năng chưa được thiết kế (tránh over-engineering).
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    """Config embedding model — dùng để đếm token (chunking/) và sinh vector
    (embedding/, ngoài phạm vi package này)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_name: str = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
    max_tokens: int = 236


class VectorDBSettings(BaseSettings):
    """Config Pinecone — VectorDB đã chốt (đổi từ Postgres/pgvector ban đầu).

    `docker-compose.yml` hiện vẫn có thể chạy Postgres cho việc khác ngoài
    lưu vector — spec không quyết định có bỏ Postgres hay không, để ngỏ cho
    quyết định sau.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    pinecone_api_key: str = Field(validation_alias="PINECONE_API_KEY")
    index_name: str = Field(validation_alias="PINECONE_INDEX_NAME")


class LLMSettings(BaseSettings):
    """Config LLM. Tên model chưa chốt — thuộc phạm vi bước retrieval/
    generation sau này, không đoán trước ở đây."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = Field(validation_alias="GROQ_API_KEY")
