"""Application factory và lifecycle cho dependency chatlog của API.

Endpoint OpenAI-compliant sẽ được thêm ở api_spec.md phase API. Lifecycle ở
đây đã khởi tạo duy nhất engine/repository và đóng task ghi an toàn, để mọi
route dùng ``stream_chat_turn`` có cùng hành vi logging theo chatlog_spec.md.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from production_legal_qa_rag.api.routes import ChatLogTaskManager
from production_legal_qa_rag.cache.keys import compute_corpus_version
from production_legal_qa_rag.chatlog.models import ChatLogMetadata
from production_legal_qa_rag.chatlog.repository import ChatLogRepository, create_engine
from production_legal_qa_rag.config import (
    CacheSettings,
    DatabaseSettings,
    GenerationSettings,
)
from production_legal_qa_rag.generation.generator import PROMPT_VERSION


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Khởi tạo và đóng dependency chatlog một lần cho cả API process."""
    database_settings = DatabaseSettings()
    cache_settings = CacheSettings()
    generation_settings = GenerationSettings()
    engine = create_engine(database_settings.database_url)
    repository = ChatLogRepository(engine)
    metadata = ChatLogMetadata(
        prompt_version=PROMPT_VERSION,
        corpus_version=compute_corpus_version(override=cache_settings.corpus_version),
        model_name=generation_settings.model_name,
    )
    app.state.chatlog_repository = repository
    app.state.chatlog_tasks = ChatLogTaskManager(repository, metadata)
    try:
        yield
    finally:
        await app.state.chatlog_tasks.drain()
        await engine.dispose()


def create_app() -> FastAPI:
    """Tạo FastAPI application với lifecycle chatlog đã đăng ký."""
    return FastAPI(lifespan=lifespan)
