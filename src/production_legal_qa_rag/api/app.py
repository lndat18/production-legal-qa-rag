"""Application factory và lifecycle của API (api_spec.md mục 7, 12).

``create_app()`` (dùng với ``uvicorn --factory``) khởi tạo mọi dependency đúng
một lần trong ``lifespan`` — Redis, engine Postgres, cache, admission và
``ChatOrchestrator`` — rồi lưu vào ``app.state`` để route lấy qua dependency.
Không tạo lại client (Groq/HF/Pinecone/reranker) cho mỗi request.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from production_legal_qa_rag.api.routes import router
from production_legal_qa_rag.api.schemas import ApiError, ErrorDetail, ErrorResponse
from production_legal_qa_rag.cache.keys import compute_corpus_version
from production_legal_qa_rag.cache.replay import replay
from production_legal_qa_rag.cache.singleflight import SingleFlight
from production_legal_qa_rag.cache.store import AnswerCache, RetrievalCache
from production_legal_qa_rag.config import (
    ApiSettings,
    CacheSettings,
    GenerationSettings,
    RedisSettings,
)
from production_legal_qa_rag.conversation.admission import AdmissionController
from production_legal_qa_rag.conversation.orchestrator import ChatOrchestrator
from production_legal_qa_rag.generation.generator import PROMPT_VERSION
from production_legal_qa_rag.observability import metrics
from production_legal_qa_rag.observability.tracing import get_langfuse_client
from production_legal_qa_rag.observability.turn_trace import RuntimeVersions

_logger = logging.getLogger(__name__)
# observability_spec.md mục 4.4: chặn tối đa vài giây để không mất trace của
# các lượt hỏi cuối, nhưng không giữ shutdown vô hạn nếu Langfuse không phản hồi.
_LANGFUSE_FLUSH_TIMEOUT_SECONDS = 5.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Khởi tạo mọi dependency của API một lần, đóng lại khi tắt."""
    api_settings = ApiSettings()
    cache_settings = CacheSettings()
    redis_settings = RedisSettings()
    generation_settings = GenerationSettings()

    versions = RuntimeVersions(
        prompt_version=PROMPT_VERSION,
        corpus_version=compute_corpus_version(override=cache_settings.corpus_version),
        model_name=generation_settings.model_name,
    )

    redis: Redis = Redis.from_url(redis_settings.redis_url)
    answer_cache = AnswerCache(
        redis,
        corpus_version=versions.corpus_version,
        prompt_version=versions.prompt_version,
        model_name=versions.model_name,
    )
    retrieval_cache = RetrievalCache(redis, corpus_version=versions.corpus_version)
    single_flight = SingleFlight(redis, answer_cache)
    admission = AdmissionController()
    # guardrail/condenser/generation không truyền vào đây: ChatOrchestrator tự
    # tạo (và giữ) đúng một instance mỗi loại theo default của nó — không có
    # route nào khác cần dùng riêng các thành phần này (khác registry nêu ở
    # mục 7, nhưng cùng tính chất "khởi tạo 1 lần" vì orchestrator chỉ được
    # tạo đúng một lần ở đây).
    orchestrator = ChatOrchestrator(
        admission=admission,
        answer_cache=answer_cache,
        retrieval_cache=retrieval_cache,
        single_flight=single_flight,
        replay=replay,
    )

    app.state.api_settings = api_settings
    app.state.redis = redis
    app.state.runtime_versions = versions
    app.state.orchestrator = orchestrator
    try:
        yield
    finally:
        await _flush_langfuse()
        await redis.aclose()


async def _flush_langfuse() -> None:
    """Gửi nốt trace đang chờ trước khi container dừng (observability_spec.md mục 4.4).

    Không bao giờ ném ngoại lệ hay chặn shutdown quá ``_LANGFUSE_FLUSH_TIMEOUT_SECONDS``:
    Langfuse không phản hồi (hoặc disabled) không được làm chậm việc tắt API.
    """
    try:
        await asyncio.wait_for(
            asyncio.to_thread(get_langfuse_client().flush),
            timeout=_LANGFUSE_FLUSH_TIMEOUT_SECONDS,
        )
    except Exception:
        _logger.warning("Không thể flush Langfuse khi shutdown.", exc_info=True)


def _error_response(
    status_code: int,
    message: str,
    error_type: str,
    code: str,
    *,
    retry_after_seconds: float | None = None,
) -> JSONResponse:
    headers = (
        {"Retry-After": str(int(retry_after_seconds))}
        if retry_after_seconds is not None
        else None
    )
    body = ErrorResponse(error=ErrorDetail(message=message, type=error_type, code=code))
    return JSONResponse(body.model_dump(), status_code=status_code, headers=headers)


def _register_exception_handlers(app: FastAPI) -> None:
    """Đăng ký handler chuyển mọi lỗi trước-stream thành body OpenAI (mục 12)."""

    @app.exception_handler(ApiError)
    async def _handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(
            exc.status_code,
            exc.message,
            exc.error_type,
            exc.code,
            retry_after_seconds=exc.retry_after_seconds,
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            422,
            "Request không hợp lệ.",
            "invalid_request_error",
            "invalid_request",
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        _logger.exception(
            "Lỗi không lường trước trong route (request_id=%s)", request_id
        )
        return _error_response(
            500, "Đã xảy ra lỗi nội bộ.", "internal_error", "internal_error"
        )


def create_app() -> FastAPI:
    """Tạo FastAPI application với router, lifecycle và exception handler."""
    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    _register_exception_handlers(app)
    metrics.instrument_app(app)
    return app
