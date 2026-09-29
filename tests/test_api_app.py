"""Test lifecycle của `api/app.py` (api_spec.md mục 7, observability_spec.md mục 4.4/4.5).

Lifespan được chạy với mọi dependency nặng (Redis, cache, orchestrator) thay bằng
bản giả và settings dựng tường minh (`_env_file=None`), nên test không đụng mạng, không
đọc `.env` thật. Trọng tâm: `RuntimeVersions` tạo đúng một lần và gắn vào
`app.state.runtime_versions`; `api` không còn engine Postgres; shutdown flush Langfuse
mà không bao giờ raise hay treo quá timeout.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI

import production_legal_qa_rag.api.app as app_module
from production_legal_qa_rag.config import (
    ApiSettings,
    CacheSettings,
    GenerationSettings,
)
from production_legal_qa_rag.generation.generator import PROMPT_VERSION
from production_legal_qa_rag.observability.turn_trace import RuntimeVersions


class _FakeRedis:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _BrokenClient:
    def flush(self) -> None:
        raise RuntimeError("langfuse chết")


class _SlowClient:
    def flush(self) -> None:
        time.sleep(0.6)


def _api_settings() -> ApiSettings:
    return ApiSettings(CHATBOT_API_KEY="test-key", _env_file=None)


def _cache_settings() -> CacheSettings:
    return CacheSettings(CACHE_CORPUS_VERSION="corpus-x", _env_file=None)


def _generation_settings() -> GenerationSettings:
    return GenerationSettings(
        GROQ_API_KEY_3="test-groq-key", model_name="model-x", _env_file=None
    )


def _capture(created: dict[str, Any], name: str) -> Callable[..., Any]:
    """Constructor giả: ghi lại đối số lifespan truyền vào rồi trả một đối tượng đặc."""

    def factory(*args: Any, **kwargs: Any) -> Any:
        instance = SimpleNamespace(args=args, kwargs=kwargs)
        created[name] = instance
        return instance

    return factory


def _patch_lifespan_dependencies(
    monkeypatch: pytest.MonkeyPatch, redis: _FakeRedis, created: dict[str, Any]
) -> None:
    monkeypatch.setattr(app_module, "ApiSettings", _api_settings)
    monkeypatch.setattr(app_module, "CacheSettings", _cache_settings)
    monkeypatch.setattr(app_module, "GenerationSettings", _generation_settings)
    fake_redis_class = SimpleNamespace(from_url=lambda _: redis)
    monkeypatch.setattr(app_module, "Redis", fake_redis_class)
    for name in (
        "AnswerCache",
        "RetrievalCache",
        "SingleFlight",
        "AdmissionController",
        "ChatOrchestrator",
    ):
        monkeypatch.setattr(app_module, name, _capture(created, name))


# ----------------------------------------------------------------------- lifespan


def test_lifespan_tao_runtime_versions_mot_lan_va_gan_vao_app_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = _FakeRedis()
    created: dict[str, Any] = {}
    flushed: list[bool] = []
    _patch_lifespan_dependencies(monkeypatch, redis, created)

    async def fake_flush() -> None:
        flushed.append(True)

    monkeypatch.setattr(app_module, "_flush_langfuse", fake_flush)
    app = FastAPI()
    expected = RuntimeVersions(
        prompt_version=PROMPT_VERSION,
        corpus_version="corpus-x",
        model_name="model-x",
    )

    async def scenario() -> None:
        async with app_module.lifespan(app):
            assert app.state.runtime_versions == expected
            assert app.state.redis is redis
            assert app.state.orchestrator is created["ChatOrchestrator"]
            secret = app.state.api_settings.chatbot_api_key.get_secret_value()
            assert secret == "test-key"
            assert redis.closed is False

    asyncio.run(scenario())

    assert redis.closed is True
    assert flushed == [True]


def test_lifespan_answer_cache_dung_cung_phien_ban_voi_runtime_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Version gắn vào trace phải là đúng version mà cache đã thực sự dùng để phục vụ."""
    created: dict[str, Any] = {}
    _patch_lifespan_dependencies(monkeypatch, _FakeRedis(), created)
    monkeypatch.setattr(app_module, "_flush_langfuse", lambda: asyncio.sleep(0))
    app = FastAPI()

    async def scenario() -> None:
        async with app_module.lifespan(app):
            pass

    asyncio.run(scenario())

    assert created["AnswerCache"].kwargs == {
        "corpus_version": "corpus-x",
        "prompt_version": PROMPT_VERSION,
        "model_name": "model-x",
    }
    assert created["RetrievalCache"].kwargs == {"corpus_version": "corpus-x"}


def test_lifespan_khong_con_engine_postgres_trong_app_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gỡ chatlog: `api` chỉ còn Redis, `/readyz` không có gì để ping Postgres."""
    _patch_lifespan_dependencies(monkeypatch, _FakeRedis(), {})
    monkeypatch.setattr(app_module, "_flush_langfuse", lambda: asyncio.sleep(0))
    app = FastAPI()

    async def scenario() -> None:
        async with app_module.lifespan(app):
            assert not hasattr(app.state, "database_engine")
            assert not hasattr(app.state, "chatlog_tasks")

    asyncio.run(scenario())


def test_lifespan_van_dong_redis_va_flush_khi_body_ung_dung_loi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = _FakeRedis()
    flushed: list[bool] = []
    _patch_lifespan_dependencies(monkeypatch, redis, {})

    async def fake_flush() -> None:
        flushed.append(True)

    monkeypatch.setattr(app_module, "_flush_langfuse", fake_flush)
    app = FastAPI()

    async def scenario() -> None:
        async with app_module.lifespan(app):
            raise RuntimeError("app lỗi")

    with pytest.raises(RuntimeError, match="app lỗi"):
        asyncio.run(scenario())

    assert redis.closed is True
    assert flushed == [True]


# -------------------------------------------------------------- _flush_langfuse


def test_flush_langfuse_disabled_khong_raise() -> None:
    asyncio.run(app_module._flush_langfuse())


def test_flush_langfuse_nuot_loi_va_chi_log_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(app_module, "get_langfuse_client", _BrokenClient)
    caplog.set_level(logging.WARNING, logger=app_module.__name__)

    asyncio.run(app_module._flush_langfuse())

    assert any("flush Langfuse" in record.getMessage() for record in caplog.records)


def test_flush_langfuse_khong_chan_shutdown_qua_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "get_langfuse_client", _SlowClient)
    monkeypatch.setattr(app_module, "_LANGFUSE_FLUSH_TIMEOUT_SECONDS", 0.05)

    async def scenario() -> float:
        started = time.monotonic()
        await app_module._flush_langfuse()
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())

    assert elapsed < 0.45
