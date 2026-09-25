"""Repository ghi lượt hỏi–đáp vào Postgres bất đồng bộ (chatlog_spec.md mục 4).

Engine và session factory khởi tạo một lần ở lifespan API; repository nhận
engine qua constructor để dễ test/inject.

Giao diện công khai duy nhất: :meth:`ChatLogRepository.record` — không bao giờ
ném ngoại lệ ra ngoài; lỗi ghi chỉ log warning mà không làm hỏng câu trả lời.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from production_legal_qa_rag.chatlog.models import TurnRecord
from production_legal_qa_rag.chatlog.tables import chat_turns

_logger = logging.getLogger(__name__)

# Timeout tối đa cho một lần INSERT (spec mục 4: 5 s).
_WRITE_TIMEOUT_SECONDS = 5.0


def create_engine(database_url: str) -> AsyncEngine:
    """Tạo async engine cho Postgres.

    Args:
        database_url: URL dạng ``postgresql+asyncpg://…``.

    Returns:
        AsyncEngine đã cấu hình, chưa kết nối.
    """
    return create_async_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=0,
    )


class ChatLogRepository:
    """Ghi ``TurnRecord`` vào bảng ``chat_turns`` bất đồng bộ.

    Engine phải được tạo qua :func:`create_engine` và truyền vào constructor.
    Repository không đóng engine; vòng đời engine thuộc về lifespan API.

    Args:
        engine: AsyncEngine đã được khởi tạo ở lifespan.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )

    async def record(self, turn: TurnRecord) -> None:
        """Ghi một lượt hỏi–đáp vào ``chat_turns``.

        Không bao giờ ném ngoại lệ — lỗi chỉ emit warning log.
        Timeout 5 giây để không chặn lifespan shutdown quá lâu.

        Args:
            turn: Dữ liệu lượt hỏi–đáp cần ghi.
        """
        try:
            await asyncio.wait_for(
                self._insert(turn),
                timeout=_WRITE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            _logger.warning(
                "chatlog: ghi timeout sau %.0fs (request_id=%s)",
                _WRITE_TIMEOUT_SECONDS,
                turn.request_id,
            )
        except Exception:
            # Không log nội dung turn để bảo vệ quyền riêng tư (spec mục 5).
            _logger.warning(
                "chatlog: ghi thất bại (request_id=%s)",
                turn.request_id,
                exc_info=True,
            )

    async def _insert(self, turn: TurnRecord) -> None:
        """Thực hiện INSERT đơn; không public vì không bắt ngoại lệ.

        Args:
            turn: Dữ liệu cần INSERT.
        """
        async with self._session_factory() as session:
            await session.execute(
                chat_turns.insert().values(
                    id=turn.id,
                    request_id=turn.request_id,
                    user_id=turn.user_id,
                    chat_id=turn.chat_id,
                    raw_query=turn.raw_query,
                    standalone_query=turn.standalone_query,
                    outcome=turn.outcome,
                    verdict=turn.verdict,
                    error_code=turn.error_code,
                    cache_status=turn.cache_status,
                    chunk_ids=turn.chunk_ids,
                    answer_text=turn.answer_text,
                    citations=turn.citations,
                    warnings=turn.warnings,
                    usage=turn.usage,
                    time_to_first_token_ms=turn.time_to_first_token_ms,
                    latency_ms=turn.latency_ms,
                    prompt_version=turn.prompt_version,
                    corpus_version=turn.corpus_version,
                    model_name=turn.model_name,
                )
            )
            await session.commit()
