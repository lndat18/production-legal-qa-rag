"""Định nghĩa bảng ``chat_turns`` bằng SQLAlchemy Core (không ORM class).

Dùng SQLAlchemy ``Table``/``Column`` thay vì Declarative ORM để tách rõ
contract database khỏi Pydantic model — hai lớp không được kế thừa nhau.
"""

from __future__ import annotations

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

# Metadata dùng chung cho Alembic và repository.
metadata = MetaData()

chat_turns = Table(
    "chat_turns",
    metadata,
    Column("id", UUID(as_uuid=False), primary_key=True),
    # Dùng server_default để DB tự sinh nếu app không truyền;
    # app luôn truyền giá trị để đảm bảo nhất quán với TurnRecord.id.
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default="now()",
    ),
    Column("request_id", Text, nullable=False),
    Column("user_id", Text, nullable=False),
    Column("chat_id", Text, nullable=True),
    Column("raw_query", Text, nullable=False),
    Column("standalone_query", Text, nullable=True),
    Column("outcome", String(32), nullable=False),
    Column("verdict", String(32), nullable=False),
    Column("error_code", Text, nullable=True),
    Column("cache_status", String(32), nullable=False),
    Column("chunk_ids", JSONB, nullable=False, server_default="'[]'"),  # type: ignore[call-arg]
    Column("answer_text", Text, nullable=False, server_default="''"),  # type: ignore[call-arg]
    Column("citations", JSONB, nullable=False, server_default="'[]'"),  # type: ignore[call-arg]
    Column("warnings", JSONB, nullable=False, server_default="'[]'"),  # type: ignore[call-arg]
    Column("usage", JSONB, nullable=True),
    Column("time_to_first_token_ms", Integer, nullable=True),
    Column("latency_ms", Integer, nullable=False, server_default="0"),  # type: ignore[call-arg]
    Column("prompt_version", Text, nullable=False, server_default="''"),  # type: ignore[call-arg]
    Column("corpus_version", Text, nullable=False, server_default="''"),  # type: ignore[call-arg]
    Column("model_name", Text, nullable=False, server_default="''"),  # type: ignore[call-arg]
)

# Index cho các query phổ biến (mục 8 spec: thống kê, retention).
Index("ix_chat_turns_created_at", chat_turns.c.created_at)
Index("ix_chat_turns_user_id", chat_turns.c.user_id)
Index("ix_chat_turns_chat_id", chat_turns.c.chat_id)
