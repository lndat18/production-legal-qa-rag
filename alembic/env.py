"""Alembic env.py — hỗ trợ asyncpg và autogenerate từ metadata chatlog.

Cấu hình asyncio runtime cho alembic (async engine) và trỏ
``target_metadata`` về schema ``chat_turns`` để alembic autogenerate hoạt động.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# Metadata của chatlog để autogenerate nhận diện bảng.
from production_legal_qa_rag.chatlog.tables import (
    metadata as target_metadata,
)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def run_migrations_offline() -> None:
    """Chạy migration ở chế độ offline (không kết nối DB thật).

    Dùng khi sinh SQL script để review trước khi apply.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):  # type: ignore[no-untyped-def]
    """Chạy migration với connection đã cấp.

    Args:
        connection: SQLAlchemy async connection.
    """
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Tạo async engine và chạy migration trong coroutine."""
    url = config.get_main_option("sqlalchemy.url")
    connectable = create_async_engine(url, poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point cho chế độ online; chạy coroutine async."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
