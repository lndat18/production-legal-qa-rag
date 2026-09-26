"""CLI Typer xóa dữ liệu chatlog quá hạn (chatlog_spec.md mục 5, 7).

Chạy tay hoặc qua cron của host. Mặc định xóa các dòng có ``created_at``
cũ hơn ``CHATLOG_RETENTION_DAYS`` ngày (mặc định 90 ngày).

Ví dụ:
    Xóa dữ liệu quá 90 ngày::

        uv run python tools/purge_chatlog.py

    Xóa dữ liệu quá 30 ngày::

        uv run python tools/purge_chatlog.py --days 30

    Xóa toàn bộ dữ liệu của một user cụ thể::

        uv run python tools/purge_chatlog.py --user-id abc123

    Kết hợp::

        uv run python tools/purge_chatlog.py --user-id abc123 --days 30
"""

from __future__ import annotations

import asyncio
import os
import sys

import typer
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import create_async_engine

# Đảm bảo import được package khi chạy trực tiếp.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from production_legal_qa_rag.chatlog.tables import chat_turns
from production_legal_qa_rag.config import DatabaseSettings

app = typer.Typer(help="Công cụ xóa dữ liệu chatlog quá hạn.")

_DEFAULT_DAYS = 90


@app.command()
def purge(
    days: int = typer.Option(
        _DEFAULT_DAYS,
        "--days",
        help="Xóa dòng cũ hơn số ngày này. Mặc định: 90.",
    ),
    user_id: str | None = typer.Option(
        None,
        "--user-id",
        help="Nếu truyền, chỉ xóa dòng của user này.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="In SQL và số dòng sẽ bị xóa mà không thực sự xóa.",
    ),
) -> None:
    """Xóa dòng chatlog quá hạn hoặc theo user_id.

    Args:
        days: Số ngày lưu trữ tối đa (mặc định 90).
        user_id: Chỉ xóa dữ liệu của user này nếu truyền.
        dry_run: Nếu True, chỉ in kết quả dự kiến, không xóa thật.
    """
    asyncio.run(_run_purge(days=days, user_id=user_id, dry_run=dry_run))


async def _run_purge(
    *,
    days: int,
    user_id: str | None,
    dry_run: bool,
) -> None:
    """Thực hiện DELETE bất đồng bộ.

    Args:
        days: Số ngày lưu trữ tối đa.
        user_id: Lọc theo user nếu không None.
        dry_run: Nếu True, chỉ đếm dòng, không xóa.
    """
    settings = DatabaseSettings()
    engine = create_async_engine(settings.database_url, echo=False)

    cutoff_expr = text("now() - :interval::interval").bindparams(
        interval=f"{days} days"
    )
    stmt = delete(chat_turns).where(chat_turns.c.created_at < cutoff_expr)
    if user_id is not None:
        stmt = stmt.where(chat_turns.c.user_id == user_id)

    async with engine.begin() as conn:
        if dry_run:
            # Đếm dòng sẽ bị xóa bằng select(func.count()).select_from() đúng SQLAlchemy 2.x.
            count_stmt = (
                select(func.count())
                .select_from(chat_turns)
                .where(chat_turns.c.created_at < cutoff_expr)
            )
            if user_id is not None:
                count_stmt = count_stmt.where(chat_turns.c.user_id == user_id)
            result = await conn.execute(count_stmt)
            n = result.scalar()
            typer.echo(f"[dry-run] Sẽ xóa {n} dòng (days={days}, user_id={user_id})")
        else:
            result = await conn.execute(stmt)
            typer.echo(
                f"Đã xóa {result.rowcount} dòng (days={days}, user_id={user_id})"
            )

    await engine.dispose()


if __name__ == "__main__":
    app()
