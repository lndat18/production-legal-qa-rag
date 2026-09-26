"""CLI Typer test thủ công chatlog — ghi TurnRecord mẫu và thống kê.

Chạy không cần khởi động toàn bộ API (chatlog_spec.md mục 7, 8.3).

Lệnh có sẵn:
    ``write``   — Ghi TurnRecord mẫu vào ``chat_turns``.
    ``stats``   — In thống kê: tỉ lệ cache_status, phân bố outcome,
                  top error_code, p50/p95 latency và time_to_first_token.

Ví dụ:
    Ghi 1 dòng mẫu::

        uv run python tools/chatlog.py write

    Xem thống kê (mặc định 7 ngày gần nhất)::

        uv run python tools/chatlog.py stats

    Xem thống kê 30 ngày::

        uv run python tools/chatlog.py stats --days 30
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

import typer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from production_legal_qa_rag.chatlog.models import TurnRecord
from production_legal_qa_rag.chatlog.repository import (
    ChatLogRepository,
    create_engine,
)
from production_legal_qa_rag.config import DatabaseSettings

app = typer.Typer(help="Test thủ công chatlog: ghi mẫu và thống kê.")


@app.command()
def write(
    outcome: str = typer.Option(
        "answered", "--outcome", help="answered|refused|error|client_disconnected"
    ),
    cache_status: str = typer.Option(
        "miss", "--cache-status", help="answer_hit|retrieval_hit|miss|bypass"
    ),
) -> None:
    """Ghi một TurnRecord mẫu vào bảng ``chat_turns``.

    Args:
        outcome: Giá trị outcome của TurnRecord mẫu.
        cache_status: Giá trị cache_status của TurnRecord mẫu.
    """
    asyncio.run(_write(outcome=outcome, cache_status=cache_status))


async def _write(*, outcome: str, cache_status: str) -> None:
    """Thực hiện ghi mẫu bất đồng bộ.

    Args:
        outcome: Outcome của lượt hỏi–đáp mẫu.
        cache_status: Cache status mẫu.
    """
    settings = DatabaseSettings()
    engine = create_engine(settings.database_url)
    repo = ChatLogRepository(engine)

    sample = TurnRecord(
        id=str(uuid.uuid4()),
        request_id=f"req-{uuid.uuid4().hex[:8]}",
        user_id="user-tool-test",
        chat_id="chat-test-001",
        raw_query="Điều kiện để được hưởng trợ cấp thôi việc là gì?",
        standalone_query="Điều kiện hưởng trợ cấp thôi việc?",
        outcome=outcome,
        verdict="allow",
        error_code=None,
        cache_status=cache_status,
        chunk_ids=["chunk-001", "chunk-002"],
        answer_text="Người lao động được hưởng trợ cấp thôi việc khi...",
        citations=[
            {
                "n": 1,
                "chunk_id": "chunk-001",
                "source_document": "BLLĐ 2019",
                "breadcrumb": "Điều 46",
            }
        ],
        warnings=[],
        usage={
            "prompt_tokens": 500,
            "completion_tokens": 200,
            "reasoning_tokens": None,
        },
        time_to_first_token_ms=320,
        latency_ms=1450,
        prompt_version="v1.0",
        corpus_version="2026-09",
        model_name="openai/gpt-oss-120b",
    )

    await repo.record(sample)
    typer.echo(
        f"Đã ghi TurnRecord id={sample.id} (outcome={outcome}, cache={cache_status})"
    )
    await engine.dispose()


@app.command()
def stats(
    days: int = typer.Option(7, "--days", help="Số ngày nhìn lại. Mặc định: 7."),
) -> None:
    """In thống kê chatlog trong ``days`` ngày gần nhất.

    In: tỉ lệ cache_status, phân bố outcome, top error_code,
    p50/p95 time_to_first_token_ms và latency_ms.

    Args:
        days: Số ngày nhìn lại.
    """
    asyncio.run(_stats(days=days))


async def _stats(*, days: int) -> None:
    """Thực hiện truy vấn thống kê.

    Args:
        days: Số ngày nhìn lại.
    """
    settings = DatabaseSettings()
    engine = create_async_engine(settings.database_url, echo=False)

    async with engine.connect() as conn:
        # --- Tỉ lệ cache_status ---
        cache_sql = text("""
            SELECT cache_status, count(*) AS n
            FROM chat_turns
            WHERE created_at >= now() - :interval::interval
            GROUP BY cache_status
            ORDER BY n DESC
        """)
        cache_rows = (
            await conn.execute(cache_sql, {"interval": f"{days} days"})
        ).fetchall()

        # --- Phân bố outcome ---
        outcome_sql = text("""
            SELECT outcome, count(*) AS n
            FROM chat_turns
            WHERE created_at >= now() - :interval::interval
            GROUP BY outcome
            ORDER BY n DESC
        """)
        outcome_rows = (
            await conn.execute(outcome_sql, {"interval": f"{days} days"})
        ).fetchall()

        # --- Top error_code ---
        error_sql = text("""
            SELECT error_code, count(*) AS n
            FROM chat_turns
            WHERE created_at >= now() - :interval::interval
              AND error_code IS NOT NULL
            GROUP BY error_code
            ORDER BY n DESC
            LIMIT 10
        """)
        error_rows = (
            await conn.execute(error_sql, {"interval": f"{days} days"})
        ).fetchall()

        # --- Percentile latency ---
        latency_sql = text("""
            SELECT
                percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_latency,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency,
                percentile_cont(0.50) WITHIN GROUP (ORDER BY time_to_first_token_ms)
                    FILTER (WHERE time_to_first_token_ms IS NOT NULL) AS p50_ttft,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY time_to_first_token_ms)
                    FILTER (WHERE time_to_first_token_ms IS NOT NULL) AS p95_ttft
            FROM chat_turns
            WHERE created_at >= now() - :interval::interval
        """)
        lat_row = (
            await conn.execute(latency_sql, {"interval": f"{days} days"})
        ).fetchone()

    await engine.dispose()

    # --- In kết quả ---
    typer.echo(f"\n=== Chatlog stats — {days} ngày gần nhất ===\n")

    typer.echo("📦 Cache status:")
    for row in cache_rows:
        typer.echo(f"  {row[0]:<20} {row[1]}")

    typer.echo("\n✅ Outcome:")
    for row in outcome_rows:
        typer.echo(f"  {row[0]:<20} {row[1]}")

    if error_rows:
        typer.echo("\n❌ Top error_code:")
        for row in error_rows:
            typer.echo(f"  {row[0]:<30} {row[1]}")
    else:
        typer.echo("\n❌ Top error_code: (không có)")

    if lat_row:
        typer.echo("\n⏱ Latency (ms):")
        typer.echo(f"  p50={lat_row[0]:.0f}  p95={lat_row[1]:.0f}")
        typer.echo("⏱ Time to first token (ms):")
        p50_ttft = f"{lat_row[2]:.0f}" if lat_row[2] is not None else "N/A"
        p95_ttft = f"{lat_row[3]:.0f}" if lat_row[3] is not None else "N/A"
        typer.echo(f"  p50={p50_ttft}  p95={p95_ttft}")


if __name__ == "__main__":
    app()
