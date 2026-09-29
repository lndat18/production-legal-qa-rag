"""Thêm cột langfuse_trace_id vào chat_turns.

Revision ID: 0002_add_langfuse_trace_id
Revises: 0001_create_chat_turns
Create Date: 2026-09-29

Cột nullable để đối chiếu qua lại Postgres <-> Langfuse UI
(observability_spec.md mục 4.4); NULL khi Langfuse disabled (thiếu key).
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_add_langfuse_trace_id"
down_revision: str | None = "0001_create_chat_turns"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Thêm cột ``langfuse_trace_id`` (nullable) vào ``chat_turns``."""
    op.add_column(
        "chat_turns",
        sa.Column("langfuse_trace_id", sa.Text, nullable=True),
    )


def downgrade() -> None:
    """Xoá cột ``langfuse_trace_id``."""
    op.drop_column("chat_turns", "langfuse_trace_id")
