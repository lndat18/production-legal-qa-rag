"""Tạo bảng chat_turns.

Revision ID: 0001_create_chat_turns
Revises:
Create Date: 2026-09-25

Bảng ghi lượt hỏi–đáp theo chatlog_spec.md mục 2.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_create_chat_turns"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Tạo bảng ``chat_turns`` và các index."""
    op.create_table(
        "chat_turns",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("request_id", sa.Text, nullable=False),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column("chat_id", sa.Text, nullable=True),
        sa.Column("raw_query", sa.Text, nullable=False),
        sa.Column("standalone_query", sa.Text, nullable=True),
        sa.Column("outcome", sa.String(32), nullable=False),
        sa.Column("verdict", sa.String(32), nullable=False),
        sa.Column("error_code", sa.Text, nullable=True),
        sa.Column("cache_status", sa.String(32), nullable=False),
        sa.Column(
            "chunk_ids",
            JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "answer_text",
            sa.Text,
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "citations",
            JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "warnings",
            JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("usage", JSONB, nullable=True),
        sa.Column("time_to_first_token_ms", sa.Integer, nullable=True),
        sa.Column(
            "latency_ms",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "prompt_version",
            sa.Text,
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "corpus_version",
            sa.Text,
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "model_name",
            sa.Text,
            nullable=False,
            server_default=sa.text("''"),
        ),
    )
    op.create_index("ix_chat_turns_created_at", "chat_turns", ["created_at"])
    op.create_index("ix_chat_turns_user_id", "chat_turns", ["user_id"])
    op.create_index("ix_chat_turns_chat_id", "chat_turns", ["chat_id"])


def downgrade() -> None:
    """Xóa bảng ``chat_turns`` và các index."""
    op.drop_index("ix_chat_turns_chat_id", table_name="chat_turns")
    op.drop_index("ix_chat_turns_user_id", table_name="chat_turns")
    op.drop_index("ix_chat_turns_created_at", table_name="chat_turns")
    op.drop_table("chat_turns")
