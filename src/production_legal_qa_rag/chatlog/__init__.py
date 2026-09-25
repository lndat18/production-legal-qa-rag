"""Package ``chatlog`` — ghi lượt hỏi–đáp vào Postgres (chatlog_spec.md).

Public API:
- :class:`~production_legal_qa_rag.chatlog.models.TurnRecord`
- :func:`~production_legal_qa_rag.chatlog.models.from_trace`
- :class:`~production_legal_qa_rag.chatlog.repository.ChatLogRepository`
- :func:`~production_legal_qa_rag.chatlog.repository.create_engine`
"""

from production_legal_qa_rag.chatlog.models import TurnRecord, from_trace
from production_legal_qa_rag.chatlog.repository import ChatLogRepository, create_engine

__all__ = [
    "ChatLogRepository",
    "TurnRecord",
    "create_engine",
    "from_trace",
]
