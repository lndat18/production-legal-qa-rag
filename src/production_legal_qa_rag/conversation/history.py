"""Cửa sổ history từ ``messages[]`` của client (conversation_spec.md mục 4).

``messages[]`` là dữ liệu không tin cậy: hàm ở đây chỉ lọc, cắt và làm sạch để
dùng làm *dữ liệu* trong prompt condense/guardrail, không bao giờ làm chỉ dẫn.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel

from production_legal_qa_rag.conversation.models import ChatMessage

# Khối "Nguồn" do `api/` nối vào câu trả lời; hằng số ở đây để `api/` import.
SOURCES_FOOTER_MARKER: Final = "\n\n---\n**Nguồn**\n"

# Ngày (thủ công) coi như thời điểm dữ liệu pháp luật trong corpus được cập nhật gần
# nhất. KHÔNG có nguồn tự động đáng tin cậy để suy ra giá trị này (`corpus_version` ở
# `cache_spec.md` chỉ là hash BM25, không phải ngày; mtime file không phản ánh ngày ban
# hành/sửa đổi văn bản luật thật). Theo conversation_spec.md mục 4, ngày này phải
# được cập nhật thủ công mỗi lần re-index corpus, không tự động hoá.
# Placeholder ban đầu: không tìm được mốc re-index corpus rõ ràng trong lịch sử git (các
# commit embedding/chunking chỉ phản ánh ngày merge code, không phải ngày build index
# thật), nên dùng ngày của bản spec hiện tại (2026-09-23).
CORPUS_SNAPSHOT_DATE: Final = "2026-09-23"

# Disclaimer cố định do `api/` nối vào cuối câu trả lời SSE, sau khối "Nguồn" (cùng nhóm
# "phần đuôi cố định do `api/` nối vào câu trả lời" với `SOURCES_FOOTER_MARKER`; hằng số
# ở đây để `api/` import — conversation_spec.md mục 4).
DATA_SNAPSHOT_DISCLAIMER: Final = (
    "\n\n_Dữ liệu pháp luật trong hệ thống được cập nhật tới "
    f"{CORPUS_SNAPSHOT_DATE}; có thể chưa phản ánh sửa đổi, bổ sung mới nhất. Vui lòng "
    "đối chiếu văn bản chính thức hoặc cơ quan có thẩm quyền khi cần độ chính xác cao "
    "nhất._"
)

MAX_QUERY_CHARS: Final = 1000
HISTORY_MAX_TURNS: Final = 3
HISTORY_ASSISTANT_MAX_CHARS: Final = 600
GUARDRAIL_CONTEXT_TURNS: Final = 2

_CITATION_MARK = re.compile(r"\[\d+\]")
_ELLIPSIS: Final = "…"


class InvalidConversationError(ValueError):
    """``messages[]`` không hợp lệ; lớp API trả 422 trước khi stream."""


class HistoryWindow(BaseModel):
    """Câu hỏi hiện tại cùng history đã cắt và làm sạch."""

    query: str
    history: list[ChatMessage]

    @property
    def has_history(self) -> bool:
        """Có history để condense hay không (lượt đầu thì không)."""
        return len(self.history) > 0

    @property
    def recent_user_turns(self) -> list[str]:
        """Tối đa ``GUARDRAIL_CONTEXT_TURNS`` câu user gần nhất trước ``query``."""
        user_turns = [m.content for m in self.history if m.role == "user"]
        return user_turns[-GUARDRAIL_CONTEXT_TURNS:]


def build_window(messages: Sequence[ChatMessage]) -> HistoryWindow:
    """Dựng cửa sổ history từ toàn bộ ``messages[]``.

    Args:
        messages: Danh sách message client gửi lên, theo thứ tự thời gian.

    Returns:
        Câu hỏi hiện tại và tối đa ``HISTORY_MAX_TURNS`` cặp (user, assistant).

    Raises:
        InvalidConversationError: Message cuối không phải ``user`` hoặc câu hỏi
            dài quá ``MAX_QUERY_CHARS``.
    """
    # Lọc theo role thực tế (không dựa vào kiểu) vì dữ liệu đến từ client.
    kept = [
        ChatMessage(role=m.role, content=m.content.strip())
        for m in messages
        if m.role in ("user", "assistant") and m.content.strip()
    ]
    if not kept or kept[-1].role != "user":
        raise InvalidConversationError("Message cuối phải là của người dùng.")
    query = kept[-1].content
    if len(query) > MAX_QUERY_CHARS:
        raise InvalidConversationError(f"Câu hỏi dài quá {MAX_QUERY_CHARS} ký tự.")
    turns = _pair_turns(kept[:-1])[-HISTORY_MAX_TURNS:]
    history = [message for turn in turns for message in turn]
    return HistoryWindow(query=query, history=history)


def _pair_turns(messages: list[ChatMessage]) -> list[list[ChatMessage]]:
    """Ghép (user, assistant); user không có assistant vẫn giữ một mình."""
    turns: list[list[ChatMessage]] = []
    for message in messages:
        if message.role == "user":
            turns.append([_clean_user(message)])
        elif turns and len(turns[-1]) == 1:
            turns[-1].append(_clean_assistant(message))
        # assistant mồ côi (đầu hội thoại hoặc liên tiếp) bị bỏ.
    return turns


def _clean_user(message: ChatMessage) -> ChatMessage:
    return ChatMessage(role="user", content=message.content[:MAX_QUERY_CHARS])


def _clean_assistant(message: ChatMessage) -> ChatMessage:
    """Cắt khối Nguồn + disclaimer, xoá ``[n]`` và giới hạn độ dài câu trả lời cũ."""
    text = message.content.split(SOURCES_FOOTER_MARKER, 1)[0]
    text = text.split(DATA_SNAPSHOT_DISCLAIMER, 1)[0]
    text = _CITATION_MARK.sub("", text).strip()
    if len(text) > HISTORY_ASSISTANT_MAX_CHARS:
        text = text[:HISTORY_ASSISTANT_MAX_CHARS].rstrip() + _ELLIPSIS
    return ChatMessage(role="assistant", content=text)
