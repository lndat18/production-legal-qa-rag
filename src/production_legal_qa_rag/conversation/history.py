"""Cửa sổ history từ ``messages[]`` của client (conversation_spec.md mục 4).

``messages[]`` là dữ liệu không tin cậy: hàm ở đây chỉ lọc, cắt và làm sạch để
dùng làm *dữ liệu* trong prompt condense/guardrail, không bao giờ làm chỉ dẫn.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel

from production_legal_qa_rag.conversation.models import ChatMessage
from production_legal_qa_rag.retrieval.citation import (
    extract_citation_khoans,
    extract_citation_numbers,
)

# Khối "Nguồn" do `api/` nối vào câu trả lời; hằng số ở đây để `api/` import.
SOURCES_FOOTER_MARKER: Final = "\n\n---\n**Nguồn**\n"
MAX_QUERY_CHARS: Final = 1000
HISTORY_MAX_TURNS: Final = 3
HISTORY_ASSISTANT_MAX_CHARS: Final = 600
GUARDRAIL_CONTEXT_TURNS: Final = 2

_CITATION_MARK = re.compile(r"\[\d+\]")
_ELLIPSIS: Final = "…"

# Cụm tham chiếu ngược tới nội dung đã nói ("ở trên", "vừa rồi/nói/nêu/trả lời/trích
# dẫn", "đã nói", "trước đó" (khi đi kèm nói/nêu/trả lời), "lúc nãy"/"khi nãy"/"hồi nãy"
# — đồng nghĩa khẩu ngữ của "vừa rồi"). "vừa" và "trước đó" PHẢI đi kèm hậu tố/ngữ cảnh
# tham chiếu rõ nghĩa — không để bare, vì cả hai còn có nghĩa khác không liên quan tới
# hội thoại cũ: "vừa" ("vừa mới ban hành", "vừa sinh con", "vừa đủ"...) và "trước đó"
# (mốc thời gian trong lịch sử pháp luật, ví dụ "mức lương tối thiểu vùng trước đó, giai
# đoạn 2015-2020" — không liên quan gì tới lượt hội thoại trước). Nếu để bare sẽ chặn
# oan câu hỏi pháp luật mới bất kỳ có chứa các từ này (phát hiện khi review PR #40,
# false-positive thật: "Tóm tắt giúp tôi các quy định vừa ban hành về nghỉ phép năm.",
# "Tóm tắt mức lương tối thiểu vùng trước đó, giai đoạn 2015-2020.").
_BACK_REFERENCE = (
    r"(?:ở\s*trên"
    r"|vừa\s*(?:rồi|nói|nêu|trả\s*lời|trích\s*dẫn)"
    r"|đã\s*nói"
    r"|(?:nói|nêu|trả\s*lời).{0,15}trước\s*đó"
    r"|trước\s*đó.{0,15}(?:nói|nêu|trả\s*lời)"
    r"|lúc\s*nãy|khi\s*nãy|hồi\s*nãy)"
)
# Mẫu regex nhận diện meta-request về lịch sử hội thoại (conversation_spec.md
# mục 18.2.3): "tóm tắt"/"nhắc lại" + cụm tham chiếu ngược, "(ý|điểm|phần) ...
# (bạn|vừa|đã) ... (nói|nêu|trả lời)", hoặc gọi thẳng "bạn" (địa chỉ hệ thống)
# + "vừa"/"đã" + "nói/nêu/trả lời" (vd. "Bạn vừa nói gì vậy?" — không cần tiền
# tố "tóm tắt"/"nhắc lại"/"ý/điểm/phần" mới là meta-request). Khoảng cách
# .{0,40}/.{0,20} đủ rộng cho câu tiếng Việt tự nhiên, không quá rộng để tránh
# khớp nhầm đoạn dài.
_META_HISTORY_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(rf"tóm\s*tắt.{{0,40}}{_BACK_REFERENCE}", re.IGNORECASE),
    re.compile(rf"nhắc\s*lại.{{0,40}}{_BACK_REFERENCE}", re.IGNORECASE),
    re.compile(
        r"(?:ý|điểm|phần).{0,20}(?:bạn|vừa|đã).{0,20}(?:nói|nêu|trả\s*lời)",
        re.IGNORECASE,
    ),
    re.compile(r"bạn.{0,20}(?:vừa|đã).{0,20}(?:nói|nêu|trả\s*lời)", re.IGNORECASE),
)


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


def is_meta_history_request(window: HistoryWindow) -> bool:
    """``True`` nếu ``window.query`` là meta-request về lịch sử hội thoại.

    Lớp phòng thủ code-based (regex, không LLM), chạy trước guardrail/condense
    (conversation_spec.md mục 18.2.3): ``generate()`` không nhận history nên
    không thể trả lời loại câu hỏi này (ví dụ "Tóm tắt lại các câu trả lời ở
    trên cho tôi."). Chỉ áp dụng khi có history (lượt đầu luôn ``False``) và
    câu hỏi không chứa số Điều/Khoản nào (tránh chặn oan "Nhắc lại giúp tôi
    Điều 35 nói gì" — có số Điều thì là tra cứu thật, không phải meta-request).

    Args:
        window: Cửa sổ history đã dựng bởi ``build_window``.
    """
    if not window.has_history:
        return False
    query = unicodedata.normalize("NFC", window.query)
    if not any(pattern.search(query) for pattern in _META_HISTORY_PATTERNS):
        return False
    return not extract_citation_numbers(query) and not extract_citation_khoans(query)


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
    """Cắt khối Nguồn, xoá ``[n]`` và giới hạn độ dài câu trả lời cũ."""
    text = message.content.split(SOURCES_FOOTER_MARKER, 1)[0]
    text = _CITATION_MARK.sub("", text).strip()
    if len(text) > HISTORY_ASSISTANT_MAX_CHARS:
        text = text[:HISTORY_ASSISTANT_MAX_CHARS].rstrip() + _ELLIPSIS
    return ChatMessage(role="assistant", content=text)
