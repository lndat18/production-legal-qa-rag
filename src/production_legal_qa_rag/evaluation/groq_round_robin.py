"""Round-robin 6 `ChatOpenAI` (Groq) độc lập tài khoản cho `generator_llm`.

Pattern MỚI trong repo (evaluation_spec.md mục 3.1), khác hẳn cách dùng nhiều
key Groq hiện có ở `GenerationSettings`/`JudgeSettings`/`formatting/llm_client.py`
(tách ngân sách theo bước cố định, mỗi bước luôn dùng đúng 1 key). Ở đây là
luân phiên nhiều key cho CÙNG một luồng gọi để rải tải — không tái dùng được
`_SlidingWindowRateLimiter`/`convert_chunks_concurrently` của
`formatting/llm_client.py` (thiết kế cho 2 worker thread xử lý song song một
hàng đợi job, không phải round-robin tuần tự).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from openai import RateLimitError
from pydantic import PrivateAttr


class GroqRoundRobinChatModel(BaseChatModel):
    """Proxy luân phiên round-robin qua N `ChatOpenAI` (Groq) độc lập tài khoản.

    Không phải rate-limiter: chỉ đổi client theo vòng lặp cố định trước mỗi
    lượt gọi thật, để rải tải đều qua các tài khoản độc lập. Chỉ cần
    implement `_generate` (sync) — `BaseChatModel._agenerate` mặc định
    fallback gọi `_generate` qua executor khi không override, nên round-robin
    vẫn đúng dù ragas gọi qua đường async.

    Không thêm rate-limiter mới: mỗi `ChatOpenAI` con giữ nguyên
    timeout/retry riêng từ `TestsetGeneratorSettings`; round-robin ở đây chỉ
    chọn client, không kiểm soát tốc độ.

    Rủi ro chấp nhận được: `itertools.cycle` không thread/async-safe tuyệt
    đối — nếu ragas gọi đồng thời, 2 lượt gọi cùng lúc có thể nhận cùng 1
    client thay vì luân phiên hoàn hảo. Chỉ ảnh hưởng độ đều của việc rải
    tải, không ảnh hưởng tính đúng đắn; chấp nhận được cho một script chạy 1
    lần (evaluation_spec.md mục 3.1, mục 10).
    """

    # Phase 1 dùng đúng 6 client, mỗi client gắn 1 key cố định.
    clients: list[ChatOpenAI]

    _cycle: Iterator[int] = PrivateAttr()
    _call_counts: list[int] = PrivateAttr()

    def model_post_init(self, context: Any, /) -> None:
        if not self.clients:
            raise ValueError("GroqRoundRobinChatModel cần ít nhất 1 client.")
        self._cycle = itertools.cycle(range(len(self.clients)))
        self._call_counts = [0] * len(self.clients)

    @property
    def _llm_type(self) -> str:
        return "groq-round-robin"

    @property
    def call_counts(self) -> list[int]:
        """Số lượt gọi (kể cả lượt bị 429 rồi chuyển client) đã gửi tới từng client.

        Dùng để ghi `llm_calls` theo đơn vị vào `generation_progress.json` và để
        kiểm tra tải có rải đều qua các tài khoản (evaluation_spec.md mục 4.5, 9.3).
        """
        return list(self._call_counts)

    def _next_client(self) -> ChatOpenAI:
        index = next(self._cycle)
        self._call_counts[index] += 1
        return self.clients[index]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Gọi client kế tiếp trong vòng round-robin; bounded fallback khi 429.

        Thử tối đa `len(clients)` client cho MỘT lượt gọi (tương đương pseudo-code
        thiết kế ở evaluation_spec.md mục 3.1: 1 lần đầu + tối đa `len(clients) - 1`
        lần fallback); hết vòng vẫn lỗi thì raise nguyên lỗi cuối, không giữ vòng
        lặp vô hạn, không tự ý bỏ qua lượt gọi.
        """
        last_error: RateLimitError | None = None
        for _ in range(len(self.clients)):
            client = self._next_client()
            try:
                return client._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            except RateLimitError as error:
                last_error = error
                continue

        if last_error is None:
            # Không thể xảy ra: `model_post_init` đã chặn `clients` rỗng nên
            # vòng lặp trên chạy ít nhất 1 lần và luôn return hoặc set biến này.
            raise RuntimeError("GroqRoundRobinChatModel: vòng round-robin không chạy.")
        raise last_error
