"""Tests that protect the `GenerationEvent` -> OpenAI conversion contract.

api_spec.md mục 6: thứ tự status(reasoning_content) -> token(content) ->
citations(khối "Nguồn") -> warning -> disclaimer -> done(finish_reason[,usage])
-> ``[DONE]``; non-stream gom cùng nội dung vào một JSON.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from production_legal_qa_rag.api.openai_format import build_completion, sse_stream
from production_legal_qa_rag.conversation.history import (
    DATA_SNAPSHOT_DISCLAIMER,
    SOURCES_FOOTER_MARKER,
)
from production_legal_qa_rag.generation.models import (
    Citation,
    CitationsEvent,
    DoneEvent,
    ErrorEvent,
    GenerationEvent,
    RefusalEvent,
    StatusEvent,
    TokenEvent,
    Usage,
    WarningEvent,
)

_MODEL = "legal-qa"
_CITATIONS = [
    Citation(
        n=1,
        chunk_id="c1",
        source_document="Luật Doanh nghiệp 2020",
        breadcrumb="Điều 4",
    ),
]
_SOURCES_BLOCK = f"{SOURCES_FOOTER_MARKER}[1] Điều 4 — Luật Doanh nghiệp 2020"


async def _events(*events: GenerationEvent) -> AsyncIterator[GenerationEvent]:
    for event in events:
        yield event


async def _events_with_delay(
    *events: GenerationEvent, delay_before_index: int, delay_seconds: float
) -> AsyncIterator[GenerationEvent]:
    """Yield events but pause before a given index, to exercise keep-alive."""
    for index, event in enumerate(events):
        if index == delay_before_index:
            await asyncio.sleep(delay_seconds)
        yield event


async def _collect_sse(
    events: AsyncIterator[GenerationEvent], **kwargs: object
) -> list[bytes]:
    return [chunk async for chunk in sse_stream(events, **kwargs)]


def _decode_data_lines(raw_chunks: list[bytes]) -> list[dict | str]:
    """Parse each SSE frame into either a dict (``data:`` JSON) or ``"KEEPALIVE"``."""
    parsed: list[dict | str] = []
    for raw in raw_chunks:
        text = raw.decode()
        if text == ": keep-alive\n\n":
            parsed.append("KEEPALIVE")
        elif text == "data: [DONE]\n\n":
            parsed.append("DONE")
        else:
            assert text.startswith("data: ")
            parsed.append(json.loads(text[len("data: ") : -2]))
    return parsed


# --- build_completion (stream=false) -------------------------------------


def test_build_completion_joins_tokens_sources_and_disclaimer_in_order() -> None:
    """Answered turn: tokens, then Nguồn block, then the snapshot disclaimer."""
    events = _events(
        StatusEvent(stage="guardrail"),
        TokenEvent(text="Xin chào, "),
        TokenEvent(text="đây là câu trả lời."),
        CitationsEvent(citations=_CITATIONS),
        DoneEvent(usage=Usage(prompt_tokens=10, completion_tokens=5)),
    )

    response = asyncio.run(build_completion(events, model=_MODEL, include_usage=True))

    expected_content = (
        "Xin chào, đây là câu trả lời." + _SOURCES_BLOCK + DATA_SNAPSHOT_DISCLAIMER
    )
    assert response.choices[0].message.content == expected_content
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].finish_reason == "stop"
    assert response.model == _MODEL
    assert response.usage is not None
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 15


def test_build_completion_without_citations_has_no_disclaimer() -> None:
    """A refusal never produced a Nguồn block, so the disclaimer must not appear."""
    events = _events(
        RefusalEvent(reason="out_of_scope", message="Câu hỏi ngoài phạm vi pháp luật."),
        DoneEvent(),
    )

    response = asyncio.run(build_completion(events, model=_MODEL, include_usage=False))

    assert response.choices[0].message.content == "Câu hỏi ngoài phạm vi pháp luật."
    assert DATA_SNAPSHOT_DISCLAIMER not in response.choices[0].message.content
    assert response.choices[0].finish_reason == "stop"
    assert response.usage is None


def test_build_completion_truncated_warning_sets_finish_reason_length() -> None:
    """``warning(code=truncated)`` must flip ``finish_reason`` to ``length``."""
    events = _events(
        TokenEvent(text="Một phần câu trả lời"),
        WarningEvent(code="truncated", message="Câu trả lời bị cắt do quá dài."),
        DoneEvent(),
    )

    response = asyncio.run(build_completion(events, model=_MODEL, include_usage=False))

    assert response.choices[0].finish_reason == "length"
    assert "Câu trả lời bị cắt do quá dài." in response.choices[0].message.content


def test_build_completion_rate_limited_error_mentions_retry_seconds() -> None:
    """``error(code=rate_limited)`` appends the retry hint per mục 6."""
    events = _events(
        TokenEvent(text="Đang trả lời"),
        ErrorEvent(
            code="rate_limited",
            message="Hết hạn mức.",
            retry_after_seconds=30.0,
        ),
        DoneEvent(),
    )

    response = asyncio.run(build_completion(events, model=_MODEL, include_usage=False))

    content = response.choices[0].message.content
    assert "Hết hạn mức." in content
    assert "30 giây" in content


def test_build_completion_include_usage_false_omits_usage_even_if_present() -> None:
    """``stream_options.include_usage=false`` must hide usage in the response."""
    events = _events(
        TokenEvent(text="ok"),
        DoneEvent(usage=Usage(prompt_tokens=1, completion_tokens=1)),
    )

    response = asyncio.run(build_completion(events, model=_MODEL, include_usage=False))

    assert response.usage is None


# --- sse_stream (stream=true) ---------------------------------------------


def test_sse_stream_orders_role_content_sources_disclaimer_finish_then_done() -> None:
    """First chunk carries the role, the last two frames are finish + [DONE]."""
    events = _events(
        StatusEvent(stage="guardrail"),
        StatusEvent(stage="retrieval"),
        TokenEvent(text="Xin chào"),
        CitationsEvent(citations=_CITATIONS),
        DoneEvent(),
    )

    raw = asyncio.run(
        _collect_sse(events, model=_MODEL, include_usage=False, keepalive_seconds=15)
    )
    frames = _decode_data_lines(raw)

    first_delta = frames[0]["choices"][0]["delta"]
    second_delta = frames[1]["choices"][0]["delta"]
    assert first_delta["role"] == "assistant"
    assert first_delta["reasoning_content"] == "Đang kiểm tra câu hỏi…"
    assert second_delta.get("role") is None
    assert second_delta["reasoning_content"] == "Đang tra cứu văn bản pháp luật…"
    assert frames[2]["choices"][0]["delta"]["content"] == "Xin chào"
    assert frames[3]["choices"][0]["delta"]["content"] == _SOURCES_BLOCK
    assert frames[4]["choices"][0]["delta"]["content"] == DATA_SNAPSHOT_DISCLAIMER
    assert frames[5]["choices"][0]["finish_reason"] == "stop"
    assert frames[-1] == "DONE"


def test_sse_stream_include_usage_appends_usage_chunk_before_done() -> None:
    """The optional usage chunk must sit after finish_reason but before [DONE]."""
    events = _events(
        TokenEvent(text="ok"),
        DoneEvent(usage=Usage(prompt_tokens=2, completion_tokens=3)),
    )

    raw = asyncio.run(
        _collect_sse(events, model=_MODEL, include_usage=True, keepalive_seconds=15)
    )
    frames = _decode_data_lines(raw)

    assert frames[-1] == "DONE"
    usage_frame = frames[-2]
    assert usage_frame["choices"] == []
    assert usage_frame["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    finish_frame = frames[-3]
    assert finish_frame["choices"][0]["finish_reason"] == "stop"


def test_sse_stream_no_usage_chunk_when_include_usage_false() -> None:
    """Without ``include_usage``, the finish chunk is immediately followed by [DONE]."""
    events = _events(
        TokenEvent(text="ok"),
        DoneEvent(usage=Usage(prompt_tokens=2, completion_tokens=3)),
    )

    raw = asyncio.run(
        _collect_sse(events, model=_MODEL, include_usage=False, keepalive_seconds=15)
    )
    frames = _decode_data_lines(raw)

    assert frames[-1] == "DONE"
    assert frames[-2]["choices"][0]["finish_reason"] == "stop"
    assert all("usage" not in f for f in frames if isinstance(f, dict))


def test_sse_stream_sends_keepalive_without_dropping_the_pending_event() -> None:
    """A slow event must trigger keep-alive comments but still arrive intact.

    This protects the ``asyncio.wait`` (not ``wait_for``) design documented in
    ``openai_format.sse_stream``: using ``wait_for`` would cancel the pending
    ``__anext__()`` and silently drop the event that was about to arrive.
    """
    events = _events_with_delay(
        TokenEvent(text="đầu tiên"),
        TokenEvent(text="sau khi chờ"),
        DoneEvent(),
        delay_before_index=1,
        delay_seconds=0.12,
    )

    raw = asyncio.run(
        _collect_sse(events, model=_MODEL, include_usage=False, keepalive_seconds=0.03)
    )
    frames = _decode_data_lines(raw)

    assert "KEEPALIVE" in frames
    contents = [
        frame["choices"][0]["delta"].get("content")
        for frame in frames
        if isinstance(frame, dict) and frame["choices"]
    ]
    assert "đầu tiên" in contents
    assert "sau khi chờ" in contents
    assert frames[-1] == "DONE"


def test_sse_stream_truncated_warning_sets_finish_reason_length() -> None:
    """Stream mode must mirror the non-stream ``finish_reason=length`` rule."""
    events = _events(
        TokenEvent(text="một phần"),
        WarningEvent(code="truncated", message="quá dài"),
        DoneEvent(),
    )

    raw = asyncio.run(
        _collect_sse(events, model=_MODEL, include_usage=False, keepalive_seconds=15)
    )
    frames = _decode_data_lines(raw)

    def _has_finish_reason(frame: object) -> bool:
        return (
            isinstance(frame, dict)
            and bool(frame["choices"])
            and frame["choices"][0].get("finish_reason") is not None
        )

    finish_frames = [frame for frame in frames if _has_finish_reason(frame)]
    assert finish_frames[-1]["choices"][0]["finish_reason"] == "length"
