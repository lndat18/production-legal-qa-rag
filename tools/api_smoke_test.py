"""Kiểm tra smoke test end-to-end cho HTTP API đang chạy.

Script gọi API qua mạng thật, nên xác nhận được cả server, Redis/Postgres,
xác thực và luồng chat (SSE hoặc JSON). Nó không khởi động server hay mock bất
kỳ dependency nào.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import typer

app = typer.Typer(add_completion=False)

_DEFAULT_BASE_URL = "http://localhost:8000"
_DEFAULT_QUERY = "Thời gian thử việc tối đa theo Bộ luật Lao động là bao lâu?"


@dataclass(frozen=True)
class HttpResponse:
    """Kết quả tối giản của một HTTP request.

    Args:
        status_code: HTTP status code server trả về.
        headers: Response headers, với tên đã được chuẩn hóa thành lowercase.
        body: Nội dung response đã giải mã UTF-8.
    """

    status_code: int
    headers: dict[str, str]
    body: str


def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout_seconds: float,
) -> HttpResponse:
    """Gửi một HTTP request và giữ lại response kể cả khi server trả lỗi.

    Args:
        method: HTTP method cần gọi.
        url: URL đầy đủ của endpoint.
        headers: Header bổ sung cho request.
        payload: JSON body, nếu endpoint cần body.
        timeout_seconds: Thời gian chờ tối đa cho toàn bộ response.

    Returns:
        Response đã được đọc hoàn chỉnh.

    Raises:
        typer.Exit: Khi không thể kết nối hoặc request hết thời gian chờ.
    """
    data = json.dumps(payload, ensure_ascii=False).encode() if payload else None
    request_headers = {"Accept": "application/json"}
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)
    request = Request(url, data=data, headers=request_headers, method=method)

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return HttpResponse(
                status_code=response.status,
                headers={key.lower(): value for key, value in response.headers.items()},
                body=response.read().decode("utf-8"),
            )
    except HTTPError as error:
        return HttpResponse(
            status_code=error.code,
            headers={key.lower(): value for key, value in error.headers.items()},
            body=error.read().decode("utf-8"),
        )
    except URLError as error:
        typer.secho(f"KHÔNG KẾT NỐI ĐƯỢC: {error.reason}", fg=typer.colors.RED)
        raise typer.Exit(1) from error


def _assert_status(response: HttpResponse, expected: int, endpoint: str) -> None:
    """Dừng test với body hữu ích nếu endpoint trả status ngoài mong đợi."""
    if response.status_code != expected:
        typer.secho(
            f"FAIL {endpoint}: nhận HTTP {response.status_code}, mong đợi {expected}.",
            fg=typer.colors.RED,
        )
        typer.echo(response.body)
        raise typer.Exit(1)


def _parse_json(body: str, endpoint: str) -> dict[str, Any]:
    """Parse JSON object hoặc dừng với thông tin response khi format không đúng."""
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as error:
        typer.secho(
            f"FAIL {endpoint}: response không phải JSON hợp lệ.", fg=typer.colors.RED
        )
        typer.echo(body)
        raise typer.Exit(1) from error
    if not isinstance(parsed, dict):
        typer.secho(
            f"FAIL {endpoint}: JSON response phải là object.", fg=typer.colors.RED
        )
        raise typer.Exit(1)
    return parsed


def _check_health(base_url: str, timeout_seconds: float) -> None:
    """Kiểm tra liveness và readiness của API cùng dependency nội bộ."""
    health_response = _request(
        "GET", f"{base_url}/healthz", timeout_seconds=timeout_seconds
    )
    _assert_status(health_response, 200, "/healthz")
    if _parse_json(health_response.body, "/healthz").get("status") != "ok":
        typer.secho("FAIL /healthz: body không có status=ok.", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho("PASS /healthz", fg=typer.colors.GREEN)

    ready_response = _request(
        "GET", f"{base_url}/readyz", timeout_seconds=timeout_seconds
    )
    _assert_status(ready_response, 200, "/readyz")
    if _parse_json(ready_response.body, "/readyz").get("status") != "ready":
        typer.secho("FAIL /readyz: body không có status=ready.", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho("PASS /readyz (Redis và Postgres sẵn sàng)", fg=typer.colors.GREEN)


def _check_models(
    base_url: str, authorization: dict[str, str], timeout_seconds: float
) -> None:
    """Kiểm tra Bearer authentication và model OpenAI-compatible."""
    response = _request(
        "GET",
        f"{base_url}/v1/models",
        headers=authorization,
        timeout_seconds=timeout_seconds,
    )
    _assert_status(response, 200, "/v1/models")
    models = _parse_json(response.body, "/v1/models").get("data")
    if not isinstance(models, list) or not any(
        isinstance(model, dict) and model.get("id") == "legal-qa" for model in models
    ):
        typer.secho(
            "FAIL /v1/models: không tìm thấy model legal-qa.", fg=typer.colors.RED
        )
        raise typer.Exit(1)
    typer.secho("PASS /v1/models (Bearer key hợp lệ)", fg=typer.colors.GREEN)


def _check_streaming_chat(
    base_url: str,
    authorization: dict[str, str],
    query: str,
    timeout_seconds: float,
) -> None:
    """Gọi chat SSE và kiểm tra kết thúc bằng ``[DONE]`` có nội dung trả lời."""
    response = _request(
        "POST",
        f"{base_url}/v1/chat/completions",
        headers=authorization,
        payload={
            "model": "legal-qa",
            "messages": [{"role": "user", "content": query}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        timeout_seconds=timeout_seconds,
    )
    _assert_status(response, 200, "/v1/chat/completions (SSE)")
    if "text/event-stream" not in response.headers.get("content-type", ""):
        typer.secho(
            "FAIL chat SSE: Content-Type không phải text/event-stream.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    content_parts: list[str] = []
    received_done = False
    for line in response.body.splitlines():
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            received_done = True
            continue
        event = _parse_json(data, "chat SSE")
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if isinstance(choice, dict):
                delta = choice.get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    content_parts.append(delta["content"])

    if not received_done or not "".join(content_parts).strip():
        typer.secho(
            "FAIL chat SSE: thiếu [DONE] hoặc không nhận được nội dung trả lời.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    typer.secho(
        f"PASS /v1/chat/completions SSE ({len(''.join(content_parts))} ký tự)",
        fg=typer.colors.GREEN,
    )


def _check_json_chat(
    base_url: str,
    authorization: dict[str, str],
    query: str,
    timeout_seconds: float,
) -> None:
    """Gọi chat non-stream và kiểm tra OpenAI completion có text trả lời."""
    response = _request(
        "POST",
        f"{base_url}/v1/chat/completions",
        headers=authorization,
        payload={
            "model": "legal-qa",
            "messages": [{"role": "user", "content": query}],
            "stream": False,
        },
        timeout_seconds=timeout_seconds,
    )
    _assert_status(response, 200, "/v1/chat/completions (JSON)")
    choices = _parse_json(response.body, "chat JSON").get("choices")
    if not isinstance(choices, list) or not choices:
        typer.secho("FAIL chat JSON: không có choices.", fg=typer.colors.RED)
        raise typer.Exit(1)
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        typer.secho(
            "FAIL chat JSON: choices[0].message.content đang trống.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)
    typer.secho(
        f"PASS /v1/chat/completions JSON ({len(content)} ký tự)",
        fg=typer.colors.GREEN,
    )


@app.command()
def main(
    base_url: str = typer.Option(
        _DEFAULT_BASE_URL,
        help="URL gốc của API, không kèm /v1 (ví dụ http://localhost:8000).",
    ),
    api_key: str | None = typer.Option(
        None, envvar="CHATBOT_API_KEY", help="Bearer key; mặc định đọc CHATBOT_API_KEY."
    ),
    query: str = typer.Option(
        _DEFAULT_QUERY, help="Câu hỏi thật dùng để thử luồng chat."
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Thử SSE (mặc định) hoặc OpenAI JSON completion.",
    ),
    timeout_seconds: float = typer.Option(
        120.0, min=1.0, help="Timeout mỗi request, gồm thời gian sinh câu trả lời."
    ),
) -> None:
    """Chạy kiểm tra end-to-end cho một API đã được khởi động.

    Ví dụ:
        uv run python tools/api_smoke_test.py --base-url http://localhost:8000
    """
    resolved_key = api_key or os.environ.get("CHATBOT_API_KEY")
    if not resolved_key:
        typer.secho(
            "Thiếu CHATBOT_API_KEY. Hãy đặt biến môi trường hoặc truyền --api-key.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(2)

    normalized_base_url = base_url.rstrip("/")
    authorization = {"Authorization": f"Bearer {resolved_key}"}
    started_at = time.perf_counter()

    typer.echo(f"Đang test API: {normalized_base_url}")
    _check_health(normalized_base_url, timeout_seconds)
    _check_models(normalized_base_url, authorization, timeout_seconds)
    if stream:
        _check_streaming_chat(
            normalized_base_url, authorization, query, timeout_seconds
        )
    else:
        _check_json_chat(normalized_base_url, authorization, query, timeout_seconds)
    typer.secho(
        f"\nAPI hoạt động ổn: hoàn tất trong {time.perf_counter() - started_at:.2f}s.",
        fg=typer.colors.GREEN,
        bold=True,
    )


if __name__ == "__main__":
    app()
