"""Test thủ công: lấy 3 chunk có sẵn trong data/chunks, gọi thử reranker API
đã host trên LightningAI để xác nhận end-to-end hoạt động đúng.
"""

import json
import urllib.request
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class RerankerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    endpoint_url: str = Field(validation_alias="RERANKER_ENDPOINT_URL")
    api_key: str = Field(validation_alias="RERANKER_API_KEY")


def main() -> None:
    settings = RerankerSettings()

    chunks_file = REPO_ROOT / "data/chunks/Văn bản hợp nhất bộ luật lao động.json"
    all_chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    # 1 chunk liên quan trực tiếp (Điều 113 - Nghỉ hằng năm) + 2 chunk không liên quan
    chunks = [all_chunks[344], all_chunks[12], all_chunks[14]]

    query = "Người lao động làm đủ 12 tháng được nghỉ phép năm bao nhiêu ngày?"
    passages = [c["content"] for c in chunks]

    payload = json.dumps({"query": query, "passages": passages}).encode("utf-8")
    request = urllib.request.Request(
        settings.endpoint_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": settings.api_key,
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read())

    ranked = sorted(
        zip(chunks, result["scores"]), key=lambda pair: pair[1], reverse=True
    )
    print(f"Query: {query}\n")
    for chunk, score in ranked:
        print(f"[{score:.4f}] {chunk['breadcrumb']}")
        print(f"         {chunk['content']}...\n")


if __name__ == "__main__":
    main()
