"""Chạy thử `retrieve()` end-to-end với dịch vụ thật (Groq, HF, Pinecone, reranker).

Cách dùng (từ root repo):

    uv run python src/production_legal_qa_rag/retrieval/test.py
    uv run python src/production_legal_qa_rag/retrieval/test.py "Câu hỏi của bạn"
    uv run python src/production_legal_qa_rag/retrieval/test.py --no-mmr
    uv run python src/production_legal_qa_rag/retrieval/test.py --compare

Cần: `.env` đầy đủ, sparse index + `data/bm25/bm25_params.json` đã build, và
reranker server đang chạy (runbook mục 9.1 của retrieval_spec.md). Nếu
reranker không lên, kết quả vẫn trả về nhưng `rerank_score=None` (fallback).
"""

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path

from production_legal_qa_rag.retrieval import (
    RetrievalError,
    RetrievalPipeline,
    RetrievedChunk,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_QUESTIONS = [
    "Người lao động làm đủ 12 tháng được nghỉ phép năm bao nhiêu ngày?",
    "Điều 3 khoản 1 quy định gì?",
    "Người sử dụng lao động phải báo trước bao lâu khi đơn phương chấm dứt hợp đồng lao động?",
]

SNIPPET_CHARS = 300


def _print_results(chunks: list[RetrievedChunk], elapsed: float) -> None:
    print(f"  -> {len(chunks)} chunk, {elapsed:.1f}s")
    for rank, chunk in enumerate(chunks, start=1):
        score = "None" if chunk.rerank_score is None else f"{chunk.rerank_score:.4f}"
        snippet = chunk.content.replace("\n", " ")[:SNIPPET_CHARS]
        print(f"  {rank}. [{score}] {chunk.breadcrumb}")
        print(f"     ({chunk.source_document}) {snippet}...")
    if any(chunk.rerank_score is None for chunk in chunks):
        print(
            "  ! rerank_score=None: reranker lỗi, đang dùng fallback (xem log WARNING)."
        )


async def _run(questions: list[str], modes: list[bool]) -> int:
    pipeline = RetrievalPipeline(
        bm25_params_path=REPO_ROOT / "data/bm25/bm25_params.json"
    )
    failures = 0
    for question in questions:
        print(f"\n=== {question}")
        for use_mmr in modes:
            print(f"- use_mmr={use_mmr}")
            start = time.perf_counter()
            try:
                chunks = await pipeline.retrieve(question, use_mmr=use_mmr)
            except RetrievalError as exc:
                failures += 1
                print(f"  ! RetrievalError: {exc}")
                continue
            _print_results(chunks, time.perf_counter() - start)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", nargs="*", help="Câu hỏi (mặc định: 3 câu mẫu)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--no-mmr", action="store_true", help="Tắt MMR")
    group.add_argument("--compare", action="store_true", help="Chạy cả MMR bật và tắt")
    args = parser.parse_args()

    modes = [True, False] if args.compare else [not args.no_mmr]
    questions = args.questions or DEFAULT_QUESTIONS

    # Settings đọc `.env` theo thư mục hiện tại.
    os.chdir(REPO_ROOT)
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )

    failures = asyncio.run(_run(questions, modes))
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
