"""Chạy thử thủ công `retrieve(query)` trên corpus/index thật (retrieval_spec.md mục 9).

Dùng để kiểm chứng thay đổi reranker (mục 6.1) — device đang dùng thật là `cuda`
hay fallback `cpu`, không silent lỗi. Cần `PINECONE_API_KEY`, `HF_TOKEN` và
index Pinecone đã nạp dữ liệu.

Bộ câu hỏi mẫu preset cùng quy ước với `tools/conversation.py` (câu viện dẫn cụ thể
+ câu paraphrase), chọn qua ``--preset``. ``--query`` nhập câu tùy ý.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum

import typer

from production_legal_qa_rag.retrieval.pipeline import FINAL_TOP_K, RetrievalPipeline

app = typer.Typer(
    help="Chạy thử retrieve(query) trên corpus/index thật để kiểm chứng reranker.",
    add_completion=False,
)

# Câu hỏi mẫu: viện dẫn cụ thể + paraphrase — đủ để kiểm tra cả citation recall
# lẫn semantic recall (retrieval_spec.md mục 9).
_PRESETS: dict[str, str] = {
    "dieu113_literal": "Khoản 1 Điều 113 Bộ luật Lao động quy định gì về nghỉ hằng năm?",
    "dieu113_para": "Người lao động được nghỉ hằng năm bao nhiêu ngày?",
    "dieu168_literal": "Điều 168 Bộ luật Lao động quy định về tham gia bảo hiểm xã hội?",
    "dieu168_para": "Người sử dụng lao động có nghĩa vụ gì với bảo hiểm xã hội của người lao động?",
    "dieu128_literal": "Điều 128 Bộ luật Lao động quy định về xử lý kỷ luật lao động?",
    "dieu128_para": "Hình thức kỷ luật nào được phép áp dụng với người lao động?",
}


class Preset(str, Enum):
    """Câu hỏi mẫu preset."""

    all = "all"
    dieu113_literal = "dieu113_literal"
    dieu113_para = "dieu113_para"
    dieu168_literal = "dieu168_literal"
    dieu168_para = "dieu168_para"
    dieu128_literal = "dieu128_literal"
    dieu128_para = "dieu128_para"


@app.command()
def main(
    query: str | None = typer.Option(
        None,
        "--query",
        "-q",
        help="Câu hỏi tùy ý. Bỏ qua --preset nếu truyền --query.",
    ),
    preset: Preset | None = typer.Option(
        None,
        "--preset",
        "-p",
        help="Chọn câu hỏi mẫu preset.",
        show_choices=True,
    ),
    use_mmr: bool | None = typer.Option(
        None,
        "--use-mmr/--no-use-mmr",
        help="Ghi đè USE_MMR mặc định của pipeline.",
    ),
    top_k: int = typer.Option(
        FINAL_TOP_K,
        "--top-k",
        "-k",
        help="Số chunk trả về tối đa.",
    ),
) -> None:
    """Chạy retrieve() và in kết quả chi tiết kèm thời gian wall-clock.

    Cần đặt PINECONE_API_KEY, HF_TOKEN và index Pinecone đã nạp dữ liệu.
    LocalReranker log device (cuda/cpu) một lần lúc load model — chạy ở log level
    mặc định là thấy ngay mà không cần API riêng.
    """
    selected_queries: list[str]
    if query:
        selected_queries = [query]
    elif preset is Preset.all:
        selected_queries = list(_PRESETS.values())
    elif preset is not None:
        selected_queries = [_PRESETS[preset.value]]
    else:
        _list_presets()
        raise typer.Abort()

    asyncio.run(_run_queries(selected_queries, use_mmr=use_mmr, top_k=top_k))


def _list_presets() -> None:
    typer.echo("Không có --query hoặc --preset. Danh sách preset khả dụng:")
    typer.echo("  --preset 'all'                            Chạy toàn bộ preset.")
    for key, text in _PRESETS.items():
        typer.echo(f"  --preset {key!r:30s}  {text}")


async def _run_queries(queries: list[str], *, use_mmr: bool | None, top_k: int) -> None:
    pipeline = RetrievalPipeline()
    for query in queries:
        await _run(pipeline, query, use_mmr=use_mmr, top_k=top_k)


async def _run(
    pipeline: RetrievalPipeline,
    query: str,
    *,
    use_mmr: bool | None,
    top_k: int,
) -> None:
    typer.echo("-" * 100)
    typer.echo(f"\nQuery: {query!r}\n")

    t0 = time.perf_counter()
    chunks = await pipeline.retrieve(query, use_mmr=use_mmr)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    typer.echo(f"Tổng thời gian retrieve(): {elapsed_ms:.0f} ms\n")

    fallback_active = any(c.rerank_score is None for c in chunks)
    if fallback_active:
        typer.echo(
            "⚠ CẢNH BÁO: rerank_score=None — fallback đã kích hoạt (reranker lỗi, "
            "xem mục 8 retrieval_spec.md). Kết quả dưới đây theo thứ tự fallback, "
            "không phải thứ tự rerank.\n"
        )

    displayed = chunks[:top_k]
    for rank, chunk in enumerate(displayed, start=1):
        score_str = (
            f"{chunk.rerank_score:.4f}" if chunk.rerank_score is not None else "None"
        )
        content_preview = chunk.content[:120].replace("\n", " ")
        if len(chunk.content) > 120:
            content_preview += " …"
        table_flag = " [TABLE]" if chunk.has_table else ""
        typer.echo(
            f"[{rank:02d}] rerank_score={score_str}{table_flag}\n"
            f"     breadcrumb : {chunk.breadcrumb}\n"
            f"     content    : {content_preview}\n"
        )


if __name__ == "__main__":
    app()
