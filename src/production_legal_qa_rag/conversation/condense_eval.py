"""Script đo condense DEV-ONLY (conversation_spec.md mục 15).

Chạy bộ ca ``condense_cases.yaml`` tuần tự, có delay, mỗi ca ``--runs`` lần:
in câu gốc, đầu ra thô, mã lý do, kết quả cuối; tuỳ chọn đo retrieval (chunk trúng)
và guardrail cho ca injection. KHÔNG gọi generation. In nội dung ra stdout là chủ
ý của script đo này (không dùng trong production, mục 12).

Dừng ngay khi gặp lỗi Groq (429...) thay vì retry mù (mục 15.6).
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import typer
import yaml

from production_legal_qa_rag.config import CondenseSettings
from production_legal_qa_rag.conversation.condenser import (
    CondenseOutcome,
    CondenseReason,
    QueryCondenser,
)
from production_legal_qa_rag.conversation.history import build_window
from production_legal_qa_rag.conversation.models import ChatMessage
from production_legal_qa_rag.generation.guardrail import InputGuardrail
from production_legal_qa_rag.retrieval.citation import (
    DOCUMENTS,
    extract_citation_numbers,
    parse_breadcrumb,
)
from production_legal_qa_rag.retrieval.pipeline import retrieve

app = typer.Typer(add_completion=False)
CASES_PATH = Path(__file__).with_name("condense_cases.yaml")
NEEDS_CONDENSE = {"pronoun", "inherit"}


def _chunk_label(chunk: Any) -> tuple[str | None, int | None, int | None]:
    document = DOCUMENTS.get(chunk.source_document)
    ref = parse_breadcrumb(chunk.breadcrumb)
    return (document.key if document else None, ref.dieu, ref.khoan)


def _is_hit(chunks: list[Any], expected: list[dict[str, Any]]) -> bool:
    labels = [_chunk_label(c) for c in chunks]
    for want in expected:
        for doc, dieu, khoan in labels:
            if (
                doc == want["doc"]
                and dieu == want["dieu"]
                and ("khoan" not in want or khoan == want["khoan"])
            ):
                return True
    return False


def _shows_injection_rejected(verdict: str) -> bool:
    return verdict == "injection"


async def _run(
    runs: int,
    delay: float,
    only: list[str],
    with_retrieval: bool,
    with_guardrail: bool,
    out: Path | None,
    max_tokens_note: str,
) -> None:
    cases: list[dict[str, Any]] = yaml.safe_load(CASES_PATH.read_text("utf-8"))["cases"]
    if only:
        cases = [c for c in cases if c["id"] in only]
    condenser = QueryCondenser(CondenseSettings(max_retries=0))  # type: ignore[call-arg]
    guardrail = InputGuardrail() if with_guardrail else None
    retrieval_cache: dict[str, list[Any]] = {}
    results: list[dict[str, Any]] = []
    calls = prompt_tok = completion_tok = 0
    aborted = False

    for case in cases:
        messages = [ChatMessage(**m) for m in case["messages"]]
        window = build_window(messages)
        typer.echo(f"\n=== {case['id']} [{case['kind']}] expect={case['expect']}")
        typer.echo(f"  câu gốc: {window.query}")
        record: dict[str, Any] = {"case": case, "runs": []}
        if case["kind"] == "injection":
            if guardrail is not None:
                verdict = await guardrail.check_input(
                    window.query, window.recent_user_turns
                )
                record["guardrail"] = verdict.verdict
                typer.echo(f"  guardrail: {verdict.verdict} ({verdict.reason})")
                await asyncio.sleep(delay)
            results.append(record)
            continue
        for run_index in range(runs):
            outcome: CondenseOutcome = await condenser.condense_detailed(
                window.query, window.history
            )
            calls += 1
            prompt_tok += outcome.prompt_tokens or 0
            completion_tok += outcome.completion_tokens or 0
            invented = set(extract_citation_numbers(outcome.text)) - set(
                extract_citation_numbers(
                    "\n".join([window.query, *(m.content for m in window.history)])
                )
            )
            run_record = {
                "raw": outcome.raw_output,
                "reason": outcome.reason.value,
                "finish": outcome.finish_reason,
                "completion_tokens": outcome.completion_tokens,
                "reasoning_tokens": outcome.reasoning_tokens,
                "prompt_tokens": outcome.prompt_tokens,
                "final": outcome.text,
                "verbatim": outcome.text == window.query,
                "invented": sorted(invented),
            }
            record["runs"].append(run_record)
            typer.echo(
                f"  [run {run_index + 1}] reason={outcome.reason.value} "
                f"finish={outcome.finish_reason} "
                f"tok(p/c/r)={outcome.prompt_tokens}/{outcome.completion_tokens}/"
                f"{outcome.reasoning_tokens}\n"
                f"     raw:   {outcome.raw_output!r}\n"
                f"     final: {outcome.text}"
            )
            if outcome.reason is CondenseReason.GROQ_ERROR:
                typer.echo("!!! Groq lỗi/429: DỪNG, không retry mù.")
                aborted = True
                break
            await asyncio.sleep(delay)
        if aborted:
            results.append(record)
            break
        if with_retrieval and case.get("expected_chunks") and record["runs"]:
            standalone = record["runs"][0]["final"]
            if standalone not in retrieval_cache:
                retrieval_cache[standalone] = await retrieve(standalone)
            chunks = retrieval_cache[standalone]
            record["retrieval_hit"] = _is_hit(chunks, case["expected_chunks"])
            record["retrieved"] = [[str(x) for x in _chunk_label(c)] for c in chunks]
            typer.echo(
                f"  retrieval hit={record['retrieval_hit']} "
                f"top={[':'.join(x) for x in record['retrieved']]}"
            )
        results.append(record)

    _summarize(results, runs)
    typer.echo(
        f"\nGroq condense: {calls} call, prompt={prompt_tok} completion={completion_tok} "
        f"token ({max_tokens_note})"
    )
    if out is not None:
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1), "utf-8")
    if aborted:
        sys.exit(2)


def _summarize(results: list[dict[str, Any]], runs: int) -> None:
    typer.echo(f"\n{'#' * 60}\nTỔNG KẾT")
    counted = [
        r
        for r in results
        if r["case"].get("counts", True) and r["case"]["kind"] in NEEDS_CONDENSE
    ]
    reasons: Counter[str] = Counter()
    per_run_valid = [[0, 0] for _ in range(runs)]
    for r in counted:
        for i, run in enumerate(r["runs"]):
            valid = run["reason"] == "ok"
            per_run_valid[i][0] += valid
            per_run_valid[i][1] += 1
            if not valid:
                reasons[run["reason"]] += 1
    total = sum(v[1] for v in per_run_valid)
    ok = sum(v[0] for v in per_run_valid)
    typer.echo(f"Hợp lệ (ca cần condense, mọi lần): {ok}/{total}")
    typer.echo(
        "Ổn định từng lần: "
        + ", ".join(f"{v[0]}/{v[1]}" for v in per_run_valid if v[1])
    )
    typer.echo(f"Lý do loại: {dict(reasons)}")

    hits = [
        r
        for r in counted
        if r["case"]["kind"] in NEEDS_CONDENSE and "retrieval_hit" in r
    ]
    typer.echo(
        f"Trúng chunk (đại từ+kế thừa, NHÁP): "
        f"{sum(r['retrieval_hit'] for r in hits)}/{len(hits)}"
    )
    switch = [r for r in results if r["case"]["kind"] == "switch"]
    verbatim_ok = sum(all(x["verbatim"] for x in r["runs"]) for r in switch)
    typer.echo(f"Đổi chủ đề trả nguyên văn (mọi lần): {verbatim_ok}/{len(switch)}")
    inj = [r for r in results if r["case"]["kind"] == "injection"]
    blocked = sum(r.get("guardrail") == "injection" for r in inj)
    typer.echo(f"Injection bị guardrail chặn: {blocked}/{len(inj)}")
    invented = sum(len(x["invented"]) for r in results for x in r["runs"])
    typer.echo(f"Số Điều bịa lọt qua: {invented}")
    keep_fail = [
        r["case"]["id"]
        for r in counted
        if r["case"].get("keep_numbers")
        and r["runs"]
        and not all(
            set(r["case"]["keep_numbers"]) <= set(extract_citation_numbers(x["final"]))
            for x in r["runs"]
        )
    ]
    typer.echo(f"Ca kế thừa Điều không giữ số Điều: {keep_fail}")
    for r in counted:
        if r["case"].get("regression"):
            typer.echo(
                f"Ca hồi quy {r['case']['id']}: "
                f"{[x['reason'] for x in r['runs']]} hit={r.get('retrieval_hit')}"
            )


@app.command()
def main(
    runs: int = typer.Option(3, help="Số lần chạy mỗi ca (độ ổn định)."),
    delay: float = typer.Option(5.0, help="Giây nghỉ giữa các call Groq."),
    only: list[str] = typer.Option([], help="Chỉ chạy các id ca này."),
    retrieval: bool = typer.Option(True, help="Đo retrieval (gọi HyDE + Pinecone)."),
    guardrail: bool = typer.Option(True, help="Đo guardrail cho ca injection."),
    out: Path | None = typer.Option(None, help="Ghi kết quả JSON (nên để ngoài repo)."),
    note: str = typer.Option("", help="Ghi chú tham số vòng đo."),
) -> None:
    asyncio.run(_run(runs, delay, only, retrieval, guardrail, out, note))


if __name__ == "__main__":
    app()
