"""Hiển thị điểm trung bình 4 metric RAGAS Phase 2 từ eval_result.xlsx bằng Seaborn.

Đọc sheet `Eval_result` bằng stdlib (không thêm dependency xlsx). Ô trống (câu bị từ
chối, chưa chấm) không tính vào thống kê; số câu có điểm và trung bình ghi trên từng ô.

Ví dụ:
    uv run python tools/visualize_eval_metrics.py
    uv run python tools/visualize_eval_metrics.py --output data/eval/phase2/metrics.png
"""

from __future__ import annotations

import re
import statistics
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore[import-untyped]
import typer

DEFAULT_INPUT_PATH = Path("data/eval/phase2/eval_result.xlsx")
SHEET_NAME = "Eval_result"
METRICS = (
    "Context Precision",
    "Context Recall",
    "Faithfulness",
    "Answer Relevancy",
)

_NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _column_index(reference: str) -> int:
    """Convert a cell reference such as `AB12` to a zero-based column index."""
    letters = re.match(r"[A-Z]+", reference)
    if letters is None:
        raise ValueError(f"Tham chiếu ô không hợp lệ: {reference}")
    index = 0
    for char in letters.group():
        index = index * 26 + ord(char) - ord("A") + 1
    return index - 1


def _sheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relations = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {rel.get("Id"): rel.get("Target", "") for rel in relations}
    for sheet in workbook.findall("m:sheets/m:sheet", _NS):
        if sheet.get("name") == sheet_name:
            target = targets[sheet.get(f"{{{_NS['r']}}}id")]
            return target.lstrip("/") if target.startswith("/") else f"xl/{target}"
    raise ValueError(f"Không tìm thấy sheet {sheet_name!r}.")


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(text.text or "" for text in item.iter(f"{{{_NS['m']}}}t"))
        for item in root.findall("m:si", _NS)
    ]


def load_scores(
    input_path: Path, metrics: Sequence[str] = METRICS
) -> dict[str, list[float]]:
    """Read numeric scores per metric column; blank cells are skipped."""
    try:
        archive = zipfile.ZipFile(input_path)
    except FileNotFoundError as error:
        raise ValueError(f"Không tìm thấy file input: {input_path}") from error
    with archive:
        strings = _shared_strings(archive)
        sheet = ElementTree.fromstring(archive.read(_sheet_path(archive, SHEET_NAME)))
    rows = sheet.findall("m:sheetData/m:row", _NS)
    if not rows:
        raise ValueError("Sheet rỗng.")

    def cells(row: ElementTree.Element) -> dict[int, tuple[str | None, str]]:
        result: dict[int, tuple[str | None, str]] = {}
        for cell in row.findall("m:c", _NS):
            value = cell.find("m:v", _NS)
            inline = cell.find("m:is", _NS)
            if value is not None and value.text is not None:
                text = value.text
            elif inline is not None:
                text = "".join(t.text or "" for t in inline.iter(f"{{{_NS['m']}}}t"))
            else:
                continue
            kind = cell.get("t")
            if kind == "s":
                text = strings[int(text)]
            result[_column_index(cell.get("r", "A"))] = (kind, text)
        return result

    header = {text: index for index, (_, text) in cells(rows[0]).items()}
    missing = [name for name in metrics if name not in header]
    if missing:
        raise ValueError(f"Thiếu cột metric: {', '.join(missing)}")

    scores: dict[str, list[float]] = {name: [] for name in metrics}
    for row in rows[1:]:
        row_cells = cells(row)
        for name in metrics:
            cell = row_cells.get(header[name])
            if cell is None or cell[0] in {"s", "str", "inlineStr", "e"}:
                continue
            scores[name].append(float(cell[1]))
    return scores


def show_chart(
    scores: Mapping[str, Sequence[float]], total: int, output: Path | None
) -> None:
    """Draw one bar per metric (x) with its mean score (y)."""
    means = {
        name: statistics.fmean(values) for name, values in scores.items() if values
    }
    if not means:
        raise ValueError("Không có điểm nào để vẽ biểu đồ.")

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(8, 5))
    names = list(means)
    sns.barplot(x=names, y=list(means.values()), hue=names, legend=False, ax=axis)
    for index, name in enumerate(names):
        axis.text(
            index,
            means[name] + 0.01,
            f"{means[name]:.3f}",
            ha="center",
            va="bottom",
        )
    axis.set_ylim(0, 1)
    axis.set_xlabel("Metric")
    axis.set_ylabel("Score")
    axis.set_title("RAGAS Phase 2")
    figure.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150)
        typer.echo(f"Đã lưu biểu đồ: {output}")
    else:
        plt.show()


@app.command()
def main(
    input_path: Annotated[
        Path, typer.Option("--input", help="File eval_result.xlsx.")
    ] = DEFAULT_INPUT_PATH,
    output: Annotated[
        Path | None, typer.Option(help="Lưu PNG thay vì mở cửa sổ.")
    ] = None,
) -> None:
    """In thống kê và vẽ bar chart điểm trung bình của 4 metric."""
    try:
        scores = load_scores(input_path)
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    total = max(len(values) for values in scores.values())
    for name, values in scores.items():
        if values:
            typer.echo(
                f"{name}: n={len(values)} mean={statistics.fmean(values):.3f} "
                f"median={statistics.median(values):.3f} "
                f"zero={sum(v == 0 for v in values)}"
            )
        else:
            typer.echo(f"{name}: không có điểm")
    show_chart(scores, total, output)


if __name__ == "__main__":
    app()
