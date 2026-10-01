"""Hiển thị thống kê golden testset raw bằng Seaborn.

Ví dụ:
    uv run python tools/visualize_golden_testset.py
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore[import-untyped]
import typer
from matplotlib.patches import Rectangle

DEFAULT_INPUT_PATH = Path("data/eval/golden_testset.json")

app = typer.Typer(add_completion=False, no_args_is_help=False)


def load_cases(input_path: Path) -> list[Mapping[str, object]]:
    """Read and validate the top-level list of golden test cases."""
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Không tìm thấy file input: {input_path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Input không phải JSON UTF-8 hợp lệ: {input_path}") from error

    if not isinstance(payload, list):
        raise TypeError("Golden testset raw phải là một JSON array.")
    if not all(isinstance(case, Mapping) for case in payload):
        raise TypeError("Mỗi phần tử của golden testset raw phải là JSON object.")
    return payload


def count_synthesizers(cases: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """Count test cases by synthesizer, retaining missing values in the total."""
    names: list[str] = []
    for case in cases:
        name = case.get("synthesizer_name")
        names.append(name if isinstance(name, str) and name.strip() else "(missing)")
    counts = Counter(names)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def show_chart(counts: Mapping[str, int]) -> None:
    """Show a simple bar chart of test-case counts by synthesizer."""
    total_cases = sum(counts.values())
    if total_cases == 0:
        raise ValueError("Golden testset raw không có test case để vẽ biểu đồ.")

    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=list(counts),
        y=list(counts.values()),
        hue=list(counts),
        palette="deep",
        legend=False,
        ax=axis,
    )
    for bar, count in zip(axis.patches, counts.values(), strict=True):
        rectangle = cast(Rectangle, bar)
        axis.text(
            rectangle.get_x() + rectangle.get_width() / 2,
            rectangle.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )
    axis.set(
        title=f"Phân bố synthesizer_name — {total_cases} test",
        xlabel="Synthesizer",
        ylabel="Số test",
        ylim=(0, max(counts.values()) * 1.15),
    )
    figure.tight_layout()
    plt.show()


@app.command()
def main(
    input_path: Path = typer.Option(
        DEFAULT_INPUT_PATH, help="Đường dẫn tới golden_testset_raw.json."
    ),
) -> None:
    """Mở biểu đồ thống kê theo `synthesizer_name`."""
    try:
        counts = count_synthesizers(load_cases(input_path))
        summary = ", ".join(f"{name}: {count}" for name, count in counts.items())
        typer.echo(f"Mở biểu đồ ({sum(counts.values())} test): {summary}")
        show_chart(counts)
    except (TypeError, ValueError) as error:
        typer.echo(f"Lỗi đầu vào: {error}", err=True)
        raise typer.Exit(2) from error


if __name__ == "__main__":
    app()
