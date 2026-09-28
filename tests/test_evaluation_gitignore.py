"""Test `.gitignore` cho artefact của `tools/generate_testset.py` (evaluation_spec.md mục 4.5).

KG chứa embedding nên rất lớn và dựng lại được -> phải bị ignore, cùng file `*.tmp` của
lần ghi nguyên tử. Ngược lại `golden_testset*.json`, `generation_progress.json` và
`units/` là dữ liệu cần commit nên TUYỆT ĐỐI không bị ignore (ignore nhầm = mất tiến độ
nhiều ngày khi đổi máy). Dùng `git check-ignore --no-index` nên không cần file tồn tại.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="Cần git và thư mục .git để chạy git check-ignore.",
)


def _is_ignored(relative_path: str) -> bool:
    completed = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", "--", relative_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode in (0, 1), completed.stderr  # 128 = lỗi git thật
    return completed.returncode == 0


@pytest.mark.parametrize(
    "relative_path",
    [
        "data/eval/knowledge_graph/Luật bảo hiểm y tế__01.json",
        "data/eval/knowledge_graph/A__01.json.tmp",
        "data/eval/knowledge_graph/sub/dir/B__02.json",
        "data/eval/golden_testset_raw.json.tmp",
        "data/eval/generation_progress.json.tmp",
        "data/eval/units/A__01.md.tmp",
    ],
)
def test_kg_va_file_tmp_cua_lan_ghi_nguyen_tu_bi_ignore(relative_path: str):
    assert _is_ignored(relative_path)


@pytest.mark.parametrize(
    "relative_path",
    [
        "data/eval/golden_testset.json",
        "data/eval/golden_testset_raw.json",
        "data/eval/generation_progress.json",
        "data/eval/units_plan.md",
        "data/eval/units/Luật bảo hiểm y tế__01.md",
    ],
)
def test_testset_progress_va_units_can_commit_khong_bi_ignore(relative_path: str):
    assert not _is_ignored(relative_path)
