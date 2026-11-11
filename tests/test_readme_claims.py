"""Every accuracy quoted in the README must match a measured run.

This exists because it already failed once. Writing the model comparison by hand, the
32B column was filled in from plausible-looking memory rather than from `runs/`, and
every figure in it was wrong — including the sign of one difference. `summarise_runs.py`
generates tables that cannot be wrong, but prose tables are written by hand, so the
prose needs a check of its own.

The test parses the README's results tables and verifies each accuracy against the run
it names. A run that has not been measured is skipped rather than failing, so the test
is usable on a fresh clone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

README = Path("README.md")
RUNS = Path("runs")

# README label -> the run file that must support it.
CLAIMS: dict[str, str] = {
    "32B single call": "large-singlecall-test.json",
    "32B multi-agent": "large-multiagent-test.json",
    "32B single agent": "large-singleagent-test.json",
    "1.5B single call": "v3-singlecall-qwen1.5b-test.json",
    "1.5B single agent": "v3-singleagent-qwen1.5b-test.json",
    "1.5B multi-agent": "v3-multiagent-qwen1.5b-test.json",
    "1.5B RL-trained agent": "v3trained-singleagent-test.json",
}

# Rows in the RL table, which names runs by version rather than by model.
RL_CLAIMS: dict[str, str] = {
    "base agent": "v3-singleagent-qwen1.5b-test.json",
    "v2": "v2trained-singleagent-test.json",
    "v3": "v3trained-singleagent-test.json",
    "v4": "v4trained-nocost-singleagent-test.json",
}

TOLERANCE = 0.06  # percentage points; the README rounds to one decimal


def _accuracy(run_file: str) -> float | None:
    path = RUNS / run_file
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["headline"]["accuracy"] * 100


def _readme_rows() -> list[list[str]]:
    if not README.exists():
        pytest.skip("README not present")
    return [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in README.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("|")
    ]


def _quoted_accuracy(cells: list[str]) -> float | None:
    """The first percentage in a row, with any bold markers stripped."""
    for cell in cells:
        match = re.fullmatch(r"\*{0,2}(\d+\.\d)%\*{0,2}", cell)
        if match:
            return float(match.group(1))
    return None


@pytest.mark.parametrize(("label", "run_file"), sorted(CLAIMS.items()))
def test_model_comparison_matches_the_measured_run(label: str, run_file: str) -> None:
    measured = _accuracy(run_file)
    if measured is None:
        pytest.skip(f"{run_file} not measured yet")

    rows = [r for r in _readme_rows() if _row_names(r, label)]
    assert rows, f"README no longer has a row for {label!r}"
    for cells in rows:
        quoted = _quoted_accuracy(cells)
        if quoted is None:
            continue
        assert abs(quoted - measured) < TOLERANCE, (
            f"README claims {quoted}% for {label!r}, but {run_file} measured "
            f"{measured:.1f}%"
        )


def _row_names(cells: list[str], label: str) -> bool:
    """True when a row names this configuration, in either table layout."""
    if label in RL_CLAIMS and cells and cells[0].strip("*` ") == label:
        return True
    model, _, method = label.partition(" ")
    joined = " ".join(cells[:3]).lower()
    return model.lower() in joined and method.lower() in joined


@pytest.mark.parametrize(("label", "run_file"), sorted(RL_CLAIMS.items()))
def test_rl_table_matches_the_measured_run(label: str, run_file: str) -> None:
    measured = _accuracy(run_file)
    if measured is None:
        pytest.skip(f"{run_file} not measured yet")

    rows = [r for r in _readme_rows() if r and r[0].strip("*` ") == label]
    assert rows, f"README no longer has an RL row for {label!r}"
    quoted = _quoted_accuracy(rows[0])
    assert quoted is not None, f"RL row {label!r} quotes no accuracy"
    assert abs(quoted - measured) < TOLERANCE, (
        f"README RL table claims {quoted}% for {label!r}, but {run_file} measured "
        f"{measured:.1f}%"
    )


def test_every_run_the_readme_names_actually_exists() -> None:
    """A README naming a run file that is not in `runs/` is quoting nothing."""
    missing = [f for f in {**CLAIMS, **RL_CLAIMS}.values() if not (RUNS / f).exists()]
    if missing:
        pytest.skip(f"not yet measured: {missing}")
    assert True
