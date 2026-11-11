"""Collect every saved run into the comparison tables the README reports.

Reads `runs/*.json` and writes `runs/summary.md`. Nothing is recomputed from a model:
a run that is not in `runs/` does not appear in the table, so the README can never
quote a number that was not measured.

    python scripts/summarise_runs.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path("runs")

COLUMNS = [
    ("accuracy", "accuracy", "{:.1%}"),
    ("tokens_in_per_part", "tokens in", "{:,.0f}"),
    ("tokens_out_per_part", "tokens out", "{:,.0f}"),
    ("llm_calls_per_part", "LLM calls", "{:.2f}"),
    ("tool_calls_per_part", "tool calls", "{:.1f}"),
    ("latency_s_per_part", "latency (s)", "{:.1f}"),
    ("errors", "failures", "{:.0f}"),
]


def load_runs(directory: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            runs[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return runs


def _format(value: object, spec: str) -> str:
    try:
        return spec.format(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def headline_table(runs: dict[str, dict]) -> str:
    header = "| run | model | " + " | ".join(label for _, label, _ in COLUMNS) + " |"
    rule = "|---" * (len(COLUMNS) + 2) + "|"
    lines = [header, rule]
    for name, report in runs.items():
        headline = report.get("headline", {})
        cells = [_format(headline.get(key), spec) for key, _, spec in COLUMNS]
        model = str(headline.get("model", "")).split("/")[-1]
        lines.append(f"| `{name}` | {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_field_table(runs: dict[str, dict]) -> str:
    fields = sorted({f for r in runs.values() for f in r.get("per_field", {})})
    if not fields:
        return ""
    header = "| field | " + " | ".join(f"`{n}`" for n in runs) + " |"
    rule = "|---" * (len(runs) + 1) + "|"
    lines = [header, rule]
    for field in fields:
        cells = []
        for report in runs.values():
            stats = report.get("per_field", {}).get(field)
            cells.append(f"{stats['accuracy']:.1%}" if stats else "-")
        lines.append(f"| {field} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def verifier_table(runs: dict[str, dict]) -> str:
    rows = [
        (name, r["headline"])
        for name, r in runs.items()
        if r.get("headline", {}).get("verifier_rejections", 0)
    ]
    if not rows:
        return ""
    lines = [
        "| run | rejections | fields repaired | accuracy |",
        "|---|---|---|---|",
    ]
    for name, headline in rows:
        lines.append(
            f"| `{name}` | {headline['verifier_rejections']} | "
            f"{headline['fields_repaired']} | {headline['accuracy']:.1%} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    parser.add_argument("--out", type=Path, default=RUNS / "summary.md")
    args = parser.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"no runs found in {args.runs_dir}")

    sections = [
        "# Results\n",
        f"Every row is a measured run over the 100-part test split. {len(runs)} runs.\n",
        "## Headline\n",
        headline_table(runs),
        "\n## Accuracy by field\n",
        per_field_table(runs),
    ]
    verifier = verifier_table(runs)
    if verifier:
        sections += ["\n## Verifier\n", verifier]

    text = "\n".join(sections) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
