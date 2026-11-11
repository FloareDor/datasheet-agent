"""Re-grade a saved run against the labels, without re-running any model.

Reports keep every predicted value, so a change to grading can be applied to runs
already paid for. That matters: a grading fix should never be a reason to spend GPU
time again, and re-running would also confound the fix with sampling noise.

    python scripts/regrade.py runs/baseline-singlecall-qwen1.5b-test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasheet_agent.dataset import read_jsonl
from datasheet_agent.grading import grade_part, summarise

SPLITS = Path("data/splits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--write", action="store_true", help="overwrite the report in place")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    labels = {p.part_number: p.labels for p in read_jsonl(SPLITS / f"{args.split}.jsonl")}

    scores = []
    for entry in report["parts"]:
        part_number = entry["part_number"]
        if part_number not in labels:
            continue
        score = grade_part(part_number, entry.get("values", {}), labels[part_number])
        scores.append(score)
        entry["accuracy"] = round(score.accuracy, 4)
        entry["correct"] = score.correct
        entry["graded"] = score.graded
        entry["wrong_fields"] = [
            {"field": o.field, "predicted": o.predicted, "label": o.label}
            for o in score.wrong_fields()
        ]

    summary = summarise(scores)
    was = report["headline"]["accuracy"]
    report["headline"]["accuracy"] = round(summary.accuracy, 4)
    report["headline"]["correct"] = summary.correct
    report["headline"]["graded"] = summary.graded
    report["per_field"] = summary.to_dict()["per_field"]

    print(f"{args.report.name}: accuracy {was:.1%} -> {summary.accuracy:.1%}")
    print(f"  {summary.correct}/{summary.graded} fields over {summary.parts} parts\n")
    for name, stats in report["per_field"].items():
        print(f"  {name:26s} {stats['accuracy']:.1%}  ({stats['correct']}/{stats['graded']})")

    if args.write:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nrewrote {args.report}")


if __name__ == "__main__":
    main()
