"""Grade the verifier as a classifier, not just by its effect on the score.

A verifier that hurts the final number could be doing either of two very different
things: rejecting correct values, or correctly flagging wrong ones that the retry then
fails to repair. Those call for opposite fixes, and the headline accuracy cannot tell
them apart. This needs `pre_retry_values` in the report.

    python scripts/verifier_report.py runs/v5-multiagent-verified-qwen1.5b-test.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasheet_agent.dataset import read_jsonl
from datasheet_agent.grading import grade_field
from datasheet_agent.schema import FIELDS_BY_NAME

SPLITS = Path("data/splits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    labels = {p.part_number: p.labels for p in read_jsonl(SPLITS / f"{args.split}.jsonl")}

    if not any(p.get("pre_retry_values") for p in report["parts"]):
        raise SystemExit(
            "this report has no pre_retry_values; re-run the eval with the current code"
        )

    # Confusion matrix over rejections, plus what the retry did with each.
    rejected_correct = 0  # the verifier was wrong to reject
    rejected_wrong = 0  # the verifier was right to reject
    repaired_from_wrong = 0  # right to reject, and the retry fixed it
    broken_from_correct = 0  # wrong to reject, and the retry destroyed a good value
    nulled = 0
    by_stage: Counter[str] = Counter()
    accepted_wrong = 0  # the verifier let a wrong value through

    for entry in report["parts"]:
        part_labels = labels.get(entry["part_number"], {})
        before = entry.get("pre_retry_values", {})
        after = entry.get("values", {})
        rejections = {r["field"]: r for r in entry.get("verification", {}).get("rejected", [])}

        for name, spec in FIELDS_BY_NAME.items():
            label = part_labels.get(name, "")
            if not label:
                continue
            was_right = grade_field(spec, before.get(name), label).correct
            now_right = grade_field(spec, after.get(name), label).correct

            if name in rejections:
                by_stage[rejections[name]["stage"]] += 1
                if was_right:
                    rejected_correct += 1
                    if not now_right:
                        broken_from_correct += 1
                        if not (after.get(name) or "").strip():
                            nulled += 1
                else:
                    rejected_wrong += 1
                    if now_right:
                        repaired_from_wrong += 1
            elif not was_right and (before.get(name) or "").strip():
                accepted_wrong += 1

    total = rejected_correct + rejected_wrong
    precision = rejected_wrong / total if total else 0.0

    print(f"{args.report.name}\n")
    print(f"rejections graded          : {total}")
    print(f"  correctly rejected       : {rejected_wrong}")
    print(f"  WRONGLY rejected         : {rejected_correct}")
    print(f"  precision                : {precision:.1%}")
    print(f"  by stage                 : {dict(by_stage)}\n")
    print(f"retry repaired a wrong value : {repaired_from_wrong}")
    print(f"retry BROKE a correct value  : {broken_from_correct}")
    print(f"  of which nulled out        : {nulled}")
    print(f"\nnet fields from verification : {repaired_from_wrong - broken_from_correct:+d}")
    print(f"wrong values it let through  : {accepted_wrong}")


if __name__ == "__main__":
    main()
