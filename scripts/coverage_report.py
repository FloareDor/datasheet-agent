"""Split a method's accuracy into *finding* the value and *reading* it.

A wrong answer has two very different causes, and the accuracy number cannot tell them
apart. Either the text containing the answer never reached the model -- a navigation
failure, fixable with better retrieval or better tools -- or it did reach the model and
the model still answered wrong, which is a comprehension failure and fixable only with
a better model or a better prompt.

    accuracy = coverage x reading accuracy

where coverage is the share of fields whose answer was somewhere in the context the
method saw, and reading accuracy is how often it got those right.

The presence test is deliberately generous: it accepts a bare number anywhere in the
context, read in the unit the field was asked for, with no requirement that it sit near
the right parameter name. So coverage here is an **upper bound** -- the real figure is
lower. That bias is the safe direction, because it can only make navigation look better
than it is, and the finding below is that navigation is the problem anyway.

Only the seven numeric fields are scored. Deciding whether "surface mount" in a wall of
text counts as the answer "SMD/SMT" needs the grader's judgement, not a substring test.

    python scripts/coverage_report.py runs/v8-rag-bm25-large-test.json
    DSA_SAVE_CONTEXT=1 python scripts/run_eval.py --method single_agent ...  # for agents
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasheet_agent.dataset import read_jsonl
from datasheet_agent.extract import SINGLE_CALL_CHAR_BUDGET
from datasheet_agent.grading import grade_field
from datasheet_agent.rag import RETRIEVERS, chunk_pages, page_texts, retrieve
from datasheet_agent.schema import FIELDS, FIELDS_BY_NAME
from datasheet_agent.units import parse_all_quantities, parse_quantity, quantities_match

SPLITS = Path("data/splits")
NUMERIC = tuple(spec.name for spec in FIELDS if spec.grader == "numeric")


def answer_is_present(field_name: str, label: str, context: str) -> bool:
    """Could the labelled value have been read out of this text at all?"""
    spec = FIELDS_BY_NAME[field_name]
    gold = parse_quantity(label, spec.dimension)
    if gold is None or not context:
        return False
    candidates = parse_all_quantities(context, spec.dimension, assume_unit=spec.unit_hint)
    return any(quantities_match(candidate, gold) for candidate in candidates)


def context_for(part_number: str, pdf: Path, method: str, saved: dict) -> str:
    """The text this method put in front of the model, rebuilt where it is deterministic."""
    if "context_seen" in saved:
        return str(saved["context_seen"])
    if method.startswith("rag_"):
        chunks = chunk_pages(page_texts(pdf))
        if not chunks:
            return ""
        retriever = RETRIEVERS[method.removeprefix("rag_")](chunks)  # type: ignore[abstract]
        return "\n\n".join(chunk.render() for chunk in retrieve(retriever, FIELDS))
    if method == "single_call":
        pages = page_texts(pdf)
        text = "\n".join(f"--- PAGE {i} ---\n{p}" for i, p in enumerate(pages, start=1))
        return text[:SINGLE_CALL_CHAR_BUDGET]
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    method = str(report["headline"]["method"])
    parts = {p.part_number: p for p in read_jsonl(SPLITS / f"{args.split}.jsonl")}

    rebuildable = method == "single_call" or method.startswith("rag_")
    if not rebuildable and not any("context_seen" in p for p in report["parts"]):
        raise SystemExit(
                f"{method} runs do not record their context by default. Re-run with\n"
                f"  DSA_SAVE_CONTEXT=1 python scripts/run_eval.py --method {method} ..."
            )

    present = found_and_right = absent_and_right = graded = 0
    per_field: dict[str, list[int]] = {name: [0, 0, 0] for name in NUMERIC}

    for saved in report["parts"]:
        part = parts.get(saved["part_number"])
        if part is None:
            continue
        context = context_for(part.part_number, part.pdf, method, saved)
        for name in NUMERIC:
            label = part.labels.get(name, "")
            if not str(label).strip():
                continue
            predicted = (saved.get("values") or {}).get(name)
            outcome = grade_field(FIELDS_BY_NAME[name], predicted, label)
            if not outcome.graded:
                continue
            graded += 1
            here = answer_is_present(name, label, context)
            per_field[name][0] += 1
            present += here
            per_field[name][1] += here
            if outcome.correct:
                per_field[name][2] += 1
                if here:
                    found_and_right += 1
                else:
                    absent_and_right += 1

    if not graded:
        raise SystemExit("nothing graded")

    coverage = present / graded
    reading = found_and_right / present if present else 0.0
    print(f"\n=== {method} :: {args.report.name} ===")
    print(f"  numeric fields graded   {graded}")
    print(f"  coverage (answer there) {coverage:>7.1%}   <- navigation ceiling")
    print(f"  reading accuracy        {reading:>7.1%}   <- correct when it was there")
    print(f"  correct anyway          {absent_and_right:>7d}   (answer not in context; guessed or")
    print("                                    the presence test missed it)")
    print(f"\n{'field':26s} {'n':>4s} {'coverage':>9s} {'reading':>8s}")
    for name, (total, here, right) in per_field.items():
        if not total:
            continue
        print(f"{name:26s} {total:>4d} {here / total:>8.1%} {right / here if here else 0:>8.1%}")


if __name__ == "__main__":
    main()
