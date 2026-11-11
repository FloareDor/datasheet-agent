"""Check that each joined part is actually documented by the PDF it was joined to.

The join is heuristic, so it needs an independent check. For every part, the longest
prefix of its part number that appears in the datasheet's own text is measured. A part
whose number appears nowhere in its datasheet is a bad join and should be dropped.

    python scripts/validate_dataset.py --drop   # rewrite the splits without bad joins
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pymupdf

from datasheet_agent.dataset import LabelledPart, normalise, read_jsonl, split_dataset, write_jsonl

SPLITS = Path("data/splits")
PAGES_TO_SCAN = 4

# The base part number: manufacturer prefix plus the numeric type, before the
# package / temperature / packaging suffix. "TL071CD" -> "TL071", "OPA2991IDR" ->
# "OPA2991", "MAX4471ESA+T" -> "MAX4471". This, not a fixed character count, is what
# a datasheet must name for the join to be sound -- a five-character base like TL071
# can never clear a six-character bar.
BASE_PART = re.compile(r"^([A-Za-z]{1,4}\d{2,5})")


def datasheet_text(pdf_path: Path, pages: int = PAGES_TO_SCAN) -> str:
    try:
        with pymupdf.open(pdf_path) as doc:
            return " ".join(doc[i].get_text() for i in range(min(pages, doc.page_count)))
    except Exception:  # noqa: BLE001 - third-party PDFs vary wildly
        return ""


def base_part_number(part_number: str) -> str:
    """The part's type designator, without package or packaging suffix."""
    match = BASE_PART.match(part_number.strip())
    return normalise(match.group(1)) if match else normalise(part_number)


def names_the_part(part_number: str, text: str) -> bool:
    """True when the datasheet's text names this part's base number."""
    base = base_part_number(part_number)
    if not base:
        return False
    # Datasheets hyphenate and soft-wrap part numbers; flatten before matching.
    return base in normalise(re.sub(r"[\s­-]+", "", text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drop", action="store_true", help="rewrite splits without bad joins")
    parser.add_argument("--train", type=int, default=300)
    parser.add_argument("--test", type=int, default=100)
    args = parser.parse_args()

    parts = read_jsonl(SPLITS / "all.jsonl")
    print(f"validating {len(parts)} parts against {len({p.pdf_path for p in parts})} datasheets\n")

    text_cache: dict[str, str] = {}
    by_method: dict[str, list[bool]] = {}
    good: list[LabelledPart] = []
    bad: list[LabelledPart] = []
    no_text: set[str] = set()

    for part in parts:
        if part.pdf_path not in text_cache:
            text_cache[part.pdf_path] = datasheet_text(part.pdf)
            if len(text_cache[part.pdf_path].strip()) < 200:
                no_text.add(part.pdf_path)
        ok = names_the_part(part.part_number, text_cache[part.pdf_path])
        by_method.setdefault(part.join_method, []).append(ok)
        (good if ok else bad).append(part)

    print(f"{'method':14s} {'parts':>6s} {'named':>7s} {'rate':>7s}")
    for method, flags in sorted(by_method.items()):
        hits = sum(flags)
        print(f"{method:14s} {len(flags):6d} {hits:7d} {100 * hits / len(flags):6.0f}%")
    print(f"\ngood: {len(good)}   bad: {len(bad)}")
    if no_text:
        print(f"datasheets with no usable text layer: {len(no_text)} (scanned images)")

    if bad:
        print("\ndropped joins (part -> datasheet):")
        for part in bad[:12]:
            scanned = " [no text layer]" if part.pdf_path in no_text else ""
            print(
                f"  {part.part_number:22s} -> {Path(part.pdf_path).name:34s}"
                f" [{part.join_method}]{scanned}"
            )
        print("\ndropped by method:", dict(Counter(p.join_method for p in bad)))

    if args.drop:
        write_jsonl(good, SPLITS / "all.jsonl")
        train, test = split_dataset(good, args.train, args.test)
        write_jsonl(train, SPLITS / "train.jsonl")
        write_jsonl(test, SPLITS / "test.jsonl")
        print(f"\nrewrote splits: train={len(train)} test={len(test)} from {len(good)} good parts")


if __name__ == "__main__":
    main()
