"""Build the labelled op-amp dataset and write train/test splits.

python scripts/build_dataset.py --train 300 --test 100
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from datasheet_agent.dataset import build_dataset, split_dataset, write_jsonl
from datasheet_agent.schema import FIELDS

RAW = Path("data/raw")
MOUSER = RAW / "mouser_opamps" / "mouser"
ARCHIVE = RAW / "opamp_dataset"
OUT = Path("data/splits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=300)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    parts = build_dataset(
        catalogue_csv_dir=MOUSER / "csv",
        catalogue_pdf_dir=MOUSER / "pdf",
        archive_root=ARCHIVE,
        archive_html_dirs=[*ARCHIVE.rglob("html"), MOUSER / "html"],
    )
    print(f"joined parts: {len(parts)}")
    print("by join method:", dict(Counter(p.join_method for p in parts)))
    print("distinct datasheets:", len({p.pdf_path for p in parts}))

    print("\nlabel coverage:")
    for spec in FIELDS:
        n = sum(1 for p in parts if p.labels.get(spec.name))
        pct = 100 * n / max(len(parts), 1)
        print(f"  {spec.name:26s} {spec.group:10s} {n:4d}/{len(parts)}  {pct:.0f}%")

    train, test = split_dataset(parts, args.train, args.test, seed=args.seed)
    write_jsonl(train, OUT / "train.jsonl")
    write_jsonl(test, OUT / "test.jsonl")
    write_jsonl(parts, OUT / "all.jsonl")

    print(f"\ntrain: {len(train)} parts / {len({p.pdf_path for p in train})} datasheets")
    print(f"test : {len(test)} parts / {len({p.pdf_path for p in test})} datasheets")
    overlap = {p.pdf_path for p in train} & {p.pdf_path for p in test}
    print(f"datasheet overlap between splits: {len(overlap)} (must be 0)")
    print(f"\nwritten to {OUT}/")


if __name__ == "__main__":
    main()
