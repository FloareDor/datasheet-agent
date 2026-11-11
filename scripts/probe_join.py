"""Measure how many catalogue labels can be joined to a datasheet PDF.

The two sides name parts differently. Mouser gives an orderable part number
("OPA2991IDR"); the big archive names each PDF by the manufacturer's document code
("ad8656.pdf", "1637fd.pdf"). This script reports, for each candidate join strategy,
how many of the 1,335 labelled parts it reaches -- before any of it is committed to.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

RAW = Path("data/raw")
MOUSER = RAW / "mouser_opamps" / "mouser"
ARCHIVE = RAW / "opamp_dataset"


def load_labels() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((MOUSER / "csv").glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for row in rows:
        part = (row.get("Mfr Part Number") or "").strip()
        if part and part not in seen:
            seen.add(part)
            unique.append(row)
    return unique


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def main() -> None:
    labels = load_labels()
    print(f"labelled parts: {len(labels)}")

    direct = {p.stem: p for p in (MOUSER / "pdf").glob("*.pdf")}
    archive = {p.stem: p for p in ARCHIVE.rglob("*.pdf")}
    print(f"mouser pdfs: {len(direct)}   archive pdfs: {len(archive)}")

    # Strategy A: mouser pdfs are named "Manufacturer_PartNumber".
    direct_by_part = {norm(stem.split("_", 1)[-1]): stem for stem in direct}
    hit_a = {
        row["Mfr Part Number"] for row in labels if norm(row["Mfr Part Number"]) in direct_by_part
    }
    print(f"A. mouser filename join            : {len(hit_a)}")

    # Strategy B: archive stem is a prefix of the orderable part number.
    stems = sorted(
        ((norm(s), s) for s in archive if len(norm(s)) >= 5),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    hit_b: dict[str, str] = {}
    for row in labels:
        part = norm(row["Mfr Part Number"])
        for stem_norm, stem in stems:
            if part.startswith(stem_norm):
                hit_b[row["Mfr Part Number"]] = stem
                break
    print(f"B. archive stem prefix join        : {len(hit_b)}")

    union = hit_a | set(hit_b)
    print(f"A or B                             : {len(union)}")

    print("\nsample B matches:")
    for part, stem in list(hit_b.items())[:12]:
        print(f"  {part:22s} -> {stem}")

    unmatched = [r["Mfr Part Number"] for r in labels if r["Mfr Part Number"] not in union]
    print(f"\nunmatched: {len(unmatched)}  e.g. {unmatched[:12]}")
    print(
        "unmatched by mfr:",
        Counter(r["Mfr."] for r in labels if r["Mfr Part Number"] in set(unmatched)).most_common(8),
    )


if __name__ == "__main__":
    main()
