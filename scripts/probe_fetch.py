"""Check whether the datasheet URLs in the catalogue CSV actually resolve to PDFs.

If they do, the labelled set is not capped at the 297 parts the archive happens to
cover. Samples a handful rather than hammering the hosts.
"""

from __future__ import annotations

import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, "scripts")
from probe_join import load_labels, norm  # noqa: E402

ARCHIVE = Path("data/raw/opamp_dataset")
MOUSER_PDF = Path("data/raw/mouser_opamps/mouser/pdf")
UA = "Mozilla/5.0 (compatible; DatasheetAgent/0.1; research)"
SAMPLE = 12


def main() -> None:
    labels = load_labels()
    stems = {norm(p.stem) for p in ARCHIVE.rglob("*.pdf")}
    stems |= {norm(p.stem.split("_", 1)[-1]) for p in MOUSER_PDF.glob("*.pdf")}

    unmatched = [
        row
        for row in labels
        if not any(norm(row["Mfr Part Number"]).startswith(s) for s in stems if len(s) >= 5)
        and (row.get("Datasheet") or "").strip()
    ]
    print(f"unmatched with a datasheet URL: {len(unmatched)}")

    random.seed(0)
    ok = 0
    for row in random.sample(unmatched, min(SAMPLE, len(unmatched))):
        url = row["Datasheet"].strip()
        part = row["Mfr Part Number"]
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=25) as response:  # noqa: S310
                head = response.read(5)
                content_type = response.headers.get("Content-Type", "")
                is_pdf = head.startswith(b"%PDF")
                ok += is_pdf
                print(f"  {'PDF ' if is_pdf else 'MISS'} {part:20s} {content_type:28s} {url[:70]}")
        except Exception as exc:  # noqa: BLE001 - probing third-party hosts
            print(f"  ERR  {part:20s} {type(exc).__name__}: {str(exc)[:60]}")
    print(f"\n{ok}/{SAMPLE} resolved to a real PDF")


if __name__ == "__main__":
    main()
