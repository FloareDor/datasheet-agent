"""Upload the splits and their datasheet PDFs to the Modal training volume.

Only the PDFs the splits actually reference are uploaded -- the full archive is 5 GB
and the splits touch a few hundred megabytes of it.

    python scripts/upload_training_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

from datasheet_agent.dataset import read_jsonl

SPLITS = Path("data/splits")
VOLUME_NAME = "datasheet-training-data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()

    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    pdfs: dict[str, Path] = {}
    for split in args.splits:
        for part in read_jsonl(SPLITS / f"{split}.jsonl"):
            # Flattened by basename: the archive's directory layout is irrelevant on
            # the volume, and the training job resolves PDFs by name.
            pdfs.setdefault(Path(part.pdf_path).name, part.pdf)

    total_mb = sum(p.stat().st_size for p in pdfs.values() if p.exists()) / 1e6
    print(f"uploading {len(pdfs)} datasheets ({total_mb:.0f} MB) and {len(args.splits)} splits")

    with volume.batch_upload(force=True) as batch:
        for split in args.splits:
            batch.put_file(SPLITS / f"{split}.jsonl", f"/splits/{split}.jsonl")
        for name, path in sorted(pdfs.items()):
            if path.exists():
                batch.put_file(path, f"/pdfs/{name}")

    print("done")


if __name__ == "__main__":
    main()
