"""Negative control: run the op-amp schema against transistor datasheets.

A bipolar transistor datasheet states no slew rate, no gain-bandwidth product and no
input offset voltage. Those fields do not exist for the part. So the correct answer is
null, every time, and any value returned is an invention.

This needs no labels, which is what makes it usable: the transistor archive ships 6,967
PDFs with no ground truth at all. The question it answers is the one the whole
"source-grounded" claim rests on — when the document does not contain the answer, does
the system say so, or does it make something up?

    python scripts/negative_control.py --limit 30
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasheet_agent.extract import extract_single_call
from datasheet_agent.llm import Backend
from datasheet_agent.pipeline import PipelineConfig, extract_single_agent
from datasheet_agent.schema import FIELDS

TRANSISTORS = Path("data/raw/transistor_dataset/test/pdf")
CACHE = Path("data/cache")
RUNS = Path("runs")

# Fields a bipolar transistor datasheet cannot state. Package and channel-count are
# excluded: a transistor does have a package, and "channels" is at least arguable, so
# only the unambiguous op-amp-only parameters are scored.
IMPOSSIBLE = (
    "slew_rate",
    "gain_bandwidth_product",
    "input_offset_voltage",
)


def _backend(role: str) -> Backend:
    import sys

    sys.path.insert(0, "scripts")
    from run_eval import make_backend

    return make_backend(role)


def run_one(method: str, backend: Backend, pdf: Path) -> dict[str, str]:
    part = pdf.stem
    if method == "single_call":
        return extract_single_call(backend, part, pdf).values
    config = PipelineConfig(backend=backend, cache_dir=CACHE, use_verifier=False, max_steps=10)
    return extract_single_agent(config, part, pdf).extraction.values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--backend", default="small")
    parser.add_argument("--method", default="single_call", choices=["single_call", "single_agent"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pdfs = sorted(p for p in TRANSISTORS.glob("*") if p.suffix.lower() == ".pdf")
    if not pdfs:
        raise SystemExit(f"no transistor PDFs under {TRANSISTORS}")
    random.Random(args.seed).shuffle(pdfs)
    pdfs = pdfs[: args.limit]

    backend = _backend(args.backend)
    print(f"{args.method} / {args.backend} on {len(pdfs)} transistor datasheets\n")

    invented: dict[str, int] = dict.fromkeys(IMPOSSIBLE, 0)
    examples: list[tuple[str, str, str]] = []
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, args.method, backend, pdf): pdf for pdf in pdfs}
        for future in as_completed(futures):
            pdf = futures[future]
            try:
                values = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad PDF must not end the run
                print(f"  {pdf.stem}: {type(exc).__name__}")
                continue
            done += 1
            for name in IMPOSSIBLE:
                value = (values.get(name) or "").strip()
                if value:
                    invented[name] += 1
                    if len(examples) < 10:
                        examples.append((pdf.stem, name, value))

    if not done:
        raise SystemExit("no datasheets completed")

    print(f"{'field':26s} {'invented':>9s} {'rate':>7s}   (correct answer is always null)")
    for name in IMPOSSIBLE:
        print(f"{name:26s} {invented[name]:>9d} {invented[name] / done:>6.1%}")
    total = sum(invented.values())
    print(f"\noverall invention rate: {total}/{done * len(IMPOSSIBLE)} = "
          f"{total / (done * len(IMPOSSIBLE)):.1%}")

    if examples:
        print("\ninvented values:")
        for part, field, value in examples:
            print(f"  {part:28s} {field:24s} {value}")

    out = RUNS / f"negative-control-{args.method}-{args.backend}.json"
    out.write_text(
        json.dumps(
            {
                "method": args.method,
                "backend": args.backend,
                "datasheets": done,
                "fields_checked": list(IMPOSSIBLE),
                "invented": invented,
                "invention_rate": round(total / (done * len(IMPOSSIBLE)), 4),
                "examples": [
                    {"part": p, "field": f, "value": v} for p, f, v in examples
                ],
                "all_fields": [f.name for f in FIELDS],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
