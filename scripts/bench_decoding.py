"""Controlled decode-speed comparison: n-gram speculation on vs off.

The end-to-end eval cannot settle this. It runs several parts concurrently, so server
contention and per-part prompt length swamp a few percent of decode time. This sends
the *same* prompt to both servers, one request at a time, several times, and reports
output tokens per second.

The prompt is a real datasheet page and the request asks for the values *quoted back*,
which is the case n-gram speculation is supposed to win: the draft tokens are already
present in the prompt.

    python scripts/bench_decoding.py --repeats 5
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from datasheet_agent.dataset import read_jsonl
from datasheet_agent.llm import OpenAIBackend
from datasheet_agent.tools import DatasheetToolbox

URL = "https://sairavig--datasheet-agent-vllm-{role}.us-east.modal.run/v1"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def build_prompt(index: int) -> str:
    part = read_jsonl(Path("data/splits/test.jsonl"))[index]
    with DatasheetToolbox(part.pdf, Path("data/cache")) as toolbox:
        page = toolbox.read_table(3)
    return (
        "Here is one page of a datasheet:\n\n"
        f"{page}\n\n"
        "Quote back, verbatim, every table row that states an electrical "
        "characteristic, one per line. Copy the text exactly as it appears."
    )


def measure(role: str, prompt: str, repeats: int, max_tokens: int) -> list[float]:
    backend = OpenAIBackend(MODEL, base_url=URL.format(role=role), api_key="modal")
    rates: list[float] = []
    for _ in range(repeats):
        started = time.monotonic()
        reply = backend.chat(
            "You copy text exactly.",
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        elapsed = time.monotonic() - started
        if elapsed > 0 and reply.usage.output_tokens:
            rates.append(reply.usage.output_tokens / elapsed)
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--part-index", type=int, default=2)
    args = parser.parse_args()

    prompt = build_prompt(args.part_index)
    print(f"prompt: {len(prompt):,} chars\n")

    results: dict[str, list[float]] = {}
    for role in ("small", "small-spec"):
        rates = measure(role, prompt, args.repeats, args.max_tokens)
        results[role] = rates
        if rates:
            print(
                f"{role:12s} {statistics.median(rates):7.1f} tok/s median "
                f"(min {min(rates):.1f}, max {max(rates):.1f}, n={len(rates)})"
            )
        else:
            print(f"{role:12s} no successful measurements")

    if all(results.values()):
        off = statistics.median(results["small"])
        on = statistics.median(results["small-spec"])
        print(f"\nspeculation: {(on / off - 1) * 100:+.1f}% decode throughput")


if __name__ == "__main__":
    main()
