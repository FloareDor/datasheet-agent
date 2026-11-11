"""Run one extraction method over a split and write a report.

    python scripts/run_eval.py --method multi_agent --split test --limit 20

Backends are chosen by role, not by name, so the same command works whether the strong
model is Claude or the open stand-in:

    --backend small     the Qwen 1.5B agent on Modal
    --backend large     the strong open reference on Modal
    --backend claude    the Claude API (needs ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasheet_agent.dataset import read_jsonl
from datasheet_agent.evaluate import evaluate, print_headline
from datasheet_agent.llm import AnthropicBackend, Backend, OpenAIBackend
from datasheet_agent.modal_vllm import LARGE_MODEL, SMALL_MODEL, VERIFIER_MODEL
from datasheet_agent.pipeline import PipelineConfig

SPLITS = Path("data/splits")
RUNS = Path("runs")
CACHE = Path("data/cache")

MODAL_URL = "https://sairavig--datasheet-agent-vllm-{role}.us-east.modal.run/v1"
BACKEND_MODELS = {
    "small": SMALL_MODEL,
    "large": LARGE_MODEL,
    "verifier": VERIFIER_MODEL,
    # The trained server exposes the LoRA adapter under this name.
    "trained": "datasheet-grpo",
    # Same weights as `small`; the difference is n-gram speculative decoding.
    "small-spec": SMALL_MODEL,
}


def make_backend(name: str, model_override: str | None = None) -> Backend:
    if name == "claude":
        return AnthropicBackend(model=model_override or "claude-sonnet-5-5")
    if name not in BACKEND_MODELS:
        raise SystemExit(f"unknown backend {name!r}")
    return OpenAIBackend(
        model=model_override or BACKEND_MODELS[name],
        base_url=MODAL_URL.format(role=name),
        api_key="modal",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="multi_agent")
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument(
        "--backend",
        default="small",
        choices=["small", "small-spec", "trained", "large", "claude"],
    )
    parser.add_argument("--model", default=None, help="override the backend's model id")
    parser.add_argument("--limit", type=int, default=0, help="0 means the whole split")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--no-verifier", action="store_true")
    parser.add_argument(
        "--verifier-backend",
        default="verifier",
        choices=["verifier", "large", "claude", "none"],
        help="'none' runs code checks only",
    )
    parser.add_argument("--tag", default="", help="suffix for the report filename")
    args = parser.parse_args()

    parts = read_jsonl(SPLITS / f"{args.split}.jsonl")
    if args.limit:
        parts = parts[: args.limit]

    verifier = None if args.verifier_backend == "none" else make_backend(args.verifier_backend)
    config = PipelineConfig(
        backend=make_backend(args.backend, args.model),
        cache_dir=CACHE,
        verifier_backend=verifier,
        use_verifier=not args.no_verifier,
        max_steps=args.max_steps,
    )

    label = args.tag or (
        f"{args.method}-{args.backend}" + ("-noverify" if args.no_verifier else "")
    )
    print(f"running {args.method} on {len(parts)} parts of {args.split} via {args.backend}")
    report = evaluate(args.method, config, parts, workers=args.workers)
    print_headline(report)

    out = RUNS / f"{label}-{args.split}.json"
    report.write(out)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
