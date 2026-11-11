"""Start a GRPO run that outlives this terminal.

Neither `modal run` nor `modal run --detach` survived here. A detached run was
cancelled anyway when the launching shell was killed, at step ~250 of 400, and
`modal run` with `.spawn()` stops the app the moment the local entrypoint returns
("Stopping app - local entrypoint completed").

What does work is spawning against a *deployed* function. A deployed app is not tied
to any local process, so the call keeps running whatever happens here.

    modal deploy -m datasheet_agent.modal_train
    python scripts/launch_training.py --steps 500 --learning-rate 5e-6
"""

from __future__ import annotations

import argparse

import modal

APP_NAME = "datasheet-agent-train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--run-name", default="grpo-qwen1.5b-v2")
    parser.add_argument(
        "--tool-call-penalty",
        type=float,
        default=None,
        help="override the per-tool-call cost; 0 removes it entirely",
    )
    args = parser.parse_args()

    train = modal.Function.from_name(APP_NAME, "train")
    handle = train.spawn(
        steps=args.steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        run_name=args.run_name,
        tool_call_penalty=args.tool_call_penalty,
    )
    print(f"spawned {args.run_name}")
    print(f"  call id : {handle.object_id}")
    print(f"  steps   : {args.steps}   lr: {args.learning_rate}")
    print("\nThis run is independent of this terminal. Follow it with:")
    print(f"  modal app logs {APP_NAME}")
    print("  or the W&B run in project 'datasheet-agent'")


if __name__ == "__main__":
    main()
