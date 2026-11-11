"""Block until a vLLM server answers, then print what it is serving.

Cold starts pull weights and build the engine, so the first call after a deploy can
take many minutes. Unbuffered so progress is visible while it waits.

    python scripts/wait_for_server.py small --minutes 25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "https://sairavig--datasheet-agent-vllm-{role}.us-east.modal.run"
POLL_SECONDS = 20


def wait(role: str, minutes: float) -> int:
    url = BASE.format(role=role) + "/v1/models"
    deadline = time.monotonic() + minutes * 60
    started = time.monotonic()
    while time.monotonic() < deadline:
        elapsed = time.monotonic() - started
        try:
            with urllib.request.urlopen(url, timeout=90) as response:  # noqa: S310
                served = json.loads(response.read())["data"][0]["id"]
            print(f"[{elapsed:6.0f}s] UP: {role} serving {served}", flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001 - any failure means "not yet"
            detail = getattr(exc, "code", type(exc).__name__)
            print(f"[{elapsed:6.0f}s] waiting ({detail})", flush=True)
            time.sleep(POLL_SECONDS)
    print(f"TIMED OUT after {minutes} minutes", flush=True)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["small", "small-spec", "trained", "large", "verifier"])
    parser.add_argument("--minutes", type=float, default=25.0)
    args = parser.parse_args()
    sys.exit(wait(args.role, args.minutes))


if __name__ == "__main__":
    main()
