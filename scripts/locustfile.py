"""Load test for the served extraction endpoint.

    locust -f scripts/locustfile.py --host https://<app>.modal.run

Two user classes, because they stress different things. `ExtractionUser` posts whole
datasheets and measures the end-to-end agent path. `ChatUser` hits the raw vLLM
completion endpoint, which isolates serving throughput from agent overhead -- without
it, a slow result cannot be attributed to either.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

SPLITS = Path("data/splits/test.jsonl")


def _part_numbers(limit: int = 50) -> list[str]:
    """Real part numbers from the test split, so prefix caching behaves realistically."""
    if not SPLITS.exists():
        return ["OPA2991IDR", "TL072CP", "LM358N"]
    with SPLITS.open(encoding="utf-8") as handle:
        parts = [json.loads(line)["part_number"] for line in handle if line.strip()]
    return parts[:limit] or ["OPA2991IDR"]


PARTS = _part_numbers()


class ExtractionUser(HttpUser):
    """Exercises the full agent path: tools, several model turns, verification."""

    wait_time = between(1, 3)

    @task
    def extract(self) -> None:
        part = random.choice(PARTS)
        with self.client.post(
            "/extract",
            json={"part_number": part},
            name="/extract",
            catch_response=True,
            timeout=300,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            try:
                payload = response.json()
            except ValueError:
                response.failure("response was not JSON")
                return
            if not payload.get("values"):
                response.failure("no values returned")


class ChatUser(HttpUser):
    """Baseline: raw vLLM throughput, with the agent path removed."""

    wait_time = between(1, 2)

    @task
    def completion(self) -> None:
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "messages": [{"role": "user", "content": "State the slew rate unit."}],
                "max_tokens": 32,
                "temperature": 0.0,
            },
            name="/v1/chat/completions",
            timeout=120,
        )
