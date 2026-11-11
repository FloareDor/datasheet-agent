"""Cost per datasheet, for models billed by GPU time rather than by token.

A hosted API bills per token, so cost follows from usage directly. A model served on
rented GPUs does not: the bill is GPU-seconds, and the cost of one extraction depends
on how many requests share the GPU while it runs. Comparing a self-hosted 1.5B against
a per-token API therefore needs an explicit throughput assumption, stated rather than
buried.

The honest unit here is **cost per datasheet at measured throughput**: take the GPU's
hourly rate, divide by the parts per hour the server actually sustained in the
evaluation, and say which run the throughput came from.
"""

from __future__ import annotations

from dataclasses import dataclass

# USD per GPU-second, as published at https://modal.com/pricing, read 2026-11-11.
# Recorded with the date because a price quoted without one silently becomes a guess.
PRICE_SOURCE = "https://modal.com/pricing"
PRICE_READ_ON = "2026-11-11"

GPU_PER_SECOND_USD: dict[str, float] = {
    "L40S": 0.000542,
    "H100": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
    "A100-80GB": 0.000694,
    "A100-40GB": 0.000583,
    "A10": 0.000306,
    "L4": 0.000222,
    "T4": 0.000164,
}

GPU_HOURLY_USD: dict[str, float] = {
    name: round(per_second * 3600, 4) for name, per_second in GPU_PER_SECOND_USD.items()
}


@dataclass(frozen=True)
class ServingCost:
    """What one datasheet cost on rented GPUs."""

    gpu: str
    gpu_count: int
    hourly_usd: float
    parts_per_hour: float
    source_run: str
    """Which measured run the throughput came from. Never an assumption."""

    @property
    def usd_per_part(self) -> float:
        if self.parts_per_hour <= 0:
            return 0.0
        return self.hourly_usd * self.gpu_count / self.parts_per_hour

    def to_dict(self) -> dict[str, object]:
        return {
            "gpu": f"{self.gpu} x{self.gpu_count}",
            "gpu_hourly_usd": round(self.hourly_usd, 4),
            "parts_per_hour": round(self.parts_per_hour, 1),
            "usd_per_part": round(self.usd_per_part, 6),
            "source_run": self.source_run,
        }


def parts_per_hour(parts: int, wall_seconds: float) -> float:
    """Throughput a run actually sustained, including all concurrency it used."""
    if wall_seconds <= 0:
        return 0.0
    return parts * 3600.0 / wall_seconds


def serving_cost(
    gpu: str,
    gpu_count: int,
    parts: int,
    wall_seconds: float,
    source_run: str,
) -> ServingCost | None:
    """Cost per datasheet, or None when the GPU's rate has not been recorded."""
    hourly = GPU_HOURLY_USD.get(gpu)
    if hourly is None:
        return None
    return ServingCost(
        gpu=gpu,
        gpu_count=gpu_count,
        hourly_usd=hourly,
        parts_per_hour=parts_per_hour(parts, wall_seconds),
        source_run=source_run,
    )
