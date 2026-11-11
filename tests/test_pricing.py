"""Cost per datasheet, for models billed by GPU time rather than per token."""

from __future__ import annotations

import pytest

from datasheet_agent.pricing import (
    GPU_HOURLY_USD,
    GPU_PER_SECOND_USD,
    parts_per_hour,
    serving_cost,
)


def test_hourly_rates_derive_from_the_published_per_second_ones() -> None:
    for name, per_second in GPU_PER_SECOND_USD.items():
        assert GPU_HOURLY_USD[name] == pytest.approx(per_second * 3600, rel=1e-6)


def test_throughput_is_measured_not_assumed() -> None:
    assert parts_per_hour(100, 90.0) == pytest.approx(4000.0)
    assert parts_per_hour(100, 0.0) == 0.0


def test_two_gpus_cost_twice_as_much_per_part() -> None:
    """Tensor parallelism rents both cards for the whole run."""
    one = serving_cost("H100", 1, 100, 600.0, "run")
    two = serving_cost("H100", 2, 100, 600.0, "run")
    assert one and two
    assert two.usd_per_part == pytest.approx(2 * one.usd_per_part)


def test_faster_serving_is_cheaper_per_part() -> None:
    slow = serving_cost("L40S", 1, 100, 600.0, "run")
    fast = serving_cost("L40S", 1, 100, 60.0, "run")
    assert slow and fast
    assert fast.usd_per_part < slow.usd_per_part


def test_an_unknown_gpu_returns_none_rather_than_inventing_a_price() -> None:
    assert serving_cost("GTX-1080", 1, 100, 600.0, "run") is None


def test_cost_records_which_run_the_throughput_came_from() -> None:
    cost = serving_cost("L40S", 1, 100, 90.0, "v3-singlecall")
    assert cost is not None
    assert cost.to_dict()["source_run"] == "v3-singlecall"
    assert cost.to_dict()["gpu"] == "L40S x1"
