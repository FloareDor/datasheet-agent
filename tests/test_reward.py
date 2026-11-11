"""The reward has to rank rollouts the way we would rank them by hand.

The grounding term exists because an earlier reward did not: trained on accuracy minus
tool cost, the policy converged to zero tool calls and guessed from priors. These tests
pin that failure shut.
"""

from __future__ import annotations

import pytest

from datasheet_agent.reward import TOOL_CALL_PENALTY, reward_value, score_rollout

DOCUMENT = (
    "ELECTRICAL CHARACTERISTICS TA = 25 C. SR Slew Rate AV = 1 21 V/us. "
    "Gain Bandwidth Product 4.5 MHz. Dual operational amplifier in an 8-lead SOIC."
)

LABELS = {
    "slew_rate": "21 V/us",
    "gain_bandwidth_product": "4.5 MHz",
    "num_channels": "2 Channel",
    "package_case": "SOIC-8",
}
PERFECT = {
    "slew_rate": "21 V/us",
    "gain_bandwidth_product": "4.5 MHz",
    "num_channels": "2",
    "package_case": "SOIC-8",
}
EVIDENCE = {
    "slew_rate": "SR Slew Rate AV = 1 21 V/us",
    "gain_bandwidth_product": "Gain Bandwidth Product 4.5 MHz",
    "num_channels": "Dual operational amplifier",
    "package_case": "8-lead SOIC",
}


def grounded(values=PERFECT, evidence=EVIDENCE, tool_calls=4) -> float:
    return reward_value(
        "X", values, LABELS, tool_calls, evidence=evidence, document_text=DOCUMENT
    )


def test_guessing_without_reading_earns_nothing() -> None:
    """The failure this term was added for: right answers, no tool calls, no reward.

    Trained without it, the policy stopped calling tools entirely and guessed ~30% of
    fields from priors.
    """
    assert reward_value("X", PERFECT, LABELS, tool_calls=0, document_text=DOCUMENT) <= 0.0


def test_reading_the_document_beats_guessing_the_same_answers() -> None:
    assert grounded() > reward_value(
        "X", PERFECT, LABELS, tool_calls=0, evidence=EVIDENCE, document_text=DOCUMENT
    )


def test_invented_evidence_earns_nothing_even_when_the_value_is_right() -> None:
    invented = dict.fromkeys(EVIDENCE, "Table 9: quantum flux coefficient 998 parsecs")
    assert grounded(evidence=invented) <= 0.0


def test_raw_accuracy_is_still_reported_so_the_gap_is_visible() -> None:
    breakdown, _ = score_rollout(
        "X", PERFECT, LABELS, tool_calls=0, evidence=EVIDENCE, document_text=DOCUMENT
    )
    assert breakdown.raw_accuracy == pytest.approx(1.0)
    assert breakdown.accuracy == 0.0
    assert breakdown.grounded_fraction == 0.0


def test_a_perfect_cheap_rollout_beats_a_perfect_expensive_one() -> None:
    assert grounded(tool_calls=3) > grounded(tool_calls=15)


def test_accuracy_outweighs_tool_cost() -> None:
    """One more correct field must be worth more than the calls to find it."""
    half = reward_value(
        "X",
        {"slew_rate": "21 V/us"},
        LABELS,
        tool_calls=2,
        evidence={"slew_rate": EVIDENCE["slew_rate"]},
        document_text=DOCUMENT,
    )
    assert grounded(tool_calls=10) > half


def test_answering_nothing_is_worse_than_answering_wrongly() -> None:
    silent = reward_value("X", {}, LABELS, tool_calls=2, document_text=DOCUMENT)
    wrong = reward_value(
        "X",
        {"slew_rate": "1 V/us"},
        LABELS,
        tool_calls=4,
        evidence={"slew_rate": EVIDENCE["slew_rate"]},
        document_text=DOCUMENT,
    )
    assert wrong > silent


def test_tool_cost_is_capped_so_one_runaway_rollout_cannot_dominate() -> None:
    assert grounded(tool_calls=20) == pytest.approx(grounded(tool_calls=200))


def test_breakdown_sums_to_the_total() -> None:
    breakdown, score = score_rollout(
        "X", PERFECT, LABELS, tool_calls=5, evidence=EVIDENCE, document_text=DOCUMENT
    )
    assert breakdown.total == pytest.approx(
        breakdown.accuracy - breakdown.tool_cost - breakdown.format_penalty
    )
    assert breakdown.tool_cost == pytest.approx(5 * TOOL_CALL_PENALTY)
    assert score.correct == score.graded == 4


def test_unlabelled_part_scores_zero_rather_than_failing() -> None:
    breakdown, score = score_rollout(
        "X", PERFECT, {}, tool_calls=2, evidence=EVIDENCE, document_text=DOCUMENT
    )
    assert score.graded == 0
    assert breakdown.accuracy == 0.0


def test_reward_stays_in_a_sane_range() -> None:
    """GRPO normalises within a group, so wild magnitudes destabilise training."""
    best = grounded(tool_calls=1)
    worst = reward_value("X", {}, LABELS, tool_calls=20, document_text=DOCUMENT)
    assert 0.0 < best <= 1.0
    assert -1.0 <= worst < 0.0
