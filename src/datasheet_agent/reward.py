"""The reward an extraction rollout earns.

**A field only counts if it is grounded**: correct against the label *and* supported by
evidence that actually appears in this datasheet, from a rollout that read the document
at all. That is not fastidiousness, it is the fix for a measured failure.

The first version rewarded accuracy minus a small per-tool-call cost. Trained on it, the
policy converged to **zero tool calls**: Qwen2.5-1.5B can guess roughly 30% of these
fields from priors alone -- SOIC-8 is the common package, -40/+85 the common temperature
range, a "2" in the part number often means two channels -- and guessing at zero cost
beats tool use at the same accuracy plus the cost. The reward was not wrong about what it
asked for; it was wrong about what it left unsaid.

Grounding closes that without a special case. A rollout that calls no tools has read no
document text, so it can quote nothing from the document, so nothing it says is grounded,
so its accuracy term is zero. Guessing becomes worthless rather than merely discouraged,
and the reward now states the actual task: extract values from *this* datasheet, do not
recall what op-amps usually look like.

The terms, in order of weight:

1. **Grounded accuracy** - the fraction of labelled fields both correct and evidenced.
   This is the task, and it dominates.
2. **Tool cost** - a small penalty per call, so the policy does not simply read the whole
   document. The premise of an agent is that it is cheaper than pasting everything in.
3. **Format** - a penalty for returning nothing parseable. The 1.5B baseline failed to
   emit JSON on 8-10% of parts, so this is common enough to be worth shaping.

Reward stays in roughly [-1, 1] so GRPO's group-relative advantage remains well-scaled
and no single term can swamp the others.
"""

from __future__ import annotations

from dataclasses import dataclass

from datasheet_agent.grading import PartScore, grade_part
from datasheet_agent.verify import evidence_is_grounded

# Per tool call. Chosen so the full ten-step budget costs 0.2 -- enough that a wasted
# call is never free, small enough that the policy will still pay for a call that
# turns one field from wrong to right (worth ~0.11 at nine fields).
TOOL_CALL_PENALTY = 0.02
NO_ANSWER_PENALTY = 0.5
MAX_PENALISED_CALLS = 20


@dataclass(frozen=True)
class RewardBreakdown:
    """The reward and its parts, so training curves can be read."""

    accuracy: float
    """Fraction of labelled fields both correct and grounded. The term that is optimised."""

    raw_accuracy: float
    """Fraction correct ignoring grounding. Logged, so the gap guessing creates is visible."""

    grounded_fraction: float
    """Of the fields answered, how many were evidenced from this document."""

    tool_cost: float
    format_penalty: float

    @property
    def total(self) -> float:
        return self.accuracy - self.tool_cost - self.format_penalty

    def to_dict(self) -> dict[str, float]:
        return {
            "reward": round(self.total, 4),
            "accuracy": round(self.accuracy, 4),
            "raw_accuracy": round(self.raw_accuracy, 4),
            "grounded_fraction": round(self.grounded_fraction, 4),
            "tool_cost": round(self.tool_cost, 4),
            "format_penalty": round(self.format_penalty, 4),
        }


def score_rollout(
    part_number: str,
    values: dict[str, str],
    labels: dict[str, str],
    tool_calls: int,
    tool_call_penalty: float = TOOL_CALL_PENALTY,
    evidence: dict[str, str] | None = None,
    document_text: str = "",
) -> tuple[RewardBreakdown, PartScore]:
    """Reward one rollout, and return the score it came from for logging."""
    score = grade_part(part_number, values, labels)

    # A part with no labels at all cannot teach anything; give it zero rather than a
    # spurious 0/0 = 0.0 that would look like total failure.
    raw_accuracy = score.accuracy if score.graded else 0.0

    quotes = evidence or {}
    # A rollout that read nothing can ground nothing, whatever it claims to quote.
    can_ground = bool(document_text) and tool_calls > 0
    grounded = (
        {name for name, quote in quotes.items() if evidence_is_grounded(quote, document_text)}
        if can_ground
        else set()
    )

    grounded_correct = sum(
        1 for o in score.outcomes if o.graded and o.correct and o.field in grounded
    )
    accuracy = grounded_correct / score.graded if score.graded else 0.0

    answered = [name for name, value in values.items() if value.strip()]
    grounded_fraction = (
        len([name for name in answered if name in grounded]) / len(answered) if answered else 0.0
    )

    return (
        RewardBreakdown(
            accuracy=accuracy,
            raw_accuracy=raw_accuracy,
            grounded_fraction=grounded_fraction,
            tool_cost=tool_call_penalty * min(tool_calls, MAX_PENALISED_CALLS),
            format_penalty=0.0 if answered else NO_ANSWER_PENALTY,
        ),
        score,
    )


def reward_value(
    part_number: str,
    values: dict[str, str],
    labels: dict[str, str],
    tool_calls: int,
    evidence: dict[str, str] | None = None,
    document_text: str = "",
) -> float:
    """Just the scalar, for a trainer that wants one number."""
    breakdown, _ = score_rollout(
        part_number,
        values,
        labels,
        tool_calls,
        evidence=evidence,
        document_text=document_text,
    )
    return breakdown.total
