"""Checking extracted values against the datasheet they came from.

Two stages, cheap first. Code checks catch the failures that do not need a model: a
value in the wrong unit, a channel count of 3, a supply voltage of 900 V, a value
offered with no quote at all, and the two that matter most -- an evidence quote that
appears nowhere in the datasheet, and a quote that does appear but does not state the
number reported beside it. Grounding alone only proves the line exists; it says
nothing about the value next to it, because the quote is text the model typed, not
text a tool handed over.

What survives goes to a model from a *different family* than the agent that produced
it. A model asked to check its own work tends to agree with it, so the verifier is
Phi-4 when the agents are Qwen. See https://arxiv.org/pdf/2402.08115v1.pdf.

Rejected fields go back to their own agent with the reason attached, for one retry.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from datasheet_agent.llm import Backend, Usage
from datasheet_agent.schema import FIELDS_BY_NAME, FieldSpec
from datasheet_agent.units import (
    Quantity,
    clean_text,
    parse_all_quantities,
    parse_quantity,
    quantities_match,
)

# Plausible ranges in base SI units. Deliberately wide: these catch nonsense
# (a 900 V op-amp supply), not merely unusual parts.
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "supply_voltage_max": (1.0, 100.0),
    "gain_bandwidth_product": (1e3, 1e10),
    "slew_rate": (1e3, 1e10),
    "input_offset_voltage": (1e-9, 1.0),
    "operating_temp_min": (-80.0, 30.0),
    "operating_temp_max": (40.0, 220.0),
    "num_channels": (1.0, 8.0),
}

VALID_CHANNEL_COUNTS = {1, 2, 3, 4, 8}
# Datasheets state the channel count as a word far more often than as a digit:
# "Quad Buffer", "Dual Operational Amplifier". A quote saying "Quad" does support
# the answer 4, and rejecting it would punish the correct reading.
CHANNEL_WORDS: dict[str, int] = {
    "single": 1,
    "mono": 1,
    "dual": 2,
    "triple": 3,
    "quad": 4,
    "quadruple": 4,
    "octal": 8,
}
MIN_EVIDENCE_CHARS = 8

# The two strict evidence checks are switchable, because measuring them meant
# running the pipeline with and without each one, and an ablation that needs a
# source edit is an ablation nobody can reproduce. Both default off: they were
# measured to cost accuracy. See the verifier table in the README.
REQUIRE_EVIDENCE = "DSA_REQUIRE_EVIDENCE"
REQUIRE_VALUE_IN_QUOTE = "DSA_REQUIRE_VALUE_IN_QUOTE"


def _enabled(flag: str) -> bool:
    return os.environ.get(flag, "0").strip().lower() in {"1", "true", "yes"}

# The tools prefix their output with "[page 7]", and models echo that prefix into
# the quote. It is a pointer, not quoted text, and its page number is not a
# measurement -- left in, it makes a bare "[page 1]" look like a grounded quote
# stating the value 1.
_PAGE_MARKER = re.compile(r"^\s*(?:\[\s*page\s*\d+\s*\]|page\s+\d+\s*[:.-])\s*", re.I)
EVIDENCE_TOKEN_THRESHOLD = 0.6


@dataclass(frozen=True)
class Rejection:
    """One field the verifier will not accept, and why."""

    field: str
    value: str
    reason: str
    stage: str
    """"code" or "model" -- which stage rejected it."""


@dataclass
class VerificationReport:
    rejections: list[Rejection] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    checked: int = 0

    @property
    def rejected_fields(self) -> tuple[str, ...]:
        return tuple(r.field for r in self.rejections)

    def reason_for(self, name: str) -> str:
        for rejection in self.rejections:
            if rejection.field == name:
                return rejection.reason
        return ""

    def to_dict(self, model: str = "") -> dict[str, object]:
        return {
            "checked": self.checked,
            "rejected": [
                {"field": r.field, "value": r.value, "reason": r.reason, "stage": r.stage}
                for r in self.rejections
            ],
            "usage": self.usage.to_dict(model),
        }


# ---- stage 1: code checks ----------------------------------------------


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def quoted_line(evidence: str) -> str:
    """The quoted text itself, with any page pointer stripped off the front."""
    return _PAGE_MARKER.sub("", clean_text(evidence)).strip()


def evidence_is_grounded(evidence: str, document_text: str) -> bool:
    """True when the quoted evidence really comes from this datasheet.

    An exact match is too strict -- models normalise whitespace, drop subscripts and
    re-flow columns when they quote a table row. So the check is on tokens: most of
    the quote's tokens must occur in the document.
    """
    quote = clean_text(evidence)
    if len(quote) < MIN_EVIDENCE_CHARS:
        return False
    quote_tokens = _tokens(quote)
    if not quote_tokens:
        return False
    document_tokens = set(_tokens(document_text))
    present = sum(1 for token in quote_tokens if token in document_tokens)
    return present / len(quote_tokens) >= EVIDENCE_TOKEN_THRESHOLD


def value_appears_in_quote(spec: FieldSpec, value: str, evidence: str) -> bool | None:
    """Whether the reported value is one of the numbers in its own quote.

    Grounding proves the quoted line exists in the datasheet. It does not connect the
    line to the number reported beside it, so a real table row can be quoted under a
    value it never states -- an error the token-overlap check cannot see, because every
    token of the quote really is in the document.

    Returns None when the quote offers nothing comparable (no parseable number, or an
    unreadable value), because the absence of a number is not evidence of a mismatch.

    This catches a value detached from its quote. It cannot catch the right number read
    off the wrong parameter -- a transition frequency quoted as a gain-bandwidth product
    passes, because the number really is on the line. That one needs the model stage.
    """
    reported = parse_quantity(value, spec.dimension, assume_unit=spec.unit_hint)
    if reported is None:
        return None
    candidates = parse_all_quantities(evidence, spec.dimension, assume_unit=spec.unit_hint)
    if spec.name == "num_channels":
        words = set(_tokens(evidence))
        candidates += [
            Quantity(float(count), spec.dimension)
            for word, count in CHANNEL_WORDS.items()
            if word in words
        ]
    if not candidates:
        return None

    for candidate in candidates:
        if quantities_match(candidate, reported):
            return True
        # A dual-supply row states "+-18 V" for a part whose total supply is 36 V.
        if spec.name == "supply_voltage_max" and quantities_match(
            Quantity(abs(candidate.value) * 2, candidate.dimension), reported
        ):
            return True
        # A range may carry its sign on the range rather than the number:
        # "40 to 125 C" for a minimum of -40.
        if spec.dimension == "temperature" and quantities_match(
            Quantity(-candidate.value, candidate.dimension), reported
        ):
            return True
    return False


def code_check(spec: FieldSpec, value: str, evidence: str, document_text: str) -> str:
    """Return a rejection reason, or "" when the value passes every code check."""
    text = clean_text(value)
    if not text:
        return ""  # an honest null is not an error; it is simply not an answer

    if spec.grader == "numeric":
        # `assume_unit` for the same reason grading uses it: the prompt names the unit
        # to answer in, so a bare "36" for a volts field is 36 V. Without this the
        # verifier rejects correct answers for their formatting and sends the agent
        # back to re-find a value it already had.
        quantity = parse_quantity(text, spec.dimension, assume_unit=spec.unit_hint)
        if quantity is None:
            return (
                f"{text!r} could not be read as a {spec.dimension} value. "
                f"Answer in {spec.unit_hint}."
            )
        low, high = PLAUSIBLE.get(spec.name, (float("-inf"), float("inf")))
        if not low <= quantity.value <= high:
            return (
                f"{text!r} is {quantity} in base units, outside the plausible range "
                f"{low:g} to {high:g} for this field. Check the unit and the column."
            )
        if spec.name == "num_channels" and int(quantity.value) not in VALID_CHANNEL_COUNTS:
            return f"{text!r} is not a real channel count for an op-amp."

    quote = quoted_line(evidence)
    if not quote:
        if _enabled(REQUIRE_EVIDENCE):
            return (
                "No evidence was quoted for this value. A page number is not a quote: "
                "quote the datasheet line you took the value from, so it can be checked."
            )
        return ""

    if not evidence_is_grounded(quote, document_text):
        return (
            "The quoted evidence does not appear in this datasheet. Re-read the page "
            "and quote the line you took the value from, exactly."
        )

    if (
        _enabled(REQUIRE_VALUE_IN_QUOTE)
        and spec.grader == "numeric"
        and value_appears_in_quote(spec, text, quote) is False
    ):
        return (
            f"The line you quoted does not state {text!r}. Either quote the line the "
            f"value actually came from, or report the value that line states."
        )
    return ""


# ---- stage 2: the cross-family model check -----------------------------

VERIFIER_SYSTEM = (
    "You audit specs extracted from component datasheets. You are given a field, the "
    "value an extraction agent reported, and the datasheet line it says the value came "
    "from. You decide only whether the evidence supports the value. You do not re-"
    "extract, and you do not reject a value merely because you would have phrased it "
    "differently."
)

_VERIFIER_CONTRACT = """\
Reply with JSON only:
{"verdict": "accept" | "reject", "reason": "<one sentence, only if rejecting>"}

Accept when the evidence line states this value for this field, allowing for unit
conversion (4.5 V/us and 4,500,000 V/s are the same value) and for a total supply
voltage reported from a +-18 V dual-supply spec.
Reject when the evidence states a different value, is about a different parameter, or
is about a different part in a family datasheet."""


def model_check(
    backend: Backend,
    part_number: str,
    spec: FieldSpec,
    value: str,
    evidence: str,
) -> tuple[bool, str, Usage]:
    """Ask the cross-family verifier whether the evidence supports the value."""
    prompt = (
        f"Part: {part_number}\n"
        f"Field: {spec.name} ({spec.unit_hint}) - {spec.description}\n"
        f"Reported value: {value}\n"
        f"Evidence the agent quoted: {evidence or '(none given)'}\n\n"
        f"{_VERIFIER_CONTRACT}"
    )
    reply = backend.chat(VERIFIER_SYSTEM, [{"role": "user", "content": prompt}], max_tokens=200)
    match = re.search(r"\{.*\}", reply.text or "", re.DOTALL)
    if match is None:
        # An unreadable verdict must not silently reject a good value.
        return True, "", reply.usage
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return True, "", reply.usage
    accepted = str(payload.get("verdict", "accept")).lower() != "reject"
    return accepted, str(payload.get("reason", "")).strip(), reply.usage


# ---- the verifier -------------------------------------------------------


def verify(
    values: dict[str, str],
    evidence: dict[str, str],
    document_text: str,
    part_number: str,
    backend: Backend | None = None,
    fields: tuple[str, ...] | None = None,
) -> VerificationReport:
    """Run code checks, then the model check on whatever survives.

    `backend` is optional so the code-only verifier can be measured on its own -- the
    plan's with/without-verifier comparison needs both halves separable.
    """
    report = VerificationReport()
    names = fields if fields is not None else tuple(values)

    for name in names:
        spec = FIELDS_BY_NAME.get(name)
        if spec is None:
            continue
        value = clean_text(values.get(name))
        if not value:
            continue
        report.checked += 1
        quote = evidence.get(name, "")

        reason = code_check(spec, value, quote, document_text)
        if reason:
            report.rejections.append(Rejection(name, value, reason, "code"))
            continue

        if backend is not None:
            accepted, model_reason, usage = model_check(backend, part_number, spec, value, quote)
            report.usage.add(usage)
            if not accepted:
                report.rejections.append(
                    Rejection(
                        name,
                        value,
                        model_reason or "The verifier judged the evidence unsupportive.",
                        "model",
                    )
                )
    return report


def retry_prompt(part_number: str, rejections: list[Rejection]) -> str:
    """The message a field agent gets when its values come back rejected."""
    lines = "\n".join(f"- {r.field}: you answered {r.value!r}. {r.reason}" for r in rejections)
    return (
        f"A verifier rejected some of your values for {part_number}:\n\n{lines}\n\n"
        "Re-read the datasheet for these fields only and correct them. Use the tools "
        "again if you need to. If after re-reading the datasheet genuinely does not "
        "state a value, answer null -- that is a correct answer, not a failure. "
        "Reply with the same JSON shape, containing only the fields listed above."
    )
