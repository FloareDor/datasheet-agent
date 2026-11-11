"""Grading an extraction against the catalogue labels.

Numeric fields are compared after normalising to base units, so "4.5 V/us" and
"4,500,000 V/s" are the same answer. Categorical fields (package, mounting style) are
compared on a canonical form, because catalogues and datasheets spell the same package
a dozen ways: "SOIC-8", "8-SOIC", "SO-8", "8 Pin SOIC".

Scoring counts only fields the label actually has. A part whose catalogue entry has no
slew rate is not evidence for or against the agent, so it is excluded rather than
counted wrong -- otherwise sparse labels would cap accuracy below 100% for everyone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from datasheet_agent.schema import FIELDS, FieldSpec
from datasheet_agent.units import clean_text, parse_quantity, quantities_match

# Package-family synonyms. The left side is what we canonicalise to.
_PACKAGE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "soic": ("soic", "so", "sop", "soic/so", "narrow soic", "r-pdso"),
    "tssop": ("tssop", "pw", "htssop"),
    "msop": ("msop", "mini-so", "vssop", "us8"),
    "sot23": ("sot23", "sot-23", "thin sot23", "sot"),
    "pdip": ("pdip", "dip", "plastic dip", "p-dip"),
    "qfn": ("qfn", "dfn", "lfcsp", "wson", "son"),
    "to92": ("to92", "to-92"),
    "bga": ("bga", "csp", "wlcsp", "dsbga", "ubga"),
    "sc70": ("sc70", "sc-70", "sot353"),
}

# Digits that belong to the family name itself, not to the pin count. Without this,
# "SOT-23-5" reads as a 23-pin part instead of a 5-pin one.
_FAMILY_DIGITS: dict[str, tuple[str, ...]] = {
    "sot23": ("23", "353"),
    "to92": ("92",),
    "sc70": ("70", "353"),
}

# Longest spelling first, globally: "so" is a SOIC synonym but also a substring of
# "tssop" and "sot23", so a shorter spelling must never win over a longer one.
_SPELLINGS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            (re.sub(r"[^a-z0-9]", "", spelling), canonical)
            for canonical, spellings in _PACKAGE_SYNONYMS.items()
            for spelling in spellings
        ),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
)

_SMD_WORDS = ("smd", "smt", "surface mount")
_THT_WORDS = ("through hole", "thru hole", "tht", "radial", "axial")


@dataclass(frozen=True)
class FieldOutcome:
    """How one field scored."""

    field: str
    predicted: str
    label: str
    correct: bool
    graded: bool
    """False when the label is missing, in which case the field is excluded."""

    @property
    def wrong(self) -> bool:
        return self.graded and not self.correct


@dataclass(frozen=True)
class PartScore:
    """How one datasheet scored, field by field."""

    part_number: str
    outcomes: tuple[FieldOutcome, ...]

    @property
    def graded(self) -> int:
        return sum(1 for o in self.outcomes if o.graded)

    @property
    def correct(self) -> int:
        return sum(1 for o in self.outcomes if o.correct and o.graded)

    @property
    def accuracy(self) -> float:
        return self.correct / self.graded if self.graded else 0.0

    def wrong_fields(self) -> tuple[FieldOutcome, ...]:
        return tuple(o for o in self.outcomes if o.wrong)


def canonical_package(raw: str | None) -> str:
    """Reduce a package name to `family-pincount`, e.g. "8-SOIC" -> "soic-8".

    Returns "" when nothing recognisable is present.
    """
    text = clean_text(raw).lower()
    if not text:
        return ""

    family = ""
    stripped = re.sub(r"[^a-z0-9]", "", text)
    for spelling, canonical in _SPELLINGS:
        if spelling in stripped:
            family = canonical
            break
    if not family:
        # Unknown family: keep the alphabetic stem so at least like compares with like.
        family = re.sub(r"[^a-z]", "", text)[:6]

    pins = _pin_count(text, family)
    return f"{family}-{pins}" if pins else family


def _pin_count(text: str, family: str) -> str:
    """Pull the pin count out of a package name, ignoring digits in the family name."""
    # An explicit "8-pin" / "8 lead" wins outright.
    explicit = re.search(r"(\d{1,3})\s*-?\s*(?:pin|lead|ld)\b", text)
    if explicit:
        return explicit.group(1)
    # Body dimensions ("QFN16 3x3", "DFN 4x4 mm") are millimetres, not pins.
    text = re.sub(r"\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?(?:\s*mm)?", " ", text)
    family_digits = _FAMILY_DIGITS.get(family, ())
    candidates = [n for n in re.findall(r"\d{1,3}", text) if n not in family_digits]
    if not candidates:
        return ""
    # "SOT-23-5" and "8-SOIC" both put the pin count last among the remaining digits.
    return candidates[-1]


def canonical_mounting(raw: str | None) -> str:
    """Reduce a mounting style to "smd", "tht" or ""."""
    # Separators vary ("Thru-Hole", "thru hole"), so flatten them to single spaces.
    text = re.sub(r"[^a-z0-9]+", " ", clean_text(raw).lower())
    if any(word in text for word in _SMD_WORDS):
        return "smd"
    if any(word in text for word in _THT_WORDS):
        return "tht"
    return ""


def grade_field(spec: FieldSpec, predicted: str | None, label: str | None) -> FieldOutcome:
    """Grade one field. A field with no label is recorded but not graded."""
    predicted_text = clean_text(predicted)
    label_text = clean_text(label)
    if not label_text:
        return FieldOutcome(spec.name, predicted_text, label_text, correct=False, graded=False)

    if spec.grader == "numeric":
        # The prediction may omit the unit the prompt asked for; the catalogue label
        # always carries its own, so only the prediction gets `assume_unit`.
        correct = quantities_match(
            parse_quantity(predicted_text, spec.dimension, assume_unit=spec.unit_hint),
            parse_quantity(label_text, spec.dimension),
        )
    elif spec.name == "mounting_style":
        canonical = canonical_mounting(predicted_text)
        correct = bool(canonical) and canonical == canonical_mounting(label_text)
    else:
        canonical = canonical_package(predicted_text)
        correct = bool(canonical) and canonical == canonical_package(label_text)

    return FieldOutcome(spec.name, predicted_text, label_text, correct=correct, graded=True)


def grade_part(
    part_number: str,
    predicted: dict[str, str | None],
    labels: dict[str, str | None],
) -> PartScore:
    """Grade every field of one part."""
    outcomes = tuple(
        grade_field(spec, predicted.get(spec.name), labels.get(spec.name)) for spec in FIELDS
    )
    return PartScore(part_number=part_number, outcomes=outcomes)


@dataclass(frozen=True)
class RunSummary:
    """Aggregate accuracy over a run, overall and per field."""

    parts: int
    graded: int
    correct: int
    per_field: dict[str, tuple[int, int]]
    """field -> (correct, graded)"""

    @property
    def accuracy(self) -> float:
        return self.correct / self.graded if self.graded else 0.0

    def field_accuracy(self, name: str) -> float:
        correct, graded = self.per_field.get(name, (0, 0))
        return correct / graded if graded else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "parts": self.parts,
            "graded": self.graded,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "per_field": {
                name: {
                    "correct": c,
                    "graded": g,
                    "accuracy": round(c / g, 4) if g else 0.0,
                }
                for name, (c, g) in sorted(self.per_field.items())
            },
        }


def summarise(scores: list[PartScore]) -> RunSummary:
    per_field: dict[str, tuple[int, int]] = {}
    for score in scores:
        for outcome in score.outcomes:
            correct, graded = per_field.get(outcome.field, (0, 0))
            per_field[outcome.field] = (
                correct + int(outcome.correct and outcome.graded),
                graded + int(outcome.graded),
            )
    return RunSummary(
        parts=len(scores),
        graded=sum(s.graded for s in scores),
        correct=sum(s.correct for s in scores),
        per_field=per_field,
    )
