"""Grading must be generous about spelling and strict about values."""

from __future__ import annotations

import pytest

from datasheet_agent.grading import (
    canonical_mounting,
    canonical_package,
    grade_field,
    grade_part,
    summarise,
)
from datasheet_agent.schema import FIELDS_BY_NAME


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("SOIC-8", "8-SOIC"),
        ("SOIC-8", "SO-8"),
        ("SOIC-8", "8 Pin SOIC"),
        ("TSSOP-14", "14-TSSOP"),
        ("SOT-23-5", "5-Lead SOT-23"),
        ("PDIP-8", "8-DIP"),
        ("MSOP-8", "8-VSSOP"),
    ],
)
def test_package_spellings_agree(a: str, b: str) -> None:
    assert canonical_package(a) == canonical_package(b) != ""


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("SOIC-8", "SOIC-14"),
        ("SOIC-8", "TSSOP-8"),
        ("SOT-23-5", "SOT-23-6"),
    ],
)
def test_different_packages_disagree(a: str, b: str) -> None:
    assert canonical_package(a) != canonical_package(b)


def test_sot23_family_digits_are_not_the_pin_count() -> None:
    assert canonical_package("SOT-23-5") == "sot23-5"
    assert canonical_package("TO-92-3") == "to92-3"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SMD/SMT", "smd"),
        ("Surface Mount", "smd"),
        ("Through Hole", "tht"),
        ("Thru-Hole", "tht"),
        ("", ""),
    ],
)
def test_mounting_canonicalisation(raw: str, expected: str) -> None:
    assert canonical_mounting(raw) == expected


def test_missing_label_is_excluded_not_counted_wrong() -> None:
    spec = FIELDS_BY_NAME["slew_rate"]
    outcome = grade_field(spec, "21 V/us", label="")
    assert not outcome.graded
    assert not outcome.wrong


def test_missing_prediction_against_real_label_is_wrong() -> None:
    spec = FIELDS_BY_NAME["slew_rate"]
    outcome = grade_field(spec, None, label="21 V/us")
    assert outcome.graded and outcome.wrong


def test_numeric_field_ignores_unit_spelling() -> None:
    spec = FIELDS_BY_NAME["gain_bandwidth_product"]
    assert grade_field(spec, "4500 kHz", "4.5 MHz").correct


def test_summary_excludes_ungraded_fields_from_the_denominator() -> None:
    labels = {"slew_rate": "21 V/us", "num_channels": "2 Channel"}
    predicted = {"slew_rate": "21 V/us", "num_channels": "4"}
    score = grade_part("OPA2991", predicted, labels)
    assert score.graded == 2
    assert score.correct == 1
    assert score.accuracy == pytest.approx(0.5)

    summary = summarise([score])
    assert summary.graded == 2
    assert summary.accuracy == pytest.approx(0.5)
    assert summary.field_accuracy("slew_rate") == 1.0
    assert summary.field_accuracy("num_channels") == 0.0
    # A field nobody had a label for contributes nothing at all.
    assert summary.field_accuracy("package_case") == 0.0
    assert summary.per_field["package_case"] == (0, 0)


def test_wrong_fields_are_reportable_for_the_verifier_retry() -> None:
    score = grade_part("X", {"slew_rate": "1 V/us"}, {"slew_rate": "21 V/us"})
    assert [o.field for o in score.wrong_fields()] == ["slew_rate"]


def test_body_dimensions_are_not_pin_counts() -> None:
    """"QFN16 3x3" is a 16-pin part in a 3mm x 3mm body, not a 3-pin part."""
    assert canonical_package("QFN16 3x3") == canonical_package("QFN-16")
    assert canonical_package("DFN 4x4 mm 8-lead") == canonical_package("DFN-8")
