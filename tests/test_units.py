"""Unit parsing is what makes grading fair, so it is tested hardest."""

from __future__ import annotations

import pytest

from datasheet_agent.units import clean_text, parse_quantity, quantities_match


@pytest.mark.parametrize(
    ("raw", "dimension", "expected"),
    [
        ("16 V", "voltage", 16.0),
        ("0.3 mV", "voltage", 3e-4),
        ("125 uV", "voltage", 125e-6),
        ("125 µV", "voltage", 125e-6),
        ("1.1 MHz", "frequency", 1.1e6),
        ("900 kHz", "frequency", 9e5),
        ("4.5 V/us", "slew_rate", 4.5e6),
        ("4,500,000 V/s", "slew_rate", 4.5e6),
        ("90 dB", "gain", 90.0),
        ('="10 pA"', "current", 1e-11),
        ("80 mA", "current", 0.08),
        ("- 40 C", "temperature", -40.0),
        ("+ 125 C", "temperature", 125.0),
        ("-40°C", "temperature", -40.0),
        ("1 Channel", "count", 1.0),
        ("4 Channels", "count", 4.0),
    ],
)
def test_parses_catalogue_and_datasheet_spellings(
    raw: str, dimension: str, expected: float
) -> None:
    quantity = parse_quantity(raw, dimension)
    assert quantity is not None, f"failed to parse {raw!r}"
    assert quantity.value == pytest.approx(expected, rel=1e-9)


def test_prefix_case_distinguishes_milli_from_mega() -> None:
    """The classic datasheet trap: mV is milli, MHz is mega."""
    millivolt = parse_quantity("5 mV", "voltage")
    megahertz = parse_quantity("5 MHz", "frequency")
    assert millivolt is not None and millivolt.value == pytest.approx(5e-3)
    assert megahertz is not None and megahertz.value == pytest.approx(5e6)


def test_unparseable_returns_none_not_zero() -> None:
    """A missing value must not be confused with a value of zero."""
    assert parse_quantity("", "voltage") is None
    assert parse_quantity("N/A", "voltage") is None
    assert parse_quantity(None, "voltage") is None
    assert parse_quantity("0 V", "voltage") is not None


def test_clean_text_strips_spreadsheet_armour() -> None:
    assert clean_text('="10 pA"') == "10 pA"
    assert clean_text("  a  b ") == "a b"


def test_match_tolerates_unit_differences_but_not_wrong_values() -> None:
    per_us = parse_quantity("4.5 V/us", "slew_rate")
    per_s = parse_quantity("4500000 V/s", "slew_rate")
    other = parse_quantity("21 V/us", "slew_rate")
    assert quantities_match(per_us, per_s)
    assert not quantities_match(per_us, other)


def test_match_requires_same_dimension() -> None:
    assert not quantities_match(parse_quantity("5 V", "voltage"), parse_quantity("5 dB", "gain"))


def test_temperature_uses_absolute_tolerance() -> None:
    """Relative tolerance is meaningless near 0 degC, so temperature gets +-2 degC."""
    assert quantities_match(
        parse_quantity("0 C", "temperature"), parse_quantity("1 C", "temperature")
    )
    assert not quantities_match(
        parse_quantity("0 C", "temperature"), parse_quantity("10 C", "temperature")
    )


def test_missing_value_never_matches() -> None:
    assert not quantities_match(None, parse_quantity("5 V", "voltage"))
    assert not quantities_match(None, None)


def test_bare_number_is_read_in_the_unit_that_was_asked_for() -> None:
    """The prompt names the answer unit, so "36" to a volts question means 36 V.

    Without this, grading measures formatting rather than extraction.
    """
    assert parse_quantity("36", "voltage", assume_unit="V").value == pytest.approx(36.0)
    assert parse_quantity("0.4", "slew_rate", assume_unit="V/us").value == pytest.approx(4e5)
    assert parse_quantity("0.9", "frequency", assume_unit="MHz").value == pytest.approx(9e5)
    assert parse_quantity("4", "voltage", assume_unit="mV").value == pytest.approx(4e-3)


def test_an_explicit_unit_always_wins_over_the_assumed_one() -> None:
    quantity = parse_quantity("4500 kHz", "frequency", assume_unit="MHz")
    assert quantity is not None and quantity.value == pytest.approx(4.5e6)


def test_assume_unit_does_not_invent_a_value_from_nothing() -> None:
    assert parse_quantity("", "voltage", assume_unit="V") is None
    assert parse_quantity("N/A", "voltage", assume_unit="V") is None
