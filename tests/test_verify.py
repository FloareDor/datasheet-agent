"""The verifier must catch wrong values without rejecting right ones."""

from __future__ import annotations

from datasheet_agent.schema import FIELDS_BY_NAME
from datasheet_agent.verify import code_check, evidence_is_grounded, verify

DOCUMENT = (
    "ELECTRICAL CHARACTERISTICS TA = 25 C. SR Slew Rate AV = 1 0.4 V/us. "
    "Supply Voltage 36 V. Operating Temperature Range -40 to 105 C."
)


def test_a_bare_number_in_the_asked_for_unit_is_accepted() -> None:
    """The prompt names the unit, so "36" to a volts field is 36 V, not a format error."""
    spec = FIELDS_BY_NAME["supply_voltage_max"]
    assert code_check(spec, "36", "Supply Voltage 36 V", DOCUMENT) == ""
    assert code_check(FIELDS_BY_NAME["input_offset_voltage"], "3", "", DOCUMENT) == ""


def test_an_implausible_value_is_rejected() -> None:
    spec = FIELDS_BY_NAME["supply_voltage_max"]
    assert "plausible range" in code_check(spec, "900 V", "", DOCUMENT)


def test_an_impossible_channel_count_is_rejected() -> None:
    spec = FIELDS_BY_NAME["num_channels"]
    assert code_check(spec, "3", "", DOCUMENT) == ""  # 3 is unusual but real
    assert code_check(spec, "7", "", DOCUMENT) != ""


def test_invented_evidence_is_caught() -> None:
    spec = FIELDS_BY_NAME["slew_rate"]
    invented = "Table 7: Zylophone Coefficient quantum flux 998 parsecs"
    assert "does not appear" in code_check(spec, "0.4 V/us", invented, DOCUMENT)


def test_real_evidence_passes_despite_reflowing() -> None:
    """Models re-flow table rows when they quote them; exact matching is too strict."""
    assert evidence_is_grounded("SR Slew Rate 0.4 V/us", DOCUMENT)
    assert evidence_is_grounded("Supply Voltage 36 V", DOCUMENT)


def test_an_empty_answer_is_not_an_error() -> None:
    """Declining to answer is a correct response, not something to retry."""
    spec = FIELDS_BY_NAME["slew_rate"]
    assert code_check(spec, "", "", DOCUMENT) == ""


def test_verify_reports_only_what_it_rejected() -> None:
    report = verify(
        values={"supply_voltage_max": "36", "num_channels": "7"},
        evidence={"supply_voltage_max": "Supply Voltage 36 V"},
        document_text=DOCUMENT,
        part_number="X",
        backend=None,
    )
    assert report.checked == 2
    assert report.rejected_fields == ("num_channels",)
    assert "channel count" in report.reason_for("num_channels")


def test_unlabelled_or_empty_values_are_not_checked() -> None:
    report = verify({"slew_rate": ""}, {}, DOCUMENT, "X", backend=None)
    assert report.checked == 0
    assert not report.rejections
