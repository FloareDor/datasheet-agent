"""The verifier must catch wrong values without rejecting right ones."""

from __future__ import annotations

import pytest

from datasheet_agent.schema import FIELDS_BY_NAME
from datasheet_agent.verify import code_check, evidence_is_grounded, verify

DOCUMENT = (
    "ELECTRICAL CHARACTERISTICS TA = 25 C. SR Slew Rate AV = 1 0.4 V/us. "
    "Supply Voltage 36 V. Operating Temperature Range -40 to 105 C. "
    "Input Offset Voltage VOS 3 mV. Number of Channels 3."
)


def test_a_bare_number_in_the_asked_for_unit_is_accepted() -> None:
    """The prompt names the unit, so "36" to a volts field is 36 V, not a format error."""
    spec = FIELDS_BY_NAME["supply_voltage_max"]
    assert code_check(spec, "36", "Supply Voltage 36 V", DOCUMENT) == ""
    offset = FIELDS_BY_NAME["input_offset_voltage"]
    assert code_check(offset, "3", "Input Offset Voltage VOS 3 mV", DOCUMENT) == ""


def test_an_implausible_value_is_rejected() -> None:
    spec = FIELDS_BY_NAME["supply_voltage_max"]
    assert "plausible range" in code_check(spec, "900 V", "Supply Voltage 36 V", DOCUMENT)


def test_an_impossible_channel_count_is_rejected() -> None:
    spec = FIELDS_BY_NAME["num_channels"]
    quote = "Number of Channels 3."
    assert code_check(spec, "3", quote, DOCUMENT) == ""  # 3 is unusual but real
    assert code_check(spec, "7", quote, DOCUMENT) != ""


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


# ---- the quote must actually state the value ---------------------------
#
# These exist because grounding was the only evidence check for the whole of the first
# eval, and grounding is token overlap against the document -- not against the number.
# A real table row quoted under a number it never states passed every code check.

from datasheet_agent.verify import (  # noqa: E402
    REQUIRE_EVIDENCE,
    REQUIRE_VALUE_IN_QUOTE,
    quoted_line,
    value_appears_in_quote,
)


@pytest.fixture
def strict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both strict evidence checks on. They ship off; see the README ablation."""
    monkeypatch.setenv(REQUIRE_EVIDENCE, "1")
    monkeypatch.setenv(REQUIRE_VALUE_IN_QUOTE, "1")

SLEW = FIELDS_BY_NAME["slew_rate"]
SUPPLY = FIELDS_BY_NAME["supply_voltage_max"]
TMIN = FIELDS_BY_NAME["operating_temp_min"]
GBP = FIELDS_BY_NAME["gain_bandwidth_product"]


def test_a_value_on_its_quoted_line_is_supported() -> None:
    assert value_appears_in_quote(SLEW, "20", "Slew Rate SR G = +1, 10-V step 20 V/us") is True


def test_a_value_absent_from_its_quoted_line_is_not() -> None:
    """The failure this check exists for: a real row, a number from somewhere else."""
    assert value_appears_in_quote(SLEW, "20", "Slew Rate SR G = +1 0.5 V/us") is False


def test_unit_conversion_still_counts_as_present() -> None:
    assert value_appears_in_quote(GBP, "1.1 MHz", "Gain Bandwidth Product GBW 1100 kHz") is True


def test_a_dual_supply_row_supports_the_total() -> None:
    """A +-18 V part is a 36 V part; rejecting that would destroy correct answers."""
    assert value_appears_in_quote(SUPPLY, "36", "Supply Voltage, V+ to V- +-18 V") is True


def test_a_range_quoted_without_the_sign_supports_its_minimum() -> None:
    assert value_appears_in_quote(TMIN, "-40", "Operating Temperature 40 to 125 C") is True


def test_a_quote_with_no_numbers_proves_nothing() -> None:
    """Absence of a number is not evidence of a mismatch, so this must not reject."""
    assert value_appears_in_quote(SLEW, "20", "Slew Rate is specified over temperature") is None


def test_the_right_number_from_the_wrong_parameter_still_passes() -> None:
    """The limit of this check, stated as a test so it is not mistaken for coverage.

    A transition frequency quoted as a gain-bandwidth product is the error the negative
    control found, and the number really is on the line -- so only the model stage can
    see it.
    """
    assert value_appears_in_quote(GBP, "300 MHz", "Transition Frequency fT 300 MHz") is True


def test_code_check_rejects_a_value_its_quote_does_not_state(strict: None) -> None:
    document = "Slew Rate SR G = +1 0.5 V/us Gain Bandwidth Product 1.1 MHz"
    reason = code_check(SLEW, "20", "Slew Rate SR G = +1 0.5 V/us", document)
    assert "does not state" in reason


def test_code_check_rejects_a_value_offered_with_no_quote(strict: None) -> None:
    """An unevidenced value is unverifiable, which is a rejection, not a pass."""
    reason = code_check(SLEW, "20", "", "Slew Rate SR 20 V/us")
    assert "No evidence" in reason


def test_a_null_with_no_quote_is_still_not_an_error(strict: None) -> None:
    assert code_check(SLEW, "", "", "anything") == ""


def test_a_page_pointer_is_not_a_quote(strict: None) -> None:
    """Models echo the tool's "[page 1]" prefix. Alone, it quotes nothing.

    Left unstripped it was worse than useless: its page number parsed as a measurement,
    so "[page 1]" read as a grounded quote stating the value 1.
    """
    assert quoted_line("[page 7] Slew Rate SR 0.4 V/us") == "Slew Rate SR 0.4 V/us"
    assert quoted_line("[page 1]") == ""
    assert "not a quote" in code_check(SLEW, "0.4 V/us", "[page 1]", DOCUMENT)


def test_a_real_quote_behind_a_page_pointer_still_passes() -> None:
    assert code_check(SLEW, "0.4 V/us", "[page 2] SR Slew Rate AV = 1 0.4 V/us", DOCUMENT) == ""


def test_a_channel_count_written_as_a_word_supports_the_number() -> None:
    """Datasheets say "Quad Buffer", not "4 channels"."""
    channels = FIELDS_BY_NAME["num_channels"]
    quote = "Low Power, Precision FET Input Quad Buffer"
    assert value_appears_in_quote(channels, "4", quote) is True
    assert value_appears_in_quote(channels, "2", quote) is False


def test_the_strict_checks_are_off_unless_asked_for() -> None:
    """They ship off because they were measured to cost accuracy, not because of a
    default nobody chose. A silent flip of this is a silent regression."""
    assert code_check(SLEW, "20", "", DOCUMENT) == ""
    assert code_check(SLEW, "20", "SR Slew Rate AV = 1 0.4 V/us", DOCUMENT) == ""
