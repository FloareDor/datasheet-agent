"""The search tool has to recover from the queries models actually write."""

from __future__ import annotations

from datasheet_agent.tools import _query_variants


def test_machine_names_relax_towards_datasheet_wording() -> None:
    """A model reaching for `slew_rate` must still find "Slew Rate"."""
    variants = _query_variants("slew_rate")
    assert variants[0] == "slew_rate"
    assert "slew rate" in variants


def test_trailing_qualifiers_are_dropped_last() -> None:
    variants = _query_variants("supply_voltage_max")
    assert variants.index("supply voltage max") < variants.index("supply voltage")


def test_a_good_query_is_tried_first_and_unchanged() -> None:
    assert _query_variants("Slew Rate")[0] == "Slew Rate"


def test_variants_are_unique_and_non_empty() -> None:
    for query in ("slew_rate", "VOS", "a_b_c", "  padded  "):
        variants = _query_variants(query)
        assert len(variants) == len(set(variants))
        assert all(v.strip() for v in variants)


def test_single_word_query_has_no_looser_form() -> None:
    assert _query_variants("VOS") == ["VOS"]


def test_field_hints_are_tried_before_blind_shortening() -> None:
    """Blind relaxation is not just weaker than the real vocabulary, it is harmful.

    `operating_temp_min` shortens to "operating temp", which substring-matches
    "Full Temperature" inside a distortion spec and returns a number in percent.
    The field's own wording has to be tried first.
    """
    from pathlib import Path

    from datasheet_agent.tools import DatasheetToolbox

    toolbox = DatasheetToolbox(
        pdf_path=Path("unused.pdf"),
        cache_dir=Path("unused"),
        term_hints={"operating_temp_min": ("Operating Temperature Range", "TA")},
    )
    candidates = toolbox._candidates("operating_temp_min")
    assert candidates[0] == "operating_temp_min"
    assert candidates[1] == "Operating Temperature Range"
    assert candidates.index("Operating Temperature Range") < candidates.index("operating temp")


def test_hint_lookup_ignores_separators_and_case() -> None:
    from pathlib import Path

    from datasheet_agent.tools import DatasheetToolbox

    toolbox = DatasheetToolbox(
        pdf_path=Path("unused.pdf"),
        cache_dir=Path("unused"),
        term_hints={"slew_rate": ("Slew Rate", "SR")},
    )
    assert "Slew Rate" in toolbox._candidates("Slew-Rate")
    assert "Slew Rate" in toolbox._candidates("slewrate")
