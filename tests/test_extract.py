"""Response parsing has to survive whatever a small model emits."""

from __future__ import annotations

from datasheet_agent.extract import Extraction, parse_response
from datasheet_agent.llm import Usage
from datasheet_agent.schema import FIELDS, fields_in_group

ELECTRICAL = fields_in_group("electrical")


def test_parses_the_contracted_shape() -> None:
    values, evidence = parse_response(
        '{"slew_rate": {"value": "21 V/us", "evidence": "SR Slew Rate 21 V/us"}}',
        ELECTRICAL,
    )
    assert values["slew_rate"] == "21 V/us"
    assert evidence["slew_rate"] == "SR Slew Rate 21 V/us"


def test_tolerates_prose_and_code_fences_around_the_json() -> None:
    text = 'Here are the values:\n```json\n{"slew_rate": {"value": "21 V/us"}}\n```\nDone.'
    values, _ = parse_response(text, ELECTRICAL)
    assert values["slew_rate"] == "21 V/us"


def test_accepts_a_bare_string_instead_of_the_object() -> None:
    values, _ = parse_response('{"slew_rate": "21 V/us"}', ELECTRICAL)
    assert values["slew_rate"] == "21 V/us"


def test_null_becomes_empty_not_the_string_none() -> None:
    values, _ = parse_response('{"slew_rate": {"value": null}}', ELECTRICAL)
    assert values["slew_rate"] == ""


def test_ignores_fields_outside_this_agents_group() -> None:
    values, _ = parse_response(
        '{"slew_rate": {"value": "21 V/us"}, "package_case": {"value": "SOIC-8"}}',
        ELECTRICAL,
    )
    assert "package_case" not in values


def test_malformed_json_costs_one_part_not_an_exception() -> None:
    assert parse_response('{"slew_rate": {"value": ', ELECTRICAL) == ({}, {})
    assert parse_response("I could not find the values.", ELECTRICAL) == ({}, {})
    assert parse_response("", ELECTRICAL) == ({}, {})


def test_merge_folds_values_and_costs_together() -> None:
    a = Extraction("X", {"slew_rate": "21 V/us"}, usage=Usage(10, 5, 1, 0.5), steps=2)
    b = Extraction(
        "X",
        {"package_case": "SOIC-8"},
        evidence={"package_case": "8-Lead SOIC"},
        usage=Usage(20, 7, 2, 1.0),
        steps=3,
    )
    a.merge(b)
    assert a.values == {"slew_rate": "21 V/us", "package_case": "SOIC-8"}
    assert a.evidence == {"package_case": "8-Lead SOIC"}
    assert (a.usage.input_tokens, a.usage.output_tokens, a.usage.calls) == (30, 12, 3)
    assert a.steps == 5


def test_merge_keeps_both_errors() -> None:
    a = Extraction("X", {}, error="first")
    a.merge(Extraction("X", {}, error="second"))
    assert "first" in a.error and "second" in a.error


def test_every_field_is_reachable_by_exactly_one_group() -> None:
    from datasheet_agent.schema import GROUPS

    covered = [f.name for g in GROUPS for f in fields_in_group(g)]
    assert sorted(covered) == sorted(f.name for f in FIELDS)
    assert len(covered) == len(set(covered))
