"""Pipeline behaviour that does not need a model: merging, and what a retry may do.

These use a scripted fake backend rather than a served model, so they assert the
pipeline's own logic and run in milliseconds.
"""

from __future__ import annotations

import json
from typing import Any

from datasheet_agent.extract import Extraction, parse_response
from datasheet_agent.llm import Backend, Reply, ToolRequest, Usage
from datasheet_agent.pipeline import PipelineResult, _merge, apply_retry_values
from datasheet_agent.schema import fields_in_group


class ScriptedBackend(Backend):
    """Replays a fixed list of replies, recording the prompts it was given."""

    def __init__(self, replies: list[Reply]) -> None:
        self.model = "scripted"
        self._replies = list(replies)
        self.seen: list[list[dict[str, Any]]] = []

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> Reply:
        self.seen.append(list(messages))
        return self._replies.pop(0) if self._replies else Reply(text="{}", usage=Usage(1, 1, 1))

    def tool_result_message(self, request: ToolRequest, result: str) -> dict[str, Any]:
        return {"role": "tool", "tool_call_id": request.id, "content": result}

    def assistant_message(self, reply: Reply) -> dict[str, Any]:
        return {"role": "assistant", "content": reply.text}

    def format_tools(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return schemas


def test_merge_is_a_union_because_the_groups_are_disjoint() -> None:
    electrical = Extraction("X", {"slew_rate": "21 V/us"}, usage=Usage(10, 2, 1))
    thermal = Extraction("X", {"operating_temp_min": "-40 C"}, usage=Usage(5, 1, 1))
    package = Extraction("X", {"package_case": "SOIC-8"}, usage=Usage(7, 3, 1))

    merged = _merge([electrical, thermal, package], "X")
    assert merged.values == {
        "slew_rate": "21 V/us",
        "operating_temp_min": "-40 C",
        "package_case": "SOIC-8",
    }
    assert merged.usage.input_tokens == 22
    assert merged.usage.calls == 3


def test_merge_accumulates_cost_rather_than_replacing_it() -> None:
    """Per-agent cost must sum; otherwise multi-agent looks artificially cheap."""
    merged = _merge(
        [Extraction("X", {}, usage=Usage(100, 10, 2)), Extraction("X", {}, usage=Usage(50, 5, 3))],
        "X",
    )
    assert (merged.usage.input_tokens, merged.usage.output_tokens) == (150, 15)
    assert merged.usage.calls == 5


def test_result_round_trips_pre_retry_values() -> None:
    """The verifier can only be graded as a classifier if the originals are kept."""
    result = PipelineResult(
        extraction=Extraction("X", {"slew_rate": ""}),
        pre_retry_values={"slew_rate": "21 V/us"},
    )
    payload = result.to_dict()
    assert payload["pre_retry_values"] == {"slew_rate": "21 V/us"}
    assert payload["values"] == {"slew_rate": ""}


def test_agent_answers_from_a_scripted_reply() -> None:
    specs = fields_in_group("thermal")
    answer = json.dumps(
        {
            "operating_temp_min": {"value": "-40 degC", "evidence": "TA -40 to 105"},
            "operating_temp_max": {"value": "105 degC", "evidence": "TA -40 to 105"},
        }
    )
    values, evidence = parse_response(answer, specs)
    assert values["operating_temp_min"] == "-40 degC"
    assert evidence["operating_temp_max"] == "TA -40 to 105"


def test_scripted_backend_records_what_the_agent_was_told() -> None:
    backend = ScriptedBackend([Reply(text='{"slew_rate": {"value": "21 V/us"}}')])
    reply = backend.chat("sys", [{"role": "user", "content": "find the slew rate"}])
    assert "21 V/us" in reply.text
    assert backend.seen[0][0]["content"] == "find the slew rate"


def test_a_retry_may_correct_a_value_but_never_erase_one() -> None:
    """Rejection is not evidence of absence.

    Measured over 100 parts, retries destroyed 98 correct values (94 by nulling them)
    while repairing 9. This guard is what makes the verifier safe to run.
    """
    values = {"slew_rate": "21 V/us", "supply_voltage_max": "36 V"}
    changed = apply_retry_values(values, {"slew_rate": "", "supply_voltage_max": "   "})
    assert values == {"slew_rate": "21 V/us", "supply_voltage_max": "36 V"}
    assert changed == []


def test_a_retry_that_supplies_a_better_value_is_accepted() -> None:
    values = {"slew_rate": "1 V/us"}
    changed = apply_retry_values(values, {"slew_rate": "21 V/us"})
    assert values["slew_rate"] == "21 V/us"
    assert changed == ["slew_rate"]


def test_a_retry_repeating_the_same_value_is_not_counted_as_a_repair() -> None:
    values = {"slew_rate": "21 V/us"}
    assert apply_retry_values(values, {"slew_rate": "21 V/us"}) == []


def test_a_retry_can_fill_a_field_that_was_empty() -> None:
    values = {"slew_rate": ""}
    assert apply_retry_values(values, {"slew_rate": "21 V/us"}) == ["slew_rate"]
    assert values["slew_rate"] == "21 V/us"
