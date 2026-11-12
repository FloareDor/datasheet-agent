"""Two ways to get specs out of a datasheet: one big call, or an agent with tools.

`extract_single_call` is the baseline the plan measures everything against: paste as
much of the datasheet as fits and ask for every field at once. `run_agent` is the
alternative: give the model three tools and let it navigate, which is the only option
once a datasheet is longer than the context the small model can afford.

Both return the same `Extraction`, so the evaluation harness cannot tell them apart.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from datasheet_agent.llm import Backend, Usage
from datasheet_agent.schema import FIELDS, FieldSpec, search_vocabulary
from datasheet_agent.tools import DatasheetToolbox, dispatch

DEFAULT_MAX_STEPS = 10
# Roughly 15k tokens, which leaves room for the prompt and the reply inside the 32k
# context the served models run with. A bigger budget silently truncates the tail of
# the datasheet -- where the package tables usually are.
SINGLE_CALL_CHAR_BUDGET = 60_000

# Opt-in, because it makes a run report tens of megabytes. It exists for the
# coverage analysis: to ask whether a wrong answer was a failure to *find* the
# value or a failure to *read* it, you need the text the agent actually saw.
SAVE_CONTEXT = "DSA_SAVE_CONTEXT"


@dataclass
class Extraction:
    """What one extraction run produced, and what it cost."""

    part_number: str
    values: dict[str, str]
    evidence: dict[str, str] = field(default_factory=dict)
    """field -> the quoted datasheet line the value came from."""
    usage: Usage = field(default_factory=Usage)
    tool_calls: list[dict[str, object]] = field(default_factory=list)
    steps: int = 0
    error: str = ""

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def context_seen(self) -> str:
        """Everything the tools handed back, in the order the agent received it."""
        return "\n".join(str(call.get("result", "")) for call in self.tool_calls)

    def to_dict(self, model: str = "") -> dict[str, object]:
        payload: dict[str, object] = {
            "part_number": self.part_number,
            "values": self.values,
            "evidence": self.evidence,
            "usage": self.usage.to_dict(model),
            "tool_calls": self.tool_call_count,
            "steps": self.steps,
            "error": self.error,
        }
        if os.environ.get(SAVE_CONTEXT, "0").strip().lower() in {"1", "true", "yes"}:
            payload["context_seen"] = self.context_seen
        return payload

    def merge(self, other: Extraction) -> None:
        """Fold another extraction's results and costs into this one."""
        self.values.update(other.values)
        self.evidence.update(other.evidence)
        self.usage.add(other.usage)
        self.tool_calls.extend(other.tool_calls)
        self.steps += other.steps
        if other.error:
            self.error = f"{self.error}; {other.error}".strip("; ")


# ---- prompts ------------------------------------------------------------

SYSTEM = (
    "You are a hardware engineer reading a component datasheet to fill in a parametric "
    "spec sheet. You report only values the datasheet states. You never guess a value "
    "from the part number or from what is typical for the family: a value you cannot "
    "find is reported as null, which is always better than a plausible invention."
)

_RULES = """\
Rules:
- Report the value for the specific orderable part number requested. Datasheets often
  cover a family (e.g. TL061/TL062/TL064) with different grades per part; pick the
  column or row matching this part's suffix where the datasheet distinguishes them.
- Use the unit asked for in each field. Include the unit in your answer.
- For a dual-supply part specified as +-18 V, report supply voltage as the total, 36 V.
- Vos is the maximum at 25 degC; GBP and slew rate are typical values.
- If the datasheet does not state a value, use null. Do not infer it."""


def _field_block(specs: tuple[FieldSpec, ...]) -> str:
    return "\n".join(f"- {s.name} ({s.unit_hint}): {s.description}" for s in specs)


def _output_contract(specs: tuple[FieldSpec, ...]) -> str:
    example = ",\n".join(
        f'    "{s.name}": {{"value": "<number with unit, or null>", "evidence": "<the '
        f'datasheet line you took it from>"}}'
        for s in specs
    )
    return (
        "When you have the values, reply with JSON only, in exactly this shape, and "
        "nothing else:\n{\n" + example + "\n}"
    )


def single_call_prompt(part_number: str, specs: tuple[FieldSpec, ...], text: str) -> str:
    return (
        f"Datasheet text for part {part_number}:\n\n"
        f"<datasheet>\n{text}\n</datasheet>\n\n"
        f"Extract these fields for {part_number}:\n{_field_block(specs)}\n\n"
        f"{_RULES}\n\n{_output_contract(specs)}"
    )


def agent_prompt(part_number: str, specs: tuple[FieldSpec, ...], orientation: str) -> str:
    return (
        f"Extract specs for part {part_number} from its datasheet.\n\n"
        f"{orientation}\n\n"
        f"Fields to find:\n{_field_block(specs)}\n\n"
        f"{_RULES}\n\n"
        "Work like this: call search with a phrase that appears in the datasheet "
        "itself, note the page it hits, then call read_table on that page and read the "
        "value out of the table.\n"
        "Do NOT search for the field names above -- no datasheet contains the string "
        "'supply_voltage_max'. Search for these instead:\n"
        f"{search_vocabulary(specs)}\n\n"
        "Several of these fields usually share one specification table, so a single "
        "read_table often answers most of them. A few well-chosen calls beat many "
        "vague ones.\n\n"
        f"{_output_contract(specs)}"
    )


# ---- response parsing ---------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

# Models asked for `null` often emit the *string* "null" instead. Treating that as a
# value would score it wrong on a field the model correctly declined to answer, and
# would let it through the verifier as if it were real.
_NULL_WORDS = {"null", "none", "n/a", "na", "not specified", "not stated", "unknown", "-"}


def _as_value(raw: object) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    return "" if text.lower() in _NULL_WORDS else text


def parse_response(
    text: str, specs: tuple[FieldSpec, ...]
) -> tuple[dict[str, str], dict[str, str]]:
    """Pull values and evidence out of a model reply.

    Models wrap JSON in prose or fences, and sometimes answer a field with a bare
    string instead of the object asked for. Both are accepted; anything genuinely
    unparseable yields empty results rather than an exception, so one malformed reply
    costs one part rather than the whole run.
    """
    match = _JSON_BLOCK.search(text or "")
    if match is None:
        return {}, {}
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return {}, {}
    if not isinstance(payload, dict):
        return {}, {}

    names = {s.name for s in specs}
    values: dict[str, str] = {}
    evidence: dict[str, str] = {}
    for name, entry in payload.items():
        if name not in names:
            continue
        if isinstance(entry, dict):
            raw_value, raw_evidence = entry.get("value"), entry.get("evidence")
        else:
            raw_value, raw_evidence = entry, None
        values[name] = _as_value(raw_value)
        if raw_evidence:
            evidence[name] = str(raw_evidence).strip()
    return values, evidence


# ---- the two strategies -------------------------------------------------


def extract_single_call(
    backend: Backend,
    part_number: str,
    pdf_path: Path,
    specs: tuple[FieldSpec, ...] = FIELDS,
    char_budget: int = SINGLE_CALL_CHAR_BUDGET,
) -> Extraction:
    """Baseline: one call with as much of the datasheet as the budget allows."""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        pages = [page.get_text() for page in doc]
    text = "\n".join(f"--- PAGE {i} ---\n{p}" for i, p in enumerate(pages, start=1))
    truncated = len(text) > char_budget
    if truncated:
        text = text[:char_budget] + "\n... [datasheet truncated to fit the context budget]"

    reply = backend.chat(
        SYSTEM,
        [{"role": "user", "content": single_call_prompt(part_number, specs, text)}],
    )
    values, evidence = parse_response(reply.text, specs)
    return Extraction(
        part_number=part_number,
        values=values,
        evidence=evidence,
        usage=reply.usage,
        steps=1,
        error="" if values else "unparseable response",
    )


def _force_answer(
    backend: Backend,
    messages: list[dict[str, object]],
    specs: tuple[FieldSpec, ...],
) -> tuple[dict[str, str], dict[str, str], Usage]:
    """One last call with no tools, demanding the answer in the contracted shape.

    Everything the agent read is still in `messages`, so this is not a guess from
    nothing -- it asks the model to commit to what it already found.
    """
    closing = list(messages) + [
        {
            "role": "user",
            "content": (
                "You have no tool calls left. Using only what you have already read, "
                "give your best answer now. Any field the datasheet did not state is "
                "null.\n\n" + _output_contract(specs)
            ),
        }
    ]
    reply = backend.chat(SYSTEM, closing, tools=None)
    values, evidence = parse_response(reply.text, specs)
    return values, evidence, reply.usage


def run_agent(
    backend: Backend,
    part_number: str,
    toolbox: DatasheetToolbox,
    specs: tuple[FieldSpec, ...] = FIELDS,
    tool_schemas: list[dict[str, object]] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Extraction:
    """Let the model navigate the datasheet with tools until it answers.

    `toolbox` is shared across the agents working on one datasheet, so its index is
    built once. Tool calls made here are attributed to this agent by slicing the
    toolbox transcript, which keeps per-agent cost honest under parallel execution.
    """
    from datasheet_agent.tools import TOOL_SCHEMAS

    schemas = backend.format_tools(tool_schemas if tool_schemas is not None else TOOL_SCHEMAS)
    first_call = len(toolbox.calls)
    # Let the tool recover when the model searches for a field name instead of the
    # words the datasheet actually prints.
    toolbox.term_hints.update({spec.name: spec.search_terms for spec in specs})

    messages: list[dict[str, object]] = [
        {"role": "user", "content": agent_prompt(part_number, specs, toolbox.orientation())}
    ]
    usage = Usage()
    steps = 0
    error = ""
    values: dict[str, str] = {}
    evidence: dict[str, str] = {}

    while steps < max_steps:
        steps += 1
        reply = backend.chat(SYSTEM, messages, tools=schemas)  # type: ignore[arg-type]
        usage.add(reply.usage)

        if reply.wants_tools:
            messages.append(backend.assistant_message(reply))
            for request in reply.tool_requests:
                result = dispatch(toolbox, request.name, request.arguments)
                messages.append(backend.tool_result_message(request, result))
            continue

        values, evidence = parse_response(reply.text, specs)
        if values:
            break
        # A reply with neither tools nor parseable JSON: ask once, plainly.
        messages.append(backend.assistant_message(reply))
        messages.append(
            {
                "role": "user",
                "content": "Reply with the JSON object only, no other text.",
            }
        )
        error = "needed a reprompt"
    else:
        # The budget is spent, so stop offering tools and ask for the answer. Without
        # this the agent simply runs out mid-search and reports nothing, which scores
        # a zero that measures the protocol rather than the model: more than half of
        # the equal-budget multi-agent runs ended this way.
        values, evidence, final_usage = _force_answer(backend, messages, specs)
        usage.add(final_usage)
        steps += 1
        error = "" if values else f"no answer after {max_steps} steps"

    return Extraction(
        part_number=part_number,
        values=values,
        evidence=evidence,
        usage=usage,
        tool_calls=toolbox.transcript()[first_call:],
        steps=steps,
        error="" if values else (error or "no values returned"),
    )
