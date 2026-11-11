"""The multi-agent extraction workflow.

A LangGraph graph: three field agents fan out in parallel, their results merge in
code, a verifier checks every value against the datasheet, and rejected fields go back
to the agent that produced them for one retry.

Merging in code rather than with a model is deliberate. The groups are disjoint by
construction, so the merge is a dictionary update -- there is nothing for a model to
decide, and a model asked to decide it could only introduce error.

Each agent gets its own `DatasheetToolbox` over a shared cache directory. The index is
built once and reused from disk, so parallel agents cost one build, while each keeps
its own tool-call transcript -- which is what makes per-agent cost honest.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from datasheet_agent.extract import Extraction, run_agent
from datasheet_agent.llm import Backend, Usage
from datasheet_agent.schema import FIELDS, GROUPS, FieldGroup, fields_in_group
from datasheet_agent.tools import DatasheetToolbox
from datasheet_agent.verify import Rejection, VerificationReport, retry_prompt, verify


@dataclass
class PipelineResult:
    """The merged extraction, plus what the verifier did to get there."""

    extraction: Extraction
    verification: VerificationReport = field(default_factory=VerificationReport)
    retried_fields: tuple[str, ...] = ()
    repaired_fields: tuple[str, ...] = ()
    """Fields whose value actually changed on retry."""

    pre_retry_values: dict[str, str] = field(default_factory=dict)
    """What the agents answered before the verifier touched anything.

    Kept so the verifier can be graded as a classifier rather than only by its effect
    on the final score: how many of the values it rejected were in fact correct, and
    how many of the wrong ones the retry actually repaired. Without this, a verifier
    that hurts and a verifier that helps-but-is-undone-by-a-bad-retry look identical.
    """

    def to_dict(self, model: str = "") -> dict[str, object]:
        return {
            **self.extraction.to_dict(model),
            "pre_retry_values": self.pre_retry_values,
            "verification": self.verification.to_dict(),
            "retried_fields": list(self.retried_fields),
            "repaired_fields": list(self.repaired_fields),
        }


class _State(TypedDict, total=False):
    """Graph state. `results` is reduced by concatenation so parallel branches merge."""

    results: Annotated[list[Extraction], operator.add]


@dataclass
class PipelineConfig:
    """Everything the graph needs that is not per-datasheet."""

    backend: Backend
    cache_dir: Path
    verifier_backend: Backend | None = None
    use_verifier: bool = True
    max_steps: int = 10
    groups: tuple[FieldGroup, ...] = GROUPS


def _toolbox(pdf_path: Path, cache_dir: Path) -> DatasheetToolbox:
    return DatasheetToolbox(pdf_path=pdf_path, cache_dir=cache_dir)


def _run_group(
    config: PipelineConfig,
    group: FieldGroup,
    part_number: str,
    pdf_path: Path,
) -> Extraction:
    """One field agent: its own toolbox, its own slice of the schema."""
    with _toolbox(pdf_path, config.cache_dir) as toolbox:
        return run_agent(
            config.backend,
            part_number,
            toolbox,
            specs=fields_in_group(group),
            max_steps=config.max_steps,
        )


def build_graph(config: PipelineConfig, part_number: str, pdf_path: Path) -> Any:
    """Compile a graph whose agent nodes run concurrently."""
    graph: StateGraph = StateGraph(_State)

    def make_node(group: FieldGroup):  # noqa: ANN202 - closure over the group
        def node(_: _State) -> _State:
            return {"results": [_run_group(config, group, part_number, pdf_path)]}

        return node

    for group in config.groups:
        graph.add_node(group, make_node(group))
        graph.add_edge(START, group)
        graph.add_edge(group, END)
    return graph.compile()


def _merge(results: list[Extraction], part_number: str) -> Extraction:
    merged = Extraction(part_number=part_number, values={})
    for result in results:
        merged.merge(result)
    return merged


def _datasheet_text(pdf_path: Path, max_pages: int = 40) -> str:
    """Flat text of the datasheet, for the verifier's grounding check."""
    import pymupdf

    try:
        with pymupdf.open(pdf_path) as doc:
            return " ".join(doc[i].get_text() for i in range(min(max_pages, doc.page_count)))
    except Exception:  # noqa: BLE001 - third-party PDFs vary wildly
        return ""


def _retry_rejected(
    config: PipelineConfig,
    part_number: str,
    pdf_path: Path,
    rejections: list[Rejection],
    merged: Extraction,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Send each rejected field back to its own agent, once. Returns (retried, repaired)."""
    from datasheet_agent.extract import SYSTEM, agent_prompt, parse_response
    from datasheet_agent.tools import TOOL_SCHEMAS, dispatch

    by_group: dict[FieldGroup, list[Rejection]] = {}
    for rejection in rejections:
        spec = next((f for f in FIELDS if f.name == rejection.field), None)
        if spec is not None:
            by_group.setdefault(spec.group, []).append(rejection)

    retried: list[str] = []
    repaired: list[str] = []
    for group, group_rejections in by_group.items():
        specs = tuple(
            f for f in fields_in_group(group) if f.name in {r.field for r in group_rejections}
        )
        if not specs:
            continue
        retried.extend(s.name for s in specs)

        wire_tools = config.backend.format_tools(TOOL_SCHEMAS)
        with _toolbox(pdf_path, config.cache_dir) as toolbox:
            messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": agent_prompt(part_number, specs, toolbox.orientation()),
                },
                {"role": "assistant", "content": _previous_answer(merged, specs)},
                {"role": "user", "content": retry_prompt(part_number, group_rejections)},
            ]
            values: dict[str, str] = {}
            for _ in range(config.max_steps):
                reply = config.backend.chat(SYSTEM, messages, tools=wire_tools)  # type: ignore[arg-type]
                merged.usage.add(reply.usage)
                if reply.wants_tools:
                    messages.append(config.backend.assistant_message(reply))
                    for request in reply.tool_requests:
                        result = dispatch(toolbox, request.name, request.arguments)
                        messages.append(config.backend.tool_result_message(request, result))
                    continue
                values, evidence = parse_response(reply.text, specs)
                repaired.extend(apply_retry_values(merged.values, values))
                merged.evidence.update(evidence)
                break
            merged.tool_calls.extend(toolbox.transcript())
    return tuple(retried), tuple(repaired)


def apply_retry_values(current: dict[str, str], retried: dict[str, str]) -> list[str]:
    """Fold a retry's answers into the current values. Returns the fields it changed.

    A retry may correct a value but never erase one. Being told "this looks wrong" is
    not evidence that the datasheet lacks the value, and a small model told it was
    wrong takes the null exit the retry prompt offers it. Measured over 100 parts,
    retries destroyed 98 correct values -- 94 of them by nulling -- while repairing 9.
    Refusing empty replacements is what makes the verifier safe to run at all.
    """
    changed: list[str] = []
    for name, value in retried.items():
        if not value.strip():
            continue
        if value != current.get(name, ""):
            changed.append(name)
        current[name] = value
    return changed


def _previous_answer(merged: Extraction, specs: tuple[Any, ...]) -> str:
    import json

    return json.dumps(
        {
            s.name: {
                "value": merged.values.get(s.name, ""),
                "evidence": merged.evidence.get(s.name, ""),
            }
            for s in specs
        }
    )


def extract_multi_agent(
    config: PipelineConfig,
    part_number: str,
    pdf_path: Path,
) -> PipelineResult:
    """Run the full workflow: parallel agents, merge, verify, one retry."""
    graph = build_graph(config, part_number, pdf_path)
    state = graph.invoke({"results": []})
    merged = _merge(list(state["results"]), part_number)

    if not config.use_verifier:
        return PipelineResult(extraction=merged)

    report = verify(
        values=merged.values,
        evidence=merged.evidence,
        document_text=_datasheet_text(pdf_path),
        part_number=part_number,
        backend=config.verifier_backend,
    )
    merged.usage.add(report.usage)

    pre_retry = dict(merged.values)
    retried: tuple[str, ...] = ()
    repaired: tuple[str, ...] = ()
    if report.rejections:
        retried, repaired = _retry_rejected(
            config, part_number, pdf_path, report.rejections, merged
        )

    return PipelineResult(
        extraction=merged,
        verification=report,
        retried_fields=retried,
        repaired_fields=repaired,
        pre_retry_values=pre_retry,
    )


def extract_single_agent(
    config: PipelineConfig,
    part_number: str,
    pdf_path: Path,
) -> PipelineResult:
    """One agent, all nine fields. The control for the multi-agent comparison."""
    with _toolbox(pdf_path, config.cache_dir) as toolbox:
        extraction = run_agent(
            config.backend,
            part_number,
            toolbox,
            specs=FIELDS,
            max_steps=config.max_steps,
        )
    return PipelineResult(extraction=extraction)


__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "Usage",
    "extract_multi_agent",
    "extract_single_agent",
]
