"""Running a method over a split and reporting accuracy, cost and latency.

Every method -- one big call, one agent, three agents, with or without the verifier --
is reduced to the same signature, so the harness cannot favour one over another. Parts
run concurrently because each is independent; the per-part latency reported is the
method's own wall-clock, not the batch's.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from datasheet_agent.dataset import LabelledPart
from datasheet_agent.extract import Extraction, extract_single_call
from datasheet_agent.grading import PartScore, RunSummary, grade_part, summarise
from datasheet_agent.llm import Usage
from datasheet_agent.pipeline import (
    PipelineConfig,
    PipelineResult,
    extract_multi_agent,
    extract_single_agent,
)

Method = Callable[[PipelineConfig, str, Path], PipelineResult]


@dataclass
class PartResult:
    """One part, end to end."""

    part_number: str
    score: PartScore
    result: PipelineResult
    seconds: float
    error: str = ""

    def to_dict(self, model: str = "") -> dict[str, object]:
        return {
            "part_number": self.part_number,
            "accuracy": round(self.score.accuracy, 4),
            "correct": self.score.correct,
            "graded": self.score.graded,
            "seconds": round(self.seconds, 2),
            "error": self.error,
            **self.result.to_dict(model),
            "wrong_fields": [
                {"field": o.field, "predicted": o.predicted, "label": o.label}
                for o in self.score.wrong_fields()
            ],
        }


@dataclass
class EvalReport:
    """Everything one method's run produced."""

    method: str
    model: str
    parts: list[PartResult] = field(default_factory=list)
    wall_seconds: float = 0.0

    @property
    def summary(self) -> RunSummary:
        return summarise([p.score for p in self.parts])

    @property
    def usage(self) -> Usage:
        total = Usage()
        for part in self.parts:
            total.add(part.result.extraction.usage)
        return total

    def headline(self) -> dict[str, object]:
        summary = self.summary
        n = max(len(self.parts), 1)
        usage = self.usage
        tool_calls = sum(p.result.extraction.tool_call_count for p in self.parts)
        return {
            "method": self.method,
            "model": self.model,
            "parts": len(self.parts),
            "accuracy": round(summary.accuracy, 4),
            "correct": summary.correct,
            "graded": summary.graded,
            "tokens_in_per_part": round(usage.input_tokens / n, 1),
            "tokens_out_per_part": round(usage.output_tokens / n, 1),
            "llm_calls_per_part": round(usage.calls / n, 2),
            "tool_calls_per_part": round(tool_calls / n, 2),
            "cost_usd_per_part": round(usage.cost_usd(self.model) / n, 6),
            "latency_s_per_part": round(sum(p.seconds for p in self.parts) / n, 2),
            "wall_seconds": round(self.wall_seconds, 1),
            "errors": sum(1 for p in self.parts if p.error),
            "verifier_rejections": sum(len(p.result.verification.rejections) for p in self.parts),
            "fields_repaired": sum(len(p.result.repaired_fields) for p in self.parts),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "headline": self.headline(),
            "per_field": self.summary.to_dict()["per_field"],
            "parts": [p.to_dict(self.model) for p in self.parts],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ---- the methods, behind one signature ---------------------------------


def method_single_call(config: PipelineConfig, part_number: str, pdf_path: Path) -> PipelineResult:
    """Baseline: paste the datasheet into one call."""
    extraction = extract_single_call(config.backend, part_number, pdf_path)
    return PipelineResult(extraction=extraction)


METHODS: dict[str, Method] = {
    "single_call": method_single_call,
    "single_agent": extract_single_agent,
    "multi_agent": extract_multi_agent,
}


def resolve_method(name: str) -> Method:
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; available: {', '.join(METHODS)}")
    return METHODS[name]


# ---- the harness --------------------------------------------------------


def evaluate_part(
    method: Method,
    config: PipelineConfig,
    part: LabelledPart,
) -> PartResult:
    started = time.monotonic()
    try:
        result = method(config, part.part_number, part.pdf)
        error = result.extraction.error
    except Exception as exc:  # noqa: BLE001 - one bad part must not end the run
        result = PipelineResult(extraction=Extraction(part.part_number, {}, error=str(exc)))
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    elapsed = time.monotonic() - started

    score = grade_part(part.part_number, result.extraction.values, part.labels)
    return PartResult(part.part_number, score, result, elapsed, error)


def evaluate(
    method_name: str,
    config: PipelineConfig,
    parts: list[LabelledPart],
    workers: int = 8,
    progress: bool = True,
) -> EvalReport:
    """Run one method over `parts`, concurrently, and grade every result."""
    method = resolve_method(method_name)
    report = EvalReport(method=method_name, model=config.backend.model)
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate_part, method, config, part): part for part in parts}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            report.parts.append(result)
            if progress:
                running = summarise([p.score for p in report.parts]).accuracy
                print(
                    f"  [{done:3d}/{len(parts)}] {result.part_number:22s} "
                    f"{result.score.correct}/{result.score.graded} "
                    f"({result.seconds:5.1f}s)  running acc {running:.1%}",
                    flush=True,
                )

    report.wall_seconds = time.monotonic() - started
    return report


def print_headline(report: EvalReport) -> None:
    headline = report.headline()
    print(f"\n=== {headline['method']} / {headline['model']} ===")
    for key, value in headline.items():
        if key not in {"method", "model"}:
            print(f"  {key:24s} {value}")
    print("\n  per-field accuracy:")
    per_field = report.summary.to_dict()["per_field"]
    for name, stats in per_field.items():  # type: ignore[union-attr]
        print(f"    {name:26s} {stats['accuracy']:.1%}  ({stats['correct']}/{stats['graded']})")
