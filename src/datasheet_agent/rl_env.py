"""The RL environment: one datasheet extraction episode.

TRL 1.17's `environment_factory` turns an object's public methods into the tools the
policy may call, with `reset` starting an episode. So the environment is the same
three tools the evaluated agent uses -- `search`, `read_section`, `read_table` --
which is the point: the policy is trained on exactly the interface it is scored on.

The reward lives in `extraction_reward` rather than the environment's `get_reward`,
because scoring needs the model's final message (the JSON answer) and the environment
only sees tool calls. TRL hands a reward function both `completions` and
`environments`, so one function can see the answer and the cost that produced it.

Everything here must be importable at module level and picklable: the async trainer
runs rollouts in a separate process and registers tools by `__name__`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from datasheet_agent.extract import agent_prompt, parse_response
from datasheet_agent.reward import score_rollout
from datasheet_agent.schema import FIELDS_BY_NAME, FieldSpec, fields_in_group
from datasheet_agent.tools import DatasheetToolbox

DEFAULT_CACHE = Path("/data/cache")


class DatasheetEnv:
    """One extraction episode over one datasheet.

    Constructed once per rollout slot and reset between episodes, so the PDF index is
    built at most once per datasheet per worker.
    """

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE) -> None:
        self.cache_dir = Path(cache_dir)
        self._toolbox: DatasheetToolbox | None = None
        self.part_number: str = ""
        self.labels: dict[str, str] = {}
        self.specs: tuple[FieldSpec, ...] = ()
        self.document_text: str = ""
        """Flat text of this datasheet, so the reward can check evidence against it."""

    # ---- episode lifecycle ------------------------------------------

    def reset(self, **kwargs: Any) -> str:
        """Start an episode. Returns the task prompt the policy sees.

        Dataset columns arrive here as keyword arguments.
        """
        self.part_number = str(kwargs.get("part_number", ""))
        self.labels = dict(kwargs.get("labels") or {})
        self.specs = _specs_for(kwargs.get("group"), self.labels)

        self._close()
        pdf_path = Path(str(kwargs.get("pdf_path", "")))
        self.document_text = _datasheet_text(pdf_path)
        self._toolbox = DatasheetToolbox(pdf_path=pdf_path, cache_dir=self.cache_dir)
        try:
            self._toolbox.open()
            orientation = self._toolbox.orientation()
        except Exception as exc:  # noqa: BLE001 - a broken PDF must not kill training
            self._toolbox = None
            orientation = f"(this datasheet could not be indexed: {exc})"

        # Appended to the seed user turn; the system prompt is already in the seed.
        return agent_prompt(self.part_number, self.specs, orientation)

    def _close(self) -> None:
        if self._toolbox is not None:
            self._toolbox.close()
            self._toolbox = None

    @property
    def tool_call_count(self) -> int:
        return len(self._toolbox.calls) if self._toolbox else 0

    # ---- the tools --------------------------------------------------
    # Public methods become the policy's tools, so every public method here is one.

    def search(self, query: str) -> str:
        """Search the datasheet's full text and return page-numbered snippets.

        Args:
            query: A phrase that appears in the datasheet, such as "Slew Rate".
        Returns:
            Matching snippets, each prefixed with its page number.
        """
        if self._toolbox is None:
            return "The datasheet is unavailable."
        return self._toolbox.search(query)

    def read_section(self, start_page: int, end_page: int) -> str:
        """Read the text of a page range, at most five pages.

        Args:
            start_page: First page to read, 1-indexed.
            end_page: Last page to read, 1-indexed.
        Returns:
            The extracted text of those pages.
        """
        if self._toolbox is None:
            return "The datasheet is unavailable."
        return self._toolbox.read_section(start_page, end_page)

    def read_table(self, page: int) -> str:
        """Extract one page as markdown, preserving table structure.

        Args:
            page: The page to extract, 1-indexed.
        Returns:
            The page as markdown, with specification tables kept as tables.
        """
        if self._toolbox is None:
            return "The datasheet is unavailable."
        return self._toolbox.read_table(page)


def _datasheet_text(pdf_path: Path, max_pages: int = 40) -> str:
    """Flat text of the datasheet, for the reward's grounding check."""
    import pymupdf

    try:
        with pymupdf.open(pdf_path) as doc:
            return " ".join(doc[i].get_text() for i in range(min(max_pages, doc.page_count)))
    except Exception:  # noqa: BLE001 - a broken PDF must not kill training
        return ""


def _specs_for(group: object, labels: dict[str, str]) -> tuple[FieldSpec, ...]:
    """The fields this episode asks for: a named group, or whatever is labelled."""
    if isinstance(group, str) and group:
        return fields_in_group(group)  # type: ignore[arg-type]
    return tuple(FIELDS_BY_NAME[name] for name in labels if name in FIELDS_BY_NAME)


def _final_text(completion: Any) -> str:
    """The assistant's last message, whatever shape TRL hands back."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for message in reversed(completion):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    parts = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    if any(parts):
                        return "\n".join(parts)
    return ""


def _tool_call_penalty() -> float:
    """Per-call cost, overridable so an ablation can set it to zero.

    v3 converged to exactly 1.00 tool calls per rollout -- the minimum the grounding
    gate allows. Setting this to 0 asks whether the cost term is what suppressed tool
    use, or whether the model simply gains nothing from reading more.
    """
    from datasheet_agent.reward import TOOL_CALL_PENALTY

    raw = os.environ.get("TOOL_CALL_PENALTY")
    try:
        return float(raw) if raw is not None else TOOL_CALL_PENALTY
    except ValueError:
        return TOOL_CALL_PENALTY


def extraction_reward(
    completions: list[Any],
    environments: list[DatasheetEnv] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Score each rollout: fields correct, minus the tool calls it took.

    TRL calls this once per training step with one entry per completion.
    """
    envs = environments or []
    log_metric = kwargs.get("log_metric")
    log_extra = kwargs.get("log_extra")

    rewards: list[float] = []
    accuracies: list[float] = []
    raw_accuracies: list[float] = []
    grounded: list[float] = []
    call_counts: list[int] = []

    for index, completion in enumerate(completions):
        env = envs[index] if index < len(envs) else None
        if env is None:
            rewards.append(0.0)
            continue

        values, evidence = parse_response(_final_text(completion), env.specs)
        breakdown, _ = score_rollout(
            env.part_number,
            values,
            env.labels,
            tool_calls=env.tool_call_count,
            tool_call_penalty=_tool_call_penalty(),
            evidence=evidence,
            document_text=env.document_text,
        )
        rewards.append(breakdown.total)
        accuracies.append(breakdown.accuracy)
        raw_accuracies.append(breakdown.raw_accuracy)
        grounded.append(breakdown.grounded_fraction)
        call_counts.append(env.tool_call_count)

    if callable(log_metric) and accuracies:
        log_metric("grounded_accuracy", sum(accuracies) / len(accuracies))
        # Logged side by side: a gap between them is the policy guessing rather than
        # reading, which is exactly the collapse the grounding term exists to prevent.
        log_metric("raw_accuracy", sum(raw_accuracies) / len(raw_accuracies))
        log_metric("grounded_fraction", sum(grounded) / len(grounded))
        log_metric("tool_calls", sum(call_counts) / len(call_counts))
    if callable(log_extra) and call_counts:
        log_extra("tool_calls", call_counts)

    return rewards
