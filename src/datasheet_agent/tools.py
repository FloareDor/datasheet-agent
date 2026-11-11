"""The tool surface an extraction agent sees.

Three tools, deliberately: `search` to find where a value lives, `read_section` to
read around a hit, `read_table` to pull a specification table as markdown. They wrap
Infineon's `datasheetindex`, which does the PDF work: a page-matched text layer, a
section map where the PDF has one, and layout-aware table extraction.

Indexing a datasheet costs ~2 s, so `DatasheetToolbox` builds artifacts once into a
cache directory and every later run reuses them.
"""

from __future__ import annotations

import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

from datasheetindex.tools.bound import DatasheetTools

MAX_SNIPPET_CHARS = 400
MAX_SECTION_CHARS = 6_000
MAX_TABLE_CHARS = 8_000
MAX_PAGE_SPAN = 5

# One lock per datasheet. Family datasheets cover several parts, and the three field
# agents share one PDF, so concurrent runs build the same cache files at the same
# time -- which on Windows fails outright with "Access is denied" rather than merely
# duplicating work. Serialising per PDF costs nothing: the second caller finds the
# artifacts already on disk and reuses them.
_BUILD_LOCKS: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
_LOCKS_GUARD = threading.Lock()


def _build_lock(pdf_path: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        return _BUILD_LOCKS[str(pdf_path.resolve())]


@dataclass
class ToolCall:
    """One tool invocation, recorded for cost accounting and for the verifier."""

    name: str
    arguments: dict[str, object]
    result: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "arguments": self.arguments, "result": self.result}


def _normalise_name(text: str) -> str:
    """Letters and digits only, lowercased, so `operating_temp_min` == `OperatingTempMin`."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _query_variants(query: str) -> list[str]:
    """Progressively looser forms of a search query, most specific first."""
    variants: list[str] = []

    def add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in variants:
            variants.append(candidate)

    add(query)
    spaced = re.sub(r"[_\-]+", " ", query).strip()
    add(spaced)
    # Drop trailing qualifiers: "supply voltage max" -> "supply voltage".
    words = spaced.split()
    for cut in range(len(words) - 1, 0, -1):
        add(" ".join(words[:cut]))
    return variants


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


@dataclass
class DatasheetToolbox:
    """Tools bound to one datasheet PDF.

    Every call is appended to `calls`, which is what the cost term in the RL reward
    and the verifier's evidence check both read.
    """

    pdf_path: Path
    cache_dir: Path
    term_hints: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Field name -> the words a datasheet uses for it.

    When a model searches for a field's machine name, this is what the query is
    replaced with. Blind relaxation is not enough and can be actively harmful:
    `operating_temp_min` shortens to "operating temp", which substring-matches
    "Full Temperature" inside a distortion spec and hands back a number in percent.
    """
    calls: list[ToolCall] = field(default_factory=list)
    _tools: DatasheetTools | None = None
    _manifest: dict[str, object] | None = None

    def __enter__(self) -> DatasheetToolbox:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def open(self) -> None:
        if self._tools is not None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tools = DatasheetTools(str(self.pdf_path))
        with _build_lock(self.pdf_path):
            # caption_figures would call an LLM per figure; the agent reads text, not art.
            tools.build_datasheet(output_dir=str(self.cache_dir), caption_figures=False)
        self._tools = tools
        self._manifest = tools.get_artifact_manifest()

    def close(self) -> None:
        if self._tools is not None:
            self._tools.close()
            self._tools = None

    @property
    def tools(self) -> DatasheetTools:
        if self._tools is None:
            raise RuntimeError("DatasheetToolbox is not open; call open() first")
        return self._tools

    @property
    def total_pages(self) -> int:
        return int((self._manifest or {}).get("total_pages", 0))

    def orientation(self) -> str:
        """A short briefing the agent gets before its first tool call.

        Contains the page count and, where the PDF has one, its section map. Old
        datasheets often have no outline at all, in which case `datasheetindex`
        supplies a hint telling the agent to orient from the first pages instead.
        """
        manifest = self._manifest or {}
        lines = [f"Datasheet: {self.pdf_path.name}", f"Pages: {self.total_pages}"]
        toc = manifest.get("toc") or []
        if isinstance(toc, list) and toc:
            lines.append("Section map (title -> pages):")
            lines.extend(_format_toc(toc))
        else:
            hint = str(manifest.get("hint", "")).strip()
            lines.append(f"No table of contents. {hint}".strip())
        return "\n".join(lines)

    # ---- the three tools ------------------------------------------------

    def search(self, query: str, max_results: int = 6) -> str:
        """Find `query` in the datasheet text. Returns page-numbered snippets."""
        matches, used = self._search_forgiving(query, max_results)
        if not matches:
            result = (
                f"No matches for {query!r}. Datasheets do not contain field names; "
                "search for the words printed in the document, such as 'Slew Rate', "
                "'VOS' or 'Operating Temperature'."
            )
        else:
            header = "" if used == query else f"(no matches for {query!r}; showing {used!r})\n"
            result = header + "\n".join(
                f"[page {m['page']}] {_truncate(str(m['snippet']).strip(), MAX_SNIPPET_CHARS)}"
                for m in matches
            )
        self._record("search", {"query": query, "max_results": max_results}, result)
        return result

    def _search_forgiving(self, query: str, max_results: int) -> tuple[list[dict], str]:
        """Search, retrying with looser forms of the query before giving up.

        A literal search is the wrong behaviour for this tool. Models reach for the
        field's machine name (`slew_rate`), which appears in no datasheet ever
        printed, and a bare "no matches" teaches them nothing -- in the first agent
        runs every electrical field scored exactly 0% for this reason alone.

        So the query is relaxed the way a search box would relax it: separators become
        spaces, then trailing qualifier words are dropped, so `supply_voltage_max`
        reaches "supply voltage". The result says which form actually matched, so the
        model can see what worked and the transcript stays honest.
        """
        for candidate in self._candidates(query):
            matches = self.tools.search_text(candidate, max_results=max_results)
            if matches:
                return list(matches), candidate
        return [], query

    def _candidates(self, query: str) -> list[str]:
        """Query forms to try, most specific first.

        A query naming a known field is replaced by that field's real vocabulary
        before any blind shortening is attempted.
        """
        key = _normalise_name(query)
        hinted: tuple[str, ...] = ()
        for name, terms in self.term_hints.items():
            if _normalise_name(name) == key:
                hinted = terms
                break
        candidates = [query, *hinted]
        for variant in _query_variants(query):
            if variant not in candidates:
                candidates.append(variant)
        return candidates

    def read_section(self, start_page: int, end_page: int | None = None) -> str:
        """Read the text of a page range. End page defaults to the start page."""
        end = start_page if end_page is None else end_page
        start, end = self._clamp_pages(start_page, end)
        text = self.tools.get_section_text(start, end)
        result = _truncate(text.strip() or "(no extractable text)", MAX_SECTION_CHARS)
        self._record("read_section", {"start_page": start, "end_page": end}, result)
        return result

    def read_table(self, page: int) -> str:
        """Extract one page as layout-aware markdown, preserving table structure."""
        page, _ = self._clamp_pages(page, page)
        try:
            text = self.tools.extract_table_markdown(page)
        except Exception as exc:  # pragma: no cover - depends on PDF internals
            text = f"(table extraction failed on page {page}: {exc})"
        result = _truncate(text.strip() or f"(no table structure on page {page})", MAX_TABLE_CHARS)
        self._record("read_table", {"page": page}, result)
        return result

    # ---- internals ------------------------------------------------------

    def _clamp_pages(self, start: int, end: int) -> tuple[int, int]:
        """Keep page numbers in range so a hallucinated page is a no-op, not a crash."""
        last = max(self.total_pages, 1)
        start = max(1, min(int(start), last))
        end = max(start, min(int(end), last))
        # A greedy range would blow the context window; cap it at a readable span.
        return start, min(end, start + MAX_PAGE_SPAN - 1)

    def _record(self, name: str, arguments: dict[str, object], result: str) -> None:
        self.calls.append(ToolCall(name=name, arguments=arguments, result=result))

    def transcript(self) -> list[dict[str, object]]:
        return [c.to_dict() for c in self.calls]

    def reset_calls(self) -> None:
        self.calls.clear()


def _format_toc(nodes: list[dict[str, object]], depth: int = 0, limit: int = 60) -> list[str]:
    """Flatten the section tree into indented `title ... pages` lines."""
    lines: list[str] = []
    for node in nodes:
        if len(lines) >= limit:
            break
        title = str(node.get("title", "")).strip()
        start, end = node.get("start_page"), node.get("end_page")
        lines.append(f"{'  ' * depth}- {title} (p{start}-{end})")
        children = node.get("children") or []
        if isinstance(children, list):
            lines.extend(_format_toc(children, depth + 1, limit - len(lines)))
    return lines


# JSON Schemas handed to the model. Kept here so the agent prompt and the
# executor can never drift apart.
TOOL_SCHEMAS: list[dict[str, object]] = [
    {
        "name": "search",
        "description": (
            "Search the datasheet's full text. Returns page-numbered snippets. Use the "
            "symbol an engineer would use (e.g. 'Slew Rate', 'GBW', 'Vos', 'TA')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Text to search for."}},
            "required": ["query"],
        },
    },
    {
        "name": "read_section",
        "description": (
            "Read the text of a page range (max 5 pages). Use after search to read the "
            "context around a hit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_page": {"type": "integer", "description": "First page, 1-indexed."},
                "end_page": {"type": "integer", "description": "Last page, 1-indexed."},
            },
            "required": ["start_page"],
        },
    },
    {
        "name": "read_table",
        "description": (
            "Extract one page as markdown with table structure preserved. Use for "
            "specification tables where column alignment carries the meaning."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"page": {"type": "integer", "description": "Page, 1-indexed."}},
            "required": ["page"],
        },
    },
]

TOOL_NAMES: tuple[str, ...] = tuple(str(s["name"]) for s in TOOL_SCHEMAS)


def dispatch(toolbox: DatasheetToolbox, name: str, arguments: dict[str, object]) -> str:
    """Run a tool the model asked for. Unknown names return an error the model can read."""
    try:
        if name == "search":
            return toolbox.search(str(arguments.get("query", "")))
        if name == "read_section":
            raw_end = arguments.get("end_page")
            return toolbox.read_section(
                int(arguments.get("start_page", 1)),
                int(raw_end) if raw_end is not None else None,
            )
        if name == "read_table":
            return toolbox.read_table(int(arguments.get("page", 1)))
    except (TypeError, ValueError) as exc:
        return f"Invalid arguments for {name}: {exc}"
    return f"Unknown tool {name!r}. Available: {', '.join(TOOL_NAMES)}"
