"""Retrieval baselines: chunk the datasheet, retrieve, answer in one call.

This exists to answer the first question anyone asks about a tool-using agent: *why not
just RAG?* Until it is measured, every answer to that is reasoning rather than evidence,
and the reasoning cuts both ways -- RAG is cheaper, and the Infineon work found that
handing one model the whole document works fine until the document gets too long.

Two retrievers, because the usual argument against RAG on datasheets is specifically an
argument against *embeddings*: that "SR", "Vos" and "GBW" sitting beside numbers in a
table do not embed near a query like "slew rate". If that is true, a lexical retriever
should beat a dense one here, which is a prediction worth testing rather than asserting.

Everything downstream of retrieval is shared with the single-call baseline -- same
prompt, same output contract, same parser, same grader. The only thing that varies is
which text the model is shown, which is what makes the comparison mean anything.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from datasheet_agent.extract import SYSTEM, Extraction, parse_response, single_call_prompt
from datasheet_agent.llm import Backend
from datasheet_agent.schema import FIELDS, FieldSpec

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
TOP_K_PER_FIELD = 3
MAX_CHUNKS = 20
"""Cap on the retrieved context. RAG's whole claim is that it is cheaper than reading
the document, so a baseline allowed to retrieve the document back is not a baseline."""

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of a datasheet, with the page it came from."""

    text: str
    page: int
    index: int

    def render(self) -> str:
        return f"--- PAGE {self.page} ---\n{self.text}"


def page_texts(pdf_path: Path) -> list[str]:
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return [page.get_text() for page in doc]


def chunk_pages(
    pages: list[str],
    size: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Fixed-size overlapping chunks, never spanning a page boundary.

    Fixed-size chunking is the thing being tested, so it is not quietly improved here:
    a table that runs longer than `size` gets cut, and a value is separated from the
    column header that says whether it is a minimum, a typical or a maximum. Keeping
    chunks inside one page is the one concession, and it only makes the baseline
    stronger.
    """
    chunks: list[Chunk] = []
    step = max(1, size - overlap)
    for page_number, text in enumerate(pages, start=1):
        cleaned = text.strip()
        if not cleaned:
            continue
        if len(cleaned) <= size:
            chunks.append(Chunk(cleaned, page_number, len(chunks)))
            continue
        for start in range(0, len(cleaned), step):
            piece = cleaned[start : start + size]
            if piece.strip():
                chunks.append(Chunk(piece, page_number, len(chunks)))
            if start + size >= len(cleaned):
                break
    return chunks


def field_query(spec: FieldSpec) -> str:
    """The query a RAG system would actually issue for this field.

    The field's own `search_terms` are included, exactly as the agent's tools get them.
    Withholding them would hand the agent an advantage that has nothing to do with
    agency, and the comparison is about retrieval strategy, not vocabulary.
    """
    return " ".join((spec.description, *spec.search_terms))


# ---- lexical retrieval --------------------------------------------------

_TOKEN = re.compile(r"[a-z0-9]+(?:[./][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 over the chunks of one datasheet.

    Written out rather than pulled in: it is thirty lines, it removes a dependency from
    a baseline whose whole job is to be unimpeachable, and it makes the scoring visible
    to anyone who wants to check that the lexical baseline was not hobbled.
    """

    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._docs = [tokenize(chunk.text) for chunk in chunks]
        self._lengths = [len(doc) for doc in self._docs]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._docs else 0.0
        self._counts = [Counter(doc) for doc in self._docs]
        document_frequency: Counter[str] = Counter()
        for doc in self._docs:
            document_frequency.update(set(doc))
        total = len(self._docs)
        self._idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = [0.0] * len(self.chunks)
        if not self._docs:
            return out
        for index, counts in enumerate(self._counts):
            length = self._lengths[index] or 1
            total = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self._avg_len or 1)
                )
                total += self._idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            out[index] = total
        return out


# ---- dense retrieval ----------------------------------------------------

_MODEL_LOCK = threading.Lock()
_MODEL: object | None = None


def _embedder() -> object:
    """One shared sentence-transformer. Loading it per part would dominate the run."""
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            from sentence_transformers import SentenceTransformer

            _MODEL = SentenceTransformer(EMBED_MODEL)
        return _MODEL


class DenseRetriever:
    """Cosine similarity over MiniLM embeddings -- RAG as it is usually built."""

    name = "dense"

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        model = _embedder()
        self._matrix = model.encode(  # type: ignore[attr-defined]
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def scores(self, query: str) -> list[float]:
        if not self.chunks:
            return []
        model = _embedder()
        vector = model.encode(  # type: ignore[attr-defined]
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return (self._matrix @ vector).tolist()


RETRIEVERS = {"bm25": BM25Retriever, "dense": DenseRetriever}


# ---- the baseline -------------------------------------------------------


def retrieve(
    retriever: BM25Retriever | DenseRetriever,
    specs: tuple[FieldSpec, ...],
    top_k: int = TOP_K_PER_FIELD,
    max_chunks: int = MAX_CHUNKS,
) -> list[Chunk]:
    """Top-k chunks for every field, merged and capped.

    One query per field rather than one for the whole schema: a single query covering
    nine unrelated parameters retrieves for none of them, and a RAG baseline should be
    beaten at its best, not at its worst.
    """
    best: dict[int, float] = {}
    for spec in specs:
        scored = sorted(
            enumerate(retriever.scores(field_query(spec))),
            key=lambda pair: pair[1],
            reverse=True,
        )
        for index, score in scored[:top_k]:
            if score <= 0:
                continue
            if score > best.get(index, float("-inf")):
                best[index] = score

    ranked = sorted(best, key=lambda index: best[index], reverse=True)[:max_chunks]
    # Restore reading order: a model shown page 7 before page 2 is being tested on
    # something other than retrieval.
    return [retriever.chunks[index] for index in sorted(ranked)]


def extract_rag(
    backend: Backend,
    part_number: str,
    pdf_path: Path,
    retriever_name: str = "bm25",
    specs: tuple[FieldSpec, ...] = FIELDS,
    top_k: int = TOP_K_PER_FIELD,
    max_chunks: int = MAX_CHUNKS,
) -> Extraction:
    """Chunk, retrieve, answer once -- the comparison the agents have to beat."""
    chunks = chunk_pages(page_texts(pdf_path))
    if not chunks:
        return Extraction(part_number, {}, error="no extractable text")

    retriever = RETRIEVERS[retriever_name](chunks)  # type: ignore[abstract]
    selected = retrieve(retriever, specs, top_k=top_k, max_chunks=max_chunks)
    context = "\n\n".join(chunk.render() for chunk in selected)

    reply = backend.chat(
        SYSTEM,
        [{"role": "user", "content": single_call_prompt(part_number, specs, context)}],
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
