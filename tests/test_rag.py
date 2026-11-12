"""The retrieval baselines, which exist to make "why not just RAG?" a measured question.

A baseline is only worth the comparison if it is built honestly, so what is tested here
is mostly that it has not been quietly hobbled: chunks cover the document, the queries
carry the same vocabulary the agent's tools get, and retrieval returns chunks in reading
order.
"""

from __future__ import annotations

from datasheet_agent.rag import (
    MAX_CHUNKS,
    BM25Retriever,
    chunk_pages,
    field_query,
    retrieve,
    tokenize,
)
from datasheet_agent.schema import FIELDS, FIELDS_BY_NAME

PAGES = [
    "ABSOLUTE MAXIMUM RATINGS Supply Voltage 36 V Operating Temperature -40 to 125 C",
    "ELECTRICAL CHARACTERISTICS SR Slew Rate AV = 1 0.4 V/us GBW Gain Bandwidth 1.1 MHz",
    "PACKAGE INFORMATION SOIC-8 surface mount dual operational amplifier",
]


def test_every_page_with_text_is_represented() -> None:
    chunks = chunk_pages(PAGES)
    assert {chunk.page for chunk in chunks} == {1, 2, 3}


def test_blank_pages_are_dropped_rather_than_retrieved() -> None:
    assert [chunk.page for chunk in chunk_pages(["text", "   ", "more"])] == [1, 3]


def test_long_pages_are_split_with_overlap_and_lose_nothing() -> None:
    page = "".join(f"{index:04d} " for index in range(900))  # ~4500 chars
    chunks = chunk_pages([page], size=1000, overlap=200)
    assert len(chunks) > 1
    assert "".join(chunk.text for chunk in chunks).replace(" ", "").find("0899") != -1


def test_a_field_query_carries_the_same_terms_the_agent_tools_get() -> None:
    """Withholding them would make the agent look good for the wrong reason."""
    spec = FIELDS_BY_NAME["slew_rate"]
    query = field_query(spec).lower()
    for term in spec.search_terms:
        assert term.lower() in query


def test_bm25_puts_the_page_that_states_the_parameter_first() -> None:
    chunks = chunk_pages(PAGES)
    retriever = BM25Retriever(chunks)
    scores = retriever.scores(field_query(FIELDS_BY_NAME["slew_rate"]))
    best = chunks[max(range(len(chunks)), key=lambda index: scores[index])]
    assert "Slew Rate" in best.text


def test_bm25_scores_a_query_with_no_shared_terms_at_zero() -> None:
    retriever = BM25Retriever(chunk_pages(PAGES))
    assert max(retriever.scores("zylophone quantum parsec")) == 0.0


def test_retrieval_returns_chunks_in_reading_order() -> None:
    """A model shown page 7 before page 2 is being tested on something else."""
    chunks = chunk_pages(PAGES)
    selected = retrieve(BM25Retriever(chunks), FIELDS)
    assert [chunk.index for chunk in selected] == sorted(chunk.index for chunk in selected)


def test_retrieval_is_capped_so_it_cannot_retrieve_the_document_back() -> None:
    """RAG's claim is that it is cheaper than reading the datasheet. A baseline that
    returns everything is not that baseline, and would win for the wrong reason."""
    pages = [f"page {index} slew rate supply voltage package {index}" for index in range(200)]
    selected = retrieve(BM25Retriever(chunk_pages(pages)), FIELDS)
    assert len(selected) <= MAX_CHUNKS


def test_an_empty_document_retrieves_nothing_rather_than_failing() -> None:
    assert retrieve(BM25Retriever([]), FIELDS) == []


def test_tokenizer_keeps_the_symbols_datasheets_use() -> None:
    """"V/us" and "4.5" are the tokens that matter here; splitting them loses the row."""
    assert "v/us" in tokenize("SR 0.4 V/us")
    assert "4.5" in tokenize("supply 4.5 V")


# ---- the coverage decomposition ----------------------------------------

import sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "scripts"))

from coverage_report import answer_is_present  # noqa: E402


def test_a_value_stated_in_the_context_counts_as_found() -> None:
    assert answer_is_present("slew_rate", "0.4 V/us", "SR Slew Rate AV = 1 0.4 V/us")


def test_a_value_the_context_never_states_does_not() -> None:
    """If this ever returns True for absent values, every coverage number is garbage."""
    assert not answer_is_present("slew_rate", "20 V/us", "SR Slew Rate AV = 1 0.4 V/us")


def test_a_value_in_another_unit_still_counts_as_found() -> None:
    assert answer_is_present("gain_bandwidth_product", "1.1 MHz", "GBW 1100 kHz over range")


def test_presence_is_an_upper_bound_not_a_claim_about_the_right_row() -> None:
    """Documented bias: a bare number anywhere counts, even under another parameter.

    The metric can only flatter navigation, which is the safe direction for the
    conclusion drawn from it -- that navigation is not where these methods fail.
    """
    assert answer_is_present("slew_rate", "20 V/us", "Transition Frequency fT 20 MHz, 20")


def test_an_empty_context_finds_nothing() -> None:
    assert not answer_is_present("slew_rate", "0.4 V/us", "")
