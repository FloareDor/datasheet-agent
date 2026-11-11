"""The join and the split are where silent label corruption would enter."""

from __future__ import annotations

from pathlib import Path

from datasheet_agent.dataset import (
    LabelledPart,
    _consumes_type_number,
    normalise,
    split_dataset,
)

SPLITS = Path("data/splits")


def part(number: str, pdf: str) -> LabelledPart:
    return LabelledPart(number, "ACME", pdf, "stem_prefix", {"slew_rate": "1 V/us"})


def test_prefix_must_consume_the_whole_type_number() -> None:
    """ "opa234" must not claim "OPA2342EA" -- they are different op-amps."""
    assert not _consumes_type_number(normalise("OPA2342EA"), "opa234")
    assert not _consumes_type_number(normalise("LM6144BIM"), "lm614")
    assert _consumes_type_number(normalise("OPA2342EA"), "opa2342")
    assert _consumes_type_number(normalise("TL071CD"), "tl071")


def test_split_never_puts_one_datasheet_in_both_halves() -> None:
    """Family datasheets cover several parts; splitting by part would leak."""
    parts = [
        part("TL061CD", "a.pdf"),
        part("TL062CD", "a.pdf"),
        part("TL064CD", "a.pdf"),
        part("AD8656ARZ", "b.pdf"),
        part("OPA2991IDR", "c.pdf"),
        part("LT1122CCN8", "d.pdf"),
    ]
    train, test = split_dataset(parts, train_size=3, test_size=3, seed=1)
    assert not ({p.pdf_path for p in train} & {p.pdf_path for p in test})


def test_split_is_deterministic_for_a_seed() -> None:
    parts = [part(f"P{i}", f"{i}.pdf") for i in range(20)]
    first = split_dataset(parts, 5, 5, seed=7)
    second = split_dataset(parts, 5, 5, seed=7)
    assert [p.part_number for p in first[0]] == [p.part_number for p in second[0]]
    assert [p.part_number for p in first[1]] == [p.part_number for p in second[1]]


def test_labelled_fields_skips_blanks() -> None:
    record = LabelledPart("X", "ACME", "a.pdf", "filename", {"slew_rate": "1 V/us", "gbp": ""})
    assert record.labelled_fields() == ("slew_rate",)


def test_built_splits_are_disjoint_and_sized() -> None:
    """Guards the real artefacts, when they exist, against a bad rebuild."""
    import pytest

    from datasheet_agent.dataset import read_jsonl

    if not (SPLITS / "train.jsonl").exists():
        pytest.skip("dataset not built")
    train = read_jsonl(SPLITS / "train.jsonl")
    test = read_jsonl(SPLITS / "test.jsonl")
    assert len(train) == 300
    assert len(test) == 100
    assert not ({p.pdf_path for p in train} & {p.pdf_path for p in test})
    assert all(Path(p.pdf_path).exists() for p in test)
    assert all(p.labelled_fields() for p in train + test)
