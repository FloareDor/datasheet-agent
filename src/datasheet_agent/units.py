"""Parsing and normalisation of engineering quantities found in datasheets.

Every extracted value is reduced to a canonical float in a canonical unit so that a
model answer ("4.5 V/us") and a catalogue label ("4,500,000 V/s") compare equal.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# SI prefixes. Datasheets use "u" and "µ" interchangeably for micro.
_PREFIXES: dict[str, float] = {
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "µ": 1e-6,
    "μ": 1e-6,
    "m": 1e-3,
    "": 1.0,
    "k": 1e3,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
}

# Canonical unit per dimension; base units are SI (V, Hz, A, V/s, degC, dB).
BASE_UNIT: dict[str, str] = {
    "voltage": "V",
    "frequency": "Hz",
    "current": "A",
    "slew_rate": "V/s",
    "gain": "dB",
    "temperature": "degC",
    "count": "",
}

# Unit spellings accepted for each dimension, longest-match first.
_UNIT_ALIASES: dict[str, tuple[str, ...]] = {
    "voltage": ("volts", "volt", "v"),
    "frequency": ("hertz", "hz"),
    "current": ("amps", "amperes", "ampere", "amp", "a"),
    "slew_rate": ("v/us", "v/µs", "v/s", "v/ms", "v/ns", "v per us"),
    "gain": ("db",),
    "temperature": ("°c", "degc", "deg c", "c", "celsius"),
    "count": ("channels", "channel", "ch"),
}

# Slew rate carries its prefix on the time unit, so it gets its own table.
_SLEW_DENOM: dict[str, float] = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}

_NUMBER = r"[-+−]?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+−]?\s*\d*\.?\d+"


@dataclass(frozen=True)
class Quantity:
    """A scalar reduced to its dimension's base unit."""

    value: float
    dimension: str

    @property
    def unit(self) -> str:
        return BASE_UNIT[self.dimension]

    def __str__(self) -> str:
        return f"{self.value:g} {self.unit}".strip()


def clean_text(raw: str | None) -> str:
    """Strip spreadsheet armour (`="..."`), unicode minus and stray whitespace."""
    if raw is None:
        return ""
    text = str(raw).strip()
    text = re.sub(r'^=\s*"(.*)"$', r"\1", text).strip()
    text = text.replace("−", "-").replace("–", "-").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def _parse_number(token: str) -> float | None:
    token = token.replace(",", "").replace(" ", "")
    try:
        return float(token)
    except ValueError:
        return None


def _scale_for(dimension: str, prefix: str, unit: str) -> float | None:
    """Multiplier converting `prefix+unit` into the dimension's base unit."""
    if dimension == "slew_rate":
        numer = _PREFIXES.get(prefix)
        denom = _SLEW_DENOM.get(unit.split("/", 1)[1]) if "/" in unit else None
        if numer is None or denom is None:
            return None
        return numer / denom
    return _PREFIXES.get(prefix)


def unit_scale(unit_text: str, dimension: str) -> float | None:
    """Multiplier taking a value written in `unit_text` into the base unit.

    Used for `assume_unit`: the prompt tells the model which unit to answer in, so a
    bare "36" for a field asked for in volts means 36 V, and "0.4" for a field asked
    for in V/us means 0.4 V/us.
    """
    cleaned = unit_text.strip()
    for alias in sorted(_UNIT_ALIASES[dimension], key=len, reverse=True):
        if cleaned.lower().endswith(alias):
            prefix = cleaned[: len(cleaned) - len(alias)]
            scale = _scale_for(dimension, prefix, alias)
            if scale is not None:
                return scale
    return 1.0 if dimension in ("temperature", "count") else None


def parse_quantity(
    raw: str | None, dimension: str, assume_unit: str | None = None
) -> Quantity | None:
    """Parse the first quantity of `dimension` out of `raw`.

    `assume_unit` is the unit the answer was *asked* for. When the text carries no
    unit of its own, the bare number is read in that unit -- otherwise a model that
    answers "36" to a question posed in volts is marked wrong against a "36 V" label,
    which measures formatting rather than extraction.

    Returns None when nothing parseable is present, so callers can distinguish
    "no value" from "value zero".
    """
    text = clean_text(raw)
    if not text or dimension not in BASE_UNIT:
        return None

    if dimension == "count":
        match = re.search(_NUMBER, text)
        if match and (number := _parse_number(match.group())) is not None:
            return Quantity(number, dimension)
        return None

    lowered = text.lower()
    for alias in sorted(_UNIT_ALIASES[dimension], key=len, reverse=True):
        # Optional SI prefix immediately before the unit, e.g. "1.1 MHz", "10 pA".
        pattern = rf"({_NUMBER})\s*([fpnuµμmkKMGT]?){re.escape(alias)}\b"
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            number = _parse_number(match.group(1))
            if number is None:
                continue
            prefix = match.group(2)
            # "mV" is milli-volt but "MHz" is mega-hertz: recover case from the source.
            start = match.start(2)
            if start < len(text):
                prefix = text[start : start + len(prefix)]
            scale = _scale_for(dimension, prefix, alias)
            if scale is None:
                continue
            return Quantity(number * scale, dimension)

    # Bare number with no unit: unambiguous for temperature, and for any field whose
    # answer unit the prompt specified.
    scale = 1.0 if dimension == "temperature" else None
    if assume_unit:
        scale = unit_scale(assume_unit, dimension)
    if scale is not None:
        match = re.search(_NUMBER, text)
        if match and (number := _parse_number(match.group())) is not None:
            return Quantity(number * scale, dimension)
    return None


def quantities_match(a: Quantity | None, b: Quantity | None, rel_tol: float = 0.05) -> bool:
    """True when two quantities agree within `rel_tol` (absolute near zero)."""
    if a is None or b is None:
        return False
    if a.dimension != b.dimension:
        return False
    if a.dimension == "temperature":
        return math.isclose(a.value, b.value, abs_tol=2.0)
    return math.isclose(a.value, b.value, rel_tol=rel_tol, abs_tol=1e-15)
