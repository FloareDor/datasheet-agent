"""Building the labelled dataset: catalogue rows joined to datasheet PDFs.

The two sides name parts differently. Mouser gives an orderable part number
("OPA2991IDR"); the archive names each PDF by the manufacturer's document code
("ad8656.pdf", "1637fd.pdf"). Three joins run in order of decreasing confidence:

1. **filename** - the 100 PDFs shipped with the catalogue are named
   `Manufacturer_PartNumber`, so the part number is the filename. Exact.
2. **stem prefix** - the archive stem is a prefix of the orderable part number
   ("opa388.pdf" for "OPA388IDBVT"). The suffix is package and packaging codes.
3. **title token** - the part number is named in the document's own title. This is
   what reaches family datasheets, where one document covers TL061/TL062/TL064.

The third join is the risky one, so it is deliberately narrow: the token must appear
in the *title*, where a datasheet names the family it documents, and never in body
text, where datasheets reference parts they are not about. A token claimed by more
than one document is dropped rather than guessed at.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from datasheet_agent.schema import FIELDS

MIN_TOKEN_CHARS = 5
TITLE_CHARS = 20_000
PART_TOKEN = re.compile(r"\b([A-Z]{1,4}\d{2,5}[A-Z0-9]{0,6})\b")

JoinMethod = str  # "filename" | "stem_prefix" | "title_token"


@dataclass(frozen=True)
class LabelledPart:
    """One op-amp: its datasheet, its catalogue labels, and how the two were joined."""

    part_number: str
    manufacturer: str
    pdf_path: str
    join_method: JoinMethod
    labels: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> LabelledPart:
        return cls(
            part_number=str(data["part_number"]),
            manufacturer=str(data["manufacturer"]),
            pdf_path=str(data["pdf_path"]),
            join_method=str(data["join_method"]),
            labels={k: str(v) for k, v in dict(data["labels"]).items()},  # type: ignore[arg-type]
        )

    @property
    def pdf(self) -> Path:
        return Path(self.pdf_path)

    def labelled_fields(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.labels.items() if value)


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


# ---- catalogue side -----------------------------------------------------


def load_catalogue(csv_dir: Path) -> list[dict[str, str]]:
    """Read every catalogue CSV, de-duplicated on orderable part number."""
    rows: list[dict[str, str]] = []
    for path in sorted(csv_dir.glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))

    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        part = (row.get("Mfr Part Number") or "").strip()
        if part and part not in unique:
            unique[part] = row
    return list(unique.values())


def extract_labels(row: dict[str, str]) -> dict[str, str]:
    """Pull the schema's fields out of a catalogue row, as raw catalogue strings."""
    from datasheet_agent.units import clean_text

    return {spec.name: clean_text(row.get(spec.label_column)) for spec in FIELDS}


# ---- datasheet side -----------------------------------------------------


def document_title(html_path: Path) -> str:
    try:
        raw = html_path.read_text(encoding="utf-8", errors="replace")[:TITLE_CHARS]
    except OSError:
        return ""
    match = re.search(r"<title>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def build_title_index(html_paths: Iterable[Path]) -> dict[str, Path]:
    """Map each part-number token named in a title to its one document.

    Tokens claimed by more than one document are dropped: a guess between two
    datasheets is worse than no label at all.
    """
    claims: dict[str, set[Path]] = {}
    for html in html_paths:
        title = document_title(html)
        for token in PART_TOKEN.findall(title.upper()):
            key = normalise(token)
            if len(key) >= MIN_TOKEN_CHARS:
                claims.setdefault(key, set()).add(html)
    return {token: next(iter(docs)) for token, docs in claims.items() if len(docs) == 1}


def _pdf_for_html(html_path: Path) -> Path | None:
    """The PDF beside an HTML file. The archive varies the extension case."""
    pdf_dir = html_path.parent.parent / "pdf"
    for suffix in (".pdf", ".PDF"):
        candidate = pdf_dir / (html_path.stem + suffix)
        if candidate.exists():
            return candidate
    return None


# ---- the join -----------------------------------------------------------


def _iter_pdfs(roots: Iterable[Path]) -> Iterator[Path]:
    for root in roots:
        if root.exists():
            yield from (p for p in root.rglob("*") if p.suffix.lower() == ".pdf")


def build_dataset(
    catalogue_csv_dir: Path,
    catalogue_pdf_dir: Path,
    archive_root: Path,
    archive_html_dirs: Iterable[Path] | None = None,
) -> list[LabelledPart]:
    """Join catalogue rows to datasheet PDFs. Returns one record per joined part."""
    catalogue = load_catalogue(catalogue_csv_dir)

    by_filename = {normalise(p.stem.split("_", 1)[-1]): p for p in _iter_pdfs([catalogue_pdf_dir])}
    archive_pdfs = list(_iter_pdfs([archive_root]))
    # Longest stem first, so "opa2188" wins over "opa218" for "OPA2188AQDGKRQ1".
    stems = sorted(
        ((normalise(p.stem), p) for p in archive_pdfs if len(normalise(p.stem)) >= MIN_TOKEN_CHARS),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    html_dirs = (
        list(archive_html_dirs)
        if archive_html_dirs is not None
        else [d for d in archive_root.rglob("html") if d.is_dir()]
    )
    title_index = build_title_index(
        html for directory in html_dirs for html in directory.glob("*.html")
    )

    parts: list[LabelledPart] = []
    for row in catalogue:
        part_number = row["Mfr Part Number"].strip()
        key = normalise(part_number)
        pdf, method = _match_one(key, by_filename, stems, title_index)
        if pdf is None:
            continue
        labels = extract_labels(row)
        if not any(labels.values()):
            continue  # a row with no usable labels teaches nothing
        parts.append(
            LabelledPart(
                part_number=part_number,
                manufacturer=(row.get("Mfr.") or "").strip(),
                pdf_path=pdf.as_posix(),
                join_method=method,
                labels=labels,
            )
        )
    return parts


def _consumes_type_number(key: str, prefix: str) -> bool:
    """True when `prefix` covers the part's whole numeric type, not half of it.

    Without this, "opa234" matches "OPA2342EA" -- two different op-amps -- because
    one type number is a prefix of another. Requiring the next character not to be a
    digit makes the prefix stop at a type boundary.
    """
    if not key.startswith(prefix):
        return False
    rest = key[len(prefix) :]
    return not (rest and rest[0].isdigit())


def _match_one(
    key: str,
    by_filename: dict[str, Path],
    stems: list[tuple[str, Path]],
    title_index: dict[str, Path],
) -> tuple[Path | None, JoinMethod]:
    if (exact := by_filename.get(key)) is not None:
        return exact, "filename"
    for stem, pdf in stems:
        if _consumes_type_number(key, stem):
            return pdf, "stem_prefix"
    # Longest prefix of the part number that some title names, so a family
    # datasheet is reached by the most specific token available.
    for end in range(len(key), MIN_TOKEN_CHARS - 1, -1):
        if not _consumes_type_number(key, key[:end]):
            continue
        html = title_index.get(key[:end])
        if html is not None and (pdf := _pdf_for_html(html)) is not None:
            return pdf, "title_token"
    return None, ""


# ---- splits and persistence --------------------------------------------


def split_dataset(
    parts: list[LabelledPart],
    train_size: int,
    test_size: int,
    seed: int = 0,
) -> tuple[list[LabelledPart], list[LabelledPart]]:
    """Split by datasheet, not by part.

    A family datasheet covers several orderable parts. If TL062 landed in train and
    TL064 in test, the test set would be measuring recall of a document the system had
    already been tuned on. Grouping by PDF keeps the split honest.
    """
    by_pdf: dict[str, list[LabelledPart]] = {}
    for part in parts:
        by_pdf.setdefault(part.pdf_path, []).append(part)

    groups = sorted(by_pdf.values(), key=lambda g: g[0].pdf_path)
    random.Random(seed).shuffle(groups)

    test: list[LabelledPart] = []
    train: list[LabelledPart] = []
    for group in groups:
        if len(test) < test_size:
            test.extend(group)
        elif len(train) < train_size:
            train.extend(group)
        else:
            break
    return train[:train_size], test[:test_size]


def write_jsonl(parts: list[LabelledPart], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for part in parts:
            handle.write(json.dumps(part.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[LabelledPart]:
    with path.open(encoding="utf-8") as handle:
        return [LabelledPart.from_dict(json.loads(line)) for line in handle if line.strip()]
