"""Second join strategy: match labels to the part numbers a datasheet names itself.

Filename matching misses two common cases. Manufacturers name documents by internal
code ("1637fd.pdf" is the LT1637 datasheet), and one datasheet often covers a whole
family ("lt1637" also documents LT1638 and LT1639). Both are recoverable from the
document's own title and first page, which the archive ships as HTML.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts")
from probe_join import load_labels, norm  # noqa: E402

ARCHIVE = Path("data/raw/opamp_dataset")
MOUSER = Path("data/raw/mouser_opamps/mouser")
HEAD_CHARS = 6_000

# A part-number-ish token: letters, then digits, then optional suffix.
TOKEN = re.compile(r"\b([A-Z]{1,4}\d{2,5}[A-Z0-9]{0,6})\b")


def document_tokens(html_path: Path) -> set[str]:
    """Part-number-like tokens from a datasheet's title and opening text."""
    try:
        raw = html_path.read_text(encoding="utf-8", errors="replace")[:HEAD_CHARS]
    except OSError:
        return set()
    title = re.search(r"<title>(.*?)</title>", raw, re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", raw)
    if title:
        text = title.group(1) + " " + text
    return {norm(m) for m in TOKEN.findall(text.upper()) if len(m) >= 5}


def main() -> None:
    labels = load_labels()
    html_files = list(ARCHIVE.rglob("*.html")) + list((MOUSER / "html").glob("*.html"))
    print(f"labelled parts: {len(labels)}   html docs: {len(html_files)}")

    # token -> candidate documents. A token claimed by many documents is a false
    # friend (a reference to another part), so those are dropped below.
    index: dict[str, set[Path]] = {}
    for html in html_files:
        for token in document_tokens(html):
            index.setdefault(token, set()).add(html)

    ambiguous = {t for t, docs in index.items() if len(docs) > 3}
    print(f"tokens indexed: {len(index)}   dropped as ambiguous: {len(ambiguous)}")

    matched: dict[str, Path] = {}
    for row in labels:
        part = norm(row["Mfr Part Number"])
        # Longest prefix of the part number that some document names.
        for end in range(len(part), 4, -1):
            token = part[:end]
            docs = index.get(token)
            if docs and token not in ambiguous:
                matched[row["Mfr Part Number"]] = sorted(docs)[0]
                break

    print(f"C. in-document token join: {len(matched)}")

    stems = {norm(p.stem) for p in ARCHIVE.rglob("*.pdf")}
    stems |= {norm(p.stem.split("_", 1)[-1]) for p in MOUSER.glob("pdf/*.pdf")}
    old = {
        r["Mfr Part Number"]
        for r in labels
        if any(norm(r["Mfr Part Number"]).startswith(s) for s in stems if len(s) >= 5)
    }
    print(f"   previous join: {len(old)}   union: {len(old | set(matched))}")
    print(f"   new from C   : {len(set(matched) - old)}")

    print("\nsample new matches:")
    for part in list(set(matched) - old)[:12]:
        print(f"  {part:22s} -> {matched[part].name}")

    still = [r for r in labels if r["Mfr Part Number"] not in (old | set(matched))]
    print(f"\nstill unmatched: {len(still)}")
    print("by mfr:", Counter(r["Mfr."] for r in still).most_common(6))


if __name__ == "__main__":
    main()
