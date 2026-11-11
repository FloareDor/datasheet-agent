"""End-to-end check of the deployed demo: upload a real datasheet, print the specs.

    python scripts/smoke_demo.py --index 2

Uses a part from the test split so the returned values can be read against known
labels -- a demo that returns plausible-looking nonsense should not pass as working.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import urllib.request
import uuid
from pathlib import Path

from datasheet_agent.dataset import read_jsonl

BASE = "https://sairavig--datasheet-agent-demo-web.us-east.modal.run"
SPLITS = Path("data/splits")


def _multipart(fields: dict[str, str], pdf_path: Path) -> tuple[bytes, str]:
    """Build a multipart/form-data body without pulling in `requests`."""
    boundary = uuid.uuid4().hex
    marker = f"--{boundary}".encode()
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks += [
            marker,
            f'Content-Disposition: form-data; name="{name}"'.encode(),
            b"",
            value.encode(),
        ]
    content_type = mimetypes.guess_type(pdf_path.name)[0] or "application/pdf"
    chunks += [
        marker,
        f'Content-Disposition: form-data; name="file"; filename="{pdf_path.name}"'.encode(),
        f"Content-Type: {content_type}".encode(),
        b"",
        pdf_path.read_bytes(),
        f"--{boundary}--".encode(),
        b"",
    ]
    return b"\r\n".join(chunks), f"multipart/form-data; boundary={boundary}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=2)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--no-verifier", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    part = read_jsonl(SPLITS / "test.jsonl")[args.index]
    print(f"part   : {part.part_number}")
    print(f"pdf    : {part.pdf.name}\n")

    body, content_type = _multipart(
        {
            "part_number": part.part_number,
            "verifier": "false" if args.no_verifier else "true",
        },
        part.pdf,
    )
    request = urllib.request.Request(
        f"{args.base}/extract", data=body, headers={"Content-Type": content_type}
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:  # noqa: S310
        payload = json.loads(response.read())

    print(f"{'field':26s} {'extracted':22s} {'label':16s} match")
    for field, value in payload["values"].items():
        label = part.labels.get(field, "")
        mark = "-" if not label else ("ok" if _looks_same(field, value, label) else "X")
        print(f"{field:26s} {value or '(none)':22s} {label or '(no label)':16s} {mark}")

    if payload.get("rejected"):
        print("\nverifier rejected:")
        for rejection in payload["rejected"]:
            print(f"  {rejection['field']} [{rejection['stage']}]: {rejection['reason'][:90]}")
    print(f"\nrepaired: {payload.get('repaired_fields')}")
    print(f"tool calls: {payload['tool_calls']}   usage: {payload['usage']}")


def _looks_same(field: str, predicted: str, label: str) -> bool:
    from datasheet_agent.grading import grade_field
    from datasheet_agent.schema import FIELDS_BY_NAME

    spec = FIELDS_BY_NAME.get(field)
    return bool(spec) and grade_field(spec, predicted, label).correct


if __name__ == "__main__":
    main()
