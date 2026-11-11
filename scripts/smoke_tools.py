"""Manual smoke test: run the three tools against one datasheet and print results."""

from __future__ import annotations

import sys
from pathlib import Path

from datasheet_agent.tools import DatasheetToolbox

DEFAULT_PDF = Path(
    "data/raw/mouser_opamps/mouser/pdf/AnalogDevicesLinearTechnology_LT1122CCN8PBF.pdf"
)


def main() -> None:
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    with DatasheetToolbox(pdf, Path("data/cache")) as toolbox:
        print("=== orientation ===")
        print(toolbox.orientation()[:600])
        print("\n=== search: Slew Rate ===")
        print(toolbox.search("Slew Rate", 3)[:600])
        print("\n=== read_table p3 ===")
        print(toolbox.read_table(3)[:800])
        print("\ncalls:", [c.name for c in toolbox.calls])


if __name__ == "__main__":
    main()
