"""The fields DatasheetAgent extracts, and how each one is graded.

Fields are partitioned into groups; one agent owns each group and the groups run in
parallel. The partition is by where the answer lives in a datasheet, not by data type:
electrical values sit in specification tables, thermal limits in absolute-maximum or
recommended-operating tables, package facts on the first page and in the package
section. Agents therefore search in genuinely different places.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FieldGroup = Literal["electrical", "thermal", "package"]
Grader = Literal["numeric", "categorical"]


@dataclass(frozen=True)
class FieldSpec:
    """One extractable spec: what to ask for, where it comes from, how to grade it."""

    name: str
    group: FieldGroup
    label_column: str
    """Column in the Mouser catalogue CSV that provides ground truth."""
    dimension: str
    """Unit dimension for numeric grading; "text" for categorical fields."""
    grader: Grader
    description: str
    """Shown to the agent. Phrased as an engineer would ask for the value."""
    unit_hint: str
    """Unit the agent is asked to answer in, to keep outputs comparable."""

    search_terms: tuple[str, ...]
    """How a datasheet spells this parameter.

    Without these, a model searches for the field's machine name
    ("supply_voltage_max") and finds nothing, because no datasheet contains that
    string. Supplying the vocabulary is part of designing the tool, not a hint about
    any particular answer -- every method gets the same list.
    """


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="supply_voltage_max",
        group="electrical",
        label_column="Supply Voltage - Max",
        dimension="voltage",
        grader="numeric",
        description=(
            "Maximum supply voltage the part operates at, as a total/single-supply "
            "figure (for a dual-supply part such as +-18 V, report 36 V)."
        ),
        unit_hint="V",
        search_terms=(
            "Supply Voltage",
            "VS",
            "VCC",
            "Absolute Maximum Ratings",
            "Supply Voltage Range",
        ),
    ),
    FieldSpec(
        name="gain_bandwidth_product",
        group="electrical",
        label_column="GBP - Gain Bandwidth Product",
        dimension="frequency",
        grader="numeric",
        description="Gain bandwidth product (GBP / GBW), typical value.",
        unit_hint="MHz",
        search_terms=("Gain Bandwidth Product", "GBW", "GBP", "Unity Gain Bandwidth", "Bandwidth"),
    ),
    FieldSpec(
        name="slew_rate",
        group="electrical",
        label_column="SR - Slew Rate",
        dimension="slew_rate",
        grader="numeric",
        description="Slew rate (SR), typical value.",
        unit_hint="V/us",
        search_terms=("Slew Rate", "SR"),
    ),
    FieldSpec(
        name="input_offset_voltage",
        group="electrical",
        label_column="Vos - Input Offset Voltage",
        dimension="voltage",
        grader="numeric",
        description="Input offset voltage (Vos), maximum value at 25 degC.",
        unit_hint="mV",
        search_terms=("Input Offset Voltage", "VOS", "Offset Voltage"),
    ),
    FieldSpec(
        name="operating_temp_min",
        group="thermal",
        label_column="Minimum Operating Temperature",
        dimension="temperature",
        grader="numeric",
        description="Lowest ambient operating temperature in the specified range.",
        unit_hint="degC",
        search_terms=(
            "Operating Temperature Range",
            "TA",
            "Ambient Temperature",
            "Specified Temperature Range",
        ),
    ),
    FieldSpec(
        name="operating_temp_max",
        group="thermal",
        label_column="Maximum Operating Temperature",
        dimension="temperature",
        grader="numeric",
        description="Highest ambient operating temperature in the specified range.",
        unit_hint="degC",
        search_terms=(
            "Operating Temperature Range",
            "TA",
            "Ambient Temperature",
            "Specified Temperature Range",
        ),
    ),
    FieldSpec(
        name="package_case",
        group="package",
        label_column="Package / Case",
        dimension="text",
        grader="categorical",
        description=(
            "Package / case of this exact orderable part, e.g. SOIC-8, TSSOP-14, "
            "SOT-23-5. Include the pin count."
        ),
        unit_hint="package name",
        search_terms=(
            "Ordering Information",
            "PACKAGE/ORDERING INFORMATION",
            "Package",
            "Pin Configuration",
            "SOIC",
            "TSSOP",
        ),
    ),
    FieldSpec(
        name="num_channels",
        group="package",
        label_column="Number of Channels",
        dimension="count",
        grader="numeric",
        description="Number of amplifier channels in the package (1, 2 or 4).",
        unit_hint="channels",
        search_terms=("Single", "Dual", "Quad", "Operational Amplifier", "amplifiers per package"),
    ),
    FieldSpec(
        name="mounting_style",
        group="package",
        label_column="Mounting Style",
        dimension="text",
        grader="categorical",
        description=(
            "Mounting style: 'SMD/SMT' for surface mount, 'Through Hole' for through-hole packages."
        ),
        unit_hint="SMD/SMT or Through Hole",
        search_terms=("Ordering Information", "Package", "SOIC", "PDIP", "Surface Mount"),
    ),
)

FIELDS_BY_NAME: dict[str, FieldSpec] = {f.name: f for f in FIELDS}
GROUPS: tuple[FieldGroup, ...] = ("electrical", "thermal", "package")


def search_vocabulary(specs: tuple[FieldSpec, ...]) -> str:
    """The search terms for a set of fields, as a prompt block."""
    return "\n".join(
        f"- {spec.name}: try " + ", ".join(f'"{t}"' for t in spec.search_terms) for spec in specs
    )


def fields_in_group(group: FieldGroup) -> tuple[FieldSpec, ...]:
    return tuple(f for f in FIELDS if f.group == group)
