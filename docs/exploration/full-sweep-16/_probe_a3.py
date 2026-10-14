"""Third round: pin nsimplify's exact line, the open() path mangling, and
compile() semantics. Read-only (opens the probe file itself, nothing else).
"""
from __future__ import annotations

import inspect
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

print("sympy", sympy.__version__)

print("\n===== R1 nsimplify source, lines 40-70 =====")
for i, line in enumerate(inspect.getsource(sympy.nsimplify).splitlines(), 1):
    if 38 <= i <= 72:
        print(f"   {i:3d}: {line}")

print("\n===== R2 open() primitive via forward-slash path (read-only) =====")
SELF_FWD = os.path.abspath(__file__).replace("\\", "/")
print("path :", SELF_FWD)
print("size :", os.path.getsize(__file__))
OPEN = (
    "__import__(" + chr(39) + "builtins" + chr(39)
    + ").open(" + chr(34) + SELF_FWD + chr(34) + ").read()"
)
for expr, label in [
    (OPEN, "open().read() raw (string -> sympify fails)"),
    ("len(" + OPEN + ")", "len(open().read())  -> integer, looks like a normal result"),
    ("len(" + OPEN + ") * 1", "len * 1"),
]:
    try:
        r = parse_expr(expr, evaluate=True)
        print(f"   {label:52s} -> {r.__class__.__name__} = {str(r)[:50]}")
    except Exception as exc:  # noqa: BLE001
        print(f"   {label:52s} RAISED {type(exc).__name__}: {str(exc)[:90]}")

mt = importlib.import_module("ptools.math_tool")
print("   -- through solve_math() (the actual tool surface) --")
try:
    print("   ", mt.solve_math(operation="evaluate", expression="len(" + OPEN + ")")[:150])
except Exception as exc:  # noqa: BLE001
    print("   UNCAUGHT", type(exc).__name__, exc)

print("\n===== R3 compile() semantics under the sandbox =====")
ns = {}
exec("from sympy import *", ns)
ns["__builtins__"] = {}
P = "__import__(" + chr(39) + "os" + chr(39) + ").getppid()"
print("real parent pid =", os.getppid())
for code in [
    "compile(" + chr(34) + P + chr(34) + ",0," + chr(39) + "eval" + chr(39) + ")",
    "exec(compile(" + chr(34) + P + chr(34) + ",0," + chr(39) + "eval" + chr(39) + "))",
    "eval(compile(" + chr(34) + P + chr(34) + ",0," + chr(39) + "eval" + chr(39) + "))",
    "nsimplify(" + chr(34) + P + chr(34) + ")",
    "nsimplify(" + chr(34) + "1/2" + chr(34) + ")",
]:
    try:
        r = parse_expr(code, global_dict=ns, evaluate=True)
        print(f"   {code[:52]:54s} -> {r.__class__.__name__} = {str(r)[:40]}")
    except Exception as exc:  # noqa: BLE001
        print(f"   {code[:52]:54s} RAISED {type(exc).__name__}: {str(exc)[:70]}")

print("\n===== R4 unsandboxed control =====")
for code in [
    "compile(" + chr(34) + P + chr(34) + ",0," + chr(39) + "eval" + chr(39) + ")",
    "eval(compile(" + chr(34) + P + chr(34) + ",0," + chr(39) + "eval" + chr(39) + "))",
    "nsimplify(" + chr(34) + P + chr(34) + ")",
]:
    try:
        r = parse_expr(code, evaluate=True)
        print(f"   {code[:52]:54s} -> {r.__class__.__name__} = {str(r)[:40]}")
    except Exception as exc:  # noqa: BLE001
        print(f"   {code[:52]:54s} RAISED {type(exc).__name__}: {str(exc)[:70]}")

print("\nDONE")
