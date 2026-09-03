from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.config import Settings

# Supported symbolic operations, surfaced to the LLM in the field description.
_OPERATIONS = (
    "derivative",
    "integral",
    "simplify",
    "solve",
    "evaluate",
    "limit",
    "series",
)


class MathInput(BaseModel):
    """Input schema for the symbolic math tool."""

    operation: str = Field(
        ...,
        description=(
            "What to compute: 'derivative', 'integral', 'simplify', "
            "'solve' (solve equation for the variable), 'evaluate' "
            "(numeric value), 'limit' (limit as the variable approaches a "
            "point), or 'series' (Taylor/Maclaurin series expansion)."
        ),
    )
    expression: str = Field(
        ...,
        min_length=1,
        description=(
            "Math expression in Python/SymPy syntax, e.g. 'x**2 + 3*x', "
            "'sin(x)*exp(x)', 'x**2 - 4'. Use '**' for powers. For 'solve', "
            "give the equation expression assumed equal to 0 (e.g. 'x**2 - 4') "
            "or use 'lhs = rhs' form."
        ),
    )
    variable: str = Field(
        default="x",
        description="Variable to differentiate, integrate, solve, or expand for.",
    )
    lower_bound: str | None = Field(
        default=None,
        description="Optional lower limit for a definite integral.",
    )
    upper_bound: str | None = Field(
        default=None,
        description="Optional upper limit for a definite integral.",
    )
    point: str | None = Field(
        default=None,
        description=(
            "For 'limit': the point the variable approaches, e.g. '0', 'oo' "
            "(infinity), '-oo'. For 'series': the expansion point (default 0)."
        ),
    )
    order: int = Field(
        default=6,
        ge=1,
        le=20,
        description="For 'series': number of terms / truncation order.",
    )
    direction: str = Field(
        default="+",
        description="For 'limit': approach direction '+' (right) or '-' (left).",
    )


def build_math_tool(
    _settings: Settings,
) -> BaseTool:
    """Create a SymPy-backed symbolic math tool."""

    def _run_math(
        operation: str,
        expression: str,
        variable: str = "x",
        lower_bound: str | None = None,
        upper_bound: str | None = None,
        point: str | None = None,
        order: int = 6,
        direction: str = "+",
    ) -> str:
        return solve_math(
            operation=operation,
            expression=expression,
            variable=variable,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            point=point,
            order=order,
            direction=direction,
        )

    return StructuredTool.from_function(
        func=_run_math,
        name="solve_math",
        description=(
            "Solve symbolic math problems exactly: derivatives, indefinite and "
            "definite integrals, equation solving, expression simplification, "
            "numeric evaluation, limits, and Taylor/Maclaurin series "
            "expansions. Use for calculus (differentiation, integration, "
            "limits, series), algebra (solving equations, simplifying), and "
            "exact arithmetic. Returns the exact symbolic result."
        ),
        args_schema=MathInput,
    )


def solve_math(
    *,
    operation: str,
    expression: str,
    variable: str = "x",
    lower_bound: str | None = None,
    upper_bound: str | None = None,
    point: str | None = None,
    order: int = 6,
    direction: str = "+",
) -> str:
    op = (operation or "").strip().lower()
    if op not in _OPERATIONS:
        return (
            f"Unsupported operation {operation!r}. "
            f"Use one of: {', '.join(_OPERATIONS)}."
        )

    raw_expr = (expression or "").strip()
    if not raw_expr:
        return "Math tool requires a non-empty expression."

    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except ImportError as exc:  # pragma: no cover - sympy ships with the deps
        return (
            "SymPy is not installed. Run `python -m pip install sympy` to "
            f"enable the math tool: {exc}"
        )

    transformations = standard_transformations + (
        implicit_multiplication_application,
    )

    def _parse(text: str) -> Any:
        # parse_expr is SymPy's safe parser: no Python eval of arbitrary code.
        return parse_expr(
            text,
            transformations=transformations,
            evaluate=True,
        )

    try:
        var = sympy.Symbol(variable.strip() or "x")
        if op == "solve":
            return _do_solve(sympy, _parse, raw_expr, var)
        expr = _parse(raw_expr)
        if op == "derivative":
            return _do_derivative(sympy, expr, var)
        if op == "integral":
            return _do_integral(sympy, _parse, expr, var, lower_bound, upper_bound)
        if op == "simplify":
            return f"simplify({raw_expr}) = {sympy.simplify(expr)}"
        if op == "limit":
            return _do_limit(sympy, _parse, expr, var, point, direction)
        if op == "series":
            return _do_series(sympy, _parse, expr, var, point, order)
        # evaluate
        return _do_evaluate(sympy, expr, raw_expr)
    except Exception as exc:
        return f"Math evaluation failed for {raw_expr!r}: {exc}"


def _do_derivative(sympy: Any, expr: Any, var: Any) -> str:
    result = sympy.diff(expr, var)
    return f"d/d{var}({expr}) = {result}"


def _do_integral(
    sympy: Any,
    parse: Any,
    expr: Any,
    var: Any,
    lower_bound: str | None,
    upper_bound: str | None,
) -> str:
    has_lower = lower_bound is not None and str(lower_bound).strip() != ""
    has_upper = upper_bound is not None and str(upper_bound).strip() != ""
    if has_lower and has_upper:
        low = parse(str(lower_bound))
        high = parse(str(upper_bound))
        result = sympy.integrate(expr, (var, low, high))
        return f"integral of {expr} d{var} from {low} to {high} = {result}"
    # Indefinite integral; note the constant of integration for the reader.
    result = sympy.integrate(expr, var)
    return f"integral of {expr} d{var} = {result} + C"


def _do_solve(sympy: Any, parse: Any, raw_expr: str, var: Any) -> str:
    if "=" in raw_expr:
        lhs_text, _, rhs_text = raw_expr.partition("=")
        equation = sympy.Eq(parse(lhs_text), parse(rhs_text))
    else:
        equation = sympy.Eq(parse(raw_expr), 0)
    solutions = sympy.solve(equation, var)
    if not solutions:
        return f"No solution found for {raw_expr} (variable {var})."
    rendered = ", ".join(str(sol) for sol in solutions)
    return f"solve({raw_expr}, {var}) -> {var} = {rendered}"


def _do_limit(
    sympy: Any,
    parse: Any,
    expr: Any,
    var: Any,
    point: str | None,
    direction: str,
) -> str:
    point_text = (point or "0").strip() or "0"
    target = parse(point_text)
    dir_flag = "-" if str(direction).strip() == "-" else "+"
    result = sympy.limit(expr, var, target, dir_flag)
    return f"limit of {expr} as {var}->{point_text} ({dir_flag}) = {result}"


def _do_series(
    sympy: Any,
    parse: Any,
    expr: Any,
    var: Any,
    point: str | None,
    order: int,
) -> str:
    point_text = (point or "0").strip() or "0"
    around = parse(point_text)
    n = max(1, int(order))
    result = sympy.series(expr, var, around, n)
    return f"series of {expr} around {var}={point_text} (order {n}) = {result}"


def _do_evaluate(sympy: Any, expr: Any, raw_expr: str) -> str:
    free_symbols: Any = getattr(expr, "free_symbols", set())
    if free_symbols:
        names = ", ".join(sorted(str(sym) for sym in free_symbols))
        return (
            f"Cannot evaluate {raw_expr} numerically: it still contains "
            f"unknown symbol(s) {names}. Provide concrete values or use "
            f"'simplify' instead."
        )
    value = sympy.N(expr)
    return f"{raw_expr} = {value}"


__all__ = [
    "MathInput",
    "build_math_tool",
    "solve_math",
]
