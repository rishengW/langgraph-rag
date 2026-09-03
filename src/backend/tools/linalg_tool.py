from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.config import Settings

_OPERATIONS = (
    "determinant",
    "inverse",
    "transpose",
    "multiply",
    "add",
    "eigenvalues",
    "rank",
    "solve",
)


class LinearAlgebraInput(BaseModel):
    """Input schema for the linear algebra tool."""

    operation: str = Field(
        ...,
        description=(
            "What to compute: 'determinant', 'inverse', 'transpose', "
            "'multiply' (matrix x matrix_b), 'add' (matrix + matrix_b), "
            "'eigenvalues', 'rank', or 'solve' (Ax=b with matrix=A, vector=b)."
        ),
    )
    matrix: str = Field(
        ...,
        min_length=1,
        description=(
            "Matrix as nested lists, e.g. '[[1, 2], [3, 4]]'. Rows are inner "
            "lists."
        ),
    )
    matrix_b: str | None = Field(
        default=None,
        description="Second matrix for 'multiply' or 'add', same nested-list format.",
    )
    vector: str | None = Field(
        default=None,
        description="Right-hand side vector b for 'solve', e.g. '[5, 6]'.",
    )


def build_linalg_tool(
    _settings: Settings,
) -> BaseTool:
    """Create a SymPy-backed exact linear algebra tool."""

    def _run_linalg(
        operation: str,
        matrix: str,
        matrix_b: str | None = None,
        vector: str | None = None,
    ) -> str:
        return linear_algebra(
            operation=operation,
            matrix=matrix,
            matrix_b=matrix_b,
            vector=vector,
        )

    return StructuredTool.from_function(
        func=_run_linalg,
        name="linear_algebra",
        description=(
            "Perform exact matrix operations: determinant, inverse, transpose, "
            "matrix multiplication and addition, eigenvalues, rank, and solving "
            "a linear system Ax=b. Use for matrix/vector math and systems of "
            "linear equations. Matrices are given as nested lists like "
            "'[[1, 2], [3, 4]]'."
        ),
        args_schema=LinearAlgebraInput,
    )


def linear_algebra(
    *,
    operation: str,
    matrix: str,
    matrix_b: str | None = None,
    vector: str | None = None,
) -> str:
    op = (operation or "").strip().lower()
    if op not in _OPERATIONS:
        return f"Unsupported operation {operation!r}. Use one of: {', '.join(_OPERATIONS)}."

    try:
        import sympy
    except ImportError as exc:  # pragma: no cover - sympy ships with the deps
        return (
            "SymPy is not installed. Run `python -m pip install sympy` to "
            f"enable the linear algebra tool: {exc}"
        )

    try:
        primary = _parse_matrix(sympy, matrix, "matrix")
        if op == "determinant":
            return _require_square(primary, "determinant") or (
                f"determinant = {primary.det()}"
            )
        if op == "inverse":
            squared = _require_square(primary, "inverse")
            if squared:
                return squared
            if primary.det() == 0:
                return "Matrix is singular (determinant 0); it has no inverse."
            return f"inverse =\n{_render(primary.inv())}"
        if op == "transpose":
            return f"transpose =\n{_render(primary.T)}"
        if op == "rank":
            return f"rank = {primary.rank()}"
        if op == "eigenvalues":
            squared = _require_square(primary, "eigenvalues")
            if squared:
                return squared
            return _format_eigenvalues(primary)
        if op == "multiply":
            return _do_multiply(sympy, primary, matrix_b)
        if op == "add":
            return _do_add(sympy, primary, matrix_b)
        return _do_solve(sympy, primary, vector)
    except _LinAlgError as exc:
        return f"Linear algebra failed: {exc}"
    except Exception as exc:
        return f"Linear algebra failed: {exc}"


class _LinAlgError(Exception):
    """Internal error with a user-readable message."""


def _parse_matrix(sympy: Any, raw: str, field: str) -> Any:
    text = (raw or "").strip()
    if not text:
        raise _LinAlgError(f"{field} is required.")
    try:
        data = ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise _LinAlgError(
            f"could not parse {field}={raw!r}; use nested lists like "
            "'[[1, 2], [3, 4]]'."
        ) from exc
    # Accept a flat list as a column vector.
    if isinstance(data, (list, tuple)) and data and not isinstance(
        data[0], (list, tuple)
    ):
        data = [[item] for item in data]
    if not isinstance(data, (list, tuple)) or not data:
        raise _LinAlgError(f"{field} must be a non-empty list of rows.")
    try:
        return sympy.Matrix([[_to_number(sympy, c) for c in row] for row in data])
    except (TypeError, ValueError) as exc:
        raise _LinAlgError(f"{field} has non-numeric or ragged rows: {exc}") from exc


def _to_number(sympy: Any, value: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("booleans are not valid matrix entries")
    if isinstance(value, (int, float)):
        return sympy.nsimplify(value) if isinstance(value, float) else sympy.Integer(value)
    if isinstance(value, str):
        return sympy.sympify(value)
    raise ValueError(f"unsupported entry {value!r}")


def _require_square(matrix: Any, op: str) -> str | None:
    if matrix.rows != matrix.cols:
        return f"{op} requires a square matrix (got {matrix.rows}x{matrix.cols})."
    return None


def _do_multiply(sympy: Any, primary: Any, matrix_b: str | None) -> str:
    if not matrix_b:
        raise _LinAlgError("multiply requires matrix_b.")
    other = _parse_matrix(sympy, matrix_b, "matrix_b")
    if primary.cols != other.rows:
        raise _LinAlgError(
            f"incompatible shapes for multiply: {primary.rows}x{primary.cols} "
            f"and {other.rows}x{other.cols}."
        )
    return f"product =\n{_render(primary * other)}"


def _do_add(sympy: Any, primary: Any, matrix_b: str | None) -> str:
    if not matrix_b:
        raise _LinAlgError("add requires matrix_b.")
    other = _parse_matrix(sympy, matrix_b, "matrix_b")
    if primary.shape != other.shape:
        raise _LinAlgError(
            f"incompatible shapes for add: {primary.shape} and {other.shape}."
        )
    return f"sum =\n{_render(primary + other)}"


def _do_solve(sympy: Any, primary: Any, vector: str | None) -> str:
    if not vector:
        raise _LinAlgError("solve requires the vector b (right-hand side).")
    rhs = _parse_matrix(sympy, vector, "vector")
    if rhs.cols != 1:
        rhs = rhs.T
    if primary.rows != rhs.rows:
        raise _LinAlgError(
            f"incompatible shapes for solve: A is {primary.rows}x{primary.cols}, "
            f"b has {rhs.rows} entries."
        )
    if primary.rows == primary.cols and primary.det() == 0:
        return "System has no unique solution (coefficient matrix is singular)."
    try:
        solution = primary.solve(rhs)
    except Exception as exc:
        raise _LinAlgError(f"could not solve the system: {exc}") from exc
    values = ", ".join(str(v) for v in solution)
    return f"solution x = [{values}]"


def _format_eigenvalues(matrix: Any) -> str:
    eig = matrix.eigenvals()
    parts = [
        f"{value} (multiplicity {mult})" for value, mult in eig.items()
    ]
    return "eigenvalues: " + ", ".join(parts)


def _render(matrix: Any) -> str:
    rows = []
    for r in range(matrix.rows):
        rows.append("[" + ", ".join(str(matrix[r, c]) for c in range(matrix.cols)) + "]")
    return "\n".join(rows)


__all__ = [
    "LinearAlgebraInput",
    "build_linalg_tool",
    "linear_algebra",
]
