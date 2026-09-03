from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.config import Settings

_OPERATIONS = (
    "factorize",
    "is_prime",
    "gcd",
    "lcm",
    "prime_factors",
    "next_prime",
    "to_base",
)


class NumberTheoryInput(BaseModel):
    """Input schema for the number theory tool."""

    operation: str = Field(
        ...,
        description=(
            "What to compute: 'factorize' (prime factorization), 'is_prime', "
            "'prime_factors' (distinct primes), 'gcd', 'lcm', 'next_prime', or "
            "'to_base' (convert to another base)."
        ),
    )
    number: int = Field(
        ...,
        description="The primary integer to operate on.",
    )
    second_number: int | None = Field(
        default=None,
        description="Second integer for 'gcd' and 'lcm'.",
    )
    base: int = Field(
        default=2,
        ge=2,
        le=36,
        description="Target base (2-36) for 'to_base'.",
    )


def build_number_theory_tool(
    _settings: Settings,
) -> BaseTool:
    """Create a number-theory tool (factorization, primality, GCD/LCM, bases)."""

    def _run_number_theory(
        operation: str,
        number: int,
        second_number: int | None = None,
        base: int = 2,
    ) -> str:
        return number_theory(
            operation=operation,
            number=number,
            second_number=second_number,
            base=base,
        )

    return StructuredTool.from_function(
        func=_run_number_theory,
        name="number_theory",
        description=(
            "Integer math: prime factorization, primality testing, distinct "
            "prime factors, greatest common divisor (GCD), least common "
            "multiple (LCM), next prime, and base conversion (binary, hex, "
            "etc.). Use for questions about factors, primes, divisibility, or "
            "number bases."
        ),
        args_schema=NumberTheoryInput,
    )


def number_theory(
    *,
    operation: str,
    number: int,
    second_number: int | None = None,
    base: int = 2,
) -> str:
    op = (operation or "").strip().lower()
    if op not in _OPERATIONS:
        return f"Unsupported operation {operation!r}. Use one of: {', '.join(_OPERATIONS)}."

    try:
        import sympy
    except ImportError as exc:  # pragma: no cover - sympy ships with the deps
        return (
            "SymPy is not installed. Run `python -m pip install sympy` to "
            f"enable the number theory tool: {exc}"
        )

    try:
        n = int(number)
        if op == "factorize":
            return _do_factorize(sympy, n)
        if op == "prime_factors":
            primes = sorted(sympy.primefactors(n))
            return f"distinct prime factors of {n}: {primes or 'none'}"
        if op == "is_prime":
            return f"{n} is {'prime' if sympy.isprime(n) else 'not prime'}."
        if op == "next_prime":
            return f"next prime after {n} = {sympy.nextprime(n)}"
        if op == "gcd":
            return _do_pair(sympy, "gcd", n, second_number)
        if op == "lcm":
            return _do_pair(sympy, "lcm", n, second_number)
        return _do_to_base(n, base)
    except _NumberError as exc:
        return f"Number theory failed: {exc}"
    except Exception as exc:
        return f"Number theory failed: {exc}"


class _NumberError(Exception):
    """Internal error with a user-readable message."""


def _do_factorize(sympy: Any, n: int) -> str:
    if n in (0, 1, -1):
        return f"{n} has no prime factorization."
    factors = sympy.factorint(n)
    parts = [
        f"{prime}^{power}" if power > 1 else f"{prime}"
        for prime, power in sorted(factors.items())
    ]
    return f"{n} = {' * '.join(parts)}"


def _do_pair(sympy: Any, op: str, a: int, b: int | None) -> str:
    if b is None:
        raise _NumberError(f"{op} requires second_number.")
    result = sympy.gcd(a, int(b)) if op == "gcd" else sympy.lcm(a, int(b))
    return f"{op}({a}, {b}) = {result}"


def _do_to_base(n: int, base: int) -> str:
    base = int(base)
    if not (2 <= base <= 36):
        raise _NumberError("base must be between 2 and 36.")
    if n == 0:
        return f"0 in base {base} = 0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    sign = "-" if n < 0 else ""
    value = abs(n)
    out: list[str] = []
    while value:
        value, rem = divmod(value, base)
        out.append(digits[rem])
    return f"{n} in base {base} = {sign}{''.join(reversed(out))}"


__all__ = [
    "NumberTheoryInput",
    "build_number_theory_tool",
    "number_theory",
]
