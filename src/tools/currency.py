from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from ._http import JsonRequester, request_json

if TYPE_CHECKING:
    from ..config import Settings

CURRENCY_API_URL = "https://api.frankfurter.dev/v2/rate"


class CurrencyInput(BaseModel):
    """Input schema for currency conversion."""

    amount: float = Field(..., gt=0, description="Amount to convert.")
    from_currency: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Source ISO 4217 currency code, such as USD.",
    )
    to_currency: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Target ISO 4217 currency code, such as EUR.",
    )


def build_currency_tool(
    _settings: Settings,
    *,
    requester: JsonRequester | None = None,
) -> BaseTool:
    """Create a currency converter tool backed by exchangerate.host."""

    def _run_currency(amount: float, from_currency: str, to_currency: str) -> str:
        return convert_currency(
            amount,
            from_currency,
            to_currency,
            requester=requester,
        )

    return StructuredTool.from_function(
        func=_run_currency,
        name="convert_currency",
        description=(
            "Convert money between ISO 4217 currencies using current Frankfurter "
            "exchange rates. Use for exchange-rate and currency-conversion questions."
        ),
        args_schema=CurrencyInput,
    )


def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
    *,
    requester: JsonRequester | None = None,
) -> str:
    source = from_currency.strip().upper()
    target = to_currency.strip().upper()
    if source == target:
        return f"{amount:g} {source} = {amount:g} {target}; rate: 1.0."

    try:
        payload = request_json(
            f"{CURRENCY_API_URL}/{source}/{target}",
            params={},
            requester=requester,
        )
    except Exception as exc:
        return f"Currency conversion failed for {source} to {target}: {exc}"

    rate = payload.get("rate")
    date = payload.get("date")

    if rate is None:
        error = payload.get("error") or payload.get("message") or "no exchange rate"
        return f"Currency conversion failed for {source} to {target}: {error}"

    result = amount * float(rate)
    lines = [f"{amount:g} {source} = {float(result):.4f} {target}"]
    lines.append(f"Exchange rate: 1 {source} = {float(rate):.6f} {target}")
    if date:
        lines.append(f"Rate date: {date}")
    return "\n".join(lines)


__all__ = [
    "CurrencyInput",
    "build_currency_tool",
    "convert_currency",
]
