from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..config import Settings

TickerFactory = Callable[[str], Any]


class StockInput(BaseModel):
    """Input schema for stock quote lookup."""

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Ticker symbol, such as AAPL, TSLA, MSFT, or 0700.HK.",
    )


def build_stock_tool(
    _settings: Settings,
    *,
    ticker_factory: TickerFactory | None = None,
) -> BaseTool:
    """Create a stock quote tool backed by yfinance."""

    def _run_stock(ticker: str) -> str:
        return get_stock_quote(ticker, ticker_factory=ticker_factory)

    return StructuredTool.from_function(
        func=_run_stock,
        name="get_stock_quote",
        description=(
            "Get a current stock quote and market snapshot for a ticker symbol. "
            "Use for current stock price, daily change, volume, market cap, and "
            "52-week range questions."
        ),
        args_schema=StockInput,
    )


def get_stock_quote(
    ticker: str,
    *,
    ticker_factory: TickerFactory | None = None,
) -> str:
    symbol = ticker.strip().upper()
    if not symbol:
        return "Stock quote lookup requires a ticker symbol."

    try:
        ticker_object = (ticker_factory or _default_ticker_factory)(symbol)
        fast_info = getattr(ticker_object, "fast_info", {}) or {}
        info = getattr(ticker_object, "info", {}) or {}
    except Exception as exc:
        return f"Stock quote lookup failed for {symbol}: {exc}"

    price = _first_value(
        fast_info,
        info,
        "last_price",
        "lastPrice",
        "regularMarketPrice",
        "currentPrice",
    )
    previous_close = _first_value(
        fast_info,
        info,
        "previous_close",
        "previousClose",
        "regularMarketPreviousClose",
    )
    currency = _first_value(fast_info, info, "currency") or ""

    if price is None:
        return f"No current stock quote found for {symbol}."

    lines = [f"Stock quote for {symbol}: {float(price):.2f} {currency}".rstrip()]
    if previous_close is not None:
        change = float(price) - float(previous_close)
        percent = (change / float(previous_close) * 100) if float(previous_close) else 0.0
        lines.append(f"Day change: {change:+.2f} ({percent:+.2f}%)")

    day_low = _first_value(fast_info, info, "day_low", "dayLow", "regularMarketDayLow")
    day_high = _first_value(fast_info, info, "day_high", "dayHigh", "regularMarketDayHigh")
    if day_low is not None and day_high is not None:
        lines.append(f"Day range: {float(day_low):.2f}-{float(day_high):.2f}")

    volume = _first_value(fast_info, info, "last_volume", "lastVolume", "volume")
    if volume is not None:
        lines.append(f"Volume: {int(volume):,}")

    market_cap = _first_value(fast_info, info, "market_cap", "marketCap")
    if market_cap is not None:
        lines.append(f"Market cap: {int(market_cap):,}")

    year_low = _first_value(fast_info, info, "year_low", "fiftyTwoWeekLow")
    year_high = _first_value(fast_info, info, "year_high", "fiftyTwoWeekHigh")
    if year_low is not None and year_high is not None:
        lines.append(f"52-week range: {float(year_low):.2f}-{float(year_high):.2f}")

    return "\n".join(lines)


def _default_ticker_factory(symbol: str) -> Any:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Install project dependencies or run "
            "`python -m pip install yfinance` to enable the stock tool."
        ) from exc

    return yf.Ticker(symbol)


def _first_value(*sources: Any) -> Any:
    keys = sources[2:]
    for source in sources[:2]:
        for key in keys:
            value = _value_from_source(source, key)
            if value is not None:
                return value
    return None


def _value_from_source(source: Any, key: Any) -> Any:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, str(key), None)


__all__ = [
    "StockInput",
    "build_stock_tool",
    "get_stock_quote",
]
