from __future__ import annotations

from src.config import Settings
from src.tools import (
    build_currency_tool,
    build_stock_tool,
    build_weather_tool,
    build_wikipedia_tool,
)


def test_weather_tool_uses_city_geocoding_and_formats_forecast():
    calls = []

    def requester(url, **kwargs):
        calls.append((url, kwargs))
        if "geocoding-api" in url:
            return {
                "results": [
                    {
                        "name": "Shanghai",
                        "admin1": "Shanghai",
                        "country": "China",
                        "latitude": 31.23,
                        "longitude": 121.47,
                    }
                ]
            }
        return {
            "current": {
                "temperature_2m": 26.5,
                "relative_humidity_2m": 72,
                "wind_speed_10m": 8.2,
                "weather_code": 2,
            },
            "current_units": {
                "temperature_2m": "C",
                "relative_humidity_2m": "%",
                "wind_speed_10m": " km/h",
            },
            "daily": {
                "time": ["2026-06-12"],
                "temperature_2m_max": [30.0],
                "temperature_2m_min": [22.0],
                "weather_code": [2],
            },
            "daily_units": {"temperature_2m_max": "C"},
        }

    tool = build_weather_tool(Settings(dashscope_api_key="test-key"), requester=requester)
    result = tool.invoke({"city": "Shanghai", "forecast_days": 1})

    assert "Weather for Shanghai, Shanghai, China" in result
    assert "Current temperature: 26.5C" in result
    assert "Conditions: partly cloudy" in result
    assert calls[1][1]["params"]["forecast_days"] == 1


def test_currency_tool_converts_with_frankfurter_rate():
    def requester(url, **kwargs):
        assert url == "https://api.frankfurter.dev/v2/rate/USD/EUR"
        assert kwargs["params"] == {}
        return {"rate": 0.9, "date": "2026-06-12"}

    tool = build_currency_tool(Settings(dashscope_api_key="test-key"), requester=requester)
    result = tool.invoke(
        {"amount": 10, "from_currency": "usd", "to_currency": "eur"}
    )

    assert "10 USD = 9.0000 EUR" in result
    assert "Exchange rate: 1 USD = 0.900000 EUR" in result
    assert "Rate date: 2026-06-12" in result


def test_wikipedia_tool_searches_and_fetches_summary_with_user_agent():
    calls = []

    def requester(url, **kwargs):
        calls.append((url, kwargs))
        assert kwargs["headers"]["User-Agent"] == "test-agent/1.0 (contact: tests)"
        if kwargs["params"].get("list") == "search":
            return {"query": {"search": [{"pageid": 123, "title": "LangGraph"}]}}
        return {
            "query": {
                "pages": {
                    "123": {
                        "title": "LangGraph",
                        "extract": "LangGraph is a library for building stateful agents.",
                    }
                }
            }
        }

    settings = Settings(
        dashscope_api_key="test-key",
        wikipedia_max_summary_chars=80,
        wikipedia_user_agent="test-agent/1.0 (contact: tests)",
    )
    tool = build_wikipedia_tool(settings, requester=requester)
    result = tool.invoke({"query": "LangGraph", "max_results": 1})

    assert "Wikipedia results for: LangGraph" in result
    assert "LangGraph is a library" in result
    assert "https://en.wikipedia.org/wiki/LangGraph" in result
    assert len(calls) == 2


def test_stock_tool_formats_quote_from_injected_ticker_factory():
    class FakeTicker:
        fast_info = {
            "last_price": 195.5,
            "previous_close": 190.0,
            "currency": "USD",
            "day_low": 188.0,
            "day_high": 196.0,
            "last_volume": 1234567,
            "market_cap": 3000000000,
            "year_low": 150.0,
            "year_high": 210.0,
        }
        info = {}

    def ticker_factory(symbol):
        assert symbol == "AAPL"
        return FakeTicker()

    tool = build_stock_tool(
        Settings(dashscope_api_key="test-key"),
        ticker_factory=ticker_factory,
    )
    result = tool.invoke({"ticker": "aapl"})

    assert "Stock quote for AAPL: 195.50 USD" in result
    assert "Day change: +5.50 (+2.89%)" in result
    assert "Volume: 1,234,567" in result
    assert "52-week range: 150.00-210.00" in result


def test_tool_builders_expose_expected_names():
    settings = Settings(dashscope_api_key="test-key")

    assert build_weather_tool(settings).name == "get_weather"
    assert build_stock_tool(settings).name == "get_stock_quote"
    assert build_currency_tool(settings).name == "convert_currency"
    assert build_wikipedia_tool(settings).name == "search_wikipedia"
