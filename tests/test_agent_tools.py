from __future__ import annotations

from src.config import Settings
from src.backend.tools import (
    build_currency_tool,
    build_datetime_tool,
    build_directions_tool,
    build_linalg_tool,
    build_map_tool,
    build_math_tool,
    build_number_theory_tool,
    build_statistics_tool,
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
    # Current conditions are rendered as a markdown table.
    assert "| Detail | Value |" in result
    assert "| --- | --- |" in result
    assert "| Current temperature | 26.5C |" in result
    assert "| Conditions | partly cloudy |" in result
    # Forecast is rendered as its own markdown table.
    assert "| Date | Range | Conditions |" in result
    assert "| 2026-06-12 | 22.0C-30.0C | partly cloudy |" in result
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
    assert build_directions_tool(settings).name == "get_directions"
    assert build_map_tool(settings).name == "find_on_map"
    assert build_math_tool(settings).name == "solve_math"
    assert build_datetime_tool(settings).name == "calculate_datetime"
    assert build_statistics_tool(settings).name == "compute_statistics"
    assert build_linalg_tool(settings).name == "linear_algebra"
    assert build_number_theory_tool(settings).name == "number_theory"


def test_math_tool_computes_limit():
    tool = build_math_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke(
        {"operation": "limit", "expression": "sin(x)/x", "variable": "x", "point": "0"}
    )

    assert "= 1" in result


def test_math_tool_limit_to_infinity():
    tool = build_math_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke(
        {"operation": "limit", "expression": "1/x", "point": "oo"}
    )

    assert "= 0" in result


def test_math_tool_computes_series():
    tool = build_math_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke(
        {"operation": "series", "expression": "exp(x)", "order": 4}
    )

    # Maclaurin series of e^x: 1 + x + x**2/2 + x**3/6 + ...
    assert "x**2/2" in result
    assert "x**3/6" in result


def test_statistics_tool_basic():
    tool = build_statistics_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke({"numbers": "2, 4, 6, 8, 10"})

    assert "| Mean | 6 |" in result
    assert "| Median | 6 |" in result
    assert "| Count | 5 |" in result
    assert "| Sum | 30 |" in result


def test_statistics_tool_requires_numbers():
    tool = build_statistics_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke({"numbers": "no digits here"})

    assert "at least one number" in result.lower()


def test_linalg_determinant():
    tool = build_linalg_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke(
        {"operation": "determinant", "matrix": "[[1, 2], [3, 4]]"}
    )

    # det = 1*4 - 2*3 = -2
    assert "-2" in result


def test_linalg_solve_system():
    tool = build_linalg_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke(
        {
            "operation": "solve",
            "matrix": "[[2, 1], [1, 3]]",
            "vector": "[5, 10]",
        }
    )

    # 2x + y = 5; x + 3y = 10 -> x = 1, y = 3
    assert "x = [1, 3]" in result


def test_linalg_inverse_singular_reports_clearly():
    tool = build_linalg_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke(
        {"operation": "inverse", "matrix": "[[1, 2], [2, 4]]"}
    )

    assert "singular" in result.lower()


def test_linalg_multiply_shape_mismatch():
    tool = build_linalg_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke(
        {
            "operation": "multiply",
            "matrix": "[[1, 2, 3]]",
            "matrix_b": "[[1, 2, 3]]",
        }
    )

    assert "incompatible shapes" in result.lower()


def test_number_theory_factorize():
    tool = build_number_theory_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke({"operation": "factorize", "number": 360})

    # 360 = 2^3 * 3^2 * 5
    assert "2^3" in result
    assert "3^2" in result
    assert "5" in result


def test_number_theory_is_prime():
    tool = build_number_theory_tool(Settings(dashscope_api_key="test-key"))
    assert "not prime" in tool.invoke({"operation": "is_prime", "number": 91}).lower()
    assert "is prime" in tool.invoke({"operation": "is_prime", "number": 97}).lower()


def test_number_theory_gcd_lcm():
    tool = build_number_theory_tool(Settings(dashscope_api_key="test-key"))
    gcd = tool.invoke({"operation": "gcd", "number": 12, "second_number": 18})
    lcm = tool.invoke({"operation": "lcm", "number": 4, "second_number": 6})

    assert "= 6" in gcd
    assert "= 12" in lcm


def test_number_theory_to_base():
    tool = build_number_theory_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke({"operation": "to_base", "number": 255, "base": 16})

    assert "ff" in result.lower()


def test_number_theory_gcd_requires_second_number():
    tool = build_number_theory_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke({"operation": "gcd", "number": 12})

    assert "requires second_number" in result.lower()


def test_datetime_tool_difference_in_days():
    tool = build_datetime_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke(
        {"operation": "difference", "start": "2026-01-01", "end": "2026-01-11"}
    )

    assert "10 day(s)" in result


def test_datetime_tool_add_offsets_date():
    tool = build_datetime_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke(
        {"operation": "add", "start": "2026-06-30", "days": 5}
    )

    assert "2026-07-05" in result


def test_datetime_tool_weekday():
    tool = build_datetime_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke({"operation": "weekday", "start": "2026-06-30"})

    # 2026-06-30 is a Tuesday.
    assert "Tuesday" in result


def test_datetime_tool_convert_timezone():
    tool = build_datetime_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke(
        {
            "operation": "convert_timezone",
            "start": "2026-06-30T15:00",
            "timezone_name": "Asia/Tokyo",
            "to_timezone": "UTC",
        }
    )

    # Tokyo is UTC+9, so 15:00 Tokyo -> 06:00 UTC.
    assert "06:00" in result


def test_datetime_tool_rejects_unknown_timezone():
    tool = build_datetime_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke({"operation": "now", "timezone_name": "Mars/Olympus"})

    assert "unknown timezone" in result.lower()


def test_datetime_tool_rejects_unsupported_operation():
    tool = build_datetime_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke({"operation": "teleport", "start": "2026-06-30"})

    assert "unsupported operation" in result.lower()


def test_directions_tool_geocodes_endpoints_and_formats_route():
    calls = []

    def requester(url, **kwargs):
        calls.append((url, kwargs))
        if "/place/text" in url:
            name = kwargs["params"]["keywords"]
            coords = {
                "Shanghai": (31.23, 121.47, "上海市", "中国"),
                "Hangzhou": (30.29, 120.16, "浙江省", "中国"),
            }[name]
            lat, lon, province, country = coords
            return {
                "status": "1",
                "pois": [
                    {
                        "id": f"B-{name}",
                        "name": name,
                        "pname": province,
                        "cityname": name,
                        "country": country,
                        "location": f"{lon},{lat}",
                        "type": "行政地标",
                    }
                ],
            }
        assert url.endswith("/direction/driving")
        return {
            "status": "1",
            "route": {
                "paths": [
                    {
                        "distance": "165000",
                        "duration": "7200",
                        "steps": [{"instruction": "Drive toward Hangzhou"}],
                    }
                ]
            },
        }

    tool = build_directions_tool(
        Settings(dashscope_api_key="test-key", amap_web_service_key="amap-secret"),
        requester=requester,
    )
    result = tool.invoke(
        {"origin": "Shanghai", "destination": "Hangzhou", "mode": "driving"}
    )

    assert "Directions from Shanghai, 上海市, 中国 to Hangzhou, 浙江省, 中国" in result
    assert "| Coordinate system | GCJ-02 (AMap) |" in result
    assert "| Distance | 165.0 km |" in result
    assert "| Estimated time | 2 h |" in result
    assert "Route steps:" in result
    assert isinstance(result, str)
    tool_message = tool.invoke(
        {
            "type": "tool_call",
            "name": "get_directions",
            "args": {"origin": "Shanghai", "destination": "Hangzhou", "mode": "car"},
            "id": "call-route",
        }
    )
    assert tool_message.artifact["type"] == "amap"
    assert tool_message.artifact["kind"] == "route"
    assert tool_message.artifact["coordinateSystem"] == "gcj02"
    assert tool_message.artifact["positions"][0] == {"lng": 121.47, "lat": 31.23}
    # Two AMap POI geocoding calls + one routing call per invocation.
    assert len(calls) == 6
    assert all(call[1]["params"]["key"] == "amap-secret" for call in calls)


def test_directions_tool_accepts_raw_coordinates_and_converts_wgs84_before_routing():
    calls = []

    def requester(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/assistant/coordinate/convert"):
            locations = kwargs["params"]["locations"]
            converted = {
                "121.470000,31.230000": "121.474000,31.234000",
                "120.160000,30.290000": "120.164000,30.294000",
            }[locations]
            return {"status": "1", "locations": converted}
        assert url.endswith("/direction/driving")
        assert kwargs["params"]["origin"] == "121.474000,31.234000"
        assert kwargs["params"]["destination"] == "120.164000,30.294000"
        return {"status": "1", "route": {"paths": [{"distance": "1000", "duration": "600"}]}}

    tool = build_directions_tool(
        Settings(dashscope_api_key="test-key", amap_web_service_key="amap-secret"),
        requester=requester,
    )
    result = tool.invoke(
        {"origin": "31.23,121.47", "destination": "30.29,120.16"}
    )

    assert "| Distance | 1.0 km |" in result
    assert "| Estimated time | 10 min |" in result
    assert [url for url, _kwargs in calls].count(
        "https://restapi.amap.com/v3/assistant/coordinate/convert"
    ) == 2


def test_directions_tool_reports_missing_route():
    def requester(url, **kwargs):
        if "/place/text" in url:
            return {
                "status": "1",
                "pois": [
                    {"name": "A", "latitude": 1.0, "longitude": 1.0, "location": "1.0,1.0"},
                ],
            }
        return {"status": "1", "route": {"paths": []}}

    tool = build_directions_tool(
        Settings(dashscope_api_key="test-key", amap_web_service_key="amap-secret"),
        requester=requester,
    )
    result = tool.invoke({"origin": "A", "destination": "A"})

    assert "No route found" in result


def test_directions_tool_error_returns_string_not_raises():
    def boom(url, **kwargs):
        raise RuntimeError("network down")

    tool = build_directions_tool(
        Settings(dashscope_api_key="test-key", amap_web_service_key="amap-secret"),
        requester=boom,
    )
    result = tool.invoke({"origin": "Shanghai", "destination": "Hangzhou"})

    assert "failed" in result.lower()


def test_map_tool_geocodes_place_and_returns_amap_link():
    def requester(url, **kwargs):
        assert url.endswith("/place/text")
        assert kwargs["params"]["key"] == "amap-secret"
        return {
            "status": "1",
            "pois": [
                {
                    "id": "B-PARIS",
                    "name": "Paris",
                    "pname": "Ile-de-France",
                    "country": "France",
                    "location": "2.3522,48.8566",
                    "type": "city",
                }
            ],
        }

    tool = build_map_tool(
        Settings(dashscope_api_key="test-key", amap_web_service_key="amap-secret"),
        requester=requester,
    )
    result = tool.invoke({"place": "Paris", "zoom": 12})

    assert "Map location for Paris, Ile-de-France, France" in result
    assert "| Latitude | 48.8566 |" in result
    assert "| Longitude | 2.3522 |" in result
    assert "| Coordinate system | GCJ-02 (AMap) |" in result
    assert "uri.amap.com/marker" in result
    assert "121" not in result
    assert isinstance(result, str)
    tool_message = tool.invoke(
        {
            "type": "tool_call",
            "name": "find_on_map",
            "args": {"place": "Paris", "zoom": 12},
            "id": "call-map",
        }
    )
    assert tool_message.artifact["type"] == "amap"
    assert tool_message.artifact["kind"] == "marker"
    assert tool_message.artifact["provider"] == "amap"
    assert tool_message.artifact["positions"] == [{"lng": 2.3522, "lat": 48.8566}]


def test_map_tool_reports_no_results():
    def requester(url, **kwargs):
        if url.endswith("/place/text"):
            return {"status": "1", "pois": []}
        if url.endswith("/geocode/geo"):
            return {"status": "1", "geocodes": []}
        if url.endswith("/config/district"):
            return {"status": "1", "districts": []}
        raise AssertionError(f"unexpected URL: {url}")

    tool = build_map_tool(
        Settings(dashscope_api_key="test-key", amap_web_service_key="amap-secret"),
        requester=requester,
    )
    result = tool.invoke({"place": "Nowhereville12345"})

    assert "No map location found" in result


def test_math_tool_computes_derivative():
    tool = build_math_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke(
        {"operation": "derivative", "expression": "x**2 + 3*x", "variable": "x"}
    )

    assert "2*x + 3" in result


def test_math_tool_computes_indefinite_integral_with_constant():
    tool = build_math_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke({"operation": "integral", "expression": "2*x"})

    assert "x**2" in result
    assert "+ C" in result


def test_math_tool_computes_definite_integral():
    tool = build_math_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke(
        {
            "operation": "integral",
            "expression": "x**2",
            "lower_bound": "0",
            "upper_bound": "3",
        }
    )

    # Integral of x^2 from 0 to 3 is 9.
    assert "= 9" in result


def test_math_tool_solves_equation():
    tool = build_math_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke({"operation": "solve", "expression": "x**2 - 4", "variable": "x"})

    assert "-2" in result and "2" in result


def test_math_tool_solves_equation_with_equals_sign():
    tool = build_math_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke({"operation": "solve", "expression": "2*x + 1 = 5"})

    assert "x = 2" in result


def test_math_tool_evaluates_numeric_expression():
    tool = build_math_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke({"operation": "evaluate", "expression": "sqrt(16) + 2"})

    assert "6" in result


def test_math_tool_evaluate_rejects_free_symbols():
    tool = build_math_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke({"operation": "evaluate", "expression": "x + 1"})

    assert "unknown symbol" in result.lower()


def test_math_tool_rejects_unsupported_operation():
    tool = build_math_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke({"operation": "factorize", "expression": "x**2 - 1"})

    assert "Unsupported operation" in result


def test_math_tool_error_returns_string_not_raises():
    tool = build_math_tool(Settings(dashscope_api_key="test-key"))
    result = tool.invoke({"operation": "derivative", "expression": "x**"})

    assert "failed" in result.lower()
