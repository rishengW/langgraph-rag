from __future__ import annotations

from typing import Any

from src.tools._geocoding import (
    CONFIDENT_MATCH_SCORE,
    geocode_place,
    name_match_score,
)
from src.tools.map_tool import find_on_map

_SHANGHAI = {
    "results": [
        {
            "name": "\u4e0a\u6d77",
            "latitude": 31.22222,
            "longitude": 121.45806,
            "admin1": "\u4e0a\u6d77\u5e02",
            "country": "\u4e2d\u56fd",
            "population": 24874500,
            "feature_code": "PPLA",
        }
    ]
}
_EMPTY_OPEN_METEO: dict[str, Any] = {"results": []}


def _photon_feature(name: str, lon: float, lat: float, **props: Any) -> dict[str, Any]:
    return {
        "geometry": {"coordinates": [lon, lat]},
        "properties": {"name": name, **props},
    }


class _StubRequester:
    """Return a canned payload per host and record the query parameters."""

    def __init__(self, open_meteo: dict[str, Any], photon: dict[str, Any] | None = None) -> None:
        self.open_meteo = open_meteo
        self.photon = photon or {"features": []}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, **kwargs: Any) -> dict[str, Any]:
        params = dict(kwargs.get("params") or {})
        self.calls.append((url, params))
        return self.photon if "photon" in url else self.open_meteo


def test_chinese_query_uses_the_matching_open_meteo_language():
    """`language=en` silently returns nothing for CJK names."""

    requester = _StubRequester(_SHANGHAI)

    places = geocode_place("\u4e0a\u6d77", requester=requester)

    assert requester.calls[0][1]["language"] == "zh"
    assert places[0].latitude == 31.22222
    assert places[0].population == 24874500
    assert places[0].provider == "open-meteo"
    assert places[0].is_confident


def test_latin_query_still_uses_english_names():
    requester = _StubRequester(_SHANGHAI)

    geocode_place("Shanghai", requester=requester)

    assert requester.calls[0][1]["language"] == "en"


def test_point_of_interest_falls_through_to_the_photon_provider():
    requester = _StubRequester(
        _EMPTY_OPEN_METEO,
        {
            "features": [
                _photon_feature(
                    "\u4e0a\u6d77\u6d66\u4e1c\u8f6f\u4ef6\u56ed",
                    121.5992398,
                    31.201163,
                    city="\u4e0a\u6d77",
                    country="\u4e2d\u56fd",
                    osm_value="commercial",
                )
            ]
        },
    )

    places = geocode_place("\u4e0a\u6d77\u91d1\u8776\u8f6f\u4ef6\u56ed", requester=requester)

    assert [url for url, _params in requester.calls][-1].endswith("/api")
    assert places[0].provider == "photon"
    assert places[0].kind == "commercial"
    # Partial name overlap: the entry is nearby, not the requested park.
    assert not places[0].is_confident


def test_confident_open_meteo_match_skips_the_poi_provider():
    requester = _StubRequester(_SHANGHAI)

    geocode_place("\u4e0a\u6d77", requester=requester)

    assert all("photon" not in url for url, _params in requester.calls)


def test_geocoder_returns_empty_when_every_provider_fails():
    def failing(_url: str, **_kwargs: Any) -> dict[str, Any]:
        raise TimeoutError("network down")

    assert geocode_place("anywhere", requester=failing) == []
    assert geocode_place("   ", requester=failing) == []


def test_name_match_score_rewards_full_coverage_and_penalizes_partial():
    assert name_match_score("Shanghai", "Shanghai") == 1.0
    assert name_match_score("\u4e0a\u6d77", "\u4e0a\u6d77\u5e02") == 1.0

    partial = name_match_score(
        "\u4e0a\u6d77\u91d1\u8776\u8f6f\u4ef6\u56ed",
        "\u4e0a\u6d77\u6d66\u4e1c\u8f6f\u4ef6\u56ed\u7956\u51b2\u4e4b\u56ed",
    )
    assert 0.0 < partial < CONFIDENT_MATCH_SCORE

    assert name_match_score("Kingdee Software Park", "Central Station") == 0.0


def test_map_request_phrasing_does_not_count_against_the_match():
    """ "在地图上找出X的位置" must score on X, not on the request wording."""

    score = name_match_score(
        "\u5728\u5730\u56fe\u4e0a\u627e\u51fa\u4e0a\u6d77\u7684\u4f4d\u7f6e",
        "\u4e0a\u6d77\u5e02",
    )

    assert score >= CONFIDENT_MATCH_SCORE


def test_find_on_map_flags_an_approximate_point_of_interest_match():
    requester = _StubRequester(
        _EMPTY_OPEN_METEO,
        {
            "features": [
                _photon_feature(
                    "\u4e0a\u6d77\u6d66\u4e1c\u8f6f\u4ef6\u56ed Y2",
                    121.5992398,
                    31.201163,
                    city="\u4e0a\u6d77",
                    country="\u4e2d\u56fd",
                )
            ]
        },
    )

    result = find_on_map("\u4e0a\u6d77\u91d1\u8776\u8f6f\u4ef6\u56ed", requester=requester)

    assert "| Latitude | 31.2012 |" in result
    assert "APPROXIMATE MATCH" in result
    assert "verify with a web search" in result


def test_find_on_map_flags_same_name_places_in_different_countries():
    requester = _StubRequester(
        _EMPTY_OPEN_METEO,
        {
            "features": [
                _photon_feature(
                    "Eiffel Tower", -116.235, 51.3336, state="Alberta", country="Canada"
                ),
                _photon_feature(
                    "Eiffel Tower",
                    -77.4452828,
                    37.8399146,
                    state="Virginia",
                    country="United States",
                ),
            ]
        },
    )

    result = find_on_map("Eiffel Tower", requester=requester)

    assert "AMBIGUOUS" in result
    assert "Other candidates:" in result
    assert "Eiffel Tower, Virginia, United States" in result


def test_find_on_map_does_not_flag_a_populated_place_with_namesakes():
    requester = _StubRequester(
        {
            "results": [
                {
                    "name": "Shanghai",
                    "latitude": 31.22222,
                    "longitude": 121.45806,
                    "admin1": "Shanghai",
                    "country": "China",
                    "population": 24874500,
                    "feature_code": "PPLA",
                },
                {
                    "name": "Shanghai",
                    "latitude": 34.8501,
                    "longitude": -87.085,
                    "admin1": "Alabama",
                    "country": "United States",
                },
            ]
        }
    )

    result = find_on_map("Shanghai", requester=requester)

    assert "AMBIGUOUS" not in result
    assert "APPROXIMATE MATCH" not in result
    assert "| Population | 24,874,500 |" in result


def test_find_on_map_explains_an_empty_result():
    requester = _StubRequester(_EMPTY_OPEN_METEO)

    result = find_on_map("nowhere-at-all", requester=requester)

    assert "No map location found" in result
    assert "web search" in result


def test_clean_place_query_strips_request_wording_in_both_languages():
    from src.tools._geocoding import clean_place_query

    assert (
        clean_place_query(
            "\u5728\u5730\u56fe\u4e0a\u627e\u51fa\u4e0a\u6d77\u91d1\u8776\u8f6f\u4ef6\u56ed"
            "\u7684\u4f4d\u7f6e"
        )
        == "\u4e0a\u6d77\u91d1\u8776\u8f6f\u4ef6\u56ed"
    )
    assert clean_place_query("where is Shanghai on a map") == "Shanghai"
    assert clean_place_query("show me the location of Times Square") == "Times Square"
    # An address carries no request wording and must survive unchanged.
    assert clean_place_query("\u6d66\u4e1c\u65b0\u533a\u6668\u6656\u8def88\u53f7") == (
        "\u6d66\u4e1c\u65b0\u533a\u6668\u6656\u8def88\u53f7"
    )
    # Stripping everything falls back to the original text.
    assert clean_place_query("\u5728\u5730\u56fe\u4e0a") == "\u5728\u5730\u56fe\u4e0a"


def test_a_whole_map_request_sentence_reaches_the_provider_as_a_place_name():
    requester = _StubRequester(_SHANGHAI)

    places = geocode_place(
        "\u5728\u5730\u56fe\u4e0a\u627e\u51fa\u4e0a\u6d77\u7684\u4f4d\u7f6e",
        requester=requester,
    )

    assert requester.calls[0][1]["name"] == "\u4e0a\u6d77"
    assert places[0].is_confident
