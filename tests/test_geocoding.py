from __future__ import annotations

from typing import Any

from src.tools._geocoding import (
    CONFIDENT_MATCH_SCORE,
    geocode_place,
    name_match_score,
)
from src.tools.map_tool import find_on_map

_SHANGHAI_POI = {
    "status": "1",
    "pois": [
        {
            "id": "B001",
            "name": "上海",
            "location": "121.458060,31.222220",
            "pname": "上海市",
            "cityname": "上海市",
            "adname": "黄浦区",
            "type": "行政地标",
            "adcode": "310101",
        }
    ],
}
_EMPTY_POIS: dict[str, Any] = {"status": "1", "pois": []}
_EMPTY_GEOCODES: dict[str, Any] = {"status": "1", "geocodes": []}
_EMPTY_DISTRICTS: dict[str, Any] = {"status": "1", "districts": []}


def _poi(name: str, lng: float, lat: float, **props: Any) -> dict[str, Any]:
    return {
        "id": props.pop("id", "BTEST"),
        "name": name,
        "location": f"{lng},{lat}",
        "pname": props.pop("pname", ""),
        "cityname": props.pop("cityname", ""),
        "adname": props.pop("adname", ""),
        "country": props.pop("country", ""),
        **props,
    }


class _StubRequester:
    """Return canned AMap payloads by endpoint and record request parameters."""

    def __init__(
        self,
        *,
        poi: dict[str, Any] | None = None,
        geocode: dict[str, Any] | None = None,
        district: dict[str, Any] | None = None,
    ) -> None:
        self.poi = poi or _EMPTY_POIS
        self.geocode = geocode or _EMPTY_GEOCODES
        self.district = district or _EMPTY_DISTRICTS
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, **kwargs: Any) -> dict[str, Any]:
        params = dict(kwargs.get("params") or {})
        self.calls.append((url, params))
        if "/place/text" in url:
            return self.poi
        if "/geocode/geo" in url:
            return self.geocode
        if "/config/district" in url:
            return self.district
        raise AssertionError(f"unexpected URL: {url}")


def test_amap_poi_search_is_primary_and_uses_cleaned_query():
    requester = _StubRequester(poi=_SHANGHAI_POI)

    places = geocode_place("在地图上找出上海的位置", requester=requester, api_key="test-key")

    assert requester.calls[0][0].endswith("/place/text")
    assert requester.calls[0][1]["keywords"] == "上海"
    assert requester.calls[0][1]["key"] == "test-key"
    assert places[0].latitude == 31.22222
    assert places[0].longitude == 121.45806
    assert places[0].provider == "amap"
    assert places[0].coordinate_system == "gcj02"
    assert places[0].source == "poi"
    assert places[0].is_confident


def test_confident_amap_poi_match_skips_geocode_and_district_fallbacks():
    requester = _StubRequester(poi=_SHANGHAI_POI)

    geocode_place("上海", requester=requester, api_key="test-key")

    assert [url for url, _params in requester.calls] == [
        "https://restapi.amap.com/v3/place/text"
    ]


def test_amap_address_geocode_fallback_runs_after_unconfident_poi_results():
    requester = _StubRequester(
        poi={
            "status": "1",
            "pois": [
                _poi(
                    "上海浦东软件园 Y2",
                    121.5992398,
                    31.201163,
                    cityname="上海市",
                    pname="上海市",
                    adname="浦东新区",
                    type="商务住宅",
                )
            ],
        },
        geocode={
            "status": "1",
            "geocodes": [
                {
                    "formatted_address": "上海市浦东新区晨晖路88号",
                    "location": "121.600000,31.200000",
                    "province": "上海市",
                    "city": "上海市",
                    "district": "浦东新区",
                    "level": "门牌号",
                }
            ],
        },
    )

    places = geocode_place("上海金蝶软件园", requester=requester, api_key="test-key")

    assert [url for url, _params in requester.calls][1].endswith("/geocode/geo")
    assert places[0].source == "poi"
    assert places[0].kind == "商务住宅"
    # Partial name overlap: the entry is nearby, not the requested park.
    assert not places[0].is_confident


def test_amap_district_lookup_is_final_fallback():
    requester = _StubRequester(
        district={
            "status": "1",
            "districts": [
                {
                    "name": "浦东新区",
                    "center": "121.544346,31.221461",
                    "level": "district",
                    "adcode": "310115",
                }
            ],
        }
    )

    places = geocode_place("浦东新区", requester=requester, api_key="test-key")

    assert [url for url, _params in requester.calls][-1].endswith("/config/district")
    assert places[0].provider == "amap"
    assert places[0].source == "district"
    assert places[0].kind == "district"
    assert places[0].is_confident


def test_geocoder_returns_empty_when_unconfigured_or_provider_fails():
    def failing(_url: str, **_kwargs: Any) -> dict[str, Any]:
        raise TimeoutError("network down")

    assert geocode_place("anywhere", requester=failing, api_key="test-key") == []
    assert geocode_place("anywhere", requester=failing, api_key="") == []
    assert geocode_place("   ", requester=failing, api_key="test-key") == []


def test_name_match_score_rewards_full_coverage_and_penalizes_partial():
    assert name_match_score("Shanghai", "Shanghai") == 1.0
    assert name_match_score("上海", "上海市") == 1.0

    partial = name_match_score(
        "上海金蝶软件园",
        "上海浦东软件园祖冲之园",
    )
    assert 0.0 < partial < CONFIDENT_MATCH_SCORE

    assert name_match_score("Kingdee Software Park", "Central Station") == 0.0


def test_map_request_phrasing_does_not_count_against_the_match():
    """ "在地图上找出X的位置" must score on X, not on the request wording."""

    score = name_match_score(
        "在地图上找出上海的位置",
        "上海市",
    )

    assert score >= CONFIDENT_MATCH_SCORE


def test_find_on_map_flags_an_approximate_point_of_interest_match():
    requester = _StubRequester(
        poi={
            "status": "1",
            "pois": [
                _poi(
                    "上海浦东软件园 Y2",
                    121.5992398,
                    31.201163,
                    cityname="上海市",
                    pname="上海市",
                    adname="浦东新区",
                )
            ],
        }
    )

    result = find_on_map("上海金蝶软件园", requester=requester, api_key="test-key")

    assert "| Latitude | 31.2012 |" in result
    assert "| Longitude | 121.5992 |" in result
    assert "APPROXIMATE MATCH" in result
    assert "verify with a web search" in result


def test_find_on_map_flags_same_name_places_in_different_regions():
    requester = _StubRequester(
        poi={
            "status": "1",
            "pois": [
                _poi(
                    "Eiffel Tower",
                    -116.235,
                    51.3336,
                    pname="Alberta",
                    cityname="Calgary",
                    country="Canada",
                ),
                _poi(
                    "Eiffel Tower",
                    -77.4452828,
                    37.8399146,
                    pname="Virginia",
                    cityname="Richmond",
                    country="United States",
                ),
            ],
        }
    )

    result = find_on_map("Eiffel Tower", requester=requester, api_key="test-key")

    assert "AMBIGUOUS" in result
    assert "Other candidates:" in result
    assert "Eiffel Tower, Richmond, Virginia, United States" in result


def test_find_on_map_explains_an_empty_result():
    requester = _StubRequester()

    result = find_on_map("nowhere-at-all", requester=requester, api_key="test-key")

    assert "No map location found" in result
    assert "web search" in result


def test_clean_place_query_strips_request_wording_in_both_languages():
    from src.tools._geocoding import clean_place_query

    assert clean_place_query("在地图上找出上海金蝶软件园的位置") == "上海金蝶软件园"
    assert clean_place_query("where is Shanghai on a map") == "Shanghai"
    assert clean_place_query("show me the location of Times Square") == "Times Square"
    # An address carries no request wording and must survive unchanged.
    assert clean_place_query("浦东新区晨晖路88号") == "浦东新区晨晖路88号"
    # Stripping everything falls back to the original text.
    assert clean_place_query("在地图上") == "在地图上"
