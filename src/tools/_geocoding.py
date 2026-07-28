# REFACTOR: Shared place geocoding for the map and directions tools.
#
# Open-Meteo's geocoder is a populated-place gazetteer: it resolves cities and
# towns (with population) but returns nothing for points of interest, campuses,
# or street addresses. It also needs the right ``language`` — a Chinese query
# with ``language=en`` finds nothing, which is why "上海" used to fail while
# "Shanghai" worked. Photon (OSM-backed, no API key) covers POIs and CJK names,
# so it is queried as the second stage.
#
# Photon ranks by text similarity only, so a confident-looking result can be the
# wrong place entirely ("Eiffel Tower" matches a mountain in Alberta). Every
# candidate therefore carries a name-coverage score, and callers surface low
# scores as approximate matches instead of asserting them as the answer.
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from ._http import JsonRequester, request_json

OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
PHOTON_GEOCODING_URL = "https://photon.komoot.io/api"
GEOCODER_USER_AGENT = "langgraph-rag/1.0 (map tool)"
# A candidate at or above this name coverage is treated as the place asked for.
CONFIDENT_MATCH_SCORE = 0.6

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_LATIN_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*")
_LATIN_REQUEST_RE = re.compile(
    r"\b(?:where\s+is|where\s+are|show\s+me|show|find|locate|display|"
    r"the\s+location\s+of|location\s+of|position\s+of|coordinates\s+of|"
    r"on\s+(?:the|a)\s+map|on\s+map|please)\b",
    re.I,
)
_PUNCTUATION_RE = re.compile(r"[，。！？；：、,.!?;:\u3010\u3011\uff08\uff09()\[\]]")
# Words that describe the request rather than the place itself.
_QUERY_NOISE_TOKENS = frozenset({"the", "of", "in", "at", "on", "map", "location", "where"})
# Request wording, not place names. "在地图上找出上海的位置" must score on 上海.
# Longest first so compound phrases are removed before their fragments.
_CJK_NOISE_TERMS = tuple(
    sorted(
        (
            "在地图上",
            "地图上",
            "地图",
            "位置",
            "在哪里",
            "在哪",
            "哪里",
            "地址",
            "坐标",
            "经纬度",
            "找出",
            "找到",
            "查找",
            "搜索",
            "显示",
            "标出",
            "定位",
            "告诉我",
            "帮我",
            "请问",
            "一下",
            "的",
        ),
        key=len,
        reverse=True,
    )
)


@dataclass(frozen=True)
class GeocodedPlace:
    """One geocoding candidate with enough context to judge the match."""

    name: str
    latitude: float
    longitude: float
    city: str = ""
    state: str = ""
    country: str = ""
    kind: str = ""
    population: int | None = None
    provider: str = ""
    match_score: float = 0.0

    @property
    def label(self) -> str:
        parts = [self.name, self.state or self.city, self.country]
        seen: list[str] = []
        for part in parts:
            value = (part or "").strip()
            if value and value not in seen:
                seen.append(value)
        return ", ".join(seen)

    @property
    def is_confident(self) -> bool:
        return self.match_score >= CONFIDENT_MATCH_SCORE


def geocode_place(
    query: str,
    *,
    limit: int = 3,
    requester: JsonRequester | None = None,
) -> list[GeocodedPlace]:
    """Return ranked geocoding candidates for a place, city, POI, or address.

    Populated places resolve through Open-Meteo first because it is the more
    authoritative gazetteer for cities. Anything it cannot resolve falls through
    to Photon, which covers points of interest and non-Latin names.
    """

    term = (query or "").strip()
    if not term:
        return []

    # The agent may pass a whole request ("在地图上找出上海的位置", "where is
    # Shanghai on a map"). Gazetteers match place names, not sentences, so the
    # request wording is removed before the lookup.
    search_term = clean_place_query(term)
    candidates = _open_meteo_candidates(search_term, limit=limit, requester=requester)
    confident = [candidate for candidate in candidates if candidate.is_confident]
    if not confident:
        candidates.extend(_photon_candidates(search_term, limit=limit, requester=requester))

    ranked = sorted(
        _dedupe(candidates),
        key=lambda candidate: (-candidate.match_score, 0 if candidate.population else 1),
    )
    return ranked[: max(1, int(limit))]


def _open_meteo_candidates(
    term: str,
    *,
    limit: int,
    requester: JsonRequester | None,
) -> list[GeocodedPlace]:
    try:
        payload = request_json(
            OPEN_METEO_GEOCODING_URL,
            params={
                "name": term,
                "count": max(1, int(limit)),
                # Chinese input needs the matching name index; passing "en"
                # silently returns no results for CJK place names.
                "language": "zh" if _CJK_RE.search(term) else "en",
                "format": "json",
            },
            requester=requester,
        )
    except Exception:
        return []

    places: list[GeocodedPlace] = []
    for result in payload.get("results") or []:
        if not isinstance(result, dict):
            continue
        place = _place_from_open_meteo(result, term)
        if place is not None:
            places.append(place)
    return places


def _place_from_open_meteo(result: dict[str, Any], term: str) -> GeocodedPlace | None:
    try:
        latitude = float(result["latitude"])
        longitude = float(result["longitude"])
    except (KeyError, TypeError, ValueError):
        return None

    name = str(result.get("name") or term)
    population = result.get("population")
    return GeocodedPlace(
        name=name,
        latitude=latitude,
        longitude=longitude,
        city=str(result.get("admin2") or "").strip(),
        state=str(result.get("admin1") or "").strip(),
        country=str(result.get("country") or "").strip(),
        kind=str(result.get("feature_code") or "").strip(),
        population=int(population) if isinstance(population, (int, float)) else None,
        provider="open-meteo",
        match_score=name_match_score(term, name),
    )


def _photon_candidates(
    term: str,
    *,
    limit: int,
    requester: JsonRequester | None,
) -> list[GeocodedPlace]:
    try:
        payload = request_json(
            PHOTON_GEOCODING_URL,
            params={"q": term, "limit": max(1, int(limit))},
            requester=requester,
            headers={"User-Agent": GEOCODER_USER_AGENT},
        )
    except Exception:
        return []

    places: list[GeocodedPlace] = []
    for feature in payload.get("features") or []:
        if not isinstance(feature, dict):
            continue
        place = _place_from_photon(feature, term)
        if place is not None:
            places.append(place)
    return places


def _place_from_photon(feature: dict[str, Any], term: str) -> GeocodedPlace | None:
    geometry = feature.get("geometry")
    coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if not isinstance(coordinates, list) or len(coordinates) < 2:
        return None
    try:
        longitude = float(coordinates[0])
        latitude = float(coordinates[1])
    except (TypeError, ValueError):
        return None

    properties = feature.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    name = str(properties.get("name") or "").strip()
    if not name:
        street = str(properties.get("street") or "").strip()
        house = str(properties.get("housenumber") or "").strip()
        name = " ".join(part for part in (street, house) if part) or term
    return GeocodedPlace(
        name=name,
        latitude=latitude,
        longitude=longitude,
        city=str(properties.get("city") or properties.get("district") or "").strip(),
        state=str(properties.get("state") or "").strip(),
        country=str(properties.get("country") or "").strip(),
        kind=str(properties.get("osm_value") or "").strip(),
        provider="photon",
        match_score=name_match_score(term, name),
    )


def clean_place_query(query: str) -> str:
    """Strip map-request wording so only the place name is looked up.

    Returns the original text when stripping would leave nothing, so an unusual
    phrasing degrades to today's behavior instead of an empty lookup.
    """

    cleaned = unicodedata.normalize("NFKC", query or "").strip()
    if not cleaned:
        return ""
    cleaned = _LATIN_REQUEST_RE.sub(" ", cleaned)
    for noise in _CJK_NOISE_TERMS:
        cleaned = cleaned.replace(noise, " ")
    cleaned = _PUNCTUATION_RE.sub(" ", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned or query.strip()


def name_match_score(query: str, name: str) -> float:
    """Return the share of distinctive query terms present in a candidate name.

    CJK text is compared as character bigrams because it is not space
    separated: "上海金蝶软件园" against "上海浦东软件园祖冲之园" scores partial
    rather than exact, which is what marks it an approximate match.
    """

    query_terms = _match_terms(query)
    if not query_terms:
        return 0.0
    normalized_name = _normalize(name)
    matched = sum(1 for term in query_terms if term in normalized_name)
    return matched / len(query_terms)


def _match_terms(text: str) -> list[str]:
    normalized = _normalize(text)
    for noise in _CJK_NOISE_TERMS:
        normalized = normalized.replace(noise, " ")
    terms = [
        token for token in _LATIN_TOKEN_RE.findall(normalized) if token not in _QUERY_NOISE_TOKENS
    ]
    for run in _CJK_RUN_RE.findall(normalized):
        if len(run) == 1:
            terms.append(run)
            continue
        terms.extend(run[index : index + 2] for index in range(len(run) - 1))
    return list(dict.fromkeys(terms))


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").casefold()


def _dedupe(places: list[GeocodedPlace]) -> list[GeocodedPlace]:
    seen: set[tuple[str, str]] = set()
    unique: list[GeocodedPlace] = []
    for place in places:
        key = (f"{place.latitude:.4f}", f"{place.longitude:.4f}")
        if key in seen:
            continue
        seen.add(key)
        unique.append(place)
    return unique


__all__ = [
    "CONFIDENT_MATCH_SCORE",
    "GEOCODER_USER_AGENT",
    "OPEN_METEO_GEOCODING_URL",
    "PHOTON_GEOCODING_URL",
    "GeocodedPlace",
    "clean_place_query",
    "geocode_place",
    "name_match_score",
]
