# REFACTOR: Shared AMap place geocoding for the map and directions tools.
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ._amap import (
    AMAP_COORDINATE_SYSTEM,
    AMAP_PROVIDER,
    AMapLookupResult,
    amap_geocode_address,
    amap_lookup_districts,
    amap_search_pois,
    resolve_amap_web_service_key,
)
from ._http import JsonRequester

# A candidate at or above this name coverage is treated as the place asked for.
CONFIDENT_MATCH_SCORE = 0.6

_CJK_RE = re.compile(r"[㐀-䶿一-鿿]")
_CJK_RUN_RE = re.compile(r"[㐀-䶿一-鿿]+")
_LATIN_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*")
_LATIN_REQUEST_RE = re.compile(
    r"\b(?:where\s+is|where\s+are|show\s+me|show|find|locate|display|"
    r"the\s+location\s+of|location\s+of|position\s+of|coordinates\s+of|"
    r"on\s+(?:the|a)\s+map|on\s+map|please)\b",
    re.I,
)
_PUNCTUATION_RE = re.compile(r"[，。！？；：、,.!?;:【】（）()\[\]]")
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
_SOURCE_RANK = {"poi": 0, "geocode": 1, "district": 2}


@dataclass(frozen=True)
class GeocodedPlace:
    """One AMap geocoding candidate with enough context to judge the match."""

    name: str
    latitude: float
    longitude: float
    city: str = ""
    state: str = ""
    country: str = ""
    kind: str = ""
    population: int | None = None
    provider: str = AMAP_PROVIDER
    match_score: float = 0.0
    address: str = ""
    district: str = ""
    adcode: str = ""
    amap_id: str = ""
    source: str = ""
    coordinate_system: str = AMAP_COORDINATE_SYSTEM

    @property
    def label(self) -> str:
        parts = [self.name, self.district, self.city, self.state, self.country]
        seen: list[str] = []
        for part in parts:
            value = (part or "").strip()
            if value and value not in seen:
                seen.append(value)
        return ", ".join(seen)

    @property
    def is_confident(self) -> bool:
        return self.match_score >= CONFIDENT_MATCH_SCORE

    @property
    def latitude_gcj02(self) -> float:
        return self.latitude

    @property
    def longitude_gcj02(self) -> float:
        return self.longitude


def geocode_place(
    query: str,
    *,
    limit: int = 3,
    requester: JsonRequester | None = None,
    api_key: str | None = None,
    timeout_seconds: int = 10,
) -> list[GeocodedPlace]:
    """Return ranked AMap candidates for a place, POI, address, or district.

    AMap POI text search is tried first because it covers landmarks, campuses,
    business parks, and many Chinese POIs. If that does not produce a confident
    match, address geocoding is added; administrative district lookup is the
    final fallback. Ranking, dedupe, confidence, and ambiguity semantics are the
    same as the previous shared geocoder.
    """

    term = (query or "").strip()
    if not term:
        return []

    key = resolve_amap_web_service_key(api_key)
    if not key:
        return []

    search_term = clean_place_query(term)
    requested_limit = max(1, int(limit))
    candidates = _amap_stage_candidates(
        "poi",
        search_term,
        api_key=key,
        limit=requested_limit,
        requester=requester,
        timeout_seconds=timeout_seconds,
    )

    if not any(candidate.is_confident for candidate in candidates):
        candidates.extend(
            _amap_stage_candidates(
                "geocode",
                search_term,
                api_key=key,
                limit=requested_limit,
                requester=requester,
                timeout_seconds=timeout_seconds,
            )
        )

    if not any(candidate.is_confident for candidate in candidates):
        candidates.extend(
            _amap_stage_candidates(
                "district",
                search_term,
                api_key=key,
                limit=requested_limit,
                requester=requester,
                timeout_seconds=timeout_seconds,
            )
        )

    ranked = sorted(
        _dedupe(candidates),
        key=lambda candidate: (
            -candidate.match_score,
            _SOURCE_RANK.get(candidate.source, 9),
            0 if candidate.population else 1,
        ),
    )
    return ranked[:requested_limit]


def _amap_stage_candidates(
    stage: str,
    term: str,
    *,
    api_key: str,
    limit: int,
    requester: JsonRequester | None,
    timeout_seconds: int,
) -> list[GeocodedPlace]:
    try:
        if stage == "poi":
            raw_results = amap_search_pois(
                term,
                api_key=api_key,
                limit=limit,
                requester=requester,
                timeout_seconds=timeout_seconds,
            )
        elif stage == "geocode":
            raw_results = amap_geocode_address(
                term,
                api_key=api_key,
                limit=limit,
                requester=requester,
                timeout_seconds=timeout_seconds,
            )
        else:
            raw_results = amap_lookup_districts(
                term,
                api_key=api_key,
                limit=limit,
                requester=requester,
                timeout_seconds=timeout_seconds,
            )
    except Exception:
        return []

    places: list[GeocodedPlace] = []
    for result in raw_results:
        place = _place_from_amap(result, term)
        if place is not None:
            places.append(place)
    return places


def _place_from_amap(result: AMapLookupResult, term: str) -> GeocodedPlace | None:
    name = str(result.name or result.address or term).strip()
    if not name:
        return None

    match_score = max(
        name_match_score(term, name),
        name_match_score(term, result.address),
        name_match_score(term, result.district),
        name_match_score(term, result.city),
    )
    return GeocodedPlace(
        name=name,
        latitude=result.position.lat,
        longitude=result.position.lng,
        city=result.city,
        state=result.province,
        country=result.country,
        kind=result.kind or result.source,
        provider=result.provider,
        match_score=match_score,
        address=result.address,
        district=result.district,
        adcode=result.adcode,
        amap_id=result.amap_id,
        source=result.source,
        coordinate_system=AMAP_COORDINATE_SYSTEM,
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

    CJK text is compared as character bigrams because it is not space separated:
    "上海金蝶软件园" against "上海浦东软件园祖冲之园" scores partial rather than exact,
    which is what marks it an approximate match.
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
    "GeocodedPlace",
    "clean_place_query",
    "geocode_place",
    "name_match_score",
]
