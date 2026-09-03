# Purpose: extract publication dates from fetched HTML metadata.
from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from typing import Any

_BeautifulSoupImport: Any
try:
    from bs4 import BeautifulSoup as _BeautifulSoupImport
except ImportError:  # pragma: no cover - dependency is declared for the app.
    _BeautifulSoupImport = None

BeautifulSoup: Any = _BeautifulSoupImport

DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
META_DATE_SELECTORS = (
    {"property": "article:published_time"},
    {"property": "og:published_time"},
    {"name": "pubdate"},
    {"name": "publishdate"},
    {"name": "date"},
    {"itemprop": "datePublished"},
)
MODIFIED_META_DATE_SELECTORS = (
    {"property": "article:modified_time"},
    {"property": "og:updated_time"},
    {"itemprop": "dateModified"},
)


def extract_publication_date(html: str, metadata: dict[str, Any] | None = None) -> date | None:
    """Extract a likely publication date from HTML or document metadata.

    Args:
        html: Raw HTML to inspect.
        metadata: Optional loader metadata containing date-like fields.
    """

    for candidate in _metadata_candidates(metadata or {}):
        parsed = parse_publication_date(candidate)
        if parsed is not None:
            return parsed

    if BeautifulSoup is None or not _looks_like_html(html):
        for candidate in _modified_metadata_candidates(metadata or {}):
            parsed = parse_publication_date(candidate)
            if parsed is not None:
                return parsed
        return parse_publication_date(html)

    soup = BeautifulSoup(html, "html.parser")
    for candidate in _html_candidates(soup):
        parsed = parse_publication_date(candidate)
        if parsed is not None:
            return parsed

    # Modification dates are a weaker fallback. A declared publication or
    # creation date always wins, including one found in page markup.
    for candidate in _modified_metadata_candidates(metadata or {}):
        parsed = parse_publication_date(candidate)
        if parsed is not None:
            return parsed
    for candidate in _modified_html_candidates(soup):
        parsed = parse_publication_date(candidate)
        if parsed is not None:
            return parsed
    return None


def parse_publication_date(value: Any) -> date | None:
    """Parse common HTML publication-date values into a date."""

    if isinstance(value, datetime):
        return _datetime_to_date(value)
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None

    parsed = _parse_iso_datetime(text) or _parse_rfc_datetime(text)
    if parsed is not None:
        return _datetime_to_date(parsed)

    match = DATE_RE.search(text)
    if match is None:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _looks_like_html(text: str) -> bool:
    return bool(text and "<" in text and ">" in text)


def _metadata_candidates(metadata: dict[str, Any]) -> list[Any]:
    keys = (
        "published_at",
        "publication_date",
        "datePublished",
        "date_published",
        "article:published_time",
        "created_at",
    )
    return [metadata[key] for key in keys if key in metadata]


def _modified_metadata_candidates(metadata: dict[str, Any]) -> list[Any]:
    keys = ("modified_at", "dateModified", "date_modified", "updated_at")
    return [metadata[key] for key in keys if key in metadata]


def _html_candidates(soup: Any) -> list[str]:
    candidates: list[str] = []
    candidates.extend(_meta_candidates(soup))
    candidates.extend(_json_ld_candidates(soup))
    candidates.extend(_time_candidates(soup))
    return candidates


def _modified_html_candidates(soup: Any) -> list[str]:
    candidates: list[str] = []
    for attrs in MODIFIED_META_DATE_SELECTORS:
        tag = soup.find("meta", attrs=attrs)
        if tag is not None and tag.get("content"):
            candidates.append(str(tag["content"]))
    candidates.extend(_json_ld_candidates(soup, modified=True))
    return candidates


def _meta_candidates(soup: Any) -> list[str]:
    values: list[str] = []
    for attrs in META_DATE_SELECTORS:
        tag = soup.find("meta", attrs=attrs)
        if tag is not None and tag.get("content"):
            values.append(str(tag["content"]))
    return values


def _time_candidates(soup: Any) -> list[str]:
    values: list[str] = []
    for tag in soup.find_all("time"):
        if tag.get("datetime"):
            values.append(str(tag["datetime"]))
        text = tag.get_text(" ", strip=True)
        if text:
            values.append(text)
    return values


def _json_ld_candidates(soup: Any, *, modified: bool = False) -> list[str]:
    values: list[str] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(tag.string or tag.get_text() or "{}")
        except json.JSONDecodeError:
            continue
        values.extend(_json_ld_date_values(payload, modified=modified))
    return values


def _json_ld_date_values(payload: Any, *, modified: bool = False) -> list[str]:
    if isinstance(payload, list):
        return [
            value
            for item in payload
            for value in _json_ld_date_values(item, modified=modified)
        ]
    if not isinstance(payload, dict):
        return []

    values: list[str] = []
    keys = ("dateModified",) if modified else ("datePublished", "dateCreated")
    for key in keys:
        if payload.get(key):
            values.append(str(payload[key]))
    for key in ("@graph", "mainEntity", "articleBody"):
        values.extend(_json_ld_date_values(payload.get(key), modified=modified))
    return values


def _parse_iso_datetime(text: str) -> datetime | None:
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _parse_rfc_datetime(text: str) -> datetime | None:
    try:
        return parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None


def _datetime_to_date(value: datetime) -> date:
    if value.tzinfo is not None:
        value = value.astimezone(UTC)
    return value.date()
