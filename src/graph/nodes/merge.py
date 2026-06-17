# REFACTOR: Conditional-expansion merge node. Deduplicates URLs across the
# first-attempt and expanded search results, ranks by
# ``(hit_count, best_provider_rank)``, and keeps the top-K per
# ``settings.web_search_top_k``. The first-attempt URLs are merged WITH
# (not discarded in favor of) the expanded URLs so the original search's
# provider ranking still anchors the result set.
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ...config import Settings
from .common import qa_question_resolver

logger = logging.getLogger(__name__)

QuestionResolver = Callable[[dict[str, Any]], str]

# REFACTOR: Same unicode-aware URL regex used by ``web_answer`` so the
# merge node's URL extraction matches the answer node's URL extraction
# exactly. Without the unicode ranges, a URL written inline in Chinese
# prose would greedily swallow the surrounding sentence and produce an
# unreachable garbage URL.
_URL_RE = re.compile(
    r"https?://[^\s<>()\[\]{}　-〿＀-￯一-鿿]+"
)


def merge_factory(
    settings: Settings,
    question_resolver: QuestionResolver = qa_question_resolver,  # noqa: ARG001
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a node that combines and ranks search URL candidates.

    The node reads ``state["source_urls"]`` (the entry-point set or the
    previous merge's output) plus the URLs emitted by the most recent
    ``live_web_search`` ToolMessage(s), deduplicates them by canonical
    URL, ranks by ``(hit_count, best_provider_rank, first_seen)``, and
    emits a single ranked ``source_urls`` list of length up to
    ``settings.web_search_top_k``. The original entry-point URLs are
    kept AND combined with the tool-message URLs rather than replaced
    -- the design decision in the 2026-06-17 form.
    """

    def merge(state: dict[str, Any]) -> dict[str, Any]:
        logger.info("MERGE SEARCH RESULTS")
        explicit = _clean_urls(state.get("source_urls") or [])
        tool_urls = _clean_urls(_urls_from_tool_messages(state.get("messages") or []))
        combined = _combine_and_rank(
            first_attempt=explicit,
            expanded=tool_urls,
            top_k=int(getattr(settings, "web_search_top_k", 0) or 0),
        )
        return {"source_urls": combined}

    return merge


def _combine_and_rank(
    first_attempt: list[str], expanded: list[str], top_k: int
) -> list[str]:
    """Combine first-attempt and expanded URLs, then rank and clamp.

    Ranking is ``(hit_count desc, best_provider_rank asc, first_seen
    asc)``:
    - ``hit_count`` rewards URLs returned by multiple queries
      (overlap = stronger evidence).
    - ``best_provider_rank`` anchors the first-attempt URLs that
      were already quality-filtered by the provider.
    - ``first_seen`` preserves the original ordering as a final
      tie-breaker.
    """

    canonical_to_records: dict[str, dict[str, Any]] = {}

    def record(url: str, provider_rank: int) -> None:
        canonical = _canonical_url(url)
        if not canonical:
            return
        entry = canonical_to_records.get(canonical)
        if entry is None:
            entry = {
                "canonical": canonical,
                "display_url": url,
                "hit_count": 0,
                "best_provider_rank": provider_rank,
                "first_seen": len(canonical_to_records),
            }
            canonical_to_records[canonical] = entry
        entry["hit_count"] += 1
        if provider_rank < entry["best_provider_rank"]:
            entry["best_provider_rank"] = provider_rank

    for index, url in enumerate(first_attempt):
        record(url, provider_rank=index)
    # REFACTOR: The expanded URLs are offset by the first-attempt rank
    # so an expanded URL that was also in the first attempt wins on
    # ``best_provider_rank`` and stays anchored in the result set.
    for index, url in enumerate(expanded):
        record(url, provider_rank=len(first_attempt) + index)

    ranked = sorted(
        canonical_to_records.values(),
        key=lambda item: (
            -item["hit_count"],
            item["best_provider_rank"],
            item["first_seen"],
        ),
    )
    chosen = ranked if top_k <= 0 else ranked[:top_k]
    return [item["display_url"] for item in chosen]


def _urls_from_tool_messages(messages: Any) -> list[str]:
    urls: list[str] = []
    if not isinstance(messages, list):
        return urls
    for message in messages:
        if _message_role(message) != "tool":
            continue
        content = getattr(message, "content", None) or ""
        urls.extend(_URL_RE.findall(str(content)))
    return urls


def _message_role(message: Any) -> str:
    role = getattr(message, "type", None)
    if role:
        return str(role)
    return str(message.__class__.__name__).lower()


def _clean_urls(values: Any) -> list[str]:
    """Trim, dedupe, and drop empty URLs from an iterable of values."""

    seen: set[str] = set()
    urls: list[str] = []
    if not isinstance(values, list):
        return urls
    for value in values:
        url = str(value).strip().rstrip(".,;）)】」》")
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _canonical_url(url: str) -> str:
    """Return a canonical form for dedup. Lowercased scheme/host/path."""

    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return ""
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    if not scheme or not netloc:
        return ""
    return urlunsplit((scheme, netloc, parts.path, "", ""))


__all__ = ["merge_factory"]
