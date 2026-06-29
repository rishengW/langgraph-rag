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
        # REFACTOR: Scope tool-message URL harvesting to the CURRENT turn only.
        # ``state["messages"]`` is the full per-thread checkpoint transcript
        # (the ``add_messages`` reducer appends and never resets between turns),
        # so scanning all of it pulls prior turns' ``live_web_search`` URLs into
        # this turn's candidate pool. Those stale URLs have valid hit_count /
        # provider_rank / first_seen values, so ranking never removes them and
        # the top-K clamp keeps them. Restricting the scan to messages after the
        # last HumanMessage ensures only this turn's search results are eligible.
        current_turn_messages = _current_turn_messages(state.get("messages") or [])
        tool_urls = _clean_urls(_urls_from_tool_messages(current_turn_messages))
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
      (overlap across the pre-graph refresh AND the in-graph search =
      stronger evidence) and is the dominant signal.
    - ``best_provider_rank`` is each URL's own position within whichever
      search returned it. The two sets are NOT offset against each other,
      so position 0 of the in-graph contextualized search ties with
      position 0 of the pre-graph refresh rather than always losing to it.
    - ``first_seen`` preserves insertion order as a final tie-breaker.
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

    # REFACTOR: Rank both sets by their OWN provider position (no offset).
    # Previously the expanded/in-graph URLs were offset by
    # ``len(first_attempt)``, which guaranteed every pre-graph refresh URL
    # out-ranked every in-graph search URL on ``best_provider_rank`` whenever
    # hit counts tied. That anchored the pre-graph refresh -- which for a vague
    # follow-up is the WEAKER, less-contextualized search -- above the
    # contextualized in-graph results. Scoring both by their own position lets
    # the better-ranked URL win regardless of which search produced it, while
    # overlap (hit_count) still promotes URLs returned by both.
    for index, url in enumerate(first_attempt):
        record(url, provider_rank=index)
    for index, url in enumerate(expanded):
        record(url, provider_rank=index)

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


def _current_turn_messages(messages: Any) -> list[Any]:
    """Return only the messages belonging to the current chat turn.

    The current turn is everything from the most recent human/user message
    onward. In chat mode ``messages`` is the full per-thread checkpoint
    transcript, so this slice excludes prior turns' tool messages (whose
    ``live_web_search`` URLs would otherwise contaminate this turn's source
    pool). When no human message is present (e.g. single-shot QA state), the
    full list is returned unchanged.
    """

    if not isinstance(messages, list):
        return []
    last_human = -1
    for index, message in enumerate(messages):
        if _message_role(message).startswith("human") or _message_role(message) == "user":
            last_human = index
    if last_human < 0:
        return list(messages)
    return list(messages[last_human:])


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
