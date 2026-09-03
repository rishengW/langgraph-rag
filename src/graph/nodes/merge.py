# REFACTOR: Conditional-expansion merge node. Deduplicates URLs across the
# first-attempt and expanded search results, ranks by semantic relevance,
# quality, overlap, and provider rank, and keeps the top-K per
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
from ...web_search.common import (
    host_quality_score,
    registrable_domain,
    text_relevance_delta,
)
from ...web_search.reputation import build_reputation_store, reputation_delta
from .common import chat_question_resolver

logger = logging.getLogger(__name__)

QuestionResolver = Callable[[dict[str, Any]], str]

# REFACTOR: Same unicode-aware URL regex used by ``web_answer`` so the
# merge node's URL extraction matches the answer node's URL extraction
# exactly. Without the unicode ranges, a URL written inline in Chinese
# prose would greedily swallow the surrounding sentence and produce an
# unreachable garbage URL.
_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}　-〿＀-￯一-鿿]+")


def merge_factory(
    settings: Settings,
    question_resolver: QuestionResolver = chat_question_resolver,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a node that combines and ranks search URL candidates.

    The node reads ``state["source_urls"]`` (the entry-point set or the
    previous merge's output) plus the per-query URL sets emitted by bounded
    web search. It falls back to recent ``live_web_search`` ToolMessages for
    compatibility, deduplicates by canonical URL, ranks semantic relevance
    before cross-query overlap, and
    emits a single ranked ``source_urls`` list of length up to
    ``settings.web_search_top_k``. The original entry-point URLs are
    kept AND combined with the tool-message URLs rather than replaced
    -- the design decision in the 2026-06-17 form.
    """

    reputation_store = build_reputation_store(settings)
    min_samples = int(getattr(settings, "web_search_domain_reputation_min_samples", 5) or 5)

    def domain_prior(url: str) -> int:
        """Return the learned per-domain ranking prior for one URL."""

        if reputation_store is None:
            return 0
        try:
            stats = reputation_store.stats(url)
        except Exception as exc:  # noqa: BLE001 - reputation must never break merge
            logger.warning("Domain reputation lookup failed for %s: %s", url, exc)
            return 0
        return reputation_delta(stats, min_samples=min_samples)

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
        message_result_sets = _url_sets_from_tool_messages(current_turn_messages)
        state_result_sets = _clean_url_sets(state.get("web_search_results"))
        result_sets = state_result_sets or message_result_sets
        metadata_sets = _clean_result_metadata_sets(state.get("web_search_result_metadata"))
        original_question = str(state.get("current_question") or "").strip()
        if not original_question:
            try:
                original_question = question_resolver(state).strip()
            except (IndexError, KeyError, TypeError):
                original_question = ""
        combined = _combine_and_rank_sets(
            first_attempt=explicit,
            result_sets=result_sets,
            metadata_sets=metadata_sets,
            top_k=int(getattr(settings, "web_search_top_k", 0) or 0),
            original_question=original_question,
            domain_prior=domain_prior,
        )
        return {"source_urls": combined}

    return merge


def _combine_and_rank(first_attempt: list[str], expanded: list[str], top_k: int) -> list[str]:
    """Combine first-attempt and expanded URLs, then rank and clamp.

    Ranking uses semantic relevance, overall quality, overlap, provider rank,
    then insertion order:
    - ``relevance_score`` prevents an off-topic URL repeated by several weak
      queries from outranking one strongly relevant result.
    - ``quality_score`` preserves the provider result's complete URL + text
      score as a secondary relevance signal.
    - ``hit_count`` rewards URLs returned by multiple queries
      (overlap across the pre-graph refresh AND the in-graph search =
      stronger evidence) and is the dominant signal.
    - ``best_provider_rank`` is each URL's own position within whichever
      search returned it. The two sets are NOT offset against each other,
      so position 0 of the in-graph contextualized search ties with
      position 0 of the pre-graph refresh rather than always losing to it.
    - ``first_seen`` preserves insertion order as a final tie-breaker.
    """

    return _combine_and_rank_sets(first_attempt, [expanded], top_k=top_k)


def _combine_and_rank_sets(
    first_attempt: list[str],
    result_sets: list[list[str]],
    top_k: int,
    metadata_sets: list[list[dict[str, Any]]] | None = None,
    original_question: str = "",
    domain_prior: Callable[[str], int] | None = None,
) -> list[str]:
    """Rank against the original question, then diversify selected domains."""

    canonical_to_records: dict[str, dict[str, Any]] = {}
    prior = domain_prior or (lambda _url: 0)

    def record(
        url: str,
        provider_rank: int,
        *,
        relevance_score: int = 0,
        original_relevance_score: int = 0,
        quality_score: int = 0,
        has_score: bool = False,
        count_hit: bool = True,
    ) -> None:
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
                "best_relevance_score": relevance_score,
                "best_original_relevance_score": original_relevance_score,
                "best_quality_score": quality_score,
                "host_quality_score": host_quality_score(url) + prior(url),
                "has_score": has_score,
                "first_seen": len(canonical_to_records),
            }
            canonical_to_records[canonical] = entry
        if count_hit:
            entry["hit_count"] += 1
        if provider_rank < entry["best_provider_rank"]:
            entry["best_provider_rank"] = provider_rank
        if has_score and not entry["has_score"]:
            entry["best_relevance_score"] = relevance_score
            entry["best_original_relevance_score"] = original_relevance_score
            entry["best_quality_score"] = quality_score
            entry["has_score"] = True
        elif has_score:
            entry["best_relevance_score"] = max(entry["best_relevance_score"], relevance_score)
            entry["best_quality_score"] = max(entry["best_quality_score"], quality_score)
            entry["best_original_relevance_score"] = max(
                entry["best_original_relevance_score"],
                original_relevance_score,
            )

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

    # URL sets remain the compatibility source of truth for membership and
    # cross-query overlap. Metadata mirrors those URLs when artifacts are
    # available, but a mixed/legacy batch may contain metadata for only some
    # results. Recording metadata instead of URL sets would silently drop the
    # unscored URLs and lose their overlap signal.
    result_canonicals: set[str] = set()
    for result_set in result_sets:
        for index, url in enumerate(result_set):
            canonical = _canonical_url(url)
            if canonical:
                result_canonicals.add(canonical)
            record(url, provider_rank=index)

    if metadata_sets:
        for metadata_set in metadata_sets:
            for index, item in enumerate(metadata_set):
                url = str(item.get("url") or "").strip()
                if not url:
                    continue
                canonical = _canonical_url(url)
                result_text = " ".join(
                    part
                    for part in (
                        str(item.get("title") or "").strip(),
                        str(item.get("snippet") or "").strip(),
                    )
                    if part
                )
                record(
                    url,
                    provider_rank=_safe_int(item.get("provider_rank"), index),
                    relevance_score=_safe_int(item.get("relevance_score"), 0),
                    original_relevance_score=(
                        text_relevance_delta(result_text, original_question)
                        if result_text and original_question
                        else 0
                    ),
                    quality_score=_safe_int(item.get("quality_score"), 0),
                    has_score=True,
                    # Usually metadata enriches a URL already counted above.
                    # Still accept metadata-only callers and count each
                    # per-query occurrence when no URL set contains it.
                    count_hit=canonical not in result_canonicals,
                )

    ranked = sorted(
        canonical_to_records.values(),
        key=lambda item: (
            -(item["best_original_relevance_score"] + item["host_quality_score"]),
            -item["best_original_relevance_score"],
            -item["host_quality_score"],
            -item["best_relevance_score"],
            -item["best_quality_score"],
            -item["hit_count"],
            item["best_provider_rank"],
            item["first_seen"],
        ),
    )
    chosen = _select_domain_diverse(ranked, top_k=top_k)
    return [item["display_url"] for item in chosen]


def _select_domain_diverse(
    ranked: list[dict[str, Any]],
    *,
    top_k: int,
    preferred_per_domain: int = 2,
) -> list[dict[str, Any]]:
    """Prefer domain diversity, then backfill when too few alternatives exist."""

    if top_k <= 0:
        return ranked

    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    domain_counts: dict[str, int] = {}
    for item in ranked:
        domain = registrable_domain(str(item.get("display_url") or ""))
        if not domain or domain_counts.get(domain, 0) < preferred_per_domain:
            selected.append(item)
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        else:
            deferred.append(item)
        if len(selected) >= top_k:
            return selected[:top_k]

    selected.extend(deferred[: max(0, top_k - len(selected))])
    return selected[:top_k]


def _urls_from_tool_messages(messages: Any) -> list[str]:
    return [url for result_set in _url_sets_from_tool_messages(messages) for url in result_set]


def _url_sets_from_tool_messages(messages: Any) -> list[list[str]]:
    result_sets: list[list[str]] = []
    if not isinstance(messages, list):
        return result_sets
    for message in messages:
        if _message_role(message) != "tool":
            continue
        content = getattr(message, "content", None) or ""
        urls = _clean_urls(_URL_RE.findall(str(content)))
        if urls:
            result_sets.append(urls)
    return result_sets


def _clean_url_sets(values: Any) -> list[list[str]]:
    if not isinstance(values, list):
        return []
    result_sets: list[list[str]] = []
    for value in values:
        urls = _clean_urls(value)
        if urls:
            result_sets.append(urls)
    return result_sets


def _clean_result_metadata_sets(values: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(values, list):
        return []
    result_sets: list[list[dict[str, Any]]] = []
    for value in values:
        if not isinstance(value, list):
            continue
        result_set = [item for item in value if isinstance(item, dict) and item.get("url")]
        if result_set:
            result_sets.append(result_set)
    return result_sets


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
