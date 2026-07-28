"""Lightweight web-search answer node for direct page-context prompting."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable, Sequence
from difflib import SequenceMatcher
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from ...config import Settings
from ...llm.sanitize import strip_citation_artifacts
from ...utils.retry import invoke_with_retry
from .common import message_text, new_chat_model, qa_question_resolver

logger = logging.getLogger(__name__)

QuestionResolver = Callable[[dict[str, Any]], str]

# Match URLs but stop at whitespace, brackets/parens, and any CJK/full-width
# punctuation or ideographs. Without the unicode ranges, a URL written inline in
# Chinese prose (e.g. "https://deepseek.net/zh）虽为官方渠道") would greedily
# swallow the surrounding sentence and produce an unreachable garbage URL.
_URL_RE = re.compile(r"https?://[^\s<>()\[\]{}\u3000-\u303f\uff00-\uffef\u4e00-\u9fff]+")
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def web_answer_factory(
    settings: Settings,
    question_resolver: QuestionResolver = qa_question_resolver,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a node that answers directly from fetched web pages.

    The node intentionally avoids retrievers, embeddings, Chroma, grading, and
    query rewriting. It delegates page loading and prompt assembly to the web
    search lightweight-path modules.
    """

    def web_answer(state: dict[str, Any]) -> dict[str, Any]:
        logger.info("GENERATE WEB ANSWER")
        question = question_resolver(state)
        urls = _extract_source_urls(state, settings)
        # REFACTOR: Increment the lightweight web-answer attempt counter on
        # every entry so the post-web-answer edge can bound the fallback loop
        # to a single retry. The counter is read by ``route_after_web_answer``
        # in src/graph/edges.py.
        attempts = int(state.get("web_answer_attempts", 0) or 0) + 1

        from ...web_search.claim_consensus import assess_status_consensus
        from ...web_search.common import is_page_text_relevant
        from ...web_search.content_fetcher import fetch_pages, is_readable_page
        from ...web_search.page_structure import structure_rejection_reason
        from ...web_search.prompt_builder import build_web_search_prompt

        pages = fetch_pages(
            urls,
            timeout=settings.page_load_timeout,
            max_tokens_per_page=settings.web_search_max_page_tokens,
            cache_ttl_seconds=settings.page_load_cache_ttl_seconds,
            max_concurrent_loads=settings.page_load_max_concurrency,
            min_readable_chars=settings.web_search_min_page_chars,
            min_readable_tokens=settings.web_search_min_page_tokens,
            relevance_query=question,
            js_fallback_enabled=settings.web_search_js_fallback_enabled,
            js_fallback_domains=settings.web_search_js_fallback_domains,
            js_force_domains=settings.web_search_js_force_domains,
            js_retry_budget=settings.web_search_js_retry_budget,
            max_link_density=settings.web_search_max_link_density,
            min_content_words=settings.web_search_min_content_words,
        )

        # Guard against ungrounded answers: if no fetched page produced
        # readable text (all URLs unreachable, empty, or junk), do NOT prompt
        # the model. With an empty context the LLM falls back to its training
        # data and silently answers from stale/parametric knowledge, which is
        # exactly the failure we want to avoid in a retrieval-grounded app.
        readable_pages = [
            page
            for page in pages
            if is_readable_page(
                page.text or "",
                url=page.url or "",
                title=page.title or "",
                query=question,
                min_chars=settings.web_search_min_page_chars,
                min_tokens=settings.web_search_min_page_tokens,
            )
        ]
        # Structural filtering measures the fetched page instead of guessing
        # from its URL: index/tag listings, login walls, and thin shells are
        # removed here so the lexical gate only sees real prose.
        structural_pages = readable_pages
        if settings.web_search_structure_filter_enabled:
            structural_pages = []
            for page in readable_pages:
                # Page objects supplied by callers or tests may predate
                # structural measurement, so an absent structure abstains.
                structure = getattr(page, "structure", None)
                reason = None if structure is None else structure_rejection_reason(structure)
                if reason is None or structure is None:
                    structural_pages.append(page)
                    continue
                logger.info(
                    "Filtered web source by page structure: url=%s reason=%s "
                    "link_density=%.3f content_words=%d",
                    page.url,
                    reason,
                    structure.link_density,
                    structure.content_words,
                )
        topical_pages = [
            page
            for page in structural_pages
            if is_page_text_relevant(
                page.text or "",
                question,
                title=page.title or "",
            )
        ]
        topical_pages = _rescue_semantically_relevant_pages(
            structural_pages,
            topical_pages,
            question,
            settings,
        )
        date_ranked_pages, date_conflict_urls = _rank_pages_by_publication_date(
            topical_pages,
            question,
        )
        if date_conflict_urls:
            logger.info(
                "Filtered web sources with conflicting publication years: urls=%s",
                ", ".join(date_conflict_urls),
            )
        relevant_pages, duplicate_urls = _dedupe_near_duplicate_pages(date_ranked_pages)
        if duplicate_urls:
            logger.info(
                "Filtered near-duplicate web source content: urls=%s",
                ", ".join(duplicate_urls),
            )
        missing_years = _missing_year_evidence(relevant_pages, question)
        if missing_years:
            logger.info(
                "Filtered web sources without complete requested-year coverage: years=%s",
                ", ".join(missing_years),
            )
            relevant_pages = []
        filtered_urls = [page.url for page in structural_pages if page not in topical_pages]
        if filtered_urls:
            logger.info(
                "Filtered readable but off-topic web source content: urls=%s",
                ", ".join(filtered_urls),
            )
        if not relevant_pages:
            _record_domain_outcomes(settings, pages, relevant_pages)
            attempted = ", ".join(page.url for page in pages if page.url) or "none"
            logger.warning(
                "No relevant readable web source content for question; skipping LLM call "
                "to avoid an ungrounded answer (attempted URLs: %s)",
                attempted,
            )
            return {
                # Do not persist or display candidates that failed the final
                # readability/relevance gate as if they grounded an answer.
                "source_urls": [],
                "messages": [
                    AIMessage(
                        content=(
                            "I couldn't retrieve readable content relevant to this "
                            "question from the web sources, so I don't have grounded "
                            "information to answer it and won't guess.\n\n"
                            f"Attempted sources: {attempted}\n\n"
                            "Suggestions:\n"
                            "- Try using specific keywords instead of a full question "
                            "(e.g. 'DeepSeek V4 Pro release' instead of 'what is the "
                            "latest model of deepseek')\n"
                            "- Include a year if your question is time-sensitive\n"
                            "- Provide a specific source URL you'd like me to read"
                        )
                    )
                ],
                # Signal the post-web-answer edge to run one expanded search
                # pass. A second failure terminates with this grounded refusal;
                # web-search mode never substitutes model training knowledge.
                "web_answer_no_readable_content": True,
                "web_answer_attempts": attempts,
            }

        _record_domain_outcomes(settings, pages, relevant_pages)

        status_consensus = assess_status_consensus(relevant_pages, question)
        if status_consensus.prompt_instruction:
            prompt = build_web_search_prompt(
                question,
                relevant_pages,
                grounding_note=status_consensus.prompt_instruction,
            )
        else:
            prompt = build_web_search_prompt(question, relevant_pages)

        try:
            result = invoke_with_retry(
                new_chat_model(settings),
                [HumanMessage(content=prompt)],
                max_retries=settings.dashscope_max_retries,
            )
            # Models sometimes reproduce other assistants' citation syntax
            # (e.g. the source-index/line-range marker shape). Our sources
            # carry no indices or line numbers, so those markers reference
            # nothing and are removed.
            content = strip_citation_artifacts(message_text(result))
        except Exception as exc:
            logger.error("Web answer error: %s", exc)
            content = (
                "I could not reach the chat model to synthesize an answer from "
                "the fetched web sources."
            )

        return {
            # The chat UI and session metadata should expose only sources that
            # were actually admitted to the grounded answer prompt.
            "source_urls": [page.url for page in relevant_pages if page.url],
            "messages": [AIMessage(content=content)],
            # REFACTOR: Clear the fallback flag on the success branch so a
            # later ``web_answer`` failure (e.g. a different question in the
            # same session) starts from a clean state, and stamp the attempt
            # counter for parity with the failure branch.
            "web_answer_no_readable_content": False,
            "web_answer_attempts": attempts,
        }

    return web_answer


def _extract_source_urls(state: dict[str, Any], settings: Settings) -> list[str]:
    # 1. URLs explicitly placed in graph state by the entry point.
    explicit_urls = _clean_urls(state.get("source_urls") or [])
    if explicit_urls:
        return explicit_urls

    # 2. The discovered/configured URLs the entry point set on Settings. These
    #    are the clean, provider-ranked URLs from the session's web-search
    #    discovery (e.g. the top-K kept after quality filtering) and are the
    #    authoritative source set for the session. They take precedence over
    #    the agent's in-graph live_web_search tool output, which is a narrower,
    #    lower-quality re-search that can otherwise shadow the curated URLs and
    #    send the answer node to chase junk/unreachable pages.
    settings_urls = _clean_urls(getattr(settings, "source_urls", None) or [])
    if settings_urls:
        return settings_urls

    # 3. URLs emitted by the live_web_search tool at runtime, used only when the
    #    session has no curated URLs. Only tool messages are parsed -- never
    #    AI/agent prose, which may contain URLs glued to surrounding text
    #    (e.g. "https://x.com/zh）虽为官方渠道").
    return _clean_urls(_urls_from_tool_messages(state.get("messages") or []))


def _urls_from_tool_messages(messages: Sequence[Any]) -> list[str]:
    urls: list[str] = []
    for message in messages:
        if _message_role(message) != "tool":
            continue
        urls.extend(_URL_RE.findall(message_text(message)))
    return urls


def _message_role(message: Any) -> str:
    role = getattr(message, "type", None)
    if role:
        return str(role)
    if isinstance(message, (tuple, list)) and message:
        return str(message[0])
    return str(message.__class__.__name__).lower()


def _clean_urls(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for value in values:
        url = str(value).strip().rstrip(".,;）)】」》")
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _record_domain_outcomes(
    settings: Settings,
    pages: Sequence[Any],
    admitted: Sequence[Any],
) -> None:
    """Feed this turn's fetch outcomes into the adaptive domain reputation prior."""

    from ...web_search.reputation import (
        OUTCOME_GROUNDED,
        OUTCOME_REJECTED,
        OUTCOME_UNREACHABLE,
        build_reputation_store,
    )

    store = build_reputation_store(settings)
    if store is None:
        return

    admitted_urls = {getattr(page, "url", "") for page in admitted}
    outcomes: list[tuple[str, str]] = []
    for page in pages:
        url = str(getattr(page, "url", "") or "")
        if not url:
            continue
        if url in admitted_urls:
            outcomes.append((url, OUTCOME_GROUNDED))
        elif getattr(page, "text", ""):
            outcomes.append((url, OUTCOME_REJECTED))
        else:
            outcomes.append((url, OUTCOME_UNREACHABLE))
    try:
        store.record_many(outcomes)
    except Exception as exc:  # noqa: BLE001 - telemetry must never break answering
        logger.warning("Could not record domain reputation outcomes: %s", exc)


def _rescue_semantically_relevant_pages(
    candidates: Sequence[Any],
    admitted: list[Any],
    question: str,
    settings: Settings,
) -> list[Any]:
    """Re-admit lexically rejected pages that are semantically on topic.

    Lexical coverage misses paraphrases and cross-language pairs. Similarity is
    only allowed to add pages back; the hard year, quantity, and typed-evidence
    checks already ran inside ``is_page_text_relevant`` and are not revisited
    here, so a rescued page still has to survive the later evidence gates.
    """

    from ...web_search.semantic import build_semantic_scorer

    scorer = build_semantic_scorer(settings)
    rejected = [page for page in candidates if page not in admitted]
    if scorer is None or not rejected:
        return admitted

    leads = [
        f"{getattr(page, 'title', '') or ''} {str(getattr(page, 'text', '') or '')[:1200]}".strip()
        for page in rejected
    ]
    similarities = scorer.similarities(question, leads)
    if len(similarities) != len(rejected):
        return admitted

    threshold = float(settings.web_search_semantic_min_similarity)
    rescued = {
        id(page): similarity
        for page, similarity in zip(rejected, similarities, strict=True)
        if similarity >= threshold
    }
    if not rescued:
        return admitted

    for page in rejected:
        if id(page) in rescued:
            logger.info(
                "Rescued semantically relevant web source: url=%s similarity=%.3f",
                getattr(page, "url", ""),
                rescued[id(page)],
            )
    # Preserve the original ranked order of the candidate list.
    return [page for page in candidates if page in admitted or id(page) in rescued]


def _dedupe_near_duplicate_pages(
    pages: Sequence[Any],
    *,
    similarity_threshold: float = 0.88,
) -> tuple[list[Any], list[str]]:
    """Preserve ranked pages while removing syndicated copies of the same text."""

    kept: list[Any] = []
    kept_fingerprints: list[str] = []
    duplicate_urls: list[str] = []
    for page in pages:
        fingerprint = _page_fingerprint(page)
        is_duplicate = bool(fingerprint) and any(
            SequenceMatcher(None, fingerprint, existing, autojunk=False).ratio()
            >= similarity_threshold
            for existing in kept_fingerprints
        )
        if is_duplicate:
            url = str(getattr(page, "url", "") or "").strip()
            if url:
                duplicate_urls.append(url)
            continue
        kept.append(page)
        kept_fingerprints.append(fingerprint)
    return kept, duplicate_urls


def _rank_pages_by_publication_date(
    pages: Sequence[Any],
    question: str,
) -> tuple[list[Any], list[str]]:
    """Apply explicit-year compatibility and stable freshness ordering."""

    from ...web_search.recency import assess_publication_date

    ranked: list[tuple[int, int, Any]] = []
    conflict_urls: list[str] = []
    for position, page in enumerate(pages):
        assessment = assess_publication_date(
            getattr(page, "publication_date", None),
            question,
        )
        if assessment.conflicts_with_required_year:
            url = str(getattr(page, "url", "") or "").strip()
            if url:
                conflict_urls.append(url)
            continue
        ranked.append((assessment.score, position, page))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [page for _score, _position, page in ranked], conflict_urls


def _page_fingerprint(page: Any) -> str:
    title = str(getattr(page, "title", "") or "")
    text = str(getattr(page, "text", "") or "")[:4000]
    normalized = unicodedata.normalize("NFKC", f"{title} {text}").casefold()
    return " ".join(normalized.split())


def _missing_year_evidence(pages: Sequence[Any], question: str) -> list[str]:
    """Return requested years that no admitted page can answer independently."""

    years = list(dict.fromkeys(_YEAR_RE.findall(question)))
    if len(years) < 2:
        return []

    from ...web_search.common import is_page_text_relevant

    without_years = _YEAR_RE.sub(" ", question)
    return [
        year
        for year in years
        if not any(
            is_page_text_relevant(
                str(getattr(page, "text", "") or ""),
                f"{' '.join(without_years.split())} {year}".strip(),
                title=str(getattr(page, "title", "") or ""),
            )
            for page in pages
        )
    ]


__all__ = ["web_answer_factory"]
