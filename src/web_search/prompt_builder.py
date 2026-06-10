# REFACTOR: Direct-answer prompt assembly for lightweight web-search context.
from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from .content_fetcher import FetchedPage, estimate_tokens, truncate_to_token_budget


def build_web_search_prompt(
    question: str,
    pages: Sequence[FetchedPage],
    *,
    max_total_tokens: int = 100000,
    today: date | None = None,
) -> str:
    """Assemble a direct-answer prompt from fetched web pages.

    Args:
        question: User question to answer.
        pages: Fetched pages to include as cited source context.
        max_total_tokens: Rough total prompt token budget.
        today: Current date used to anchor the model in the present. Defaults
            to the system date. Passed explicitly in tests for determinism.

    Returns:
        Prompt text containing the question and included web source excerpts.
    """

    current_date = (today or date.today()).isoformat()
    # Fixed instructional preamble. This is overhead that anchors the model in
    # the present and tells it to trust the sources; it must not be charged
    # against the per-source token budget, otherwise a longer instruction
    # block would silently starve the actual source content.
    instructions = (
        f"Today's date is {current_date}.\n\n"
        "You are answering using live web sources retrieved just now. These "
        "sources reflect the current state of the world and are MORE UP TO "
        "DATE than your own training data. When a source conflicts with your "
        "prior knowledge, trust the source. Do not dismiss information as "
        "future, unreleased, or non-existent simply because it postdates your "
        "training cutoff — if the sources describe it, treat it as real and "
        "current.\n\n"
        "IMPORTANT — Semantic matching: The question may use different words "
        "than the sources. Match on MEANING, not exact phrasing. For example: "
        "if the question asks about the 'latest model' and a source describes "
        "a 'new release' or 'V4 Pro launched in 2026', those ARE the answer — "
        "do NOT reject them because the source doesn't use the exact words "
        "'latest model'. Bridge vocabulary gaps between the question and the "
        "sources.\n\n"
        "Answer the question using the web sources below. Cite sources by URL "
        "where possible. If the sources genuinely do not contain the answer "
        "after semantic matching, say so plainly rather than guessing from "
        "prior knowledge.\n\n"
    )
    question_block = f"Question: {question.strip()}\n\nSources:"
    header = f"{instructions}{question_block}"
    footer = "\n\nAnswer:"
    # Budget governs the question block + source sections. The fixed
    # instructional preamble is excluded so it can never crowd out sources.
    budget = max(0, int(max_total_tokens))
    remaining_tokens = budget - estimate_tokens(question_block) - estimate_tokens(footer)
    sections: list[str] = []

    for page in pages:
        section = _source_section(page, remaining_tokens)
        if not section:
            continue
        sections.append(section)
        remaining_tokens -= estimate_tokens(section)
        if remaining_tokens <= 0:
            break

    if not sections:
        sections.append("No readable web source content was available.")
    source_context = "\n\n".join(sections)
    return f"{header}\n\n{source_context}{footer}"


def _source_section(page: FetchedPage, remaining_tokens: int) -> str:
    if remaining_tokens <= 0 or not page.text.strip():
        return ""

    title_suffix = f" (Title: {page.title})" if page.title else ""
    prefix = f"--- Source: {page.url}{title_suffix} ---\n"
    suffix = "\n--- End Source ---"
    content_budget = remaining_tokens - estimate_tokens(prefix) - estimate_tokens(suffix)
    if content_budget <= 0:
        return ""

    content = truncate_to_token_budget(page.text, content_budget)
    if not content:
        return ""
    return f"{prefix}{content}{suffix}"


__all__ = ["build_web_search_prompt"]
