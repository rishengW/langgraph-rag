# REFACTOR: Dynamic structural page filtering. Measures the page we actually
# fetched instead of guessing article-ness from the URL shape. Link density and
# content volume separate articles from index/tag listings, login walls, and
# thin doorway shells that no hand-written denylist can enumerate.
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from bs4 import BeautifulSoup

PageShape = Literal["article", "listing", "thin", "gateway", "unknown"]

DEFAULT_MAX_LINK_DENSITY = 0.5
DEFAULT_MIN_CONTENT_WORDS = 60
# A listing page repeats short anchors; an article links out occasionally. The
# ratio is per 100 words so it does not punish long, richly cited articles.
LISTING_ANCHORS_PER_100_WORDS = 12
GATEWAY_TEXT_MAX_WORDS = 25
_WORD_RE = re.compile(r"[A-Za-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?。！？\n]+")
_GATEWAY_TEXT_RE = re.compile(
    r"(?:sign\s?in|log\s?in|register|subscribe to continue|enable javascript|"
    r"verify you are human|access denied|page not found|"
    r"\u767b\u5f55|\u6ce8\u518c|\u9a8c\u8bc1\u7801|\u8bf7\u5f00\u542f|"
    r"\u9875\u9762\u4e0d\u5b58\u5728|\u8bbf\u95ee\u53d7\u9650)",
    re.I,
)


@dataclass(frozen=True)
class PageStructure:
    """Structural measurements of one fetched page."""

    shape: PageShape = "unknown"
    link_density: float = 0.0
    anchor_count: int = 0
    content_words: int = 0
    mean_sentence_words: float = 0.0
    measured: bool = False

    @property
    def is_content_page(self) -> bool:
        """Return whether the page looks like readable prose rather than an index."""

        return self.shape in {"article", "unknown"}


def word_count(text: str) -> int:
    """Count CJK characters and Latin words as comparable content units."""

    return len(_WORD_RE.findall(text or ""))


def assess_page_structure(
    html: str,
    text: str = "",
    *,
    max_link_density: float = DEFAULT_MAX_LINK_DENSITY,
    min_content_words: int = DEFAULT_MIN_CONTENT_WORDS,
) -> PageStructure:
    """Measure link density and content volume to classify a fetched page.

    Args:
        html: Raw page markup. Plain-text documents (e.g. extracted PDFs) yield
            no anchors, so they are judged on content volume alone.
        text: Already-extracted article text. When empty, the markup's own text
            is used.
        max_link_density: Anchor-text share above which a page is a listing.
        min_content_words: Content units below which a page is too thin to ground
            an answer.

    Returns:
        A ``PageStructure`` with ``measured=False`` when there was nothing to
        measure, so callers can abstain instead of rejecting.
    """

    soup = BeautifulSoup(html or "", "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    page_text = (text or "").strip() or soup.get_text(" ", strip=True)
    total_words = word_count(page_text)
    if not total_words:
        return PageStructure()

    anchors = soup.find_all("a")
    anchor_words = sum(word_count(anchor.get_text(" ", strip=True)) for anchor in anchors)
    # Anchor text can be counted from full markup while ``text`` holds only the
    # extracted article, so clamp the ratio into a meaningful range.
    link_density = min(1.0, anchor_words / max(total_words, anchor_words or 1))
    anchors_per_100_words = (len(anchors) / total_words) * 100

    sentences = [part for part in _SENTENCE_SPLIT_RE.split(page_text) if word_count(part) > 0]
    mean_sentence_words = total_words / len(sentences) if sentences else float(total_words)

    shape: PageShape = "article"
    if total_words <= GATEWAY_TEXT_MAX_WORDS and _GATEWAY_TEXT_RE.search(page_text):
        shape = "gateway"
    elif _is_listing(link_density, anchors_per_100_words, max_link_density):
        shape = "listing"
    elif total_words < max(0, int(min_content_words)):
        shape = "thin"

    return PageStructure(
        shape=shape,
        link_density=round(link_density, 3),
        anchor_count=len(anchors),
        content_words=total_words,
        mean_sentence_words=round(mean_sentence_words, 2),
        measured=True,
    )


def _is_listing(
    link_density: float,
    anchors_per_100_words: float,
    max_link_density: float,
) -> bool:
    """Require both a high anchor share and anchor repetition for a listing."""

    threshold = max(0.0, min(1.0, float(max_link_density)))
    if threshold <= 0:
        return False
    return link_density >= threshold and anchors_per_100_words >= LISTING_ANCHORS_PER_100_WORDS


def structure_rejection_reason(structure: PageStructure) -> str | None:
    """Return why a measured page cannot ground an answer, or ``None``."""

    if not structure.measured or structure.is_content_page:
        return None
    return f"{structure.shape}_page"


__all__ = [
    "DEFAULT_MAX_LINK_DENSITY",
    "DEFAULT_MIN_CONTENT_WORDS",
    "GATEWAY_TEXT_MAX_WORDS",
    "LISTING_ANCHORS_PER_100_WORDS",
    "PageShape",
    "PageStructure",
    "assess_page_structure",
    "structure_rejection_reason",
    "word_count",
]
