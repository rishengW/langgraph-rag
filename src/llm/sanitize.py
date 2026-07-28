# REFACTOR: Strip hallucinated citation markers from user-facing answers.
#
# Models trained on transcripts from tool-augmented assistants sometimes
# reproduce those assistants' internal citation syntax — for example
# "【199†L91-L126】" (a source index, a dagger, and a line range) or
# "[oaicite:0]". Our prompts number nothing and expose no line ranges, so such
# markers point at nothing: they cannot be verified and they look broken in the
# UI. Stripping them is safe because they carry no information about our
# sources, which are cited by URL.
from __future__ import annotations

import re

# 【…†…】 and its ASCII-bracket variants, plus the OpenAI tool markers.
_CITATION_ARTIFACT_PATTERNS = (
    # Full-width bracket citation containing a dagger, e.g. 【199†L91-L126】.
    re.compile(
        r"[\u3010\uff3b][^\u3011\uff3d\n]{0,120}?[\u2020\u2021][^\u3011\uff3d\n]{0,120}?[\u3011\uff3d]"
    ),
    # ASCII bracket variant of the same shape, e.g. [199†L91-L126].
    re.compile(r"\[[^\]\n]{0,120}?[\u2020\u2021][^\]\n]{0,120}?\]"),
    # Tool-call residue: [oaicite:0], 【oaicite:1】, citeturn3search2.
    re.compile(r"[\u3010\[]\s*oaicite\s*:[^\u3011\]\n]{0,60}[\u3011\]]", re.I),
    re.compile(r"\bcite(?:turn|start|end)\w*", re.I),
    re.compile(r"\u2020L\d+(?:-L\d+)?"),
)
# Longest marker prefix that could still be completed by later stream chunks.
MAX_PARTIAL_MARKER_CHARS = 160
_OPENING_MARKER_RE = re.compile(r"[\u3010\[\uff3b][^\u3011\]\uff3d\n]{0,160}$")
_PARTIAL_CITE_RE = re.compile(
    r"\bcite(?:t(?:u(?:r(?:n)?)?)?|s(?:t(?:a(?:r(?:t)?)?)?)?|e(?:n(?:d)?)?)?\w*$",
    re.I,
)
_PARTIAL_LINE_REF_RE = re.compile(r"[\u2020\u2021]L?\d*(?:-L?\d*)?$")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"[ \t]+([.,;:!?)\]\u3002\uff0c\uff09])")
_REPEATED_SPACE_RE = re.compile(r"[ \t]{2,}")


def strip_citation_artifacts(text: str) -> str:
    """Remove fabricated citation markers and tidy the whitespace they leave.

    URLs, ordinary bracketed text, and footnote-style references without a
    dagger are preserved: only the tool-marker shapes above are removed.
    """

    if not text:
        return text

    cleaned = text
    for pattern in _CITATION_ARTIFACT_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    if cleaned == text:
        return text

    cleaned = _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", cleaned)
    cleaned = _REPEATED_SPACE_RE.sub(" ", cleaned)
    return "\n".join(line.rstrip() for line in cleaned.split("\n"))


class CitationArtifactFilter:
    """Streaming-safe sanitizer for token deltas.

    A marker can be split across chunks ("【199", "†L91-", "L126】"), so a
    tail that could still become a marker is held back until the marker either
    completes or is ruled out. Call :meth:`flush` at the end of a stream to
    release whatever is still buffered.
    """

    def __init__(self, max_buffer_chars: int = MAX_PARTIAL_MARKER_CHARS) -> None:
        self._buffer = ""
        self._max_buffer_chars = max(1, int(max_buffer_chars))

    def feed(self, chunk: str) -> str:
        """Return the sanitized text that is safe to emit for this chunk."""

        if not chunk:
            return ""
        self._buffer += chunk
        # Choose the hold point on the RAW buffer, then sanitize only the part
        # being released. Sanitizing first would let a pattern match half of an
        # in-flight marker and mangle the text around it.
        hold_from = _partial_marker_start(self._buffer, self._max_buffer_chars)
        emit_raw, self._buffer = self._buffer[:hold_from], self._buffer[hold_from:]
        return strip_citation_artifacts(emit_raw)

    def flush(self) -> str:
        """Return any remaining buffered text, sanitized."""

        remaining = strip_citation_artifacts(self._buffer)
        self._buffer = ""
        return remaining


def _partial_marker_start(text: str, max_buffer_chars: int) -> int:
    """Return the index from which text may still be part of a marker."""

    tail_start = max(0, len(text) - max_buffer_chars)
    tail = text[tail_start:]
    for pattern in (_OPENING_MARKER_RE, _PARTIAL_CITE_RE, _PARTIAL_LINE_REF_RE):
        match = pattern.search(tail)
        if match:
            return tail_start + match.start()
    return len(text)


__all__ = [
    "MAX_PARTIAL_MARKER_CHARS",
    "CitationArtifactFilter",
    "strip_citation_artifacts",
]
