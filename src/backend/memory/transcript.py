"""Pure transcript handling for automatic memory extraction.

Turn counting, slice selection, and prompt rendering, with no ``Settings`` and no
I/O. ``count_turns`` is the authoritative turn counter: it reads the thread's
checkpointed message list, so it cannot drift from the transcript the way a
separate counter would.

Role classification mirrors ``src/frontend/chat/api.py::_serialize_messages`` so the
extractor and the history endpoint agree on what a user or assistant message is.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

USER_LABEL: Final[str] = "User:"
ASSISTANT_LABEL: Final[str] = "Assistant:"
TRUNCATION_MARKER: Final[str] = " ... [truncated]"

#: Hard cap on retained messages, applied before the character cap. Bounds the
#: work done per extraction independently of message size.
MAX_SLICE_MESSAGES: Final[int] = 200

#: Zero-width marker the chat layer prepends to an injected recall note so this
#: module can exclude it without inspecting its text. Imported lazily to avoid a
#: cycle with :mod:`.recall`.
_NOTE_MARKER_CACHE: list[str] = []


def _note_marker() -> str:
    if not _NOTE_MARKER_CACHE:
        from .recall import MEMORY_NOTE_LABEL, MEMORY_NOTE_MARKER

        _NOTE_MARKER_CACHE.extend((MEMORY_NOTE_MARKER, MEMORY_NOTE_LABEL))
    return _NOTE_MARKER_CACHE[0]


def _note_label() -> str:
    _note_marker()
    return _NOTE_MARKER_CACHE[1]


def message_role(message: Any) -> str:
    """Classify a message as user, assistant, system, tool, or its raw kind."""

    kind = getattr(message, "type", None) or message.__class__.__name__.lower()
    kind = str(kind)
    if kind.startswith("human") or kind == "user":
        return "user"
    if kind.startswith("ai") or kind == "assistant":
        return "assistant"
    if kind in ("tool", "function"):
        return "tool"
    if kind.startswith("system"):
        return "system"
    return kind


def normalize_message_content(message: Any) -> str:
    """Return a message's content as trimmed text.

    Non-string content (a list of content blocks, as some providers emit) is
    flattened by concatenating the text of every element that carries text.
    Elements with no text are discarded rather than stringified, so a tool-call
    block does not leak ``{'type': 'tool_use', ...}`` into the prompt.
    """

    content = getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for element in content:
            text = _element_text(element)
            if text:
                parts.append(text)
        return " ".join(parts).strip()
    if isinstance(content, dict):
        return _element_text(content).strip()
    return str(content).strip()


def _element_text(element: Any) -> str:
    if isinstance(element, str):
        return element.strip()
    if isinstance(element, dict):
        for key in ("text", "content", "value"):
            value = element.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    text = getattr(element, "text", None)
    return text.strip() if isinstance(text, str) else ""


def count_turns(messages: Sequence[Any]) -> int:
    """Return the number of user messages, i.e. the authoritative Turn_Count."""

    return sum(1 for message in messages if message_role(message) == "user")


def is_memory_note(message: Any) -> bool:
    """True when the message carries injected recall content.

    Decided by marker, not by content inspection, so a recalled memory can never
    be re-extracted into a new record.
    """

    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return False
    head = content[:200]
    return _note_marker() in head or head.startswith(_note_label())


def select_slice(
    messages: Sequence[Any],
    *,
    watermark: int,
    trigger: str = "round_complete",
) -> list[Any]:
    """Return the extractable messages after the watermark-th user turn.

    Both triggers reduce to the same rule: everything after the watermark. For
    ``round_complete`` the upper bound is the latest turn, which is also the end
    of the checkpointed list, so no separate branch is needed.

    Excludes system messages, tool messages, injected recall notes, and messages
    whose normalized content is blank (which removes tool-call carrier messages).
    """

    del trigger  # both triggers share one rule; kept for call-site clarity

    seen_user = 0
    start = len(messages)
    for index, message in enumerate(messages):
        if message_role(message) == "user":
            seen_user += 1
            if seen_user == watermark + 1:
                start = index
                break
    if watermark <= 0:
        start = 0

    kept: list[Any] = []
    for message in messages[start:]:
        role = message_role(message)
        if role not in ("user", "assistant"):
            continue
        if is_memory_note(message):
            continue
        if not normalize_message_content(message):
            continue
        kept.append(message)
    return kept


def render_slice(messages: Sequence[Any], *, max_chars: int) -> str:
    """Render a slice as role-labelled blocks separated by one blank line.

    Applies the message cap first, then the character cap, dropping whole
    messages from the start so the most recent turns survive. A single message
    that alone exceeds the budget is truncated with a marker rather than dropped,
    which keeps a long final turn extractable instead of yielding an empty slice.
    """

    limit = max(1, int(max_chars))
    retained = list(messages)[-MAX_SLICE_MESSAGES:]

    blocks: list[str] = []
    for message in retained:
        label = USER_LABEL if message_role(message) == "user" else ASSISTANT_LABEL
        blocks.append(f"{label} {normalize_message_content(message)}")

    while blocks:
        rendered = "\n\n".join(blocks)
        if len(rendered) <= limit:
            return rendered
        if len(blocks) == 1:
            # A lone oversized message is truncated rather than dropped, so a
            # long final turn stays extractable. When the budget is smaller than
            # the marker itself there is no room to signal truncation, so the
            # length bound wins over the marker.
            if limit <= len(TRUNCATION_MARKER):
                return blocks[0][:limit]
            return blocks[0][: limit - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER
        blocks.pop(0)

    return ""


__all__ = [
    "ASSISTANT_LABEL",
    "MAX_SLICE_MESSAGES",
    "TRUNCATION_MARKER",
    "USER_LABEL",
    "count_turns",
    "is_memory_note",
    "message_role",
    "normalize_message_content",
    "render_slice",
    "select_slice",
]
