from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..config import Settings


def settings_for_session(
    base: Settings,
    urls: list[str],
    thread_id: str,
    isolated: bool,
) -> Settings:
    """Return settings with per-session Chroma isolation when requested."""

    session_urls = list(urls)
    if not isolated:
        return replace(base, source_urls=session_urls)

    base_chroma_dir = Path(base.chroma_dir)
    return replace(
        base,
        source_urls=session_urls,
        chroma_dir=base_chroma_dir / "chat" / thread_id,
        collection_name=f"{base.collection_name}-chat-{thread_id}",
    )


_settings_for_session = settings_for_session


__all__ = ["_settings_for_session", "settings_for_session"]
