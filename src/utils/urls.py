from __future__ import annotations


def parse_url_input(raw_urls: str | list[str] | None) -> list[str] | None:
    """Normalize API/CLI URL input to a clean list or ``None``.

    ``None``, an empty string, an empty list, and CSV strings containing only
    whitespace all mean "no URLs supplied".
    """

    if raw_urls is None:
        return None

    if isinstance(raw_urls, str):
        candidates = raw_urls.split(",")
    else:
        candidates = []
        for raw_url in raw_urls:
            candidates.extend(str(raw_url).split(","))

    urls = [url.strip() for url in candidates if url and url.strip()]
    return urls or None

