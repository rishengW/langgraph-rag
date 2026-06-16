# REFACTOR: Optional Playwright-backed loader adapter for JS-rendered pages.
from __future__ import annotations

from contextlib import suppress

from langchain_core.documents import Document


class PlaywrightLoaderUnavailableError(RuntimeError):
    """Raised when the optional browser loader cannot be constructed."""


class PlaywrightPageLoader:
    """Minimal loader-compatible wrapper around Playwright Chromium."""

    def __init__(self, url: str, page_timeout: int) -> None:
        self.url = url
        self.timeout_ms = max(1, int(page_timeout)) * 1000

    def load(self) -> list[Document]:
        """Render one URL and return the browser DOM HTML as a Document."""

        try:
            from playwright.sync_api import TimeoutError, sync_playwright
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise PlaywrightLoaderUnavailableError(_PLAYWRIGHT_SETUP_MESSAGE) from exc

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                response = page.goto(
                    self.url,
                    wait_until="load",
                    timeout=self.timeout_ms,
                )
                with suppress(TimeoutError):
                    page.wait_for_load_state("networkidle", timeout=3000)
                metadata = {
                    "source": self.url,
                    "url": self.url,
                    "title": page.title(),
                    "status_code": response.status if response else None,
                }
                return [Document(page_content=page.content(), metadata=metadata)]
            finally:
                browser.close()


_PLAYWRIGHT_SETUP_MESSAGE = (
    "Playwright web-search fallback requires the optional playwright package "
    "and browser runtime. Install it and run "
    "`python -m playwright install chromium` before enabling "
    "WEB_SEARCH_JS_FALLBACK_ENABLED."
)


def playwright_loader_factory(url: str, page_timeout: int) -> PlaywrightPageLoader:
    """Build a loader-compatible Playwright renderer for one URL.

    The import is intentionally lazy so normal web search, tests, and server
    startup do not require the optional Playwright runtime or browser install.
    """

    return PlaywrightPageLoader(url, page_timeout)


__all__ = [
    "PlaywrightLoaderUnavailableError",
    "PlaywrightPageLoader",
    "playwright_loader_factory",
]
