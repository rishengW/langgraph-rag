---
name: web-search-architect
description: >
  Use this skill whenever working on web search URL discovery — Baidu HTML scraping,
  DuckDuckGo search, result normalization, noise filtering, redirect resolution,
  or the WebSearchProvider Protocol abstraction. Covers the full query→search→
  normalize→filter→rank pipeline that discovers source URLs for indexing. Trigger
  on mentions of web_search, Baidu, DuckDuckGo, ddgs, discover_urls_from_web,
  URL discovery, search provider, or URL filtering.
---

# Web Search Architect — src/web_search/

Domain: web search URL discovery, provider implementations, URL normalization and filtering.
Parent: `SKILL.md` (root). Siblings: `src/graph/SKILL.md`, `src/rag/SKILL.md`, `src/api/SKILL.md`, `src/sessions/SKILL.md`.

## Quick Reference

| Fact | Value |
|---|---|
| Search providers | Baidu (HTML scraping), DuckDuckGo (ddgs library + HTML fallback) |
| Config key | `web_search_provider` (default: "baidu") |
| Max results | `web_search_max_results` (default: 20) |
| Top-K after filtering | `web_search_top_k` (default: 3) |
| Region | `web_search_region` (default: "wt-wt") |
| Time limit | `web_search_timelimit` (optional: "d", "w", "m") |
| SSL verify | `web_search_verify_ssl` (default: True) |
| Page fetch fallback | HTTP first; optional Playwright fallback planned for JS-shell domains |
| Source file (current) | `core/web_search.py` — 363 LOC |
| Target location | `src/web_search/` |

## File Map (Current)

```
src/core/
+-- web_search.py           # discover_urls_from_web() + Baidu/DDG providers + URL filtering — 363 LOC
```

### Detailed File Responsibilities

| Function / Class | Responsibility | Method |
|------|---------------|--------|
| `discover_urls_from_web(settings)` | Top-level entry: select provider, search, normalize, filter, return top-k | Calls provider search |
| `settings_for_discovered_urls(settings, discovered)` | Return new Settings object with discovered URLs as source_urls | -- |
| `BaiduWebSearch` | Baidu HTML scraping via `requests.get()`, parse result links | Regex extraction from HTML |
| `DuckDuckGoWebSearch` | DDGS library (`ddgs.text()`) + HTML fallback (`_duckduckgo_search_html()`) | Library + requests |
| `_normalize_urls()` | Redirect resolution + deduplication | HEAD requests |
| `_filter_urls()` | Remove PDF, video, social media, and other noise | Domain + extension blocklist |
| `_resolve_redirects()` | Follow HTTP redirects to canonical URLs | requests.head() |

## Search Pipeline

```
User Query (from settings.source_urls used as search terms)
  │
  ▼
Provider Selection (Baidu or DuckDuckGo)
  │
  ▼
Raw Results (URLs + snippets)
  │
  ▼
Redirect Resolution (_resolve_redirects)
  │
  ▼
Deduplication (canonical URL set)
  │
  ▼
Noise Filtering (_filter_urls)
  │   Remove: PDFs, videos (.mp4, .avi), social media (facebook, twitter, instagram),
  │   login pages, archive sites, and other non-text content
  │
  ▼
Top-K Selection (web_search_top_k)
  │
  ▼
Return list[str] of clean URLs
```

## Provider Comparison

| Provider | Implementation | Method | Strengths | Weaknesses |
|----------|---------------|--------|-----------|------------|
| Baidu | HTML scraping of search results | `_baidu_search()` via `requests.get()` | Better Chinese-language results, no API key needed | Fragile to HTML structure changes, rate limiting |
| DuckDuckGo | `ddgs` library + HTML fallback | `_duckduckgo_search()` via `ddgs.text()` + `_duckduckgo_search_html()` | Privacy-focused, more stable API | Fewer Chinese results, library version sensitivity |

### Inconsistency Note

```
Baidu:   raw HTTP → parse HTML → extract URLs
DDGS:    library call → structured results → extract URLs

Target (Phase 2): Both behind a common WebSearchProvider Protocol.
```

## Key Design Pattern

```python
def discover_urls_from_web(settings: Settings) -> list[str]:
    # 1. Select provider based on settings.web_search_provider
    # 2. Search with query = settings.source_urls (used as search terms)
    # 3. Normalize and filter URLs
    # 4. Return top web_search_top_k results
```

## Known Issues

| # | Issue | Severity | Location | Fix |
|---|-------|----------|----------|-----|
| 1 | Baidu HTML scraping is fragile — page structure changes break extraction | High | `_baidu_search()` | Add integration test with captured HTML snapshot |
| 2 | Inconsistent provider patterns — DDGS uses library, Baidu uses raw HTTP | Medium | Both providers | Unify behind `WebSearchProvider` Protocol (Phase 2) |
| 3 | URL normalization is complex (redirect resolution, noise filtering, dedup) | Medium | Multiple functions | Add comprehensive unit tests |
| 4 | No retry logic on search failures — single attempt per provider | Medium | `discover_urls_from_web()` | Add retry with fallback to alternate provider |
| 5 | `web_search_region` default "wt-wt" is undocumented | Low | config defaults | Document region code semantics |
| 6 | `WebBaseLoader` cannot render JS or browser-gated pages | High | `content_fetcher.py`, `playwright_loader.py` | Implemented: optional Playwright fallback after low-text extraction |
| 7 | No domain-specific fetch policy for known JS/anti-bot hosts | Medium | `fetch_policy.py`, `content_fetcher.py` | Implemented: configured fallback/force-JS domains |

## Refactoring Target — src/web_search/

```
src/web_search/
+-- __init__.py
+-- protocol.py            # WebSearchProvider Protocol
+-- baidu.py               # BaiduWebSearch implementation
+-- duckduckgo.py          # DuckDuckGoWebSearch implementation
+-- factory.py             # Provider selection + instantiation
+-- content_fetcher.py     # Lightweight page fetch/extract/prompt preparation
+-- fetch_policy.py        # Domain-aware HTTP/JS page-fetch policy
+-- playwright_loader.py   # Optional JS-capable page loader adapter
```

### Protocol Definition (Phase 2)

```python
@runtime_checkable
class WebSearchProvider(Protocol):
    def search(self, query: str, max_results: int = 20) -> list[str]: ...
    @property
    def provider_name(self) -> str: ...
```

## JS Fallback / Domain Fetch Policy Target

Problem 3A/3E is solved with an optional browser-backed fetch path, not by
weakening the readability guard. Keep `WebBaseLoader` as the fast default and use
Playwright only when policy says it is worth the cost.

### Config Shape

```python
web_search_js_fallback_enabled: bool = False
web_search_js_fallback_domains: list[str] = [
    "baike.baidu.com",
    "zhuanlan.zhihu.com",
    "apps.microsoft.com",
    "deepseek.net",
]
web_search_js_force_domains: list[str] = []
```

### Fetch Policy Shape

```python
@dataclass(frozen=True)
class FetchPolicy:
    force_js: bool = False
    retry_js_on_low_text: bool = True
    request_headers: dict[str, str] = field(default_factory=dict)
```

### Execution Flow

1. Resolve the URL host through `fetch_policy.py`.
2. Use `WebBaseLoader` first unless `force_js=True`.
3. Extract text and check `is_readable_text(...)` with configured thresholds.
4. If text is below threshold and `retry_js_on_low_text=True`, retry once with
   `playwright_loader.py`.
5. If the JS result is still below threshold or Playwright fails, keep the current
   grounded refusal path. Do not pass shell text to the LLM.

### Implementation Notes

- Keep Playwright optional and disabled by default; document `python -m playwright
  install chromium` for users who enable it.
- Normalize metadata from the JS loader to match `WebBaseLoader`: `source`, `url`,
  and `title`.
- Use tight per-page timeouts and log whether each page used HTTP, JS fallback, or
  force-JS.
- Do not add a broad anti-bot bypass. The goal is rendering legitimate pages when
  operators opt in, not evading access controls.

## Refactoring To-Do List

> Source: [`REFACTORING_PLAN.md`](../../REFACTORING_PLAN.md). Web search scope items.

### Phase 1 — Extract Without Behavioral Change

- [ ] **1.3 Replace `print()` with `logging`** in web_search module
- [ ] **1.2 Extract networking utilities** — SSL config, timeout helpers → `src/utils/networking.py`

### Phase 2 — Interfaces & Abstractions

- [ ] **2.1 Define `WebSearchProvider` Protocol** — `src/web_search/protocol.py`
  - [ ] `search(query, max_results)` → `list[str]`
  - [ ] `provider_name` property
- [ ] **2.1 Implement Protocol classes**
  - [ ] `BaiduWebSearch` → `src/web_search/baidu.py`
  - [ ] `DuckDuckGoWebSearch` → `src/web_search/duckduckgo.py`
  - [ ] Verify: `isinstance(provider, WebSearchProvider)` passes for both
- [ ] **2.1 Provider factory** — `src/web_search/factory.py` with `get_search_provider(name, config)` selector
- [ ] **2.2 DI integration** — `discover_urls_from_web()` accepts `WebSearchProvider`, not reads from Settings directly

### Phase 3 — Optimizations

- [ ] **3.4 Retry with provider fallback** — if Baidu fails, try DDGS automatically
- [ ] **3.4 Search result caching** — cache results per query for TTL to reduce external calls
- [ ] **3.5 Deprecation shim** — `src/core/web_search.py` → re-exports from `src/web_search/`

- [x] **3.5a JS-capable page fetch fallback** -- optional Playwright loader after low-text HTTP extraction
- [x] **3.5b Domain fetch policy** -- `src/web_search/fetch_policy.py` for fallback/force-JS hosts and request profiles

## Testing Strategy

| Test | Approach | Phase |
|------|----------|-------|
| Baidu HTML parsing | Capture real HTML response; verify URL extraction | Phase 1 |
| DDGS library mock | Mock `ddgs.text()` return; verify integration | Phase 1 |
| URL normalization | Redirect resolution, dedup, noise filtering edge cases | Phase 1 |
| Provider fallback | Mock Baidu failure; verify DDGS kicks in | Phase 2 |
| Protocol conformance | `isinstance(provider, WebSearchProvider)` | Phase 2 |
| Low-text JS fallback | Mock HTTP shell text, verify JS loader is retried once | Phase 3 |
| Force-JS domains | Configured domain skips HTTP and uses JS loader first | Phase 3 |
| JS fallback failure | Browser loader failure still returns grounded refusal | Phase 3 |

## Dependencies

- `src/config/` — WebSearchConfig (provider, max_results, top_k, region, timelimit, verify_ssl)
- `src/utils/networking.py` — SSL config, redirect resolution helpers
- External: `requests`, `beautifulsoup4`, `ddgs`
- Optional external: `playwright` (+ Chromium browser install), `html2text` if using
  `AsyncChromiumLoader` + `Html2TextTransformer`
