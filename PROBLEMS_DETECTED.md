# Problems Detected — Web Search Performance & Answer Quality

> **Status legend:** ✅ Fixed · 🟡 Partially fixed / mitigated · 🔁 Bypassed in the
> primary code path (still applies elsewhere) · ❌ Not fixed.
> Status lines below reflect the current state of the codebase, not the original report.

## Problem 1: Web Search Is Too Slow (Takes Minutes)

### Root Cause: Sequential, Synchronous URL Loading With Per-URL Timeouts

The pipeline from web search to answer has multiple sequential bottlenecks:

#### 1A. URLs Are Loaded One-At-A-Time, Not In Parallel

**Status:** ✅ **Fixed.** `src/rag/document_loader.py` now loads URLs concurrently via `_load_url_documents_batch` using a `ThreadPoolExecutor`, with concurrency capped by `Settings.page_load_max_concurrency` (default `4`). Failed URLs are isolated per-future and skipped instead of aborting the batch.

**Original finding (kept for history):**

**File:** `src/rag/document_loader.py:38-76` (`load_source_documents`)

The function iterates through URLs in a plain `for` loop:

```python
for url in source_urls:
    try:
        loader = loader_factory(url, page_timeout)
        docs_nested.append(list(loader.load()))
    except Exception as exc:
        ...
```

Each URL is fetched and parsed **serially**. If `web_search_top_k` is 6 (per `.env.example`) or 8 (per `Settings` default), and each URL takes 10–15 seconds, the loading phase alone takes **60–120 seconds**. There is no concurrency (no `asyncio`, no `ThreadPoolExecutor`, no `concurrent.futures`).

#### 1B. Per-URL Timeout Compounds The Delay

**Status:** 🟡 **Mitigated, not directly fixed.** `page_load_timeout` is still `15s` in `src/config/settings.py`. The compounding effect is neutralized by 1A's parallelism — with `page_load_max_concurrency=4` and `web_search_top_k=6`, the worst case is now `ceil(6/4) × 15s ≈ 30s` instead of `90s`. The per-URL ceiling itself is unchanged.

**Original finding (kept for history):**

**File:** `src/config/settings.py:48` — `page_load_timeout: int = 15`

Each slow or unreachable URL blocks the pipeline for up to 15 seconds. With 6–8 URLs, worst-case wall-clock time is `8 × 15 = 120 seconds` just for HTTP fetching.

#### 1C. No HTTP-Level Caching Of Fetched Content

**Status:** ✅ **Fixed (opt-in).** A `SourceDocumentCache` (`src/rag/document_loader.py`) with per-(url, timeout) entries, TTL eviction, and per-key load locks is now wired into `_load_url_documents`. Activation is controlled by `Settings.page_load_cache_ttl_seconds` (default `0` = disabled). Set it to a non-zero value (e.g. 300) to actually benefit; the plumbing is in place.

**Original finding (kept for history):**

Every query triggers a **fresh** download of every URL, even if the same URL was fetched moments earlier for a similar question. There is no `requests_cache` or disk-based cache layer.

#### 1D. Full Embedding Pass On Every Rebuild

**Status:** 🔁 **Bypassed for the web-search path; still applies for explicit-URL Chroma rebuilds.** `Settings.web_search_lightweight` defaults to `True`, and `src/qa/api.py` routes web-search queries through `build_lightweight_graph` + `fetch_pages` (no Chroma, no embedding) when URLs come from `discover_urls_from_web`. The heavy "rebuild Chroma → re-embed all chunks" path in `src/rag/chroma_retriever.py` is still used when the user supplies explicit URLs that differ from `settings.source_urls`, or when `web_search_lightweight=False`.

**Original finding (kept for history):**

**File:** `src/rag/chroma_retriever.py:271-284`

When `rebuild=True` (which is always the case for web-search discoveries — see `src/qa/api.py:247`), all document chunks are re-embedded via the DashScope API. For large web pages, this can produce dozens of chunks, each requiring an API call. With `dashscope_request_timeout=120` and `dashscope_max_retries=3`, a single failing embedding call can block for 6+ minutes before giving up.

#### 1E. No Lazy/Paginated Content Loading

**Status:** ✅ **Fixed for the lightweight web-search path.** `src/web_search/content_fetcher.py::fetch_pages` calls `truncate_to_token_budget` per page using `Settings.web_search_max_page_tokens` (default `8000`) so each fetched page is bounded before reaching the prompt. Whole-page download still happens at the HTTP layer (no range requests), but downstream memory and prompt-token usage are now capped. This does **not** apply to the heavy Chroma rebuild path, which still chunks the full page.

**Original finding (kept for history):**

The entire web page HTML is loaded into memory and chunked before any answer can be generated. There's no option to:
- Fetch only the first N KB of each page
- Stream content progressively
- Use a lightweight text-extraction mode instead of full BeautifulSoup parsing

#### 1F. Chroma Rebuild Is Unnecessarily Heavy

**Status:** 🔁 **Bypassed for the web-search path** (same fix as 1D). When `web_search_lightweight=True` (default), web-search queries never touch Chroma — `build_lightweight_graph` is used and the Chroma "clear → load → split → embed → persist" cycle is skipped entirely. Still applies when explicit URLs differ from configured source URLs.

**Original finding (kept for history):**

**File:** `src/rag/chroma_retriever.py:242-287`

Every web search triggers a full Chroma rebuild: clear old store → load all URLs → split all docs → embed all chunks → persist. For a single question, most of this work is wasted since the index is used once and discarded.

---

## Problem 2: Incorrect Answers Due To Poor Content Ranking

### Root Cause: Search-Engine Ranking Is The Only Quality Signal

The pipeline has no content-quality gate between "URLs from search engine" and "content in vector store."

#### 2A. No Content Relevance Or Quality Scoring Before Embedding

**Status:** 🟡 **Partially fixed.** Two layers were added:
- **URL-level:** `src/web_search/common.py` now has `url_quality_score`, `MIN_USABLE_URL_SCORE=45`, path/extension/leaf blocklists, search-query URL detection, and `canonical_url_key` deduplication. `select_top_urls` calls `ranked_usable_urls`, which drops anything scoring below the threshold.
- **Document-level (Chroma path only):** `src/rag/document_quality.py::filter_quality_documents` (length, signal ratio, repeated-line ratio, boilerplate term ratio, unique-term count, optional query overlap) is wired into `load_and_split_documents` for the Chroma path.

The lightweight web-search path (`src/web_search/content_fetcher.py::fetch_pages` → `build_web_search_prompt`) does **not** apply the full `filter_quality_documents` scoring used by the Chroma path. It now applies a readability threshold before prompt assembly (see Problem 3), so bare-shell HTML no longer slips through as near-empty excerpts.

**Original finding (kept for history):**

**File:** `src/web_search/common.py:86-92` (`select_top_urls`)

```python
def select_top_urls(urls: list[str], top_k: int | None) -> list[str]:
    filtered = [url for url in urls if not is_noise_url(url)]
    if top_k and top_k > 0:
        return filtered[:top_k]
    return filtered
```

This function only:
1. Drops a hardcoded list of noise hostnames (baidu image/video/map, bing internal pages)
2. Takes the top-K URLs in the search engine's ranked order

It does **not**:
- Fetch and score page content for relevance to the query
- Check if the page actually contains substantive text vs. navigation boilerplate
- Detect paywalls, login walls, or JavaScript-only pages before spending time fetching them
- Deduplicate pages that are mirrors or near-duplicates

If the search engine ranks an SEO-optimized but content-poor page at position #1, it goes straight into the vector store.

#### 2B. No Re-Ranking After Retrieval

**Status:** ✅ **Fixed (lexical, not neural).** `src/graph/nodes/common.py` now defines `rerank_retrieved_context`, which is called from both `grade_documents_factory` and `generate_factory`. It splits retrieved chunks (preferring `Document` artifacts, falling back to blank-line splits) and reranks them by lexical relevance: token-presence + token-frequency + 3× n-gram phrase hits (2- and 3-grams). It is **not** a cross-encoder, so its impact on keyword-stuffed adversarial content is limited, but the original gap (no re-ranking at all) is closed.

**Original finding (kept for history):**

**File:** `src/graph/nodes/common.py:157-221` (`grade_documents_factory`)

After Chroma retrieves the top-k vector-similarity chunks, the grader only performs a **binary yes/no relevance check**. There is no re-ranking step (e.g., cross-encoder) between retrieval and generation. The order of retrieved chunks is purely based on Chroma's cosine similarity, which can be gamed by keyword-stuffed content.

#### 2C. Basic Content Filtering Only — Misses Many Content Farms

**Status:** ✅ **Fixed at the URL layer.** `src/web_search/common.py::is_noise_url` now also rejects URLs whose path segments fall in `NOISE_PATH_SEGMENTS` (account/auth/login/search/tag/etc.), whose leaves match `LOW_VALUE_PATH_LEAFS` (about/contact/feed/privacy/rss/sitemap/terms), whose extension is in `LOW_VALUE_FILE_EXTENSIONS` (binary/media/archive/document types), or that look like search-of-search URLs via `has_search_query`. `url_quality_score` then rewards `CONTENT_PATH_CUES` (article/blog/news/post/release/etc.) and HTTPS, penalizing rootless / query-string-heavy URLs. Domain-list-based content-farm filtering (e.g. SEO-spam hosts) is still out of scope by design.

**Original finding (kept for history):**

**File:** `src/web_search/common.py:57-83`

The `NOISE_HOSTNAMES` set and `is_noise_url()` filter only catch:
- A few Baidu subdomains (image, tieba, zhidao, fanyi, map, video, wenku, passport, login)
- Bing's own search result pages
- Baidu `/s` search-of-search paths

It does **not** filter:
- Generic content farms / SEO spam domains
- Pages that are mostly ads
- Pages blocked by robots.txt that will fail after timeout
- Pages with very low text-to-HTML ratio

#### 2D. No Timeliness/Recency Bias For Bing

**Status:** ✅ **Fixed.** `src/web_search/bing.py` defines `BING_TIMELIMIT_FILTERS` (mapping shared `web_search_timelimit` values `d`/`day`, `w`/`week`, `m`/`month` to Bing's `ex1:"ezN"` HTML filters) plus `bing_timelimit_filter` and a `timelimit` field on `BingWebSearch`. `build_bing_search_url` injects the `filters=` query parameter when the value is set. `Settings.web_search_timelimit` (and `default.yaml`/`.env.example`) expose the knob.

**Original finding (kept for history):**

**File:** `src/web_search/bing.py:60-104`

Bing's `search()` does not accept or pass a time-range filter. DuckDuckGo supports `timelimit` via its settings, but Bing — the default provider — has no equivalent. For questions where recency matters (e.g., "latest version of X"), old stale pages can rank above fresh ones.

#### 2E. Search Provider Fallback Can Surface Low-Quality Results

**Status:** ✅ **Fixed at the "no usable URLs" boundary.** `src/web_search/discovery.py::_discover_with_provider` now returns an empty list when `select_top_urls` filters everything out (logged: "returned no usable URLs after filtering; falling back when another provider is available"). The outer `discover_urls_from_web` loop iterates `_fallback_provider_names`, so an empty post-filter result from Bing now triggers Baidu / DuckDuckGo. Quality-based fallback when results pass the URL filter but are still semantically poor (no on-topic content) remains out of scope.

**Original finding (kept for history):**

**File:** `src/web_search/discovery.py:28-34, 53-81`

The fallback chain is `bing → baidu → duckduckgo`. If Bing returns results but they're low-quality (not captcha/verification), there's no quality check to fall back to DuckDuckGo. The code only skips a provider on `BaiduVerificationError` or `BingVerificationError` — not on poor result quality.

#### 2F. `web_search_top_k` Configuration Drift

**Status:** ✅ **Fixed.** All three sources now agree on `6`:

| Source | Value |
|---|---|
| `src/config/settings.py::Settings.web_search_top_k` | `6` |
| `config/default.yaml` (`web_search_top_k`) | `6` |
| `.env.example` (`WEB_SEARCH_TOP_K`) | `6` |

**Original finding (kept for history):**

Three different defaults exist for `web_search_top_k`:
| Source | Value |
|---|---|
| `src/config/settings.py:44` | `8` |
| `config/default.yaml:30` | `3` |
| `.env.example:46` | `6` |

Depending on which config path is used, the number of URLs fetched varies 3×, changing both latency and result diversity without the user realizing it.

---

## Summary

Verified against the current codebase. Status legend: ✅ Fixed · 🟡 Partial / mitigated · 🔁 Bypassed (still applies in non-default path) · ❌ Not fixed.

| # | Problem | Severity | Impact | Status |
|---|---|---|---|---|
| 1A | Sequential URL loading (no parallelism) | High | 60–120s added latency | ✅ Fixed (ThreadPoolExecutor, `page_load_max_concurrency=4`) |
| 1B | Per-URL 15s timeout compounds delay | High | Worst-case 120s+ for 8 URLs | 🟡 Mitigated by 1A; per-URL ceiling unchanged |
| 1C | No HTTP caching | Medium | Repeated queries re-fetch same pages | ✅ Fixed (opt-in via `page_load_cache_ttl_seconds`, default 0) |
| 1D | Full re-embedding on every search | High | 30s–6min per rebuild | 🔁 Bypassed when `web_search_lightweight=True` (default); still applies for explicit-URL Chroma rebuilds |
| 1E | No lazy/partial content loading | Medium | Full page always loaded | ✅ Fixed for lightweight path (`web_search_max_page_tokens` truncation) |
| 1F | Full Chroma rebuild for single-use index | Medium | Wasted I/O + compute | 🔁 Bypassed when `web_search_lightweight=True` (default) |
| 2A | No content quality scoring before indexing | High | Spam/SEO pages get embedded | 🟡 URL-level scoring + Chroma-path doc filter added; lightweight path now has a readability gate |
| 2B | No re-ranking after retrieval | Medium | Vector similarity is the only rank | ✅ Fixed (lexical reranker `rerank_retrieved_context`) |
| 2C | Noise filter catches only ~15 domains | Medium | Most content farms pass through | ✅ Fixed (path/leaf/extension/search-query filters + content-cue scoring) |
| 2D | No recency filter for Bing | Low | Stale pages may outrank fresh ones | ✅ Fixed (`BING_TIMELIMIT_FILTERS` + `web_search_timelimit`) |
| 2E | No quality-based provider fallback | Low | Low-quality Bing results never trigger DDG | ✅ Fixed at the "no usable URLs" boundary; semantic-quality fallback still out of scope |
| 2F | `web_search_top_k` inconsistent across configs | Low | Unpredictable behavior | ✅ Fixed (all three sources = `6`) |
| 3A | `WebBaseLoader` cannot render JS / handle anti‑bot pages | High | Major Chinese tech sources yield empty bodies | ✅ Fixed as opt-in JS fallback (`WEB_SEARCH_JS_FALLBACK_ENABLED`) |
| 3B | Extraction has nothing to extract from JS shells | High | Symptom of 3A | 🟡 Mitigated by readability threshold/refusal |
| 3C | "Readable page" guard accepts trivial text | High | Existing safety branch never fires on JS‑rendered failures | ✅ Fixed (`web_search_min_page_chars` / `web_search_min_page_tokens`) |
| 3D | No per‑page extraction‑size logging | Medium | Cannot distinguish discovery miss from extraction miss | ✅ Fixed (`fetch_pages` logs extraction and prompt sizes) |
| 3E | No domain‑specific loader policy / fallback | Medium | Same loader applied to news sites and SPAs alike | ✅ Fixed (`fetch_policy.py`, fallback/force-JS domains) |

---

## Problem 3: "No Content In Sources" Answers Despite Valid URLs

### Symptom

For queries like "DeepSeek's latest models in June 2026", the model returns answers of the form:

> A Baidu Baike entry on "DeepSeek" (https://baike.baidu.com/item/DeepSeek/65368136), but the content of that page is not included in the retrieved snippet — only the URL and title are given... A Zhihu column (...) — again, only the URL is provided; no excerpted text...

The URLs in the answer are real, on‑topic, and reachable. The model then concludes "the provided web sources do not contain information," which is a hallucinated framing of an extraction failure as a content failure.

### Root Cause: Content‑Extraction Failure On JS‑Rendered / Bot‑Gated Pages

This is **not** a discovery problem and **not** a URL‑validity problem. Discovery returns valid URLs and the URL‑quality filter keeps them. The failure is in the **fetch + extract** step: the loader returns a page shell with no readable article body, and the downstream guard is too lenient to catch it.

#### 3A. `WebBaseLoader` Cannot Render JavaScript

**Status:** ✅ **Fixed as an opt-in fallback.** The fast default remains `WebBaseLoader`, but `src/web_search/content_fetcher.py` can now retry low-text known JS domains through `src/web_search/playwright_loader.py` when `WEB_SEARCH_JS_FALLBACK_ENABLED=true`. Browser automation remains disabled by default and requires Playwright plus a Chromium install.

**File:** `src/rag/document_loader.py:115-119`

```python
def default_loader_factory(url: str, page_timeout: int) -> WebBaseLoader:
    return WebBaseLoader(
        url,
        requests_kwargs={"timeout": page_timeout},
    )
```

`WebBaseLoader` is a plain `requests` + BeautifulSoup loader. It does not execute JavaScript and does not present as a real browser. The pages most commonly surfaced for Chinese tech queries are exactly the ones this loader cannot render:

| Domain | Failure mode |
|---|---|
| `baike.baidu.com` | Body content hydrated by JS; raw HTML is mostly a shell. |
| `zhuanlan.zhihu.com` | Anti‑bot interstitial returned to non‑browser clients; the article never loads. |
| `apps.microsoft.com` | JS‑rendered SPA; raw HTML is `<title>` + bootstrapper. |
| `deepseek.net` | Marketing SPA; minimal server‑rendered text. |

For these URLs `loader.load()` succeeds (HTTP 200), so the existing retry/error paths never fire — but the resulting `Document.page_content` carries no real body text.

#### 3B. `extract_text` Cannot Recover Text That Isn't In The HTML

**Status:** 🟡 **Mitigated.** `src/web_search/content_fetcher.py::fetch_pages` now records extracted character/token counts and treats text below the configured readability threshold as unusable prompt context. This does not make JS-only HTML extractable; it prevents the failure from being framed as source content.

**File:** `src/web_search/content_fetcher.py:80-101`

```python
def extract_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for element in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]):
        element.decompose()
    candidates = [soup.find("article"), soup.find("main"), soup.find(attrs={"role": "main"}), soup.body, soup]
    for candidate in candidates:
        ...
```

When the HTML body is a JS shell, the `article`/`main`/`body` candidates contain at most a few tokens of nav/title/loading text. Extraction is doing the right thing; there is just nothing to extract.

#### 3C. The "Readable Page" Guard Is Too Lenient

**Status:** ✅ **Fixed.** `src/graph/nodes/web_answer.py` now uses `is_readable_text` with `Settings.web_search_min_page_chars` (default `200`) and `Settings.web_search_min_page_tokens` (default `50`) before building a prompt. `fetch_pages` applies the same thresholds before returning page text.

**File:** `src/graph/nodes/web_answer.py:54-58`

```python
readable_pages = [page for page in pages if (page.text or "").strip()]
if not readable_pages:
    ...
    return {"messages": [AIMessage(content="I couldn't retrieve readable content...")]}
```

A page with 30 characters of nav text passes this filter. The existing "I couldn't retrieve readable content" safety branch is never taken on JS‑rendered failures, so the prompt builder cites the URL with a near‑empty excerpt:

```
--- Source: https://baike.baidu.com/item/DeepSeek/65368136 (Title: DeepSeek_百度百科) ---
<a few nav-ish tokens>
--- End Source ---
```

The LLM dutifully reports back exactly that — "URL and title given but no excerpt" — which is the symptom the user observed.

#### 3D. No Per‑Page Length Visibility In Logs

**Status:** ✅ **Fixed.** `src/web_search/content_fetcher.py::fetch_pages` now logs one INFO line per URL with extracted chars/tokens, prompt chars/tokens, and the page error.

**File:** `src/web_search/content_fetcher.py:33-77` (`fetch_pages`)

There is no log line recording `len(page.text)` (or token estimate) per URL. From the outside, a session that failed because of empty extraction looks identical to a session that failed because no URLs were discovered. This blocks operational diagnosis of the difference between "discovery miss" and "fetch/extract miss."

#### 3E. No Domain‑Specific Fetch Profile / Blacklist For Known Anti‑Bot Hosts

**Status:** ✅ **Fixed.** `src/web_search/fetch_policy.py` now resolves per-URL policy for known and configured JS-heavy domains. The policy can add browser-like HTTP headers, retry low-text pages through the JS loader, or force JS first for configured hosts via `WEB_SEARCH_JS_FORCE_DOMAINS`.

**File:** `src/web_search/common.py` (URL filtering) and `src/rag/document_loader.py` (loader factory)

Domains known to either gate non‑browser clients (zhihu) or require JS hydration (baidu baike, apps.microsoft.com, marketing SPAs) now receive a domain-specific fetch policy when JS fallback is enabled. Plain server-rendered news pages still use normal HTTP loading.

### Suggested Fixes (For The Tool That Fixes This)

The fixes below are listed in increasing order of effort/blast radius. Pick the smallest set that resolves the symptom; do not over‑engineer.

1. **Tighten the readability gate.** In `src/graph/nodes/web_answer.py`, change the `readable_pages` filter from "any non‑empty text" to a minimum useful length after `normalize_whitespace`, e.g. ≥ 200 characters or ≥ 50 estimated tokens. This alone correctly routes the current failure into the existing "couldn't retrieve readable content" branch instead of producing a confidently empty answer.
2. **Log per‑page extraction size.** In `src/web_search/content_fetcher.py::fetch_pages`, emit one INFO line per URL with `url`, `len(page.text)`, `estimate_tokens(page.text)`, and `page.error`. This makes "discovery miss" vs. "fetch/extract miss" trivially separable in logs.
3. **Add a JS‑capable fetch fallback.** Introduce a second loader factory backed by Playwright (`langchain_community.document_loaders.PlaywrightURLLoader` or `AsyncChromiumLoader` + `Html2TextTransformer`). Strategy: try `WebBaseLoader` first; if extracted text falls below the readability threshold, retry that URL once with the JS loader. Apply unconditionally for a small allowlist of known SPA / anti‑bot hosts (`zhuanlan.zhihu.com`, `baike.baidu.com`, `apps.microsoft.com`, plus a configurable list).
4. **Domain‑aware request profile for `WebBaseLoader`.** Where Playwright is unavailable, at least set a realistic `User-Agent`, `Accept-Language`, and reasonable `Referer` via `requests_kwargs` for known anti‑bot hosts. This is a partial mitigation only — it will not solve JS hydration.
5. **Optional: extend `is_noise_url` / quality filter** to deprioritize (not drop) hosts that have repeatedly produced empty extractions in a session, so the top‑K is not crowded out by URLs that the loader is known to fail on.

### Diagnosis Summary

| Step | Status |
|---|---|
| URL discovery (`discover_urls_from_web`) | ✅ Working — returns valid, on‑topic URLs |
| URL quality / liveness filter (`select_top_urls`, `is_noise_url`) | ✅ Working — URLs are reachable |
| HTTP fetch (`WebBaseLoader`) | ⚠️ Can still return JS-shell HTML, but policy domains can retry/force a JS loader |
| Text extraction (`extract_text`) | 🟡 Still produces near-empty text for JS shells; `fetch_pages` now retries JS when policy allows and otherwise marks it below threshold |
| Readability guard (`web_answer.readable_pages`) | ✅ Uses configurable character/token thresholds instead of accepting any non-empty text |
| Prompt assembly (`build_web_search_prompt`) | ✅ Working — now receives only readable pages from the lightweight path |
| LLM answer | ✅ No longer called when every fetched page is empty/trivial; the grounded refusal is returned instead |

| # | Problem | Severity | Impact |
|---|---|---|---|
| 3A | `WebBaseLoader` cannot render JS / handle anti‑bot pages | **High** | Fixed as opt-in Playwright fallback for configured domains |
| 3B | Extraction has nothing to extract from JS shells | **High** | Mitigated by thresholding/refusal; JS fallback can recover when enabled |
| 3C | "Readable page" guard accepts trivial text | **High** | Fixed by configurable readability thresholds |
| 3D | No per‑page extraction‑size logging | **Medium** | Fixed by per-URL extraction/prompt size logs |
| 3E | No domain‑specific loader policy / fallback | **Medium** | Fixed with `FetchPolicy`, fallback domains, force-JS domains, and browser-like HTTP headers |
