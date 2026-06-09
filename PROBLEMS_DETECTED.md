# Problems Detected — Web Search Performance & Answer Quality

## Problem 1: Web Search Is Too Slow (Takes Minutes)

### Root Cause: Sequential, Synchronous URL Loading With Per-URL Timeouts

The pipeline from web search to answer has multiple sequential bottlenecks:

#### 1A. URLs Are Loaded One-At-A-Time, Not In Parallel

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

**File:** `src/config/settings.py:48` — `page_load_timeout: int = 15`

Each slow or unreachable URL blocks the pipeline for up to 15 seconds. With 6–8 URLs, worst-case wall-clock time is `8 × 15 = 120 seconds` just for HTTP fetching.

#### 1C. No HTTP-Level Caching Of Fetched Content

Every query triggers a **fresh** download of every URL, even if the same URL was fetched moments earlier for a similar question. There is no `requests_cache` or disk-based cache layer.

#### 1D. Full Embedding Pass On Every Rebuild

**File:** `src/rag/chroma_retriever.py:271-284`

When `rebuild=True` (which is always the case for web-search discoveries — see `src/qa/api.py:247`), all document chunks are re-embedded via the DashScope API. For large web pages, this can produce dozens of chunks, each requiring an API call. With `dashscope_request_timeout=120` and `dashscope_max_retries=3`, a single failing embedding call can block for 6+ minutes before giving up.

#### 1E. No Lazy/Paginated Content Loading

The entire web page HTML is loaded into memory and chunked before any answer can be generated. There's no option to:
- Fetch only the first N KB of each page
- Stream content progressively
- Use a lightweight text-extraction mode instead of full BeautifulSoup parsing

#### 1F. Chroma Rebuild Is Unnecessarily Heavy

**File:** `src/rag/chroma_retriever.py:242-287`

Every web search triggers a full Chroma rebuild: clear old store → load all URLs → split all docs → embed all chunks → persist. For a single question, most of this work is wasted since the index is used once and discarded.

---

## Problem 2: Incorrect Answers Due To Poor Content Ranking

### Root Cause: Search-Engine Ranking Is The Only Quality Signal

The pipeline has no content-quality gate between "URLs from search engine" and "content in vector store."

#### 2A. No Content Relevance Or Quality Scoring Before Embedding

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

**File:** `src/graph/nodes/common.py:157-221` (`grade_documents_factory`)

After Chroma retrieves the top-k vector-similarity chunks, the grader only performs a **binary yes/no relevance check**. There is no re-ranking step (e.g., cross-encoder) between retrieval and generation. The order of retrieved chunks is purely based on Chroma's cosine similarity, which can be gamed by keyword-stuffed content.

#### 2C. Basic Content Filtering Only — Misses Many Content Farms

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

**File:** `src/web_search/bing.py:60-104`

Bing's `search()` does not accept or pass a time-range filter. DuckDuckGo supports `timelimit` via its settings, but Bing — the default provider — has no equivalent. For questions where recency matters (e.g., "latest version of X"), old stale pages can rank above fresh ones.

#### 2E. Search Provider Fallback Can Surface Low-Quality Results

**File:** `src/web_search/discovery.py:28-34, 53-81`

The fallback chain is `bing → baidu → duckduckgo`. If Bing returns results but they're low-quality (not captcha/verification), there's no quality check to fall back to DuckDuckGo. The code only skips a provider on `BaiduVerificationError` or `BingVerificationError` — not on poor result quality.

#### 2F. `web_search_top_k` Configuration Drift

Three different defaults exist for `web_search_top_k`:
| Source | Value |
|---|---|
| `src/config/settings.py:44` | `8` |
| `config/default.yaml:30` | `3` |
| `.env.example:46` | `6` |

Depending on which config path is used, the number of URLs fetched varies 3×, changing both latency and result diversity without the user realizing it.

---

## Summary

| # | Problem | Severity | Impact |
|---|---|---|---|
| 1A | Sequential URL loading (no parallelism) | **High** | 60–120s added latency |
| 1B | Per-URL 15s timeout compounds delay | **High** | Worst-case 120s+ for 8 URLs |
| 1C | No HTTP caching | **Medium** | Repeated queries re-fetch same pages |
| 1D | Full re-embedding on every search | **High** | 30s–6min per rebuild |
| 1E | No lazy/partial content loading | **Medium** | Full page always loaded |
| 1F | Full Chroma rebuild for single-use index | **Medium** | Wasted I/O + compute |
| 2A | No content quality scoring before indexing | **High** | Spam/SEO pages get embedded |
| 2B | No re-ranking after retrieval | **Medium** | Vector similarity is the only rank |
| 2C | Noise filter catches only ~15 domains | **Medium** | Most content farms pass through |
| 2D | No recency filter for Bing | **Low** | Stale pages may outrank fresh ones |
| 2E | No quality-based provider fallback | **Low** | Low-quality Bing results never trigger DDG |
| 2F | `web_search_top_k` inconsistent across configs | **Low** | Unpredictable behavior |
