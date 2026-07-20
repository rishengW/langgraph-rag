---
name: web-search-architect
description: >
  Use this skill when changing live web discovery, provider adapters, URL
  filtering and ranking, page fetching, evidence gates, or lightweight
  web-search answer generation under src/web_search and its graph callers.
---

# Web Search Architect

Domain: live URL discovery, page fetching, evidence filtering, and grounded
answer context for chat and QA. Parent: `SKILL.md`. Graph ownership lives in
`src/graph/`; shared HTTP loading lives in `src/rag/document_loader.py`.

## Runtime Defaults

| Setting | Default |
|---|---:|
| `web_search_enabled` | `True` |
| `web_search_provider` | `bing` |
| `web_search_provider_fanout` | `2` |
| `web_search_max_results` | `20` |
| `web_search_top_k` | `6` |
| `web_search_min_url_score` | `45` |
| `web_search_provider_timeout_seconds` | `8` |
| `web_search_api_timeout_seconds` | `20` |
| `web_search_deadline_seconds` | `30` |
| `web_search_lightweight` | `True` |
| `page_load_timeout` | `15` |
| `page_load_max_concurrency` | `4` |
| `web_search_js_fallback_enabled` | `False` |

Environment and YAML configuration may override these values. LLM query
rewriting is disabled by default; deterministic query preparation remains the
normal path.

## Supported Providers

The provider IDs in this table must match
`src.web_search.factory.SUPPORTED_PROVIDER_NAMES`.

| Provider ID | Transport | Notes |
|---|---|---|
| `serper` | JSON API | Google results through Serper; API key required |
| `brave` | JSON API | Brave Search API; API key required |
| `tavily` | JSON API | Basic search without generated answers or raw content; API key required |
| `bing_api` | JSON API | Configured Bing-compatible endpoint; API key required |
| `bing` | HTML | Bing result parsing with CAPTCHA/shell detection |
| `baidu` | HTML | Baidu parsing plus bounded redirect resolution and CAPTCHA detection |
| `duckduckgo` | HTML | Bounded DuckDuckGo HTML endpoint; DDGS remains an explicit compatibility helper |

All providers implement `WebSearchProvider` and return `SearchResult` values
containing URL, title, and snippet. `RankedSearchResult` preserves provider,
provider rank, relevance score, and quality score through the graph artifact.

## File Map

| File | Responsibility |
|---|---|
| `protocol.py` | Provider protocol and ranked result value object |
| `factory.py` | Provider registry, aliases, key detection, construction |
| `api_providers.py` | Serper, Brave, Tavily, and Bing API adapters |
| `baidu.py`, `bing.py`, `duckduckgo.py` | HTML provider adapters and redirect/CAPTCHA handling |
| `discovery.py` | Provider ordering, staged fan-out, deadlines, circuits, final provider merge |
| `query_prep.py`, `query_constraints.py` | Deterministic query planning and hard-constraint preservation |
| `common.py` | URL normalization, authenticity checks, authority and relevance scoring |
| `tool.py` | LangChain `live_web_search` wrapper and ranked result artifact |
| `content_fetcher.py` | Direct page loading, extraction, readability, token limits, publication dates |
| `fetch_policy.py`, `playwright_loader.py` | Domain-aware HTTP headers and optional JS rendering |
| `date_extractor.py`, `recency.py` | Publication-date extraction and query-aware freshness assessment |
| `evidence.py`, `claim_consensus.py` | Typed answer evidence and current-status consensus |
| `prompt_builder.py` | Source-only prompt assembly with URL, title, and publication date |
| `benchmark.py` | Offline Mandarin search quality and latency evaluation |
| `__init__.py` | Public exports |

## Lightweight Chat Flow

With `web_search_lightweight=True`, chat owns discovery inside the graph. Each
turn resets stale URL state before execution.

```text
START
  -> agent
     -> direct answer -> END
     -> one live_web_search call
        -> decompose
        -> search_queries
        -> merge
        -> web_answer
           -> grounded answer -> END
           -> no grounded page -> expand -> search_queries -> merge -> web_answer
           -> second miss -> grounded refusal -> END
```

`decompose` bypasses its LLM for atomic questions and emits at most three
validated sub-questions for compound questions. `search_queries` executes at
most six queries with at most three concurrent tool calls. Each tool call enters
provider discovery, which runs at most `web_search_provider_fanout` providers in
one stage. The default peak is therefore three query calls times two provider
calls, subject to provider availability and deadlines.

The `web_search` ToolNode in the lightweight graph is the generic fallback for
mixed or multiple tool calls. A single pure live-search call uses the bounded
`decompose -> search_queries` path.

## Discovery and Pre-Fetch Filtering

1. `build_search_query` performs deterministic cleanup. Optional LLM rewriting
   is accepted only when language, quoted phrases, identifiers, and years remain
   valid.
2. Mandarin planning adds an exact query and an official-source variant, or one
   official query per year for a two-year comparison.
3. Configured APIs are preferred for Mandarin, followed by Baidu, Bing HTML,
   and DuckDuckGo unless an explicit provider priority overrides the order.
4. Providers in one stage run concurrently. The first stage with usable results
   ends provider fallback for that query.
5. Verification failures open a five-minute HTML-provider cooldown. Two
   consecutive ordinary failures open a one-minute circuit.
6. `prefetch_rejection_reason` rejects noise/search/login/download URLs,
   `site:` mismatches, missing quoted titles or identifiers, and recognized
   owner-name lookalike domains.
7. `result_quality_score` combines URL shape, source authority, language,
   title/snippet coverage, requested years, quantities, and typed evidence.
8. Results below `web_search_min_url_score` are removed. Canonical duplicates
   are merged and the provider blend is clamped to `web_search_top_k`.
9. The graph merge ranks all query sets against the original question, retains
   provider metadata, rewards cross-query overlap, and prefers domain diversity.

## Page Fetch and Evidence Gates

The page path is shallow: it fetches selected result URLs only and never follows
links recursively.

1. HTTP pages load concurrently through `load_source_documents`, preserving URL
   order and skipping individual failures.
2. Fetch policy may add browser request headers. When explicitly enabled, known
   JS domains retry low-text HTTP results with Playwright; force-JS domains skip
   HTTP. Do not add anti-bot bypass behavior.
3. Extraction removes scripts, navigation, headers, footers, forms, and sidebars.
   It prefers JSON-LD and article/main containers, then falls back to body text.
4. Normal pages must satisfy configured character or token thresholds. Concise
   official pages have a stricter relevance-based exception.
5. `FetchedPage.publication_date` is populated from loader metadata or page
   markup. Published/created dates win; modification dates are fallback only.
6. Page relevance requires lexical/entity coverage and, when requested, the
   correct year, quantity, date, or price evidence. Title and the first 1,200
   characters receive most of the weight.
7. For one explicit query year, a known conflicting publication year is
   rejected. Missing dates stay neutral. Current/latest questions rank recent
   pages first; ordinary questions preserve provider order.
8. Near-duplicate content is removed. Multi-year questions need independent
   evidence for every requested year.
9. Current-status claims require one authoritative source or agreement from two
   independent domains. Conflicts become explicit prompt constraints.
10. Only admitted pages enter the prompt. If no evidence survives, the LLM is
    skipped; after one expanded search pass, the graph returns a refusal.

## Testing

Use focused tests before the broader suite:

```text
tests/test_web_search_providers.py
tests/test_search_api_providers.py
tests/test_web_search_discovery_blending.py
tests/test_web_search_lightweight_primitives.py
tests/test_web_search_relevance.py
tests/test_web_search_evidence.py
tests/test_web_search_claim_consensus.py
tests/test_web_search_recency.py
tests/test_lightweight_graph_web_answer_integration.py
```

Network-backed provider behavior should be tested with captured responses or
mock sessions. Never require paid API keys for the deterministic unit suite.

## Known Limitations

- HTML provider layouts and anti-bot pages can change without notice.
- A successful early provider stage prevents later stages from contributing
  additional recall for that query.
- Publication dates are missing or mislabeled on many valid pages, so undated
  pages must remain eligible.
- Publication year is only a safe hard constraint for a single explicit-year
  query; multi-year comparisons rely on page-text evidence instead.
- Search filtering cannot recover relevant pages a provider never returned.
