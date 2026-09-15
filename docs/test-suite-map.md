# Test Suite Map

Every file in `tests/` read and catalogued. 80 `test_*.py` files + `conftest.py` (81 files total, ~1.1 MB of Python), split into 8 thematic groups.

**Read method:** each group was read in full (not sampled) by a dedicated reader; shared infrastructure and the cross-cutting claims below were re-verified directly against the working tree.

**Read on:** branch `production`, 2026-09-14.

---

## 0. Environment status — the suite cannot run here

This is the single most important thing in this document. `tests/` is well-structured but the local venv is missing declared dev dependencies, so **nothing collects**:

```
$ python -c "import langchain_core"
ModuleNotFoundError: No module named 'langchain_core'

$ python -m pytest --version
__main__.py: error: unrecognized arguments: --timeout=120
```

- `tests/conftest.py:7` does `from langchain_core.messages import AIMessage` — a hard import at collection time, not inside a test. Every test file dies before the first assertion runs.
- `pyproject.toml:44-48` sets `addopts = ["-q", "--tb=short", "--timeout=120"]`. The `--timeout` flag requires the `pytest-timeout` plugin, which is also absent.
- Both are declared in `requirements-dev.txt` (`pytest>=8.0,<9`, `pytest-cov>=5.0,<7`, `pytest-timeout>=2.3,<3`, `hypothesis>=6.100,<7`). So this is an **uninstalled-environment problem, not a test-suite problem** — `pip install -r requirements-dev.txt` should clear both.
- Consequence for this document: no statement below is backed by a test run. Everything is static analysis — imports resolved against `src/`, assertions read off the source. **Line numbers and names are verifiable; pass/fail is not.**

---

## 1. Shared infrastructure — `tests/conftest.py` (49 lines)

Tiny and deliberately minimal. Three fixtures, one hook:

| Symbol | Kind | What it does |
|---|---|---|
| `pytest_collection_modifyitems` | hook | Auto-marks any item whose filename stem contains `integration` with `pytest.mark.integration`. No decorator upkeep. 3 files qualify today. |
| `mock_settings(tmp_path)` | fixture | `Settings(dashscope_api_key="test-key", chroma_dir=tmp_path/"chroma", source_urls=["https://example.com/a"], dashscope_max_retries=1)` |
| `isolated_settings(mock_settings)` | fixture | Returns a factory wrapping `dataclasses.replace(mock_settings, **kwargs)` — the workhorse, used across ~20 files. |
| `ai_message` | fixture | Returns a factory producing `AIMessage(content=...)`. |

**Side effects:** none. No autouse fixtures, no env mutation, no global state. Every fixture is function-scoped and takes `tmp_path`, so isolation is per-test by construction.

**Notable absence:** there is no LLM fake, no fetch fake, no session fixture in conftest. Each group builds its own doubles locally — which is good for independence but means the same `_FakeModel` / `RecordingCheckpointer` / `FakeGraph` shapes are re-derived several times (see §10.6).

---

## Group 1 — Web search & fetch (17 files)

The largest cluster and the most duplicated. Covers the whole retrieval-then-rank pipeline under `src/backend/web_search/` (24 submodules) plus the tool and utility seams around it.

Counts below are `def test` definitions, verified by grep — two independent readers both reported different (wrong) figures, so these are the authoritative ones. Parametrization multiplies collected instances.

| File | `def test` | Source module under test | What it asserts |
|---|---|---|---|
| test_web_search_providers.py | **67** | `web_search.factory`, `baidu`, `bing`, `duckduckgo`, `discovery`, `query_prep`, `common` | The largest file (1,397 lines). Provider protocol conformance; factory aliases (`"ddg"` → DuckDuckGo, default Bing, market `en-US`, timelimit `w`, `verify_ssl=False`, timeout 7); `bing_timelimit_filter` d/w/M → ez1/ez2/ez3 and y/latest/None → None; URL builder edge cases (`count=1` for `max_results=0`); Baidu/Bing verification errors on captcha bodies; bounded timeouts; concurrent Baidu redirect resolution (`max_active_redirects == 2`); Bing base64 `/ck/a?u=` unwrapping; DDGS kwargs `{"verify": False, "timeout": 5}`; `discover_urls` injection, `min_url_score=80`, alternate-provider fallback, Chinese blend, `RankedSearchResult` metadata; verification cooldown across two calls; disabled provider → zero calls; `build_web_search_tool` named `live_web_search`; `query_prep` filler stripping, 2026 append, named-entity preservation, metro clarification, relative-year handling with "fiscal year" preserved; `rewrite_search_query_llm` skipped without an API key; `build_search_query` LLM-off by default; `url_quality_score`, `text_relevance_delta`, snippet-aware `result_quality_score`, `select_top_results`; MCP generic-term hardening and Jordan/Argentina brand collision both pushing junk below `DEFAULT_MIN_USABLE_URL_SCORE`; `_term_weight("cup") == 0.2` vs `_term_weight("argentina") == 1.0`; CJK bigrams; current-year tournament page outranking the evergreen winners list |
| test_web_search_lightweight_primitives.py | **17** | `content_fetcher.fetch_pages`, `extract_text`, `fetch_policy.resolve_fetch_policy`, `is_readable_text`, `prompt_builder.build_web_search_prompt` | `fetch_pages` forwards `page_load_timeout` (3.8 → 3), `max_concurrent_loads`, `page_load_cache_ttl_seconds`; exact log line format `Fetched web page content: url=… extracted_chars=19 extracted_tokens=5 prompt_chars=19 prompt_tokens=5`; loader failure → `"network unavailable"` pages; `extract_text` prefers `main` and strips header/footer; JSON-LD `articleBody` beats a loading shell; sub-threshold text cleared to `""`; JS retry → `["http","js"]` / `"js_fallback"`; force-JS domains skip HTTP entirely; JS error text rejected; HTTP error kept as `"http+js_failed"`; `resolve_fetch_policy` subdomain matching and zh-CN Accept-Language; `is_readable_text` boundaries at 197/200 chars; prompt ends `Answer:`, 70-token source budget, embeds `EVIDENCE CONSTRAINT` |
| test_web_search_relevance.py | **15** | `web_search.common` | Owner-lookalike `deepseek-seek.com.cn` penalised by exactly `OWNER_LOOKALIKE_PENALTY` yet still eligible; `deepseek-ai.github.io` and `xiaomiev.com` usable outside the allowlist; `.pdf` and `/download/` paths not noise; `/category/` listings demoted not rejected; login/account/search/assets hard-rejected; `prefetch_rejection_reason` → `site_constraint_mismatch`, `missing_quoted_title`, `missing_identifier`; api-docs outranking a matching secondary; unresolved baidu redirects and doorway `.php` rejected; year match > absent > conflicting; concrete count answers rewarded; 欧几里得几何 does **not** trigger the count requirement; six negative pages; `page_relevance_score` ranks h1/lead above a buried match |
| test_web_search_authority_readability.py | **9** | `web_search.common`, `content_fetcher.fetch_pages` | `host_authority_class` → government / education / recognized_owner / standard, suffix-safe (so `gov.cn.example.test` and `example.edu` stay standard); `host_quality_score` > 0 for gov/edu/deepseek, == 0 for example.com, < 0 for sohu/baijiahao; `result_quality_score` orders official > neutral > syndicated with syndicated still > 0; `registrable_domain` groups `jtj/www.nanjing.gov.cn` → `nanjing.gov.cn`, `127.0.0.1` for localhost; `is_readable_page` accepts a < 200-char official answer with year+count evidence and rejects missing year/count/syndicated/off-topic; `fetch_pages` keeps the short official page when `relevance_query` is set and retries an unlisted `.gov.cn` host, `fetch_method == "js_fallback"` |
| test_web_search_discovery_blending.py | **10** | `web_search.discovery.discover_search_results_from_web` | Configured API pair preferred (`{serper, brave}`, baidu/bing uncalled); two consecutive `TimeoutError` opens the circuit (`_provider_cooldown_remaining("serper") > 0`); Chinese blend hits baidu+bing first with query rewritten to `… 运营线路总数` at `max_results=20`; duckduckgo only after the primary pair misses; `provider_stage_timeout=0.05` lets the fallback land in < 0.4s; `build_search_query` runs once and both providers get the same string; `_fallback_provider_names("duckduckgo", …) == ["bing","baidu","duckduckgo"]`; single-provider stage must not construct a `ThreadPoolExecutor`; thin first-stage URL still triggers fallbacks; `_stage_result_threshold` 0→2, 1→1, 6→2 |
| test_web_search_page_structure.py | **11** | `web_search.page_structure` + `content_fetcher.fetch_pages` | CJK-aware `word_count` (南京地铁 == 4); article shape with `link_density < 0.5` and no rejection; listing rejected as `"listing_page"` with `link_density >= 0.5`; "enable JavaScript" shell → `"gateway_page"`; `min_content_words=60` short body → `"thin_page"`; empty page abstains (measured `False`, still content); 12 citation-heavy paragraphs stay `"article"`; `fetch_pages` attaches structure; JS retry fires once outside the domain list, honours `js_retry_budget=1` preferring the listed host, and is skipped when `js_fallback_enabled=False` |
| test_web_search_pdf_loader.py | **8** | `web_search.pdf_loader`, `content_fetcher`, `rag.document_loader` | Builds real PDFs byte-by-byte with a valid xref. `is_pdf_url` accepts `.PDF` with a query, rejects `pdf-guide`; `extract_pdf_text` returns 1 page and the text layer; `PdfPageLoader` returns one Document with `fetch_method="pdf"` after exactly one session call; textless scan raises `ValueError("no extractable text")`; `fetch_pages` keeps PDF text with no HTML extraction; `looks_like_pdf_payload` detects `%PDF` after leading whitespace; extensionless arxiv URL recovered by payload sniffing; `default_loader_factory("…pdf")` returns `PdfAwareLoader`. `pytest.importorskip("pypdf")` — pypdf is installed, so this never silently skips |
| test_web_search_evidence.py | **8** | `web_search.evidence`, `web_search.common` | `detect_query_intents` → price; date+policy; comparison+status; date questions need a concrete date (`2026年8月1日` passes, `近期施行` fails, `answer_evidence_delta` > 0 / < 0); price needs a currency amount near **every** product identifier (小米 SU7 Ultra vs 小米手环 499元 fails); quoted policy title must exist on the date page; policy/comparison are soft ranking signals; `is_page_text_relevant` and `text_relevance_delta` require the date/price evidence |
| test_web_search_recency.py | **8** | `date_extractor`, `recency`, `graph.nodes.web_answer._rank_pages_by_publication_date`, `factory` | Prefers `datePublished` 2026-06-12 over `modified` 2026-07-18 then falls back; `assess_publication_date` neutral for a non-temporal query (applies `False`, score 0) and for an undated current query (applies `True`, score 0); newer pages score higher for "latest release"; a single explicit year flags only genuine conflicts; multi-year queries stay neutral at 0; `_rank_pages_by_publication_date` orders recent/undated/old and returns conflicting 2026 URLs for a "2025" query; **SKILL.md provider table must match `factory.SUPPORTED_PROVIDER_NAMES` exactly** (7 entries, currently matching — so that doc is not stale) |
| test_web_search_reputation.py | **7** | `web_search.reputation`, `graph.nodes.merge.merge_factory` | `DomainReputationStore` aggregates per registrable domain (example.com → 3 attempts / 2 grounded / 1 rejected, `other.example.org` kept separate, unseen → `None`); `reputation_delta` is 0 below `min_samples=5` and for `None`, `+MAX_REPUTATION_BONUS` / `-MAX_REPUTATION_PENALTY`, 0 for a 5/5 split; stats survive reopening the SQLite file; `build_reputation_store` returns `None` when disabled, else a path under `"web-search"`; `merge_factory` promotes a learned-good domain over a junk domain with identical `relevance_score=20` / `quality_score=80`; ranking unchanged when reputation is off |
| test_web_search_semantic.py | **5** | `web_search.semantic` + `discovery` | `cosine_similarity` returns 1.0 / 0.0 and 0.0 for empty or zero-norm vectors; `semantic_bonus` is 0 at and below the 0.35 threshold, `SEMANTIC_MAX_BONUS` at 1.0, strictly between at 0.6; `build_semantic_scorer` is `None` by default; a `StubScorer` rescues an English query against a Mandarin snippet (`without_semantics == []` vs one URL, `scorer.calls` non-empty); scoring skipped when the flag is off |
| test_search_api_providers.py | **3** (1 parametrize) | `web_search.api_providers` + `factory` | Parametrised over Serper (POST, `organic.link/title/snippet`), Brave (GET, `web.results`), Tavily (POST, `results`), BingApi (GET, `webPages.value`) — each normalised to one `SearchResult` with `provider._session.calls[0][0] == method` and timeout 8.0; `SerperWebSearch(api_key="")` raises `ValueError("requires an API key")`; `get_search_provider` builds all four, `"bing_api"` → `BingApiWebSearch`, Tavily timeout 20.0 |
| test_summarize_url_tool.py | **4** | `backend.tools`, `tools.summarize_tool` | `ftp://` → `"unsupported url"`; a readable page is fetched, summarised, and returned as `"Summary of Example Article"` with focus and page text both verified inside the captured prompt; unreadable page → `"could not read content"`; `build_summarize_url_tool(...).name == "summarize_url"` |
| test_web_search_claim_consensus.py | **3** | `web_search.claim_consensus.assess_status_consensus` | One official page → `"supported"`, `categorical_allowed=True`; secondaries need two independent domains (single → `"insufficient"`, `categorical_allowed=False`, `"unverified"` in `prompt_instruction`); two contradicting secondaries → `"conflicting"` with `"conflict"` in `prompt_instruction` |
| test_web_search_query_constraints.py | **3** | `web_search.query_constraints` + `query_prep` | `extract_query_constraints` on `小米 SU7 Ultra 2026 款价格` → language `"mixed"`, identifiers `{su7, ultra}`, 小米 in entities, years `{2026}`, `PRICE_INTENT` set; `validate_query_candidate` rejects an English translation and an invented 2026 while accepting a same-language partial; `plan_search_queries` for 《南京市住房租赁管理办法》 yields 2 queries, the second carrying the quoted title, `site:gov.cn` and `生效日期`, no `COMPARISON_INTENT` |
| test_web_fetch_events.py | **2** | `graph.builder`, `graph.events`, `graph.executor`, `graph.nodes.web_answer` | `WebFetchEvent` URLs from merged `source_urls` arrive as `["https://a.test/one","https://b.test/two"]` through `GraphExecutor.astream(stream_tokens=True)` with `total == 2` and `node == "web_answer"`, followed by a done event carrying `"grounded answer"`; a direct `web_answer_factory` call without a stream context does not raise on announcement. Only test in the suite building a module via `ModuleType` + `monkeypatch.setitem(sys.modules, …)` |
| test_urls.py | **2** | `src.utils.urls.parse_url_input` | Returns `None` for `None`, `""`, `" , "` and `[]`; CSV strings and nested lists split and strip to `["https://a.test","https://b.test","https://c.test"]` |
| | **172** | | **17 files, 172 test functions** |

**Zero live network in the entire group.** HTTP is faked at module level (`monkeypatch.setattr(baidu_module, "urlopen", ...)`, `bing_module.urlopen`, `duckduckgo_module.load_ddgs`, injectable `FakeSession`/`FakeResponse`). API keys are hardcoded fake strings passed into `Settings` — never read from the environment, so there is no CI env-variable failure mode. The only non-faked filesystem dependency is the SKILL.md drift check.

`test_web_search_providers.py` carries a local autouse fixture clearing `discovery._provider_cooldowns` / `_provider_failure_counts`; the PDF and reputation tests use `tmp_path` for their SQLite DBs.

---

## Group 2 — Query planning & Mandarin search benchmark (4 files)

Mandarin query rewriting/expansion, bounded search fan-out, and an offline search-quality evaluator. All offline: search comes from an injected `discover` callback, LLM calls are monkeypatched.

| File | Source module under test | What it asserts | Live net / LLM? |
|---|---|---|---|
| test_bounded_search_queries.py | `graph.builder.build_lightweight_graph`, `graph.nodes.search_queries.search_queries_factory` (`WEB_SEARCH_MAX_QUERIES = 6`), `tools.live_web_search`, `web_search.protocol.RankedSearchResult` | 8 raw queries dedupe + clamp to 6; `tool_call_id` preserved; all `RankedSearchResult` fields round-tripped and duplicated across 2 planned variants; 4 Mandarin queries → exactly 6 variants; per-sub-question fan-out; expand-retry clamps 7 to `WEB_SEARCH_MAX_QUERIES-1` with 2 answer attempts; original Mandarin kept, English translations dropped | No |
| test_mandarin_query_planning.py | `graph.nodes.decompose.decompose_factory`, `web_search.query_prep` | Strips `请帮我查一下` / `请问` wrappers and appends intent hint `发布时间`; adds `官方` variant; splits 2025/2026 into two year-scoped queries each ending `官方 数据`; non-Mandarin passthrough; output bounded to 2 variants; compound questions make exactly 1 LLM call; atomic connectors (`与` `以及` `和`) short-circuit with **no** LLM call | No |
| test_mandarin_search_benchmark.py | `web_search.benchmark` | On 2 fixtures: P@3=0.5, Official@3=0.4, Answerable@3=0.5, Irrelevant=0.4, latency mean/median=200, p95=300, min=100, max=300, group `bing/exact`; `ValueError` on missing judgments unless `allow_unjudged=True`; empty capture rejected; JSONL normalization (`BING→bing`, `EXACT→exact`, `:443` and trailing slash stripped); manifest must have ≥15 queries, full category coverage, CJK in every question, unique ids; CLI `--format json` returns 0 | No |
| test_provider_publication.py | `src.backend.mcp` + `adapters.mcp_client` | *Off-topic for this group* — MCP catalog lifecycle: atomic required-provider failure retaining the prior generation, optional failure degrading readiness, collision quarantining, snapshot-lease retention of retired generations, bounded cleanup timeout, config→registration binding, plus a Hypothesis property over Req 8.1–8.6 | No |

**Benchmark wiring.** `DEFAULT_QUERY_SET` resolves at import time in `src/backend/web_search/benchmark.py:14-16` as `Path(__file__).resolve().parents[2] / "benchmarks" / "mandarin_search" / "queries.json"` — relative to the **module file**, not cwd. The file holds 18 queries with 4 fields each (`query_id`, `category`, `question`, `as_of_date`), 3 per category across all 6 `ALLOWED_CATEGORIES`, all Simplified Chinese.

**Gating — and a stale skip reason.** The only gate is `@pytest.mark.skipif(not DEFAULT_QUERY_SET.exists(), ...)` at `test_mandarin_search_benchmark.py:69-72`: a filesystem check, not a marker or env var. No `integration`/`slow` decorators, no env var read anywhere, no `pytestmark`. **The skipif's stated reason is wrong** — it claims `benchmarks/` is gitignored, but `.gitignore:24-27` tracks `queries.json` and excludes only `runs.jsonl`, `judgments.jsonl` and `runs/`. So in a real CI checkout `.exists()` is `True` and the test runs. The practical risk is inverted: in a sparse checkout the only structural guarantee on the dataset silently disappears as a *skip*, not a failure.

**CI** (`.github/workflows/ci.yml:32,41`, Python 3.11): `python -m pytest --tb=short --cov=src --cov-report=term --cov-fail-under=70`, no `-m` filter, on push to `main`/`production` and on PR. All of group 2 runs.

---

## Group 3 — File & language tool suites (13 files)

Every language/format tool family has its own `<lang>_edit.py` + `<lang>_file.py` pair under `src/backend/tools/`, and public tools are never merged into shared modules. **The convention holds.**

| File | Source module | Operations covered | Third-party lib |
|---|---|---|---|
| test_file_tools.py | `src/backend/tools` package root (read-only readers) | `read_text_file`, `read_word_document`, `read_excel_spreadsheet`, `read_pdf` + ~18 parametrized code/file readers | openpyxl (hard) |
| test_typescript_edit.py | `tools.typescript_edit` | `create_` / `edit_` / `inspect_typescript_file` — the reference suite, 13 tests | — |
| test_text_edit.py | `tools.text_edit` | `create_` / `edit_` / `inspect_text_file` | — |
| test_markdown_edit.py | `tools.markdown_edit` | `create_` / `edit_` / `inspect_markdown_file` | — |
| test_json_edit.py | `tools.json_edit` | same 3 + JSON-value edit semantics | — |
| test_yaml_edit.py | `tools.yaml_edit` | same 3 + multi-document guards | PyYAML (hard) |
| test_csv_edit.py | `tools.csv_edit` | row/cell operations (structurally different from the line-based family) | — |
| test_excel_create.py | `tools.excel_create` | `create_excel_spreadsheet` (Node `@oai/artifact-tool`) | openpyxl + Node runtime |
| test_excel_edit.py | `tools.excel_edit` | `edit_excel`, `inspect_excel` — deepest coverage of the group | openpyxl (hard) |
| test_word_edit.py | `tools.word_edit` | `create_` / `edit_` / `inspect_word_document` | python-docx (`importorskip`) |
| test_powerpoint_edit.py | `tools.powerpoint_edit` | `edit_powerpoint`, `inspect_powerpoint` — thinnest | python-pptx (`importorskip`) |
| test_zip_file.py | `tools.zip_file` | `read_zip_entry`, `inspect_zip_file` only | — |
| test_code_edit.py | `build_*_edit_tools` across 16 code langs + `jsonl_edit` | `create_` / `inspect_` / `edit_<lang>_file` × 16 | — |

**Sandbox.** `conftest.py` provides no file fixtures; every file test builds its own scope under pytest `tmp_path` — `file_root = tmp_path/"files"`, `session_root = tmp_path/"files"/"chat_uploads"/"thread-a"` — passed in as `file_read_root`. Fully isolated per test. All suites test path-traversal and cross-session denial. One exception: `test_excel_create.py:15-24` resolves `node_modules` relative to CWD and shells out to `node`, skipping if the `@oai/artifact-tool` runtime is absent.

**Convention compliance.** Pairs exist for: c, go, groovy, haskell, html, java, javascript, json, julia, latex, log, lua, markdown, matlab, php, prolog, python, r, ruby, rust, shell, sql, swift, text, typescript, yaml. Document family: `excel_{create,edit,file}`, `word_{edit,file}`, `pdf_file`. Deviations from the pair pattern: `powerpoint_edit.py` has **no** `powerpoint_file.py` (inspect bundled into the edit module); `csv_edit.py` has **no** `csv_file.py` (no CSV read tool); `jsonl_edit.py` has no file reader; `zip_file.py` has no edit counterpart. `test_file_tools.py:9-35` and `test_code_edit.py:7-21` import builders from the **package root** `src.backend.tools`, so they pin the `__all__` publish list; dedicated suites import from the module path.

---

## Group 4 — Memory subsystem (12 files)

Long-term memory is a JSON document store (`long_term_memory.json`, written atomically via mkstemp + `os.replace`) behind four layers: a pure store, agent-facing tools, automatic recall injection, and background extraction from transcripts. **All 12 files are fully deterministic** — no file makes a model call or needs an API key; the extractor is always handed a `model_factory` returning a canned stub.

| File | Source module | What it asserts |
|---|---|---|
| test_memory_extraction.py | `memory.extraction` | Turn count from checkpointer; `should_extract` fires exactly at turns 10/20/30 with interval 10; candidate parser tolerates JSON prefix/suffix, dedupes case-insensitively, caps at `max_candidates`, never evaluates entry 101; unusable response advances watermark while writing nothing; one store refusal skips only that candidate; model failure **holds** watermark at 0 |
| test_memory_extraction_properties.py | `memory.extraction` | 8-way (case → status, watermark) matrix; timeout gives status `failed` / detail `model_timeout` and discards the late response; `run()` never raises for any of four failure seams; watermark above turn count clamped to 1 and repaired on session start; session scope binds to the extracted thread; exactly one content-free log record per outcome |
| test_memory_extraction_integration.py | `frontend.chat.memory_hooks`, `memory.scheduler`, `memory.watermark` | `build_extraction_runtime` returns `None` without touching the checkpointer when either flag is off; `SessionWatermarkStore` prefers persisted 7 over session attribute 2 and writes both; `on_session_start` picks most-recent with greatest-id tiebreak, filters deleted/zero-turn/current, marks an over-age winner done without replacement, schedules 1 of 100 on adoption; HTTP returns **before** a blocked worker finishes; a hook raising `ZeroDivisionError` still returns 200; failed turns never call `after_turn` and mask the real error; CLI shuts down on exit and on `KeyboardInterrupt` |
| test_memory_integration.py | `graph.builder`, `llm.prompts`, `memory.recall` | Both heavy and lightweight resolvers expose save/recall/forget when `memory_enabled` and omit them when disabled; enabling memory adds exactly those 3 tool names; prompt names every memory tool and says REMEMBER and RECALL; memory note precedes upload note precedes `HumanMessage`; injection is **read-only** (file bytes and `last_recalled_at` unchanged); session memory injected only for its own thread; history serialization drops the note |
| test_memory_mutations.py | `memory.store` | `save` trims and returns a 32-char hex id, casefolds categories, dedupes tags; duplicate save updates only tags + `updated_at`, keyed on category+scope; validation order content→category→scope→tags; limit checks run **before** secret screening; credential-looking content refused without echoing the secret; LRU eviction prefers never-recalled, spans scopes, ties on ascending id; `recall` stamps `last_recalled_at` only; `forget` by id is idempotent and deletes at most one; `forget` by query caps at `MAX_FORGET_DELETES`; `purge_session` removes one thread only; **8 threads × 20 saves lose nothing** |
| test_memory_store.py | `memory.serialization`, `memory.store` | 12 hand-enumerated documents round-trip field by field with zero warnings and byte-identical re-serialization, including a document at the 500-record cap; missing/unusable required field skips only that record and names its position; `resolve_store_path` refuses `..` traversal; corrupt file quarantined as `.corrupt-*` that never collides; atomic writes retry `os.replace` 4 times with 0.05/0.10/0.20 backoff then raise `MemoryWriteError` leaving target and temp untouched; concurrent readers never see a partial document |
| test_memory_tools.py | `tools.memory_tool` | Pydantic inputs reject blank/over-long content, over-long query, >10 tags; the three tools are `save_memory`, `recall_memory`, `forget_memory`; no tool exposes `config`, `thread_id`, `path`, `file` or `directory` in its args; `thread_id_from_config` trims and rejects non-dict/blank/`None`; validation failure, secret refusal, any exception including `MemoryError`, and a rejected store path all return a `MEMORY_ERROR:` string under 500 chars; a broken logger doesn't change the outcome; the **11th call in a turn is refused** without persisting; budget shared across all three tools |
| test_memory_recall_note.py | `memory.recall`, `memory.transcript` | `build_memory_note` respects `max_chars` emitting only whole records, scores against the first 500 chars, is `None` for no match/empty store/1-char terms; note building changes neither file bytes, `mtime_ns` nor `last_recalled_at`; `MemoryCallBudget` allows exactly 10 then refuses, resets per thread and turn; `build_turn_messages` orders note, upload, question |
| test_memory_transcript.py | `memory.transcript` | `message_role` maps Human/AI/System/Tool; `normalize_message_content` flattens content blocks and drops non-text; `count_turns` counts only user messages; `is_memory_note` detects the marker; `select_slice` keeps only user/assistant pairs above the watermark and skips an injected note mid-transcript; `render_slice` never exceeds its budget across six budgets and applies the 200-message cap before the character cap |
| test_memory_primitives.py | `memory.relevance`, `memory.secrets` | `normalize_content` casefolds and collapses whitespace; `derive_query_terms` drops 1-char and duplicate terms, caps at 50; `relevance_score` substring-matches CJK; `rank_records` orders by score → recency → ascending id and sorts unparseable timestamps oldest; `find_secret_match` detects 10 patterns (PEM, `sk_token`, AWS, bearer, `password=`, `API_KEY:`, `db_passwd`, `token:`) and never returns the secret |
| test_memory_scheduler.py | `memory.scheduler` | Two slots allow two submissions then refuse while both run; `wait_idle` frees the slot; workers are daemon threads and **shutdown never joins them** (verified with a thread class that raises if `join` is called); a raising runner still releases its slot |
| test_memory_watermark.py | `memory.watermark` | `coerce_watermark` accepts 0…1,000,000 and maps above-max, negatives, `True`, `False`, `"3"`, `3.0`, `None`, `{}`, `[]` to 0; survives a hostile object raising from `__int__` and `__eq__`; booleans rejected rather than read as ints; `WATERMARK_KEY == "extraction_watermark"` |

**Persistence & isolation.** JSON, not SQLite — SQLite exists only in `backend/sessions` for chat checkpoints. Watermarks are a separate seam: `InMemoryWatermarkStore` is a plain dict by default, `SessionWatermarkStore` persists them into `SessionMetadata.config["extraction_watermark"]`. Every test gets its own `tmp_path` file plus freshly built store, budget and scheduler; clocks are injected (`StepClock`) so timestamp assertions are deterministic. The one genuinely shared resource is the module-level store singleton behind `get_memory_store` — `test_memory_integration.py` guards it with an autouse `reset_store_cache` fixture, `test_memory_store.py` and `test_memory_recall_note.py` call `reset_store_cache()` inline.

**Naming trap.** `test_memory_extraction_properties.py` is **not** property-based — it uses `@pytest.mark.parametrize` over explicit matrices. `hypothesis` *is* a real project dependency (`requirements-dev.txt:12`, used in 7 other files) and `test_memory_store.py:5` claims "No hypothesis in this project" — that comment is wrong.

---

## Group 5 — Graph & RAG core (10 files)

The LangGraph topology, state schema and RAG interfaces.

| File | Source module | What it asserts |
|---|---|---|
| test_lightweight_graph_web_answer_integration.py | `graph.nodes.web_answer`, `build_lightweight_graph`, `route_after_web_answer*` | The heavyweight file: web answer grounding, content-fetch / readability / structure filtering, retry/fallback routing |
| test_conditional_expansion_nodes.py | `nodes.decompose`, `nodes.expand`, `nodes.merge` | Atomic-question bypass, clamping, dedupe, URL ranking/canonicalization, top-k, domain diversity |
| test_graph_builder.py | `graph.builder` | `build_graph`, `_resolve_tools`, `_resolve_lightweight_tools`, node/tool override injection, legacy `src.frontend.chat.graph.build_chat_graph` delegation |
| test_graph_state_and_nodes.py | `graph.state` (`RAGState`), `nodes.common`, `nodes.condense` | State channels, bounded messages, rerank, grade factory, `GRADE_PROMPT`; pins `ChatState is RAGState` |
| test_rag_interfaces.py | `src/backend/rag/*` | Embedding protocols, document loading/quality, Chroma retriever, cache, `build_retriever_tool` |
| test_graph_artifacts.py | `graph.artifacts` | AMap marker/route + file artifact normalization, bounds, rejection, extraction |
| test_planning_reflection.py | `nodes.planning` | `normalize_plan`, `planner_node`, `subgoal_dispatcher_node`, `route_subgoals`, `reflection_revise_node`, `route_after_self_critique`, plus end-to-end wiring in both graphs |
| test_graph_executor.py | `graph.executor`, `graph.metrics` | Event translation, token streaming, artifact dedupe/accumulation, `MetricsCollector` |
| test_fallback_answer.py | `nodes.fallback_answer` | Preserved-question prompt, refusal fallback on model failure |
| test_retry.py | `src/utils/retry` | *Not graph* — `call_with_retry`, `is_retryable_connection_error` |

**Topology as the tests describe it.** State is `RAGState` (TypedDict, `total=False`) with `ChatState` as an alias; sub-models `SubGoal`, `SubGoalResult`, `AnswerCritique`, `WebSearchResultMetadata`; channels include `messages` (Annotated `add_messages`), `subgoal_results` (Annotated `operator.add`), `search_queries`, `web_search_results(_metadata)`, `web_answer_attempts`, `expansion_attempted`, `answer_critique`, `reflection_retry_count`.

Edges live in `graph/edges.py`: `route_after_agent` (+ `_with_critique`), `route_after_lightweight_agent` (+ `_with_critique`), `route_after_lightweight_tool`, `route_after_web_answer` (+ `_with_fallback`), `route_after_self_critique`, `route_subgoals`, `route_after_subgoal_aggregation`. Maps: `AGENT_EDGE_MAP` (tools→retrieve), `GRADE_EDGE_MAP` (generate|rewrite), `LIGHTWEIGHT_TOOL_EDGE_MAP` (merge|agent), `WEB_ANSWER_EDGE_MAP` (expand|fallback_answer|answer_self_critique|END), `WEB_ANSWER_FALLBACK_MAX_ATTEMPTS = 2`.

- **Full graph nodes:** condense, planner, subgoal_dispatcher, execute_subgoal, subgoal_aggregator, agent, retrieve, generate, rewrite, fallback_answer, answer_self_critique, reflection_revise. Flow: START→condense→agent; agent⇄retrieve via `grade_documents`; generate→END (or →answer_self_critique→reflection_revise↔).
- **Lightweight nodes:** agent, web_search (ToolNode), execute_search_queries, decompose, expand, merge, web_answer, fallback_answer (+ planning/critique). No retriever/vector store. The agent calls the real `live_web_search` tool; `route_after_lightweight_tool` sends only `live_web_search` results to `merge→web_answer` and every other tool (weather, stock, …) back to `agent`. Adds the decompose/expand/merge fan-out retry and a tool-free `fallback_answer` terminal.

**LLM faking.** No shared fake class — three seam patterns, all `monkeypatch.setattr` on the node module: swap `new_chat_model`/`new_structured_chat_model` + `invoke_with_retry` (dominant); `RunnableLambda(lambda prompt: AIMessage(...))` as a stand-in chain; tiny local classes (`FakeModel` with only `bind_tools`, `FakeEmbeddings`, `_FakeStructured`, plus `FakeGraph`/`TokenStreamGraph`/`SummaryGraph`/`ErrorGraph`/`ArtifactGraph` in `test_graph_executor.py`). Web fetching faked by monkeypatching `sys.modules` for `content_fetcher`/`prompt_builder`.

---

## Group 6 — Chat application, HTTP API, sessions, config (7 files)

The user-facing layer and its configuration.

| File | Source module | What it asserts |
|---|---|---|
| test_chat_upload.py | `frontend.chat.uploads`, `frontend.chat.api` | Upload context notes advertise editing tools **only when the matching flag is enabled** (text, typescript, a 19-language block incl. json/jsonl/r/rust/go/sql/php/ruby/latex/prolog/haskell/lua/julia/shell/matlab/groovy/swift/log, powerpoint, excel); context injected as a `SystemMessage` into the **next** turn only, never repeated, never leaked into history; uploads land inside the file-read root and are readable by the file tool; `.docx`/`.pptx`/`.xlsx` built in-test from real bytes; `evil.exe` → `errors: ["unsupported file type"]`; upload/delete/edit/download round-trips with 404s for cross-thread and unknown-thread access; deleting a thread removes the upload dir; edit artifacts downloadable with correct MIME type; `DELETE /chat/{id}/files/{name}` returns the cumulative remaining list |
| test_sessions.py | `backend.sessions` | Registry create/get/list/delete with cleanup; `delete` removes the isolated Chroma dir; `cleanup_expired` removes only stale sessions and only their dir; `get` touches for TTL (and `touch=False` does not); `settings_for_session` shares Chroma for defaults, isolates under `chroma_dir/chat/<thread>` with a suffixed collection name for custom sources; `InMemoryStorage` copies metadata (mutation of a loaded copy doesn't leak); SQLite round-trip, update, delete, `list_ids`/`list_metadata` ordering; optional persistence with touch throttling at `TOUCH_PERSIST_INTERVAL_SECONDS`; `update_sources` replaces the graph and persists; `restore` from metadata; extraction watermark round-trips and survives a registry touch; legacy metadata defaults the watermark and backfills `config`; schema version written to a `schema_version` table; `SQLiteMemorySaver` restores graph state after reopen, persists one row per thread in `thread_state`, migrates a legacy single-`checkpoint_state` pickled blob, and serializes writes |
| test_chat_application_services.py | `backend.application` (`TurnExecutionService`, `SessionLifecycleService`) | Snapshot/restore discipline: completion snapshots and runs the hook; `ErrorEvent` followed by `DoneEvent` restores the **exact** snapshot and runs no hook; stream cancellation closes the generator, waits, and rolls back; stream preserves event order and artifacts; non-streaming failure restores the snapshot, skips `sync_sources` and `after_turn`, and sanitizes to `"Internal server error. Request ID: correlation-id"`; success calls `sync_sources` **before** `after_turn`; `SessionLifecycleService.start` builds the graph once; `/chat/{id}/history` is guarded by the global API key (401 vs 200) |
| test_config.py | `config.loader`, `config.settings` | 48 tests. Parse helpers; web-search/page-load defaults aligning with `config/default.yaml`; env-only AMap secrets with configurable timeout; `.env.example` documentation checks; per-language edit flag defaults (`word`/`powerpoint`/`excel`/`text`/`markdown`/`typescript`/`excel_create`) cross-checked against env and docs; YAML flat values and lists; **precedence CLI > env > yaml > defaults**; clamping of chat context bounds; opt-in flags for lightweight web search, LLM query rewrite, agent tools, JS policy, readability thresholds, page-load cache TTL, document quality, rerank strategy, `page_load_max_concurrency`; DeepSeek provider (requires key; defaults to DashScope); long-term-memory settings (env, boolean accepted values, out-of-range ints, non-integer ints, invalid default scope, `recall_top_k` may exceed `max_records`, YAML); extraction settings (same four validation shapes, YAML, transcript vs record char bounds are independent); and `test_no_extraction_model_setting_exists` |
| test_api_dependencies.py | `frontend.api.dependencies`, `frontend.chat.api` | Legacy model import path still resolves (`chat_api.StartChatRequest is StartChatRequest`); `initialize_chat_app_state` preserves an existing registry and graph-factory lock across re-initialization; the full chat contract — `POST /chat`, `GET /health` → `{"status": "ok"}`, `POST /chat/{id}/message` → `{thread_id, answer, error, artifacts}`, `GET /chat/{id}/history` → turns, `DELETE /chat/{id}` → `{"status": "deleted"}` and registry emptied, graph built exactly once with `rebuild_vectorstore=False`; the graph factory receives `session_root = file_read_root/chat_uploads/<thread>` and `thread_id`; web-search mode **ignores the request's `web_search` toggle** and refreshes source URLs every turn with `rebuild_vectorstore=True`; lightweight mode is graph-owned (no `discover_urls_from_web` call, graph built once) |
| test_resource_ownership_quotas.py | `backend.security`, `backend.sessions`, `frontend.chat.api` | Ownership and quotas. Registry authorizes a complete owner and never touches a foreign probe (`get_owned` by wrong principal or wrong tenant returns `None` without bumping `last_accessed_at`); v1 SQLite rows migrate as ownerless; checkpoint operations require the recorded owner and normalize every failure to the same `"Resource not found."`; each of 6 quota dimensions enforced individually; concurrency separate per principal and tenant; window expiry bounded by `max_tracked_identities`; the HTTP boundary covers session, stream, upload, artifact download and delete in one test with caller identity headers spoofed to prove they're ignored; `principal_id`/`tenant_id` in the request body → **422**; tenant request quota shared across principals → **429** with `{"detail": {"code": "QUOTA_EXCEEDED", "message": "Request quota exceeded."}}`; out-of-range quota env rejected; plus a Hypothesis ownership-confinement property (Req 4.1–4.6) |
| test_api_auth.py | `frontend.api.auth`, `frontend.api.amap_proxy.ClientRateLimiter` | `require_principal` fails closed with **503** and an "API key" detail in `production`/`staging` when no `API_KEY` is set; stays open outside protected environments (default principal `api-key-client`); with a key set, missing → **401**, `Bearer secret-key` → 200. `ClientRateLimiter` bounds requests per window, slides the window, and evicts stale clients |
| test_chat_turn_inputs.py | `frontend.chat.api._graph_inputs_for_turn` | Each turn seeds the session's current URLs into graph state, overwriting stale checkpoint values; omits `source_urls`/`source_mode` entirely when the session has none (no empty override); lightweight web turns reset all checkpointed search state (`sub_questions`, `expanded_queries`, `search_queries`, `web_search_results(_metadata)`, `web_answer_attempts`, `web_answer_no_readable_content`, `expansion_attempted`) |

**Regression guards worth knowing about.** Three tests in this group encode specific production bugs rather than features: `test_session_registry_keeps_storage_backed_instance_and_persists` (`ChatSessionRegistry.__len__` makes an empty registry falsy, so an `or`-fallback in `initialize_chat_app_state` silently replaced the storage-backed registry and lost every session on restart); `test_sqlite_memory_saver_serializes_writes_under_lock` + `test_sqlite_memory_saver_concurrent_writers_do_not_corrupt_state` (the `"dictionary changed size during iteration"` crash when two `/chat` requests hit one session — the first asserts the structural lock invariant via monkeypatched `MemorySaver.put` so it fires without depending on timing); and `test_upload_returns_all_session_files_after_sequential_uploads` (upload responses used to list only the current request, making the client's chip strip drop older chips — commit `296c5ef chips-overlap`).

---

## Group 7 — Agent tools, persona, answer sanitizer, AMap & geo (8 files)

| File | Source module | What it asserts |
|---|---|---|
| test_tool_catalog_policy.py | `backend.mcp`, `backend.mcp.catalog`, `graph.builder`/`events`/`executor`/`metrics` | Immutable catalog composition + policy pipeline. Deep immutability of `ToolDescriptor`/`ToolCatalogSnapshot`; duplicate-name rejection before publication; name validation (leading digit, spaces, reserved `mcp__` namespace); schema validation (unsupported constructs, property/enum/byte/depth caps, JSON-serializability); source/risk enum enforcement; descriptor schema must match the dispatch schema; MCP namespace must match `server_name`; atomic generational publication with the failed candidate closed and the prior generation retained; policy wrapping preserves `BaseTool` and applies redaction + char/byte bounds + audit events stamped with the generation; authorize-before-validate with non-disclosing denials; catalog × runtime allowlist **intersection**; unknown-input rejection before invocation; principal/tenant rate limits; concurrency slot retained until blocked work exits; cancellation releases limits; audit-sink failure does not bypass redaction; a Hypothesis non-disclosure/bounds property; the graph attaches one immutable snapshot to both model and dispatcher and revalidates injected snapshots |
| test_agent_tools.py | `backend.tools.*` | Behavior of the 11 non-file agent tools: weather, currency, wikipedia, stock, directions, map, math, datetime, statistics, linalg, number-theory |
| test_prompt_persona.py | `backend.llm.prompts`, `config.loader._coerce_setting` | Default prompts contain "Donald Trump"; `agent_system_prompt("")`/`rag_prompt("")` drop the persona but keep tool rules and `{current_date}`; `TRUMP_AGENT_PERSONA`/`TRUMP_RAG_PERSONA` are literal substrings of the defaults; `agent_persona_style` defaults to `"trump"`, coerces lowercase, `None` falls back, unknown values raise `ValueError` |
| test_answer_sanitizer.py | `backend.llm.sanitize`, `graph.nodes.web_answer`, `graph.executor`, `web_search.*` | Strips `【199†L91-L126】` / ASCII `[199†L91-L126]` dagger citations, bare `†L9-L9` refs, `[oaicite:0]` / `【oaicite:12】` tool residue, `citeTurn3Search2` tokens; tidies the whitespace it leaves; **preserves** URLs, markdown links, ordinary `[1]`/`[sic]`, and a lone `†`. Streaming `CitationArtifactFilter` buffers partial marker prefixes (≤160 chars), drops markers split across chunks, releases non-marker text, and **flushes** unterminated prefixes rather than swallowing real output. Wired at the `web_answer` node and `GraphExecutor.stream`; prompt-side invariant: the RAG prompt explicitly forbids inventing markers |
| test_amap.py | `backend.tools._amap`, `graph.artifacts`, `graph.events` | Lat/lng parsing, WGS84→GCJ02 call shape, envelope validation incl. v4 numeric `errcode: 0`, credential-override protection in `amap_request_json`, `safe_amap_error` redaction, route parsing/bounds/polyline sampling, URI + artifact builders, travel-mode aliases, `ArtifactEvent` identity |
| test_amap_proxy.py | `frontend.api.amap_proxy` | Server-side browser proxy. Browser-safe client config (no server key or security code leaks); disable-when-incomplete; path allowlist (traversal, double-slash, encoded traversal, leading space, wrong API version, backslashes); fixed restapi/webapi hosts; `jscode` always server-overridden; redirects off; query name/size bounds including discarded `jscode`; streamed response byte cap with forced close; `image/svg+xml` → `application/octet-stream`; stream errors wrapped as 502 without leaking detail |
| test_geocoding.py | `backend.tools._geocoding`, `backend.tools.map_tool.find_on_map` | POI-primary → `/geocode/geo` → `/config/district` fallback chain; skip fallback on a confident match; empty on blank/unconfigured/network failure; `name_match_score`; bilingual request-wording stripping; APPROXIMATE / AMBIGUOUS / empty messaging from `find_on_map` |
| test_chat_static_amap.py | `frontend/chat/static/script.js` + `index.html` | Static string assertions over the frontend bundle — no Python imports, nothing executed. No NUL bytes; math-token escaping; link tokens protected before `escapeHtml`; lazy cached `/chat/config` AMap loading; no server key names in the bundle; artifact caps/dedupe/history restore; deferred markdown/KaTeX/artifact rendering during streaming; DOM-text-only map cards with origin-pinned fallback links; file-download cards; map teardown; responsive CSS + numeric cache-bust |

**Tool catalog source of truth.** `src/backend/mcp/providers.py` — `_default_providers()` and `default_provider_tools()`, consumed by `graph/builder.py:_resolve_raw_tools`. Builtins are registered by `BuiltinToolProvider`, each gated by a `*_enabled` setting; `RetrieverToolProvider` is heavy-path only, `WebSearchToolProvider` required on the lightweight path, alongside `MemoryToolProvider`, `DocumentToolProvider` and `SessionEditingToolProvider`.

**None of the four AMap/geo files is network-dependent.** Every HTTP call goes through an injected `requester` returning canned payloads; `test_chat_static_amap.py` performs no I/O at all. Keys are split: server-side `amap_web_service_key` for the backend, browser `amap_js_api_key` + `amap_js_security_code`, and the proxy refuses to forward (503) without a server-side security code.

---

## Group 8 — MCP, security, deployment, release gate (9 files)

The security-and-operations envelope. Every file here is requirement-traceable: many tests carry a `**Validates: Requirements N.N…**` docstring.

| File | Source module | What it asserts |
|---|---|---|
| test_mcp_observability.py | `backend.mcp.observability` | Bounded observability (spec task 7.1). `MCPObservability` emits to log/metric/trace/audit; every JSON export ≤ `MAX_OBSERVATION_RECORD_BYTES`; `redaction_values` scrubbed from all four channels and from `repr()`; the metric record has a **fixed dimension set** (`schema_version`, `category`, `signal`, `outcome`, `transport`, `duration_ms`, `generation`) and provably excludes `request_id`/`principal_id`/`tenant_id`/`tool`/`source_server`; optional sink failures are isolated and counted in `snapshot().export_failures` while `require_audit_delivery=True` raises `RequiredAuditDeliveryError`; shared exporters closed exactly once; policy denial exports `authorization` + `policy_denial` outcomes with no argument leakage; Hypothesis non-disclosure property over 40 examples (Req 5.6, 7.1–7.6, 9.3) |
| test_outbound_mcp_config.py | `adapters.mcp_client` | Outbound config is a separate boundary from inbound. `load_outbound_mcp_settings({})` is a no-op; enablement **fails closed** without `OUTBOUND_MCP_CONFIG_FILE`; unknown env vars rejected; only typed `SecretReference`s accepted as authorization (raw values and `{"value": ...}` rejected, marker absent from the error); runtime resolution redacts the value from `repr()` and `to_config()`; disabled settings **never resolve credentials**; `SecretResolver.resolve()` raises `TypeError` on a non-reference; confinement property (Req 1.1, 1.4, 9.1, 9.2) |
| test_outbound_mcp_security.py | `adapters.mcp_client` | Endpoint and process security. HTTPS mandatory, arbitrary process fields rejected (`command`/`args` unknown — stdio must reference an allowlisted `command_template`); `OutboundEndpointPolicy` validates redirects, DNS and the **actually-connected peer** (`not pre-resolved`, `not allowlisted`, `limit exceeded`); private destinations require narrow host **and** CIDR approval; `ManagedOutboundMCPProvider` enforces the redirect contract with `follow_redirects=False` always, request/result byte bounds (`request exceeds`), reconnect-and-retry **only** when `retry_safe=True`, invocation and shutdown timeouts that cancel upstream work (asserting `invocation_cancelled is True`), and startup-failure cleanup that closes session and connector with no leakage; `ExecutableAllowlist` refuses stdio in hosted production unless explicitly opted in, uses fixed argv without a shell, and terminates without killing. Includes a 40-example property over prohibited IPv4/IPv6 ranges (10/8, 127/8, 169.254/16, 224/4, link-local, ULA, multicast) — Req 6.3–6.7, 9.5 |
| test_deployment_topology.py | `src.deployment.topology` | Single-instance enforcement. `" Production "` is stripped and lowercased; accepts exactly 1 worker + 1 replica; **every** worker-count env alias (`RAG_WORKER_COUNT`, `WEB_CONCURRENCY`, `UVICORN_WORKERS`) individually rejects scaled production; replica scaling names the local blockers (`local SQLite`, `local Chroma`, `process-local locks`) and requires capabilities that are "implemented, explicitly configured, and validated"; invalid counts (`0`, `not-a-number`, `-1`, `""`) fail closed with the offending var name in the error; conflicting aliases fail closed; development retains multi-process flexibility; scaled production accepts only a **complete** validated capability generation (all required validators called, in order); each missing capability rejected **before any validator runs** (`validation_calls == 0`); incomplete evidence (no implementation / not configured / no validator) rejected; a failing validator's detail is sanitized out of the error (`postgresql://admin:secret@...` absent); duplicate registrations fail closed before validation; single-instance startup **never runs** shared-state validators; FastAPI startup rejects scaled production before local resources are touched; two Hypothesis properties (Req 11.1, 11.4–11.6); and the Dockerfile + `.env.example` both declare `RAG_WORKER_COUNT=1` / `RAG_REPLICA_COUNT=1` |
| test_resource_ownership_quotas.py | `backend.security`, `backend.sessions` | *Also in Group 6.* Ownership confinement and quota enforcement across the HTTP boundary — see the Group 6 row |
| test_production_release_gate.py | `deployment.release_gate` | Release evidence validation. A valid document closes every gate (`schema_version` 1, release id, `production` environment, worker/replica counts, 8 rehearsal drills, backup/restore digests equal, `public_summary()` → `{"status": "passed", "release_id": ..., "scenario_count": 8}`). Rejects missing/duplicate/failed/stale/future drills; requires recoverable quiesced state (`writers_quiesced`, `isolated_restore`, `restore_sha256`); rejects scaled local state (`unsupported topology`); accepts **only** secret references and sanitizes rejection (raw marker absent from the error); key rotation reuses the reference without persisting the value (old and new secrets absent from `to_config()` and `repr()`); file gate + CLI emit only a bounded JSON summary — exit 0 with the image digest and volume ref absent, exit **2** on failure with `{"code": "production_release_gate_failed", ...}` and an empty stdout; and the ops runbook + `config/production-rehearsal.example.json` both cover the closed gate, including the exact key set |
| test_llm_provider.py | `backend.llm.provider` | `build_structured_chat_model` + `structured_output_method`: DeepSeek uses tool-free `json_mode` (`tool_choice` never passed), DashScope keeps the library default (`None`, no kwargs) |
| test_compat.py | `src.frontend.chat.sessions` | Reloading the legacy compatibility module emits a `DeprecationWarning` naming itself and keeps `ChatSessionRegistry` importable |

---

## 10. Cross-cutting findings

### 10.1 Environment blockers (verified)

1. `langchain_core` missing → `tests/conftest.py:7` kills collection of all 80 files.
2. `pytest-timeout` missing → `addopts` in `pyproject.toml:47` fails pytest at argument parse.
3. Both are declared in `requirements-dev.txt`. Fix: install dev requirements. **Nothing here indicates the suite itself is broken.**

### 10.2 The deleted `core/` layer is fully cleaned up

The working tree deletes the entire old layer — `src/backend/core/` (config, embeddings, graph_executor, nodes, retriever, state, web_search) and `src/backend/application/rag_service.py`. Checked across all 80 files: **zero references remain.** Every test was already repointed to the new locations (`backend.graph.*`, `backend.rag.*`, `backend.web_search.*`, `backend.sessions`, `backend.memory`, `backend.security`, `backend.application.{chat_service,turn_execution,session_lifecycle}`), and `tests/test_rag_application_service.py` was deleted in step with its subject. The only residue is stale `__pycache__/*.pyc` files, which pytest will not collect.

### 10.3 Confirmed code bug — `calculate_datetime` is misclassified as read-only

`src/backend/mcp/providers.py:445-452`, `_builtin_risk()`:

```python
if tool.name in {
    "solve_math",
    "compute_statistics",
    "linear_algebra",
    "number_theory",
    "date_time",      # <-- never matches
}:
    return "execute"
```

The real tool name is `calculate_datetime` (`src/backend/tools/datetime_tool.py:79`, re-exported at `:234`). So the datetime tool is risk-classed `"read"` instead of `"execute"`. Verified by direct inspection of both files. **No test covers risk mapping for real builtins** — `test_tool_catalog_policy.py` uses only dummy echo/capture tools, and `test_agent_tools.py:160` asserts the tool's *name* but not its risk class. This is the one substantive defect found by reading the suite: a missing test, exposing a real misconfiguration in a security-relevant field.

### 10.4 Document / metadata drift

| Where | Problem |
|---|---|
| `benchmarks/mandarin_search/README.md:46` | Documents `python -m src.web_search.benchmark`, but the module is `src.backend.web_search.benchmark`. The documented command fails. |
| `test_mandarin_search_benchmark.py:69-72` | Skipif reason says `benchmarks/` is gitignored. `.gitignore:24-27` explicitly comments "queries.json and docs stay tracked". The guard's *reason* is wrong even though its *condition* is right. |
| `test_mandarin_search_benchmark.py:76` | Asserts `>= 15` queries while the README documents 18 — the assertion is looser than the documented count, so drift is undetectable. |
| `tests/test_memory_store.py:5` | "No `hypothesis` in this project, so the domains are enumerated explicitly" — false; `hypothesis>=6.100,<7` is in `requirements-dev.txt:12` and 7 test files use `@given`. |
| `tests/test_memory_extraction_properties.py` | Filename implies property-based testing; it uses only `parametrize`. Fine, but misleading when triaging coverage. |
| `tests/__pycache__/` | `test_rag_application_service.cpython-311.pyc` and other compiled remnants of deleted files. Harmless but noisy in `git status`-adjacent scans. |

### 10.5 Duplication hotspots

- **`test_web_search_providers.py` (62 tests) re-asserts the scoring rules owned by three dedicated files.** Lines ~948–1395 duplicate `url_quality_score`, `select_top_urls`, `text_relevance_delta`, `result_quality_score`, `select_top_results`, `_term_weight`, `_acronym_terms`, `is_page_text_relevant` from `test_web_search_relevance.py`, `test_web_search_evidence.py` and `test_web_search_authority_readability.py`. `prepare_search_query` is tested in both `providers.py:822-875,1308-1337` and `query_constraints`. JS-retry-on-unreadable is asserted in three files. This is where a change to one scoring rule becomes a three-file edit.
- **Per-language suites are carbon copies.** `test_typescript_edit.py`, `test_text_edit.py` and `test_markdown_edit.py` are ~90% identical (all wrap `_source_edit.py`); JSON/YAML add schema-specific cases, CSV is genuinely different. Shared harnesses (`_FakeModel`, `RecordingCheckpointer`, `FakeGraph`, `_client`, `isolated_settings` boilerplate) are re-derived per file instead of living in conftest.
- **Mandarin variant strings asserted twice** — end-to-end in `test_bounded_search_queries.py:120-127,266-269` and at pure-function level in `test_mandarin_query_planning.py`. Any `query_prep` wording tweak breaks two files.
- **Memory: watermark semantics asserted twice** across the extraction pair; turn-message ordering asserted at two seams; "injection is read-only" asserted three times; failing-store degradation covered at four overlapping layers; session-scope isolation asserted four separate times.
- **Artifact-shape assertions** (`type=amap`, `kind=marker|route`, `coordinateSystem=gcj02`) duplicated between `test_agent_tools.py:400-403,511-514` and `test_amap.py:219-297`.
- **`registers X_with_session_scope`** duplicated near-verbatim for the full vs lightweight graphs in `test_graph_builder.py:255-676`.

### 10.6 Coverage gaps

**Untested source modules.**
- **5 language tool pairs are exported but have zero coverage:** `c_edit/c_file`, `html_edit/html_file`, `java_edit/java_file`, `javascript_edit/javascript_file`, `python_edit/python_file` — absent from both `test_code_edit.LANGUAGES` and `test_file_tools` parametrization. They ship in `src/backend/tools/__init__.py` unverified.
- `web_search/benchmark.py` `evaluate()` is exercised only on hand-built 2-query fixtures; the committed 18-query manifest is loaded only for coverage checks, so precision and latency math are never validated against real data.
- `web_search/playwright_loader.py` has no tests.
- `web_search/claim_consensus.py` (3 tests) and `web_search/query_constraints.py` (3 tests) are thin.
- The deleted `application/rag_service.py` layer's replacement is never covered at the application layer — `test_rag_application_service.py` was removed, not replaced.
- `graph/metrics.py` covered by one happy-path snapshot; `graph/events.py` error/recoverable variants thin.
- Non-critique edges `route_after_agent`, `route_after_lightweight_agent`, `route_after_subgoal_aggregation` are untested.

**Unexercised failure paths.**
- `SessionWatermarkStore` has only a happy path — no test for session storage raising or a non-int persisted key; nothing records that `InMemoryWatermarkStore` resets watermarks on restart.
- `purge_session` has no error-path test; `get_memory_store` has no test for a `MemoryPathError` from path resolution.
- `MAX_SCOPE_ID_CHARS` / `MAX_QUERY_CHARS` enforced only at the schema/tool layer, never at the store layer.
- `ExtractionScheduler` exception-release tested only at concurrency=1.
- Memory concurrency is saves-only: nothing races save against forget, `purge_session` or recall.
- Nothing asserts the content of the extraction **prompt** — the model is always a stub, so the transcript is only observed as `render_slice` output, never checked end to end.

**Unverified-by-design (worth stating plainly).**
- **The suite contains no live-network and no live-LLM test.** Every HTTP call, model call, provider response and DNS lookup is a fake. This makes CI fast and hermetic — and it means the suite guarantees nothing about real search-provider behavior, real embedding calls, or the real AMap API. A regression that only manifests against live services would pass CI.
- `test_chat_static_amap.py` is string-matching on production JavaScript — brittle to any refactor, and it provides zero behavioral or JS-level coverage of the frontend bundle.
- `test_amap_proxy.py` tests only the pure functions (`fetch_amap_proxy_response`, `validate_amap_proxy_path`) — never the FastAPI route, middleware, or auth layer.
- Nothing pins `benchmark.py`'s `parents[2]` path resolution, so moving the module turns a run into a silent skip.
- `test_tool_catalog_policy.py` uses only dummy echo/capture tools — nothing pins the *actual* builtin catalog against a golden name/risk list, which is exactly how §10.3 slipped through.

### 10.7 What the suite is genuinely strong at

Read fairly: the security and operations envelope is unusually disciplined.
- **Secret non-disclosure is a cross-cutting invariant**, asserted in at least 8 places: observability (Hypothesis, Req 9.3), outbound config confinement, endpoint policy, startup-failure cleanup, release gate (both validation and CLI), key rotation, quota errors, and the memory secret screen. Every one of these tests that a secret-bearing failure path does not echo the secret in the exception string or the log record.
- **Fail-closed defaults** are consistently asserted: auth 503 without a key in protected environments, topology 500-equivalent startup refusal, unknown MCP env vars, non-reference secrets, stdio in hosted production, scaled production without a complete validated capability generation, missing rehearsal drills.
- **Caller-supplied identity is provably ignored** (`X-Principal-Id`/`X-Tenant-Id` spoofed in headers; `principal_id`/`tenant_id` in the body → 422).
- **Resource ownership** is tested across every HTTP surface of one thread (session, stream, upload, artifact, delete) with the caller switched to a foreign principal mid-test.
- **Concurrency is tested for real** — 8 threads × 20 memory saves, 6 threads × 30 checkpoint writes, scheduler slot release under a raising runner, SQLite write serialization with a structural lock invariant that doesn't depend on timing.
- **The three big production regressions are pinned** with explanatory comments that read like postmortems (registry `or`-fallback, checkpoint dict-iteration crash, upload chip overlap).

---

## Appendix — group sizes

| Group | Files | Approx. Python |
|---|---|---|
| 1. Web search & fetch | 17 | ~200 KB |
| 2. Query planning & benchmark | 4 | ~25 KB |
| 3. File & language tool suites | 13 | ~150 KB |
| 4. Memory subsystem | 12 | ~165 KB |
| 5. Graph & RAG core | 10 | ~185 KB |
| 6. Chat app, API, sessions, config | 7 | ~130 KB |
| 7. Agent tools, persona, AMap/geo | 8 | ~120 KB |
| 8. MCP, security, deployment, release gate | 9 | ~75 KB |
| `conftest.py` | 1 | 1 KB |
| **Total** | **80 + conftest** | **~1.1 MB** |

`test_resource_ownership_quotas.py` appears in both Group 6 and Group 8 because it spans the HTTP/ownership boundary and the quota engine.
