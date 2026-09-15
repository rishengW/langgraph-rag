# Web Search Providers

`WEB_SEARCH_PROVIDER` selects the discovery backend used by the web-search
pipeline. The table below is the authoritative provider registry; the test
`tests/test_web_search_recency.py::test_web_search_skill_provider_table_matches_factory_registry`
keeps this table in sync with `SUPPORTED_PROVIDER_NAMES` in
`src/backend/web_search/factory.py`.

| Provider | Type | Credential |
| --- | --- | --- |
| `serper` | JSON API | `serper_api_key` |
| `brave` | JSON API | `brave_search_api_key` |
| `tavily` | JSON API | `tavily_api_key` |
| `bing_api` | JSON API | `bing_search_api_key` |
| `bing` | HTML | none (keyless HTML scrape) |
| `baidu` | HTML | none (keyless HTML scrape) |
| `duckduckgo` | HTML | none (keyless HTML scrape) |

JSON API providers require their credential field to be set; HTML providers
work without configuration. Unknown names raise a `ValueError` listing the
supported choices. The default provider is `bing`.
