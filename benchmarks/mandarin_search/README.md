# Mandarin Search Quality Benchmark

This offline benchmark makes Mandarin search changes measurable without calling a search
engine or an LLM during tests. `queries.json` contains 18 versioned chat questions across
count, date, policy, price, comparison, and current-status intents.

## Capture a run

For each provider and query strategy, record one JSON object per line in `runs.jsonl`:

```json
{"query_id":"count_nanjing_metro_2025","provider":"bing","strategy":"exact","query":"截至2025年底 南京地铁 投入运营 线路 总数","latency_ms":482.6,"results":[{"rank":1,"url":"https://example.gov.cn/article/metro","title":"...","snippet":"..."}]}
```

`strategy` should identify query formulation, for example `original`, `exact`, or
`official`. Measure `latency_ms` around the provider call only. Preserve the order returned
by the provider and capture an empty `results` array when it returns nothing. Do not put API
keys, cookies, or request headers in a capture file.

Use the same query manifest and capture date when comparing providers or strategies.
Current-status results should be relabeled when their `as_of_date` becomes stale.

## Label the results

Fetch and inspect each distinct query/URL pair, then write one object per line to
`judgments.jsonl`:

```json
{"query_id":"count_nanjing_metro_2025","url":"https://example.gov.cn/article/metro","relevant":true,"official":true,"answerable":true}
```

- `relevant`: the page addresses the original benchmark question, including its entity,
  time scope, and requested relationship.
- `official`: the page is published by the responsible government agency, institution,
  standards body, or product/service owner. Reposts and aggregators are not official.
- `answerable`: the fetched page contains enough explicit evidence to answer the question,
  rather than only mentioning its topic. An answerable page must also be relevant.

Judge the original benchmark question, not the rewritten provider query. Keep judgments
blind to provider and strategy where practical. A second reviewer should resolve borderline
labels before using the scores as a release gate.

## Evaluate

```powershell
python -m src.web_search.benchmark `
  --runs benchmarks/mandarin_search/runs.jsonl `
  --judgments benchmarks/mandarin_search/judgments.jsonl
```

The evaluator reports aggregate and provider/strategy metrics:

- `P@3`: relevant results in the first three positions divided by three per run. Returning
  fewer than three results therefore lowers precision.
- `Official@3`: official sources divided by returned results in the first three positions.
- `Answerable@3`: runs with at least one answerable result in the first three positions.
- `Irrelevant`: irrelevant URLs divided by all returned URLs, not only the first three.
- Latency: provider-call mean, median, nearest-rank p95, minimum, and maximum in JSON output;
  the table shows mean and p95.

Every returned URL must be labeled by default. During an unfinished labeling pass,
`--allow-unjudged` treats missing judgments as irrelevant and exposes their count in the
`Missing` column. Use `--format json` for machine-readable output and `--k` to inspect a
different cutoff.
