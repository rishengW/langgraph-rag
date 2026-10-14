# RAG Retrieval Quality Benchmark

Offline benchmark that makes RAG retrieval changes measurable. A golden set of
questions with expected source documents is answered by the persisted Chroma
store; capture records what the retriever returns, evaluate scores it. The
evaluate step never calls an LLM or the embedding provider, so it is safe in CI.

## Golden set

`golden_set.jsonl` holds one object per line:

```json
{"query_id":"pai_claude_code_model_guidelines_overview","question":"What does the PAI model-guidelines page say about using PAI models in Claude Code?","expected_sources":["https://help.aliyun.com/zh/pai/user-guide/use-pai-model-guidelines-in-claude-code"],"note":"top-level content question"}
```

`expected_sources` are matched loosely: query strings and trailing slashes are
stripped before comparison, so a golden URL with tracking parameters matches a
chunk stored under the canonical link. `note` is optional commentary.

## Capture a run

```powershell
python -m src.backend.rag.benchmark capture `
  --top-k 4 `
  --strategy default `
  --out benchmarks/rag_quality/runs.jsonl
```

Capture loads the persisted Chroma store via `ChromaRetriever` (no rebuild) and
records each question's ranked chunks with their `source` metadata and a text
head. It needs `DASHSCOPE_API_KEY` for query embeddings. Use `--strategy` to
label what you are measuring (for example `default`, `hybrid`, `lexical-rerank`)
and keep one file per configuration.

## Evaluate

```powershell
python -m src.backend.rag.benchmark evaluate `
  --runs benchmarks/rag_quality/runs.jsonl `
  --k 4
```

Metrics per strategy and in aggregate:

- `recall@k` — expected sources found among the top-k chunks, averaged over
  queries. A cross-page question with two expected sources scores 0.5 when
  only one page is retrieved.
- `mrr` — reciprocal rank of the first expected source within the top k.
- `provenance` — share of runs where every retrieved chunk carries a non-empty
  `source`; a regression here means citation grounding (phase 1) would break.
- `empty` — share of runs returning no chunks at all.

`--format json` emits the machine-readable report; `--k` re-scores an existing
capture at a different cutoff without re-embedding.

Keep `runs.jsonl` captures out of git (they are machine-specific); commit only
golden-set changes so a quality regression always has a stable baseline to
diff against.
