# Agent Guide — Otology Literature Research Agent

Operational notes for AI agents working in this repo.

## Starting the server

The Flask backend runs on **port 8080** (port 5000 is taken by macOS AirPlay and returns 403).

```bash
set -a && source .env && set +a
python3 agent/server.py
```

Confirm it's up:

```bash
curl -s http://localhost:8080/ | grep -o 'input-box'
```

Or find the port if uncertain:

```bash
lsof -iTCP -sTCP:LISTEN | grep python
```

## Environment variables

All secrets live in `.env` at the repo root. The file is gitignored and cannot be read directly. Source it inline in every shell command that needs the keys:

```bash
set -a && source .env && set +a && <your command>
```

Key variables: `GEMINI_API_KEY`, `MEILI_URL`, `MEILI_INDEX`, `MEILI_SEARCH_KEY`, `MEILI_WRITE_KEY`.

## Testing the /chat endpoint

```bash
set -a && source .env && set +a && curl -s -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"YOUR QUESTION HERE"}],"trace":true}' \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
trace = d.get('trace', {})
for tc in trace.get('tool_calls', []):
    print(f\"turn {tc['turn']}: {tc['query']!r} → {tc['count']} results\")
    print(f\"  variants: {tc.get('query_variants', [])}\")
print()
print(d.get('reply', ''))
"
```

Add `"trace":true` to the request body to see which queries were issued, which variants were generated, how many results each returned, and whether any citation warnings fired.

## Flask auto-reload

The server runs with `debug=True` and will auto-reload on file changes to `agent/server.py`. No restart needed after editing that file. Changes to `.env` do require a restart.

## Key files

| File | Purpose |
|---|---|
| `agent/server.py` | Core backend — retrieval, reranking, agentic loop, system prompts |
| `benchmark.md` | 27 evaluation questions across 7 clinical categories |
| `scripts/run_benchmark.py` | Benchmark harness — runs questions through `/chat` and saves results |
| `scripts/fetch_pubmed.py` | Fetches otology articles from PubMed into `my-data/pubmed-otology.json` |
| `scripts/upload.py` | Uploads JSON corpus to Meilisearch |
| `agent-review.md` | Architecture review: known gaps, completed fixes, open items |
| `search-app/chat.html` | Chat UI (served at `/`) |

## Running the benchmark

```bash
set -a && source .env && set +a
python3 scripts/run_benchmark.py --questions 1-8
python3 scripts/run_benchmark.py --questions all
```

Results land in `benchmark-runs/<timestamp>/` as `results.json` (full traces) and `summary.md` (table).

## Architecture in brief

Query → expand to up to 5 variants → BM25 fetch (top 60 per variant) → RRF merge → PMID dedup → semantic rerank (`gemini-embedding-001`, asymmetric task types) → top 10 returned to model. Up to 5 agentic tool-call turns per request. See `README.md` for the full composite score formula.

## Known gotchas

- `current_year = 2026` is hardcoded in `recency_boost_for_year` (`server.py:355`) — should use `datetime.date.today().year`
- Journal filter is exact-match and case-sensitive — brittle for journal name variants
- Embedding cache lives at `data/runtime/embedding-cache.sqlite` (gitignored); set `DISABLE_EMBEDDING_CACHE=1` to bypass
- The model is `gemma-4-31b-it` (Gemma 4, not Gemini) — it supports native function-calling via the Google GenAI API
