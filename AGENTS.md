# Agent Guide — Otology Literature Research Agent

Operational notes for AI agents working in this repo.

## Starting the server

The Flask backend runs on **port 8080** (port 5000 is taken by macOS AirPlay and returns 403).

```bash
set -a && source .env && set +a
python3 agent/server.py
```

This defaults to Meilisearch native hybrid retrieval with embedder `otology_openai_large` and OpenAI `text-embedding-3-large` rerank embeddings. The startup logs should include `Meili hybrid` and `rerank embeddings=openai:text-embedding-3-large`.

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

Key variables: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `MEILI_URL`, `MEILI_INDEX`, `MEILI_SEARCH_KEY`, `MEILI_WRITE_KEY`.

Hybrid retrieval variables, now matching the runtime defaults:

```bash
MEILI_HYBRID_SEARCH=1
MEILI_HYBRID_EMBEDDER=otology_openai_large
MEILI_HYBRID_PROVIDER=openai
MEILI_HYBRID_MODEL=text-embedding-3-large
MEILI_HYBRID_SEMANTIC_RATIO=0.3
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
```

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
| `scripts/vectorize_and_upload.py` | Precomputes document embeddings and uploads `_vectors` for Meilisearch hybrid search |
| `agent-review.md` | Architecture review: known gaps, completed fixes, open items |
| `search-app/chat.html` | Chat UI (served at `/`) |

## Running the benchmark

```bash
set -a && source .env && set +a
python3 scripts/run_benchmark.py --questions 1-8
python3 scripts/run_benchmark.py --questions all
```

Full-agent results land in `benchmark-runs/<timestamp>/` as `results.json` (full traces), `summary.md` (table), and `answers.md` (readable answers/citations).

Retrieval-only benchmark for fast hybrid/search tuning:

```bash
set -a && source .env && set +a
EMBEDDING_PROVIDER=openai \
EMBEDDING_MODEL=text-embedding-3-large \
MEILI_HYBRID_SEARCH=1 \
MEILI_HYBRID_EMBEDDER=otology_openai_large \
MEILI_HYBRID_PROVIDER=openai \
MEILI_HYBRID_MODEL=text-embedding-3-large \
MEILI_HYBRID_SEMANTIC_RATIO=0.3 \
python3 scripts/run_benchmark.py --questions all --retrieval-only --max-results 10 --stop-on-rerank-fallback
```

Retrieval-only results include `retrieval.md` with titles, scores, semantic scores, score components, boosts, and topic-gate diagnostics.

## Vectorizing the index

Use the existing local corpus; do not re-scrape PubMed unless the data itself needs refreshing.

```bash
set -a && source .env && set +a
EMBEDDING_MODEL=text-embedding-3-large \
python3 scripts/vectorize_and_upload.py --embedding-batch-size 64 --upload-batch-size 100
```

The hosted index currently has `text-embedding-3-large` vectors for all 16,496 docs under Meilisearch embedder `otology_openai_large`.

## Architecture in brief

Query → expand to up to 5 variants → Meilisearch native hybrid fetch by default (BM25 + vector search) → RRF merge → PMID dedup → semantic rerank (configurable embedding provider, asymmetric task types, topic-gated boosts) → top 10 returned to model. Up to 5 agentic tool-call turns per request. See `README.md` for the full composite score formula.

## Known gotchas

- `current_year = 2026` is hardcoded in `recency_boost_for_year` (`server.py:355`) — should use `datetime.date.today().year`
- Journal filter is exact-match and case-sensitive — brittle for journal name variants
- Embedding cache lives at `data/runtime/embedding-cache.sqlite` (gitignored); set `DISABLE_EMBEDDING_CACHE=1` to bypass
- Native hybrid search is enabled by default; set `MEILI_HYBRID_SEARCH=0` to force BM25-only first-stage fetch. The app still applies its own semantic reranker after Meili returns candidates
- The model is `gemma-4-31b-it` (Gemma 4, not Gemini) — it supports native function-calling via the Google GenAI API
